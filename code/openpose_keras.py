import os
import argparse
import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model
import tanaka
import download123
import delete
import getUrl
import upload
import pyrebase
import delete_storage
import uploadImg
import firebase_admin
from multiprocessing import Process
from firebase_admin import credentials

mode = tanaka.model_1
mode_vector = tanaka.model_1_vector
mode_frame = tanaka.model_1_frame

array_data = []
array_vector = []
count = 1

body_index = ["脖子","肩膀","右臂","右手","左臂","左手","右腳","右腿","左腳","左腿"]

tic=0
# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]

# the middle joints heatmap correpondence
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
          [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
          [55, 56], [37, 38], [45, 46]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def process (input_image, params, model_params):

    set_data = []

    oriImg = cv2.imread(input_image)  # B,G,R order
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    #for m in range(len(multiplier)):
    for m in range(1):
        scale = multiplier[m]

        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])

        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)

        
        output_blobs = model.predict(input_img)
      
        
        
        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []
    peak_counter = 0
    prinfTick(1)
    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    test_all_peak_connection = []       #######

    connection_all = []
    special_k = []
    mid_num = 10
    prinfTick(2)
    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):           #當兩個特徵點數量>0時，運算彼此間的向量。 
            connection_candidate = []       #要是彼此兩特徵點符合連接條件的話，+入connection_candidate[]。
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])           #(x,y)位移。
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])     #兩特徵點間的距離。
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(
                        score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    #test_all_peak_connection.append([candA[i][3], candB[j][3], i, j])
                    #print('特徵點',candA[i][3]+1,'第',i+1,'個與特徵點',candB[j][3]+1,'第',j+1,'個特徵點相連')
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    prinfTick(3)
    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    '''canvas = cv2.imread('bang.png')  # B,G,R order
    for i in range(18):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)'''

    canvas = cv2.imread(input_image)  # B,G,R order
    for i in range(18):
        #print('特徵點',i,': ')
        if(len(array_data) == 0 and len(all_peaks[i]) == 0):
            set_data.append((0,0))
        elif(len(all_peaks[i]) == 0 and len(array_data) != 0 ):
            set_data.append(array_data[len(array_data)-1][i][0:2])
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
            #print('第',j+1,'個位置: ',all_peaks[i][j][0:2])
            if(j == 0):
                set_data.append(all_peaks[i][j][0:2])

    #if(array_data[len(array_data)])

    array_data.append(set_data)
    #print(array_data[0][0][0])

    h = array_data[0][10][1] - array_data[0][0][1]
    h2 = mode[0][10][1] - mode[0][0][1]
    ratio = h2 / h

    omg = len(array_data)
    array_vector.append([(ratio * array_data[omg-1][1][1]-ratio * array_data[omg-1][0][1], ratio * array_data[omg-1][1][0]-ratio * array_data[omg-1][0][0]), \
    (ratio * array_data[omg-1][5][1]-ratio * array_data[omg-1][2][1], ratio * array_data[omg-1][5][0]-ratio * array_data[omg-1][2][0]), \
    (ratio * array_data[omg-1][2][1]-ratio * array_data[omg-1][3][1], ratio * array_data[omg-1][2][0]-ratio * array_data[omg-1][3][0]), \
    (ratio * array_data[omg-1][3][1]-ratio * array_data[omg-1][4][1], ratio * array_data[omg-1][3][0]-ratio * array_data[omg-1][4][0]), \
    (ratio * array_data[omg-1][6][1]-ratio * array_data[omg-1][5][1], ratio * array_data[omg-1][6][0]-ratio * array_data[omg-1][5][0]), \
    (ratio * array_data[omg-1][7][1]-ratio * array_data[omg-1][6][1], ratio * array_data[omg-1][7][0]-ratio * array_data[omg-1][6][0]), \
    (ratio * array_data[omg-1][9][1]-ratio * array_data[omg-1][8][1], ratio * array_data[omg-1][9][0]-ratio * array_data[omg-1][8][0]), \
    (ratio * array_data[omg-1][10][1]-ratio * array_data[omg-1][9][1], ratio * array_data[omg-1][10][0]-ratio * array_data[omg-1][9][0]), \
    (ratio * array_data[omg-1][12][1]-ratio * array_data[omg-1][11][1], ratio * array_data[omg-1][12][0]-ratio * array_data[omg-1][11][0]), \
    (ratio * array_data[omg-1][13][1]-ratio * array_data[omg-1][12][1], ratio * array_data[omg-1][13][0]-ratio * array_data[omg-1][12][0])])

    for i in range(len(array_data)):
        print(array_data[i])
        #print(len(array_data))

    for i in range(len(array_vector)):
        print(array_vector[i])
        #print(len(array_vector))
    #print(all_peaks)

    stickwidth = 4
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0,
                                       360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    #print(test_all_peak_connection)
    #print(len(test_all_peak_connection))
    return canvas

def prinfTick(i):
    toc = time.time()
    #print ('processing time%d is %.5f' % (i,toc - tic))            ########

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='sample_images/ski1.jpg', help='input image')
    parser.add_argument('--output', type=str, default='result.png', help='output image')
    parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')

    args = parser.parse_args()
    input_image = args.image
    output = args.output
    keras_weights_file = args.model

    tic = time.time()
    print('start processing...')

    # load model

    # authors of original model don't use
    # vgg normalization (subtracting mean) on input images
    model = get_testing_model()
    model.load_weights(keras_weights_file)

####################################################main#############################################

    while(True):
        image_prtsc_index = 1
        download123.download_video()
        if os.path.exists('success.mp4'):
            cap = cv2.VideoCapture('success.mp4')
            cap2 = cv2.VideoCapture('success.mp4')
            vi = cap.isOpened()

            frame_count = 0  # Counting Frame numbers.
            all_frames = []
            while (True):
                rr, ff = cap2.read()
                if rr is False:
                    break
                all_frames.append(ff)
                frame_count = frame_count + 1

            if (vi == True):
                cap.set(3, 720)
                cap.set(4, 480)
                time.sleep(0)
                for i in range(frame_count):
                    tic = time.time()
                    ret, frame = cap.read()
                    writeStatus = cv2.imwrite(input_image, frame)
                    params, model_params = config_reader()

                    # generate image with body parts
                    print('第', count, '張影像')
                    count = count + 1
                    canvas = process(input_image, params, model_params)

                    if(i == int(image_prtsc_index * (frame_count / 7)) - 1):
                        cv2.imwrite(str(image_prtsc_index) + ".png", canvas)
                        image_prtsc_index = image_prtsc_index + 1
                    #print(image_prtsc_index)

                    cv2.imshow("capture", canvas)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cap.release()
            cv2.destroyAllWindows()

            distance = []
            for i in range(frame_count):
                d1 = []
                for j in range(10):
                    d2 = []
                    for k in range(mode_frame):
                        d2.append(math.sqrt((array_vector[i][j][1] - mode_vector[k][j][1]) ** 2 + (
                                    array_vector[i][j][0] - mode_vector[k][j][0]) ** 2))
                    d1.append(d2)
                distance.append(d1)

            print(len(distance), len(distance[0]), len(distance[0][0]))

            '''for i in range(frame_count):             #test
                for j in range(10):
                    for k in range(mode_frame):
                        if(i == k):
                            print(distance[i][j][k])'''

            wrap = distance.copy()
            for i in range(frame_count):
                for j in range(10):
                    for k in range(mode_frame):
                        if (k >= 1 and i >= 1):
                            wrap[i][j][k] = wrap[i][j][k] + min(wrap[i - 1][j][k - 1], wrap[i][j][k - 1],
                                                                wrap[i - 1][j][k])
                        elif (k >= 1 and i == 0):
                            wrap[i][j][k] = wrap[i][j][k] + wrap[i][j][k - 1]
                        elif (k == 0 and i >= 1):
                            wrap[i][j][k] = wrap[i][j][k] + wrap[i - 1][j][k]

            grade = 0
            wraping_grade = 0
            warning = []
            for i in range(10):
                grade_part = 2100 / (20 + 22 ** (wrap[frame_count - 1][i][mode_frame - 1] / 3000))
                if (grade_part < 30):
                    warning.append(body_index[i])
                print("向量 ", i, " 分數:", grade_part, "\twraping:", wrap[frame_count - 1][i][mode_frame - 1])
                grade = grade_part + grade
                wraping_grade = wraping_grade + wrap[frame_count - 1][i][mode_frame - 1]
            print("平均分數 : ", grade / 10, "\t平均wraping:", wraping_grade / 10)
            if(len(warning)==0):
                print("Great!")
                warning.append("Great!")
            else:
                for i in range(len(warning)-1):
                    warning[0] = warning[0] + ",\n"
                    warning[0] = warning[0] + warning[i + 1]
                print(warning[0])
            print_string = "經由分析後，您的:\n" +warning[0] + "\n為不合格。\n以上是投球姿勢檢測圖。\n感謝使用~"
            upload.feedback(print_string)
            delete.delete_video()
            cap = None
            cap2 = None

            delete_storage.delete_png()
            uploadImg.upload_img()
            getUrl.getAndUpload()

            os.remove('success.mp4')

########################################## TW ##################################

'''wrap = []

for i in range(frame_count):
    d1 = []
    for j in range(10):
        d2 = []
        for k in range(mode_frame):
            if(k>=1 and i>=1):
                var = distance[i][j][k] + min(wrap[i-1][j][k-1], wrap[i][j][k-1], wrap[i-1][j][k])
            elif(k>=1 and i==0):
                var = distance[i][j][k] + wrap[i][j][k-1]
            elif(k==0 and i>=1):
                var = distance[i][j][k] + wrap[i-1][j][k]
            elif(k==0 and i==0):
                var = distance[i][j][k]
            d2.append(var)
        d1.append(d2)
    wrap.append(d1)'''



'''for i in range(frame_count):
    for j in range(10):
        for k in range(mode_frame):
            if(i == k):
                print(wrap[i][j][k])'''


'''grade = 0
for i in range(10):
    grade_part = wrap[frame_count-1][i][mode_frame-1]
    print("向量 ",i," ",grade_part)
    grade = grade_part + grade
print("平均 : ",grade/10)'''





