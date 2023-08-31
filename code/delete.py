import os
import pyrebase
import firebase_admin
from firebase_admin import credentials

config = {
    "apiKey" : "AIzaSyBetij4oV0OI07i1dejkUbR_WLKiGTVKt4",
    "authDomain" : "finalproject-4d00b.firebaseapp.com",
    "databaseURL" : "https://finalproject-4d00b.firebaseio.com",
    "projectId" : "finalproject-4d00b",
    "storageBucket" : "finalproject-4d00b.appspot.com",
    "messagingSenderId": "333251010728",
    "appId": "1:333251010728:web:bcde17f5a5fff61ee2a29d",
    "measurementId": "G-23GJKYQJK0",
    "serviceAccount": "A:/final/firebaseImportant.json"
}
def delete_video():

    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()

#storage.child("testUpload/video.mp4").download("","AAAAA.mp4")

    storage.delete("testUpload/video.mp4")

#storage.child("testUpload/").delete("video.mp4")
#path_on_cloud = "testUpload/video.mp4"

#storage.child(path_on_cloud).download("","success.mp4")





