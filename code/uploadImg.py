import pyrebase
config = {
    "apiKey" : "AIzaSyBetij4oV0OI07i1dejkUbR_WLKiGTVKt4",
    "authDomain" : "finalproject-4d00b.firebaseapp.com",
    "databaseURL" : "https://finalproject-4d00b.firebaseio.com",
    "projectId" : "finalproject-4d00b",
    "storageBucket" : "finalproject-4d00b.appspot.com",
    "messagingSenderId": "333251010728",
    "appId": "1:333251010728:web:bcde17f5a5fff61ee2a29d",
    "measurementId": "G-23GJKYQJK0",
}

def upload_img():
    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()
    storage.child("image/1.png").put("1.png")
    storage.child("image/2.png").put("2.png")
    storage.child("image/3.png").put("3.png")
    storage.child("image/4.png").put("4.png")
    storage.child("image/5.png").put("5.png")
    storage.child("image/6.png").put("6.png")
    storage.child("image/7.png").put("7.png")
