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
firebase = pyrebase.initialize_app(config)
db = firebase.database()
db.child("image").child("obj1")