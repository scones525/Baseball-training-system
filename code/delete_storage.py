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
def delete_png():
    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()
    storage.delete("image/1.png")
    storage.delete("image/2.png")
    storage.delete("image/3.png")
    storage.delete("image/4.png")
    storage.delete("image/5.png")
    storage.delete("image/6.png")
    storage.delete("image/7.png")
