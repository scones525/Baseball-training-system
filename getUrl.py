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
def getAndUpload():
    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()
    url1 = storage.child("image/1.png").get_url(None)
    url2 = storage.child("image/2.png").get_url(None)
    url3 = storage.child("image/3.png").get_url(None)
    url4 = storage.child("image/4.png").get_url(None)
    url5 = storage.child("image/5.png").get_url(None)
    url6 = storage.child("image/6.png").get_url(None)
    url7 = storage.child("image/7.png").get_url(None)


    database=firebase.database()

    database.child("message").child("obj1").child("image").set(url1)

    database.child("message").child("obj2").child("image").set(url2)
    database.child("message").child("obj3").child("image").set(url3)
    database.child("message").child("obj4").child("image").set(url4)
    database.child("message").child("obj5").child("image").set(url5)
    database.child("message").child("obj6").child("image").set(url6)
    database.child("message").child("obj7").child("image").set(url7)
