import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore
firebase_config = {
  "apiKey": "AIzaSyChAnD1ryKlGV9rW_CYsYHHzrETt5mF69M",
  "authDomain": "inteliiscreen.firebaseapp.com",
  "databaseURL": "https://ai-powred-default-rtdb.asia-southeast1.firebasedatabase.app//",
  "projectId": "inteliiscreen",
  "storageBucket": "inteliiscreen.firebasestorage.app",
  "messagingSenderId": "889455936299",
  "appId": "1:889455936299:web:bdbfea86f8c48c6b323b6e"
}
firebase_config = {

  "apiKey": "AIzaSyDYd-pSwm-n8FoweL1UQ26ycJD-r1IRYt8",
  "authDomain": "ai-powred.firebaseapp.com",
  "databaseURL": "https://ai-powred-default-rtdb.asia-southeast1.firebasedatabase.app",
  "projectId": "ai-powred",
  "storageBucket": "ai-powred.firebasestorage.app",
  "messagingSenderId": "856605504519",
  "appId": "1:856605504519:web:15d75308b69491d918e6a2"

}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
# Firebase Admin SDK for Firestore
def init_firestore():
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase_credentials.json")
        firebase_admin.initialize_app(cred)
    return firestore.client()

def login_user(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        return user, None
    except Exception as e:
        error_message = parse_firebase_error(e)
        return None, error_message

def send_password_reset(email):
    try:
        auth.send_password_reset_email(email)
        return "Password reset email sent!"
    except Exception as e:
        return parse_firebase_error(e)

def parse_firebase_error(e):
    msg = str(e)
    if "EMAIL_NOT_FOUND" in msg:
        return "Email not found."
    elif "INVALID_PASSWORD" in msg:
        return "Invalid password."
    elif "MISSING_EMAIL" in msg:
        return "Please enter your email."
    else:
        return "Authentication failed."