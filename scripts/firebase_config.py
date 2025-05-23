import os
import pyrebase
import firebase_admin
from dotenv import load_dotenv

from firebase_admin import credentials, firestore

# Load environment variables from .env file
load_dotenv()

# Firebase config from environment variables
firebase_config = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID")
}

# Initialize Pyrebase
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# Initialize Firebase Admin SDK for Firestore
def init_firestore():
    if not firebase_admin._apps:
        cred = credentials.Certificate("scripts/firebase_credentials.json")
        firebase_admin.initialize_app(cred)
    return firestore.client()

# Login user with Firebase Auth
def login_user(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        return user, None
    except Exception as e:
        error_message = parse_firebase_error(e)
        return None, error_message

# Send password reset email
def send_password_reset(email):
    try:
        auth.send_password_reset_email(email)
        return "Password reset email sent!"
    except Exception as e:
        return parse_firebase_error(e)

# Parse Firebase authentication errors
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