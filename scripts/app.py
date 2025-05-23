import streamlit as st  # Moved to top as st.set_page_config must be the first Streamlit call

# ✅ Streamlit Page Config (must come before any other Streamlit command)
st.set_page_config(page_title="JobMatchAI", layout="wide")

import firebase_admin
from firebase_admin import credentials, firestore
import pdfplumber
import logging
import warnings
import ollama
import os
import json
import pandas as pd
import numpy as np
import datetime
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
from flask import Flask
from firebase_config import auth

# Load environment variables
load_dotenv()
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_credentials.json")
    firebase_admin.initialize_app(cred)
db_firestore = firestore.client()

# Title after page config
st.title("🚀 AI-Powered Job Screening Platform")

# ✅ Initialize session state
if "user" not in st.session_state:
    st.session_state.user = None



# --- SIDEBAR: Authentication ---
st.sidebar.title("🔐 Authentication")

# ✅ If user is logged in, show Logout
if st.session_state.get("user"):
    st.sidebar.success(f"Logged in as: {st.session_state.user['email']}")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.success("You have been logged out.")
        st.experimental_rerun()

# ✅ Login form if user is not logged in
else:
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")

    login_clicked = st.sidebar.button("Login")
    if login_clicked:
        if not email or not password:
            st.sidebar.warning("Please enter both email and password.")
        else:
            try:
                user = auth.sign_in_with_email_and_password(email, password)
                st.session_state.user = user
                st.success("✅ Login successful!")
                st.experimental_rerun()
            except Exception as e:
                st.sidebar.error("❌ Invalid email or password.")
# --- MAIN AREA ---
if not st.session_state.get("user"):
    st.title("🔐 Login Required")
    st.write("Please log in using the sidebar to access the JobMatchAI Dashboard.")
    st.stop()


# ✅ Main Dashboard (only after login)
st.title("📊 JobMatchAI Dashboard")
st.success(f"Welcome, {st.session_state.user['email']}!")

# --- Add your main dashboard content below ---
st.write("This is your job screening and AI matching platform.")

def parse_cv(file):
    try:
        pdf = pdfplumber.open(file)
        text = " ".join([page.extract_text() or "" for page in pdf.pages])
        pdf.close()
        return text
    except Exception as e:
        st.error(f"Error parsing PDF: {e}")
        return None

def match_candidate(jd_summary, cv_text):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([jd_summary, cv_text])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
        return round(score, 2)
    except Exception as e:
        st.error(f"Error matching CV with JD: {e}")
        return 0.0

def save_candidate(name, email, cv_text):
    try:
        doc_ref = db_firestore.collection("candidates").document(email)
        doc_ref.set({
            "name": name,
            "email": email,
            "cv_text": cv_text,
            "upload_date": firestore.SERVER_TIMESTAMP
        })
        return True
    except Exception as e:
        st.error(f"Error saving candidate to Firebase: {e}")
        return False

def save_matches(email, matches):
    try:
        for jd_id, score in matches:
            db_firestore.collection("matches").add({
                "candidate_email": email,
                "jd_id": jd_id,
                "score": score,
                "match_date": firestore.SERVER_TIMESTAMP
            })
        return True
    except Exception as e:
        st.error(f"Error saving matches to Firebase: {e}")
        return False

def save_interview(email, jd_id, date, time, notes):
    try:
        db_firestore.collection("interviews").add({
            "candidate_email": email,
            "jd_id": jd_id,
            "scheduled_date": date.isoformat(),
            "scheduled_time": time.strftime("%H:%M"),
            "notes": notes,
            "created_at": firestore.SERVER_TIMESTAMP
        })
        return True
    except Exception as e:
        st.error(f"Error saving interview to Firebase: {e}")
        return False

def send_interview_invite(email, name, jd_title, date, time, notes):
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        from_email = os.getenv("EMAIL_USER")
        subject = f"Interview Invitation: {jd_title} Position"
        
        body = f"""
        <p>Dear {name},</p>
        <p>We are pleased to inform you that you have been shortlisted for an interview for the position of <b>{jd_title}</b> at JobMatchAI.</p>
        <p><b>Date:</b> {date.strftime('%A, %B %d, %Y')}<br>
        <b>Time:</b> {time.strftime('%I:%M %p')}<br>
        <b>Notes:</b> {notes}</p>
        <p>Best regards,<br>JobMatchAI Team</p>
        """

        message = Mail(from_email=from_email, to_emails=email, subject=subject, html_content=body)
        sg.send(message)
        return True
    except Exception as e:
        st.warning(f"SendGrid Email failed: {e}")
        return False

def get_stats():
    try:
        candidates = [doc.to_dict() for doc in db_firestore.collection("candidates").stream()]
        matches = [doc.to_dict() for doc in db_firestore.collection("matches").stream()]
        interviews = [doc.to_dict() for doc in db_firestore.collection("interviews").stream()]
        avg_score = np.mean([m['score'] for m in matches]) if matches else 0
        return len(candidates), len(matches), avg_score, len(interviews)
    except Exception as e:
        st.error(f"Error fetching stats: {e}")
        return 0, 0, 0, 0

# Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Upload & Match", "Schedule Interview"])

# Upload & Match
if section == "Upload & Match":
    st.header("📄 Upload Candidate CV")
    name = st.text_input("👤 Candidate Name")
    email = st.text_input("📧 Email Address")
    cv = st.file_uploader("📎 Upload PDF CV", type='pdf')

    if st.button("🔍 Match Jobs"):
        if not all([name, email, cv]):
            st.warning("Please fill all required fields.")
            st.stop()

        cv_text = parse_cv(cv)
        if not cv_text:
            st.stop()

        if save_candidate(name, email, cv_text):
            try:
                with open("data/jd_summaries.json") as f:
                    jds = json.load(f)
            except Exception as e:
                st.error(f"Could not load JD summaries: {e}")
                st.stop()

            results = [(jd_id, match_candidate(jd.get("summary", ""), cv_text)) for jd_id, jd in jds.items()]
            results.sort(key=lambda x: x[1], reverse=True)

            if save_matches(email, results):
                st.success("🎯 Matching Complete! Top 3 Matches:")
                for jd_id, score in results[:3]:
                    title = jds[jd_id].get("title", jd_id)
                    st.markdown(f"""
                        <div style='background-color:#E75E5B;padding:10px;border-radius:8px;margin-bottom:10px;'>
                        <b>{title}</b><br>Match Score: {score:.2f}%</div>
                    """, unsafe_allow_html=True)
            else:
                st.error("❌ Failed to save matches to Firebase.")

# Schedule Interview
elif section == "Schedule Interview":
    st.header("📅 Schedule Interview")
    with st.form(key="interview_form"):
        email = st.text_input("Candidate Email")
        jd_id = st.text_input("Job ID")
        jd_title = st.text_input("Job Title")
        date = st.date_input("Interview Date", datetime.date.today())
        time = st.time_input("Interview Time", datetime.time(10, 0))
        notes = st.text_area("Notes", "Please bring portfolio.")

        submit_btn = st.form_submit_button("📨 Send Interview Invite")
        if submit_btn:
            if save_interview(email, jd_id, date, time, notes):
                success = send_interview_invite(email, email.split("@")[0], jd_title, date, time, notes)
                if success:
                    st.success("✅ Interview scheduled and email sent!")
                else:
                    st.warning("⚠️ Interview saved, but email failed to send.")
            else:
                st.error("❌ Failed to save interview details.")
