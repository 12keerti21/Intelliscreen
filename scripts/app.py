import streamlit as st  # Moved to top as st.set_page_config must be the first Streamlit call

# ✅ Streamlit Page Config (must come before any other Streamlit call)
st.set_page_config(
    page_title="JobMatchAI", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved styling
st.markdown("""
<style>
    /* Main background and text colors with gradient */
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9f2ff 100%);
        color: #212529;
    }
    
    /* Header styling */
    h1, h2, h3, h4 {
        color: #0d2f81;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-weight: 600;
        text-shadow: 0px 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    /* Sidebar styling with gradient */
    .css-1d391kg {
        background: linear-gradient(180deg, #e9ecef 0%, #dbeafe 100%);
    }
    
    /* Form elements styling */
    .stTextInput, .stTextArea, .stSelectbox {
        border-radius: 8px;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Login button specific styling */
    .login-btn {
        background-color: #2563eb !important;
        color: white !important;
        font-size: 16px !important;
        width: 100%;
    }
    
    /* Success messages */
    .element-container div[data-testid="stAlert"] {
        border-radius: 8px;
    }
    
    /* Match card styling - IMPROVED */
    .match-card {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
        border-left: 4px solid #2563eb;
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .match-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(37, 99, 235, 0.15);
    }
    
    /* Footer styling with gradient */
    .footer {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-radius: 10px;
        color: #1e3a8a;
        padding: 20px;
        margin-top: 40px;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.03);
    }
    
    /* Dashboard metrics - IMPROVED */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    .metric-label {
        font-size: 14px;
        color: #4b5563;
        font-weight: 500;
    }
    
    /* Resume parsing results section - NEW */
    .results-header {
        color: #0d2f81;
        margin-top: 30px;
        margin-bottom: 15px;
        font-weight: 700;
        font-size: 24px;
    }
    
    .match-score-circle {
        width: 90px;
        height: 90px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        border: 3px solid white;
    }
    
    .match-score-text {
        color: white;
        font-weight: bold;
        font-size: 26px;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    .match-details {
        margin: 0;
        color: #1e293b;
        font-weight: 500;
    }
    
    .match-job-id {
        margin: 5px 0 0 0;
        font-size: 14px;
        color: #64748b;
    }
    
    .match-title {
        margin: 0 0 12px 0;
        color: #0d2f81;
        font-size: 20px;
        font-weight: 600;
    }
    
    /* Welcome banner with gradient */
    .welcome-banner {
        background: linear-gradient(135deg, #bfdbfe 0%, #93c5fd 100%);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1);
    }
    
    .welcome-heading {
        margin: 0;
        color: #0d2f81;
        font-weight: 700;
    }
    
    .welcome-text {
        margin: 8px 0 0 0;
        color: #1e3a8a;
        font-size: 16px;
    }
    
    /* Form containers with gradient */
    .form-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

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

# Title after page config with logo
st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 30px;">
        <h1 style="margin: 0; color: #0d2f81; font-size: 32px;">🚀 AI-Powered Job Screening Platform</h1>
    </div>
""", unsafe_allow_html=True)

# ✅ Initialize session state
if "user" not in st.session_state:
    st.session_state.user = None

# --- SIDEBAR: Authentication ---
st.sidebar.markdown("""
    <h2 style="color: #0d2f81; text-align: center; font-weight: 700;">🔐 Authentication</h2>
    <hr style="margin: 0.5rem 0 1.5rem 0;">
""", unsafe_allow_html=True)

# ✅ If user is logged in, show Logout
if st.session_state.get("user"):
    st.sidebar.markdown(f"""
        <div style="padding: 15px; background: linear-gradient(135deg, #bbf7d0 0%, #86efac 100%); border-radius: 10px; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
            <p style="margin: 0; font-weight: bold; color: #166534;">Logged in as:</p>
            <p style="margin: 0; color: #166534; font-size: 16px;">{st.session_state.user['email']}</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("Logout", key="logout_button"):
        st.session_state.user = None
        st.success("You have been logged out.")
        st.rerun()

# ✅ Login form if user is not logged in
else:
    with st.sidebar.container():
        st.markdown("""
            <div style="background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);">
                <h3 style="text-align: center; color: #0d2f81; margin-bottom: 20px; font-weight: 700;">Welcome Back!</h3>
        """, unsafe_allow_html=True)
        
        email = st.text_input("📧 Email", placeholder="Enter your email")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            login_clicked = st.button("Login", key="login_button", use_container_width=True)
        
        if login_clicked:
            if not email or not password:
                st.warning("Please enter both email and password.")
            else:
                try:
                    with st.spinner("Authenticating..."):
                        user = auth.sign_in_with_email_and_password(email, password)
                        st.session_state.user = user
                        st.success("✅ Login successful!")
                        st.experimental_rerun()
                except Exception as e:
                    st.error("❌ Invalid email or password.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# --- MAIN AREA ---
if not st.session_state.get("user"):
    st.markdown("""
        <div style="text-align: center; padding: 60px 30px; background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); border-radius: 15px; margin: 20px 0; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);">
            <h1 style="color: #0d2f81; font-weight: 700; margin-bottom: 20px;">🔐 Login Required</h1>
            <p style="font-size: 18px; color: #334155; margin-bottom: 30px;">Please log in using the sidebar to access the JobMatchAI Dashboard.</p>
            <p style="font-size: 14px; color: #64748b;">Need help? Contact support@404tech.com</p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# ✅ Main Dashboard (only after login)
st.markdown("""
    <h1 style="color: #0d2f81; font-weight: 700; margin-bottom: 20px;">📊 JobMatchAI Dashboard</h1>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="welcome-banner">
        <h3 class="welcome-heading">👋 Welcome, HR!</h3>
        <p class="welcome-text">Your AI-powered job screening platform is ready to match the best talent.</p>
    </div>
""", unsafe_allow_html=True)

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

# Display dashboard metrics
candidates, matches, avg_score, interviews = get_stats()
st.markdown("""<h3 style="color: #0d2f81; margin: 20px 0; font-weight: 700;">📈 Overview</h3>""", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        <div class="metric-card">
            <p class="metric-value">{}</p>
            <p class="metric-label">Total Candidates</p>
        </div>
    """.format(candidates), unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-card">
            <p class="metric-value">{}</p>
            <p class="metric-label">Total Matches</p>
        </div>
    """.format(matches), unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="metric-card">
            <p class="metric-value">{:.1f}%</p>
            <p class="metric-label">Average Match Score</p>
        </div>
    """.format(avg_score), unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="metric-card">
            <p class="metric-value">{}</p>
            <p class="metric-label">Interviews Scheduled</p>
        </div>
    """.format(interviews), unsafe_allow_html=True)

# Navigation
st.sidebar.markdown("""
    <h3 style="color: #0d2f81; margin-top: 30px; text-align: center; font-weight: 700;">📋 Navigation</h3>
    <hr style="margin: 0.5rem 0 1.5rem 0;">
""", unsafe_allow_html=True)

section = st.sidebar.radio("Go to", ["Upload & Match", "Schedule Interview"])

# Upload & Match
if section == "Upload & Match":
    st.markdown("""
        <h2 style="color: #0d2f81; margin-top: 40px; font-weight: 700;">📄 Upload Candidate CV</h2>
        <p style="color: #334155; margin-bottom: 25px; font-size: 16px;">Upload a candidate's CV to match with available job positions</p>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
            <div class="form-container">
        """, unsafe_allow_html=True)
        
        name = st.text_input("👤 Candidate Name", placeholder="Enter full name")
        email = st.text_input("📧 Email Address", placeholder="Enter email address")
        cv = st.file_uploader("📎 Upload PDF CV", type='pdf')

        match_col1, match_col2 = st.columns([3, 1])
        with match_col2:
            match_btn = st.button("🔍 Match Jobs", type="primary", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if match_btn:
            if not all([name, email, cv]):
                st.warning("Please fill all required fields.")
                st.stop()

            with st.spinner("Processing CV and finding matches..."):
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
                        st.success("🎯 Matching Complete!")
                        st.markdown("""
                            <h3 class="results-header">✨ Top 3 Matches for {}</h3>
                        """.format(name), unsafe_allow_html=True)
                        
                        for jd_id, score in results[:3]:
                            title = jds[jd_id].get("title", jd_id)
                            # Set gradient background for score circle based on score
                            if score >= 75:
                                score_bg = "linear-gradient(135deg, #10b981 0%, #059669 100%)"
                            elif score >= 50:
                                score_bg = "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)"
                            else:
                                score_bg = "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)"
                                
                            st.markdown(f"""
                                <div class='match-card'>
                                    <h4 class="match-title">{title}</h4>
                                    <div style="display: flex; align-items: center;">
                                        <div style="flex-grow: 1;">
                                            <p class="match-details"><b>Match Score:</b> {score:.2f}%</p>
                                            <p class="match-details"><b>Key Skills:</b> {', '.join(jds[jd_id].get("skills", ["Not specified"])[:3])}</p>
                                            <p class="match-job-id">Job ID: {jd_id}</p>
                                        </div>
                                        <div class="match-score-circle" style="background: {score_bg}">
                                            <span class="match-score-text">{score:.0f}%</span>
                                        </div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                    else:
                        st.error("❌ Failed to save matches to Firebase.")

# Schedule Interview
elif section == "Schedule Interview":
    st.markdown("""
        <h2 style="color: #0d2f81; margin-top: 40px; font-weight: 700;">📅 Schedule Interview</h2>
        <p style="color: #334155; margin-bottom: 25px; font-size: 16px;">Schedule interviews with matched candidates</p>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
            <div class="form-container">
        """, unsafe_allow_html=True)
        
        with st.form(key="interview_form"):
            st.markdown("<h4 style='color: #0d2f81; font-weight: 600;'>Candidate & Job Details</h4>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                email = st.text_input("📧 Candidate Email", placeholder="Enter email address")
                jd_id = st.text_input("🆔 Job ID", placeholder="Enter job ID")
            
            with col2:
                jd_title = st.text_input("💼 Job Title", placeholder="Enter job title")
            
            st.markdown("<h4 style='color: #0d2f81; margin-top: 25px; font-weight: 600;'>Interview Details</h4>", unsafe_allow_html=True)
            
            col3, col4 = st.columns(2)
            with col3:
                date = st.date_input("📆 Interview Date", datetime.date.today())
            
            with col4:
                time = st.time_input("🕒 Interview Time", datetime.time(10, 0))
            
            notes = st.text_area("📝 Notes", "Please bring portfolio and be prepared to discuss your past projects.", height=100)

            col5, col6, col7 = st.columns([1, 2, 1])
            with col6:
                submit_btn = st.form_submit_button("📨 Send Interview Invite", use_container_width=True)
            
            if submit_btn:
                with st.spinner("Scheduling interview and sending invitation..."):
                    if save_interview(email, jd_id, date, time, notes):
                        success = send_interview_invite(email, email.split("@")[0], jd_title, date, time, notes)
                        if success:
                            st.success("✅ Interview scheduled and email sent!")
                        else:
                            st.warning("⚠️ Interview saved, but email failed to send.")
                    else:
                        st.error("❌ Failed to save interview details.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <div style='display: flex; justify-content: space-between; align-items: flex-start;'>
            <div style='flex: 1; text-align: left;'>
                <h3 style='margin: 0 0 10px 0; color: #0d2f81; font-weight: 700;'>JobMatchAI</h3>
                <p style='font-size: 14px;'>© 2025 <b>Fluminatiion Technologies Pvt Ltd</b><br>
                #404, 2nd Floor, Tech Valley,<br> Bengaluru, India - 560001</p>
            </div>
            <div style='flex: 1; text-align: center;'>
                <h4 style='margin-bottom: 8px; color: #0d2f81; font-weight: 600;'>📞 Contact</h4>
                <p style='font-size: 14px;'>support@404tech.com<br>+91-98765-43210</p>
            </div>
            <div style='flex: 1; text-align: right;'>
                <h4 style='margin-bottom: 8px; color: #0d2f81; font-weight: 600;'>🤖 How to Use JobMatchAI</h4>
                <ol style='font-size: 13px; padding-left: 20px; text-align: left;'>
                    <li>Login with company credentials</li>
                    <li>Upload candidate's PDF CV</li>
                    <li>Let AI match with job descriptions</li>
                    <li>View top job matches</li>
                    <li>Schedule & send interview invites</li>
                </ol>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)