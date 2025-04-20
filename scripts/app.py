import streamlit as st
import pdfplumber
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import yagmail
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
data_dir = "../data"
jd_summaries_file = os.path.join(data_dir, "jd_summaries.json")

# Initialize vectorizer globally
vectorizer = TfidfVectorizer()

# Load JD summaries with caching
@st.cache_data
def load_jd_summaries():
    try:
        with open(jd_summaries_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.error("⚠️ Job descriptions database not found or invalid.")
        return {}

# CV parser
def parse_cv(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        st.error(f"Failed to parse CV: {str(e)}")
        return None

# TF-IDF Matcher
def match_candidate(jd_summary, cv_text):
    try:
        tfidf_matrix = vectorizer.fit_transform([jd_summary, cv_text])
        return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0] * 100
    except Exception as e:
        st.error(f"Matching error: {str(e)}")
        return 0

# Set page config
st.set_page_config(page_title="Job Screening AI", page_icon="🚀", layout="wide")

# Custom CSS (optional external file)
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load JD data
jd_summaries = load_jd_summaries()

# Title
st.markdown("""
    <div style='padding: 2rem 0;'>
        <h1 style='text-align: center;'>🚀 AI-Powered Job Screening Platform</h1>
        <p style='text-align: center; color: gray;'>Find your best fit job match instantly using AI</p>
    </div>
""", unsafe_allow_html=True)

# Section: Job Roles
if jd_summaries:
    st.subheader("🔍 Explore Open Roles")
    cols = st.columns(2)
    for idx, (jd_id, jd_data) in enumerate(jd_summaries.items()):
        with cols[idx % 2]:
            with st.expander(f"📋 {jd_data['title']} (JD {jd_id})"):
                st.markdown(f"""
                    **Description:** {jd_data['text']}

                    **Summary:** {jd_data['summary']}
                """)
else:
    st.warning("No job descriptions available. Please check the database.")

# Section: Upload CV
st.markdown("---")
st.subheader("📁 Upload Your CV")
uploaded_cv = st.file_uploader("Upload PDF CV (max 10MB)", type="pdf")

if uploaded_cv:
    if uploaded_cv.size > 10 * 1024 * 1024:
        st.error("File too large. Limit is 10MB.")
    else:
        with st.spinner("Parsing your CV..."):
            cv_text = parse_cv(uploaded_cv)

        if cv_text:
            st.success("✅ CV parsed successfully!")
            st.subheader("📊 Compatibility Scores")

            scores = [(jd_id, jd_data['title'], match_candidate(jd_data['summary'], cv_text))
                      for jd_id, jd_data in jd_summaries.items()]

            for jd_id, title, score in scores:
                if score >= 70:
                    st.success(f"{title} (JD {jd_id}): {score:.2f}%")
                elif score >= 40:
                    st.warning(f"{title} (JD {jd_id}): {score:.2f}%")
                else:
                    st.error(f"{title} (JD {jd_id}): {score:.2f}%")

            # Top Matches Graph
            top_matches = sorted(scores, key=lambda x: x[2], reverse=True)[:3]
            top_df = pd.DataFrame({
                "Job": [title[:30] + "..." for _, title, _ in top_matches],
                "Score": [score for _, _, score in top_matches]
            })
            fig, ax = plt.subplots()
            top_df.plot.barh(x='Job', y='Score', ax=ax, color='mediumseagreen')
            ax.set_xlim(0, 100)
            ax.set_title("Top Matching Roles")
            st.pyplot(fig)

            # Interview Scheduler
            if any(score >= 70 for _, _, score in scores):
                with st.expander("📅 Schedule Interview"):
                    email = st.text_input("Candidate Email")
                    date = st.date_input("Interview Date", min_value=datetime.date.today())
                    time = st.time_input("Interview Time")
                    notes = st.text_area("Notes", "Available for online or offline interview.")

                    if st.button("📧 Send Interview Invitation"):
                        if not email or "@" not in email:
                            st.warning("Enter a valid email address.")
                        else:
                            try:
                                yag = yagmail.SMTP(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
                                yag.send(to=email, subject="Interview Invitation",
                                         contents=f"Interview scheduled on {date} at {time}.\n\n{notes}")
                                st.success("✅ Email invitation sent successfully!")
                            except Exception as e:
                                st.error(f"Email sending failed: {str(e)}")
st.markdown("""
    <style>
        /* General Styling */
        body {
            background-color: #f4f7fa;
            color: #333;
            font-family: 'Roboto', sans-serif;
        }

        /* Header */
        h1 {
            color: #1a73e8;
            font-family: 'Montserrat', sans-serif;
            font-size: 3rem;
        }

        h2, h3 {
            color: #333;
            font-family: 'Poppins', sans-serif;
        }

        /* Subheader */
        .subheader {
            color: #4CAF50;
            font-family: 'Roboto', sans-serif;
        }

        /* Buttons */
        button, .stButton {
            background-color: #1a73e8;
            color: #fff;
            font-family: 'Poppins', sans-serif;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
            transition: background-color 0.3s;
        }

        button:hover, .stButton:hover {
            background-color: #166bb5;
        }

        /* Input Fields */
        .stTextInput, .stFileUploader, .stTextArea, .stDateInput, .stTimeInput {
            background-color: #fff;
            border-radius: 5px;
            border: 1px solid #ccc;
            padding: 10px;
            margin-top: 5px;
        }

        /* Card Elements */
        .stExpanderHeader {
            font-family: 'Montserrat', sans-serif;
            font-weight: bold;
            color: #333;
            background-color: #e0e7ff;
            padding: 10px;
        }

        .stExpanderContent {
            padding: 15px;
            background-color: #f9fafb;
        }

        /* Footer */
        footer {
            background-color: #2d2d2d;
            color: #fff;
            padding: 20px 0;
            text-align: center;
            font-size: 0.9rem;
        }

        footer a {
            color: #fff;
            text-decoration: none;
            font-weight: bold;
        }

        /* Compatibility Scores */
        .stBarChart .stChart {
            background-color: #e0e7ff;
            border-radius: 5px;
        }

        .stProgressBar {
            background-color: #1a73e8;
        }

        /* Success / Error Messages */
        .stSuccess, .stWarning, .stError {
            font-family: 'Poppins', sans-serif;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }

        .stSuccess {
            background-color: #4CAF50;
            color: #fff;
        }

        .stWarning {
            background-color: #ff9800;
            color: #fff;
        }

        .stError {
            background-color: #f44336;
            color: #fff;
        }

        /* Adding Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500&family=Poppins:wght@400;600&family=Roboto:wght@400;500&display=swap');
    </style>
""", unsafe_allow_html=True)

# Analytics Section
st.markdown("---")
with st.expander("📈 Recruitment Analytics"):
    if jd_summaries and uploaded_cv:
        scores_data = [(jd_id, jd_data['title'], match_candidate(jd_data['summary'], cv_text))
                       for jd_id, jd_data in jd_summaries.items()]
        metrics = pd.DataFrame({
            "Job Role": [title for _, title, _ in scores_data],
            "Your Match": [score for _, _, score in scores_data],
            "Benchmark": [60] * len(scores_data),
            "Openings": [2] * len(scores_data)
        })
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Your Compatibility Scores**")
            st.bar_chart(metrics.set_index("Job Role")["Your Match"])
        with col2:
            st.write("**Match vs Benchmark**")
            st.bar_chart(metrics.set_index("Job Role")[["Your Match", "Benchmark"]])

        st.dataframe(metrics[['Job Role', 'Openings']], use_container_width=True)

# Cover Letter Generator Tab
from cover_letter_generator import generate_cover_letter

with st.expander("📝 Generate Cover Letter"):
    st.subheader("Custom Cover Letter Tool")
    resume_input = st.text_area("Paste Your Resume Text")
    jd_input = st.text_area("Paste the Job Description")
    if st.button("🧠 Generate Cover Letter"):
        with st.spinner("Crafting letter..."):
            letter = generate_cover_letter(resume_input, jd_input)
            st.code(letter)

# Footer
st.markdown("""
    <hr style='margin-top: 2rem;'>
    <div style='text-align: center; color: gray;'>
        Job Screening AI — All Rights Reserved © 2025
    </div>
""", unsafe_allow_html=True)
