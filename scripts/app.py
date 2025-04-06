import streamlit as st
import pdfplumber
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
data_dir = "../data"
jd_summaries_file = os.path.join(data_dir, "jd_summaries.json")

# Load pre-computed JDs
@st.cache_data
def load_jd_summaries():
    with open(jd_summaries_file, "r") as f:
        return json.load(f)

# Parse CV
def parse_cv(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        cv_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return cv_text

# Match CV to JD
def match_candidate(jd_summary, cv_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([jd_summary, cv_text])
    score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0] * 100
    return score

# Streamlit Interface
st.set_page_config(page_title="Job Screening AI", page_icon="🚀", layout="wide")

# Header
st.markdown("""
    <style>
    .title { font-size: 40px; color: #1E90FF; font-weight: bold; text-align: center; }
    .subtitle { font-size: 20px; color: #4682B4; text-align: center; }
    .jd-box { background-color: #F0F8FF; padding: 10px; border-radius: 10px; margin-bottom: 10px;color: black; }
    .score-box { background-color: #E6F3FF; padding: 10px; border-radius: 10px; color: black;}
    </style>
    <div class='title'>Job Screening AI</div>
    <div class='subtitle'>Find Your Perfect Job Match with AI Precision</div>
""", unsafe_allow_html=True)

# Load JDs
jd_summaries = load_jd_summaries()

# Display JDs
st.subheader("Explore Our Job Openings")
col1, col2 = st.columns(2)
for jd_id, jd_data in jd_summaries.items():
    jd_id_int = int(jd_id)  # Convert string to integer
    if jd_id_int % 2 == 1:
        with col1:
            with st.expander(f"📋 {jd_data['title']}"):
                st.markdown(f"<div class='jd-box'>"
                            f"<b>Description:</b> {jd_data['text']}<br>"
                            f"<b>Summary:</b> {jd_data['summary']}"
                            f"</div>", unsafe_allow_html=True)
    else:
        with col2:
            with st.expander(f"📋 {jd_data['title']}"):
                st.markdown(f"<div class='jd-box'>"
                            f"<b>Description:</b> {jd_data['text']}<br>"
                            f"<b>Summary:</b> {jd_data['summary']}"
                            f"</div>", unsafe_allow_html=True)

# CV Upload Section
st.subheader("Upload Your CV")
st.write("Submit your CV (PDF) to see how you match with our job openings!")
uploaded_cv = st.file_uploader("Choose a PDF file", type="pdf", help="Upload a single PDF CV")

# Process CV
if uploaded_cv:
    with st.spinner("Processing your CV..."):
        cv_text = parse_cv(uploaded_cv)
    st.success("CV parsed successfully!")

    # Match against all JDs
    scores = []
    st.subheader("Your Compatibility Scores")
    for jd_id, jd_data in jd_summaries.items():
        score = match_candidate(jd_data["summary"], cv_text)
        scores.append((jd_id, jd_data["title"], score))
        st.markdown(f"<div class='score-box'>"
                    f"JD {jd_id} - {jd_data['title']}: <b>{score:.2f}%</b>"
                    f"</div>", unsafe_allow_html=True)

    # Top 3 Matches
    top_matches = sorted(scores, key=lambda x: x[2], reverse=True)[:3]
    st.subheader("Your Top 3 Matches")
    for rank, (jd_id, title, score) in enumerate(top_matches, 1):
        st.write(f"{rank}. JD {jd_id} - {title}: **{score:.2f}%**")

# Footer
st.markdown("<hr><div style='text-align: center; color: #696969;'></div>", unsafe_allow_html=True)