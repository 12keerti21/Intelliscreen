import streamlit as st
import pdfplumber
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#st.set_page_config(page_title="Job Screening AI", page_icon="🚀", layout="wide")

import datetime
import yagmail
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv  # For secure credential management

# Load environment variables
load_dotenv()

# Paths
data_dir = "../data"
jd_summaries_file = os.path.join(data_dir, "jd_summaries.json")

# Initialize vectorizer globally for better performance
vectorizer = TfidfVectorizer()

# Load pre-computed JDs with error handling
@st.cache_data
def load_jd_summaries():
    try:
        with open(jd_summaries_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Job descriptions database not found.")
        return {}
    except json.JSONDecodeError:
        st.error("Invalid job descriptions format.")
        return {}

# Parse CV with better error handling
def parse_cv(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            cv_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return cv_text
    except Exception as e:
        st.error(f"Failed to parse CV: {str(e)}")
        return None

# Match CV to JD using pre-initialized vectorizer
def match_candidate(jd_summary, cv_text):
    try:
        tfidf_matrix = vectorizer.fit_transform([jd_summary, cv_text])
        score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0] * 100
        return score
    except Exception as e:
        st.error(f"Matching error: {str(e)}")
        return 0

# Streamlit Interface
st.set_page_config(page_title="Job Screening AI", page_icon="🚀", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #f9fafb;
        font-family: 'Segoe UI', sans-serif;
    }

    .title {
        font-size: 48px;
        color: #0a58ca;
        font-weight: 800;
        text-align: center;
        margin-top: 30px;
    }

    .subtitle {
        font-size: 22px;
        color: #495057;
        text-align: center;
        margin-bottom: 40px;
    }

    .jd-box {
        background-color: #ffffff;
        padding: 20px;
        border-left: 6px solid #0a58ca;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        color: #212529;
    }

    .score-box {
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-weight: 600;
        color: #212529;
    }

    .high-match {
        background-color: #d1e7dd;
        border-left: 5px solid #0f5132;
    }

    .medium-match {
        background-color: #fff3cd;
        border-left: 5px solid #664d03;
    }

    .low-match {
        background-color: #f8d7da;
        border-left: 5px solid #842029;
    }

    .footer {
        text-align: center;
        font-size: 14px;
        color: #adb5bd;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)


# Load JDs
jd_summaries = load_jd_summaries()

if not jd_summaries:
    st.warning("No job descriptions available. Please check the database.")
else:
    # Display JDs in columns
    st.subheader("Explore Our Job Openings")
    col1, col2 = st.columns(2)
    
    for jd_id, jd_data in jd_summaries.items():
        jd_id_int = int(jd_id)
        container = col1 if jd_id_int % 2 == 1 else col2
        
        with container:
            with st.expander(f"📋 {jd_data['title']} (JD {jd_id})", expanded=False):
                st.markdown(f"<div class='jd-box'>"
                           f"<b>Description:</b> {jd_data['text']}<br><br>"
                           f"<b>Summary:</b> {jd_data['summary']}"
                           f"</div>", unsafe_allow_html=True)

# CV Upload Section
st.subheader("Upload Your CV")
uploaded_cv = st.file_uploader("Choose a PDF file", type="pdf", 
                              help="Upload a single PDF CV (max 10MB)",
                              accept_multiple_files=False)

if uploaded_cv:
    if uploaded_cv.size > 10 * 1024 * 1024:  # 10MB limit
        st.error("File size too large. Please upload a PDF smaller than 10MB.")
    else:
        with st.spinner("Analyzing your CV..."):
            cv_text = parse_cv(uploaded_cv)
        
        if cv_text:
            st.success("CV analysis complete!")
            
            # Match against all JDs
            scores = []
            st.subheader("Your Compatibility Scores")
            
            for jd_id, jd_data in jd_summaries.items():
                score = match_candidate(jd_data["summary"], cv_text)
                scores.append((jd_id, jd_data["title"], score))
                
                # Determine score class for styling
                if score >= 70:
                    score_class = "high-match"
                elif score >= 40:
                    score_class = "medium-match"
                else:
                    score_class = "low-match"
                
                st.markdown(f"<div class='score-box {score_class}'>"
                           f"<b>JD {jd_id} - {jd_data['title']}:</b> {score:.2f}%"
                           f"</div>", unsafe_allow_html=True)

            # Top 3 Matches with visualization
            if scores:
                top_matches = sorted(scores, key=lambda x: x[2], reverse=True)[:3]
                
                st.subheader("Your Best Matches")
                fig, ax = plt.subplots()
                pd.DataFrame({
                    'Job': [f"{title[:20]}..." for _, title, _ in top_matches],
                    'Score': [score for _, _, score in top_matches]
                }).plot.barh(x='Job', y='Score', ax=ax, color='skyblue')
                ax.set_xlim(0, 100)
                st.pyplot(fig)
                
                # Interview scheduling for high matches
                if any(score >= 70 for _, _, score in scores):
                    with st.expander("📅 Schedule an Interview", expanded=False):
                        candidate_email = st.text_input("Your Email Address")
                        interview_date = st.date_input("Preferred Date", 
                                                      min_value=datetime.date.today())
                        interview_time = st.time_input("Preferred Time")
                        message = st.text_area("Additional Notes", 
                                              "Please let us know any special requirements...")
                        
                        if st.button("Request Interview"):
                            if not candidate_email or "@" not in candidate_email:
                                st.warning("Please enter a valid email address")
                            else:
                                try:
                                    yag = yagmail.SMTP(os.getenv("EMAIL_USER"), 
                                                      os.getenv("EMAIL_PASS"))
                                    full_message = f"""
                                    Dear Candidate,
                                    
                                    Thank you for your application. We'd like to invite you for an interview.
                                    
                                    Proposed Interview Details:
                                    - Date: {interview_date}
                                    - Time: {interview_time}
                                    
                                    Your Notes:
                                    {message}
                                    
                                    Best regards,
                                    Hiring Team
                                    """
                                    yag.send(
                                        to=candidate_email,
                                        subject="Interview Invitation",
                                        contents=full_message
                                    )
                                    st.success("Invitation sent successfully!")
                                except Exception as e:
                                    st.error(f"Failed to send email: {str(e)}")

# Analytics Dashboard

# ==============================================
# ANALYTICS DASHBOARD
# ==============================================
st.markdown("---")  # Visual separator
with st.expander("📊 Recruitment Analytics Dashboard", expanded=False):
    st.subheader("Hiring Metrics Overview")
    
    if jd_summaries and uploaded_cv:  # Only show if JDs exist and CV is uploaded
        # Get matching scores data from earlier
        scores_data = [(jd_id, jd_data['title'], match_candidate(jd_data["summary"], cv_text)) 
                      for jd_id, jd_data in jd_summaries.items()]

        # Check if scores_data has elements
        if scores_data:
            # Extract data into separate lists
            job_roles = [title for _, title, _ in scores_data]
            match_scores = [score for _, _, score in scores_data]

            # Ensure consistent length for company_avg_scores and open_positions
            # Make sure these lists have the same number of items as `scores_data`
            company_avg_scores = [70, 65, 80][:len(scores_data)]  # Example benchmark data
            open_positions = [3, 2, 1][:len(scores_data)]  # Example positions data
            
            # Print the lengths to debug
            st.write(f"Length of job_roles: {len(job_roles)}")
            st.write(f"Length of match_scores: {len(match_scores)}")
            st.write(f"Length of company_avg_scores: {len(company_avg_scores)}")
            st.write(f"Length of open_positions: {len(open_positions)}")
            
            # Check if the lengths match
            if len(job_roles) == len(match_scores) == len(company_avg_scores) == len(open_positions):
                # Create the DataFrame
                metrics = pd.DataFrame({
                    'Job Role': job_roles,
                    'Your Match Score': match_scores,
                    'Company Avg Score': company_avg_scores,
                    'Open Positions': open_positions
                })
                
                # Visualization columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Your Match Scores**")
                    st.bar_chart(metrics.set_index('Job Role')['Your Match Score'])
                    
                with col2:
                    st.write("**Comparison to Company Average**")
                    st.bar_chart(metrics[['Job Role', 'Your Match Score', 'Company Avg Score']].set_index('Job Role'))
                
                # Additional metrics table
                st.write("**Position Availability**")
                st.table(metrics[['Job Role', 'Open Positions', 'Company Avg Score']])
            else:
                # Handle the case where lengths do not match
                st.error("Mismatch in the lengths of data lists. Please check the data source.")
        
        else:
            st.warning("No scores data available. Please check your input.")
        
    elif not uploaded_cv:
        st.info("Upload a CV to see personalized analytics")
    else:
        st.warning("No job description data available")
from cover_letter_generator import generate_cover_letter

cl_tab, *_ = st.tabs(["Cover Letter Generator", "Other Tabs..."])

with cl_tab:
    resume_input = st.text_area("Paste Resume Text")
    jd_input = st.text_area("Paste Job Description")
    
    if st.button("Generate Cover Letter"):
        with st.spinner("Generating Cover Letter..."):
            letter = generate_cover_letter(resume_input, jd_input)
            st.code(letter)

        with st.spinner("Matching Resume with JD..."):
            def run_matching():
             pass  # replace this with actual implementation


    
# ==============================================
# FOOTER
# ==============================================
st.markdown("""
    <hr style="margin-top: 50px;">
    <div class='footer'>
        Job Screening AI — All Rights Reserved © 2025
    </div>
""", unsafe_allow_html=True)
