import ollama
import sqlite3
import pandas as pd
import pdfplumber
import os
import datetime
import logging
import warnings
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# To Suppress warnings
warnings.filterwarnings("ignore")

# Reduce verbosity of ollama HTTP requests
logging.getLogger("httpx").setLevel(logging.WARNING)

# Simple clean logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

SCORE_THRESHOLD = 30  # You can vary this threshold

with sqlite3.connect("../output/recruitment.db") as conn:
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY, 
            job_title TEXT, 
            jd_text TEXT, 
            summary TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            cv_id INTEGER, 
            cv_text TEXT, 
            jd_id INTEGER, 
            score REAL, 
            UNIQUE(cv_id, jd_id)
        )
    """)
    logging.info("SQLite tables ready.")

    def summarize_jd(jd_text, jd_id, job_title):
        # Check if summary already exists
        cursor.execute("SELECT summary FROM jobs WHERE id = ?", (jd_id,))
        result = cursor.fetchone()
        if result and result[0]:
            return result[0]  

        prompt = f"Summarize this job description into key skills, experience, and qualifications: {jd_text}"
        response = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": prompt}])
        summary = response["message"]["content"]
        cursor.execute("INSERT OR REPLACE INTO jobs (id, job_title, jd_text, summary) VALUES (?, ?, ?, ?)",
                       (jd_id, job_title, jd_text, summary))
        return summary

    def parse_cv(pdf_path):
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    def match_candidate(jd_summary, cv_text, jd_id, cv_id):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([jd_summary, cv_text])
        score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0] * 100
        cursor.execute("INSERT OR IGNORE INTO candidates (cv_id, cv_text, jd_id, score) VALUES (?, ?, ?, ?)",
                       (cv_id, cv_text, jd_id, score))
        return score

    def schedule_interview(cv_id, jd_id, score):
        interview_date = datetime.datetime.now() + datetime.timedelta(days=2)
        email_content = (
            f"Email: Dear Candidate {cv_id},\n\n"
            f"For Job {jd_id}, your match score is {score:.2f}%. "
            f"You are invited for an interview on {interview_date.strftime('%B %d, %Y')} at 10 AM.\n\n"
            f"Best Regards,\nHR Team"
        )
        email_path = f"../output/email_cv{cv_id}_jd{jd_id}.txt"
        with open(email_path, "w") as f:
            f.write(email_content)
        return email_content

    # Main Execution 
    data_dir = "../data"
    cv_folder = os.path.join(data_dir, "CVs1")
    jd_df = pd.read_csv(os.path.join(data_dir, "job_description.csv"), encoding='ISO-8859-1')
    logging.info(f"Loaded {len(jd_df)} Job Descriptions.")

    # Parse CVs (first 10)
    cv_files = sorted([f for f in os.listdir(cv_folder) if f.endswith(".pdf")])[:10]
    cv_texts = {}
    for cv_id, cv_file in enumerate(cv_files, 1):
        cv_texts[cv_id] = parse_cv(os.path.join(cv_folder, cv_file))

    # Matching logic
    all_matches = []
    cv_matches = defaultdict(list)

    for index, row in jd_df.iterrows():
        jd_id = index + 1
        job_title = row["Job Title"]
        jd_text = row["Job Description"]
        jd_summary = summarize_jd(jd_text, jd_id, job_title)

        for cv_id, cv_text in cv_texts.items():
            score = match_candidate(jd_summary, cv_text, jd_id, cv_id)
            all_matches.append((cv_id, jd_id, job_title, score))
            cv_matches[cv_id].append((jd_id, job_title, score))

    # Top 3 matches per candidate
    print("\n📊 Final Matching Results:\n")

    for cv_id in sorted(cv_matches):
        top3 = sorted(cv_matches[cv_id], key=lambda x: x[2], reverse=True)[:3]
        print(f"🧾 Candidate {cv_id} Top Matches:")
        email_sent = False
        for rank, (jd_id, job_title, score) in enumerate(top3, 1):
            print(f"   {rank}. JD {jd_id} - {job_title} ({score:.2f}%)")
            if score >= SCORE_THRESHOLD:
                schedule_interview(cv_id, jd_id, score)
                email_sent = True
        if email_sent:
            print(f"✅ Interview scheduled for Candidate {cv_id}\n")
        else:
            print(f"❌ No interview scheduled for Candidate {cv_id} (all scores < {SCORE_THRESHOLD}%)\n")

    print(" All done!\n")
