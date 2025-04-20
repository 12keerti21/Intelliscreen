import os
import datetime
import logging
import warnings
import requests
from collections import defaultdict

import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import firebase_admin
from firebase_admin import credentials, firestore
import ollama

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Suppress warnings and noisy logs
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(message)s')

SCORE_THRESHOLD = 30  # Minimum score for scheduling interview


# ------------------- 🔍 UTILITY FUNCTIONS -------------------

# ✅ Function to summarize JD using Ollama
def summarize_jd(jd_text, jd_id, job_title):
    doc_ref = db.collection("jobs").document(str(jd_id))
    existing_doc = doc_ref.get()

    if existing_doc.exists:
        return existing_doc.to_dict().get("summary", "")

    prompt = f"Summarize this job description into key skills, experience, and qualifications:\n\n{jd_text}"
    response = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": prompt}])
    summary = response["message"]["content"]

    doc_ref.set({
        "job_title": job_title,
        "jd_text": jd_text,
        "summary": summary
    })

    return summary


# ✅ Parse CV from PDF
def parse_cv(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


# ✅ Match CV with JD using TF-IDF
def match_candidate(jd_summary, cv_text, jd_id, cv_id):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([jd_summary, cv_text])
    score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0] * 100

    doc_ref = db.collection("candidates").document(f"cv{cv_id}_jd{jd_id}")
    doc_ref.set({
        "cv_id": cv_id,
        "cv_text": cv_text,
        "jd_id": jd_id,
        "score": score
    })

    return score


# ✅ Schedule interview if score is good
def schedule_interview(cv_id, jd_id, score):
    interview_date = datetime.datetime.now() + datetime.timedelta(days=2)
    email_content = (
        f"Email: Dear Candidate {cv_id},\n\n"
        f"For Job {jd_id}, your match score is {score:.2f}%. "
        f"You are invited for an interview on {interview_date.strftime('%B %d, %Y')} at 10 AM.\n\n"
        f"Best Regards,\nHR Team"
    )

    email_ref = db.collection("interviews").document(f"cv{cv_id}_jd{jd_id}")
    email_ref.set({
        "cv_id": cv_id,
        "jd_id": jd_id,
        "email_content": email_content,
        "interview_date": interview_date,
        "score": score
    })

    email_path = f"../output/email_cv{cv_id}_jd{jd_id}.txt"
    with open(email_path, "w", encoding="utf-8") as f:
        f.write(email_content)

    return email_content


# ✅ Generate cover letter using AIML API


# ------------------- 🚀 MAIN EXECUTION -------------------

data_dir = "../data"
cv_folder = os.path.join(data_dir, "CVs1")
jd_df = pd.read_csv(os.path.join(data_dir, "job_description.csv"), encoding='ISO-8859-1')
logging.info(f"✅ Loaded {len(jd_df)} Job Descriptions.")

cv_files = sorted([f for f in os.listdir(cv_folder) if f.endswith(".pdf")])[:10]
cv_texts = {cv_id: parse_cv(os.path.join(cv_folder, cv_file)) for cv_id, cv_file in enumerate(cv_files, 1)}

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

        # ✅ Generate cover letter
       

# ✅ Show top 3 matches & schedule interview if eligible
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

print("🎉 All done!")
