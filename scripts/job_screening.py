import os
import time
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

# Terminal color codes for enhanced display
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Initialize Firebase
cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Suppress warnings and noisy logs
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(message)s')

SCORE_THRESHOLD = 30  # Minimum score for scheduling interview


# ------------------- 🔍 UTILITY FUNCTIONS -------------------

# Animation function for loading effects
def loading_animation(message, duration=1.5):
    animation_chars = "|/-\\"
    end_time = time.time() + duration
    i = 0
    
    print()  # Start with empty line
    while time.time() < end_time:
        print(f"\r{Colors.CYAN}{message} {animation_chars[i % len(animation_chars)]}{Colors.ENDC}", end="")
        time.sleep(0.1)
        i += 1
    
    print(f"\r{Colors.GREEN}✅ {message} Done!{Colors.ENDC}")
    time.sleep(0.3)

# ✅ Function to summarize JD using Ollama
def summarize_jd(jd_text, jd_id, job_title):
    doc_ref = db.collection("jobs").document(str(jd_id))
    existing_doc = doc_ref.get()

    if existing_doc.exists:
        return existing_doc.to_dict().get("summary", "")

    loading_animation(f"Summarizing job {jd_id}: {job_title}")
    
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
    os.makedirs(os.path.dirname(email_path), exist_ok=True)
    with open(email_path, "w", encoding="utf-8") as f:
        f.write(email_content)

    return email_content


# Animated score bar visualization
def display_score_bar(score, width=30):
    filled_width = int(score * width / 100)
    
    # Choose color based on score
    if score >= 70:
        color = Colors.GREEN
    elif score >= 50:
        color = Colors.YELLOW
    else:
        color = Colors.RED
        
    bar = f"{color}{'█' * filled_width}{Colors.ENDC}{'░' * (width - filled_width)}"
    return f"[{bar}] {color}{score:.2f}%{Colors.ENDC}"


# ------------------- 🚀 MAIN EXECUTION -------------------

def animate_title():
    title = "HR CANDIDATE MATCHING SYSTEM"
    print("\n")
    for i in range(len(title) + 1):
        print(f"\r{Colors.HEADER}{Colors.BOLD}{title[:i]}{Colors.ENDC}", end="")
        time.sleep(0.03)
    print("\n" + "=" * len(title))
    time.sleep(0.5)

animate_title()

data_dir = "../data"
cv_folder = os.path.join(data_dir, "CVs1")
jd_df = pd.read_csv(os.path.join(data_dir, "job_description.csv"), encoding='ISO-8859-1')
print(f"{Colors.BLUE}✅ Loaded {Colors.BOLD}{len(jd_df)}{Colors.ENDC}{Colors.BLUE} Job Descriptions.{Colors.ENDC}")

loading_animation("Scanning CV directory")
cv_files = sorted([f for f in os.listdir(cv_folder) if f.endswith(".pdf")])[:10]

print(f"{Colors.BLUE}📄 Processing {Colors.BOLD}{len(cv_files)}{Colors.ENDC}{Colors.BLUE} CV files{Colors.ENDC}")
cv_texts = {}
for cv_id, cv_file in enumerate(cv_files, 1):
    print(f"\r{Colors.CYAN}⏳ Parsing CV {cv_id}/{len(cv_files)}: {cv_file}{Colors.ENDC}", end="")
    cv_texts[cv_id] = parse_cv(os.path.join(cv_folder, cv_file))
    time.sleep(0.2)  # Small delay for visual effect
print(f"\n{Colors.GREEN}✅ All CVs parsed successfully!{Colors.ENDC}\n")

all_matches = []
cv_matches = defaultdict(list)

print(f"{Colors.HEADER}{Colors.BOLD}🔍 Starting CV-JD Matching Process{Colors.ENDC}")
for index, row in jd_df.iterrows():
    jd_id = index + 1
    job_title = row["Job Title"]
    jd_text = row["Job Description"]
    
    print(f"\n{Colors.BLUE}🔍 Processing Job {jd_id}: {Colors.BOLD}{job_title}{Colors.ENDC}")
    jd_summary = summarize_jd(jd_text, jd_id, job_title)

    for cv_id, cv_text in cv_texts.items():
        print(f"\r{Colors.CYAN}  Matching CV {cv_id} with Job {jd_id}...{Colors.ENDC}", end="")
        score = match_candidate(jd_summary, cv_text, jd_id, cv_id)
        all_matches.append((cv_id, jd_id, job_title, score))
        cv_matches[cv_id].append((jd_id, job_title, score))
        time.sleep(0.1)  # Small delay for animation effect
    
    print(f"\r{Colors.GREEN}  ✅ Completed matching all CVs with Job {jd_id}{Colors.ENDC}")

# Animated results display
print(f"\n\n{Colors.HEADER}{Colors.BOLD}📊 FINAL MATCHING RESULTS{Colors.ENDC}")
print(f"{Colors.HEADER}{'=' * 50}{Colors.ENDC}\n")
time.sleep(0.5)

for cv_id in sorted(cv_matches):
    top3 = sorted(cv_matches[cv_id], key=lambda x: x[2], reverse=True)[:3]
    
    print(f"{Colors.BOLD}{Colors.YELLOW}👤 CANDIDATE {cv_id} TOP MATCHES:{Colors.ENDC}")
    print(f"{Colors.YELLOW}{'─' * 40}{Colors.ENDC}")
    
    email_sent = False
    for rank, (jd_id, job_title, score) in enumerate(top3, 1):
        # Animation effect for each score bar
        print(f"   {Colors.BLUE}#{rank}{Colors.ENDC} JD {jd_id} - {Colors.BOLD}{job_title}{Colors.ENDC}")
        
        # Animate the score bar
        empty_bar = f"[{'░' * 30}] 0.00%"
        print(f"      {empty_bar}", end="\r")
        
        # Gradually fill the bar
        steps = 10
        for i in range(steps + 1):
            temp_score = score * i / steps
            filled_width = int(temp_score * 30 / 100)
            
            if temp_score >= 70:
                color = Colors.GREEN
            elif temp_score >= 50:
                color = Colors.YELLOW
            else:
                color = Colors.RED
                
            temp_bar = f"{color}{'█' * filled_width}{Colors.ENDC}{'░' * (30 - filled_width)}"
            print(f"      [{temp_bar}] {color}{temp_score:.2f}%{Colors.ENDC}", end="\r")
            time.sleep(0.03)
        
        # Final score bar
        print(f"      {display_score_bar(score)}")
        
        if score >= SCORE_THRESHOLD:
            schedule_interview(cv_id, jd_id, score)
            email_sent = True
    
    if email_sent:
        print(f"\n   {Colors.GREEN}✅ Interview scheduled for Candidate {cv_id}{Colors.ENDC}")
    else:
        print(f"\n   {Colors.RED}❌ No interview scheduled for Candidate {cv_id} (all scores < {SCORE_THRESHOLD}%){Colors.ENDC}")
    
    print(f"{Colors.YELLOW}{'─' * 40}{Colors.ENDC}\n")
    time.sleep(0.3)  # Pause between candidates

# Animated completion message
print(f"{Colors.HEADER}{'=' * 50}{Colors.ENDC}")
completion_msg = "🎉 ALL PROCESSING COMPLETED SUCCESSFULLY! 🎉"
for i in range(len(completion_msg) + 1):
    print(f"\r{Colors.GREEN}{Colors.BOLD}{completion_msg[:i]}{Colors.ENDC}", end="")
    time.sleep(0.02)
print("\n")