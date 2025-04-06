# Enhancing Job Screening with AI and Data Intelligence

Welcome to the project repository for "Enhancing Job Screening with AI and Data Intelligence." This solution automates job screening by parsing CVs, summarizing job descriptions with AI, and matching candidates using TF-IDF similarity. It includes a batch processing script, a pre-computation tool, and an interactive Streamlit UI.

## Project Overview
- **Idea Title**: AI-Powered Job Screening Revolution
- **Team Name**: [Your Name] (or your team name, e.g., "AI Innovators")
- **Problem Statement**: Manual screening of job applications is time-consuming and inefficient, especially for large datasets (e.g., 20 job descriptions and 200 CVs). This leads to delays in hiring and potential mismatches between candidates and roles.
- **Proposed Solution**: An AI-powered system that parses CVs, summarizes job descriptions using the Gemma:2b model, and matches candidates to roles with TF-IDF. Features include a Streamlit interface for real-time uploads, batch processing for bulk analysis, and email scheduling for top matches (≥30% score).

## Project Structure
JobScreening/
├── data/
│   ├── job_description.csv    # 20 job descriptions
│   ├── CVs1/                 # 200 CV PDFs
│   └── jd_summaries.json     # Pre-computed JD summaries
├── scripts/
│   ├── job_screening.py      # Batch processing script
│   ├── precompute_summaries.py # JD summarization script
│   └── app.py               # Streamlit UI script
├── output/
│   ├── recruitment.db       # SQLite database
│   ├── demo_log.txt         # Batch processing log
│   └── email_cvX_jdY.txt    # Scheduled interview emails
├── requirements.txt         # Python dependencies
├── README.txt               # This file