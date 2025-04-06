import ollama
import pandas as pd
import json
import os

# Paths
data_dir = "../data"
jd_file = os.path.join(data_dir, "job_description.csv")
output_file = "../data/jd_summaries.json"

# Load JDs and summarize
jd_df = pd.read_csv(jd_file, encoding='ISO-8859-1')
jd_summaries = {}
for index, row in jd_df.iterrows():
    jd_id = index + 1
    job_title = row["Job Title"]
    jd_text = row["Job Description"]
    prompt = f"Summarize this job description into key skills, experience, and qualifications: {jd_text}"
    response = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": prompt}])
    summary = response["message"]["content"]
    jd_summaries[jd_id] = {"title": job_title, "text": jd_text, "summary": summary}
    print(f"Summarized JD {jd_id}: {job_title}")

# Save to JSON
with open(output_file, "w") as f:
    json.dump(jd_summaries, f)
print(f"Saved summaries to {output_file}")