import requests

def generate_cover_letter(resume_text, job_description):
    prompt = f"""
You are a professional HR assistant. Write a personalized and high-quality cover letter based on the following:

Resume:
{resume_text}

Job Description:
{job_description}

Make it confident, engaging, and highlight how the candidate is a strong fit. Avoid copying the job description directly.
"""

    API_KEY = "599105cbc41a4f6fb6d32c3fb8a57911"  # Replace with your AIML API key
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gemma:2b",
        "prompt": prompt,
        "max_tokens": 500
    }

    # ✅ FIXED endpoint spelling here
    response = requests.post("https://api.aimlapi.com/v1/completion", headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["data"]["text"]
    else:
        return f"⚠️ API Error {response.status_code}: {response.text}"
