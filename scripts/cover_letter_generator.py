from transformers import pipeline

generator = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_cover_letter(resume_text, job_description):
    prompt = f"Generate a professional cover letter based on this resume:\n{resume_text}\nAnd this job description:\n{job_description}"
    result = generator(prompt, max_length=512, do_sample=False)
    return result[0]['generated_text']
