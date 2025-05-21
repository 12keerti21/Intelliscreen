import ollama
import pandas as pd
import json
import os
from pathlib import Path

def summarize_job_descriptions():
    """Summarize job descriptions using Ollama's Gemma model and save results to JSON."""
    
    # Define paths using Path for better cross-platform compatibility
    data_dir = Path("../data")
    jd_file = data_dir / "job_description.csv"
    output_file = data_dir / "jd_summaries.json"
    
    try:
        # Load job descriptions with explicit encoding and error handling
        jd_df = pd.read_csv(jd_file, encoding='ISO-8859-1')
        
        if jd_df.empty:
            print("Error: No data found in the job description file.")
            return
        
        # Validate required columns exist
        required_columns = ["Job Title", "Job Description"]
        if not all(col in jd_df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in jd_df.columns]
            print(f"Error: Missing required columns: {', '.join(missing)}")
            return

        jd_summaries = {}
        
        for index, row in jd_df.iterrows():
            jd_id = index + 1
            job_title = row["Job Title"]
            jd_text = row["Job Description"]
            
            if pd.isna(jd_text) or not jd_text.strip():
                print(f"Skipping empty job description for {job_title}")
                continue
                
            # Create structured prompt for better results
            prompt = f"""
            Analyze this job description and extract the following information in JSON format:
            - Job Title: {job_title}
            - Key Skills: (list of 5-8 technical/hard skills)
            - Experience Requirements: (years and type of experience needed)
            - Qualifications: (education, certifications, etc.)
            - Summary: (2-3 sentence overview of the role)
            
            Job Description:
            {jd_text}
            """
            
            try:
                response = ollama.chat(
                    model="gemma:2b",
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.3}  # Lower temp for more factual responses
                )
                summary = response["message"]["content"]
                
                # Store both raw and structured data
                jd_summaries[jd_id] = {
                    "title": job_title,
                    "description": jd_text,
                    "summary": summary,
                    "structured": parse_summary(summary)  # Additional parsing function
                }
                
                print(f"Summarized JD {jd_id}: {job_title}")
                
            except Exception as e:
                print(f"Error processing JD {jd_id}: {str(e)}")
                continue
                
        # Save results with pretty formatting
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(jd_summaries, f, indent=2)
            
        print(f"Successfully saved {len(jd_summaries)} summaries to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File not found at {jd_file}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

def parse_summary(summary_text):
    """Parse the model's summary into structured data."""
    try:
        # Attempt to parse JSON if the model returned it
        if summary_text.strip().startswith("{"):
            return json.loads(summary_text)
            
        # Fallback parsing for text responses
        structured = {}
        lines = [line.strip() for line in summary_text.split("\n") if line.strip()]
        
        current_section = None
        for line in lines:
            if line.endswith(":"):
                current_section = line[:-1].lower().replace(" ", "_")
                structured[current_section] = []
            elif current_section:
                structured[current_section].append(line)
                
        return structured
        
    except Exception:
        return {"raw_summary": summary_text}

if __name__ == "__main__":
    summarize_job_descriptions()