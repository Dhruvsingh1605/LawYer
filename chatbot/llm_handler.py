import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_response_with_gemini(query, docs):
    prompt = f"User Query: {query['topic']}\nRelevant Cases:\n"
    for d in docs:
        prompt += f"- {d.get('case_name', '')}: {d.get('plain_text', '')[:500]}\n"

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text
