# chatbot/llm_handler.py

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def generate_response_with_gemini(query, docs, second_chance=False):
    """
    - If docs is non-empty: build a prompt listing the user query + relevant case excerpts, then call Gemini.
    - If docs is empty:
        - If second_chance=False: return a message asking for a legal question.
        - If second_chance=True: call Gemini directly on the raw user query (no case context).
    """

    if not docs:
        if not second_chance:
            return "Please ask me a question related to law or cases."
        else:
            prompt = f"User Query (non-legal): {query['topic']}\nAnswer:"
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text

    prompt = f"User Query: {query['topic']}\nRelevant Cases:\n"
    for d in docs:
        prompt += f"- {d.get('case_name', '')}: {d.get('text', '')[:500]}\n"
    prompt += "\nAnswer:"

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text
