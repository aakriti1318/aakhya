# src/ai_model.py

import google.generativeai as genai
from src.config import load_api_credentials

def generate_response(prompt):
    token = load_api_credentials()
    genai.configure(api_key=token)

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)

    return response.text if hasattr(response, 'text') else str(response)