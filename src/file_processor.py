import pandas as pd
from io import BytesIO
from src.ai_model import generate_response
from src.looker_url_generator import LookerURLGenerator
import json

def load_config():
    with open('config.json') as config_file:
        return json.load(config_file)

def process_file(uploaded_file):
    # Determine file type and read accordingly
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(BytesIO(uploaded_file.getvalue()))
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(BytesIO(uploaded_file.getvalue()))
    else:
        raise ValueError("Unsupported file type")

    # Get the first 5 rows
    preview_data = df.head()

    # Convert preview data to string for Gemini input
    preview_str = preview_data.to_string()

    # Use the existing generate_response function to classify the data
    prompt = f"""
    Given the following data preview, classify it as either "cosmetic" or "other" sector:

    {preview_str}

    Respond with only one word: either "cosmetic" or "other".
    """

    classification = generate_response(prompt).strip().lower()

    # Ensure the classification is valid
    if classification not in ["cosmetic", "other"]:
        classification = "other"  # Default to "other" if classification is unclear

    # Get the appropriate Looker Studio URL
    url_generator = LookerURLGenerator()
    looker_url = url_generator.get_looker_url(classification)

    return preview_data, classification, looker_url