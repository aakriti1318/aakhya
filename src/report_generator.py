from google.generativeai import client
import google.generativeai as genai
import datetime
import pandas as pd
import random
import string
import datetime
import json 

client.api_key = 'AIzaSyDhCOUW8tp3HFMYUJRjSi2mSk__VO607BI'
    
def get_record_count(data):
    return len(data)

def get_time_period(data):
    if 'date' in data.columns:
        min_date = pd.to_datetime(data['date']).min()
        max_date = pd.to_datetime(data['date']).max()
        return f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
    else:
        return "Unknown time period"

def generate_report_id():
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"RPT{timestamp}{random_str}"

def get_dynamic_report_data(data, classification, looker_url, dataset_name="Unnamed Dataset"):
    record_count = get_record_count(data)
    time_period = get_time_period(data)
    report_id = generate_report_id()
    date_generated = datetime.datetime.now().strftime("%Y-%m-%d")

    executive_summary = f"This report provides an analysis of {classification} data..."
    key_findings = "The analysis revealed several key findings including..."
    detailed_analysis = "A detailed breakdown of the data shows..."
    visualizations = "1. Key visualizations derived from the dataset."
    recommendations = "Based on the analysis, we recommend..."
    methodology = "The methodology used in this analysis includes..."
    data_quality = "Data quality was assessed and any issues are mentioned here..."
    appendices = "Appendix A, B, C..."
    revision_history = "Initial version created on " + date_generated

    report_data = {
        "report_id": report_id,
        "date_generated": date_generated,
        "classification": classification,
        "executive_summary": executive_summary,
        "dataset_name": dataset_name,
        "record_count": record_count,
        "time_period": time_period,
        "key_findings": key_findings,
        "detailed_analysis": detailed_analysis,
        "looker_url": looker_url,
        "visualizations": visualizations,
        "recommendations": recommendations,
        "methodology": methodology,
        "data_quality": data_quality,
        "appendices": appendices,
        "revision_history": revision_history
    }
    
    return report_data

def generate_report_gemini(data, classification, looker_url, dataset_name="Unnamed Dataset"):
    report_data = get_dynamic_report_data(data, classification, looker_url, dataset_name)
    
    prompt_template = """
    ISO-Compliant Data Analysis Report
    ==================================
    
    Report ID: {report_id}
    Date Generated: {date_generated}
    Classification: {classification}
    
    1. Executive Summary
    --------------------
    {executive_summary}
    
    2. Data Overview
    ----------------
    Dataset: {dataset_name}
    Records Analyzed: {record_count}
    Time Period: {time_period}
    
    3. Key Findings
    ---------------
    {key_findings}
    
    4. Detailed Analysis
    --------------------
    {detailed_analysis}
    
    5. Visualizations
    -----------------
    Looker Studio Dashboard: {looker_url}
    
    Key Visualizations:
    {visualizations}
    
    6. Recommendations
    ------------------
    {recommendations}
    
    7. Methodology
    --------------
    {methodology}
    
    8. Data Quality and Limitations
    -------------------------------
    {data_quality}
    
    9. Appendices
    -------------
    {appendices}
    
    10. Revision History
    -------------------
    {revision_history}
    
    End of Report
    """
    
    filled_prompt = prompt_template.format(**report_data)
    
    # Initialize the model
    model = genai.GenerativeModel('gemini-1.0-pro')
    
    # Generate the content
    response = model.generate_content(filled_prompt)
    
    # Return the generated text
    return response.text

