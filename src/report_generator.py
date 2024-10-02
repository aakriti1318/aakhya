import google.generativeai as genai
import datetime
import pandas as pd
import random
import string
import numpy as np

# Configure the API key
genai.configure(api_key='AIzaSyDhCOUW8tp3HFMYUJRjSi2mSk__VO607BI')

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

    # Calculate some basic statistics
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    stats = data[numeric_columns].describe().to_dict()

    executive_summary = f"""
    This report provides a comprehensive analysis of {classification} data from the {dataset_name} dataset. 
    The analysis covers {record_count} records{f' over the period of {time_period}' if time_period != "Unknown time period" else ''}. 
    Key findings and actionable recommendations are provided based on thorough data examination and statistical analysis.
    """

    # Determine time span category
    if time_period != "Unknown time period":
        try:
            start_date, end_date = time_period.split(" to ")
            time_span = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            time_category = ['short-term', 'medium-term', 'long-term'][min(2, time_span // 365)]
        except:
            time_category = "unknown duration"
    else:
        time_category = "unknown duration"

    key_findings = f"""
    1. Data Volume: The dataset contains {record_count} records, providing a {['limited', 'moderate', 'substantial'][min(2, record_count // 1000)]} sample size for analysis.
    2. Time Span: {"Data covers " + time_period if time_period != "Unknown time period" else "Time period is unknown"}, allowing for {time_category} trend analysis.
    3. Key Metrics: 
       {', '.join([f"{col}: mean = {stats[col]['mean']:.2f}, std = {stats[col]['std']:.2f}" for col in list(stats.keys())[:3]])}
    4. [Additional finding based on data patterns]
    5. [Additional finding based on data patterns]
    """

    detailed_analysis = f"""
    1. Descriptive Statistics:
       {data[numeric_columns].describe().to_string()}
    
    2. Correlation Analysis:
       [Insert insights from correlation analysis]
    
    3. Trend Analysis:
       [Insert insights from trend analysis over the time period]
    
    4. Segment Analysis:
       [Insert insights from segmentation of data]
    
    5. Outlier Detection:
       [Insert findings related to outliers in the dataset]
    """

    visualizations = f"""
    1. Time Series Plot: Showing the trend of key metrics over time.
    2. Correlation Heatmap: Visualizing the relationships between different variables.
    3. Distribution Plots: Histograms and box plots for key numeric variables.
    4. Scatter Plots: Exploring relationships between important variable pairs.
    5. Pie Charts or Bar Graphs: Representing the distribution of categorical variables.
    """

    recommendations = """
    Based on the analysis, we recommend:
    1. [Action item based on key finding 1]
    2. [Action item based on key finding 2]
    3. [Action item based on key finding 3]
    4. [Action item based on detailed analysis]
    5. [Long-term strategic recommendation]
    """

    methodology = f"""
    The analysis employed the following methodologies:
    1. Descriptive Statistics: To summarize central tendencies and variability in the data.
    2. Time Series Analysis: To identify trends and patterns over the period {time_period}.
    3. Correlation Analysis: To uncover relationships between different variables.
    4. Segmentation: To identify distinct groups or patterns within the dataset.
    5. Outlier Detection: To identify and analyze data points that significantly differ from other observations.
    """

    data_quality = f"""
    Data Quality Assessment:
    1. Completeness: {data.isnull().sum().sum()} missing values identified across all fields.
    2. Consistency: [Insert findings about data consistency]
    3. Accuracy: [Insert notes on data accuracy, if any issues were found]
    4. Timeliness: Data covers {time_period}, which is considered [current/slightly outdated/outdated] for this analysis.
    
    Limitations:
    1. [Insert any limitations of the dataset or analysis]
    2. [Insert any potential biases in the data]
    """

    appendices = """
    Appendix A: Detailed Methodology
    Appendix B: Full Statistical Output
    Appendix C: Data Dictionary
    Appendix D: Supplementary Visualizations
    """

    revision_history = f"Initial version created on {date_generated}"

    return {
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

def generate_report_gemini(data, classification, looker_url, dataset_name="Unnamed Dataset"):
    report_data = get_dynamic_report_data(data, classification, looker_url, dataset_name)
    
    prompt_template = """
    You are an expert data analyst tasked with creating a comprehensive ISO-compliant data analysis report. 
    Use the provided information to generate a detailed, insightful report. Expand on the given points, 
    add specific details where appropriate, and ensure the report is coherent and professional.

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