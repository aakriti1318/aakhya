# src/bigquery_uploader.py

import os
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import streamlit as st
from src.utils import sanitize_table_name

def upload_to_bigquery(file):
    try:
        credentials = get_credentials()
        client = bigquery.Client(credentials=credentials, project=credentials.project_id)

        df = read_file(file)
        if df is None:
            return None

        table_id = get_table_id(credentials, file)
        job_config = get_job_config()

        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()

        table = client.get_table(table_id)
        st.success(f"File {file.name} has been uploaded to BigQuery! {table.num_rows} rows loaded.")
        st.session_state.app_state['file_stored_gcp'] = True

        return table_id  # Return the table_id for further use

    except Exception as e:
        st.error(f"An error occurred while uploading {file.name} to BigQuery: {str(e)}")
        st.session_state.app_state['file_stored_gcp'] = False
        return None

def get_credentials():
    if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        return service_account.Credentials.from_service_account_file(
            os.environ['GOOGLE_APPLICATION_CREDENTIALS']
        )
    else:
        credentials_path = 'service-account-key.json'
        return service_account.Credentials.from_service_account_file(credentials_path)

def read_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    elif file.name.endswith('.txt'):
        return pd.read_csv(file, sep='\t')
    else:
        st.error(f"Unsupported file format: {file.name}")
        return None

def get_table_id(credentials, file):
    project_id = credentials.project_id
    dataset_id = "aakhya"
    table_name = sanitize_table_name(file.name.split('.')[0])
    return f"{project_id}.{dataset_id}.{table_name}"

def get_job_config():
    return bigquery.LoadJobConfig(
        autodetect=True,
        source_format=bigquery.SourceFormat.CSV,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )