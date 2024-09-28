import streamlit as st
import json

def load_api_credentials():
    with open('credentials.json', 'r') as f:
        file = json.load(f)
        return file['token']

def initialize_session_state():
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'show_upload' not in st.session_state:
        st.session_state['show_upload'] = False
    if 'app_state' not in st.session_state:
        st.session_state.app_state = {
            'file_uploaded': False,
            'file_stored_gcp': False,
            'model_trained': False,
            'email_info': False,
            'dashboard_created': False
        }
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []