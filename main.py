# main.py
import streamlit as st
from src.chat_interface import ChatInterface
from src.config import initialize_session_state
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_lottie import st_lottie
import requests

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():
    st.set_page_config(page_title="Aakhya AI Assistant", page_icon="🔍", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
    # .stApp {
    #     background-color: #f0f2f6;
    # }
    .stTextInput > div > div > input {
        color: #000000;
        background-color: #ffffff;
    }

    .status-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .upload-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("Aakhya 🔍")
        st.markdown("##### One-stop solution for training, analyzing, and sharing insights!")
        
        # Lottie Animation
        lottie_url = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
        lottie_json = load_lottie_url(lottie_url)
        st_lottie(lottie_json, speed=1, height=200, key="initial")

        add_vertical_space(2)
        st.markdown("### How to use Aakhya:")
        st.markdown("""
        1. Upload your data file
        2. Train the AI model
        3. Analyze results on BigQuery
        4. Create visualizations
        5. Share insights via email
        """)

    # Main content
    initialize_session_state()
    
    colored_header(
        label="Chat with Aakhya",
        description="Your AI-powered data analysis assistant",
        color_name="blue-70"
    )

    # Two-column layout
    col1, col2, col3 = st.columns([2,0.20,1])

    with col1:
        chat_interface = ChatInterface()
        chat_interface.run()

    with col3:
        # Project Status Card
        with st.container():
            # st.markdown("<div class='status-card'>", unsafe_allow_html=True)
            st.subheader("Project Status")
            for key, value in st.session_state.app_state.items():
                st.write(f"{'✅' if value else '❌'} {key.replace('_', ' ').title()}")
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()