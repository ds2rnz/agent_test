from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from config import apply_global_styles
from login_app import show_login_page
from main_ai_app import show_main_app

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="고성군청 AI 도우미",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_global_styles()


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_info" not in st.session_state:
    st.session_state.user_info = None


if st.session_state.logged_in:
    show_main_app()
else:
    show_login_page()
