import streamlit as st
from langchain_openai import ChatOpenAI
from datetime import datetime
import pytz
import streamlit as st
from langchain.tools import tool
import pytz
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from pathlib import Path
from users_db import USERS_DB
from main_ai_app import show_main_app     # ai agent 메인 함수
from login_app import show_login_page, check_login      # 로그인 함수


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")




# 모델 초기화
llm = ChatOpenAI(model="gpt-5")


# ==================== 메인 실행 ====================

config = {"configurable": {"thread_id": "1"}}

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large", 
    api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    )

# 페이지 설정
st.set_page_config(page_title="GPT 기반 AI 도우미", page_icon="💬", layout="wide")

# 세션 상태 초기화
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

# 로그인 상태에 따라 페이지 표시
# if not st.session_state.logged_in:
#    show_login_page()
# else:
    show_main_app()





