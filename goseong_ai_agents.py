import streamlit as st
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from datetime import datetime
import pytz
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from dotenv import load_dotenv
import os
from langchain.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from pathlib import Path
import tempfile
import traceback
import time
from users_db import USERS_DB
from main_ai_app import show_main_app     # ai agent 메인 함수
from login_app import show_login_page, check_login      # 로그인 함수



# create_agent 관련 tool 함수 / 시간, 웹검색


@tool
def get_current_time(timezone: str, location: str) -> str:
    '''  해당 지역 현재시각을 구하는 함수 '''
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        result = f'{timezone} ({location}) 현재시각 {now}'
        return result
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"  

@tool
def get_web_search(query: str) -> str:
    """
    웹 검색을 수행하는 함수.

    Args:
        query (str): 검색어
    Returns:
        str: 검색 결과
    """
    custom_wrapper = DuckDuckGoSearchAPIWrapper(region="kr-kr", time="y", max_results=10)
    search = DuckDuckGoSearchResults(
        api_wrapper=custom_wrapper,
        source="news, image, text",
        results_separator=';\n')
    
    results = search.run(query)

    st.toast("웹 검색을 통아여 알아보고 있습니다.", icon="🎉")
    return results




# ==================== 메인 실행 ====================


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

config = {"configurable": {"thread_id": "1"}}

llm = init_chat_model(
    model = "openai:gpt-4o",
    temperature=0.6, 
    max_tokens=1000, 
    timeout=10, 
    max_retries=2, 
    )

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large", 
    api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    )

agent = create_agent(
    model=llm,
    tools=[get_current_time, get_web_search],
    middleware=[],
    system_prompt="사용자가 질문을하면 구체적이고 자세하게 설명해주고 모르는 내용이면 인터넷 검색을 꼭해서 답변해줘 그리고 한글로 답해주세요", 
    )

# 페이지 설정
st.set_page_config(page_title="GPT 기반 AI 도우미", page_icon="💬", layout="wide")

# 세션 상태 초기화
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

# 로그인 상태에 따라 페이지 표시
if not st.session_state.logged_in:
    show_login_page()
else:
    show_main_app()
