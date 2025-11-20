import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from datetime import datetime

import pytz
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from dotenv import load_dotenv
import os

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyPDFLoader
 #from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import ast

# from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma
# from langchain.agents import PromptTemplate
from langchain_community.vectorstores import FAISS
from openai import OpenAI

import concurrent.futures
import traceback
import inspect
import time

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

llm = ChatOpenAI(
    model="gpt-5",
    temperature=0.4,
    timeout=30,  # 30초 타임아웃
    max_retries=2 ) 




def debug_wrap(func):
    """함수 실행 시 에러나 중단점을 추적하기 위한 디버깅 래퍼"""
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        try:
            st.write(f"[DEBUG] ▶ 실행 시작: {func_name}")
            result = func(*args, **kwargs)
            st.write(f"[DEBUG] ✅ 실행 성공: {func_name}")
            return result
        except Exception as e:
            tb = traceback.format_exc()
            st.write(f"\n[ERROR] ❌ 함수 '{func_name}' 에서 예외 발생:")
            st.write(f"  └─ {e}")
            st.write(tb)
            st.error(f"❌ 함수 '{func_name}' 실행 중 오류 발생: {e}")
            st.code(tb, language="python")
            raise
    return wrapper




# -- 도구 정의 --
# @tool
# def get_current_time(timezone: str, location: str) -> str:
#     """현재 시간을 지정된 타임존과 위치에 맞게 반환합니다."""
#     import pytz
#     from datetime import datetime
#     try:
#         tz = pytz.timezone(timezone)
#         now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
#         return f'{timezone} ({location}) 현재시각 {now}'
#     except pytz.UnknownTimeZoneError:
#         return f"알 수 없는 타임존: {timezone}"

# @tool
# def get_web_search(query: str, search_period: str) -> str:
#     """DuckDuckGo API를 이용해 지정된 기간 내의 뉴스를 검색하여 결과를 반환합니다."""
#     wrapper = DuckDuckGoSearchAPIWrapper(region="kr-kr", time=search_period)
#     search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news", results_separator=';\n')
#     return search.invoke(query)

# tools = [get_current_time, get_web_search]
# tool_dict = {tool.name: tool for tool in tools}
# llm_with_tools = llm.bind_tools(tools)


@debug_wrap
def get_ai_response(messages):
    response = llm.invoke(messages)
    return response





# --- Streamlit 앱 설정 ---
st.set_page_config(page_title="AI 도우미", page_icon="💬", layout="wide")

st.title("💬 고성군청 :blue[AI] 도우미")


# --- 화면 디자인 ---
st.markdown("""
    <style>
    /* 기본 바디 폰트 및 배경 */
    body {
        background-color: #f0f2f6;
        font-family: 'Noto Sans KR', sans-serif;
        color: #333;
    }

    
     /* 입력창 스타일 */
    .stChatInput input {
        border: 2px solid #3b82f6;
        border-radius: 25px;
        padding: 15px 25px;
        font-size: 16px;
        background: linear-gradient(to right, #f0f9ff, #ffffff);
        transition: all 0.3s ease;
    }
    
    .stChatInput input:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        background: white;
    }
    
    .stChatInput button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        border-radius: 50%;
        transition: transform 0.3s ease;
    }
    
    .stChatInput button:hover {
        transform: scale(1.1) rotate(15deg);
    }

    </style>
""", unsafe_allow_html=True)

animated_input_css = """
    <style>
    /* 입력창 등장 애니메이션 */
    .stChatInput {
        animation: slide-up 0.5s ease-out;
    }
    
    @keyframes slide-up {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* 타이핑 효과 */
    .stChatInput input:focus {
        animation: typing-glow 2s ease-in-out infinite;
    }
    
    @keyframes typing-glow {
        0%, 100% { box-shadow: 0 0 5px rgba(59, 130, 246, 0.3); }
        50% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.6); }
    }
    
    /* 버튼 회전 효과 */
    .stChatInput button:hover {
        animation: rotate 0.5s ease;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    </style>
"""

st.markdown(animated_input_css, unsafe_allow_html=True)

   

# 스트림릿 session_state에 메시지 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("저는 고성군청 직원을 위해 최선을 다하는 인공지능 도우미입니다. "),  
        AIMessage("무엇을 도와 드릴까요?")
    ]

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

# 스트림릿 화면에 메시지 출력
for msg in st.session_state.messages:
    if msg.content:
        if isinstance(msg, SystemMessage):
            st.chat_message("system").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)


# 사용자 입력 처리
if prompt := st.chat_input(placeholder="✨ 무엇이든 물어보세요?"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(prompt))

    # vectorstore 존재 여부 확인
    if st.session_state.get("vectorstore") is not None:
        # 벡터스토어 기반 답변
        st.write("📚 학습된 문서를 기반으로 답변합니다...")
        answer = get_ai_response(prompt)
        
    else:
        st.write("🤖 일반 AI 모드로 답변합니다...")
        response = get_ai_response(st.session_state["messages"])
        result = st.chat_message("assistant").write_stream(response)
        st.session_state.messages.append(AIMessage(result)) 
else:
    # 기존 도구 결합 LLM 답변
    st.write("🤖 일반 AI 모드로 답변합니다...")
    response = llm.invoke(st.session_state["messages"])
    result = st.chat_message("assistant").write_stream(response)
    st.session_state.messages.append(AIMessage(result)) 











