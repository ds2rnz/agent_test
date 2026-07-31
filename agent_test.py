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
import base64
from openai import OpenAI
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


@tool
def generate_image(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "medium"
) -> str:
    """
    사용자의 설명을 바탕으로 새로운 이미지를 생성합니다.

    사용자가 그림, 포스터, 일러스트, 사진, 배너, 캐릭터 등의
    생성을 요청했을 때 사용합니다.

    Args:
        prompt: 생성할 이미지에 대한 구체적인 설명
        size: 이미지 크기. 1024x1024, 1536x1024, 1024x1536 중 하나
        quality: 이미지 품질. low, medium, high 중 하나
    """

    allowed_sizes = {
        "1024x1024",
        "1536x1024",
        "1024x1536",
    }
    allowed_qualities = {"low", "medium", "high"}

    if size not in allowed_sizes:
        size = "1024x1024"

    if quality not in allowed_qualities:
        quality = "medium"

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        result = client.images.generate(
            model="gpt-image-2",
            prompt=prompt,
            size=size,
            quality=quality,
            output_format="png",
            n=1,
        )

        image_base64 = result.data[0].b64_json

        if not image_base64:
            return "이미지 데이터가 반환되지 않았습니다."

        image_bytes = base64.b64decode(image_base64)

        # 도구 실행 결과를 Streamlit 화면 출력용으로 임시 보관
        if "pending_images" not in st.session_state:
            st.session_state.pending_images = []

        st.session_state.pending_images.append({
            "prompt": prompt,
            "image_bytes": image_bytes,
        })

        return (
            "이미지를 성공적으로 생성했습니다. "
            "이미지는 사용자의 채팅 화면에 표시됩니다."
        )

    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"

        if "pending_image_errors" not in st.session_state:
            st.session_state.pending_image_errors = []

        st.session_state.pending_image_errors.append(error_message)

        return f"이미지 생성에 실패했습니다. 오류: {error_message}"


# ==================== 메인 실행 ====================


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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