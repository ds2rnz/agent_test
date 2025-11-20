import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from langchain_core.tools import tool
from datetime import datetime
import pytz

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

system_prompt_text = """
당신은 고성군청 직원을 위한 친절한 고성군청 AI 도우미입니다.

1. 직원들이 질문하면 구체적이고 자세하게 설명해주세요 .
2. 모르는 내용이면 도구를 이용하여 인터넷 검색을 꼭해서 답변해주세요.
3. 인터넷 검색에 대하여 링크를 표시해 주세요.
4. 이 지역은 강원도 고성군입니다.
   - 고성군청 주소는 강원특별자치도 고성군 간성읍 고성중앙길9입니다.
5. 강원도 고성군 관련 관광지 질문이 들어오면 아래 홈페이지를 참고하여 답해주세요.
   - 고성군 관광포털 사이트 : https://gwgs.go.kr/tour/index.do
6. 강원도 고성군 고성군청에 관하여 질문이 들어오면 아래 홈페이지를 참고하여 답해주세요
   - 고성군청 홈페이지 : https://gwgs.go.kr
7. 고성군수는 함명준입니다.
   - 고성군수는 고성군 발전을 위하여 노력하시는분입니다.
8. 고성군청 ai 도우미는 고성군청 총무행정관 정보관리팀에서 agent를 제작하였습니다.
   - langchain을 기반으로 제작하였으며, RAG기술과 학습기능을 탐재하였으며, 지속적으로 기능추가 예정입니다.
9. 한글로 답해주세요
"""


# 모델 초기화
llm = ChatOpenAI(model="gpt-5")


# def get_ai_response(messages):
#     for chunk in llm.stream(messages):
#         text = getattr(chunk, "content", None)
#         if isinstance(text, str) and text:
#            yield text

def get_ai_response(messages):
    messages = llm.stream(messages)
    return messages



# Streamlit 앱
st.title("💬 GPT-4o Langchain Chat")

# 스트림릿 session_state에 메시지 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage(content=system_prompt_text),  
        AIMessage("How can I help you?")
    ]

# 스트림릿 화면에 메시지 출력
for msg in st.session_state.messages:
    if msg.content:
        if isinstance(msg, SystemMessage):
            st.chat_message("system").write("저는 사용자를 돕기 위해 최선을 다하는 고성군청 AI 도우미입니다")
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, ToolMessage):
            st.chat_message("tool").write(msg.content)


# 사용자 입력 처리
if prompt := st.chat_input():
    st.chat_message("user").write(prompt) # 사용자 메시지 출력
    st.session_state.messages.append(HumanMessage(prompt)) # 사용자 메시지 저장

    response = get_ai_response(st.session_state["messages"])
    
    result = st.chat_message("assistant").write_stream(response) # AI 메시지 출력
    st.session_state["messages"].append(AIMessage(result)) # AI 메시지 저장 




















