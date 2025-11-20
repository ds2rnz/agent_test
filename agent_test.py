import streamlit as st
import re
from typing import Iterator, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage

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



def get_ai_response(messages: List[BaseMessage]) -> Iterator[str]:
     """ LangChain ChatOpenAI의 .stream이 산출하는 chunk에서 content만 추출해 yield. 
          Streamlit의 write_stream에 넘길 수 있는 문자열 이터레이터를 반환. """ 
     for chunk in llm.stream(messages):
         text = getattr(chunk, "content", None) 
         if isinstance(text, str) and text:
             yield text

def strip_tool_noise(text: str) -> str:
    """ 모델이 응답 본문에 실수로 내보낼 수 있는 도구/검색 관련 로그를 제거.
      예: query, search_period, Searching..., Calling web search tool 등. """ 
    if not text: 
        return text

    # 코드 펜스 블록 제거
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    drop_patterns = [
        r'\"?\bquery\b\"?\s*:',          # "query": ... / query:
        r'\"?\bsearch_period\b\"?\s*:',  # "search_period": ... / search_period:
        r'^\{.*\}$',                     # 한 줄 JSON 블록
        r'^\[.*\]$',                     # 한 줄 배열
        r'^\s*Calling web search tool.*',
        r'^\s*Initiating web search.*',
        r'^\s*Proceeding to gather.*',
        r'^\s*Searching the web.*',
        r'^\s*Searching\.\.\.*',
        r'^\s*Now fetching.*',
        r'^\s*Wrapping.*',
        r'^\s*Contacting search engine.*',
        r'^\s*I will look this up.*',
        r'^\s*I will.*search.*',
        r'^\s*\{.*\"query\".*\}.*',
        r'^\s*\{.*\"search_period\".*\}.*',
    ]

    filtered_lines = []
    for line in text.splitlines():
        keep = True
        for pat in drop_patterns:
            if re.search(pat, line, flags=re.IGNORECASE):
                keep = False
                break
        if keep:
            filtered_lines.append(line)

    return "\n".join(filtered_lines).strip()



# Streamlit 앱
st.title("💬 GPT-4o Langchain Chat")

# 스트림릿 session_state에 메시지 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage(content=system_prompt_text),  
    st.chat_message("assistant").write("무엇을 도와드릴까요?")
    ]


def render_chat_history(messages: List[BaseMessage]) -> None: 
    for msg in messages: 
        if isinstance(msg, HumanMessage): 
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage): 
            st.chat_message("assistant").write(msg.content) # SystemMessage/ToolMessage는 출력하지 않음



render_chat_history(st.session_state["messages"])

# 사용자 입력 처리
if prompt := st.chat_input():
    st.chat_message("user").write(prompt) # 사용자 메시지 출력
    st.session_state.messages.append(HumanMessage(prompt)) # 사용자 메시지 저장

   # 스트리밍 응답
   stream = get_ai_response(st.session_state["messages"])
   final_text = st.chat_message("assistant").write_stream(stream)

   # 후처리: 도구/검색 로그 제거
   if isinstance(final_text, str):
       final_text = strip_tool_noise(final_text)

# 최종 응답 저장
if final_text:
    st.session_state["messages"].append(AIMessage(content=final_text))   

















