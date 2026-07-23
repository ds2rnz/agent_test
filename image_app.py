import streamlit as st
from langchain_openai import ChatOpenAI
from datetime import datetime
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
import pytz
from pathlib import Path
import tempfile
import traceback
import time
import base64
from openai import OpenAI

def display_image_errors():
    errors = st.session_state.pop("pending_image_errors", [])

    for error in errors:
        st.error(f"❌ 이미지 생성 실패\n\n{error}") 


def display_pending_images():
    """이미지 도구가 만든 이미지를 채팅 기록에 추가하고 표시합니다."""

    pending_images = st.session_state.pop("pending_images", [])

    for image_info in pending_images:
        file_name = f"generated_{int(time.time() * 1000)}.png"
        download_key = f"download_{time.time_ns()}"

        image_message = {
            "role": "assistant",
            "content": "🎨 요청하신 이미지를 생성했습니다.",
            "image_bytes": image_info["image_bytes"],
            "image_prompt": image_info["prompt"],
            "file_name": file_name,
            "download_key": download_key,
        }

        st.session_state.messages.append(image_message)

        with st.chat_message("assistant"):
            st.write(image_message["content"])

            st.image(
                image_info["image_bytes"],
                caption=image_info["prompt"],
                # use_container_width=True,
                width = 512,
            )

            st.download_button(
                label="📥 이미지 다운로드",
                data=image_info["image_bytes"],
                file_name=file_name,
                mime="image/png",
                key=download_key,
            )

IMAGE_KEYWORDS = (
    "이미지",
    "그림",
    "사진",
    "포스터",
    "배너",
    "일러스트",
    "캐릭터",
    "로고",
    "썸네일",
    "그려줘",
    "만들어줘",
    "생성해줘",
)


def is_image_request(prompt: str | None) -> bool:
    """사용자 입력이 이미지 생성 요청인지 확인합니다."""

    if not isinstance(prompt, str) or not prompt.strip():
        return False

    normalized_prompt = prompt.strip().lower()

    image_words = (
        "이미지",
        "그림",
        "사진",
        "포스터",
        "배너",
        "일러스트",
        "캐릭터",
        "로고",
        "썸네일",
    )

    action_words = (
        "그려",
        "만들어",
        "생성",
        "제작",
    )

    return (
        any(word in normalized_prompt for word in image_words)
        and any(word in normalized_prompt for word in action_words)
    )


def select_image_size(prompt: str | None) -> str:
    if not isinstance(prompt, str):
        return "1024x1024"

    normalized_prompt = prompt.strip().lower()

    if any(word in normalized_prompt for word in ("가로", "배너", "와이드")):
        return "1536x1024"

    if any(word in normalized_prompt for word in ("세로", "스토리", "휴대폰")):
        return "1024x1536"

    return "1024x1024"