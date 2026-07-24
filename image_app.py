import time

import streamlit as st


def display_image_errors():
    """대기 중인 이미지 오류를 화면에 표시합니다."""
    for error in st.session_state.pop("pending_image_errors", []):
        st.error(f"이미지 처리에 실패했습니다.\n\n{error}")


def display_pending_images():
    """생성·편집된 이미지를 채팅 기록에 한 번만 추가하고 표시합니다."""
    for image_info in st.session_state.pop("pending_images", []):
        timestamp = time.time_ns()
        file_name = f"goseong_ai_image_{timestamp}.png"
        download_key = f"download_{timestamp}"
        image_message = {
            "role": "assistant",
            "content": "요청하신 이미지가 완성되었습니다.",
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
                width=512,
            )
            st.download_button(
                "이미지 다운로드",
                data=image_info["image_bytes"],
                file_name=file_name,
                mime="image/png",
                key=download_key,
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
    action_words = ("그려", "만들어", "생성", "제작")
    return (
        any(word in normalized_prompt for word in image_words)
        and any(word in normalized_prompt for word in action_words)
    )


def select_image_size(prompt: str | None) -> str:
    """문장에 포함된 비율 표현으로 이미지 크기를 선택합니다."""
    if not isinstance(prompt, str):
        return "1024x1024"

    normalized_prompt = prompt.strip().lower()
    if any(word in normalized_prompt for word in ("가로", "배너", "와이드")):
        return "1536x1024"
    if any(word in normalized_prompt for word in ("세로", "스토리", "휴대폰")):
        return "1024x1536"
    return "1024x1024"
