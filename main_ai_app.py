import html
import os

import streamlit as st
from langchain_community.vectorstores import FAISS

from ai_qna_app import (
    ai_answer,
    answer_question,
    edit_image,
    generate_image,
    process1_f,
)
from image_app import (
    display_image_errors,
    display_pending_images,
    is_image_request,
    select_image_size,
)


DOCUMENT_TYPES = ["pdf", "xlsx", "xls", "xlsm", "csv", "pptx", "pptm", "ppt"]
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]


def _short_file_name(file_name: str, limit: int = 31) -> str:
    safe_name = html.escape(file_name)
    return safe_name if len(safe_name) <= limit else f"{safe_name[:limit]}…"


def _render_file_list(files, label: str):
    file_rows = "".join(
        f'<div class="gs-file-item">{index}. {_short_file_name(file.name)}</div>'
        for index, file in enumerate(files[:3], start=1)
    )
    st.markdown(
        f"""
        <div class="gs-file-list">
            <div class="gs-file-count">✓ {len(files)}개 {label} 선택</div>
            {file_rows}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _logout():
    st.session_state.logged_in = False
    st.session_state.user_info = None
    for key in (
        "messages",
        "vectorstore",
        "pending_images",
        "pending_image_errors",
        "uploader1",
        "edit_image_uploader",
        "edit_image_prompt",
    ):
        st.session_state.pop(key, None)
    st.rerun()


def _render_sidebar():
    user_info = st.session_state.get("user_info") or {}
    user_name = html.escape(str(user_info.get("name", "사용자")))
    user_id = html.escape(str(st.session_state.get("logged_in", "")))

    with st.sidebar:
        st.markdown(
            f"""
            <div class="gs-user-card">
                <div class="gs-user-label">SIGNED IN</div>
                <div class="gs-user-name">👤 {user_name}님</div>
                <div class="gs-user-id">새올 ID · {user_id}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("로그아웃", type="secondary", use_container_width=True):
            _logout()

        st.markdown("#### 업무 도구")
        st.caption("필요한 기능을 열어 바로 사용할 수 있습니다.")

        with st.expander("📚 문서 학습", expanded=False):
            st.markdown(
                '<div class="gs-section-note">'
                "PDF·Excel·PowerPoint 내용을 학습해 문서 기반 답변을 제공합니다."
                "</div>",
                unsafe_allow_html=True,
            )
            uploaded_files = st.file_uploader(
                "학습 문서",
                type=DOCUMENT_TYPES,
                accept_multiple_files=True,
                key="uploader1",
                help="한 번에 최대 3개 파일을 선택할 수 있습니다.",
            )

            if uploaded_files:
                _render_file_list(uploaded_files, "문서")
                if len(uploaded_files) > 3:
                    st.warning("문서는 최대 3개까지 선택해 주세요.")

            process_button = st.button(
                "문서 학습 시작",
                key="process1",
                type="primary",
                use_container_width=True,
                disabled=not uploaded_files or len(uploaded_files) > 3,
            )
            st.caption("구형 .ppt 파일은 .pptx로 변환 후 업로드해 주세요.")

        with st.expander("🎨 이미지 편집", expanded=False):
            st.markdown(
                '<div class="gs-section-note">'
                "최대 3장의 이미지를 참고해 배경·색상·구성을 수정합니다."
                "</div>",
                unsafe_allow_html=True,
            )
            edit_images = st.file_uploader(
                "원본 이미지",
                type=IMAGE_TYPES,
                accept_multiple_files=True,
                key="edit_image_uploader",
                help="PNG, JPG, JPEG, WEBP · 최대 3개",
            )

            if edit_images:
                _render_file_list(edit_images, "이미지")
                if len(edit_images) <= 3:
                    preview_columns = st.columns(min(len(edit_images), 3))
                    for index, uploaded_image in enumerate(edit_images[:3]):
                        with preview_columns[index]:
                            st.image(uploaded_image, use_container_width=True)
                else:
                    st.warning("이미지는 최대 3개까지 선택해 주세요.")

            edit_prompt = st.text_area(
                "수정 요청",
                placeholder="예: 배경을 고성 해변으로 바꾸고 맑은 하늘을 추가해 주세요.",
                key="edit_image_prompt",
                height=100,
            )
            edit_size = st.selectbox(
                "결과 이미지 비율",
                options=["auto", "1024x1024", "1536x1024", "1024x1536"],
                format_func=lambda value: {
                    "auto": "자동",
                    "1024x1024": "정사각형 · 1024 × 1024",
                    "1536x1024": "가로형 · 1536 × 1024",
                    "1024x1536": "세로형 · 1024 × 1536",
                }[value],
                key="edit_image_size",
            )
            edit_button = st.button(
                "이미지 편집 시작",
                key="edit_image_button",
                type="primary",
                use_container_width=True,
                disabled=(
                    not edit_images
                    or len(edit_images) > 3
                    or not edit_prompt.strip()
                ),
            )

        with st.expander("이용 안내"):
            st.markdown(
                """
                1. 문서를 학습하면 업로드 자료를 우선 검색합니다.
                2. 일반 질문은 AI가 바로 답변합니다.
                3. “고성 관광 포스터를 만들어줘”처럼 입력하면 이미지를 생성합니다.
                4. 생성·편집한 이미지는 PNG로 내려받을 수 있습니다.
                """
            )

        st.markdown(
            """
            <div class="gs-footer">
                총무행정관 정보관리팀<br>
                Goseong County AI Assistant · v1.1.0
            </div>
            """,
            unsafe_allow_html=True,
        )

    return uploaded_files, process_button, edit_images, edit_prompt, edit_size, edit_button


def _render_header():
    st.markdown(
        """
        <div class="gs-hero">
            <div class="gs-eyebrow">
                <span class="gs-dot"></span>
                GOSEONG COUNTY · AI WORKSPACE
            </div>
            <h1>고성군청 <span>AI 도우미</span></h1>
            <p>
                질문에 답하고, 검색하고, 필요한 이미지를 만들고, 도움되는
                고성군청 직원 전용 AI 업무 공간입니다.
            </p>
            <div class="gs-chip-row">
                <span class="gs-chip">📚 문서 학습</span>
                <span class="gs-chip">🔎 정보 검색</span>
                <span class="gs-chip">🎨 이미지 생성</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_messages():
    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        if role == "system":
            continue

        with st.chat_message(role):
            content = message.get("content")
            if content:
                st.write(content)

            if message.get("image_bytes"):
                st.image(
                    message["image_bytes"],
                    caption=message.get("image_prompt", "생성된 이미지"),
                    width=512,
                )
                st.download_button(
                    "이미지 다운로드",
                    data=message["image_bytes"],
                    file_name=message.get("file_name", "generated_image.png"),
                    mime="image/png",
                    key=message.get("download_key"),
                )


def _append_assistant_message(content: str, is_error: bool = False):
    st.session_state.messages.append({"role": "assistant", "content": content})
    message_box = st.chat_message("assistant")
    if is_error:
        message_box.error(content)
    else:
        message_box.write(content)


def _answer_general_question():
    response = ai_answer(st.session_state.messages)
    ai_response = response["messages"][-1].content
    _append_assistant_message(ai_response)


def show_main_app():
    """고성군청 AI 도우미 메인 화면을 표시합니다."""
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "messages" not in st.session_state:
        user_name = (st.session_state.get("user_info") or {}).get("name", "사용자")
        st.session_state.messages = [
            {
                "role": "system",
                "content": "저는 고성군청 직원을 위해 최선을 다하는 인공지능 도우미입니다.",
            },
            {
                "role": "assistant",
                "content": f"안녕하세요, {user_name}님. 오늘 어떤 업무를 도와드릴까요?",
            },
        ]

    (
        uploaded_files,
        process_button,
        edit_images,
        edit_prompt,
        edit_size,
        edit_button,
    ) = _render_sidebar()

    _render_header()
    _render_messages()

    if process_button:
        learned_vectorstore = process1_f(uploaded_files)
        if learned_vectorstore is not None:
            st.session_state.vectorstore = learned_vectorstore

    if edit_button:
        with st.spinner("이미지를 편집하고 있습니다..."):
            edit_result = edit_image(
                uploaded_image=edit_images,
                prompt=edit_prompt,
                size=edit_size,
                quality="medium",
            )
        if st.session_state.get("pending_images"):
            display_pending_images()
        else:
            st.error(edit_result)
            display_image_errors()

    prompt = st.chat_input(
        "질문을 입력하거나 만들고 싶은 이미지를 설명해 주세요."
    )
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if is_image_request(prompt):
        with st.spinner("요청하신 이미지를 생성하고 있습니다..."):
            try:
                tool_result = generate_image.invoke(
                    {
                        "prompt": prompt,
                        "size": select_image_size(prompt),
                        "quality": "medium",
                    }
                )
                if st.session_state.get("pending_images"):
                    display_pending_images()
                else:
                    _append_assistant_message(str(tool_result), is_error=True)
            except Exception as error:
                _append_assistant_message(
                    f"이미지 생성 중 오류가 발생했습니다: "
                    f"{type(error).__name__}: {error}",
                    is_error=True,
                )
        return

    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is not None:
        with st.spinner("학습된 문서에서 관련 내용을 찾고 있습니다..."):
            answer = answer_question(prompt)

        if not answer or "죄송합니다." in answer or len(answer) < 30:
            with st.spinner("답변을 작성하고 있습니다..."):
                try:
                    _answer_general_question()
                except Exception as error:
                    _append_assistant_message(
                        f"답변 생성 중 오류가 발생했습니다: "
                        f"{type(error).__name__}: {error}",
                        is_error=True,
                    )
        else:
            _append_assistant_message(answer)
    else:
        with st.spinner("답변을 작성하고 있습니다..."):
            try:
                _answer_general_question()
            except Exception as error:
                _append_assistant_message(
                    f"답변 생성 중 오류가 발생했습니다: "
                    f"{type(error).__name__}: {error}",
                    is_error=True,
                )


def load_vectorstore(embedding, persist_directory="C:/faiss_store"):
    """저장된 FAISS 학습 데이터를 불러옵니다."""
    if not os.path.isdir(persist_directory):
        return None

    index_file = os.path.join(persist_directory, "index.faiss")
    pkl_file = os.path.join(persist_directory, "index.pkl")
    if not (os.path.exists(index_file) and os.path.exists(pkl_file)):
        return None

    try:
        vectorstore = FAISS.load_local(
            persist_directory,
            embedding,
            allow_dangerous_deserialization=True,
        )
        st.toast("기존 학습 데이터를 불러왔습니다.", icon="📚")
        return vectorstore
    except Exception as error:
        st.warning(f"기존 학습 데이터 로드 실패: {error}")
        return None
