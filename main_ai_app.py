import html
import os
import re
import streamlit as st
from langchain_community.vectorstores import FAISS
from ai_qna_app import (
    analyze_images,
    ai_answer,
    answer_question,
    edit_image,
    generate_image,
    process1_f)
from image_app import (
    display_image_errors,
    display_pending_images,
    is_image_edit_request,
    is_image_request,
    select_image_size)
from pdf_converter import PdfConversionError, convert_pdf
import logging


logger = logging.getLogger(__name__)



DOCUMENT_TYPES = ["pdf", "xlsx", "xls", "xlsm", "csv", "pptx", "pptm", "ppt"]
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]
CHAT_FILE_TYPES = DOCUMENT_TYPES + IMAGE_TYPES


def _get_requested_conversion_formats(prompt: str) -> set[str]:
    """사용자가 요청한 PDF 변환 형식(DOCX/TXT)을 반환합니다."""
    normalized = re.sub(r"\s+", "", (prompt or "").lower())
    if not normalized:
        return set()

    conversion_terms = ("변환", "바꿔", "바꾸", "만들", "저장")
    if not any(term in normalized for term in conversion_terms):
        return set()

    requested_formats = set()
    if any(term in normalized for term in ("docx", "워드", "word")):
        requested_formats.add("docx")
    if any(term in normalized for term in ("txt", "텍스트", "text")):
        requested_formats.add("txt")
    return requested_formats


def _short_file_name(file_name: str, limit: int = 31) -> str:
    """ 화면에 업드로 파일명 표시 함수 """
    safe_name = html.escape(file_name)
    return safe_name if len(safe_name) <= limit else f"{safe_name[:limit]}…"


def _render_file_list(files, label: str):
    """ 사이드바에 선택된 파일 목록 표시 함수 """
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
        "pdf_converter_uploader",
        "pdf_conversion_results",
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

        with st.expander("📄 PDF 파일 변환", expanded=False):
            st.markdown(
                '<div class="gs-section-note">'
                "PDF의 텍스트를 추출해 DOCX와 TXT 파일로 변환합니다."
                "</div>",
                unsafe_allow_html=True,
            )
            conversion_files = st.file_uploader(
                "변환할 PDF",
                type=["pdf"],
                accept_multiple_files=True,
                key="pdf_converter_uploader",
                help="한 번에 최대 3개 · 파일당 50MB 이하",
            )
            if conversion_files:
                _render_file_list(conversion_files, "PDF")
                if len(conversion_files) > 3:
                    st.warning("PDF는 최대 3개까지 선택해 주세요.")

            conversion_button = st.button(
                "DOCX·TXT 변환",
                key="pdf_conversion_button",
                type="primary",
                use_container_width=True,
                disabled=(
                    not conversion_files
                    or len(conversion_files) > 3
                ),
            )

            if conversion_button:
                results = []
                with st.spinner("PDF를 변환하고 있습니다..."):
                    for uploaded_pdf in conversion_files:
                        try:
                            results.append(
                                convert_pdf(
                                    uploaded_pdf.getvalue(),
                                    uploaded_pdf.name,
                                )
                            )
                        except PdfConversionError as error:
                            st.error(f"{uploaded_pdf.name}: {error}")
                        except Exception:
                            logger.exception("PDF 변환 중 예기치 않은 오류")
                            st.error(
                                f"{uploaded_pdf.name}: 변환 중 오류가 발생했습니다."
                            )
                st.session_state.pdf_conversion_results = results

            for index, result in enumerate(
                st.session_state.get("pdf_conversion_results", [])
            ):
                st.success(
                    f"{result.source_name} · {result.page_count}페이지 변환 완료"
                )
                if result.text_page_count < result.page_count:
                    st.warning(
                        "일부 페이지에서 텍스트를 추출하지 못했습니다. "
                        "스캔된 페이지는 OCR이 필요할 수 있습니다."
                    )
                download_columns = st.columns(2)
                with download_columns[0]:
                    st.download_button(
                        "DOCX 다운로드",
                        data=result.docx_bytes,
                        file_name=result.docx_name,
                        mime=(
                            "application/vnd.openxmlformats-officedocument."
                            "wordprocessingml.document"
                        ),
                        key=f"pdf_docx_download_{index}_{result.docx_name}",
                        use_container_width=True,
                    )
                with download_columns[1]:
                    st.download_button(
                        "TXT 다운로드",
                        data=result.txt_bytes,
                        file_name=result.txt_name,
                        mime="text/plain; charset=utf-8",
                        key=f"pdf_txt_download_{index}_{result.txt_name}",
                        use_container_width=True,
                    )

        with st.expander("이용 안내"):
            st.markdown(
                """
                1. 문서를 학습하면 업로드 자료를 우선 검색합니다.
                2. 일반 질문은 AI가 바로 답변합니다.
                3. “고성 관광 포스터를 만들어줘”처럼 입력하면 이미지를 생성합니다.
                4. 생성·편집한 이미지는 PNG로 내려받을 수 있습니다.
                5. PDF 파일 변환에서 DOCX와 TXT를 함께 내려받을 수 있습니다.
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
                <span class="gs-chip">🔎 정보 검색</span>
                <span class="gs-chip">📚 문서 학습</span>
                <span class="gs-chip">🎨 이미지 생성·편집</span>
                <span class="gs-chip">📄 PDF 파일 변환(DOCX, TXT)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_conversion_downloads(message: dict, message_index: int):
    """대화 메시지에 저장된 PDF 변환 결과와 다운로드 버튼을 표시합니다."""
    conversions = message.get("conversions") or []
    for conversion_index, result in enumerate(conversions):
        st.success(
            f"{result['source_name']} · {result['page_count']}페이지 변환 완료"
        )
        if result["text_page_count"] < result["page_count"]:
            st.warning(
                "일부 페이지에서 텍스트를 추출하지 못했습니다. "
                "스캔된 페이지는 OCR이 필요할 수 있습니다."
            )
        requested_formats = set(result.get("requested_formats") or ("docx",))
        download_columns = st.columns(len(requested_formats))

        column_index = 0
        if "docx" in requested_formats:
            with download_columns[column_index]:
                st.download_button(
                    "DOCX 다운로드",
                    data=result["docx_bytes"],
                    file_name=result["docx_name"],
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"
                    ),
                    key=f"chat_docx_{message_index}_{conversion_index}",
                    use_container_width=False,
                )
            column_index += 1

        if "txt" in requested_formats:
            with download_columns[column_index]:
                st.download_button(
                    "TXT 다운로드",
                    data=result["txt_bytes"],
                    file_name=result["txt_name"],
                    mime="text/plain; charset=utf-8",
                    key=f"chat_txt_{message_index}_{conversion_index}",
                    use_container_width=False,
                )


def _render_messages():
    """ 저장된 대화 출력 함수 """
    for message_index, message in enumerate(st.session_state.messages):
        role = message.get("role", "assistant")
        if role == "system":
            continue

        with st.chat_message(role):
            content = message.get("content")
            if content:
                st.write(content)

            attachments = message.get("attachments") or []
            if attachments:
                st.caption(
                    "첨부: " + ", ".join(item["name"] for item in attachments)
                )
                image_attachments = [
                    item for item in attachments if item.get("image_bytes")
                ]
                if image_attachments:
                    columns = st.columns(min(len(image_attachments), 3))
                    for index, item in enumerate(image_attachments):
                        with columns[index % len(columns)]:
                            st.image(
                                item["image_bytes"],
                                caption=item["name"],
                                use_container_width=True,
                            )

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
            _render_conversion_downloads(message, message_index)


def _append_assistant_message(
    content: str,
    is_error: bool = False,
    conversions: list[dict] | None = None,
):
    """ 답변 저장 및 출력 함수 """
    message = {"role": "assistant", "content": content}
    if conversions:
        message["conversions"] = conversions
    st.session_state.messages.append(message)
    message_box = st.chat_message("assistant")
    if is_error:
        message_box.error(content)
    else:
        message_box.write(content)
        if conversions:
            _render_conversion_downloads(
                message,
                len(st.session_state.messages) - 1,
            )


def _convert_chat_pdfs(document_files, requested_formats: set[str]):
    """대화창에 첨부된 PDF를 요청한 형식으로 변환합니다."""
    non_pdf_files = [
        file.name
        for file in document_files
        if os.path.splitext(file.name)[1].lower() != ".pdf"
    ]
    if non_pdf_files:
        names = ", ".join(non_pdf_files)
        _append_assistant_message(
            "현재 대화창의 DOCX·TXT 변환은 PDF 파일만 지원합니다. "
            f"다음 파일을 PDF로 저장한 뒤 다시 첨부해 주세요: {names}",
            is_error=True,
        )
        return

    conversions = []
    errors = []
    format_label = "·".join(value.upper() for value in sorted(requested_formats))
    with st.spinner(f"첨부한 PDF를 {format_label} 형식으로 변환하고 있습니다..."):
        for uploaded_pdf in document_files:
            try:
                result = convert_pdf(
                    uploaded_pdf.getvalue(),
                    uploaded_pdf.name,
                )
                conversions.append(
                    {
                        "source_name": result.source_name,
                        "docx_name": result.docx_name,
                        "docx_bytes": result.docx_bytes,
                        "txt_name": result.txt_name,
                        "txt_bytes": result.txt_bytes,
                        "page_count": result.page_count,
                        "text_page_count": result.text_page_count,
                        "requested_formats": sorted(requested_formats),
                    }
                )
            except PdfConversionError as error:
                errors.append(f"{uploaded_pdf.name}: {error}")
            except Exception:
                logger.exception("대화창 PDF 변환 중 예기치 않은 오류")
                errors.append(
                    f"{uploaded_pdf.name}: 변환 중 오류가 발생했습니다."
                )

    if conversions:
        content = (
            f"{len(conversions)}개 PDF를 {format_label} 형식으로 변환했습니다."
        )
        if errors:
            content += "\n\n변환하지 못한 파일:\n- " + "\n- ".join(errors)
        _append_assistant_message(content, conversions=conversions)
    else:
        _append_assistant_message(
            "PDF를 DOCX로 변환하지 못했습니다.\n\n- " + "\n- ".join(errors),
            is_error=True,
        )


def _build_attachment_records(files):
    """채팅 기록에 표시할 안전한 첨부 메타데이터 함수"""
    records = []
    for uploaded_file in files:
        extension = os.path.splitext(uploaded_file.name)[1].lower().lstrip(".")
        record = {"name": uploaded_file.name, "file_type": extension}
        if extension in IMAGE_TYPES:
            record["image_bytes"] = uploaded_file.getvalue()
        records.append(record)
    return records


def _render_current_user_message(content: str, attachments):
    """이번에 제출된 사용자 메시지와 첨부 파일 표시 험수 """
    with st.chat_message("user"):
        if content:
            st.write(content)
        if attachments:
            st.caption("첨부: " + ", ".join(item["name"] for item in attachments))
            image_attachments = [
                item for item in attachments if item.get("image_bytes")
            ]
            if image_attachments:
                columns = st.columns(min(len(image_attachments), 3))
                for index, item in enumerate(image_attachments):
                    with columns[index % len(columns)]:
                        st.image(
                            item["image_bytes"],
                            caption=item["name"],
                            width=512,
                        )


def _get_safe_error_message(error: Exception) -> str:
    """내부 정보를 노출하지 않는 사용자용 오류 문구 함수 """
    error_text = str(error).lower()
    error_name = type(error).__name__.lower()

    if (
        "timeout" in error_name
        or "timed out" in error_text
        or "operation timed out" in error_text
    ):
        return (
            "외부 검색 서버의 응답 시간이 초과되었습니다. "
            "잠시 후 다시 질문해 주세요."
        )

    if (
        "rate limit" in error_text
        or "ratelimit" in error_name
        or "429" in error_text
    ):
        return (
            "현재 요청이 많아 잠시 응답할 수 없습니다. "
            "잠시 후 다시 시도해 주세요."
        )

    if (
        "authentication" in error_text
        or "api key" in error_text
        or "401" in error_text
    ):
        return (
            "AI 서비스 인증 설정에 문제가 있습니다. "
            "관리자에게 문의해 주세요."
        )

    return (
        "답변을 생성하는 중 일시적인 오류가 발생했습니다. "
        "잠시 후 다시 시도해 주세요."
    )


# def _extract_ai_text(message) -> str:
#     """LangChain의 문자열·구조화 응답에서 최종 텍스트를 추출합니다.  gpt-5.6 사용용 함수"""
#     content = getattr(message, "content", "")

#     if isinstance(content, str):
#         return content

#     if isinstance(content, list):
#         final_texts = []
#         other_texts = []

#         for block in content:
#             if not isinstance(block, dict):
#                 continue

#             text = block.get("text")
#             if not isinstance(text, str) or not text.strip():
#                 continue

#             if block.get("phase") == "final_answer":
#                 final_texts.append(text.strip())
#             else:
#                 other_texts.append(text.strip())

#         return "\n\n".join(final_texts or other_texts)

#     return str(content) if content else ""


# def _answer_general_question():
#     """AI 답변을 전달하고 구조화된 응답에서 텍스트를 추출합니다.  gpt-5.6사용용 함수"""
#     response = ai_answer(st.session_state.messages)
#     last_message = response["messages"][-1]
#     ai_response = _extract_ai_text(last_message)

#     if not ai_response:
#         ai_response = "답변 내용이 반환되지 않았습니다."

#     _append_assistant_message(ai_response)


def _answer_general_question():
    """ 질문 전달 함수    gpt-5.5 이하 버전 사용용 """
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

    submission = st.chat_input(
        "질문을 입력하거나 + 버튼으로 문서·이미지를 첨부해 주세요.",
        accept_file="multiple",
        file_type=CHAT_FILE_TYPES,
        key="main_chat_input",
    )
    if not submission:
        return

    prompt = submission.text.strip()
    attached_files = list(submission.files)

    if len(attached_files) > 3:
        _append_assistant_message(
            "대화창에는 한 번에 최대 3개 파일까지 첨부할 수 있습니다.",
            is_error=True,
        )
        return

    document_files = [
        file
        for file in attached_files
        if os.path.splitext(file.name)[1].lower().lstrip(".") in DOCUMENT_TYPES
    ]
    image_files = [
        file
        for file in attached_files
        if os.path.splitext(file.name)[1].lower().lstrip(".") in IMAGE_TYPES
    ]

    if document_files and image_files:
        _append_assistant_message(
            "문서와 이미지는 한 번에 함께 처리할 수 없습니다. "
            "문서 또는 이미지만 선택해 다시 보내 주세요.",
            is_error=True,
        )
        return

    if not prompt and not attached_files:
        return

    effective_prompt = prompt
    if document_files and not effective_prompt:
        effective_prompt = "첨부한 문서의 핵심 내용을 요약해 주세요."
    elif image_files and not effective_prompt:
        effective_prompt = "첨부한 이미지의 내용을 자세히 설명해 주세요."

    attachment_records = _build_attachment_records(attached_files)
    user_message = {
        "role": "user",
        "content": effective_prompt,
        "attachments": attachment_records,
    }
    st.session_state.messages.append(user_message)
    _render_current_user_message(effective_prompt, attachment_records)

    if document_files:
        requested_formats = _get_requested_conversion_formats(effective_prompt)
        if requested_formats:
            _convert_chat_pdfs(document_files, requested_formats)
            return

        with st.spinner("첨부 문서를 학습하고 있습니다..."):
            learned_vectorstore = process1_f(document_files)
        if learned_vectorstore is None:
            _append_assistant_message(
                "첨부 문서를 처리하지 못했습니다. 파일 형식과 내용을 확인해 주세요.",
                is_error=True,
            )
            return

        st.session_state.vectorstore = learned_vectorstore
        with st.spinner("첨부 문서에서 답변을 찾고 있습니다..."):
            answer = answer_question(effective_prompt)
        _append_assistant_message(answer)
        return

    if image_files:
        if is_image_edit_request(effective_prompt):
            with st.spinner("첨부 이미지를 편집하고 있습니다..."):
                try:
                    edit_result = edit_image(
                        uploaded_image=image_files,
                        prompt=effective_prompt,
                        size=select_image_size(effective_prompt),
                        quality="medium",
                    )
                    if st.session_state.get("pending_images"):
                        display_pending_images()
                    else:
                        _append_assistant_message(str(edit_result), is_error=True)
                        display_image_errors()
                except Exception as error:
                    safe_message = _get_safe_error_message(error)
                    _append_assistant_message(safe_message, is_error=True)
            return

        with st.spinner("첨부 이미지를 분석하고 있습니다..."):
            try:
                analysis = analyze_images(image_files, effective_prompt)
                _append_assistant_message(analysis)
            except Exception as error:
                safe_message = _get_safe_error_message(error)
                _append_assistant_message(safe_message, is_error=True)
        return

    if is_image_request(effective_prompt):
        with st.spinner("요청하신 이미지를 생성하고 있습니다..."):
            try:
                tool_result = generate_image.invoke(
                    {
                        "prompt": effective_prompt,
                        "size": select_image_size(effective_prompt),
                        "quality": "medium",
                    }
                )
                if st.session_state.get("pending_images"):
                    display_pending_images()
                else:
                    _append_assistant_message(str(tool_result), is_error=True)
            except Exception as error:
                safe_message = _get_safe_error_message(error)
                _append_assistant_message(safe_message, is_error=True)
        return

    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is not None:
        with st.spinner("학습된 문서에서 관련 내용을 찾고 있습니다..."):
            answer = answer_question(effective_prompt)

        if not answer or "죄송합니다." in answer or len(answer) < 30:
            with st.spinner("답변을 작성하고 있습니다..."):
                try:
                    _answer_general_question()
                except Exception as error:
                    safe_message = _get_safe_error_message(error)
                    _append_assistant_message(safe_message, is_error=True)
        else:
            _append_assistant_message(answer)
    else:
        with st.spinner("답변을 작성하고 있습니다..."):
            try:
                _answer_general_question()
            except Exception as error:
                safe_message = _get_safe_error_message(error)
                _append_assistant_message(safe_message, is_error=True)


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