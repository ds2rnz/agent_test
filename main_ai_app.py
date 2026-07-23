import streamlit as st
import os
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from ai_qna_app import process1_f, ai_answer, answer_question, generate_image, edit_image
from config import get_embedding
from image_app import display_image_errors, display_pending_images, select_image_size, is_image_request




def show_main_app():
    """메인 AI 도우미 앱"""
    
    # 페이지 설정
    st.markdown("""
        <style>
            .centered-title {
                text-align: center;
                font-size: 3rem;
                color: #1e293b;
                margin-top: 0px;
                margin-bottom: 3px;
            }
            .ai-text {
                font-size: 3.5rem;
                color: #2563eb;
                margin-left: 10px;
                margin-right: 10px;
            }
        </style> 
        <h1 style="text-align: center; font-size: 3rem; color: #1e293b;">
        💬 고성군청 <span class="ai-text">AI</span> 도우미 </h1>
    """, unsafe_allow_html=True)

    # 사이드바
    with st.sidebar:
        # 사용자 정보 표시
        st.markdown(f"""
            <div style="background: #e0f2fe; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <p style="margin: 0; font-weight: bold; font-size: 1.2rem; color: #0c4a6e;">👤 {st.session_state.user_info['name']}</p>
                <h3 style="margin: 0.5rem 0 0 0; font-size: 1.2rem; color: #075985;">ID: {st.session_state.logged_in}</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # 로그아웃 버튼
        if st.button("🚪 로그아웃", type="secondary", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_info = None
            # 다음 로그인 사용자에게 이전 대화/이미지/PDF 상태가 보이지 않도록 초기화
            for key in (
                "messages",
                "vectorstore",
                "pending_images",
                "pending_image_errors",
            ):
                st.session_state.pop(key, None)
            st.rerun()
        
        st.markdown("---")
        
        # 문서 학습기
        st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
        
        st.markdown("""
            <h2 style="text-align: center; font-size: 1.7rem; color: #000000;">📚 문서 학습기</h2>
            """, unsafe_allow_html=True)

        st.markdown("""
            <p class="upload-label">
                📎 PDF 파일 업로드 
                <span class="badge">최대 3개</span>
            </p>
        """, unsafe_allow_html=True)
        
        uploaded_files1 = st.file_uploader(
            "학습할 PDF 선택",
            type=['pdf'],
            accept_multiple_files=True,
            key="uploader1",
            label_visibility="collapsed"
        )
        
        if uploaded_files1:
            st.markdown("""
                <div style="background: #f0fdf4; padding: 0.5rem; border-radius: 8px; margin-top: 0.5rem;">
                    <p style="margin: 0; font-size: 0.85rem; color: #15803d; font-weight: 500;">
                        ✅ {}개 파일 선택됨
                    </p>
                </div>
            """.format(len(uploaded_files1)), unsafe_allow_html=True)
            
            for i, file in enumerate(uploaded_files1[:3], 1):
                st.markdown(f"""
                    <div style="font-size: 0.8rem; color: #475569; padding: 0.2rem 0.5rem;">
                        {i}. {file.name[:30]}{'...' if len(file.name) > 30 else ''}
                    </div>
                """, unsafe_allow_html=True)
        
        process1 = st.button(
            "🚀 학습 시작",
            key="process1",
            type="primary",
            use_container_width=True
        )
#        st.markdown("---")
#        st.markdown("### 📖 :blue[사용방법]")
#        st.markdown("""
#            1. PDF 파일(최대 3개만) 업로드 가능
#            2. "학습시작" 버튼을 클릭하세요
#            3. 학습한 문서를 바탕으로 사용자 요청에 따라 답변합니다.
#            """)
        
        st.markdown("---")

        # 이미지 수정기
        # st.markdown("### 🎨 이미지 수정기")
        st.markdown("""
            <h2 style="text-align: center; font-size: 1.5rem; color: #000000;">🎨 Image Editer</h2>
            """, unsafe_allow_html=True)
        st.caption("Image를 업로드해 수정합니다.")

        edit_uploaded_image = st.file_uploader(
            "Image(최대3개)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="edit_image_uploader",
            )

        if edit_uploaded_image is not None:
            for index, uploaded_image in enumerate(edit_uploaded_image[:3], 1):
                st.image(uploaded_image, caption="원본", width=256)
        
        edit_prompt = st.text_area(
            "수정 내용",
            placeholder="예: 배경을 바다로 바꾸고 하늘에 뭉게구름을 추가해줘",
            key="edit_image_prompt",
        )
        edit_size = st.selectbox(
            "수정 Image 크기",
            options=["auto", "1024x1024", "1536x1024", "1024x1536"],
            key="edit_image_size",
        )
        edit_button = st.button(
            "✨ Image 수정",
            key="edit_image_button",
            type="primary",
            use_container_width=True,
        )

        st.markdown("---")

        st.markdown("""
            <div style="text-align: center; padding: 1rem; color: #000000; font-size: 0.9rem;">
                <p style="margin: 0;">Made by 🔍 총무행정관 정보관리팀</p>
                <p style="margin: 0.5rem 0 0 0;">v1.0.1 | 2026</p>
            </div>
        """, unsafe_allow_html=True)

    # 메시지 초기화
    if "messages" not in st.session_state:
        usr_name = f"무엇을 도와 드릴까요?  {str(st.session_state.user_info['name'])}님" 
        st.session_state.messages = [
            {"role": "system", "content": "저는 고성군청 직원을 위해 최선을 다하는 인공지능 도우미입니다."},
            {"role": "assistant", "content": usr_name}
        ]

    # 메시지 출력
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg.get("content", "")

        with st.chat_message(role):
            if content:
                st.write(content)

            if msg.get("image_bytes"):
                st.image(
                    msg["image_bytes"],
                    caption=msg.get("image_prompt", "생성된 이미지"),
                    # use_container_width=True,
                    width = 512,
                )

                st.download_button(
                    label="📥 이미지 다운로드",
                    data=msg["image_bytes"],
                    file_name=msg.get(
                        "file_name",
                        "generated_image.png",
                    ),
                    mime="image/png",
                    key=msg.get("download_key"),
                )


    # 세션 상태 초기화
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None

    # PDF 학습 버튼은 채팅 입력 여부와 무관하게 처리
    if process1:
        learned_vectorstore = process1_f(uploaded_files1)
        if learned_vectorstore is not None:
            st.session_state["vectorstore"] = learned_vectorstore

    # 사이드바의 이미지 수정 요청 처리
    if edit_button:
        if edit_uploaded_image is None:
            st.warning("⚠️ 수정할 이미지를 먼저 업로드해 주세요.")
        elif not edit_prompt.strip():
            st.warning("⚠️ 어떻게 수정할지 내용을 입력해 주세요.")
        else:
            with st.spinner("🎨 이미지를 수정하는 중입니다..."):
                edit_result = edit_image(
                    uploaded_image=edit_uploaded_image,
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
        placeholder="무엇이든 물어보세요?"
    )

    # 화면 최초 진입/재실행 시 prompt는 None이므로 메시지를 만들지 않음
    if not prompt:
        return

    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    st.chat_message("user").write(prompt)

    # 이미지 요청
    if is_image_request(prompt):
        with st.spinner("🎨 이미지를 생성하는 중입니다..."):
            try:
                image_size = select_image_size(prompt)

                tool_result = generate_image.invoke({
                    "prompt": prompt,
                    "size": image_size,
                    "quality": "medium",
                })

                if st.session_state.get("pending_images"):
                    # 이미지 메시지 저장과 화면 출력은 이 함수에서 한 번만 처리
                    display_pending_images()
                else:
                    error_message = str(tool_result)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                    })

                    st.chat_message("assistant").error(
                        error_message
                    )

            except Exception as e:
                error_message = (
                    "이미지 생성 중 오류가 발생했습니다: "
                    f"{type(e).__name__}: {e}"
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                })

                st.chat_message("assistant").error(
                    error_message
                )

    # 이미지 요청이 아닌 일반 질문
    else:
        vectorstore = st.session_state.get("vectorstore")

        if vectorstore is not None:
            with st.spinner("📚 학습된 문서를 검색하는 중..."):
                answer = answer_question(prompt)

            if (
                not answer
                or "죄송합니다." in answer
                or len(answer) < 30
            ):
                with st.spinner("답변 생성 중..."):
                    response = ai_answer(
                        st.session_state.messages
                    )
                    ai_response = response["messages"][-1].content

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ai_response,
                    })

                    st.chat_message("assistant").write(
                        ai_response
                    )
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                })

                st.chat_message("assistant").write(answer)

        else:
            # 벡터스토어가 없으면 일반 AI 답변
            with st.spinner("답변 생성 중..."):
                try:
                    response = ai_answer(
                        st.session_state.messages
                    )
                    ai_response = response["messages"][-1].content

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ai_response,
                    })

                    st.chat_message("assistant").write(
                        ai_response
                    )

                except Exception as e:
                    error_message = (
                        "답변 생성 중 오류가 발생했습니다: "
                        f"{type(e).__name__}: {e}"
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                    })

                    st.chat_message("assistant").error(
                        error_message
                    )

def load_vectorstore(embedding, persist_directory="C:/faiss_store"):
    
    # 저장 디렉토리가 존재하는지 확인
    if not os.path.isdir(persist_directory):
        return None
        
    index_file = os.path.join(persist_directory, "index.faiss")
    pkl_file = os.path.join(persist_directory, "index.pkl")
    

    if os.path.exists(index_file) and os.path.exists(pkl_file):
        try:
            st.spinner("📂 기존 학습한 자료를 불러오는 중...")
            vectorstore = FAISS.load_local(
                persist_directory, 
                embedding,
                allow_dangerous_deserialization=True
            )
            st.toast("기존 학습 데이터를 사용합니다!", icon="📚")
            return vectorstore
        except Exception as e:
            st.warning(f"⚠️ 기존 파일 로드 실패: {e}")
            return None
    else:
        return None        









