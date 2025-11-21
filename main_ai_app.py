import streamlit as st
import os
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from ai_qna_app import process1_f, ai_answer, answer_question
from config import get_embedding




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
        
        st.markdown("---")
        st.markdown("### 📖 :blue[사용방법]")
        st.markdown("""
            1. PDF 파일을 업로드하세요(최대 3개만)
            2. "학습시작" 버튼을 클릭하세요
            3. 학습한 문서를 바탕으로 사용자 요청에 따라 답변합니다.
            """)
            
        st.markdown("---")

        st.markdown("""
            <div style="text-align: center; padding: 1rem; color: #000000; font-size: 0.9rem;">
                <p style="margin: 0;">Made by 🔍 총무행정관 정보관리팀</p>
                <p style="margin: 0.5rem 0 0 0;">v1.0.0 | 2025</p>
            </div>
        """, unsafe_allow_html=True)

    # 메시지 초기화
    if "messages" not in st.session_state:
        usr_name = f"무엇을 도와 드릴까요? :{str(st.session_state.user_info['name'])}님" 
        st.session_state.messages = [
            {"role": "system", "content": "저는 고성군청 직원을 위해 최선을 다하는 인공지능 도우미입니다."},
            {"role": "assistant", "content": usr_name}
        ]

    # 메시지 출력
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        st.chat_message(role).write(content)

    # vectorstore 로드
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = load_vectorstore(
            embedding=get_embedding(),
            persist_directory="C:/faiss_store"
        )

    # 사용자 입력 처리
    if prompt := st.chat_input(placeholder="무엇이든 물어보세요?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        vectorstore = st.session_state.get("vectorstore")

        if vectorstore is not None:
            with st.spinner("📚 학습된 문서를 검색하는 중..."):
                answer = answer_question(prompt)

            if answer and "죄송합니다. " in answer or len(answer) < 30:
                # st.info("💡 학습된 문서에서 관련 내용을 찾지 못했습니다. 일반 AI 모드로 전환합니다.")
                
                with st.spinner("답변 생성 중..."):
                    try:
                        response = ai_answer(st.session_state.messages)
                        ai_response = response['messages'][-1].content
                        st.toast("일반 AI 모드로 답변합니다....!", icon="🎉")
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        st.chat_message("assistant").write(ai_response)
                    except Exception as e:
                        error_msg = f"오류가 발생했습니다: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        st.chat_message("assistant").write(error_msg)
            else:
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.chat_message("assistant").write(answer)
        else:
            with st.spinner("답변 생성 중..."):
                try:
                    response = ai_answer(st.session_state.messages)
                    ai_response = response['messages'][-1].content
                    st.toast("일반 AI 모드로 답변합니다....!", icon="🎉")
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    st.chat_message("assistant").write(ai_response)
                except Exception as e:
                    error_msg = f"오류가 발생했습니다: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.chat_message("assistant").write(error_msg)

    # 문서 학습 처리
    if process1:
        st.session_state["vectorstore"] = process1_f(uploaded_files1)



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


















