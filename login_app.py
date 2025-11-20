import streamlit as st
import os
import time
from users_db import USERS_DB


def check_login(user_id, name):
    if user_id in USERS_DB:
        user_info = USERS_DB[user_id]
        if user_info.get("name") == name:
            return True, user_info
    return False, None


def show_login_page():
    """로그인 페이지 표시"""

    st.title("🔐 로그인")
    
    # 중앙 정렬을 위한 컬럼
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
  
        with st.form("login_form"):
            user_id = st.text_input("아이디", placeholder="새올아이디를 입력하세요")
            name = st.text_input("사용자이름",  placeholder="사용자 이름을 입력하세요")
            
            submit = st.form_submit_button("로그인", use_container_width=True)
            
            if submit:
                if user_id and name:
                    is_valid, user_info = check_login(user_id, name)
                    
                    if is_valid:
                        with st.spinner("로그인 중..."):
                            time.sleep(1)
                        
                        st.session_state.logged_in = user_id
                        st.session_state.user_info = user_info
                        st.success(f"환영합니다, {user_info['name']}님!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ 로그인 ID 또는 사용자이름을 올바르지 않습니다.")
                else:
                    st.warning("⚠️ 로그인 ID 또는 사용자이름을 모든 입력해주세요.")

        st.markdown('</div>', unsafe_allow_html=True)

        # 하단 정보
        st.markdown("""
            <div style="text-align: center; margin-top: 3rem; color: #64748b;">
                <p>Made by 🔍 총무행정관 정보관리팀</p>
                <p>v1.0.0 | 2025</p>
            </div>
        """, unsafe_allow_html=True)
       
        # 계정 안내
        with st.expander("📝 사용자 계정 입력방법"):
            st.info("""
            **사용자 계정:**
            - 아이디: user12345  / 새올 로그인 ID 입력
            - 사용자이름: 홍길동  /  새올 ID 사용자명
            - 로그인 에러시 정보관리팀 📞680-3463으로 연락주세요
            """)
