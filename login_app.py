import streamlit as st

from users_db import USERS_DB


def check_login(user_id, name):
    """사용자 ID와 이름이 등록 정보와 일치하는지 확인합니다."""
    normalized_id = user_id.strip()
    normalized_name = name.strip()
    user_info = USERS_DB.get(normalized_id)

    if user_info and user_info.get("name") == normalized_name:
        return True, user_info
    return False, None


def show_login_page():
    """고성군청 AI 도우미 로그인 화면을 표시합니다."""
    st.markdown(
        """
        <div class="gs-hero">
            <div class="gs-eyebrow">
                <span class="gs-dot"></span>
                GOSEONG COUNTY · DIGITAL WORKSPACE
            </div>
            <h1>고성군청 <span>AI 도우미</span></h1>
            <p>
                생성형 AI 대화부터 이미지 생성·편집까지, 직원의 일상 업무를
                더 빠르고 편리하게 지원합니다.
            </p>
            <div class="gs-chip-row">
                <span class="gs-chip">문서 기반 답변</span>
                <span class="gs-chip">실시간 정보 검색</span>
                <span class="gs-chip">이미지 생성·편집</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([0.5, 1, 0.5])

    with center:
        with st.container(border=True):
            st.markdown("### 사용자 로그인")
            st.caption("새올행정시스템에서 사용하는 정보를 입력해 주세요.")

            with st.form("login_form"):
                user_id = st.text_input(
                    "새올 아이디",
                    placeholder="아이디를 입력하세요",
                    autocomplete="username",
                )
                name = st.text_input(
                    "사용자 이름",
                    placeholder="이름을 입력하세요",
                )
                submit = st.form_submit_button(
                    "로그인",
                    type="primary",
                    use_container_width=True,
                )

                if submit:
                    if not user_id.strip() or not name.strip():
                        st.warning("아이디와 사용자 이름을 모두 입력해 주세요.")
                    else:
                        is_valid, user_info = check_login(user_id, name)
                        if is_valid:
                            st.session_state.logged_in = user_id.strip()
                            st.session_state.user_info = user_info
                            st.rerun()
                        else:
                            st.error("아이디 또는 사용자 이름을 다시 확인해 주세요.")

            with st.expander("계정 입력 안내"):
                st.markdown(
                    """
                    - **아이디**: 새올 로그인 ID
                    - **사용자 이름**: 새올 ID에 등록된 이름
                    - 로그인 문의: 정보관리팀 **680-3463**
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
