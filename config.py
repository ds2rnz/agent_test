import os

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


load_dotenv()


def apply_global_styles():
    """앱 전체에 공통으로 사용하는 고성군청 UI 테마를 적용합니다."""
    st.markdown(
        """
        <style>
        :root {
            --gs-navy: #123b5d;
            --gs-blue: #2563eb;
            --gs-sky: #eaf4fb;
            --gs-bg: #f4f7fb;
            --gs-card: #ffffff;
            --gs-text: #172033;
            --gs-muted: #64748b;
            --gs-line: #dfe7f1;
            --gs-success: #0f766e;
        }

        html, body, [class*="css"] {
            font-family: Pretendard, "Noto Sans KR", "Apple SD Gothic Neo",
                         "Malgun Gothic", "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 88% 0%, rgba(37, 99, 235, .06), transparent 28rem),
                var(--gs-bg);
            color: var(--gs-text);
        }

        [data-testid="stHeader"] {
            background: rgba(244, 247, 251, .86);
            backdrop-filter: blur(12px);
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        .block-container {
            max-width: 1120px;
            padding-top: 2.4rem;
            padding-bottom: 5.5rem;
        }

        [data-testid="stSidebar"] {
            background: #f9fbfd;
            border-right: 1px solid var(--gs-line);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.4rem;
        }

        h1, h2, h3 {
            color: var(--gs-text);
            letter-spacing: -0.035em;
        }

        p, label {
            letter-spacing: -0.012em;
        }

        .gs-hero {
            position: relative;
            overflow: hidden;
            padding: 2rem 2.25rem;
            margin-bottom: 1.4rem;
            border: 1px solid rgba(37, 99, 235, .14);
            border-radius: 24px;
            background: linear-gradient(125deg, #ffffff 0%, #f7fbff 68%, #e7f3fb 100%);
            box-shadow: 0 14px 38px rgba(30, 64, 175, .07);
        }

        .gs-hero::after {
            content: "";
            position: absolute;
            width: 210px;
            height: 210px;
            right: -70px;
            top: -95px;
            border-radius: 999px;
            background: rgba(37, 99, 235, .08);
        }

        .gs-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            margin-bottom: .7rem;
            color: var(--gs-blue);
            font-size: .76rem;
            font-weight: 800;
            letter-spacing: .08em;
        }

        .gs-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #16a34a;
            box-shadow: 0 0 0 4px rgba(22, 163, 74, .11);
        }

        .gs-hero h1 {
            margin: 0;
            font-size: clamp(2rem, 4vw, 3rem);
            line-height: 1.18;
            font-weight: 850;
        }

        .gs-hero h1 span {
            color: var(--gs-blue);
        }

        .gs-hero p {
            max-width: 690px;
            margin: .75rem 0 0;
            color: var(--gs-muted);
            font-size: .98rem;
            line-height: 1.75;
        }

        .gs-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: .5rem;
            margin-top: 1.15rem;
        }

        .gs-chip {
            padding: .42rem .72rem;
            color: #355168;
            font-size: .78rem;
            font-weight: 700;
            border: 1px solid #d8e5ef;
            border-radius: 999px;
            background: rgba(255, 255, 255, .78);
        }

        .gs-user-card {
            padding: 1rem;
            margin-bottom: .75rem;
            border: 1px solid #d9e6ef;
            border-radius: 16px;
            background: linear-gradient(145deg, #ffffff, #edf7fc);
        }

        .gs-user-label {
            margin-bottom: .3rem;
            color: var(--gs-muted);
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .06em;
        }

        .gs-user-name {
            color: var(--gs-navy);
            font-size: 1.08rem;
            font-weight: 800;
        }

        .gs-user-id {
            margin-top: .2rem;
            color: var(--gs-muted);
            font-size: .78rem;
        }

        .gs-section-note {
            margin: -.1rem 0 .85rem;
            color: var(--gs-muted);
            font-size: .78rem;
            line-height: 1.55;
        }

        .gs-file-list {
            margin: .55rem 0 .75rem;
            padding: .6rem .7rem;
            border: 1px solid #dcebe6;
            border-radius: 12px;
            background: #f2fbf7;
        }

        .gs-file-count {
            margin-bottom: .35rem;
            color: var(--gs-success);
            font-size: .78rem;
            font-weight: 800;
        }

        .gs-file-item {
            overflow: hidden;
            padding: .16rem 0;
            color: #526376;
            font-size: .76rem;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .gs-footer {
            padding: 1.2rem .2rem .4rem;
            color: #94a3b8;
            font-size: .72rem;
            line-height: 1.7;
            text-align: center;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--gs-line);
            border-radius: 20px;
            background: rgba(255, 255, 255, .92);
            box-shadow: 0 12px 30px rgba(15, 23, 42, .05);
        }

        [data-testid="stChatMessage"] {
            margin-bottom: .7rem;
            padding: 1.05rem 1.15rem;
            border: 1px solid var(--gs-line);
            border-radius: 18px;
            background: rgba(255, 255, 255, .9);
            box-shadow: 0 5px 18px rgba(15, 23, 42, .035);
        }

        [data-testid="stChatInput"] {
            border-color: #cbd8e6;
            border-radius: 18px;
            background: #ffffff;
            box-shadow: 0 10px 28px rgba(15, 23, 42, .09);
        }

        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stFormSubmitButton"] > button {
            min-height: 2.65rem;
            border-radius: 11px;
            font-weight: 750;
            transition: transform .15s ease, box-shadow .15s ease;
        }

        .stButton > button[kind="primary"],
        [data-testid="stFormSubmitButton"] > button {
            border-color: var(--gs-blue);
            background: var(--gs-blue);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        [data-testid="stFormSubmitButton"] > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(37, 99, 235, .14);
        }

        .stTextInput input, .stTextArea textarea {
            border-color: #d7e1ec;
            border-radius: 11px;
            background: #fbfdff;
        }

        [data-testid="stFileUploaderDropzone"] {
            border: 1px dashed #b9ccdc;
            border-radius: 13px;
            background: #f8fbfe;
        }

        [data-testid="stExpander"] {
            overflow: hidden;
            border: 1px solid var(--gs-line);
            border-radius: 14px;
            background: rgba(255, 255, 255, .75);
        }

        [data-testid="stAlert"] {
            border-radius: 12px;
        }

        @media (max-width: 720px) {
            .block-container {
                padding: 1rem .85rem 5rem;
            }
            .gs-hero {
                padding: 1.45rem;
                border-radius: 18px;
            }
            .gs-hero h1 {
                font-size: 2rem;
            }
            .gs-chip-row {
                display: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_embedding():
    """Embedding 모델 반환 (캐시됨)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되지 않았습니다. 프로젝트의 .env 파일을 확인해 주세요."
        )
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=api_key,
    )
