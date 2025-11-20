import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings

@st.cache_resource
def get_embedding():
    """Embedding 모델 반환 (캐시됨)"""
    return OpenAIEmbeddings(
        model="text-embedding-3-large", 
        api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    )


