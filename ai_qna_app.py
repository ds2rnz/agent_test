import streamlit as st
import os
from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import RetrievalQA
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from pathlib import Path
import tempfile
import traceback
import time
import pytz
from datetime import datetime
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

@tool
def get_current_time(timezone: str, location: str) -> str:
    '''  해당 지역 현재시각을 구하는 함수 '''
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        result = f'{timezone} ({location}) 현재시각 {now}'
        return result
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"  

@tool
def get_web_search(query: str) -> str:
    """
    웹 검색을 수행하는 함수.

    Args:
        query (str): 검색어
    Returns:
        str: 검색 결과
    """
    custom_wrapper = DuckDuckGoSearchAPIWrapper(region="kr-kr", time="y", max_results=10)
    search = DuckDuckGoSearchResults(
        api_wrapper=custom_wrapper,
        source="news, image, text",
        results_separator=';\n')
    
    results = search.run(query)

    st.toast("웹 검색을 통아여 알아보고 있습니다.", icon="🎉")
    return results



config = {"configurable": {"thread_id": "1"}}

system_prompt_text = """
당신은 고성군청 직원을 위한 친절한 고성군청 AI 도우미입니다.

1. 직원들이 질문하면 구체적이고 자세하게 설명해주세요 .
2. 모르는 내용이면 도구를 이용하여 인터넷 검색을 꼭해서 답변해주세요.
3. 인터넷 검색에 대하여 링크를 표시해 주세요.
4. 이 지역은 강원도 고성군입니다.
   - 고성군청 주소는 강원특별자치도 고성군 간성읍 고성중앙길9입니다.
5. 강원도 고성군 관련 관광지 질문이 들어오면 아래 홈페이지를 참고하여 답해주세요.
   - 고성군 관광포털 사이트 : https://gwgs.go.kr/tour/index.do
6. 강원도 고성군 고성군청에 관하여 질문이 들어오면 아래 홈페이지를 참고하여 답해주세요
   - 고성군청 홈페이지 : https://gwgs.go.kr
7. 고성군수는 함명준입니다.
   - 고성군수는 고성군 발전을 위하여 노력하시는분입니다.
8. 고성군청 ai 도우미는 고성군청 총무행정관 정보관리팀에서 agent를 제작하였습니다.
   - langchain을 기반으로 제작하였으며, RAG기술과 학습기능을 탐재하였으며, 지속적으로 기능추가 예정입니다.
9. 한글로 답해주세요
"""

llm = init_chat_model(
    model = "openai:gpt-4o",
    temperature=0.9, 
    max_tokens=2000, 
    timeout=100,
    max_retries=2, 
    )


embedding = OpenAIEmbeddings(
    model="text-embedding-3-large", 
    api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    )


agent = create_agent(
    model=llm,
    tools=[get_current_time, get_web_search],
    middleware=[],
    system_prompt=system_prompt_text, 
    )



def answer_question(query: str):
    st.toast("🚀 질문 처리 시작")
    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is None:
        st.warning("⚠️ PDF 학습이 아직 완료되지 않았습니다.")
        return "먼저 PDF 문서를 업로드하고 학습시켜 주세요."

    st.toast("✅ vectorstore 확인 완료")
    try:
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            st.toast(f"  문서 {i} 유사도: {score:.4f}", icon="🎉")

        SIMILARITY_THRESHOLD = 1.1
        relevant_docs = [doc for doc, score in docs_with_scores if score < SIMILARITY_THRESHOLD]
        if not relevant_docs:
            return "죄송합니다. 관련된 정보를 찾지 못했습니다."
        
        template = """당신은 친절한 AI 도우미입니다. 주어진 문서 내용을 바탕으로 질문에 답변해주세요.
    
                    문서 내용:
                    {context}

                    질문: {question}

                    답변 시 다음을 지켜주세요:
                    1. 문서 내용에 기반하여 정확하게 답변해주세요.
                    2. 가능한 한 구체적이고 자세하게 설명해주세요.
                    3. 한국어로 답변해주세요.

                    답변:"""

        prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
                )
        retriever = vectorstore.as_retriever(search_kwargs={"k":3})
        qa_chain = RetrievalQA.from_chain_type(
               llm=llm,
               chain_type="stuff",
               retriever=retriever,
               chain_type_kwargs={"prompt": prompt},
               return_source_documents=False
                )
        result = qa_chain.invoke({"query": query})
        if isinstance(result, dict):
            return result.get("result", "답변을 생성할 수 없습니다.")
        else:
            return str(result)
    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")
        st.code(traceback.format_exc(), language="python")
        return f"오류가 발생했습니다: {e}"
                

def ai_answer(messages):
    response = agent.invoke(
    {"messages": messages},
        config=config,
        tool_choice='any'
        )
    return response


def process1_f(uploaded_files1):
    """PDF 파일을 학습하여 벡터스토어 생성"""
    
    if uploaded_files1 and len(uploaded_files1) > 3:
        st.error("❌ PDF는 최대 3개까지 업로드 가능합니다!")
        st.warning("⚠️ PDF파일을 3개만 선택하여 주세요!")
        return None
    
    if not uploaded_files1:
        st.warning("⚠️ PDF 파일을 업로드해주세요.")
        return None

    try:
        with st.spinner("📚 PDF 임베딩 및 벡터스토어 생성 중... 잠시만 기다려주세요"):
            all_splits = []
            
            for idx, uploaded_file in enumerate(uploaded_files1, 1):
                st.write(f"📄 {idx}/{len(uploaded_files1)} 파일 처리 중: {uploaded_file.name}")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                try:
                    loader = PyPDFLoader(tmp_path)
                    data = loader.load()
                    
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=600, 
                        chunk_overlap=100
                    )
                    splits = splitter.split_documents(data)
                    all_splits.extend(splits)
                    
                    st.success(f"✅ {uploaded_file.name}: {len(splits)}개 문서로 분할")
                    
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            st.info(f"📊 총 문서 분할 수: {len(all_splits)}")

            embedding = OpenAIEmbeddings(
                model="text-embedding-3-large", 
                api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
            )
            
            persist_directory = "C:/faiss_store"
            try:
                os.makedirs(persist_directory, exist_ok=True)
            except Exception as e:
                st.error(f"❌ 디렉토리 생성 실패: {e}")
                return None

            batch_size = 20
            vectorstore = None
            total_batches = (len(all_splits) + batch_size - 1) // batch_size
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(0, len(all_splits), batch_size):
                batch = all_splits[i:i+batch_size]
                batch_num = i//batch_size + 1
                
                status_text.text(f"🔄 배치 {batch_num}/{total_batches} 학습자료 저장 중...")
                progress_bar.progress(batch_num / total_batches)
                
                try:
                    if vectorstore is None:
                        vectorstore = FAISS.from_documents(batch, embedding)
                    else:
                        vectorstore.add_documents(batch)
                    
                    vectorstore.save_local(persist_directory)
                    time.sleep(1.5)
                    
                except Exception as e:
                    st.error(f"❌ 배치 {batch_num} 학습자료 저장 실패: {e}")
                    continue
            
            progress_bar.progress(1.0)
            status_text.text("✅ 학습자료 저장 완료!")
            st.success("🎉 학습이 완료되었습니다!")
            st.toast("학습한 문서를 바탕으로 질문해 보세요!", icon="🎉")
            
            return vectorstore
    except Exception as e:
        st.error(f"❌ 학습 중 오류 발생: {e}")
        st.code(traceback.format_exc(), language="python")
        return None























































