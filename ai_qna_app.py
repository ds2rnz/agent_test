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
import base64
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from pptx import Presentation
from langchain_core.documents import Document

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


@tool
def generate_image(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "medium"
) -> str:
    """
    사용자의 설명을 바탕으로 새로운 이미지를 생성합니다.

    사용자가 그림, 포스터, 일러스트, 사진, 배너, 캐릭터 등의
    생성을 요청했을 때 사용합니다.

    Args:
        prompt: 생성할 이미지에 대한 구체적인 설명
        size: 이미지 크기. 1024x1024, 1536x1024, 1024x1536 중 하나
        quality: 이미지 품질. low, medium, high 중 하나
    """

    allowed_sizes = {
        "1024x1024",
        "1536x1024",
        "1024x1536",
    }
    allowed_qualities = {"low", "medium", "high"}

    if size not in allowed_sizes:
        size = "1024x1024"

    if quality not in allowed_qualities:
        quality = "medium"

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        result = client.images.generate(
            model="gpt-image-2",
            prompt=prompt,
            size=size,
            quality=quality,
            output_format="png",
            n=1,
        )

        image_base64 = result.data[0].b64_json

        if not image_base64:
            return "이미지 데이터가 반환되지 않았습니다."

        image_bytes = base64.b64decode(image_base64)

        # 도구 실행 결과를 Streamlit 화면 출력용으로 임시 보관
        if "pending_images" not in st.session_state:
            st.session_state.pending_images = []

        st.session_state.pending_images.append({
            "prompt": prompt,
            "image_bytes": image_bytes,
        })

        return (
            "이미지를 성공적으로 생성했습니다. "
            "이미지는 사용자의 채팅 화면에 표시됩니다."
        )

    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"

        if "pending_image_errors" not in st.session_state:
            st.session_state.pending_image_errors = []

        st.session_state.pending_image_errors.append(error_message)

        return f"이미지 생성에 실패했습니다. 오류: {error_message}"


def edit_image(uploaded_image, prompt: str, size: str = "auto", quality: str = "medium") -> str:
    
    """ 업로드한 이미지를 사용자의 지시에 맞게 수정합니다."""
    
    if not uploaded_image:
        return "수정할 이미지를 업로드해 주세요."
    if not isinstance(prompt, str) or not prompt.strip():
        return "이미지 수정 내용을 입력해 주세요."

    if not isinstance(uploaded_image, (list, tuple)):
        uploaded_image = [uploaded_image]
    if len(uploaded_image) > 3:
        return "수정할 이미지는 최대 3개까지 업로드할 수 있습니다."

    if size not in {"auto", "1024x1024", "1536x1024", "1024x1536"}:
        size = "auto"
    if quality not in {"low", "medium", "high", "auto"}:
        quality = "medium"

    temp_paths = []
    image_files = []
    try:
        for index, uploaded_image in enumerate(uploaded_image, 1):
            file_name = getattr(uploaded_image, "name", f"input_{index}.png")
            suffix = Path(file_name).suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
                return "PNG, JPG, JPEG, WEBP 이미지만 수정할 수 있습니다."

            image_data = uploaded_image.getvalue()
            if not image_data:
                return f"{file_name} 이미지가 비어 있습니다."
            if len(image_data) > 50 * 1024 * 1024:
                return f"{file_name}: 각 이미지는 50MB 이하여야 합니다."

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(image_data)
                temp_paths.append(temp_file.name)

        client = OpenAI(api_key=OPENAI_API_KEY)
        image_files = [open(path, "rb") for path in temp_paths]
        image_input = image_files[0] if len(image_files) == 1 else image_files
        result = client.images.edit(
            model="gpt-image-2",
            image=image_input,
            prompt=prompt.strip(),
            size=size,
            quality=quality,
            output_format="png",
            n=1,
        )

        image_base64 = result.data[0].b64_json
        if not image_base64:
            return "수정된 이미지 데이터가 반환되지 않았습니다."

        if "pending_images" not in st.session_state:
            st.session_state.pending_images = []
        st.session_state.pending_images.append({
            "prompt": f"이미지 수정: {prompt.strip()}",
            "image_bytes": base64.b64decode(image_base64),
        })
        return "이미지를 성공적으로 수정했습니다."

    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"
        if "pending_image_errors" not in st.session_state:
            st.session_state.pending_image_errors = []
        st.session_state.pending_image_errors.append(error_message)
        return f"이미지 수정에 실패했습니다. 오류: {error_message}"

    finally:
        for image_file in image_files:
            image_file.close()
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    

def answer_question(query: str):
    st.toast("🚀 질문 처리 시작")
    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is None:
        st.warning("문서 학습이 아직 완료되지 않았습니다.")
        return "먼저 PDF, Excel 또는 PowerPoint 문서를 업로드하고 학습해 주세요."

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


def load_excel_documents(file_path: str, original_name: str) -> list[Document]:
    """Excel/CSV 파일을 시트별 LangChain Document로 변환합니다."""
    extension = Path(original_name).suffix.lower()

    if extension == ".csv":
        try:
            sheets = {"CSV": pd.read_csv(file_path, dtype=str, keep_default_na=False)}
        except UnicodeDecodeError:
            sheets = {
                "CSV": pd.read_csv(
                    file_path,
                    dtype=str,
                    keep_default_na=False,
                    encoding="cp949",
                )
            }
    else:
        sheets = pd.read_excel(
            file_path,
            sheet_name=None,
            dtype=str,
            keep_default_na=False,
        )

    documents = []
    for sheet_name, dataframe in sheets.items():
        dataframe = dataframe.fillna("")
        lines = [
            f"파일명: {original_name}",
            f"시트명: {sheet_name}",
            "열 구성: " + " | ".join(str(column) for column in dataframe.columns),
        ]

        for row_number, (_, row) in enumerate(dataframe.iterrows(), start=2):
            cells = [
                f"{column}: {value}"
                for column, value in row.items()
                if str(value).strip()
            ]
            if cells:
                lines.append(f"{row_number}행 | " + " | ".join(cells))

        if len(lines) > 3:
            documents.append(
                Document(
                    page_content="\n".join(lines),
                    metadata={
                        "source": original_name,
                        "file_type": "excel",
                        "sheet": str(sheet_name),
                    },
                )
            )

    return documents


def load_powerpoint_documents(file_path: str, original_name: str) -> list[Document]:
    """PPTX 파일에서 슬라이드별 텍스트와 표를 추출합니다."""
    presentation = Presentation(file_path)
    documents = []

    for slide_number, slide in enumerate(presentation.slides, start=1):
        parts = [f"파일명: {original_name}", f"슬라이드: {slide_number}"]

        for shape in slide.shapes:
            if getattr(shape, "has_table", False):
                for row_number, row in enumerate(shape.table.rows, start=1):
                    values = [cell.text.strip() for cell in row.cells]
                    if any(values):
                        parts.append(
                            f"표 {row_number}행 | " + " | ".join(values)
                        )
            elif getattr(shape, "has_text_frame", False):
                text = shape.text.strip()
                if text:
                    parts.append(text)

        if len(parts) > 2:
            documents.append(
                Document(
                    page_content="\n".join(parts),
                    metadata={
                        "source": original_name,
                        "file_type": "powerpoint",
                        "slide": slide_number,
                    },
                )
            )

    return documents


def load_uploaded_documents(file_path: str, original_name: str) -> list[Document]:
    """파일 확장자에 맞는 로더로 문서를 추출합니다."""
    extension = Path(original_name).suffix.lower()

    if extension == ".pdf":
        return PyPDFLoader(file_path).load()
    if extension in {".xlsx", ".xls", ".xlsm", ".csv"}:
        return load_excel_documents(file_path, original_name)
    if extension in {".pptx", ".pptm"}:
        return load_powerpoint_documents(file_path, original_name)
    if extension == ".ppt":
        raise ValueError("구형 .ppt 파일은 PowerPoint에서 .pptx로 변환한 후 업로드해 주세요.")

    raise ValueError(f"지원하지 않는 파일 형식입니다: {extension}")



def process1_f(uploaded_files1):
    """PDF, Excel, PowerPoint 파일을 학습하여 벡터스토어를 생성합니다."""
    
    if uploaded_files1 and len(uploaded_files1) > 3:
        st.error("문서는 최대 3개까지 업로드할 수 있습니다.")
        return None
    
    if not uploaded_files1:
        st.warning("⚠️ PDF, Excel 또는 PowerPoint 파일을 업로드해 주세요.")
        return None

    try:
        with st.spinner("📚 문서 임베딩 및 벡터스토어 생성 중... 잠시만 기다려주세요"):
            all_splits = []
            
            for idx, uploaded_file in enumerate(uploaded_files1, 1):
                st.write(f"📄 {idx}/{len(uploaded_files1)} 파일 처리 중: {uploaded_file.name}")
                
                extension = Path(uploaded_file.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    data = load_uploaded_documents(tmp_path, uploaded_file.name)
                    
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=600, 
                        chunk_overlap=100
                    )
                    splits = splitter.split_documents(data)
                    all_splits.extend(splits)
                    
                    st.success(f"✅ {uploaded_file.name}: {len(splits)}개 학습 조각으로 분할")

                except Exception as file_error:
                    st.error(f"❌ {uploaded_file.name} 처리 실패: {file_error}")
                    
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            if not all_splits:
                st.error("❌ 학습할 텍스트를 추출하지 못했습니다.")
                return None


            st.info(f"📊 총 문서 분할 수: {len(all_splits)}")

            embedding = OpenAIEmbeddings(
                model="text-embedding-3-large", 
                api_key=OPENAI_API_KEY,
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





load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

config = {"configurable": {"thread_id": "1"}}

system_prompt_text = """
당신은 고성군청 직원을 위한 친절한 고성군청 AI 도우미입니다.

1. 직원들이 질문하면 구체적이고 자세하게 설명해주세요 .
2. 모르는 내용이면 도구를 이용하여 인터넷 검색을 꼭해서 답변해주세요.
3. 인터넷 검색에 대하여 링크를 표시해 주세요.
4. 이 지역은 강원특별자치도 고성군입니다.
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
    model = "openai:gpt-5.5",
    # temperature=0.9, 
    # max_tokens=2000, 
    # timeout=100,
    # max_retries=2, 
    )


embedding = OpenAIEmbeddings(
    model="text-embedding-3-large", 
    api_key = os.getenv("OPENAI_API_KEY"))


agent = create_agent(
    model=llm,
    tools=[get_current_time, get_web_search, generate_image],
    middleware=[],
    system_prompt=system_prompt_text, 
    )
