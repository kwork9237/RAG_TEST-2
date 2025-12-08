import streamlit as st
import tiktoken
from loguru import logger

from langchain_core.messages import ChatMessage
# from langchain_ollama import ChatOllama

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langserve import RemoteRunnable

# TEST LINK
# link = "http://localhost:8000/llm"
link = "https://phalangeal-kenley-nonethnologically.ngrok-free.dev/llm"

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)

    return len(tokens)

def get_text(docs):
    doc_list = []

    for doc in docs:
        fname = doc.name

        with open(fname, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Uploaded {fname}")

        if ".pdf" in doc.name:
            loader = PyPDFLoader(fname)
            documents = loader.load_and_split()

        elif '.docx' in doc.name:
            loader = Docx2txtLoader(fname)
            documents = loader.load_and_split()

        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(fname)
            documents = loader.load_and_split()

        doc_list.extend(documents)

    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 900,
        chunk_overlap = 100,
        length_function = tiktoken_len
    )

    chunks = text_splitter.split_documents(text)

    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name = "jhgan/ko-sroberta-multitask",
        model_kwargs = { "device" : "cpu" },
        encode_kwargs = { "normalize_embeddings" : True }
    )
    
    vectordb = FAISS.from_documents(text_chunks, embeddings)

    return vectordb


def main():
    global retriever

    st.set_page_config(
        page_title="Streamlit_Remote_RAG",
        page_icon=":books:"
    )

    st.title("_RAG_TEST4 :red[Q/A Chat]_ :books:")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "store" not in st.session_state:
        st.session_state["store"] = dict()

    def print_history():
        for msg in st.session_state.messages:
            st.chat_message(msg.role).write(msg.content)

    def add_history(role, content):
        st.session_state.messages.append(ChatMessage(role = role, content = content))

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    with st.sidebar:
        upload_files = st.file_uploader("Uploat your file", type=["pdf", "docs"], accept_multiple_files=True)
        process = st.button("Process")

    if process:
        files_text = get_text(upload_files)
        text_chunks = get_text_chunks(files_text)
        vector_store = get_vectorstore(text_chunks)
        retriever = vector_store.as_retriever(search_type="mmr", vervose=True)
        # retriever = vector_store.as_retriever(search_type="similarity", k=2)
        
        st.session_state["retriever"] = retriever
        st.session_state.processComplete = True

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role" : "assistant",
                "content" : "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요."
            }
        ]

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    print_history()

    if user_input := st.chat_input("질문을 입력해주세요."):
        add_history("user", user_input)
        st.chat_message("user").write(f"{user_input}")

        with st.chat_message("assistant"):
            llm = RemoteRunnable(link)
            chat_container = st.empty()

            if st.session_state.processComplete == True:
                retriever = st.session_state["retriever"]

                ## 아마 버전이슈였던 것 같음. GPT code를 이용하여 단일 text로 보내서 해결

                # 1) 업로드한 문서에서 관련 문맥 추출
                docs = retriever.invoke(user_input)
                context = format_docs(docs)

                # 2) RAG용 최종 프롬프트 문자열 (ChatML 형식)
                rag_input = f"""
                    당신은 동서울대학교 컴퓨터소프트웨어학과 안내 AI입니다.
                    검색된 문맥을 사용하여 질문에 맞는 답변을 하세요.
                    문맥을 찾을 수 없으면 모르겠다고 답변하시오.

                    검색된 문맥: {context}

                    질문: {user_input}
                """

                # 3) LangServe / Ollama 서버 호출
                answer = llm.stream(rag_input)

                # 4) 스트리밍 출력 + 히스토리 저장
                chunks = []
                for chunk in answer:
                    # chunk는 AIMessageChunk일 수도 있고 문자열일 수도 있으므로 안전하게 처리
                    text = getattr(chunk, "content", str(chunk))
                    chunks.append(text)
                    chat_container.markdown("".join(chunks))

                add_history("ai", "".join(chunks))

            else:
                prompt2 = ChatPromptTemplate.from_template(
                    "다음의 질문에 간결하게 답변해 주세요:\n {input}"
                )

                chain = prompt2 | llm | StrOutputParser()

                answer = chain.stream(user_input)
                chunks = []

                for chunk in answer:
                    chunks.append(chunk)
                    chat_container.markdown("".join(chunks))

                add_history("ai", "".join(chunks))

if __name__ == '__main__':
    main()
