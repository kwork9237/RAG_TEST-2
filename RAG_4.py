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
link = "https://phalangeal-kenley-nonethnologically.ngrok-free.dev"

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

    RAG_PROMPT_TEMPLATE = """
        당신은 동서울대학교 컴퓨터소프트웨어학과 안내 AI입니다.
        검색된 문맥을 사용하여 질문에 맞는 답변을 30단어 이내로 하세요.
        답을 모른다면 모른다고 답변하세요.
        Question: {question}
        Context: {context}
        Answer:
    """

    print_history()

    if user_input := st.chat_input("질문을 입력해주세요."):
        add_history("user", user_input)
        st.chat_message("user").write(f"{user_input}")

        with st.chat_message("assistant"):
            llm = RemoteRunnable(link)
            chat_container = st.empty()

            if st.session_state.processComplete == True:
                prompt1 = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
                retriever = st.session_state["retriever"]

                rag_chain = (
                    {
                        "context" : retriever | format_docs,
                        "question" : RunnablePassthrough(),
                    }
                    | prompt1
                    | llm
                    | StrOutputParser()
                )

                answer = rag_chain.stream(user_input)
                chunks = []

                for chunk in answer:
                    chunks.append(chunk)
                    chat_container.markdown("".join(chunks))

                add_history("ai", "".join(chunks))

            else:
                prompt2 = ChatPromptTemplate.from_template(
                    "다음의 질문에 간결하게 답변해 주세요:\n{input}"
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