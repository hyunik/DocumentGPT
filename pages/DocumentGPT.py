# Import required modules
import streamlit as st
import os
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler

# Set page
st.set_page_config(
    page_title="Document GPT",
    page_icon="🐟",
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ]
)

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file_path):
    with open(file_path, "rb") as f:
        file_content = f.read()

    cache_dir = LocalFileStore(os.path.join('/tmp/cache/embeddings', os.path.basename(file_path)))
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator='\n',
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)

    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    if "messages" in st.session_state:
        for message in st.session_state["messages"]:
            send_message(
                message["message"],
                message["role"],
                save=False
            )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Answer the question using ONLY the following context. If you don't know the answer,
    just say you don't know. DO NOT MAKE ANYTHING UP.

    context: {context}
    """),
    ("human", "{question}")
])

# START
st.title("DocumentGPT")

# Uploading file
st.markdown("""
Welcome!!

Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
""")

# 문서 폴더 안의 첫 번째 파일 경로 가져오기
documents_folder = "documents"
always_loaded_file_path = None
if os.path.exists(documents_folder) and os.listdir(documents_folder):
    always_loaded_file_path = os.path.join(documents_folder, os.listdir(documents_folder)[0])

# 문서 임베딩
if always_loaded_file_path:
    retriever = embed_file(always_loaded_file_path)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")

        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(message)
else:
    st.session_state["messages"] = []
