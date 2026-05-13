from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from config import GEMINI_API_KEY, FAISS_INDEX_DIR
import os

# ── Step 1: Load & chunk the PDF ──────────────────────────────────────────────
def load_and_split(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()                        # list of LangChain Document objects

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # each chunk = ~1000 characters
        chunk_overlap=200       # 200-char overlap keeps context between chunks
    )
    chunks = splitter.split_documents(documents)
    return chunks

# ── Step 2: Embed chunks → store in FAISS ────────────────────────────────────
def build_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_DIR)         # saves to disk
    return vector_store

# ── Step 3: Load existing FAISS index ────────────────────────────────────────
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    vector_store = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True         # required by LangChain
    )
    return vector_store

# ── Step 4: Build RAG chain with memory ──────────────────────────────────────
def get_qa_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3                              # lower = more factual answers
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),  # top 4 chunks
        memory=memory,
        return_source_documents=True
    )
    return chain

# ── Step 5: Ask a question ────────────────────────────────────────────────────
def ask_question(chain, question: str):
    result = chain.invoke({"question": question})
    return {
        "answer": result["answer"],
        "sources": [doc.page_content[:200] for doc in result["source_documents"]]
    }