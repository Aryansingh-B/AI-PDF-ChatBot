from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil, os

from config import UPLOAD_DIR
from rag_pipeline import (
    load_and_split,
    build_vector_store,
    load_vector_store,
    get_qa_chain,
    ask_question
)

app = FastAPI(title="AI PDF Chatbot")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global chain object (lives in memory per session)
qa_chain = None

# ── Route 1: Upload PDF ───────────────────────────────────────────────────────
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global qa_chain

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save uploaded file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # RAG pipeline: load → chunk → embed → store
    chunks = load_and_split(file_path)
    vector_store = build_vector_store(chunks)
    qa_chain = get_qa_chain(vector_store)

    return {"message": f"'{file.filename}' uploaded and indexed successfully. You can now ask questions!"}

# ── Route 2: Ask a Question ───────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(request: QuestionRequest):
    global qa_chain

    if qa_chain is None:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet. Please upload a PDF first.")

    result = ask_question(qa_chain, request.question)
    return result

# ── Route 3: Health check ─────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "AI PDF Chatbot is running!"}