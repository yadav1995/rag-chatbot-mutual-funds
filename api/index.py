from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path

# Add project root so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_pipeline import RAGPipeline

app = FastAPI(title="Mutual Fund FAQ API", version="1.0.0")

# CORS for local dev connecting to Vercel/NextJS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global pipeline (loads models once on startup)
pipeline = None

@app.on_event("startup")
def startup_event():
    global pipeline
    # Load pipeline without reranker by default to save memory in serverless environments
    pipeline = RAGPipeline(use_reranker=False)

class ChatMessage(BaseModel):
    role: str
    content: str
    citations: Optional[List[str]] = []

class ChatRequest(BaseModel):
    query: str
    thread_id: Optional[str] = "default-thread"
    history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    answer: str
    citations: List[str]
    intent: str
    guardrail_passed: bool
    guardrail_violations: List[str]
    scrape_date: str

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse)
def compute_chat(request: ChatRequest):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    # Format history for RAG pipeline
    formatted_history = []
    for msg in dict(request).get("history", []):
        if hasattr(msg, "role"):
            formatted_history.append({"role": msg.role, "content": msg.content})
        else:
            formatted_history.append(msg)
            
    response = pipeline.answer(
        query=request.query,
        thread_id=request.thread_id,
        conversation_history=formatted_history
    )
    
    return ChatResponse(
        answer=response.answer,
        citations=response.citations,
        intent=response.intent,
        guardrail_passed=response.guardrail_passed,
        guardrail_violations=response.guardrail_violations,
        scrape_date=response.scrape_date
    )
