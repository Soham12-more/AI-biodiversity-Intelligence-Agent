from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from . import store
from .dialogue import handle
from .knowledge import get_retriever, load_kb
from .models import ChatRequest, ChatResponse

app = FastAPI(title="Darukaa Biodiversity Intelligence Agent", version="1.0.0")
STATIC = Path(__file__).parent / "static"


@app.get("/health")
def health():
    r = get_retriever()
    return {"status": "ok", "retrieval_mode": r.mode, "chunks": len(r.kb.chunks)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message and not req.site:
        raise HTTPException(400, "Send 'message' (text) and/or 'site' (structured JSON).")
    return handle(req)


@app.get("/kb/search")
def kb_search(q: str, k: int = 5):
    return {"query": q, "mode": get_retriever().mode, "hits": get_retriever().search(q, k=k)}


@app.get("/kb/evidence")
def kb_evidence():
    return list(load_kb().evidence.values())


@app.get("/sessions/{sid}")
def session(sid: str):
    s = store.get_or_create(sid)
    return {**s, "history": store.history(sid)}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
