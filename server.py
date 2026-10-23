"""
Web front-end for the History Bot (FastAPI), like the reference repo's Day-4 RAG website.

Run (from the project root, after `python -m rag.ingest`):
    uvicorn server:app --port 8000      then open http://localhost:8000

GET  /          chat page; /?q=<question> opens it with that question already answered
POST /ask       {"question": "..."} -> {"found", "answer", "quote", "references": [{"page", "page_label"}]}
                (a "Not found" reply carries "searched_pages" instead of a quote)
"""

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag import config
from rag.ask import EXAM_QUESTIONS, ask

PAGE = Path(__file__).parent / "static" / "index.html"

app = FastAPI(title="Orient BlackSwan History Bot")


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask_endpoint(body: Question) -> dict:
    return ask(body.question)


@app.get("/", response_class=HTMLResponse)
def home(q: str | None = None) -> str:
    # Answer ?q= on the server so the page is complete on first paint (used for the README screenshots).
    initial = [ask(q)] if q else []
    boot = {"initial": initial, "examples": EXAM_QUESTIONS, "model": config.MODEL_NAME}
    boot = json.dumps(boot).replace("</", "<\\/")
    return PAGE.read_text().replace("/*__BOOT__*/", f"window.BOOT = {boot};")
