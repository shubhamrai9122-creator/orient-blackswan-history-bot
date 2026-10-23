"""
Web front-end for the History Bot (FastAPI), like the reference repo's Day-4 RAG website.

Run (from the project root, after `python -m rag.ingest`):
    uvicorn server:app --port 8000      then open http://localhost:8000

GET  /          chat page; /?q=<question> opens it with that question already answered
GET  /pdf/History_2024-25.pdf#page=N     the catalogue, opened at page N (reference links)
POST /query     {"name": "...", "message": "..."} -> saved to data/queries.jsonl ("Raise a query" button)
GET  /queries   every query raised so far, newest first
POST /ask       {"question": "..."} -> {"found", "answer", "quote", "references": [{"page", "page_label"}]}
                (a "Not found" reply carries "searched_pages" instead of a quote)
"""

import json
from datetime import datetime
from pathlib import Path
from threading import Lock

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel

from rag import config
from rag.ask import EXAM_QUESTIONS, ask

PAGE = Path(__file__).parent / "static" / "index.html"
QUERIES = Path(__file__).parent / "data" / "queries.jsonl"   # queries raised from the web UI
_queries_lock = Lock()

app = FastAPI(title="Orient BlackSwan History Bot")


class Question(BaseModel):
    question: str


class RaisedQuery(BaseModel):
    name: str = ""
    message: str


@app.post("/query")
def raise_query(body: RaisedQuery) -> dict:
    # Anyone can raise a query (a problem, a wrong answer, a request); it is appended to data/queries.jsonl.
    message = body.message.strip()
    if not message:
        return {"ok": False, "detail": "Please write your query."}
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "name": body.name.strip()[:100] or "Anonymous",
        "message": message[:2000],
    }
    with _queries_lock:
        QUERIES.parent.mkdir(exist_ok=True)
        with QUERIES.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True}


@app.get("/queries")
def list_queries() -> list:
    # Every query raised so far, newest first.
    if not QUERIES.exists():
        return []
    return [json.loads(line) for line in QUERIES.read_text().splitlines() if line.strip()][::-1]


@app.post("/ask")
def ask_endpoint(body: Question) -> dict:
    # ask() never raises: anything it can't answer from the catalogue comes back as "Not found".
    return ask(body.question)


@app.get(f"/pdf/{config.PDF_PATH.name}", response_model=None)
def pdf() -> FileResponse | RedirectResponse:
    # The catalogue itself, so a reference link like /pdf/History_2024-25.pdf#page=91 opens it at that page.
    if config.PDF_PATH.exists():
        return FileResponse(config.PDF_PATH, media_type="application/pdf")
    return RedirectResponse(config.PDF_URL)


@app.get("/", response_class=HTMLResponse)
def home(q: str | None = None) -> str:
    # Answer ?q= on the server so the page is complete on first paint (used for the README screenshots).
    initial = [ask(q)] if q else []
    boot = {"initial": initial, "examples": EXAM_QUESTIONS, "model": config.MODEL_NAME}
    boot = json.dumps(boot).replace("</", "<\\/")
    return PAGE.read_text().replace("/*__BOOT__*/", f"window.BOOT = {boot};")
