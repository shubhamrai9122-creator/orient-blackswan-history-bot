"""
Step 2 - Ask the History Bot

    question -> retriever (top-k chunks) -> grounded prompt -> Gemini -> JSON
             -> guardrail -> answer + page reference, or exactly "Not found"

The bot never shows an error. If the Gemini embeddings quota is used up it falls
back to keyword search over the stored chunks, and anything it cannot answer from
the catalogue (gibberish, off-topic, or a failed model call) is "Not found".

Run (from the project root, after `python -m rag.ingest`):
    python -m rag.ask                  interactive
    python -m rag.ask -q "question"    one question
    python -m rag.ask --all            the 5 exam questions
"""

import argparse
import json
import logging
import math
import re
import threading
from collections import Counter
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from rag import config
from rag.gemini_llm import GeminiInteractionsLLM
from rag.ingest import get_embeddings

log = logging.getLogger("history_bot")

EXAM_QUESTIONS = [
    "On what date were the prices in this catalogue last updated?",
    "In which year was the publisher (Orient BlackSwan) incorporated?",
    "What is the price of a Physics textbook?",
    "Who is the current Minister of Education?",
    'Which is the "best" history book for a beginner?',
]

# The prompt is what makes answers grounded: only the excerpts count, every answer
# must point at the page it came from, and anything else becomes found=false.
PROMPT = ChatPromptTemplate.from_template(
    """You answer questions about the Orient BlackSwan History Catalogue 2024-25.
Use ONLY the catalogue excerpts below. Each excerpt starts with its page number.

Rules:
1. Use only facts written in the excerpts. No outside knowledge, no guessing.
2. If the excerpts do not contain the answer, set "found" to false.
3. The catalogue lists and describes books; it does not rank them. A question asking
   for an opinion (for example which book is "best") is not answered unless an excerpt
   says so explicitly: set "found" to false.
4. Questions about other subjects, people or current affairs that the excerpts do not
   cover: set "found" to false.
5. "answer" is one complete sentence that directly answers the question.
6. "pages" lists the page number(s) of the excerpt(s) holding the answer.
7. "quote" is the full supporting passage, copied word for word from that excerpt: the
   sentence that contains the answer plus the sentences around it that give its context
   (one to three complete sentences, not a fragment).
8. When "found" is false, "reason" explains in one or two sentences what the excerpts do
   or do not contain about the question (for example, that the catalogue lists only
   history titles). Use only the excerpts; never invent facts.

Reply with ONLY this JSON object, no markdown:
{{"found": true or false, "answer": "<one complete sentence>", "pages": [<page numbers>], "quote": "<full supporting passage>", "reason": "<only when found is false>"}}

Catalogue excerpts:
{context}

Question: {question}"""
)

STOPWORDS = set(
    "a an the is are was were be been am of in on at to for and or what which who whom whose when where why how "
    "this that these those it its by with from as do does did i you we they he she me my our your their about any "
    "there can could would should will has have had not no if than then so into also only very most more".split()
)


def words(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def get_store() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(config.DB_DIR),
    )


@lru_cache(maxsize=1)
def keyword_index():
    """Every stored chunk with its word counts, for keyword search."""
    data = get_store().get(include=["documents", "metadatas"])
    chunks = [Document(page_content=t, metadata=m) for t, m in zip(data["documents"], data["metadatas"])]
    counts = [Counter(words(c.page_content)) for c in chunks]
    doc_freq = Counter(w for c in counts for w in c)
    return chunks, counts, doc_freq


def keyword_search(question: str, k: int) -> list:
    """TF-IDF keyword search over the stored chunks; needs no API call."""
    chunks, counts, doc_freq = keyword_index()
    terms = set(words(question)) - STOPWORDS
    scored = []
    for chunk, count in zip(chunks, counts):
        score = sum((1 + math.log(count[t])) * math.log(len(chunks) / doc_freq[t]) for t in terms if count[t])
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in scored[:k]]


def retrieve(question: str) -> dict:
    """Semantic search with Gemini embeddings; keyword search if the embeddings API is unavailable."""
    try:
        retriever = get_store().as_retriever(search_kwargs={"k": config.TOP_K})
        return {"chunks": retriever.invoke(question), "retrieval": "embeddings"}
    except Exception as err:  # e.g. the free-tier embeddings quota is used up
        log.warning("Embedding search failed, using keyword search: %s", str(err)[:150])
        return {"chunks": keyword_search(question, config.TOP_K), "retrieval": "keywords"}


def format_chunks(chunks) -> str:
    # Tag every chunk with its page so the model can cite where the answer is.
    return "\n\n".join(f"[Page {c.metadata['page_number']}]\n{c.page_content}" for c in chunks)


def not_found(question: str, searched=(), reason: str = "") -> dict:
    result = {"question": question, "found": False, "answer": config.NOT_FOUND, "references": []}
    if reason:
        result["reason"] = reason
    if searched:
        result["searched_pages"] = sorted(searched)
    return result


def apply_guardrail(state: dict) -> dict:
    """Return "Not found" unless the model found the answer on a page we actually retrieved."""
    result = state["result"] if isinstance(state["result"], dict) else {}
    retrieved = {c.metadata["page_number"]: c.metadata.get("page_label") for c in state["chunks"]}
    pages = sorted({int(p) for p in result.get("pages", []) if str(p).isdigit() and int(p) in retrieved})
    answer = str(result.get("answer", "")).strip()
    if not result.get("found") or not answer or not pages:
        reason = str(result.get("reason", "")).strip() or "The catalogue excerpts closest to this question do not contain the answer."
        reply = not_found(state["question"], retrieved, reason)
    else:
        reply = {
            "question": state["question"],
            "found": True,
            "answer": answer,
            "quote": str(result.get("quote", "")).strip(),
            "references": [{"page": p, "page_label": retrieved[p]} for p in pages],
        }
    reply["retrieval"] = state["retrieval"]
    return reply


@lru_cache(maxsize=1)
def build_chain():
    llm = GeminiInteractionsLLM(
        model=config.MODEL_NAME, fallback_models=config.FALLBACK_MODELS, api_key=config.GEMINI_API_KEY
    )
    answer_chain = (
        {"context": lambda s: format_chunks(s["chunks"]), "question": lambda s: s["question"]}
        | PROMPT
        | llm
        | JsonOutputParser()
    )

    def answer(state: dict) -> dict:
        # Nothing relevant retrieved (e.g. gibberish): no need to ask the model.
        if not state["chunks"]:
            return {"found": False, "reason": "Nothing in the catalogue matches the words in this question."}
        return answer_chain.invoke(state)

    # Retrieve once, keep the chunks for the guardrail, then answer.
    return (
        RunnableLambda(lambda s: {**s, **retrieve(s["question"])})
        | RunnablePassthrough.assign(result=answer)
        | RunnableLambda(apply_guardrail)
    )


def _key(question: str) -> str:
    return " ".join(question.lower().split())


def _load_answers() -> dict:
    try:
        return json.loads(config.ANSWER_CACHE.read_text())
    except (OSError, ValueError):
        return {}


ANSWERS = _load_answers()
_answers_lock = threading.Lock()


def ask(question: str) -> dict:
    """Answer one question. Never raises: whatever can't be answered from the catalogue is "Not found"."""
    question = question.strip()
    key = _key(question)
    if key in ANSWERS:
        return {**ANSWERS[key], "question": question}
    if not re.search(r"[a-z0-9]", key):
        return not_found(question, reason="The question has no words to search the catalogue for.")
    try:
        result = build_chain().invoke({"question": question})
    except Exception as err:  # quota, network or an unreadable model reply
        log.error("Could not answer %r: %s", question, str(err)[:200])
        return not_found(question, reason="No answer from the catalogue could be confirmed for this question.")
    # Keep grounded answers and semantic-search results. A keyword-search "Not found" is not kept,
    # so it gets redone properly once embeddings work again.
    if result["retrieval"] == "embeddings" or result["found"]:
        with _answers_lock:
            ANSWERS[key] = result
            try:
                config.ANSWER_CACHE.write_text(json.dumps(ANSWERS, indent=2, ensure_ascii=False))
            except OSError as err:
                log.warning("Could not save answer cache: %s", err)
    return result


def format_reply(res: dict) -> str:
    if not res["found"]:
        reply = config.NOT_FOUND
        if res.get("reason"):
            reply += f"\nReason: {res['reason']}"
        searched = ", ".join(str(p) for p in res.get("searched_pages", []))
        if searched:
            reply += f"\nReference: none (closest pages searched: {searched})"
        return reply
    pages = ", ".join(
        f"page {r['page']}"
        + (f" (printed page {r['page_label']})" if r["page_label"] not in (None, str(r["page"])) else "")
        for r in res["references"]
    )
    reply = f"{res['answer']}\nReference: {config.PDF_PATH.name}, {pages}"
    if res.get("quote"):
        reply += f'\nQuote: "{res["quote"]}"'
    reply += f"\nOpen: {config.PDF_URL}#page={res['references'][0]['page']}"
    return reply


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the Orient BlackSwan History Bot")
    parser.add_argument("-q", "--question", help="ask one question and exit")
    parser.add_argument("--all", action="store_true", help="run the 5 exam questions")
    args = parser.parse_args()

    if args.question or args.all:
        questions = [args.question] if args.question else EXAM_QUESTIONS
        for i, q in enumerate(questions, 1):
            print(f"\nQ{i}: {q}\nA: {format_reply(ask(q))}")
        return

    print("Orient BlackSwan History Bot - ask about the History Catalogue 2024-25 ('quit' to stop)")
    while True:
        q = input("\n> ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        if q:
            print(format_reply(ask(q)))


if __name__ == "__main__":
    main()
