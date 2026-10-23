"""
Step 2 - Ask the History Bot

    question -> retriever (top-k chunks) -> grounded prompt -> Gemini -> JSON
             -> guardrail -> answer + page reference, or exactly "Not found"

Run (from the project root, after `python -m rag.ingest`):
    python -m rag.ask                  interactive
    python -m rag.ask -q "question"    one question
    python -m rag.ask --all            the 5 exam questions
"""

import argparse
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from rag import config
from rag.gemini_llm import GeminiInteractionsLLM
from rag.ingest import get_embeddings

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
5. "pages" lists the page number(s) of the excerpt(s) holding the answer, and "quote"
   is copied word for word from that excerpt.

Reply with ONLY this JSON object, no markdown:
{{"found": true or false, "answer": "<short answer>", "pages": [<page numbers>], "quote": "<exact supporting text>"}}

Catalogue excerpts:
{context}

Question: {question}"""
)


def format_chunks(chunks) -> str:
    # Tag every chunk with its page so the model can cite where the answer is.
    return "\n\n".join(f"[Page {c.metadata['page_number']}]\n{c.page_content}" for c in chunks)


def apply_guardrail(state: dict) -> dict:
    """Return "Not found" unless the model found the answer on a page we actually retrieved."""
    result = state["result"] if isinstance(state["result"], dict) else {}
    retrieved = {c.metadata["page_number"]: c.metadata.get("page_label") for c in state["chunks"]}
    pages = sorted({int(p) for p in result.get("pages", []) if str(p).isdigit() and int(p) in retrieved})
    answer = str(result.get("answer", "")).strip()
    if not result.get("found") or not answer or not pages:
        return {
            "question": state["question"],
            "found": False,
            "answer": config.NOT_FOUND,
            "references": [],
            "searched_pages": sorted(retrieved),
        }
    return {
        "question": state["question"],
        "found": True,
        "answer": answer,
        "quote": str(result.get("quote", "")).strip(),
        "references": [{"page": p, "page_label": retrieved[p]} for p in pages],
    }


@lru_cache(maxsize=1)
def build_chain():
    store = Chroma(
        collection_name=config.COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(config.DB_DIR),
    )
    retriever = store.as_retriever(search_kwargs={"k": config.TOP_K})
    llm = GeminiInteractionsLLM(model=config.MODEL_NAME, api_key=config.GEMINI_API_KEY)

    answer_chain = (
        {"context": lambda s: format_chunks(s["chunks"]), "question": lambda s: s["question"]}
        | PROMPT
        | llm
        | JsonOutputParser()
    )
    # Retrieve once, keep the chunks for the guardrail, then answer.
    return (
        RunnablePassthrough.assign(chunks=lambda s: retriever.invoke(s["question"]))
        | RunnablePassthrough.assign(result=answer_chain)
        | RunnableLambda(apply_guardrail)
    )


@lru_cache(maxsize=128)
def ask(question: str) -> dict:
    return build_chain().invoke({"question": question.strip()})


def format_reply(res: dict) -> str:
    if not res["found"]:
        searched = ", ".join(str(p) for p in res.get("searched_pages", []))
        return f"{config.NOT_FOUND}\nReference: none (closest pages searched: {searched})"
    pages = ", ".join(
        f"page {r['page']}"
        + (f" (printed page {r['page_label']})" if r["page_label"] not in (None, str(r["page"])) else "")
        for r in res["references"]
    )
    reply = f"{res['answer']}\nReference: {config.PDF_PATH.name}, {pages}"
    if res.get("quote"):
        reply += f'\nQuote: "{res["quote"]}"'
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
