"""
Step 1 - Ingest the catalogue into ChromaDB

    PDF -> one Document per page -> overlapping chunks -> Gemini embeddings -> Chroma (./db)

Run (from the project root):
    python -m rag.ingest
"""

import shutil
import sys
import time

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import config

BATCH_SIZE = 40  # chunks per embedding call; small batches stay under the free-tier rate limit


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL, google_api_key=config.GEMINI_API_KEY)


def add_with_retry(store: Chroma, batch: list) -> None:
    for attempt in range(6):
        try:
            store.add_documents(batch)
            return
        except Exception as err:  # the free tier answers 429 / RESOURCE_EXHAUSTED when we go too fast
            if "429" not in str(err) and "RESOURCE_EXHAUSTED" not in str(err):
                raise
            wait = 15 * (attempt + 1)
            print(f"  rate limited, retrying in {wait}s ...")
            time.sleep(wait)
    raise RuntimeError("Embedding kept hitting the rate limit")


def main() -> None:
    if not config.PDF_PATH.exists():
        sys.exit(f"Missing {config.PDF_PATH}. Download it first:\n  curl -L -o {config.PDF_PATH} {config.PDF_URL}")

    # 1. DOCUMENT LOADER - one Document per page; metadata["page"] is the 0-based page index.
    pages = PyPDFLoader(str(config.PDF_PATH)).load()
    pages_with_text = [p for p in pages if p.page_content.strip()]
    print(f"Loaded {len(pages)} pages ({len(pages_with_text)} with text) from {config.PDF_PATH.name}")

    # 2. TEXT SPLITTER - overlapping chunks, small enough to retrieve precisely.
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages_with_text)
    for chunk in chunks:
        # Chroma only stores scalar metadata; page_number (1-based) is the reference we show.
        chunk.metadata = {k: v for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool))}
        chunk.metadata["page_number"] = chunk.metadata["page"] + 1
    print(f"Split into {len(chunks)} chunks")

    # 3. EMBEDDINGS + 4. VECTOR STORE - rebuilt from scratch so re-running never duplicates chunks.
    shutil.rmtree(config.DB_DIR, ignore_errors=True)
    store = Chroma(
        collection_name=config.COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(config.DB_DIR),
    )
    for start in range(0, len(chunks), BATCH_SIZE):
        add_with_retry(store, chunks[start:start + BATCH_SIZE])
        print(f"  embedded {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}")
    print(f"Saved vector store to {config.DB_DIR}")


if __name__ == "__main__":
    main()
