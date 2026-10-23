"""Shared settings for the History Bot. Model names can be overridden in .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.5-flash")
# Tried in order when a model's free-tier quota is used up; each model has its own quota.
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3-flash-preview"]
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

PDF_PATH = ROOT / "data" / "History_2024-25.pdf"
PDF_URL = "https://www.orientblackswan.com/pdfs/downloads/History_2024-25.pdf"
DB_DIR = ROOT / "db"
COLLECTION = "history_catalogue_2024_25"
ANSWER_CACHE = DB_DIR / "answer_cache.json"   # answers already worked out, kept across restarts

CHUNK_SIZE = 1000     # characters per chunk: roughly one or two catalogue entries
CHUNK_OVERLAP = 150   # an entry cut at a chunk boundary still appears whole in one chunk
TOP_K = 6             # chunks retrieved per question

NOT_FOUND = "Not found"
