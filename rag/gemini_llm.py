"""
A LangChain LLM that calls Gemini through Google's Interactions API.

Why not ChatGoogleGenerativeAI (used in the reference repo)? On newly created
API keys, gemini-2.5-* returns "no longer available to new users", and the
newer models (gemini-3.5-flash, ...) answer only through the Interactions API,
not generateContent. Wrapping that endpoint as a LangChain LLM keeps the RAG
chain plain LCEL:  prompt | llm | parser
"""

import time
from typing import Any, List, Optional

import requests
from langchain_core.language_models.llms import LLM

INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
RETRY_STATUS = {500, 502, 503, 504}


class GeminiInteractionsLLM(LLM):
    model: str = "gemini-3.5-flash"
    fallback_models: List[str] = []  # tried in order when a model's quota is used up (HTTP 429)
    api_key: str
    timeout: int = 120
    max_retries: int = 3

    @property
    def _llm_type(self) -> str:
        return "gemini-interactions"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs: Any) -> str:
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        for model in dict.fromkeys([self.model, *self.fallback_models]):
            for attempt in range(self.max_retries):
                try:
                    resp = requests.post(
                        INTERACTIONS_URL, json={"model": model, "input": prompt}, headers=headers, timeout=self.timeout
                    )
                except requests.RequestException:
                    resp = None
                if resp is not None and resp.ok:
                    return _extract_text(resp.json())
                if resp is not None and resp.status_code == 429:
                    break  # this model's quota is used up: try the next model
                if resp is not None and resp.status_code not in RETRY_STATUS:
                    resp.raise_for_status()
                if attempt < self.max_retries - 1:
                    time.sleep(3 * 2**attempt)  # 3s, 6s: back off on overload and timeouts
        raise RuntimeError("Every Gemini model failed or is out of quota")


def _extract_text(data: dict) -> str:
    # Response shape: steps[] -> {"type": "model_output", "content": [{"type": "text", "text": ...}]}
    # (older responses used outputs[] -> {"type": "text", "text": ...})
    parts = [
        part.get("text", "")
        for step in data.get("steps", [])
        if step.get("type") == "model_output"
        for part in step.get("content", [])
        if part.get("type") == "text"
    ]
    if not parts:
        parts = [out.get("text", "") for out in data.get("outputs", []) if out.get("type") == "text"]
    return "".join(parts).strip()
