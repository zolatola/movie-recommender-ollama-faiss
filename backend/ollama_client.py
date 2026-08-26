"""
Minimal client for a local Ollama server (https://ollama.com).
No third-party ollama package required -- just plain HTTP.
"""

from __future__ import annotations
import requests
from typing import Iterable


class OllamaError(Exception):
    pass


def is_running(host: str, timeout: float = 2.0) -> bool:
    try:
        r = requests.get(f"{host}/api/tags", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def list_models(host: str, timeout: float = 3.0) -> list[str]:
    try:
        r = requests.get(f"{host}/api/tags", timeout=timeout)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except requests.RequestException as e:
        raise OllamaError(f"Could not reach Ollama at {host}: {e}")


def get_embedding(text: str, model: str, host: str, timeout: float = 30.0) -> list[float]:
    """Get a single embedding vector from Ollama's /api/embeddings endpoint."""
    try:
        r = requests.post(
            f"{host}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        vec = data.get("embedding")
        if not vec:
            raise OllamaError(f"Ollama returned no embedding: {data}")
        return vec
    except requests.RequestException as e:
        raise OllamaError(
            f"Embedding request failed (model='{model}' at {host}): {e}"
        )


def chat(
    messages: list[dict],
    model: str,
    host: str,
    timeout: float = 60.0,
    temperature: float = 0.7,
) -> str:
    """Non-streaming chat completion. Returns the assistant's text."""
    try:
        r = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "").strip()
    except requests.RequestException as e:
        raise OllamaError(f"Chat request failed (model='{model}' at {host}): {e}")
