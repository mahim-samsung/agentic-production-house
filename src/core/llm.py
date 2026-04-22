"""
Unified LLM client supporting OpenAI and Ollama backends.

Both are accessed through the OpenAI-compatible chat completions API,
so switching providers only requires changing the base_url and model.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from src.utils.config import get as cfg
from src.utils.logger import get_logger

log = get_logger("llm")

T = TypeVar("T", bound=BaseModel)


def _ollama_openai_base_url(host: str) -> str:
    """Ollama OpenAI-compatible calls use base …/v1 + POST /chat/completions."""
    h = host.strip().rstrip("/")
    if h.lower().endswith("/v1"):
        h = h[:-3].rstrip("/")
    return f"{h}/v1"


class LLMClient:
    """Thin wrapper that talks to any OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.provider = provider or cfg("llm.provider", "ollama")

        if self.provider == "openai":
            self.model = model or cfg("llm.openai.model", "gpt-4o")
            self.base_url = base_url or cfg("llm.openai.base_url", "https://api.openai.com/v1")
            self.api_key = api_key or cfg("llm.openai.api_key", "")
        else:
            self.model = model or cfg("llm.ollama.model", "llama3.1:8b")
            ollama_host = base_url or cfg("llm.ollama.base_url", "http://localhost:11434")
            self.base_url = _ollama_openai_base_url(ollama_host)
            self.api_key = "ollama"

        self.temperature = cfg("llm.temperature", 0.7)
        self.max_tokens = cfg("llm.max_tokens", 4096)

        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120.0,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        log.debug(f"LLM request → {self.provider}/{self.model}")
        resp = self._client.post("/chat/completions", json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404 and self.provider != "openai":
                body = ""
                try:
                    body = (e.response.text or "")[:400]
                except Exception:
                    pass
                hint = (
                    f"Ollama returned HTTP 404 for {e.request.url!s}. "
                    f"Model in use: {self.model!r}. "
                    "Fixes: (1) `ollama pull " + self.model + "` so that tag exists; "
                    "(2) upgrade Ollama — OpenAI-compatible /v1/chat/completions needs a current release; "
                    "(3) set llm.ollama.base_url / OLLAMA_BASE_URL to the API host only, e.g. "
                    "http://127.0.0.1:11434 (not …/v1 or …/chat/completions)."
                )
                if body.strip():
                    hint += f" Response: {body.strip()!r}"
                raise httpx.HTTPStatusError(hint, request=e.request, response=e.response) from e
            raise
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Chat and parse the response as JSON (with resilient extraction)."""
        raw = self.chat(messages, temperature=temperature)
        return _extract_json(raw)

    def chat_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        temperature: float | None = None,
        inject_schema: bool = True,
        max_tokens: int | None = None,
    ) -> T:
        """Chat and parse into a Pydantic model.

        For Ollama, ``inject_schema=False`` avoids pasting a huge JSON Schema (saves context
        and reduces garbled / truncated outputs). Use when the system prompt already lists fields.
        """
        augmented = list(messages)
        if inject_schema:
            schema = response_model.model_json_schema()
            schema_hint = (
                "You MUST respond with valid JSON matching this schema. "
                "No markdown fences, no extra text — pure JSON only.\n\n"
                f"Schema:\n{json.dumps(schema, indent=2)}"
            )
            if augmented and augmented[0]["role"] == "system":
                augmented[0] = {
                    "role": "system",
                    "content": augmented[0]["content"] + "\n\n" + schema_hint,
                }
            else:
                augmented.insert(0, {"role": "system", "content": schema_hint})

        raw = self.chat(augmented, temperature=temperature, max_tokens=max_tokens)
        data = _extract_json(raw)
        return response_model.model_validate(data)

    def close(self):
        self._client.close()


def _collapse_whitespace_inside_json_strings(s: str) -> str:
    """
    Local LLMs often break JSON by putting real newlines (or tabs) inside "..." values.
    Strict json.loads rejects that; replace with spaces inside quoted segments only.
    """
    out: list[str] = []
    i = 0
    in_string = False
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == '"':
            n_bs = 0
            j = i - 1
            while j >= 0 and s[j] == "\\":
                n_bs += 1
                j -= 1
            if n_bs % 2 == 0:
                in_string = not in_string
            out.append(ch)
            i += 1
            continue
        if in_string and ch in "\n\r\t":
            out.append(" ")
            if ch == "\r" and i + 1 < n and s[i + 1] == "\n":
                i += 2
            else:
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _strip_incomplete_trailing_key(s: str) -> str:
    """Remove a trailing ', "key":' or ', "key": "partial…' that breaks json.loads."""
    s = s.rstrip()
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r",\s*\"[^\"]+\"\s*:\s*$", "", s.rstrip())
        s = re.sub(r",\s*\"[^\"]+\"\s*:\s*(\"([^\"\\]|\\.)*)?$", "", s.rstrip())
    return s.rstrip().rstrip(",")


def _salvage_truncated_object(blob: str) -> dict[str, Any] | None:
    """Close LLM JSON objects that were cut off mid-field (common with local models)."""
    blob = _collapse_whitespace_inside_json_strings(blob.strip())
    blob = _strip_incomplete_trailing_key(blob)
    blob = blob.rstrip().rstrip(",")
    if not blob.endswith("}"):
        blob += "}"
    try:
        data = json.loads(blob)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _loads_json_lenient(blob: str) -> dict[str, Any] | None:
    blob = blob.strip()
    if not blob:
        return None
    for candidate in (_collapse_whitespace_inside_json_strings(blob), blob):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            salvaged = _salvage_truncated_object(candidate)
            if salvaged is not None:
                return salvaged
    return None


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from LLM output that may include markdown fences."""
    text = text.strip()

    parsed = _loads_json_lenient(text)
    if parsed is not None:
        return parsed

    # Strip markdown code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        parsed = _loads_json_lenient(match.group(1))
        if parsed is not None:
            return parsed

    # Find first balanced { ... } block
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                parsed = _loads_json_lenient(text[start : i + 1])
                if parsed is not None:
                    return parsed
                start = None

    raise ValueError(f"Could not extract JSON from LLM response:\n{text[:500]}")
