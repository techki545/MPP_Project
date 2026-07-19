import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _disabled(value: str) -> bool:
    return str(value or "").strip().lower() in {"0", "false", "no", "n", "off", "disabled"}


@dataclass
class LLMConfig:
    api_key: str
    model: str
    base_url: str
    timeout: int
    enabled: bool


def load_llm_config() -> LLMConfig:
    api_key = os.getenv("MPP_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    explicit = os.getenv("MPP_USE_LLM", "")
    allow_no_key = _truthy(os.getenv("MPP_LLM_ALLOW_NO_KEY", ""))
    enabled = bool(api_key)
    if explicit:
        enabled = _truthy(explicit) and not _disabled(explicit) and (bool(api_key) or allow_no_key)
    return LLMConfig(
        api_key=api_key,
        model=os.getenv("MPP_LLM_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("MPP_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        timeout=int(os.getenv("MPP_LLM_TIMEOUT", "60")),
        enabled=enabled,
    )


def extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        chunks = [part.strip() for part in text.split("```") if part.strip()]
        for chunk in chunks:
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("{"):
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


class OpenAICompatibleLLM:
    """Minimal OpenAI-compatible chat-completions client.

    It supports OpenAI and compatible local/cloud gateways through:
    - MPP_LLM_API_KEY or OPENAI_API_KEY
    - MPP_LLM_BASE_URL, default https://api.openai.com/v1
    - MPP_LLM_MODEL, default gpt-4o-mini
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or load_llm_config()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def chat_json(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("LLM is disabled or API key is missing.")

        payload = {
            "model": self.config.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False, indent=2),
                },
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        request = urllib.request.Request(
            url=f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP error {exc.code}: {detail}") from exc

        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        return extract_json_object(content)


def get_llm_client() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM()
