import json
import os
import re
from typing import Any, Dict, List


DEFAULT_KB_PATH = os.path.join("MycoplasmaPneumonia", "knowledge_base.json")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_KB_PATH = os.path.join(BASE_DIR, "MycoplasmaPneumonia", "knowledge_base.json")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


class RAG_Module:
    """Local evidence retriever for pediatric MPP guidance and literature notes."""

    def __init__(self, openai_api_key: str = "", url_list=None, knowledge_base_path: str = DEFAULT_KB_PATH):
        self.openai_api_key = openai_api_key
        self.knowledge_base_path = knowledge_base_path
        self.url_list = url_list or []
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.knowledge_base_path):
            return []
        with open(self.knowledge_base_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_tokens = _tokens(query)
        scored = []
        for entry in self.knowledge_base:
            evidence_text = " ".join(
                [
                    entry.get("id", ""),
                    entry.get("topic", ""),
                    " ".join(entry.get("tags", [])),
                    entry.get("statement", ""),
                    entry.get("clinical_use", ""),
                ]
            )
            score = len(query_tokens & _tokens(evidence_text))
            if score:
                scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def query(self, query: str, top_k: int = 5) -> str:
        evidence = self.retrieve(query, top_k=top_k)
        if not evidence:
            return "No relevant evidence found in the local pediatric MPP knowledge base."
        lines = []
        for item in evidence:
            lines.append(
                f"[{item.get('id')}] {item.get('topic')}: {item.get('statement')} "
                f"Use: {item.get('clinical_use')}"
            )
        return "\n".join(lines)
