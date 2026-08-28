"""Conservative cleanup for evidence text extracted from journal PDFs."""

from __future__ import annotations

import re
import unicodedata


_INTERNAL_CITATION_PATTERN = re.compile(
    r"[\[［]\s*\d+(?:\s*(?:[-–—,，;；]|至)\s*\d+)*\s*[\]］]"
)
_REFERENCE_MARKER_PATTERN = re.compile(
    r"(?:参考文献|references\b|[\[［]\s*[JM]\s*[\]］])",
    re.IGNORECASE,
)
_JOURNAL_CITATION_PATTERN = re.compile(
    r"(?:19|20)\d{2}\s*[,，]\s*\d+\s*[（(]\s*\d+\s*[)）]\s*[:：]\s*\d+"
)
_PRIVATE_USE_PATTERN = re.compile(r"[\ue000-\uf8ff]")
_CJK_GAP_PATTERN = re.compile(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])")
_DOCUMENT_HEADER_PATTERN = re.compile(
    r"(?:新疆医科大学|重庆医科大学)(?:硕士|博士)(?:研究生)?学位论文"
)


def clean_evidence_text(text: object, *, limit: int | None = None) -> str:
    """Remove citation spill and extraction artifacts without rewriting facts."""

    if not isinstance(text, str):
        return ""
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = _PRIVATE_USE_PATTERN.sub("", cleaned)
    cleaned = _DOCUMENT_HEADER_PATTERN.sub("", cleaned)
    marker = _REFERENCE_MARKER_PATTERN.search(cleaned)
    if marker and marker.start() >= 24:
        cutoff = marker.start()
        earlier_citations = [
            match
            for match in _INTERNAL_CITATION_PATTERN.finditer(cleaned[:cutoff])
            if match.start() >= 12 and cutoff - match.start() <= 160
        ]
        if earlier_citations:
            cutoff = earlier_citations[-1].start()
        cleaned = cleaned[:cutoff]
    cleaned = _INTERNAL_CITATION_PATTERN.sub("", cleaned)
    cleaned = _CJK_GAP_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\r\n;；")
    if limit is not None and len(cleaned) > limit:
        cleaned = _truncate_sentence(cleaned, limit)
    return cleaned


def reference_like_text(text: object) -> bool:
    """Identify clauses dominated by bibliography rather than clinical prose."""

    if not isinstance(text, str) or not text.strip():
        return True
    normalized = unicodedata.normalize("NFKC", text)
    if _REFERENCE_MARKER_PATTERN.search(normalized):
        return True
    if _JOURNAL_CITATION_PATTERN.search(normalized):
        return True
    return False


def _truncate_sentence(text: str, limit: int) -> str:
    candidate = text[:limit]
    stop = max(candidate.rfind(character) for character in "。！？.!?")
    if stop >= max(40, limit // 2):
        return candidate[: stop + 1].strip()
    return candidate.rstrip()
