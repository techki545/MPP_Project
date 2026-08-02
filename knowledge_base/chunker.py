from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Iterable, Sequence

from .models import ChunkRecord
from .pdf_parser import ParsedPage


DEFAULT_TARGET_TOKENS = 700
DEFAULT_OVERLAP_TOKENS = 100
DEFAULT_MAX_CHUNK_TOKENS = 900
_TOKEN_RE = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9]+")
_HEADING_RE = re.compile(
    r"^(摘要|abstract|方法|methods|结果|results|讨论|discussion|结论|conclusion|参考文献|references)\s*[:：]?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Fragment:
    text: str
    tokens: tuple[str, ...]
    page_number: int
    section: str
    used_ocr: bool
    quality: str


def approximate_token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def chunk_pages(
    document_id: str,
    file_id: str,
    pages: Sequence[ParsedPage],
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
) -> tuple[ChunkRecord, ...]:
    _validate_window(target_tokens, overlap_tokens, max_chunk_tokens)
    fragments = tuple(_page_fragments(pages))
    if not fragments:
        return ()

    chunks: list[ChunkRecord] = []
    current: list[_Fragment] = []
    current_section = ""
    for fragment in fragments:
        if current and fragment.section != current_section:
            chunks.append(_make_chunk(document_id, file_id, current))
            current = []
        if not current:
            current_section = fragment.section

        for part in _split_fragment(fragment, target_tokens):
            if current and _token_total(current) + len(part.tokens) > target_tokens:
                chunks.append(_make_chunk(document_id, file_id, current))
                current = _tail_fragments(current, overlap_tokens)
                # The retained tail belongs to the next window.  It must be
                # combined with the incoming part rather than emitted alone.
            if current and _token_total(current) + len(part.tokens) > max_chunk_tokens:
                current = _tail_fragments(current, overlap_tokens)
            current.append(part)
            current_section = part.section

    if current:
        chunks.append(_make_chunk(document_id, file_id, current))
    return tuple(chunks)


def _validate_window(target_tokens: int, overlap_tokens: int, max_chunk_tokens: int) -> None:
    if target_tokens <= 0 or max_chunk_tokens <= 0:
        raise ValueError("target and maximum chunk token counts must be positive")
    if overlap_tokens < 0 or overlap_tokens >= target_tokens:
        raise ValueError("overlap token count must be non-negative and less than target")
    if target_tokens > max_chunk_tokens or target_tokens + overlap_tokens > max_chunk_tokens:
        raise ValueError("target plus overlap must not exceed maximum chunk size")


def _page_fragments(pages: Sequence[ParsedPage]) -> Iterable[_Fragment]:
    section = ""
    for page in pages:
        paragraph_lines: list[str] = []
        for raw_line in page.text.splitlines():
            line = raw_line.strip()
            if not line:
                yield from _paragraph_fragment(paragraph_lines, page, section)
                paragraph_lines = []
                continue
            heading = _HEADING_RE.match(line)
            if heading:
                yield from _paragraph_fragment(paragraph_lines, page, section)
                paragraph_lines = []
                section = heading.group(1)
            else:
                paragraph_lines.append(line)
        yield from _paragraph_fragment(paragraph_lines, page, section)


def _paragraph_fragment(
    lines: list[str], page: ParsedPage, section: str
) -> Iterable[_Fragment]:
    text = " ".join(lines).strip()
    tokens = tuple(_TOKEN_RE.findall(text))
    if tokens:
        yield _Fragment(text, tokens, page.page_number, section, page.used_ocr, page.quality)


def _split_fragment(fragment: _Fragment, target_tokens: int) -> Iterable[_Fragment]:
    if len(fragment.tokens) <= target_tokens:
        yield fragment
        return
    for start in range(0, len(fragment.tokens), target_tokens):
        tokens = fragment.tokens[start : start + target_tokens]
        yield _Fragment(
            " ".join(tokens),
            tokens,
            fragment.page_number,
            fragment.section,
            fragment.used_ocr,
            fragment.quality,
        )


def _tail_fragments(fragments: list[_Fragment], overlap_tokens: int) -> list[_Fragment]:
    if overlap_tokens == 0:
        return []
    selected: list[_Fragment] = []
    remaining = overlap_tokens
    for fragment in reversed(fragments):
        if remaining <= 0:
            break
        tail = fragment.tokens[-remaining:]
        selected.append(
            _Fragment(
                " ".join(tail),
                tail,
                fragment.page_number,
                fragment.section,
                fragment.used_ocr,
                fragment.quality,
            )
        )
        remaining -= len(tail)
    return list(reversed(selected))


def _token_total(fragments: Sequence[_Fragment]) -> int:
    return sum(len(fragment.tokens) for fragment in fragments)


def _make_chunk(document_id: str, file_id: str, fragments: Sequence[_Fragment]) -> ChunkRecord:
    text = "\n\n".join(fragment.text for fragment in fragments)
    page_start = min(fragment.page_number for fragment in fragments)
    page_end = max(fragment.page_number for fragment in fragments)
    section = fragments[0].section
    is_ocr = any(fragment.used_ocr for fragment in fragments)
    quality = _conservative_quality(fragments, is_ocr)
    token_count = _token_total(fragments)
    digest = sha256(
        f"{document_id}|{file_id}|{page_start}|{page_end}|{section}|{text}".encode("utf-8")
    ).hexdigest()
    return ChunkRecord(
        chunk_id=digest,
        document_id=document_id,
        file_id=file_id,
        section=section,
        page_start=page_start,
        page_end=page_end,
        text=text,
        token_count=token_count,
        is_ocr=is_ocr,
        quality=quality,
        content_hash=sha256(text.encode("utf-8")).hexdigest(),
    )


def _conservative_quality(fragments: Sequence[_Fragment], is_ocr: bool) -> str:
    if is_ocr:
        return "ocr"
    qualities = {fragment.quality for fragment in fragments}
    if "low_quality" in qualities:
        return "low_quality"
    if "ocr_error" in qualities or "page_error" in qualities:
        return "degraded"
    return "extracted"
