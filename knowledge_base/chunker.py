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
    token_spans: tuple[tuple[int, int], ...]
    source_id: int
    page_number: int
    section: str
    used_ocr: bool
    quality: str

    def slice_tokens(self, start: int, end: int) -> "_Fragment":
        if start < 0 or end <= start or end > len(self.token_spans):
            raise ValueError("invalid token slice")
        start_offset = 0 if start == 0 else self.token_spans[start][0]
        end_offset = len(self.text) if end == len(self.token_spans) else self.token_spans[end][0]
        return _Fragment(
            text=self.text[start_offset:end_offset],
            token_spans=tuple(
                (match_start - start_offset, match_end - start_offset)
                for match_start, match_end in self.token_spans[start:end]
            ),
            source_id=self.source_id,
            page_number=self.page_number,
            section=self.section,
            used_ocr=self.used_ocr,
            quality=self.quality,
        )


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
            if current and _token_total(current) + len(part.token_spans) > target_tokens:
                chunks.append(_make_chunk(document_id, file_id, current))
                current = _tail_fragments(current, overlap_tokens)
                # The retained tail belongs to the next window.  It must be
                # combined with the incoming part rather than emitted alone.
            if current and _token_total(current) + len(part.token_spans) > max_chunk_tokens:
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
    source_id = 0
    for page in pages:
        paragraph_lines: list[str] = []
        for raw_line in page.text.splitlines(keepends=True):
            line = raw_line.strip()
            if not line:
                yield from _paragraph_fragment("".join(paragraph_lines), page, section, source_id)
                source_id += 1
                paragraph_lines = []
                continue
            heading = _HEADING_RE.match(line)
            if heading:
                yield from _paragraph_fragment("".join(paragraph_lines), page, section, source_id)
                source_id += 1
                paragraph_lines = []
                section = heading.group(1)
            else:
                paragraph_lines.append(raw_line)
        yield from _paragraph_fragment("".join(paragraph_lines), page, section, source_id)
        source_id += 1


def _paragraph_fragment(
    text: str, page: ParsedPage, section: str, source_id: int
) -> Iterable[_Fragment]:
    token_spans = tuple((match.start(), match.end()) for match in _TOKEN_RE.finditer(text))
    if token_spans:
        yield _Fragment(
            text,
            token_spans,
            source_id,
            page.page_number,
            section,
            page.used_ocr,
            page.quality,
        )


def _split_fragment(fragment: _Fragment, target_tokens: int) -> Iterable[_Fragment]:
    if len(fragment.token_spans) <= target_tokens:
        yield fragment
        return
    for start in range(0, len(fragment.token_spans), target_tokens):
        yield fragment.slice_tokens(start, min(start + target_tokens, len(fragment.token_spans)))


def _tail_fragments(fragments: list[_Fragment], overlap_tokens: int) -> list[_Fragment]:
    if overlap_tokens == 0:
        return []
    selected: list[_Fragment] = []
    remaining = overlap_tokens
    for fragment in reversed(fragments):
        if remaining <= 0:
            break
        take = min(remaining, len(fragment.token_spans))
        selected.append(fragment.slice_tokens(len(fragment.token_spans) - take, len(fragment.token_spans)))
        remaining -= take
    return list(reversed(selected))


def _token_total(fragments: Sequence[_Fragment]) -> int:
    return sum(len(fragment.token_spans) for fragment in fragments)


def _make_chunk(document_id: str, file_id: str, fragments: Sequence[_Fragment]) -> ChunkRecord:
    text = _join_fragment_texts(fragments)
    page_start = min(fragment.page_number for fragment in fragments)
    page_end = max(fragment.page_number for fragment in fragments)
    section = fragments[0].section
    is_ocr = any(fragment.used_ocr for fragment in fragments)
    quality = _conservative_quality(fragments, is_ocr)
    token_count = approximate_token_count(text)
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


def _join_fragment_texts(fragments: Sequence[_Fragment]) -> str:
    parts = [fragments[0].text]
    for previous, fragment in zip(fragments, fragments[1:]):
        separator = "" if previous.source_id == fragment.source_id else "\n\n"
        parts.append(separator + fragment.text)
    return "".join(parts)


def _conservative_quality(fragments: Sequence[_Fragment], is_ocr: bool) -> str:
    if is_ocr:
        return "ocr"
    qualities = {fragment.quality for fragment in fragments}
    if "low_quality" in qualities:
        return "low_quality"
    if "ocr_error" in qualities or "page_error" in qualities:
        return "degraded"
    return "extracted"
