from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import unicodedata

import pymupdf

try:
    from rapidocr import RapidOCR
except ImportError:  # pragma: no cover - dependency is installed in production.
    RapidOCR = None  # type: ignore[assignment,misc]


OCRCallable = Callable[[bytes], str]


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str
    used_ocr: bool
    quality: str


@dataclass(frozen=True)
class ParsedPDF:
    status: str
    pages: tuple[ParsedPage, ...]
    error_code: str = ""
    error_message: str = ""


def _non_whitespace_count(text: str) -> int:
    return sum(not character.isspace() for character in text)


def _bad_character_ratio(text: str) -> float:
    visible_count = 0
    bad_count = 0
    for character in text:
        if character == "\ufffd" or unicodedata.category(character) == "Cc":
            visible_count += 1
            bad_count += 1
        elif not character.isspace():
            visible_count += 1
    if visible_count == 0:
        return 1.0
    return bad_count / visible_count


def is_low_quality_text(text: str) -> bool:
    """Return whether extracted text warrants local OCR recovery."""

    return _non_whitespace_count(text) < 40 or _bad_character_ratio(text) > 0.30


def _ocr_improves(extracted: str, candidate: str) -> bool:
    candidate_count = _non_whitespace_count(candidate)
    if candidate_count == 0:
        return False
    extracted_count = _non_whitespace_count(extracted)
    return candidate_count > extracted_count or (
        _bad_character_ratio(extracted) > 0.30 and _bad_character_ratio(candidate) < _bad_character_ratio(extracted)
    )


def rapidocr_ocr_factory() -> OCRCallable:
    """Build a local lazy RapidOCR boundary, one engine per parser worker."""

    engine: object | None = None

    def ocr(image_bytes: bytes) -> str:
        nonlocal engine
        if engine is None:
            if RapidOCR is None:
                raise RuntimeError("RapidOCR is not installed")
            engine = RapidOCR()
        output = engine(image_bytes)  # type: ignore[operator]
        return _rapidocr_text(output)

    return ocr


class PDFParser:
    def __init__(self, *, ocr: OCRCallable | None = None) -> None:
        self._ocr = ocr

    def parse(self, path: str | Path) -> ParsedPDF:
        file_path = Path(path)
        if not file_path.is_file():
            return ParsedPDF("failed", (), "pdf_missing", "PDF file does not exist")

        try:
            document = pymupdf.open(file_path)
        except Exception as error:
            return ParsedPDF(
                "failed",
                (),
                _open_error_code(error),
                "Unable to open PDF file",
            )

        try:
            if document.is_encrypted:
                result = ParsedPDF(
                    "failed", (), "pdf_encrypted", "PDF file is encrypted"
                )
            else:
                result = self._parse_document(document)
        except Exception:
            result = ParsedPDF("failed", (), "pdf_open_error", "Unable to read PDF file")
        try:
            document.close()
        except Exception:
            if result.status in {"parsed", "partial"}:
                return result
            return ParsedPDF("failed", (), "pdf_close_error", "Unable to close PDF file")
        return result

    def _parse_document(self, document: pymupdf.Document) -> ParsedPDF:
        pages: list[ParsedPage] = []
        errors: list[str] = []
        for page_number, page in enumerate(document, start=1):
            try:
                extracted = page.get_text("text", sort=True)
            except Exception:
                pages.append(ParsedPage(page_number, "", False, "page_error"))
                errors.append("pdf_page_error")
                continue

            if not is_low_quality_text(extracted):
                pages.append(ParsedPage(page_number, extracted, False, "extracted"))
                continue

            pages.append(self._recover_low_quality_page(page_number, page, extracted, errors))

        if not errors:
            return ParsedPDF("parsed", tuple(pages))
        return ParsedPDF("partial", tuple(pages), errors[0], "One or more PDF pages could not be processed")

    def _recover_low_quality_page(
        self,
        page_number: int,
        page: pymupdf.Page,
        extracted: str,
        errors: list[str],
    ) -> ParsedPage:
        if self._ocr is None:
            return ParsedPage(page_number, extracted, False, "low_quality")
        try:
            pixmap = page.get_pixmap(dpi=200, alpha=False)
            try:
                candidate = self._ocr(pixmap.tobytes("png"))
            finally:
                del pixmap
        except Exception:
            errors.append("pdf_ocr_error")
            return ParsedPage(page_number, extracted, False, "ocr_error")
        if _ocr_improves(extracted, candidate):
            return ParsedPage(page_number, candidate.strip(), True, "ocr")
        return ParsedPage(page_number, extracted, False, "low_quality")


def _open_error_code(error: Exception) -> str:
    message = str(error).lower()
    if "truncated" in message or "unexpected end" in message:
        return "pdf_truncated"
    return "pdf_open_error"


def _rapidocr_text(output: object) -> str:
    if output is None:
        return ""
    texts = getattr(output, "txts", None)
    if texts is not None:
        return _join_texts(texts)
    if not isinstance(output, (list, tuple)):
        return ""
    values: object = output
    if len(output) == 2 and isinstance(output[1], (int, float)):
        values = output[0]
    if not isinstance(values, (list, tuple)):
        return ""
    if all(isinstance(value, str) for value in values):
        return _join_texts(values)
    legacy_texts = [
        row[1]
        for row in values
        if isinstance(row, (list, tuple)) and len(row) >= 2 and isinstance(row[1], str)
    ]
    return _join_texts(legacy_texts)


def _join_texts(values: object) -> str:
    if not isinstance(values, (list, tuple)):
        return ""
    return "\n".join(
        normalized
        for value in values
        if isinstance(value, str) and (normalized := value.strip())
    )
