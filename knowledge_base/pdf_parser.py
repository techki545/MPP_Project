from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Callable
import unicodedata

import pymupdf

try:
    from rapidocr import RapidOCR
except ImportError:  # pragma: no cover - dependency is installed in production.
    RapidOCR = None  # type: ignore[assignment,misc]


OCRCallable = Callable[[bytes], str]
DEFAULT_OCR_DPI = 200
DEFAULT_MAX_OCR_PIXELS = 4_000_000
_PAGE_NUMBER_RE = re.compile(r"^(?:第\s*)?\d{1,4}(?:\s*页)?$")


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


@dataclass(frozen=True)
class _TextBlock:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2


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
    def __init__(
        self,
        *,
        ocr: OCRCallable | None = None,
        max_ocr_pixels: int = DEFAULT_MAX_OCR_PIXELS,
    ) -> None:
        if not isinstance(max_ocr_pixels, int) or max_ocr_pixels <= 0:
            raise ValueError("max_ocr_pixels must be a positive integer")
        self._ocr = ocr
        self._max_ocr_pixels = max_ocr_pixels

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
        try:
            page_count = int(document.page_count)
        except Exception:
            return ParsedPDF("failed", (), "pdf_open_error", "Unable to read PDF page count")
        for page_index in range(page_count):
            page_number = page_index + 1
            try:
                page = document.load_page(page_index)
                extracted = _layout_aware_page_text(page)
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
        render_dpi = self._safe_ocr_dpi(page)
        if render_dpi is None:
            errors.append("pdf_ocr_too_large")
            return ParsedPage(page_number, extracted, False, "ocr_too_large")
        try:
            pixmap = page.get_pixmap(dpi=render_dpi, alpha=False)
            try:
                candidate = self._ocr(pixmap.tobytes("png"))
            finally:
                del pixmap
        except Exception:
            errors.append("pdf_ocr_error")
            return ParsedPage(page_number, extracted, False, "ocr_error")
        if _ocr_improves(extracted, candidate):
            quality = "ocr" if render_dpi == DEFAULT_OCR_DPI else "ocr_reduced"
            return ParsedPage(page_number, candidate.strip(), True, quality)
        quality = "low_quality" if render_dpi == DEFAULT_OCR_DPI else "low_quality_reduced"
        return ParsedPage(page_number, extracted, False, quality)

    def _safe_ocr_dpi(self, page: pymupdf.Page) -> int | None:
        try:
            width_points = float(page.rect.width)
            height_points = float(page.rect.height)
        except Exception:
            return None
        if width_points <= 0 or height_points <= 0:
            return None
        if _pixel_count(width_points, height_points, DEFAULT_OCR_DPI) <= self._max_ocr_pixels:
            return DEFAULT_OCR_DPI
        scale = math.sqrt(
            self._max_ocr_pixels
            / _pixel_count(width_points, height_points, DEFAULT_OCR_DPI)
        )
        dpi = max(1, math.floor(DEFAULT_OCR_DPI * scale))
        while dpi >= 1 and _pixel_count(width_points, height_points, dpi) > self._max_ocr_pixels:
            dpi -= 1
        return dpi or None


def _layout_aware_page_text(page: pymupdf.Page) -> str:
    """Extract text in human reading order, including common two-column pages."""

    try:
        raw_blocks = page.get_text("blocks", sort=False)
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)
    except Exception:
        return str(page.get_text("text", sort=True)).strip()
    blocks = _text_blocks(raw_blocks, page_height)
    if not blocks:
        return str(page.get_text("text", sort=True)).strip()
    ordered = _reading_order(blocks, page_width)
    return "\n\n".join(block.text for block in ordered if block.text).strip()


def _text_blocks(raw_blocks: object, page_height: float) -> list[_TextBlock]:
    if not isinstance(raw_blocks, (list, tuple)):
        return []
    blocks: list[_TextBlock] = []
    for raw in raw_blocks:
        if not isinstance(raw, (list, tuple)) or len(raw) < 5:
            continue
        if len(raw) >= 7 and raw[6] != 0:
            continue
        try:
            x0, y0, x1, y1 = (float(raw[index]) for index in range(4))
        except (TypeError, ValueError):
            continue
        text = _normalize_block_text(raw[4])
        if not text or x1 <= x0 or y1 <= y0:
            continue
        if y0 >= page_height * 0.94 and _PAGE_NUMBER_RE.fullmatch(text):
            continue
        blocks.append(_TextBlock(x0, y0, x1, y1, text))
    return blocks


def _normalize_block_text(value: object) -> str:
    lines = [" ".join(line.split()) for line in str(value or "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _reading_order(blocks: list[_TextBlock], page_width: float) -> list[_TextBlock]:
    if len(blocks) < 2 or page_width <= 0:
        return sorted(blocks, key=lambda block: (block.y0, block.x0))
    center = page_width / 2
    gutter = max(12.0, page_width * 0.035)
    left: list[_TextBlock] = []
    right: list[_TextBlock] = []
    spanning: list[_TextBlock] = []
    for block in blocks:
        if block.x1 <= center + gutter and block.center_x < center:
            left.append(block)
        elif block.x0 >= center - gutter and block.center_x >= center:
            right.append(block)
        else:
            spanning.append(block)
    if not _is_two_column(left, right):
        return sorted(blocks, key=lambda block: (block.y0, block.x0))

    remaining_left = sorted(left, key=lambda block: (block.y0, block.x0))
    remaining_right = sorted(right, key=lambda block: (block.y0, block.x0))
    ordered: list[_TextBlock] = []
    for divider in sorted(spanning, key=lambda block: (block.y0, block.x0)):
        ordered.extend(_take_region(remaining_left, divider.y0))
        ordered.extend(_take_region(remaining_right, divider.y0))
        ordered.append(divider)
    ordered.extend(remaining_left)
    ordered.extend(remaining_right)
    return ordered


def _is_two_column(left: list[_TextBlock], right: list[_TextBlock]) -> bool:
    if not left or not right:
        return False
    left_chars = sum(_non_whitespace_count(block.text) for block in left)
    right_chars = sum(_non_whitespace_count(block.text) for block in right)
    if min(left_chars, right_chars) < 30:
        return False
    left_top, left_bottom = min(block.y0 for block in left), max(block.y1 for block in left)
    right_top, right_bottom = min(block.y0 for block in right), max(block.y1 for block in right)
    overlap = min(left_bottom, right_bottom) - max(left_top, right_top)
    shorter_height = min(left_bottom - left_top, right_bottom - right_top)
    return overlap >= max(18.0, shorter_height * 0.15)


def _take_region(blocks: list[_TextBlock], before_y: float) -> list[_TextBlock]:
    selected = [block for block in blocks if block.center_y < before_y]
    if selected:
        selected_ids = {id(block) for block in selected}
        blocks[:] = [block for block in blocks if id(block) not in selected_ids]
    return selected


def _open_error_code(error: Exception) -> str:
    message = str(error).lower()
    if "truncated" in message or "unexpected end" in message:
        return "pdf_truncated"
    return "pdf_open_error"


def _pixel_count(width_points: float, height_points: float, dpi: int) -> int:
    width = math.ceil(width_points * dpi / 72)
    height = math.ceil(height_points * dpi / 72)
    return width * height


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
