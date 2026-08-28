from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest
from rapidocr.utils.output import RapidOCROutput

import knowledge_base.pdf_parser as pdf_parser_module
from knowledge_base.pdf_parser import (
    PDFParser,
    ParsedPDF,
    ParsedPage,
    is_low_quality_text,
    rapidocr_ocr_factory,
)


def _save_pdf(path: Path, page_texts: list[str]) -> None:
    document = pymupdf.open()
    try:
        for text in page_texts:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        document.save(path)
    finally:
        document.close()


def test_parser_keeps_page_numbers_and_avoids_ocr_for_text_pdf(tmp_path: Path) -> None:
    path = tmp_path / "article.pdf"
    _save_pdf(path, ["Methods randomized trial with enough readable text to skip OCR entirely."])
    calls: list[bytes] = []

    result = PDFParser(ocr=lambda image: calls.append(image) or "unexpected").parse(path)

    assert result.status == "parsed"
    assert result.error_code == ""
    assert result.pages[0] == ParsedPage(
        page_number=1,
        text="Methods randomized trial with enough readable text to skip OCR entirely.",
        used_ocr=False,
        quality="extracted",
    )
    assert calls == []


def test_layout_extraction_reads_each_column_before_moving_to_the_next() -> None:
    class TwoColumnPage:
        rect = SimpleNamespace(width=600.0, height=800.0)

        def get_text(self, kind: str, *, sort: bool) -> object:
            if kind == "text":
                return "legacy alternating extraction"
            assert kind == "blocks"
            assert sort is False
            return [
                (40, 20, 560, 45, "跨栏标题", 0, 0),
                (40, 80, 270, 130, "左栏第一段包含足够多的正文字符用于识别双栏版面。", 1, 0),
                (330, 85, 560, 135, "右栏第一段包含足够多的正文字符用于识别双栏版面。", 2, 0),
                (40, 150, 270, 210, "左栏第二段应当紧跟左栏第一段而不是右栏第一段。", 3, 0),
                (330, 155, 560, 215, "右栏第二段应当在左栏内容全部完成之后出现。", 4, 0),
            ]

    text = pdf_parser_module._layout_aware_page_text(TwoColumnPage())

    assert text.index("跨栏标题") < text.index("左栏第一段")
    assert text.index("左栏第一段") < text.index("左栏第二段")
    assert text.index("左栏第二段") < text.index("右栏第一段")
    assert text.index("右栏第一段") < text.index("右栏第二段")


def test_layout_extraction_resets_column_order_after_cross_column_divider() -> None:
    class SectionedPage:
        rect = SimpleNamespace(width=600.0, height=800.0)

        def get_text(self, kind: str, *, sort: bool) -> object:
            if kind == "text":
                return "legacy extraction"
            return [
                (40, 60, 270, 120, "上半页左栏正文内容足够长以建立双栏判断。", 1, 0),
                (330, 60, 560, 120, "上半页右栏正文内容足够长以建立双栏判断。", 2, 0),
                (40, 150, 560, 180, "跨栏小节标题", 3, 0),
                (40, 210, 270, 270, "下半页左栏正文内容应先于下半页右栏内容。", 4, 0),
                (330, 210, 560, 270, "下半页右栏正文内容应在最后读取。", 5, 0),
            ]

    text = pdf_parser_module._layout_aware_page_text(SectionedPage())

    assert text.index("上半页左栏") < text.index("上半页右栏")
    assert text.index("上半页右栏") < text.index("跨栏小节标题")
    assert text.index("跨栏小节标题") < text.index("下半页左栏")
    assert text.index("下半页左栏") < text.index("下半页右栏")


def test_parser_uses_injected_ocr_only_for_empty_low_quality_page(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    _save_pdf(path, [""])
    images: list[bytes] = []

    result = PDFParser(ocr=lambda image: images.append(image) or "OCR result").parse(path)

    assert result.status == "parsed"
    assert result.pages == (ParsedPage(1, "OCR result", True, "ocr"),)
    assert len(images) == 1
    assert images[0].startswith(b"\x89PNG")


def test_quality_threshold_and_control_character_ratio_trigger_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "quality.pdf"
    _save_pdf(path, ["short", "normal text that is long enough to be usable for indexing."])
    original_get_text = pymupdf.Page.get_text

    def controlled_text(page: pymupdf.Page, *args: object, **kwargs: object) -> str:
        if page.number == 1:
            return "normal text that is long enough to be usable for indexing but " + "\ufffd" * 32
        return original_get_text(page, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_text", controlled_text)
    calls: list[bytes] = []

    result = PDFParser(ocr=lambda image: calls.append(image) or "replacement-free OCR text").parse(path)

    assert [page.used_ocr for page in result.pages] == [True, True]
    assert [page.quality for page in result.pages] == ["ocr", "ocr"]
    assert len(calls) == 2
    assert is_low_quality_text("valid text " + "\ufffd" * 32) is True


@pytest.mark.parametrize("control", ["\x7f", "\x80", "\x85"])
def test_quality_threshold_counts_del_and_c1_control_characters(control: str) -> None:
    assert is_low_quality_text("A" * 69 + control * 31) is True
    assert is_low_quality_text("A" * 70 + control * 30) is False


def test_parser_retains_other_pages_when_one_page_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "two-pages.pdf"
    _save_pdf(path, ["First readable page contains enough normal content for extraction.", "Second readable page contains enough normal content for extraction."])
    original_get_text = pymupdf.Page.get_text

    def failing_second_page(page: pymupdf.Page, *args: object, **kwargs: object) -> str:
        if page.number == 1:
            raise RuntimeError("broken page")
        return original_get_text(page, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_text", failing_second_page)

    result = PDFParser().parse(path)

    assert result.status == "partial"
    assert result.error_code == "pdf_page_error"
    assert result.pages[0].text.startswith("First readable")
    assert result.pages[1] == ParsedPage(2, "", False, "page_error")


@pytest.mark.parametrize(
    ("path_name", "payload", "error_code"),
    [
        ("missing.pdf", None, "pdf_missing"),
        ("corrupt.pdf", b"not a PDF", "pdf_open_error"),
    ],
)
def test_parser_has_stable_file_boundary_errors(
    tmp_path: Path, path_name: str, payload: bytes | None, error_code: str
) -> None:
    path = tmp_path / path_name
    if payload is not None:
        path.write_bytes(payload)

    result = PDFParser().parse(path)

    assert result == ParsedPDF(status="failed", pages=(), error_code=error_code, error_message=result.error_message)
    assert result.error_message


def test_parser_reports_encrypted_pdf_without_throwing(tmp_path: Path) -> None:
    path = tmp_path / "encrypted.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    document.close()

    result = PDFParser().parse(path)

    assert result.status == "failed"
    assert result.error_code == "pdf_encrypted"


def test_parser_distinguishes_truncated_open_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "truncated.pdf"
    path.write_bytes(b"placeholder")

    def truncated(_: Path) -> pymupdf.Document:
        raise RuntimeError("truncated PDF stream")

    monkeypatch.setattr(pdf_parser_module.pymupdf, "open", truncated)

    assert PDFParser().parse(path).error_code == "pdf_truncated"


def test_parser_preserves_successful_parse_when_document_close_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "close-error.pdf"
    path.write_bytes(b"placeholder")

    class ClosingDocument:
        is_encrypted = False
        page_count = 0

        def __iter__(self):
            return iter(())

        def close(self) -> None:
            raise RuntimeError("close failed")

    monkeypatch.setattr(pdf_parser_module.pymupdf, "open", lambda _: ClosingDocument())

    assert PDFParser().parse(path) == ParsedPDF("parsed", ())


def test_parser_uses_page_indexes_and_retains_pages_around_load_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "load-error.pdf"
    path.write_bytes(b"placeholder")

    class FakePage:
        def __init__(self, text: str) -> None:
            self.text = text

        def get_text(self, _: str, *, sort: bool) -> str:
            assert sort is True
            return self.text

    class IndexedDocument:
        is_encrypted = False
        page_count = 3

        def __iter__(self):
            raise AssertionError("parser must not use the document iterator")

        def load_page(self, index: int) -> FakePage:
            if index == 1:
                raise RuntimeError("page load failure")
            return FakePage("Readable extracted text that is long enough for normal indexing.")

        def close(self) -> None:
            return None

    monkeypatch.setattr(pdf_parser_module.pymupdf, "open", lambda _: IndexedDocument())

    result = PDFParser().parse(path)

    assert result.status == "partial"
    assert result.error_code == "pdf_page_error"
    assert [page.page_number for page in result.pages] == [1, 2, 3]
    assert result.pages[1] == ParsedPage(2, "", False, "page_error")
    assert result.pages[2].quality == "extracted"


def test_parser_keeps_exact_200_dpi_for_normal_ocr_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "normal-render.pdf"
    path.write_bytes(b"placeholder")
    calls: list[tuple[int, bool]] = []

    class FakePixmap:
        def tobytes(self, kind: str) -> bytes:
            assert kind == "png"
            return b"png"

    class FakePage:
        rect = SimpleNamespace(width=72.0, height=72.0)

        def get_text(self, _: str, *, sort: bool) -> str:
            return ""

        def get_pixmap(self, *, dpi: int, alpha: bool) -> FakePixmap:
            calls.append((dpi, alpha))
            return FakePixmap()

    class FakeDocument:
        is_encrypted = False
        page_count = 1

        def load_page(self, _: int) -> FakePage:
            return FakePage()

        def close(self) -> None:
            return None

    monkeypatch.setattr(pdf_parser_module.pymupdf, "open", lambda _: FakeDocument())

    result = PDFParser(ocr=lambda _: "usable OCR text").parse(path)

    assert calls == [(200, False)]
    assert result.pages[0].quality == "ocr"


def test_parser_reduces_ocr_dpi_to_stay_within_pixel_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "large-render.pdf"
    path.write_bytes(b"placeholder")
    calls: list[tuple[int, bool]] = []

    class FakePixmap:
        def tobytes(self, _: str) -> bytes:
            return b"png"

    class FakePage:
        rect = SimpleNamespace(width=720.0, height=720.0)

        def get_text(self, _: str, *, sort: bool) -> str:
            return ""

        def get_pixmap(self, *, dpi: int, alpha: bool) -> FakePixmap:
            calls.append((dpi, alpha))
            return FakePixmap()

    class FakeDocument:
        is_encrypted = False
        page_count = 1

        def load_page(self, _: int) -> FakePage:
            return FakePage()

        def close(self) -> None:
            return None

    monkeypatch.setattr(pdf_parser_module.pymupdf, "open", lambda _: FakeDocument())

    result = PDFParser(
        ocr=lambda _: "usable OCR text", max_ocr_pixels=1_000_000
    ).parse(path)

    assert calls == [(100, False)]
    assert result.pages[0].quality == "ocr_reduced"


def test_parser_skips_ocr_when_even_minimum_dpi_exceeds_pixel_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "too-large.pdf"
    path.write_bytes(b"placeholder")

    class FakePage:
        rect = SimpleNamespace(width=7200.0, height=7200.0)

        def get_text(self, _: str, *, sort: bool) -> str:
            return ""

        def get_pixmap(self, **_: object) -> object:
            raise AssertionError("unsafe pixmap rendering should not occur")

    class FakeDocument:
        is_encrypted = False
        page_count = 1

        def load_page(self, _: int) -> FakePage:
            return FakePage()

        def close(self) -> None:
            return None

    monkeypatch.setattr(pdf_parser_module.pymupdf, "open", lambda _: FakeDocument())

    result = PDFParser(ocr=lambda _: "unused", max_ocr_pixels=1).parse(path)

    assert result.status == "partial"
    assert result.error_code == "pdf_ocr_too_large"
    assert result.pages == (ParsedPage(1, "", False, "ocr_too_large"),)


def test_ocr_failure_stays_at_page_boundary(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    _save_pdf(path, [""])

    def bad_ocr(_: bytes) -> str:
        raise RuntimeError("OCR unavailable")

    result = PDFParser(ocr=bad_ocr).parse(path)

    assert result.status == "partial"
    assert result.error_code == "pdf_ocr_error"
    assert result.pages == (ParsedPage(1, "", False, "ocr_error"),)


def test_ocr_text_must_improve_low_quality_extraction(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    _save_pdf(path, ["tiny"])

    result = PDFParser(ocr=lambda _: " ").parse(path)

    assert result.pages == (ParsedPage(1, "tiny", False, "low_quality"),)


def test_rapidocr_factory_initializes_once_and_joins_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[object] = []

    class FakeOutput:
        result = object()
        txts = [" first ", "", "second"]

    class FakeRapidOCR:
        def __init__(self) -> None:
            created.append(self)

        def __call__(self, image: bytes) -> FakeOutput:
            assert image == b"png"
            return FakeOutput()

    monkeypatch.setattr(pdf_parser_module, "RapidOCR", FakeRapidOCR)
    ocr = rapidocr_ocr_factory()

    assert ocr(b"png") == "first\nsecond"
    assert ocr(b"png") == "first\nsecond"
    assert len(created) == 1


def test_rapidocr_factory_accepts_installed_rapidocr_output_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRapidOCR:
        def __call__(self, _: bytes) -> RapidOCROutput:
            return RapidOCROutput(txts=("first", "", "second"))

    monkeypatch.setattr(pdf_parser_module, "RapidOCR", FakeRapidOCR)

    assert rapidocr_ocr_factory()(b"png") == "first\nsecond"


def test_rapidocr_factory_accepts_legacy_result_and_elapsed_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRapidOCR:
        def __call__(self, _: bytes) -> tuple[list[list[object]], float]:
            return ([[[0, 0, 1, 1], "first", 0.9], [[0, 0, 1, 1], "second", 0.8]], 0.01)

    monkeypatch.setattr(pdf_parser_module, "RapidOCR", FakeRapidOCR)

    assert rapidocr_ocr_factory()(b"png") == "first\nsecond"


@pytest.mark.parametrize("output", [None, object()])
def test_rapidocr_factory_handles_missing_or_empty_results(
    monkeypatch: pytest.MonkeyPatch, output: object | None
) -> None:
    class FakeRapidOCR:
        def __call__(self, _: bytes) -> object | None:
            return output

    monkeypatch.setattr(pdf_parser_module, "RapidOCR", FakeRapidOCR)

    assert rapidocr_ocr_factory()(b"png") == ""
