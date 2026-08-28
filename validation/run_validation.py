"""Run a reproducible retrieval baseline over the standard MPP questions."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_base.config import Settings
from knowledge_base.errors import KnowledgeBaseError
from knowledge_base.models import SearchFilters
from knowledge_base.service import create_production_service


_CITATION_PATTERN = re.compile(r"\[(\d+)]")


@dataclass(frozen=True)
class CaseAssessment:
    case_id: str
    expected_source_hit: bool
    duplicate_document_count: int
    unsupported_citation_count: int
    all_citations_supported: bool
    evidence_type_coverage: float
    retrieval_mode: str
    citation_count: int
    retrieved_evidence_types: tuple[str, ...]
    query_understanding_present: bool
    question_type_match: bool | None
    citation_locator_coverage: float

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["retrieved_evidence_types"] = list(self.retrieved_evidence_types)
        return payload


class HttpValidationService:
    """Run validation through the same HTTP route used by the browser."""

    def __init__(self, base_url: str, timeout_seconds: int = 180) -> None:
        self.query_url = f"{base_url.rstrip('/')}/api/query"
        self.timeout_seconds = timeout_seconds

    def query(self, question: str, filters: SearchFilters) -> dict[str, Any]:
        payload = {"question": question, **_filters_payload(filters)}
        request = Request(
            self.query_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Validation API returned HTTP {error.code}: {body[:500]}") from error
        except URLError as error:
            raise RuntimeError(f"Validation API is unavailable: {error.reason}") from error
        if not isinstance(result, dict):
            raise RuntimeError("Validation API returned a non-object response")
        return result


def evaluate_case(
    case: Mapping[str, Any], result: Mapping[str, Any]
) -> CaseAssessment:
    sources = [
        dict(source)
        for source in result.get("sources", [])
        if isinstance(source, Mapping)
    ]
    expected_title_terms = _normalized_terms(case.get("expected_title_terms", []))
    expected_doi_terms = _normalized_terms(case.get("expected_doi_terms", []))
    title_match = bool(expected_title_terms) and any(
        all(term in _normalize(source.get("title", "")) for term in expected_title_terms)
        for source in sources
    )
    doi_match = bool(expected_doi_terms) and any(
        any(term in _normalize_doi(source.get("doi", "")) for term in expected_doi_terms)
        for source in sources
    )
    expected_source_hit = title_match or doi_match

    document_keys = [_document_key(source) for source in sources]
    duplicate_document_count = len(document_keys) - len(set(document_keys))

    citations = {
        int(value)
        for value in _CITATION_PATTERN.findall(str(result.get("answer_markdown", "")))
    }
    valid_numbers = {
        _source_number(source, index)
        for index, source in enumerate(sources, start=1)
    }
    explicit_support = result.get("source_support", {})
    if not isinstance(explicit_support, Mapping):
        explicit_support = {}
    source_by_number = {
        _source_number(source, index): source
        for index, source in enumerate(sources, start=1)
    }
    unsupported = set()
    for citation in citations:
        source = source_by_number.get(citation)
        explicit = explicit_support.get(
            str(citation), explicit_support.get(citation, None)
        )
        has_evidence_text = bool(source and _source_has_evidence_text(source))
        if (
            citation not in valid_numbers
            or explicit is False
            or (explicit is not True and not has_evidence_text)
        ):
            unsupported.add(citation)
    unsupported_count = len(unsupported) if citations else 1

    expected_types = set(_normalized_terms(case.get("expected_evidence_types", [])))
    retrieved_types = tuple(
        sorted(
            {
                _normalize(source.get("evidence_type", "unknown")) or "unknown"
                for source in sources
            }
        )
    )
    type_coverage = (
        len(expected_types.intersection(retrieved_types)) / len(expected_types)
        if expected_types
        else 1.0
    )
    query_context = result.get("query_context", {})
    if not isinstance(query_context, Mapping):
        query_context = {}
    actual_question_type = str(query_context.get("question_type", "")).strip()
    expected_question_type = str(case.get("expected_question_type", "")).strip()
    pico = query_context.get("pico", {})
    query_understanding_present = bool(
        actual_question_type
        and isinstance(pico, Mapping)
        and any(pico.get(key) for key in ("population", "intervention", "comparator", "outcome"))
    )
    question_type_match = (
        actual_question_type == expected_question_type
        if expected_question_type
        else None
    )
    located_citations = sum(
        bool(source_by_number.get(number, {}).get("page_ranges"))
        for number in citations
        if number in source_by_number
    )
    citation_locator_coverage = (
        located_citations / len(citations) if citations else 0.0
    )
    return CaseAssessment(
        case_id=str(case.get("id", "")),
        expected_source_hit=expected_source_hit,
        duplicate_document_count=duplicate_document_count,
        unsupported_citation_count=unsupported_count,
        all_citations_supported=bool(citations) and not unsupported,
        evidence_type_coverage=round(type_coverage, 4),
        retrieval_mode=str(result.get("mode", "unknown")),
        citation_count=len(citations),
        retrieved_evidence_types=retrieved_types,
        query_understanding_present=query_understanding_present,
        question_type_match=question_type_match,
        citation_locator_coverage=round(citation_locator_coverage, 4),
    )


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Validation fixture must be a non-empty JSON array")
    cases: list[dict[str, Any]] = []
    ids: set[str] = set()
    for raw_case in payload:
        if not isinstance(raw_case, Mapping):
            raise ValueError("Every validation case must be an object")
        case = dict(raw_case)
        case_id = str(case.get("id", "")).strip()
        question = str(case.get("question", "")).strip()
        if not case_id or not question:
            raise ValueError("Every validation case needs id and question")
        if case_id in ids:
            raise ValueError(f"Duplicate validation id: {case_id}")
        ids.add(case_id)
        cases.append(case)
    return cases


def run_cases(
    cases: Iterable[Mapping[str, Any]], service: Any
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["id"])
        try:
            filters = _filters_from_case(case.get("filters", {}))
            result = service.query(str(case["question"]), filters)
            assessment = evaluate_case(case, result)
            records.append(
                {
                    "id": case_id,
                    "question": str(case["question"]),
                    "status": "completed",
                    "filters": _filters_payload(filters),
                    "assessment": assessment.as_dict(),
                    "result": result,
                }
            )
        except KnowledgeBaseError as error:
            records.append(
                {
                    "id": case_id,
                    "question": str(case["question"]),
                    "status": "failed",
                    "error": {
                        "code": error.code,
                        "message": error.message,
                        "details": error.details,
                    },
                }
            )
        except Exception as error:
            records.append(
                {
                    "id": case_id,
                    "question": str(case["question"]),
                    "status": "failed",
                    "error": {
                        "code": "validation_case_failed",
                        "message": str(error),
                    },
                }
            )
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "baseline_only": True,
        "target_threshold": None,
        "aggregate": _aggregate(records),
        "cases": records,
    }


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [
        record for record in records if record.get("status") == "completed"
    ]
    assessments = [record["assessment"] for record in completed]
    count = len(assessments)
    mode_counts = Counter(item["retrieval_mode"] for item in assessments)
    typed_assessments = [
        item for item in assessments if item["question_type_match"] is not None
    ]
    return {
        "case_count": len(records),
        "completed_count": count,
        "failed_count": len(records) - count,
        "expected_source_hit_count": sum(
            bool(item["expected_source_hit"]) for item in assessments
        ),
        "expected_source_hit_rate": round(
            sum(bool(item["expected_source_hit"]) for item in assessments) / count, 4
        )
        if count
        else None,
        "duplicate_document_count": sum(
            int(item["duplicate_document_count"]) for item in assessments
        ),
        "unsupported_citation_count": sum(
            int(item["unsupported_citation_count"]) for item in assessments
        ),
        "all_citations_supported_count": sum(
            bool(item["all_citations_supported"]) for item in assessments
        ),
        "mean_evidence_type_coverage": round(
            sum(float(item["evidence_type_coverage"]) for item in assessments) / count,
            4,
        )
        if count
        else None,
        "query_understanding_count": sum(
            bool(item["query_understanding_present"]) for item in assessments
        ),
        "question_type_match_rate": round(
            sum(bool(item["question_type_match"]) for item in typed_assessments)
            / len(typed_assessments),
            4,
        )
        if typed_assessments
        else None,
        "mean_citation_locator_coverage": round(
            sum(float(item["citation_locator_coverage"]) for item in assessments) / count,
            4,
        )
        if count
        else None,
        "retrieval_modes": dict(sorted(mode_counts.items())),
    }


def _filters_from_case(raw: object) -> SearchFilters:
    values = dict(raw) if isinstance(raw, Mapping) else {}
    evidence_types = values.get("evidence_types", [])
    return SearchFilters(
        evidence_types=frozenset(str(value) for value in evidence_types),
        year_from=_optional_int(values.get("year_from")),
        year_to=_optional_int(values.get("year_to")),
        fulltext_only=bool(values.get("fulltext_only", False)),
    )


def _filters_payload(filters: SearchFilters) -> dict[str, Any]:
    return {
        "evidence_types": sorted(filters.evidence_types),
        "year_from": filters.year_from,
        "year_to": filters.year_to,
        "fulltext_only": filters.fulltext_only,
    }


def _source_number(source: Mapping[str, Any], fallback: int) -> int:
    value = source.get("source_number", fallback)
    return value if isinstance(value, int) and not isinstance(value, bool) else fallback


def _document_key(source: Mapping[str, Any]) -> str:
    document_id = _normalize(source.get("document_id", ""))
    doi = _normalize_doi(source.get("doi", ""))
    title = _normalize(source.get("title", ""))
    return document_id or doi or title


def _source_has_evidence_text(source: Mapping[str, Any]) -> bool:
    snippets = source.get("snippets", [])
    if isinstance(snippets, (list, tuple)) and any(
        isinstance(value, str) and value.strip() for value in snippets
    ):
        return True
    return bool(str(source.get("abstract", "")).strip())


def _normalized_terms(values: object) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(_normalize(value) for value in values if _normalize(value))


def _normalize(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _normalize_doi(value: object) -> str:
    normalized = _normalize(value)
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
    return normalized


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).with_name("mpp_questions.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / ".local" / "validation",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--api-url",
        help="Use a running web service, for example http://127.0.0.1:8765.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = load_cases(args.questions)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        cases = cases[: args.limit]
    if args.api_url:
        service = HttpValidationService(args.api_url)
    else:
        settings = Settings.from_mapping(os.environ, project_root=PROJECT_ROOT)
        service = create_production_service(settings)
    report = run_cases(cases, service)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    output_path = args.output_dir / f"mpp-validation-{stamp}.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output_path), **report["aggregate"]}, ensure_ascii=False))
    return 0 if report["aggregate"]["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
