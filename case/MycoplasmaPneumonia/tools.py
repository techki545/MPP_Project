import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .llm_client import get_llm_client


PACKAGE_DIR = os.path.dirname(__file__)
KNOWLEDGE_BASE_PATH = os.path.join(PACKAGE_DIR, "knowledge_base.json")


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "positive", "present"}
    return False


def _first_input(inputs: List[Any]) -> Any:
    if not inputs:
        raise ValueError("Tool input list is empty.")
    return inputs[0]


def _load_case(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        return _read_json(value)
    if isinstance(value, dict) and "case" in value and isinstance(value["case"], dict):
        return value["case"]
    if isinstance(value, dict):
        return value
    raise TypeError(f"Unsupported case input type: {type(value)!r}")


def _find_dict_with_key(inputs: Iterable[Any], key: str) -> Dict[str, Any]:
    for item in inputs:
        if isinstance(item, dict) and key in item:
            return item
    return {}


def _text_has_any(text: str, patterns: Iterable[str]) -> bool:
    low = (text or "").lower()
    return any(pattern in low for pattern in patterns)


def _text_has_positive_finding(text: str, patterns: Iterable[str]) -> bool:
    low = (text or "").lower()
    for pattern in patterns:
        if pattern not in low:
            continue
        negated = re.search(rf"\b(no|without|absent|negative for)\b[^.]*\b{re.escape(pattern)}\b", low)
        if not negated:
            return True
    return False


def _normalise_imaging_features(imaging: Dict[str, Any]) -> Dict[str, bool]:
    report = str(imaging.get("radiology_report", ""))
    raw_features = imaging.get("image_features") or {}

    features = {
        "consolidation": _as_bool(raw_features.get("consolidation"))
        or _text_has_positive_finding(report, ["consolidation", "infiltration", "opacity"]),
        "multilobar": _as_bool(raw_features.get("multilobar"))
        or _text_has_positive_finding(report, ["multilobar", "bilateral", "multiple lobes"]),
        "pleural_effusion": _as_bool(raw_features.get("pleural_effusion"))
        or _text_has_positive_finding(report, ["pleural effusion", "effusion"]),
        "atelectasis": _as_bool(raw_features.get("atelectasis"))
        or _text_has_positive_finding(report, ["atelectasis"]),
        "necrotizing_pneumonia": _as_bool(raw_features.get("necrotizing_pneumonia"))
        or _text_has_positive_finding(report, ["necrotizing", "necrosis", "cavitation"]),
        "rapid_progression": _as_bool(raw_features.get("rapid_progression"))
        or _text_has_positive_finding(report, ["rapid progression", "progressed", "worsening"]),
    }
    return features


def _collect_modalities(case: Dict[str, Any]) -> List[str]:
    modalities = []
    for key in ["symptoms", "vitals", "labs", "pathogen", "imaging", "treatment_history", "follow_up"]:
        if case.get(key):
            modalities.append(key)
    return modalities


def _load_kb() -> List[Dict[str, Any]]:
    return _read_json(KNOWLEDGE_BASE_PATH)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _level(score: float, high: float, moderate: float, labels) -> str:
    """Map a numeric score to high, moderate, or low labels."""
    if score >= high:
        return labels[0]
    if score >= moderate:
        return labels[1]
    return labels[2]


def normalize_patient_case(inputs: List[Any], save_dir: str = "", save_name: str = "") -> Dict[str, Any]:
    """Normalize raw pediatric MPP case data into a stable multimodal schema."""
    raw = _load_case(_first_input(inputs))
    symptoms = raw.get("symptoms") or {}
    vitals = raw.get("vitals") or {}
    labs = raw.get("labs") or {}
    pathogen = raw.get("pathogen") or {}
    imaging = raw.get("imaging") or {}
    treatment = raw.get("treatment_history") or {}
    follow_up = raw.get("follow_up") or {}

    case = {
        "case_id": str(raw.get("case_id", "unknown_case")),
        "age_years": _as_float(raw.get("age_years")),
        "sex": raw.get("sex", ""),
        "symptoms": {
            "fever_days": _as_float(symptoms.get("fever_days"), 0),
            "max_temperature_c": _as_float(symptoms.get("max_temperature_c")),
            "cough": _as_bool(symptoms.get("cough")),
            "wheezing": _as_bool(symptoms.get("wheezing")),
            "dyspnea": _as_bool(symptoms.get("dyspnea")),
            "chest_pain": _as_bool(symptoms.get("chest_pain")),
        },
        "vitals": {
            "spo2": _as_float(vitals.get("spo2")),
            "respiratory_distress": _as_bool(vitals.get("respiratory_distress")),
            "heart_rate": _as_float(vitals.get("heart_rate")),
            "respiratory_rate": _as_float(vitals.get("respiratory_rate")),
        },
        "labs": {
            "wbc_10e9_l": _as_float(labs.get("wbc_10e9_l")),
            "neutrophil_ratio": _as_float(labs.get("neutrophil_ratio")),
            "crp_mg_l": _as_float(labs.get("crp_mg_l")),
            "pct_ng_ml": _as_float(labs.get("pct_ng_ml")),
            "ldh_u_l": _as_float(labs.get("ldh_u_l")),
        },
        "pathogen": {
            "mycoplasma_pcr": str(pathogen.get("mycoplasma_pcr", "")).lower(),
            "mycoplasma_igm": str(pathogen.get("mycoplasma_igm", "")).lower(),
            "macrolide_resistance": str(pathogen.get("macrolide_resistance", "unknown")).lower(),
        },
        "imaging": {
            "image_path": imaging.get("image_path", ""),
            "radiology_report": imaging.get("radiology_report", ""),
            "image_features": _normalise_imaging_features(imaging),
        },
        "treatment_history": {
            "macrolide_days": _as_float(treatment.get("macrolide_days"), 0),
            "response": str(treatment.get("response", "")).lower(),
            "persistent_fever_after_treatment": _as_bool(treatment.get("persistent_fever_after_treatment")),
        },
        "follow_up": {
            "poor_oral_intake": _as_bool(follow_up.get("poor_oral_intake")),
            "extrapulmonary_symptoms": _as_bool(follow_up.get("extrapulmonary_symptoms")),
        },
    }

    missing = []
    for section in ["symptoms", "vitals", "labs", "pathogen", "imaging"]:
        if not raw.get(section):
            missing.append(section)

    return {
        "case": case,
        "modalities": _collect_modalities(raw),
        "missing_sections": missing,
        "note": "Standardized multimodal case object for evidence-based MPP reasoning.",
    }


def assess_suspected_mpp(inputs: List[Any], save_dir: str = "", save_name: str = "") -> Dict[str, Any]:
    """Assess whether the standardized case supports suspected pediatric MPP."""
    case = _load_case(_first_input(inputs))
    symptoms = case["symptoms"]
    pathogen = case["pathogen"]
    imaging_features = case["imaging"]["image_features"]

    score = 0.0
    signals = []

    if symptoms["fever_days"] and symptoms["fever_days"] >= 3:
        score += 0.15
        signals.append("persistent fever")
    if symptoms["cough"]:
        score += 0.15
        signals.append("cough")
    if pathogen["mycoplasma_pcr"] in {"positive", "+", "detected"}:
        score += 0.35
        signals.append("Mycoplasma PCR positive")
    if pathogen["mycoplasma_igm"] in {"positive", "+", "detected"}:
        score += 0.2
        signals.append("Mycoplasma IgM positive")
    if any(imaging_features.values()):
        score += 0.15
        signals.append("compatible radiologic pneumonia burden")

    score = min(score, 1.0)
    level = _level(score, high=0.7, moderate=0.4, labels=("high", "moderate", "low"))
    return {
        "indicator": "suspected_mpp",
        "suspicion_level": level,
        "score": round(score, 3),
        "signals": signals,
        "evidence_ids": ["MPP-DX-001", "MPP-IMG-001"],
    }


def stratify_severity(inputs: List[Any], save_dir: str = "", save_name: str = "") -> Dict[str, Any]:
    """Classify disease severity using clinical, laboratory, and imaging signals."""
    case = _load_case(_first_input(inputs))
    vitals = case["vitals"]
    labs = case["labs"]
    imaging_features = case["imaging"]["image_features"]
    follow_up = case["follow_up"]
    symptoms = case["symptoms"]

    points = 0
    rationale = []

    spo2 = vitals.get("spo2")
    if spo2 is not None and spo2 < 92:
        points += 3
        rationale.append("SpO2 < 92")
    elif spo2 is not None and spo2 < 95:
        points += 2
        rationale.append("SpO2 92-94")
    if vitals.get("respiratory_distress") or symptoms.get("dyspnea"):
        points += 2
        rationale.append("respiratory distress or dyspnea")
    if imaging_features.get("multilobar"):
        points += 2
        rationale.append("multilobar lung involvement")
    if imaging_features.get("pleural_effusion"):
        points += 1
        rationale.append("pleural effusion")
    if imaging_features.get("atelectasis"):
        points += 1
        rationale.append("atelectasis")
    if imaging_features.get("necrotizing_pneumonia"):
        points += 3
        rationale.append("necrotizing pneumonia")
    if labs.get("crp_mg_l") is not None and labs["crp_mg_l"] >= 40:
        points += 1
        rationale.append("CRP >= 40 mg/L")
    if labs.get("pct_ng_ml") is not None and labs["pct_ng_ml"] >= 0.5:
        points += 1
        rationale.append("PCT >= 0.5 ng/mL")
    if follow_up.get("extrapulmonary_symptoms"):
        points += 2
        rationale.append("extrapulmonary symptoms")

    if points >= 5:
        severity = "severe"
    elif points >= 2:
        severity = "moderate"
    else:
        severity = "mild"

    return {
        "indicator": "severity",
        "severity": severity,
        "points": points,
        "rationale": rationale,
        "evidence_ids": ["MPP-SEV-001", "MPP-IMG-001"],
    }


def assess_refractory_risk(inputs: List[Any], save_dir: str = "", save_name: str = "") -> Dict[str, Any]:
    """Estimate refractory or severe progression risk for clinician review."""
    case = _load_case(_first_input(inputs))
    symptoms = case["symptoms"]
    labs = case["labs"]
    imaging_features = case["imaging"]["image_features"]
    treatment = case["treatment_history"]

    points = 0
    rationale = []

    if symptoms.get("fever_days") and symptoms["fever_days"] >= 7:
        points += 2
        rationale.append("fever duration >= 7 days")
    if treatment.get("macrolide_days", 0) >= 3 and treatment.get("response") in {"poor", "none", "worse"}:
        points += 3
        rationale.append("poor response after at least 3 days of macrolide therapy")
    if treatment.get("persistent_fever_after_treatment"):
        points += 2
        rationale.append("persistent fever after treatment")
    if labs.get("crp_mg_l") is not None and labs["crp_mg_l"] >= 40:
        points += 1
        rationale.append("elevated CRP")
    if labs.get("ldh_u_l") is not None and labs["ldh_u_l"] >= 350:
        points += 1
        rationale.append("elevated LDH")
    if imaging_features.get("pleural_effusion") or imaging_features.get("multilobar"):
        points += 1
        rationale.append("significant radiologic involvement")

    if points >= 6:
        risk = "high"
    elif points >= 3:
        risk = "moderate"
    else:
        risk = "low"

    return {
        "indicator": "refractory_or_progression_risk",
        "refractory_risk": risk,
        "points": points,
        "rationale": rationale,
        "evidence_ids": ["MPP-RISK-001"],
    }


def retrieve_evidence(inputs: List[Any], save_dir: str = "", save_name: str = "") -> Dict[str, Any]:
    """Retrieve traceable evidence snippets using lightweight lexical scoring."""
    query_text = json.dumps(inputs, ensure_ascii=False)
    query_tokens = set(_tokenize(query_text))
    kb = _load_kb()

    scored = []
    for entry in kb:
        haystack = " ".join(
            [
                entry.get("topic", ""),
                " ".join(entry.get("tags", [])),
                entry.get("statement", ""),
                entry.get("clinical_use", ""),
            ]
        )
        tokens = set(_tokenize(haystack))
        score = len(query_tokens & tokens)
        if score == 0 and entry.get("topic") in {"diagnosis", "severity", "treatment", "follow_up"}:
            score = 1
        scored.append((score, entry))

    ranked = [entry for score, entry in sorted(scored, key=lambda item: item[0], reverse=True) if score > 0]
    return {
        "retriever": "local_lexical_rag",
        "evidence": ranked[:5],
        "evidence_ids": [entry["id"] for entry in ranked[:5]],
    }


def recommend_management(inputs: List[Any], save_dir: str = "", save_name: str = "") -> Dict[str, Any]:
    """Generate evidence-linked management recommendations for clinician review."""
    normalized = _find_dict_with_key(inputs, "case")
    case = normalized.get("case", _load_case(_first_input(inputs)))
    diagnosis = _find_dict_with_key(inputs, "suspicion_level")
    severity = _find_dict_with_key(inputs, "severity")
    refractory = _find_dict_with_key(inputs, "refractory_risk")
    evidence = _find_dict_with_key(inputs, "evidence")

    items = []
    warnings = []

    suspicion_level = diagnosis.get("suspicion_level", "unknown")
    severity_level = severity.get("severity", "unknown")
    refractory_level = refractory.get("refractory_risk", "unknown")
    age = case.get("age_years")

    if suspicion_level in {"high", "moderate"}:
        items.append(
            {
                "domain": "diagnosis",
                "recommendation": "Treat the case as suspected pediatric MPP and review pathogen results, imaging, and differential diagnoses together.",
                "evidence_ids": ["MPP-DX-001", "MPP-IMG-001"],
            }
        )
    else:
        items.append(
            {
                "domain": "diagnosis",
                "recommendation": "MPP evidence is weak; prioritize differential diagnosis and repeat pathogen evaluation if clinically indicated.",
                "evidence_ids": ["MPP-DX-001"],
            }
        )

    if severity_level == "severe":
        items.append(
            {
                "domain": "severity",
                "recommendation": "Recommend urgent clinician review for severe pneumonia features, oxygenation monitoring, complication assessment, and escalation planning.",
                "evidence_ids": ["MPP-SEV-001"],
            }
        )
        warnings.append("severe disease signals present")
    elif severity_level == "moderate":
        items.append(
            {
                "domain": "severity",
                "recommendation": "Recommend close monitoring and reassessment of respiratory status, inflammatory markers, and imaging evolution.",
                "evidence_ids": ["MPP-SEV-001"],
            }
        )

    if refractory_level in {"high", "moderate"}:
        alternative_note = "review age-limited alternative antimicrobial options with a pediatric specialist"
        if age is not None and age >= 8:
            alternative_note = "clinician may review age-appropriate non-macrolide options under local guidance"
        items.append(
            {
                "domain": "treatment_review",
                "recommendation": f"Poor response or refractory-risk signals are present; {alternative_note}. Do not auto-prescribe.",
                "evidence_ids": ["MPP-RISK-001", "MPP-TX-001"],
            }
        )
        warnings.append("refractory or severe progression risk")
    else:
        items.append(
            {
                "domain": "treatment_review",
                "recommendation": "Continue guideline-concordant therapy review and monitor treatment response.",
                "evidence_ids": ["MPP-TX-001"],
            }
        )

    items.append(
        {
            "domain": "follow_up",
            "recommendation": "Follow up for persistent fever, worsening cough or dyspnea, falling oxygen saturation, poor oral intake, extrapulmonary symptoms, or delayed imaging recovery.",
            "evidence_ids": ["MPP-FU-001"],
        }
    )

    result = {
        "case_id": case.get("case_id", "unknown_case"),
        "recommendation_items": items,
        "warnings": warnings,
        "retrieved_evidence_ids": evidence.get("evidence_ids", []),
        "safety_note": "Clinical decision-support output for physician review only. It is not a diagnosis or prescription.",
        "llm_used": False,
    }
    llm = get_llm_client()
    if not llm.enabled:
        return result

    system_prompt = (
        "You are a pediatric infectious disease clinical decision-support assistant. "
        "Generate cautious, evidence-linked recommendations for clinician review only. "
        "Do not prescribe automatically. Return only valid JSON with keys: "
        "recommendation_items, warnings, safety_note, llm_summary."
    )
    payload = {
        "case": case,
        "rule_outputs": {
            "diagnosis": diagnosis,
            "severity": severity,
            "refractory": refractory,
            "draft_recommendation": result,
        },
        "evidence": evidence.get("evidence", []),
        "required_constraints": [
            "Keep each recommendation linked to evidence_ids.",
            "Mention clinician review for antimicrobial changes.",
            "Include follow-up warning signals.",
            "Do not claim definitive diagnosis or prescription authority.",
        ],
    }
    try:
        generated = llm.chat_json(system_prompt, payload)
    except Exception as exc:
        result["llm_error"] = str(exc)
        return result

    result["llm_used"] = True
    result["recommendation_items"] = generated.get("recommendation_items", result["recommendation_items"])
    result["warnings"] = generated.get("warnings", result["warnings"])
    result["safety_note"] = generated.get("safety_note", result["safety_note"])
    result["llm_summary"] = generated.get("llm_summary", "")
    return result


def build_final_report(inputs: List[Any], save_dir: str = "", save_name: str = "") -> Dict[str, Any]:
    """Build the final evidence-traceable pediatric MPP recommendation report."""
    normalized = _find_dict_with_key(inputs, "case")
    diagnosis = _find_dict_with_key(inputs, "suspicion_level")
    severity = _find_dict_with_key(inputs, "severity")
    refractory = _find_dict_with_key(inputs, "refractory_risk")
    evidence = _find_dict_with_key(inputs, "evidence")
    recommendation = _find_dict_with_key(inputs, "recommendation_items")
    case = normalized.get("case", {})

    evidence_map = {entry["id"]: entry for entry in _load_kb()}
    used_ids = set()
    for result in [diagnosis, severity, refractory, recommendation]:
        for evidence_id in result.get("evidence_ids", []):
            used_ids.add(evidence_id)
        for item in result.get("recommendation_items", []):
            for evidence_id in item.get("evidence_ids", []):
                used_ids.add(evidence_id)

    report = {
        "case_id": case.get("case_id", "unknown_case"),
        "diagnosis": {
            "suspected_mpp": diagnosis.get("suspicion_level", "unknown"),
            "score": diagnosis.get("score"),
            "signals": diagnosis.get("signals", []),
        },
        "risk": {
            "severity": severity.get("severity", "unknown"),
            "severity_rationale": severity.get("rationale", []),
            "refractory_risk": refractory.get("refractory_risk", "unknown"),
            "refractory_rationale": refractory.get("rationale", []),
        },
        "recommendations": recommendation.get("recommendation_items", []),
        "warnings": recommendation.get("warnings", []),
        "evidence_trace": [
            {
                "id": evidence_id,
                "topic": evidence_map.get(evidence_id, {}).get("topic", "not_retrieved"),
                "statement": evidence_map.get(evidence_id, {}).get("statement", ""),
            }
            for evidence_id in sorted(used_ids)
        ],
        "limitations": [
            "The structured indicators are generated by rule-based reasoning and local lexical evidence retrieval.",
            "When enabled, the large model is used to refine recommendation wording and evidence synthesis, not to replace clinician judgment.",
            "It requires clinician validation and does not replace medical judgment.",
            "Missing or low-quality case data may reduce recommendation reliability.",
        ],
        "llm_used": bool(recommendation.get("llm_used")),
    }
    llm = get_llm_client()
    if not llm.enabled:
        return report

    system_prompt = (
        "You are preparing an evidence-based pediatric MPP clinical decision-support report. "
        "Use the structured rule outputs as fixed facts. Refine the final report narrative and "
        "quality notes, but do not change numeric scores, severity labels, or refractory-risk labels. "
        "Return only valid JSON with keys: executive_summary, clinician_review_points, data_quality_notes."
    )
    payload = {
        "fixed_report": report,
        "evidence_trace": report["evidence_trace"],
        "constraints": [
            "Do not change diagnosis, severity, refractory_risk, score, or evidence IDs.",
            "Use cautious clinical language.",
            "State that the output is not a prescription.",
        ],
    }
    try:
        generated = llm.chat_json(system_prompt, payload)
    except Exception as exc:
        report["llm_error"] = str(exc)
        return report

    report["llm_used"] = True
    report["llm_synthesis"] = {
        "executive_summary": generated.get("executive_summary", ""),
        "clinician_review_points": generated.get("clinician_review_points", []),
        "data_quality_notes": generated.get("data_quality_notes", []),
    }
    return report
