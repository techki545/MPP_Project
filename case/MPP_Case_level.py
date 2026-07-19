import argparse
import json
import os
from typing import Any, Dict, List

try:
    from .MycoplasmaPneumonia.tools import (
        assess_refractory_risk,
        assess_suspected_mpp,
        build_final_report,
        normalize_patient_case,
        recommend_management,
        retrieve_evidence,
        stratify_severity,
    )
except ImportError:
    from MycoplasmaPneumonia.tools import (
    assess_refractory_risk,
    assess_suspected_mpp,
    build_final_report,
    normalize_patient_case,
    recommend_management,
    retrieve_evidence,
    stratify_severity,
)


DATA_ROOT = "MycoplasmaPneumonia"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "MycoplasmaPneumonia")

TOOL_FN_REGISTRY = {
    "normalize_patient_case": normalize_patient_case,
    "assess_suspected_mpp": assess_suspected_mpp,
    "stratify_severity": stratify_severity,
    "assess_refractory_risk": assess_refractory_risk,
    "retrieve_evidence": retrieve_evidence,
    "recommend_management": recommend_management,
    "build_final_report": build_final_report,
}


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _merge_json(path: str, field: str, value: Any) -> None:
    if os.path.exists(path):
        data = _read_json(path)
    else:
        data = {}
    data[field] = value
    _write_json(path, data)


def _command_to_fn_name(command: str) -> str:
    command = (command or "").strip()
    return command.split("(", 1)[0].strip() if "(" in command else command


def _ensure_plan(data_root: str) -> List[Dict[str, Any]]:
    plan_path = os.path.join(data_root, "plan.json")
    if not os.path.exists(plan_path):
        try:
            from disease.MPP_Task_level import generate_plan
        except ImportError:
            import sys

            sys.path.insert(0, os.path.dirname(BASE_DIR))
            from disease.MPP_Task_level import generate_plan

        generate_plan(data_root)
    return _read_json(plan_path)


def _default_output_dir(data_root: str, case: Dict[str, Any]) -> str:
    case_id = str(case.get("case_id", "unknown_case")).replace(os.sep, "_")
    return os.path.join(data_root, "record", case_id)


def run_case(case_file: str, data_root: str = DATA_ROOT, output_dir: str = "") -> Dict[str, Any]:
    """Execute the patient-level evidence reasoning workflow for one pediatric MPP case."""
    task = _read_json(os.path.join(data_root, "task.json"))
    toolset = _read_json(os.path.join(data_root, "toolset.json"))
    plan = _ensure_plan(data_root)
    case = _read_json(case_file)
    tool_by_id = {int(tool["id"]): tool for tool in toolset}

    save_dir = output_dir or _default_output_dir(data_root, case)
    os.makedirs(save_dir, exist_ok=True)

    results: Dict[int, Any] = {}
    run_record: Dict[str, Any] = {
        "task": task,
        "case_file": case_file,
        "steps": {},
    }

    for step in plan:
        step_id = int(step["id"])
        tool_ids = step.get("tool") or []
        if not isinstance(tool_ids, list):
            tool_ids = [tool_ids]
        if len(tool_ids) != 1:
            raise ValueError(f"Step {step_id} must reference exactly one tool in this prototype.")

        tool = tool_by_id[int(tool_ids[0])]
        fn_name = _command_to_fn_name(tool.get("command", ""))
        fn = TOOL_FN_REGISTRY.get(fn_name)
        if fn is None:
            raise KeyError(f"Tool function '{fn_name}' is not registered.")

        resolved = []
        for dep in step.get("input_type", []):
            dep = int(dep)
            resolved.append(case if dep == 0 else results[dep])

        result = fn(resolved, save_dir, step.get("output_path", "diagnosis.json"))
        results[step_id] = result

        step_payload = {
            "id": step_id,
            "action": step.get("action", ""),
            "tool": fn_name,
            "output_type": step.get("output_type", ""),
            "result": result,
        }
        run_record["steps"][f"step_{step_id}"] = step_payload

        output_file = os.path.join(save_dir, step.get("output_path", "diagnosis.json"))
        if step.get("output_type") == "final report":
            _write_json(output_file, result)
        else:
            _merge_json(output_file, f"step_{step_id}", step_payload)

    _write_json(os.path.join(save_dir, "run_record.json"), run_record)
    return {
        "output_dir": save_dir,
        "diagnosis_file": os.path.join(save_dir, "diagnosis.json"),
        "final_file": os.path.join(save_dir, "final_recommendation.json"),
        "final_report": results[max(results.keys())],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pediatric MPP evidence-based recommendation workflow.")
    parser.add_argument("--case", default=os.path.join(DATA_ROOT, "sample_case.json"), help="Path to case JSON.")
    parser.add_argument("--data-root", default=DATA_ROOT, help="Path to MPP task package.")
    parser.add_argument("--output-dir", default="", help="Optional output directory.")
    args = parser.parse_args()

    result = run_case(args.case, data_root=args.data_root, output_dir=args.output_dir)
    final = result["final_report"]
    print(json.dumps(
        {
            "output_dir": result["output_dir"],
            "final_file": result["final_file"],
            "case_id": final.get("case_id"),
            "suspected_mpp": final.get("diagnosis", {}).get("suspected_mpp"),
            "severity": final.get("risk", {}).get("severity"),
            "refractory_risk": final.get("risk", {}).get("refractory_risk"),
            "warnings": final.get("warnings", []),
            "llm_used": final.get("llm_used", False),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
