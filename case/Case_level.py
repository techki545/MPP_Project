import argparse
import json
import os
from typing import Any, Dict

try:
    from .MPP_Case_level import DATA_ROOT, run_case
except ImportError:
    from MPP_Case_level import DATA_ROOT, run_case


def run_case_level(case_file: str, data_root: str = DATA_ROOT, output_dir: str = "") -> Dict[str, Any]:
    """Run patient-level individualized evidence reasoning."""
    return run_case(case_file=case_file, data_root=data_root, output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pediatric MPP patient-level evidence recommendation.")
    parser.add_argument("--case", default=os.path.join(DATA_ROOT, "sample_case.json"), help="Path to case JSON.")
    parser.add_argument("--data-root", default=DATA_ROOT, help="Task package root.")
    parser.add_argument("--output-dir", default="", help="Optional output directory.")
    args = parser.parse_args()

    result = run_case_level(args.case, data_root=args.data_root, output_dir=args.output_dir)
    final = result["final_report"]
    print(
        json.dumps(
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
        )
    )


if __name__ == "__main__":
    main()
