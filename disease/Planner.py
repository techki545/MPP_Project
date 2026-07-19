import json
import os
from typing import Any, Dict, List, Optional


STANDARD_MPP_PLAN: List[Dict[str, Any]] = [
    {
        "id": 1,
        "layer": "data",
        "tool": [1],
        "action_type": "quantitative",
        "action": "Normalize multimodal pediatric MPP case data",
        "input_type": [0],
        "output_type": "standardized data",
        "output_path": "diagnosis.json",
        "proposal_task": "multimodal data cleaning, de-identification, coding, and standardization",
    },
    {
        "id": 2,
        "layer": "model",
        "tool": [2],
        "action_type": "qualitative",
        "action": "Assess suspected Mycoplasma pneumoniae pneumonia",
        "input_type": [1],
        "output_type": "diagnostic indicator",
        "output_path": "diagnosis.json",
        "proposal_task": "suspected case identification",
    },
    {
        "id": 3,
        "layer": "model",
        "tool": [3],
        "action_type": "quantitative",
        "action": "Stratify disease severity",
        "input_type": [1],
        "output_type": "risk indicator",
        "output_path": "diagnosis.json",
        "proposal_task": "severity stratification",
    },
    {
        "id": 4,
        "layer": "model",
        "tool": [4],
        "action_type": "quantitative",
        "action": "Assess refractory or severe progression risk",
        "input_type": [1],
        "output_type": "risk indicator",
        "output_path": "diagnosis.json",
        "proposal_task": "severe and refractory risk assessment",
    },
    {
        "id": 5,
        "layer": "knowledge",
        "tool": [5],
        "action_type": "qualitative",
        "action": "Retrieve traceable evidence from the MPP knowledge base",
        "input_type": [2, 3, 4],
        "output_type": "evidence set",
        "output_path": "diagnosis.json",
        "proposal_task": "RAG evidence retrieval and traceable knowledge base use",
    },
    {
        "id": 6,
        "layer": "application",
        "tool": [6],
        "action_type": "qualitative",
        "action": "Generate clinician-review management recommendation",
        "input_type": [1, 2, 3, 4, 5],
        "output_type": "management recommendation",
        "output_path": "diagnosis.json",
        "proposal_task": "treatment reference, efficacy review, and follow-up warning",
    },
    {
        "id": 7,
        "layer": "application",
        "tool": [7],
        "action_type": "qualitative",
        "action": "Build final evidence-based recommendation report",
        "input_type": [1, 2, 3, 4, 5, 6],
        "output_type": "final report",
        "output_path": "final_recommendation.json",
        "proposal_task": "interpretable and evidence-traceable output",
    },
]


class Planner:
    """Disease-level pathway planner for the pediatric MPP proposal."""

    def __init__(self, api_key: str = "", disease: str = "pediatric_mpp"):
        self.api_key = api_key
        self.disease = disease

    def _allowed_tool_ids(self, toolset: Optional[List[Dict[str, Any]]]) -> set[int]:
        if not toolset:
            return set()
        return {int(item["id"]) for item in toolset if "id" in item}

    def _validate_plan(self, plan: List[Dict[str, Any]], toolset: Optional[List[Dict[str, Any]]] = None) -> None:
        required = {
            "id",
            "layer",
            "tool",
            "action_type",
            "action",
            "input_type",
            "output_type",
            "output_path",
            "proposal_task",
        }
        allowed_tools = self._allowed_tool_ids(toolset)
        seen = {0}

        for index, step in enumerate(plan, start=1):
            missing = required - set(step)
            if missing:
                raise ValueError(f"Plan step {index} missing fields: {sorted(missing)}")
            if int(step["id"]) != index:
                raise ValueError("Plan IDs must be consecutive and start from 1.")
            if step["action_type"] not in {"quantitative", "qualitative"}:
                raise ValueError(f"Invalid action_type in step {index}: {step['action_type']}")
            if not isinstance(step["tool"], list) or not step["tool"]:
                raise ValueError(f"Step {index} must contain non-empty tool list.")
            if allowed_tools:
                for tool_id in step["tool"]:
                    if int(tool_id) not in allowed_tools:
                        raise ValueError(f"Step {index} references unknown tool id {tool_id}.")
            for dep in step["input_type"]:
                if int(dep) not in seen:
                    raise ValueError(f"Step {index} has unresolved dependency {dep}.")
            seen.add(index)

    def build_standard_pathway(self, toolset: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        plan = [dict(step) for step in STANDARD_MPP_PLAN]
        self._validate_plan(plan, toolset=toolset)
        return plan

    def plan(
        self,
        output_path: str,
        prompt: str = "",
        rag_text: str = "",
        filename: str = "plan.json",
        model: str = "",
        toolset: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Write the proposal-aligned disease pathway to disk.

        The original project used an LLM planner. For this proposal prototype the
        disease-level pathway is deterministic so it is reproducible in reports
        and software-copyright materials.
        """
        os.makedirs(output_path, exist_ok=True)
        plan = self.build_standard_pathway(toolset=toolset)
        out_file = os.path.join(output_path, filename)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        return plan

