import json
import os

try:
    from .Planner import Planner
    from .RAG import RAG_Module
except ImportError:
    from Planner import Planner
    from RAG import RAG_Module


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_ROOT = os.path.join(PROJECT_DIR, "case", "MycoplasmaPneumonia")


def generate_task_level_plan(data_root: str = DATA_ROOT):
    """Generate the disease-level pathway required by the proposal."""
    task_file = os.path.join(data_root, "task.json")
    tool_file = os.path.join(data_root, "toolset.json")

    with open(task_file, "r", encoding="utf-8") as f:
        task = json.load(f)
    with open(tool_file, "r", encoding="utf-8") as f:
        toolset = json.load(f)

    rag = RAG_Module(knowledge_base_path=os.path.join(data_root, "knowledge_base.json"))
    context = rag.query(
        "pediatric Mycoplasma pneumoniae pneumonia diagnosis severity refractory treatment follow-up"
    )

    planner_prompt = (
        "Build a disease-level standardized pathway for pediatric Mycoplasma pneumoniae pneumonia. "
        "The pathway must cover multimodal data normalization, suspected identification, severity "
        "stratification, refractory risk assessment, RAG evidence retrieval, recommendation generation, "
        "and final traceable report output."
    )

    planner = Planner(disease="pediatric_mpp")
    plan = planner.plan(
        output_path=data_root,
        prompt=planner_prompt,
        rag_text=context,
        filename="plan.json",
        toolset=toolset,
    )
    return plan


if __name__ == "__main__":
    generated = generate_task_level_plan()
    print(f"Generated {len(generated)} proposal-aligned workflow steps at {os.path.join(DATA_ROOT, 'plan.json')}")
