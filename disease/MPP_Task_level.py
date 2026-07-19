import json
import os

try:
    from .Planner import Planner
except ImportError:
    from Planner import Planner


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_ROOT = os.path.join(PROJECT_DIR, "case", "MycoplasmaPneumonia")


def generate_plan(data_root: str = DATA_ROOT, filename: str = "plan.json"):
    """Generate the same proposal-aligned disease pathway as Task_level.py."""
    tool_file = os.path.join(data_root, "toolset.json")
    with open(tool_file, "r", encoding="utf-8") as f:
        toolset = json.load(f)

    planner = Planner(disease="pediatric_mpp")
    return planner.plan(output_path=data_root, filename=filename, toolset=toolset)


if __name__ == "__main__":
    plan = generate_plan()
    print(f"Generated {len(plan)} MPP workflow steps at {os.path.join(DATA_ROOT, 'plan.json')}")
