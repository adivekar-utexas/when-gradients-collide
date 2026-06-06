"""Smoke test: 2-step TextGrad on SummEval with a cheap model.

Uses a JSON config file (loaded via TypedPath) and a real LLM API.
Run from repo root:
    python scripts/smoke_test.py
"""

import os
import sys
from pathlib import Path

# Load env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from when_gradients_collide.expt.dataset import SummEval
from when_gradients_collide.expt.runner import (
    AlgorithmRunner,
    get_initial_prompt,
    get_task_losses,
)
from when_gradients_collide.experiment_config import ExperimentConfig, load_config


REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = REPO_ROOT / "expt" / "configs" / "experiments" / "summeval-openrouter-deepseek.json"


def main() -> int:
    print("=" * 60)
    print("SMOKE TEST: 2-step TextGrad on SummEval")
    print("=" * 60)

    # 1. Load experiment config (this also loads the LLM config from its JSON path)
    config_path = os.environ.get("WGC_EXPERIMENT_CONFIG", str(DEFAULT_CONFIG))
    print(f"Loading config: {config_path}")
    exp_config: ExperimentConfig = load_config(config_path)
    print(f"Dataset: {exp_config.dataset}")
    print(f"Steps: {exp_config.steps} (overridden to 2 for smoke test)")
    print(f"LLM config loaded: task_model={exp_config.llm.task_model.name}")

    # 2. Load dataset
    dataset = SummEval(data_dir="data")
    print(f"Dataset: {dataset.dataset_name}")
    print(f"Tasks: {[t.task_name for t in dataset.tasks]}")

    # 3. Build initial prompt
    initial_prompt = get_initial_prompt(dataset=dataset, tasks=dataset.tasks)
    print(f"Initial prompt skeleton length: {len(initial_prompt.skeleton)} chars")

    # 4. Get task losses
    task_losses = get_task_losses(dataset=dataset, tasks=dataset.tasks)
    print(f"Task losses: {task_losses}")

    # 5. Run a 2-step TextGrad optimization
    runner = AlgorithmRunner.options(mode="threads", max_workers=1).init()

    future = runner.run(
        dataset=dataset,
        algo_name="textgrad",
        llm_config=exp_config.llm,
        steps=2,
        batch_size=3,
        eval_every=1,
        run_name="smoke_test",
        verbosity=1,
        output_dir="smoke_test_output",
        validation_metric="accuracy",  # required for TextGrad
    )

    result = future.result()

    print("=" * 60)
    print(f"Status: {result['status']}")
    if result["status"] == "success":
        print("SMOKE TEST PASSED")
    else:
        print(f"Error: {result.get('error', 'unknown')}")
        print("SMOKE TEST FAILED")
    print("=" * 60)

    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
