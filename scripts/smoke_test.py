"""Smoke test: 2-step TextGrad on SummEval with a cheap model.

Uses the .env API key. Run from repo root:
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


def main():
    print("=" * 60)
    print("SMOKE TEST: 2-step TextGrad on SummEval")
    print("=" * 60)

    # 1. Load dataset
    dataset = SummEval(data_dir="data")
    print(f"Dataset: {dataset.dataset_name}")
    print(f"Tasks: {[t.task_name for t in dataset.tasks]}")

    # 2. Build initial prompt
    initial_prompt = get_initial_prompt(dataset=dataset, tasks=dataset.tasks)
    print(f"Initial prompt skeleton length: {len(initial_prompt.skeleton)} chars")

    # 3. Get task losses
    task_losses = get_task_losses(dataset=dataset, tasks=dataset.tasks)
    print(f"Task losses: {task_losses}")

    # 4. Run a 2-step TextGrad optimization
    #    Use a cheap model via OpenRouter
    runner = AlgorithmRunner.options(mode="threads", max_workers=1).init()

    future = runner.run(
        dataset=dataset,
        algo_name="textgrad",
        api_key=os.environ["OPENROUTER_API_KEY"],
        steps=2,
        batch_size=3,
        eval_every=1,
        run_name="smoke_test",
        llm="openrouter_deepseek",  # uses OpenRouter with DeepSeek models
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

    return result["status"] == "success"


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
