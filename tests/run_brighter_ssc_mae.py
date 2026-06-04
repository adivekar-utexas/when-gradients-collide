"""Rerun failed BRIGHTER SSC-val=mae job (single sequential run).

Usage:
    cd PromptMOO
    /Users/adivekar/miniconda3/envs/prompt_moo/bin/python tests/run_brighter_ssc_mae.py
"""

import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "expt"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

STEPS: int = 12
BATCH_SIZE: int = 3
LLM_FAMILY: str = "qwen3"

JOB_NAME: str = "TextGrad-BRIGHTER-02Apr2026-SSC-val=mae-run_1"
MODE_CONFIG: Dict[str, str] = {
    "loss_task_strategy": "separate_tasks",
    "gradient_task_strategy": "separate_tasks",
    "optimizer_task_strategy": "combine_all_tasks",
}
VAL_METRIC: str = "mae"

COMMON_PARAMS: Dict[str, Any] = {
    "validation_gate_samples": 100,
    "gradient_llm_temperature": 0.3,
    "optimizer_llm_temperature": 0.7,
    "loss_llm_temperature": 0.3,
}


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    api_key: str = os.environ["OMNIROUTE_API_KEY"]

    from runner import (
        create_gradient_llm, create_loss_llm, create_optimizer_llm,
        create_task_llm, get_initial_prompt,
    )
    from prompt_moo.algorithm import TextGrad
    from prompt_moo.config import temp_config
    from dataset import BRIGHTER

    dataset = BRIGHTER(data_dir="expt")
    tasks = dataset.tasks

    output_dir: str = f"tests/e2e_outputs/unified/{JOB_NAME}/run"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Job: {JOB_NAME}")
    print(f"Output: {output_dir}")
    print(f"Mode: SSC (separate loss, separate gradient, combine optimizer)")
    print(f"Val metric: {VAL_METRIC}")
    print(f"Steps: {STEPS}, batch_size: {BATCH_SIZE}")
    print()

    start: float = time.time()

    with temp_config(substep_delay=1.0, verbosity=1):
        initial_prompt = get_initial_prompt(dataset=dataset, tasks=tasks

        task_llm = create_task_llm(llm=LLM_FAMILY)
        optimizer_llm = create_optimizer_llm(llm=LLM_FAMILY)
        gradient_llm = create_gradient_llm(llm=LLM_FAMILY)
        loss_llm = create_loss_llm(llm=LLM_FAMILY)

        try:
            algo = TextGrad(
                task_llm=task_llm,
                gradient_llm=gradient_llm,
                optimizer_llm=optimizer_llm,
                loss_llm=loss_llm,
                tasks=tasks,
                steps=STEPS,
                batch_size=BATCH_SIZE,
                eval_every=1,
                eval_initial_prompt=True,
                eval_first_step=True,
                eval_last_step=True,
                name=JOB_NAME,
                verbosity=1,
                validation_metric=VAL_METRIC,
                **COMMON_PARAMS,
                **MODE_CONFIG,
            )
            algo.train(
                dataset=dataset,
                initial_prompt=initial_prompt,
                output_dir=output_dir,
            )
            elapsed: float = (time.time() - start) / 60
            print(f"\nCOMPLETED in {elapsed:.1f} min -> {output_dir}")
        except Exception as e:
            elapsed = (time.time() - start) / 60
            print(f"\nFAILED after {elapsed:.1f} min: {type(e).__name__}: {e}")
            raise
        finally:
            task_llm.stop()
            optimizer_llm.stop()
            gradient_llm.stop()
            loss_llm.stop()


if __name__ == "__main__":
    main()
