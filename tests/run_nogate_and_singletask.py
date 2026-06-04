"""Run TextGrad experiments: no-gate multi-task + single-task ablations.

Job groups:
1. Multi-task, val=none, 4 modes (SSS/SSC/SCC/CCC) — per dataset
2. Single-task, val=mae + val=none, CCC mode, one task at a time — per dataset

Runs SummEval first (all jobs parallel), then BRIGHTER (all jobs parallel).
"""

import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "expt"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

METRIC_NAMES: List[str] = [
    "Accuracy", "MAE", "OffByOne", "OffByTwo", "OffByThree",
    "SpearmanCorrelation", "KendallTau",
]

STEPS: int = 12
BATCH_SIZE: int = 3
LLM_FAMILY: str = "qwen3"
DATE_PREFIX: str = "02Apr2026"

MODE_CONFIGS: Dict[str, Dict[str, str]] = {
    "SSS": {
        "loss_task_strategy": "separate_tasks",
        "gradient_task_strategy": "separate_tasks",
        "optimizer_task_strategy": "separate_tasks",
    },
    "SSC": {
        "loss_task_strategy": "separate_tasks",
        "gradient_task_strategy": "separate_tasks",
        "optimizer_task_strategy": "combine_all_tasks",
    },
    "SCC": {
        "loss_task_strategy": "separate_tasks",
        "gradient_task_strategy": "combine_all_tasks",
        "optimizer_task_strategy": "combine_all_tasks",
    },
    "CCC": {
        "loss_task_strategy": "combine_all_tasks",
        "gradient_task_strategy": "combine_all_tasks",
        "optimizer_task_strategy": "combine_all_tasks",
    },
}

COMMON_PARAMS: Dict[str, Any] = {
    "validation_gate_samples": 100,
    "gradient_llm_temperature": 0.3,
    "optimizer_llm_temperature": 0.7,
    "loss_llm_temperature": 0.3,
}

DATASET_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "SummEval",
        "dataset_class": "SummEval",
        "all_task_names": ["fluency", "relevance", "coherence", "consistency"],
    },
    {
        "name": "BRIGHTER",
        "dataset_class": "BRIGHTER",
        "all_task_names": ["anger", "fear", "joy", "sadness", "surprise"],
    },
]


def run_single(
    job_name: str,
    mode_config: Dict[str, str],
    val_metric: str,
    dataset_class_name: str,
    task_names_subset: Optional[List[str]],
) -> str:
    """Run one TextGrad experiment.

    Args:
        task_names_subset: If provided, only these tasks are used (single-task runs).
            If None, all tasks from the dataset are used.
    """
    from dotenv import load_dotenv
    load_dotenv()
    api_key: str = os.environ["OMNIROUTE_API_KEY"]

    from runner import (
        create_gradient_llm, create_loss_llm, create_optimizer_llm,
        create_task_llm, get_initial_prompt,
    )
    from prompt_moo.algorithm import TextGrad
    from prompt_moo.config import temp_config

    import dataset as ds_module
    dataset_cls = getattr(ds_module, dataset_class_name)
    dataset = dataset_cls(data_dir="expt")

    if task_names_subset is not None:
        task_name_set = set(task_names_subset)
        tasks = [t for t in dataset.tasks if t.task_name in task_name_set]
        if len(tasks) != len(task_names_subset):
            found: List[str] = [t.task_name for t in tasks]
            raise ValueError(
                f"Expected tasks {task_names_subset} but found {found} "
                f"in {dataset_class_name}.tasks"
            )
    else:
        tasks = dataset.tasks

    output_dir: str = f"tests/e2e_outputs/unified/{job_name}/run"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with temp_config(substep_delay=1.0, verbosity=1):
        initial_prompt = get_initial_prompt(dataset=dataset, tasks=tasks)

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
                name=job_name,
                verbosity=1,
                validation_metric=val_metric,
                **COMMON_PARAMS,
                **mode_config,
            )
            algo.train(
                dataset=dataset,
                initial_prompt=initial_prompt,
                output_dir=output_dir,
            )
            print(f"\n[{job_name}] COMPLETED -> {output_dir}")
        finally:
            task_llm.stop()
            optimizer_llm.stop()
            gradient_llm.stop()
            loss_llm.stop()

    return output_dir


def extract_metrics(
    run_dir: str, task_names: List[str], max_step: int,
) -> Dict[int, Dict[str, float]]:
    from prompt_moo.metrics import Metric
    results: Dict[int, Dict[str, float]] = {}
    for step in range(0, max_step + 1):
        eval_path: str = os.path.join(run_dir, f"eval_step_{step}.parquet")
        if not os.path.exists(eval_path):
            continue
        df: pd.DataFrame = pd.read_parquet(eval_path)
        row: Dict[str, float] = {}
        for mname in METRIC_NAMES:
            task_vals: List[float] = []
            for task in task_names:
                gt_col: str = f"gt_{task}"
                pred_col: str = f"pred_{task}"
                if gt_col not in df.columns or pred_col not in df.columns:
                    continue
                gt = pd.to_numeric(df[gt_col], errors="coerce")
                pred = pd.to_numeric(df[pred_col], errors="coerce")
                valid = gt.notna() & pred.notna()
                y_true = gt[valid].tolist()
                y_pred = pred[valid].tolist()
                try:
                    mcls = Metric.get_subclass(mname)
                    val = mcls.compute(y_true=y_true, y_pred=y_pred)
                    task_vals.append(val)
                except Exception:
                    pass
            row[mname] = float(np.mean(task_vals)) if len(task_vals) > 0 else 0.0
        results[step] = row
    return results


# ── Build job list per dataset ──────────────────────────────────────────────

JobSpec = Tuple[str, Dict[str, str], str, Optional[List[str]], List[str]]
# (job_name, mode_config, val_metric, task_names_subset, task_names_for_eval)


def build_jobs(ds_config: Dict[str, Any]) -> List[JobSpec]:
    ds_name: str = ds_config["name"]
    all_tasks: List[str] = ds_config["all_task_names"]
    jobs: List[JobSpec] = []

    # Group 1: Multi-task, val=none, 4 modes
    for mode_name in ["SSS", "SSC", "SCC", "CCC"]:
        job_name: str = f"TextGrad-{ds_name}-{DATE_PREFIX}-{mode_name}-val=none-run_1"
        jobs.append((
            job_name, MODE_CONFIGS[mode_name], "none", None, all_tasks,
        ))

    # Group 2: Single-task, CCC mode, val=mae and val=none
    for task_name in all_tasks:
        for val_metric, val_short in [("mae", "mae"), ("none", "none")]:
            job_name = (
                f"TextGrad-{ds_name}-{DATE_PREFIX}-CCC"
                f"-val={val_short}-task={task_name}-run_1"
            )
            jobs.append((
                job_name,
                MODE_CONFIGS["CCC"],
                val_metric,
                [task_name],
                [task_name],
            ))

    return jobs


def run_dataset(ds_config: Dict[str, Any]) -> None:
    ds_name: str = ds_config["name"]
    ds_class: str = ds_config["dataset_class"]
    jobs: List[JobSpec] = build_jobs(ds_config)

    print(f"\n{'=' * 90}")
    print(f"Dataset: {ds_name} — Launching {len(jobs)} parallel jobs")
    print(f"{'=' * 90}")
    for job_name, _, val_metric, subset, _ in jobs:
        task_tag: str = f" (tasks={subset})" if subset is not None else ""
        print(f"  {job_name}{task_tag}")
    print()

    start_time: float = time.time()
    output_dirs: Dict[str, str] = {}

    max_workers: int = min(len(jobs), 16)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_single, job_name, mode_cfg, val_metric, ds_class, subset,
            ): (job_name, eval_tasks)
            for job_name, mode_cfg, val_metric, subset, eval_tasks in jobs
        }
        for future in as_completed(futures):
            job_name, eval_tasks = futures[future]
            try:
                out_dir: str = future.result()
                output_dirs[job_name] = (out_dir, eval_tasks)
                print(f"[{job_name}] Done: {out_dir}")
            except Exception as e:
                print(f"[{job_name}] FAILED: {e}")

    elapsed: float = time.time() - start_time
    print(f"\n{ds_name}: All jobs completed in {elapsed / 60:.1f} minutes")

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n{'=' * 150}")
    print(f"DELTA Step 0 -> Step {STEPS}: {ds_name}")
    print(f"{'=' * 150}")

    for job_name, _, _, _, _ in jobs:
        if job_name not in output_dirs:
            print(f"\n{job_name}: FAILED")
            continue
        out_dir, eval_tasks = output_dirs[job_name]
        metrics = extract_metrics(out_dir, eval_tasks, STEPS)
        if 0 not in metrics:
            print(f"\n{job_name}: No Step 0 data")
            continue
        last_step: int = max(metrics.keys())
        m0 = metrics[0]
        m_last = metrics[last_step]
        print(f"\n{job_name} (Step 0 -> {last_step}):")
        for mn in METRIC_NAMES:
            v0 = m0.get(mn, 0)
            v_last = m_last.get(mn, 0)
            delta = v_last - v0
            pct = (delta / abs(v0) * 100) if v0 != 0 else 0
            direction = "lower=better" if mn == "MAE" else "higher=better"
            print(
                f"  {mn:>20}: {v0:.4f} -> {v_last:.4f}  "
                f"({delta:+.4f}, {pct:+.1f}%)  [{direction}]"
            )


def main() -> None:
    total_start: float = time.time()
    for ds_config in DATASET_CONFIGS:
        run_dataset(ds_config)
    total_elapsed: float = time.time() - total_start
    print(f"\n\nTotal time for both datasets: {total_elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
