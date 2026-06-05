"""Run 12 TextGrad configs in parallel per dataset: 4 modes x 3 validation metrics.

Modes: SSS, SSC, SCC, CCC
Validation metrics: off_by_one, mae, spearman_correlation
Fixed: batch_size=3, grad_temp=0.3, opt_temp=0.7, loss_temp=0.3, val_samples=100, steps=12
Runs SummEval first (12 parallel), then BRIGHTER (12 parallel).
"""

import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


METRIC_NAMES: List[str] = [
    "Accuracy",
    "MAE",
    "OffByOne",
    "OffByTwo",
    "OffByThree",
    "SpearmanCorrelation",
    "KendallTau",
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

VAL_METRICS: List[str] = ["off_by_one", "mae", "spearman_correlation"]
VAL_SHORT: Dict[str, str] = {
    "off_by_one": "ob1",
    "mae": "mae",
    "spearman_correlation": "spearman",
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
        "tasks_attr": "tasks",
        "task_names": ["fluency", "relevance", "coherence", "consistency"],
    },
    {
        "name": "BRIGHTER",
        "dataset_class": "BRIGHTER",
        "tasks_attr": "tasks",
        "task_names": ["anger", "fear", "joy", "sadness", "surprise"],
    },
]


def run_single(
    job_name: str,
    mode_config: Dict[str, str],
    val_metric: str,
    dataset_class_name: str,
) -> str:
    from dotenv import load_dotenv

    load_dotenv()
    api_key: str = os.environ["OPENROUTER_API_KEY"]

    from runner import (
        create_gradient_llm,
        create_loss_llm,
        create_optimizer_llm,
        create_shared_limits,
        create_task_llm,
        get_initial_prompt,
    )
    from when_gradients_collide.algorithm import TextGrad
    from when_gradients_collide.config import temp_config

    from when_gradients_collide.expt import dataset as ds_module

    dataset_cls = getattr(ds_module, dataset_class_name)
    dataset = dataset_cls(data_dir="data")
    tasks = dataset.tasks

    output_dir: str = f"results/{job_name}/run"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with temp_config(substep_delay=1.0, verbosity=1):
        initial_prompt = get_initial_prompt(dataset=dataset, tasks=tasks)
        shared_limits = create_shared_limits()

        task_llm = create_task_llm(
            llm=LLM_FAMILY, api_key=api_key, limits=shared_limits
        )
        optimizer_llm = create_optimizer_llm(
            llm=LLM_FAMILY, api_key=api_key, limits=shared_limits
        )
        gradient_llm = create_gradient_llm(
            llm=LLM_FAMILY, api_key=api_key, limits=shared_limits
        )
        loss_llm = create_loss_llm(
            llm=LLM_FAMILY, api_key=api_key, limits=shared_limits
        )

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
    run_dir: str, task_names: List[str], max_step: int
) -> Dict[int, Dict[str, float]]:
    from when_gradients_collide.metrics import Metric

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
                gt = pd.to_numeric(df[f"gt_{task}"], errors="coerce")
                pred = pd.to_numeric(df[f"pred_{task}"], errors="coerce")
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


def run_dataset(ds_config: Dict[str, Any]) -> None:
    ds_name: str = ds_config["name"]
    ds_class: str = ds_config["dataset_class"]
    task_names: List[str] = ds_config["task_names"]

    jobs: List[Tuple[str, Dict[str, str], str]] = []
    for mode_name, mode_config in MODE_CONFIGS.items():
        for val_metric in VAL_METRICS:
            val_short: str = VAL_SHORT[val_metric]
            job_name: str = (
                f"TextGrad-{ds_name}-{DATE_PREFIX}-{mode_name}-val={val_short}-run_1"
            )
            jobs.append((job_name, mode_config, val_metric))

    print(f"\n{'=' * 80}")
    print(f"Dataset: {ds_name} — Launching {len(jobs)} parallel jobs")
    print(f"{'=' * 80}")
    for job_name, _, val_metric in jobs:
        print(f"  {job_name}")
    print()

    start_time: float = time.time()
    output_dirs: Dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(
                run_single, job_name, mode_config, val_metric, ds_class
            ): job_name
            for job_name, mode_config, val_metric in jobs
        }
        for future in as_completed(futures):
            job_name: str = futures[future]
            try:
                out_dir: str = future.result()
                output_dirs[job_name] = out_dir
                print(f"[{job_name}] Done: {out_dir}")
            except Exception as e:
                print(f"[{job_name}] FAILED: {e}")

    elapsed: float = time.time() - start_time
    print(f"\n{ds_name}: All jobs completed in {elapsed / 60:.1f} minutes")

    print(f"\n{'=' * 150}")
    print(f"RESULTS ({ds_name}): Average across {len(task_names)} tasks, per step")
    print(f"{'=' * 150}")

    header: str = f"{'Job':>60} | {'Step':>4}"
    for mn in METRIC_NAMES:
        short: str = mn[:7]
        header += f" | {short:>8}"
    print(header)
    print("-" * 150)

    for job_name, _, _ in jobs:
        if job_name not in output_dirs:
            print(f"{job_name:>60} | FAILED")
            continue
        metrics = extract_metrics(output_dirs[job_name], task_names, STEPS)
        for step in sorted(metrics.keys()):
            m = metrics[step]
            row_str: str = f"{job_name:>60} | {step:>4}"
            for mn in METRIC_NAMES:
                row_str += f" | {m.get(mn, 0):>8.4f}"
            print(row_str)
        print()

    print(f"\n{'=' * 150}")
    print(f"DELTA FROM BASELINE ({ds_name}): Step 0 -> Step {STEPS}")
    print(f"{'=' * 150}")
    for job_name, _, _ in jobs:
        if job_name not in output_dirs:
            continue
        metrics = extract_metrics(output_dirs[job_name], task_names, STEPS)
        if 0 not in metrics or STEPS not in metrics:
            continue
        m0 = metrics[0]
        m_last = metrics[STEPS]
        print(f"\n{job_name}:")
        for mn in METRIC_NAMES:
            v0 = m0.get(mn, 0)
            v_last = m_last.get(mn, 0)
            delta = v_last - v0
            pct = (delta / abs(v0) * 100) if v0 != 0 else 0
            direction = "lower=better" if mn == "MAE" else "higher=better"
            print(
                f"  {mn:>20}: {v0:.4f} -> {v_last:.4f}  (delta={delta:+.4f}, {pct:+.1f}%)  [{direction}]"
            )


def main() -> None:
    total_start: float = time.time()
    for ds_config in DATASET_CONFIGS:
        run_dataset(ds_config)
    total_elapsed: float = time.time() - total_start
    print(f"\n\nTotal time for both datasets: {total_elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
