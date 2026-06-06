"""Cherry-pick best single-task instructions and evaluate as a combined multi-task prompt.

For each SummEval task, scans all single-task runs (run_1, run_2, run_3) across
all steps (0-12) to find the step+run with the best value for a target metric
(Spearman, OffByOne, or MAE). Extracts the instruction from that step, combines
the 4 best per-task instructions into one multi-task PromptTemplate, and runs
full-test-set evaluation.

Produces 6 evaluations: {Spearman, OB1, MAE} × {val=none, val=mae}.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


from when_gradients_collide.metrics import Metric

TASKS: List[str] = ["fluency", "relevance", "coherence", "consistency"]
EVAL_METRIC_NAMES: List[str] = [
    "Accuracy", "MAE", "OffByOne", "OffByTwo", "OffByThree",
    "SpearmanCorrelation", "KendallTau",
]
CHERRYPICK_METRICS: List[Dict[str, Any]] = [
    {"name": "SpearmanCorrelation", "short": "spearman", "higher_is_better": True},
    {"name": "OffByOne", "short": "ob1", "higher_is_better": True},
    {"name": "MAE", "short": "mae", "higher_is_better": False},
]
DATE_PREFIX: str = "02Apr2026"
BASE: str = os.path.join(os.path.dirname(__file__), "e2e_outputs", "unified")
RUNS: List[int] = [1, 2, 3]
MAX_STEP: int = 12
LLM_CONFIG_PATH: str = "expt/configs/llm.json"


def get_instruction_at_step(run_dir: str, step: int, task_name: str) -> Optional[str]:
    """Extract the task instruction used at a given evaluation step."""
    if step == 0:
        step1_path: str = os.path.join(run_dir, "run_logs", "step_0001.parquet")
        if not os.path.exists(step1_path):
            return None
        df: pd.DataFrame = pd.read_parquet(step1_path)
        update: Dict = json.loads(df["prompt_update"].iloc[0])
        old_instructions: Dict[str, str] = update["old_instruction"]
        return old_instructions.get(task_name)
    else:
        step_path: str = os.path.join(
            run_dir, "run_logs", f"step_{step:04d}.parquet"
        )
        if not os.path.exists(step_path):
            return None
        df = pd.read_parquet(step_path)
        state: Dict = json.loads(df["algorithm_state"].iloc[0])
        instructions: Dict[str, str] = state["previous_instructions"]
        return instructions.get(task_name)


def find_best_per_task(
    val_short: str,
    target_metric_name: str,
    higher_is_better: bool,
) -> Dict[str, Dict[str, Any]]:
    """For each task, find the (run, step) with the best target metric value."""
    target_cls = Metric.get_subclass(target_metric_name)
    best: Dict[str, Dict[str, Any]] = {}

    for task_name in TASKS:
        best_val: float = -1e9 if higher_is_better else 1e9
        best_info: Optional[Dict[str, Any]] = None

        for run_n in RUNS:
            job_name: str = (
                f"TextGrad-SummEval-{DATE_PREFIX}-CCC"
                f"-val={val_short}-task={task_name}-run_{run_n}"
            )
            run_dir: str = os.path.join(BASE, job_name, "run")
            if not os.path.isdir(run_dir):
                continue

            for step in range(0, MAX_STEP + 1):
                eval_path: str = os.path.join(run_dir, f"eval_step_{step}.parquet")
                if not os.path.exists(eval_path):
                    continue
                df: pd.DataFrame = pd.read_parquet(eval_path)

                gt_col: str = f"gt_{task_name}"
                pred_col: str = f"pred_{task_name}"
                if gt_col not in df.columns or pred_col not in df.columns:
                    continue

                gt = pd.to_numeric(df[gt_col], errors="coerce")
                pred = pd.to_numeric(df[pred_col], errors="coerce")
                valid = gt.notna() & pred.notna()
                y_true: List = gt[valid].tolist()
                y_pred: List = pred[valid].tolist()

                try:
                    metric_val: float = target_cls.compute(
                        y_true=y_true, y_pred=y_pred,
                    )
                except Exception:
                    continue

                is_better: bool = (
                    metric_val > best_val if higher_is_better
                    else metric_val < best_val
                )
                if is_better:
                    all_metrics: Dict[str, float] = {}
                    for mname in EVAL_METRIC_NAMES:
                        try:
                            mcls = Metric.get_subclass(mname)
                            all_metrics[mname] = mcls.compute(
                                y_true=y_true, y_pred=y_pred,
                            )
                        except Exception:
                            pass
                    best_val = metric_val
                    best_info = {
                        "run": run_n,
                        "step": step,
                        "target_value": metric_val,
                        "all_metrics": all_metrics,
                        "run_dir": run_dir,
                    }

        if best_info is not None:
            instruction: Optional[str] = get_instruction_at_step(
                run_dir=best_info["run_dir"],
                step=best_info["step"],
                task_name=task_name,
            )
            best_info["instruction"] = instruction
            best[task_name] = best_info

    return best


def build_combined_prompt_and_evaluate(
    best_per_task: Dict[str, Dict[str, Any]],
    val_short: str,
    target_metric_short: str,
) -> None:
    """Build a multi-task PromptTemplate from cherry-picked instructions and evaluate."""
    from dotenv import load_dotenv
    load_dotenv()

    from runner import create_shared_limits, create_task_llm, get_initial_prompt
    from when_gradients_collide.config import temp_config
    from when_gradients_collide.data_structures import DatasetSample
    from when_gradients_collide.experiment_config import load_config
    from when_gradients_collide.prompt_template import PromptTemplate
    from when_gradients_collide.task_predictor import parse_task_response
    from dataset import SummEval

    # Load LLM config from JSON file
    llm_config = load_config(
        os.path.join(os.path.dirname(__file__), "..", LLM_CONFIG_PATH)
    ).llm

    dataset: SummEval = SummEval(data_dir="data")
    tasks = dataset.tasks

    initial_prompt: PromptTemplate = get_initial_prompt(
        dataset=dataset, tasks=tasks,
    )

    new_instructions: Dict[str, str] = {}
    for task_name in TASKS:
        if task_name in best_per_task and best_per_task[task_name]["instruction"] is not None:
            new_instructions[task_name] = best_per_task[task_name]["instruction"]
        else:
            new_instructions[task_name] = initial_prompt.instruction[task_name]

    combined_prompt: PromptTemplate = PromptTemplate(
        skeleton=initial_prompt.skeleton,
        tasks=initial_prompt.tasks,
        instruction=new_instructions,
        input_col_labels=initial_prompt.input_col_labels,
        task_output_formats=initial_prompt.task_output_formats,
    )

    tag: str = f"val={val_short}, pick={target_metric_short}"
    print(f"\n{'=' * 80}")
    print(f"Combined prompt instructions ({tag}):")
    print(f"{'=' * 80}")
    print(combined_prompt.render_instructions())

    output_dir: str = os.path.join(
        BASE,
        f"TextGrad-SummEval-{DATE_PREFIX}-cherrypick-val={val_short}-pick={target_metric_short}",
        "run",
    )
    os.makedirs(output_dir, exist_ok=True)

    with temp_config(substep_delay=1.0, verbosity=1):
        shared_limits = create_shared_limits()
        task_llm = create_task_llm(
            llm_config=llm_config, limits=shared_limits,
        )
        try:
            test_df: pd.DataFrame = dataset.test()

            test_samples: List[DatasetSample] = []
            for idx, row in test_df.iterrows():
                inputs: Dict[str, Any] = {
                    col: row[col] for col in dataset.input_cols if col in row
                }
                ground_truths: Dict[str, Any] = {
                    col: row[col] for col in dataset.gt_cols if col in row
                }
                test_samples.append(DatasetSample(
                    sample_id=f"test_sample{idx}",
                    inputs=inputs,
                    ground_truths=ground_truths,
                ))

            print(f"\nRunning inference on {len(test_samples)} test samples ({tag})...")

            from when_gradients_collide.config import wgc_config
            from when_gradients_collide.llm_utils import apply_prompt_suffix

            prompts_for_llm: List[str] = [
                combined_prompt.render_task_prompt(sample=s)
                for s in test_samples
            ]
            prompts_for_llm = apply_prompt_suffix(prompts_for_llm, task_llm)

            raw_responses: List[str] = task_llm.call_llm_batch(
                prompts=prompts_for_llm,
                verbosity=1,
            ).result(timeout=wgc_config.defaults.batch_invocation_timeout)

            rows: List[Dict] = []
            for i, (sample, response) in enumerate(zip(test_samples, raw_responses)):
                parsed: Dict[str, Any] = parse_task_response(response=response)
                row: Dict[str, Any] = {
                    "sample_id": sample.sample_id,
                    "prompt": prompts_for_llm[i],
                    "raw_response": response,
                    "inputs": json.dumps(sample.inputs),
                }
                for task_name in TASKS:
                    row[f"gt_{task_name}"] = sample.ground_truths.get(task_name)
                    row[f"pred_{task_name}"] = parsed.get(task_name)
                rows.append(row)

            results_df: pd.DataFrame = pd.DataFrame(rows)
            out_path: str = os.path.join(output_dir, "eval_cherrypick.parquet")
            results_df.to_parquet(out_path)
            print(f"Saved: {out_path}")

            print(f"\n{'=' * 80}")
            print(f"RESULTS: Cherry-pick ({tag})")
            print(f"{'=' * 80}")

            for mname in EVAL_METRIC_NAMES:
                task_vals: List[float] = []
                per_task_str: List[str] = []
                for task_name in TASKS:
                    gt = pd.to_numeric(results_df[f"gt_{task_name}"], errors="coerce")
                    pred = pd.to_numeric(results_df[f"pred_{task_name}"], errors="coerce")
                    valid = gt.notna() & pred.notna()
                    y_true = gt[valid].tolist()
                    y_pred = pred[valid].tolist()
                    try:
                        mcls = Metric.get_subclass(mname)
                        val = mcls.compute(y_true=y_true, y_pred=y_pred)
                        task_vals.append(val)
                        per_task_str.append(f"{task_name}={val:.4f}")
                    except Exception:
                        per_task_str.append(f"{task_name}=ERR")

                avg: float = float(np.mean(task_vals)) if len(task_vals) > 0 else 0.0
                direction: str = "lower=better" if mname == "MAE" else "higher=better"
                print(f"  {mname:>20}: avg={avg:.4f}  ({', '.join(per_task_str)})  [{direction}]")

        finally:
            task_llm.stop()


def main() -> None:
    total_start: float = time.time()

    for val_short in ["none", "mae"]:
        for cp_metric in CHERRYPICK_METRICS:
            metric_name: str = cp_metric["name"]
            metric_short: str = cp_metric["short"]
            higher_is_better: bool = cp_metric["higher_is_better"]

            print(f"\n{'#' * 90}")
            print(f"# Cherry-pick by best {metric_name} (val={val_short})")
            print(f"{'#' * 90}")

            best_per_task: Dict[str, Dict[str, Any]] = find_best_per_task(
                val_short=val_short,
                target_metric_name=metric_name,
                higher_is_better=higher_is_better,
            )

            print(f"\nBest per-task {metric_name} (val={val_short}):")
            for task_name in TASKS:
                if task_name in best_per_task:
                    info = best_per_task[task_name]
                    print(
                        f"  {task_name}: {metric_name}={info['target_value']:.4f} "
                        f"(run_{info['run']}, step {info['step']})"
                    )
                    instr: Optional[str] = info.get("instruction")
                    if instr is not None:
                        print(f"    Instruction: {instr[:120]}...")
                else:
                    print(f"  {task_name}: NO DATA")

            build_combined_prompt_and_evaluate(
                best_per_task=best_per_task,
                val_short=val_short,
                target_metric_short=metric_short,
            )

    total_elapsed: float = time.time() - total_start
    print(f"\n\nTotal time: {total_elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
