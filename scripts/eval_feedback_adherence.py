"""
Feedback Adherence Evaluation (RQ3)

Measures how well the LLM optimizer's revised instructions address the gradient.
Runs on SummEval val=mae and val=none, all modes (SSS/SSC/SCC/CCC + Single),
all 3 runs.

Usage:
    python scripts/eval_feedback_adherence.py
"""

import json
import os
import re
import sys
from ast import literal_eval
from typing import Any, Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv


from when_gradients_collide.llm_utils import apply_prompt_suffix
from when_gradients_collide.expt.runner import _build_retry_config, _stamp_prompt_suffix, create_shared_limits
from slowburn import SlowBurnLLM
from when_gradients_collide.config import wgc_config

BASE: str = os.path.join(os.path.dirname(__file__), "e2e_outputs", "unified")
OUT_DIR: str = os.path.join(BASE, "rq3_results")

DATASET_PREFIX: str = "TextGrad-SummEval-02Apr2026"
VALS: List[str] = ["mae", "none"]
MODES: List[str] = ["SSS", "SSC", "SCC", "CCC"]
SINGLE_TASK_MODE: str = "CCC"
SINGLE_TASK_NAMES: List[str] = ["fluency", "relevance", "coherence", "consistency"]
RUNS: List[int] = [1, 2, 3]

MODEL_NAME: str = "openrouter/anthropic/claude-sonnet-4.6"
PROVIDER_ORDER: List[str] = ["amazon-bedrock"]
MAX_TOKENS: int = 10
BATCH_TIMEOUT: int = 3600

ADHERENCE_PROMPT = """\
You are evaluating whether revisions to task-specific instructions correctly \
addressed the gradient (suggested changes) that was provided.

The instructions are for an LLM judge that evaluates the "{task}" task. \
The Gradient section may contain suggestions about multiple tasks; consider \
only suggestions pertaining to "{task}".

Rate from 1 to 10 how well the New Instructions address the Gradient for \
the "{task}" task. 1 = completely ignores/contradicts the gradient. \
10 = precisely addresses every point while preserving what worked.

## Old Instructions
{old_instruction}

## New Instructions
{new_instruction}

## Gradient (Suggested Changes)
{gradient_text}

Respond with ONLY a single integer from 1 to 10. No explanation."""


def create_eval_llm(api_key: str) -> SlowBurnLLM:
    cfg = wgc_config.defaults
    limits = create_shared_limits()
    llm = SlowBurnLLM.options(
        mode="asyncio",
        limits=limits,
        **_build_retry_config(cfg=cfg),
    ).init(
        name="eval_llm",
        model_name=MODEL_NAME,
        api_key=api_key,
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        timeout=cfg.optimizer_llm_timeout,
        litellm_params={
            "extra_body": {
                "provider": {"order": PROVIDER_ORDER, "allow_fallbacks": False},
                "reasoning": {"effort": "none"},
            }
        },
    )
    _stamp_prompt_suffix(llm, model_name=MODEL_NAME, reasoning=False)
    return llm


def discover_runs() -> List[Tuple[str, str]]:
    results = []
    for val in VALS:
        for mode in MODES:
            for run_id in RUNS:
                run_name = f"{DATASET_PREFIX}-{mode}-val={val}-run_{run_id}"
                run_dir = os.path.join(BASE, run_name, "run")
                if os.path.isdir(run_dir):
                    results.append((run_name, run_dir))
        for task in SINGLE_TASK_NAMES:
            for run_id in RUNS:
                run_name = f"{DATASET_PREFIX}-{SINGLE_TASK_MODE}-val={val}-task={task}-run_{run_id}"
                run_dir = os.path.join(BASE, run_name, "run")
                if os.path.isdir(run_dir):
                    results.append((run_name, run_dir))
    return sorted(results)


def load_run_config(run_dir: str) -> Dict[str, Any]:
    with open(os.path.join(run_dir, "run_summary.json")) as f:
        summary = json.load(f)
    config = summary["config"]
    return {
        "task_names": [t["task_name"] for t in config["tasks"]],
        "num_steps": config["steps"],
        "name": config["name"],
    }


def parse_mode(run_name: str) -> str:
    if "-task=" in run_name:
        return "Single"
    for mode in ["SSS", "SSC", "SCC", "CCC"]:
        if f"-{mode}-" in run_name:
            return mode
    return "unknown"


def parse_val(run_name: str) -> str:
    m = re.search(r"val=(\w+)", run_name)
    return m.group(1) if m else "unknown"


def parse_run_id(run_name: str) -> int:
    m = re.search(r"run_(\d+)$", run_name)
    return int(m.group(1)) if m else 0


def build_eval_prompts_for_run(
    run_dir: str,
    task_names: List[str],
    num_steps: int,
) -> List[Dict[str, Any]]:
    records = []
    log_dir = os.path.join(run_dir, "run_logs")
    for step in range(1, num_steps + 1):
        step_file = os.path.join(log_dir, f"step_{step:04d}.parquet")
        if not os.path.exists(step_file):
            continue
        df = pd.read_parquet(step_file)
        grads_dict = json.loads(df["gradients"][0]).get("gradients", {})
        prompt_update_str = df["prompt_update"][0]
        try:
            prompt_update = literal_eval(prompt_update_str)
        except (ValueError, SyntaxError):
            prompt_update = json.loads(prompt_update_str)
        old_instr = prompt_update["old_instruction"]
        new_instr = prompt_update["new_instruction"]

        for task in task_names:
            task_grads = grads_dict.get(task, [])
            gradient_text = "\n\n".join(g["gradient_text"] for g in task_grads if "gradient_text" in g)
            if not gradient_text.strip():
                continue
            old_i = old_instr.get(task, "") if isinstance(old_instr, dict) else str(old_instr)
            new_i = new_instr.get(task, "") if isinstance(new_instr, dict) else str(new_instr)
            records.append({
                "step": step,
                "task": task,
                "eval_prompt": ADHERENCE_PROMPT.format(
                    task=task, old_instruction=old_i, new_instruction=new_i, gradient_text=gradient_text,
                ),
            })
    return records


def parse_score(text: str) -> int:
    """Extract integer score from raw LLM response."""
    text = text.strip()
    m = re.search(r"\b(\d{1,2})\b", text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 10:
            return val
    return -1


def print_analysis(df: pd.DataFrame, metric_name: str) -> None:
    s = df["score"]
    valid = df[df["score"] > 0]
    print(f"\n{'=' * 70}")
    print(f"{metric_name} — {len(df)} evaluations, {len(valid)} valid scores")
    print(f"Score range [{s[s > 0].min()}, {s[s > 0].max()}], mean {s[s > 0].mean():.2f}")
    print(f"{'=' * 70}")

    mode_order = ["Single", "SSS", "SSC", "SCC", "CCC"]
    valid["mode"] = pd.Categorical(valid["mode"], categories=mode_order, ordered=True)

    print(f"\n--- {metric_name}: Mode x Val (mean ± std, averaged over runs) ---")
    agg = valid.groupby(["mode", "val"])["score"].agg(["mean", "std", "count"]).round(2)
    print(agg.to_string())

    print(f"\n--- {metric_name}: Mode x Val x Task ---")
    agg2 = valid.groupby(["mode", "val", "task"])["score"].agg(["mean", "std"]).round(2)
    print(agg2.to_string())

    for val in VALS:
        subset = valid[valid["val"] == val]
        if subset.empty:
            continue
        print(f"\n--- {metric_name}: Step-wise scores for val={val} (mean over runs x tasks) ---")
        pivot = subset.groupby(["step", "mode"])["score"].mean().unstack("mode").round(2)
        pivot = pivot[[c for c in mode_order if c in pivot.columns]]
        print(pivot.to_string())

        print(f"\n--- {metric_name}: Step-wise scores for val={val}, per task (mean over runs) ---")
        for task in sorted(subset["task"].unique()):
            task_sub = subset[subset["task"] == task]
            if task_sub.empty:
                continue
            print(f"\n  Task: {task}")
            pivot_t = task_sub.groupby(["step", "mode"])["score"].mean().unstack("mode").round(2)
            pivot_t = pivot_t[[c for c in mode_order if c in pivot_t.columns]]
            print(pivot_t.to_string())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "expt", "_env"))
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "_env"))
    api_key = os.getenv("OPENROUTER_API_KEY")
    assert api_key, "OPENROUTER_API_KEY not found"

    runs = discover_runs()
    print(f"Discovered {len(runs)} runs")
    print(f"Model: {MODEL_NAME}, providers: {PROVIDER_ORDER}, max_tokens: {MAX_TOKENS}")

    llm = create_eval_llm(api_key)

    all_dfs = []
    for run_name, run_dir in runs:
        config = load_run_config(run_dir)
        records = build_eval_prompts_for_run(run_dir, config["task_names"], config["num_steps"])
        if not records:
            continue
        mode = parse_mode(run_name)
        val = parse_val(run_name)
        run_id = parse_run_id(run_name)
        print(f"  {run_name}: {len(records)} evals (mode={mode}, val={val}, run={run_id})")

        prompts = apply_prompt_suffix([r["eval_prompt"] for r in records], llm)
        responses: List[str] = llm.call_llm_batch(prompts=prompts, verbosity=2).result(timeout=BATCH_TIMEOUT)

        rows = []
        for rec, resp in zip(records, responses):
            score = parse_score(resp)
            rows.append({
                "run_name": run_name, "mode": mode, "val": val, "run_id": run_id,
                "step": rec["step"], "task": rec["task"],
                "score": score, "raw_response": resp.strip(),
            })
        all_dfs.append(pd.DataFrame(rows))

    if not all_dfs:
        print("No results.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = os.path.join(OUT_DIR, "feedback_adherence.parquet")
    combined.to_parquet(out_path, index=False)
    print(f"\nSaved to: {out_path}")

    failed = combined[combined["score"] < 0]
    if len(failed) > 0:
        print(f"\nWARNING: {len(failed)} responses failed to parse. Samples:")
        for _, row in failed.head(5).iterrows():
            print(f"  {row['run_name']} step={row['step']} task={row['task']}: '{row['raw_response']}'")

    print_analysis(combined, metric_name="Feedback Adherence")


if __name__ == "__main__":
    main()
