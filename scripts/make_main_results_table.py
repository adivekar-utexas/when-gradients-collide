import glob
import itertools
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


from when_gradients_collide.metrics import Metric


BASE: str = os.path.join(os.path.dirname(__file__), "e2e_outputs", "unified")
OUT_DIR: str = os.path.join(BASE, "textgrad_mode_plots")
OUT_PATH: str = os.path.join(
    OUT_DIR,
    "textgrad_summeval_spearman_ob1_report.md",
)
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_PREFIX: str = "TextGrad-SummEval-02Apr2026"
TASKS: List[str] = ["fluency", "relevance", "coherence", "consistency"]
VALS: List[str] = ["none", "mae"]
MODES: List[str] = ["Single", "SSS", "SSC", "SCC", "CCC"]

SPEARMAN: str = "SpearmanCorrelation"
OB1: str = "OffByOne"
MAX_STEP: int = 12
SINGLE_TASK_MODE: str = "CCC"
WORST_SPEARMAN: float = -1.0
WORST_OB1: float = 0.0


def filter_dominated_maximise(*, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points

    keep_mask: np.ndarray = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        if not keep_mask[i]:
            continue
        for j in range(len(points)):
            if i == j:
                continue
            dominates: bool = bool(np.all(points[j] >= points[i]) and np.any(points[j] > points[i]))
            if dominates:
                keep_mask[i] = False
                break
    return points[keep_mask]


def exact_hypervolume_maximise(
    *,
    points: np.ndarray,
    reference_point: np.ndarray,
) -> float:
    if len(points) == 0:
        return 0.0

    shifted: np.ndarray = points - reference_point
    positive: np.ndarray = shifted[np.all(shifted > 0.0, axis=1)]
    if len(positive) == 0:
        return 0.0

    volume: float = 0.0
    point_count: int = len(positive)
    for subset_size in range(1, point_count + 1):
        sign: float = 1.0 if subset_size % 2 == 1 else -1.0
        for subset in itertools.combinations(range(point_count), subset_size):
            subset_points: np.ndarray = positive[list(subset)]
            upper_corner: np.ndarray = np.min(subset_points, axis=0)
            intersection_volume: float = float(np.prod(upper_corner))
            volume += sign * intersection_volume
    return volume


def discover_run_dirs(*, prefix: str, config_base: str) -> List[str]:
    pattern: str = os.path.join(BASE, f"{prefix}-{config_base}-run_*", "run")
    run_dirs: List[str] = sorted(glob.glob(pattern))
    return run_dirs


def compute_metric_value(
    *,
    df: pd.DataFrame,
    task_name: str,
    metric_name: str,
) -> Optional[float]:
    gt_col: str = f"gt_{task_name}"
    pred_col: str = f"pred_{task_name}"
    if gt_col not in df.columns:
        return None
    if pred_col not in df.columns:
        return None

    gt: pd.Series = pd.to_numeric(df[gt_col], errors="coerce")
    pred: pd.Series = pd.to_numeric(df[pred_col], errors="coerce")
    valid: pd.Series = gt.notna() & pred.notna()
    y_true: List = gt[valid].tolist()
    y_pred: List = pred[valid].tolist()
    if len(y_true) == 0:
        return None

    metric_cls = Metric.get_subclass(metric_name)
    value: float = float(metric_cls.compute(y_true=y_true, y_pred=y_pred))
    return value


def load_multitask_rows(
    *,
    prefix: str,
    mode_name: str,
    val_name: str,
) -> List[Dict[str, object]]:
    config_base: str = f"{mode_name}-val={val_name}"
    run_dirs: List[str] = discover_run_dirs(prefix=prefix, config_base=config_base)
    rows: List[Dict[str, object]] = []
    if len(run_dirs) == 0:
        return rows

    for run_dir in run_dirs:
        run_num_text: str = run_dir.split("-run_")[-1].split(os.sep)[0]
        run_num: int = int(run_num_text)
        for step in range(0, MAX_STEP + 1):
            parquet_path: str = os.path.join(run_dir, f"eval_step_{step}.parquet")
            if not os.path.exists(parquet_path):
                continue
            df: pd.DataFrame = pd.read_parquet(parquet_path)
            for task_name in TASKS:
                spearman_value: Optional[float] = compute_metric_value(
                    df=df,
                    task_name=task_name,
                    metric_name=SPEARMAN,
                )
                ob1_value: Optional[float] = compute_metric_value(
                    df=df,
                    task_name=task_name,
                    metric_name=OB1,
                )
                if spearman_value is None or ob1_value is None:
                    continue
                row: Dict[str, object] = {
                    "config": mode_name,
                    "val": val_name,
                    "run": run_num,
                    "step": step,
                    "task": task_name,
                    SPEARMAN: spearman_value,
                    OB1: ob1_value,
                }
                rows.append(row)
    return rows


def load_single_task_rows(
    *,
    prefix: str,
    val_name: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for task_name in TASKS:
        config_base: str = f"{SINGLE_TASK_MODE}-val={val_name}-task={task_name}"
        run_dirs: List[str] = discover_run_dirs(prefix=prefix, config_base=config_base)
        for run_dir in run_dirs:
            run_num_text: str = run_dir.split("-run_")[-1].split(os.sep)[0]
            run_num: int = int(run_num_text)
            for step in range(0, MAX_STEP + 1):
                parquet_path: str = os.path.join(run_dir, f"eval_step_{step}.parquet")
                if not os.path.exists(parquet_path):
                    continue
                df: pd.DataFrame = pd.read_parquet(parquet_path)
                spearman_value: Optional[float] = compute_metric_value(
                    df=df,
                    task_name=task_name,
                    metric_name=SPEARMAN,
                )
                ob1_value: Optional[float] = compute_metric_value(
                    df=df,
                    task_name=task_name,
                    metric_name=OB1,
                )
                if spearman_value is None or ob1_value is None:
                    continue
                row: Dict[str, object] = {
                    "config": "Single",
                    "val": val_name,
                    "run": run_num,
                    "step": step,
                    "task": task_name,
                    SPEARMAN: spearman_value,
                    OB1: ob1_value,
                }
                rows.append(row)
    return rows


def load_all_rows() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for val_name in VALS:
        rows.extend(load_single_task_rows(prefix=DATASET_PREFIX, val_name=val_name))
        for mode_name in MODES:
            if mode_name == "Single":
                continue
            rows.extend(
                load_multitask_rows(
                    prefix=DATASET_PREFIX,
                    mode_name=mode_name,
                    val_name=val_name,
                )
            )
    result: pd.DataFrame = pd.DataFrame(rows)
    return result


def compute_hvi_progression(
    *,
    config_df: pd.DataFrame,
    metric_name: str,
) -> Dict[int, float]:
    steps_sorted: List[int] = sorted(config_df["step"].unique().tolist())
    step_vectors: Dict[int, np.ndarray] = {}
    for step in steps_sorted:
        step_df: pd.DataFrame = config_df[config_df["step"] == step]
        means_by_task: Dict[str, float] = {}
        grouped: pd.DataFrame = (
            step_df.groupby("task")[metric_name]
            .mean()
            .reset_index()
        )
        for _, row in grouped.iterrows():
            task_name: str = row["task"]
            metric_value: float = float(row[metric_name])
            means_by_task[task_name] = metric_value

        values: List[float] = []
        for task_name in TASKS:
            if task_name not in means_by_task:
                values = []
                break
            values.append(means_by_task[task_name])
        if len(values) == len(TASKS):
            step_vectors[step] = np.array(values)

    archive: List[np.ndarray] = []
    hv_by_step: Dict[int, float] = {}
    for step in steps_sorted:
        if step not in step_vectors:
            continue
        archive.append(step_vectors[step])
        points: np.ndarray = np.array(archive)
        non_dominated: np.ndarray = filter_dominated_maximise(points=points)
        if metric_name == SPEARMAN:
            ref_value: float = WORST_SPEARMAN
        elif metric_name == OB1:
            ref_value = WORST_OB1
        else:
            raise ValueError(f"Unknown HVI metric {metric_name!r}")
        ref_point: np.ndarray = np.full(len(TASKS), ref_value)
        hv_value: float = float(
            exact_hypervolume_maximise(
                points=non_dominated,
                reference_point=ref_point,
            )
        )
        hv_by_step[step] = hv_value
    return hv_by_step


def format_value(*, value: Optional[float], is_best: bool = False) -> str:
    if value is None:
        return "—"
    formatted: str = f"{value:.4f}"
    if is_best:
        formatted += "†"
    return formatted


def build_markdown_table(
    *,
    all_df: pd.DataFrame,
    config_name: str,
    val_name: str,
) -> str:
    config_df: pd.DataFrame = all_df[
        (all_df["config"] == config_name) & (all_df["val"] == val_name)
    ].copy()
    if len(config_df) == 0:
        return f"## {config_name} val={val_name}\n\nNo data found.\n"

    runs: List[int] = sorted(config_df["run"].unique().tolist())
    spearman_hvi: Dict[int, float]
    ob1_hvi: Dict[int, float]
    if config_name == "Single":
        spearman_hvi = {}
        ob1_hvi = {}
    else:
        spearman_hvi = compute_hvi_progression(config_df=config_df, metric_name=SPEARMAN)
        ob1_hvi = compute_hvi_progression(config_df=config_df, metric_name=OB1)

    per_task_summary: pd.DataFrame = (
        config_df.groupby(["step", "task"])[[SPEARMAN, OB1]]
        .agg(["mean", "std"])
        .reset_index()
    )
    per_task_summary.columns = [
        "step",
        "task",
        "task_spearman_mean",
        "task_spearman_std",
        "task_ob1_mean",
        "task_ob1_std",
    ]
    per_run_avg: pd.DataFrame = (
        config_df.groupby(["run", "step"])[[SPEARMAN, OB1]]
        .mean()
        .reset_index()
    )
    per_step_summary: pd.DataFrame = (
        per_run_avg.groupby("step")[[SPEARMAN, OB1]]
        .agg(["mean", "std"])
        .reset_index()
    )
    per_step_summary.columns = [
        "step",
        "spearman_mean",
        "spearman_std",
        "ob1_mean",
        "ob1_std",
    ]
    best_step: int = int(
        per_step_summary.loc[
            per_step_summary["spearman_mean"].idxmax(),
            "step",
        ]
    )

    lines: List[str] = []
    lines.append(f"## {config_name} val={val_name}")
    lines.append("")
    lines.append(f"Runs: {runs}")
    lines.append("")
    lines.append("`†` marks the best `avg_spearman_mean` within this table.")
    lines.append("")
    lines.append(
        "| Step | "
        "fluency_spearman_mean | fluency_spearman_std | fluency_ob1_mean | fluency_ob1_std | "
        "relevance_spearman_mean | relevance_spearman_std | relevance_ob1_mean | relevance_ob1_std | "
        "coherence_spearman_mean | coherence_spearman_std | coherence_ob1_mean | coherence_ob1_std | "
        "consistency_spearman_mean | consistency_spearman_std | consistency_ob1_mean | consistency_ob1_std | "
        "avg_spearman_mean | avg_spearman_std | "
        "avg_ob1_mean | avg_ob1_std | "
        "HVI_spearman | HVI_ob1 |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    steps: List[int] = sorted(per_step_summary["step"].tolist())
    for step in steps:
        task_cells: List[str] = []
        for task_name in TASKS:
            task_row: pd.DataFrame = per_task_summary[
                (per_task_summary["step"] == step)
                & (per_task_summary["task"] == task_name)
            ]
            if len(task_row) == 0:
                task_spearman_mean: Optional[float] = None
                task_spearman_std: Optional[float] = None
                task_ob1_mean: Optional[float] = None
                task_ob1_std: Optional[float] = None
            else:
                task_spearman_mean = float(task_row.iloc[0]["task_spearman_mean"])
                task_spearman_std_raw: float = float(task_row.iloc[0]["task_spearman_std"])
                task_ob1_mean = float(task_row.iloc[0]["task_ob1_mean"])
                task_ob1_std_raw: float = float(task_row.iloc[0]["task_ob1_std"])
                task_spearman_std = (
                    0.0 if np.isnan(task_spearman_std_raw) else task_spearman_std_raw
                )
                task_ob1_std = (
                    0.0 if np.isnan(task_ob1_std_raw) else task_ob1_std_raw
                )
            task_cells.extend(
                [
                    format_value(value=task_spearman_mean),
                    format_value(value=task_spearman_std),
                    format_value(value=task_ob1_mean),
                    format_value(value=task_ob1_std),
                ]
            )

        summary_row: pd.DataFrame = per_step_summary[per_step_summary["step"] == step]
        if len(summary_row) == 0:
            continue
        spearman_mean: float = float(summary_row.iloc[0]["spearman_mean"])
        spearman_std_raw: float = float(summary_row.iloc[0]["spearman_std"])
        ob1_mean: float = float(summary_row.iloc[0]["ob1_mean"])
        ob1_std_raw: float = float(summary_row.iloc[0]["ob1_std"])

        spearman_std: float = 0.0 if np.isnan(spearman_std_raw) else spearman_std_raw
        ob1_std: float = 0.0 if np.isnan(ob1_std_raw) else ob1_std_raw
        hvi_s: Optional[float] = (
            spearman_hvi[step] if step in spearman_hvi else None
        )
        hvi_o: Optional[float] = (
            ob1_hvi[step] if step in ob1_hvi else None
        )

        avg_spearman_cell: str = format_value(
            value=spearman_mean,
            is_best=(step == best_step),
        )
        row_cells: List[str] = [
            str(step),
            *task_cells,
            avg_spearman_cell,
            format_value(value=spearman_std),
            format_value(value=ob1_mean),
            format_value(value=ob1_std),
            format_value(value=hvi_s),
            format_value(value=hvi_o),
        ]
        lines.append("| " + " | ".join(row_cells) + " |")

    lines.append("")
    return "\n".join(lines)


def build_markdown_report(*, all_df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# TextGrad SummEval Report")
    lines.append("")
    lines.append(
        "This report shows per-step summaries for `SpearmanCorrelation` and "
        "`OffByOne` across `val=none` and `val=mae` runs."
    )
    lines.append("")
    lines.append(
        "`avg_spearman_mean` is the mean Spearman score at a given step after "
        "first averaging across the 4 tasks within each run, and then averaging "
        "those run-level averages across all discovered runs for that table."
    )
    lines.append("")
    lines.append(
        "`avg_spearman_std` is the standard deviation of those run-level average "
        "Spearman scores across all discovered runs at that same step."
    )
    lines.append("")
    lines.append(
        "Here, `N` means the number of discovered runs for the current "
        "configuration block."
    )
    lines.append("")
    lines.append("So concretely, for one step:")
    lines.append("")
    lines.append("1. In each run, compute the average Spearman across `fluency`, `relevance`, `coherence`, and `consistency`.")
    lines.append("2. Take those `N` run-level averages.")
    lines.append("3. `avg_spearman_mean` = mean of those N numbers.")
    lines.append("4. `avg_spearman_std` = std of those `N` numbers.")
    lines.append("")
    lines.append("It is not:")
    lines.append("- the average of per-task standard deviations")
    lines.append("- the standard deviation across samples inside a run")
    lines.append("- the standard deviation across tasks")
    lines.append("")
    lines.append(
        "It is specifically cross-run variability of the run-level task-averaged "
        "Spearman."
    )
    lines.append("")
    for val_name in VALS:
        lines.append(f"# val={val_name}")
        lines.append("")
        for config_name in MODES:
            lines.append(
                build_markdown_table(
                    all_df=all_df,
                    config_name=config_name,
                    val_name=val_name,
                )
            )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    all_df: pd.DataFrame = load_all_rows()
    report: str = build_markdown_report(all_df=all_df)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved report: {OUT_PATH}")


if __name__ == "__main__":
    main()
