"""Plot per-task metric progression for all GPO hparam configs.

Creates an interactive HTML dashboard with hvplot (bokeh backend):
- One plot per config
- Each plot shows all 4 tasks as separate line+scatter traces
- Separate dashboard per metric (MAE, Spearman, etc.)
- Grid layout using .cols(3)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "expt"))

import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas

hv.extension("bokeh")

from prompt_moo.metrics import Metric
from bokeh.io import save, output_file
from bokeh.resources import CDN

BASE = os.path.join(os.path.dirname(__file__), "e2e_outputs", "unified")
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "e2e_outputs", "unified", "hparam_plots"
)
os.makedirs(OUT_DIR, exist_ok=True)

EVAL_STEPS = [0, 1, 3, 6, 9, 12]
TASKS = ["fluency", "relevance", "coherence", "consistency"]
METRIC_NAMES = [
    "MAE",
    "Accuracy",
    "SpearmanCorrelation",
    "PearsonCorrelation",
    "OffByOne",
    "OffByTwo",
    "F1",
    "KendallTau",
]
LOWER_IS_BETTER = {"MAE", "LCE"}

CONFIGS = [
    ("R0: MAE k=3 d=3 rel", "all_gpo_mae"),
    ("R1: acc k=5 d=3 rel", "gpo_hparam_round1"),
    ("R2: obo k=5 iss35 d=3 rel", "gpo_hparam_round2"),
    ("R3: obo k=7 iss15 d=3 rel", "gpo_hparam_round3"),
    ("R4: obo k=7 iss35 d=3 rel", "gpo_hparam_round4"),
    ("R5: obo k=5 bs24 d=3 rel", "gpo_hparam_round5"),
    ("R6: obo k=5 d=5 rel", "gpo_hparam_round6"),
    ("R7: spear k=5 d=5 rel", "gpo_hparam_round7"),
    ("R8: obo k=5 d=5 imp", "gpo_hparam_round8"),
    ("R9: obo k=6 iss40 d=5 imp", "gpo_hparam_round9"),
    ("R10: mae k=5 d=5 imp", "gpo_hparam_round10"),
    ("R8-run2", "gpo_hparam_best_run2"),
    ("R8-run3", "gpo_hparam_best_run3"),
]

TASK_COLORS = {
    "fluency": "#2196F3",
    "relevance": "#FF9800",
    "coherence": "#4CAF50",
    "consistency": "#E91E63",
}


def compute_all_per_task_metrics():
    """Read all parquets and compute per-task metrics for every config x step."""
    rows = []
    for cfg_label, cfg_dir in CONFIGS:
        run_dir = os.path.join(BASE, cfg_dir, "run")
        if not os.path.isdir(run_dir):
            print(f"  SKIP {cfg_label}: {run_dir} not found")
            continue
        for step in EVAL_STEPS:
            pq_path = os.path.join(run_dir, f"eval_step_{step}.parquet")
            if not os.path.exists(pq_path):
                continue
            df = pd.read_parquet(pq_path)
            for task in TASKS:
                gt_col = f"gt_{task}"
                pred_col = f"pred_{task}"
                if gt_col not in df.columns or pred_col not in df.columns:
                    continue
                valid = df[gt_col].notna() & df[pred_col].notna()
                y_true = df.loc[valid, gt_col].tolist()
                y_pred = df.loc[valid, pred_col].tolist()
                row = {"config": cfg_label, "step": step, "task": task}
                for mname in METRIC_NAMES:
                    try:
                        mcls = Metric.get_subclass(mname)
                        val = mcls.compute(y_true=y_true, y_pred=y_pred)
                        row[mname] = val
                    except Exception:
                        row[mname] = None
                rows.append(row)
    return pd.DataFrame(rows)


def make_metric_dashboard(all_df, metric_name):
    """Create a grid of plots for one metric, one plot per config."""
    is_lower_better = metric_name in LOWER_IS_BETTER
    direction_label = "lower=better" if is_lower_better else "higher=better"

    plots = []
    for cfg_label, _ in CONFIGS:
        cfg_df = all_df[all_df["config"] == cfg_label].copy()
        if len(cfg_df) == 0:
            continue

        baseline_vals = {}
        for task in TASKS:
            s0 = cfg_df[(cfg_df["task"] == task) & (cfg_df["step"] == 0)]
            if len(s0) > 0 and s0[metric_name].notna().any():
                baseline_vals[task] = s0[metric_name].iloc[0]

        overlay = None
        for task in TASKS:
            task_df = cfg_df[cfg_df["task"] == task].dropna(subset=[metric_name]).copy()
            if len(task_df) == 0:
                continue
            task_df = task_df.sort_values("step")

            best_opt = task_df[task_df["step"] > 0]
            if len(best_opt) > 0:
                if is_lower_better:
                    best_idx = best_opt[metric_name].idxmin()
                else:
                    best_idx = best_opt[metric_name].idxmax()
                best_step = best_opt.loc[best_idx, "step"]
                best_val = best_opt.loc[best_idx, metric_name]
            else:
                best_step = None
                best_val = None

            line = task_df.hvplot.line(
                x="step",
                y=metric_name,
                label=task,
                color=TASK_COLORS[task],
                line_width=2,
                width=420,
                height=300,
            )
            scatter = task_df.hvplot.scatter(
                x="step",
                y=metric_name,
                color=TASK_COLORS[task],
                size=60,
            )

            combined = line * scatter

            if best_step is not None:
                best_df = pd.DataFrame([{"step": best_step, metric_name: best_val}])
                star = best_df.hvplot.scatter(
                    x="step",
                    y=metric_name,
                    color=TASK_COLORS[task],
                    size=180,
                    marker="star",
                )
                combined = combined * star

            if task in baseline_vals:
                bl_val = baseline_vals[task]
                hline = hv.HLine(bl_val).opts(
                    color=TASK_COLORS[task],
                    line_dash="dashed",
                    line_width=1,
                    alpha=0.4,
                )
                combined = combined * hline

            overlay = combined if overlay is None else overlay * combined

        if overlay is not None:
            title_short = cfg_label
            overlay = overlay.opts(
                title=title_short,
                xlabel="Step",
                ylabel=metric_name,
                show_grid=True,
                legend_position="bottom_right" if not is_lower_better else "top_right",
                fontsize={"title": 10, "labels": 9, "ticks": 8, "legend": 8},
            )
            plots.append(overlay)

    if len(plots) == 0:
        return None

    layout = hv.Layout(plots).cols(3)
    layout = layout.opts(
        hv.opts.Layout(
            title=f"{metric_name} per task ({direction_label}) — ★ = best optimization step",
        )
    )
    return layout


print("Computing all per-task metrics from parquet files...")
all_df = compute_all_per_task_metrics()
print(f"  Total rows: {len(all_df)}")
print(f"  Configs: {all_df['config'].nunique()}")
print(f"  Metrics: {METRIC_NAMES}")
print()

for metric_name in METRIC_NAMES:
    print(f"Building dashboard for {metric_name}...")
    layout = make_metric_dashboard(all_df, metric_name)
    if layout is None:
        print(f"  SKIP {metric_name}: no data")
        continue

    out_path = os.path.join(OUT_DIR, f"gpo_hparam_{metric_name.lower()}.html")
    renderer = hv.renderer("bokeh")
    renderer.save(layout, out_path)
    print(f"  Saved: {out_path}")

csv_path = os.path.join(OUT_DIR, "all_per_task_metrics.csv")
all_df.to_csv(csv_path, index=False)
print(f"\nRaw data CSV: {csv_path}")

print(f"\nAll plots saved to: {OUT_DIR}/")
print("Open any .html file in a browser to interact with the plots.")
