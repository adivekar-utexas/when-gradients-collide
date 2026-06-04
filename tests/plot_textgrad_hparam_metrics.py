"""Plot TextGrad hyperparameter tuning metrics across all rounds.

Two dashboards:
1. Per-task metric progression for the 3 final runs (from parquet data)
2. Aggregate metric comparison across all R1-R10 rounds (from recorded data)

Produces separate HTML files per metric in tests/e2e_outputs/unified/textgrad_hparam_plots/.
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
OUT_DIR = os.path.join(BASE, "textgrad_hparam_plots")
os.makedirs(OUT_DIR, exist_ok=True)

TASKS = ["fluency", "relevance", "coherence", "consistency"]
METRIC_NAMES = [
    "Accuracy",
    "MAE",
    "OffByOne",
    "OffByTwo",
    "OffByThree",
    "SpearmanCorrelation",
    "KendallTau",
    "PearsonCorrelation",
    "F1",
]
LOWER_IS_BETTER = {"MAE", "LCE"}

TASK_COLORS = {
    "fluency": "#2196F3",
    "relevance": "#FF9800",
    "coherence": "#4CAF50",
    "consistency": "#E91E63",
}

ROUND_COLORS = {
    "R1": "#E53935",
    "R2": "#8E24AA",
    "R3": "#1E88E5",
    "R4": "#43A047",
    "R5": "#F4511E",
    "R6": "#6D4C41",
    "R7": "#00ACC1",
    "R8 (best)": "#FFB300",
    "R9": "#7CB342",
    "R10": "#546E7A",
    "Final avg": "#212121",
}

ROUND_AGGREGATE_DATA = {
    "R1": {
        "desc": "SSC, acc val, bs=12",
        "steps": {
            0: {
                "Acc": 0.421,
                "MAE": 0.898,
                "OB1": 0.752,
                "Spear": 0.284,
                "Kend": 0.259,
            },
            3: {
                "Acc": 0.549,
                "MAE": 0.841,
                "OB1": 0.735,
                "Spear": 0.249,
                "Kend": 0.236,
            },
            6: {
                "Acc": 0.547,
                "MAE": 0.849,
                "OB1": 0.733,
                "Spear": 0.223,
                "Kend": 0.210,
            },
        },
    },
    "R2": {
        "desc": "SSC, ob1 val, bs=12",
        "steps": {
            0: {
                "Acc": 0.422,
                "MAE": 0.894,
                "OB1": 0.755,
                "Spear": 0.284,
                "Kend": 0.260,
            },
            1: {
                "Acc": 0.551,
                "MAE": 0.753,
                "OB1": 0.789,
                "Spear": 0.298,
                "Kend": 0.280,
            },
            3: {
                "Acc": 0.545,
                "MAE": 0.802,
                "OB1": 0.758,
                "Spear": 0.294,
                "Kend": 0.277,
            },
        },
    },
    "R3": {
        "desc": "CCC, ob1 val, bs=6",
        "steps": {
            0: {
                "Acc": 0.420,
                "MAE": 0.897,
                "OB1": 0.755,
                "Spear": 0.285,
                "Kend": 0.260,
            },
            1: {
                "Acc": 0.549,
                "MAE": 0.820,
                "OB1": 0.744,
                "Spear": 0.222,
                "Kend": 0.208,
            },
            3: {
                "Acc": 0.472,
                "MAE": 0.868,
                "OB1": 0.763,
                "Spear": 0.200,
                "Kend": 0.188,
            },
        },
    },
    "R4": {
        "desc": "SSC, ob1 val, bs=4",
        "steps": {
            0: {
                "Acc": 0.427,
                "MAE": 0.890,
                "OB1": 0.755,
                "Spear": 0.296,
                "Kend": 0.271,
            },
            1: {
                "Acc": 0.511,
                "MAE": 0.859,
                "OB1": 0.745,
                "Spear": 0.235,
                "Kend": 0.220,
            },
            3: {
                "Acc": 0.540,
                "MAE": 0.837,
                "OB1": 0.739,
                "Spear": 0.258,
                "Kend": 0.242,
            },
            6: {
                "Acc": 0.540,
                "MAE": 0.857,
                "OB1": 0.734,
                "Spear": 0.212,
                "Kend": 0.200,
            },
        },
    },
    "R5": {
        "desc": "SCC, ob1 val, bs=4",
        "steps": {
            0: {
                "Acc": 0.421,
                "MAE": 0.894,
                "OB1": 0.755,
                "Spear": 0.293,
                "Kend": 0.268,
            },
            3: {
                "Acc": 0.547,
                "MAE": 0.868,
                "OB1": 0.723,
                "Spear": 0.224,
                "Kend": 0.212,
            },
            6: {
                "Acc": 0.572,
                "MAE": 0.737,
                "OB1": 0.785,
                "Spear": 0.177,
                "Kend": 0.166,
            },
        },
    },
    "R6": {
        "desc": "SSC, spearman val, bs=3",
        "steps": {
            0: {
                "Acc": 0.420,
                "MAE": 0.899,
                "OB1": 0.752,
                "Spear": 0.290,
                "Kend": 0.264,
            },
            3: {
                "Acc": 0.419,
                "MAE": 0.899,
                "OB1": 0.753,
                "Spear": 0.284,
                "Kend": 0.259,
            },
            6: {
                "Acc": 0.415,
                "MAE": 0.907,
                "OB1": 0.750,
                "Spear": 0.275,
                "Kend": 0.251,
            },
        },
    },
    "R7": {
        "desc": "SSC, ob1 val, bs=3, low temps",
        "steps": {
            0: {
                "Acc": 0.422,
                "MAE": 0.896,
                "OB1": 0.753,
                "Spear": 0.290,
                "Kend": 0.265,
            },
            3: {
                "Acc": 0.538,
                "MAE": 0.820,
                "OB1": 0.752,
                "Spear": 0.266,
                "Kend": 0.250,
            },
            6: {
                "Acc": 0.448,
                "MAE": 0.873,
                "OB1": 0.767,
                "Spear": 0.265,
                "Kend": 0.247,
            },
        },
    },
    "R8 (best)": {
        "desc": "SSC, ob1 val, MAE loss, grad=0.3 opt=0.7",
        "steps": {
            0: {
                "Acc": 0.424,
                "MAE": 0.894,
                "OB1": 0.752,
                "Spear": 0.291,
                "Kend": 0.266,
            },
            3: {
                "Acc": 0.544,
                "MAE": 0.847,
                "OB1": 0.737,
                "Spear": 0.225,
                "Kend": 0.213,
            },
            6: {
                "Acc": 0.322,
                "MAE": 0.878,
                "OB1": 0.835,
                "Spear": 0.290,
                "Kend": 0.261,
            },
        },
    },
    "R9": {
        "desc": "SSC, ob1 val, MAE loss, bs=4, grad=0.3 opt=0.7",
        "steps": {
            0: {
                "Acc": 0.423,
                "MAE": 0.894,
                "OB1": 0.752,
                "Spear": 0.285,
                "Kend": 0.261,
            },
            3: {
                "Acc": 0.543,
                "MAE": 0.845,
                "OB1": 0.741,
                "Spear": 0.231,
                "Kend": 0.219,
            },
            6: {
                "Acc": 0.543,
                "MAE": 0.844,
                "OB1": 0.741,
                "Spear": 0.235,
                "Kend": 0.222,
            },
        },
    },
    "R10": {
        "desc": "SSC, ob1 val, MAE loss, bs=3, val=60, loss=0.5",
        "steps": {
            0: {
                "Acc": 0.421,
                "MAE": 0.898,
                "OB1": 0.752,
                "Spear": 0.283,
                "Kend": 0.258,
            },
            3: {
                "Acc": 0.564,
                "MAE": 0.777,
                "OB1": 0.770,
                "Spear": 0.256,
                "Kend": 0.241,
            },
            6: {
                "Acc": 0.563,
                "MAE": 0.779,
                "OB1": 0.770,
                "Spear": 0.244,
                "Kend": 0.230,
            },
        },
    },
}

AGG_METRIC_MAP = {
    "Accuracy": "Acc",
    "MAE": "MAE",
    "OffByOne": "OB1",
    "SpearmanCorrelation": "Spear",
    "KendallTau": "Kend",
}

FINAL_RUN_CONFIGS = [
    ("R8-final-1", "final_run_1"),
    ("R8-final-2", "final_run_2"),
    ("R8-final-3", "final_run_3"),
]


def compute_per_task_metrics_from_parquet():
    """Read parquet files from the 3 final runs and compute per-task metrics."""
    rows = []
    for cfg_label, cfg_dir in FINAL_RUN_CONFIGS:
        run_dir = os.path.join(BASE, cfg_dir, "run")
        if not os.path.isdir(run_dir):
            print(f"  SKIP {cfg_label}: {run_dir} not found")
            continue
        for step in range(0, 13):
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


def build_aggregate_dataframe():
    """Convert the hardcoded round aggregate data into a DataFrame."""
    rows = []
    for round_name, round_data in ROUND_AGGREGATE_DATA.items():
        for step, metrics in round_data["steps"].items():
            for agg_metric, agg_key in AGG_METRIC_MAP.items():
                if agg_key in metrics:
                    rows.append(
                        {
                            "round": round_name,
                            "desc": round_data["desc"],
                            "step": step,
                            "metric": agg_metric,
                            "value": metrics[agg_key],
                        }
                    )
    return pd.DataFrame(rows)


def make_per_task_dashboard(all_df, metric_name):
    """Per-task metric grid: one plot per final run, 4 task lines each."""
    is_lower = metric_name in LOWER_IS_BETTER
    direction = "lower=better" if is_lower else "higher=better"
    plots = []
    for cfg_label, _ in FINAL_RUN_CONFIGS:
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
            best_step = best_val = None
            if len(best_opt) > 0:
                best_idx = (
                    best_opt[metric_name].idxmin()
                    if is_lower
                    else best_opt[metric_name].idxmax()
                )
                best_step = best_opt.loc[best_idx, "step"]
                best_val = best_opt.loc[best_idx, metric_name]
            line = task_df.hvplot.line(
                x="step",
                y=metric_name,
                label=task,
                color=TASK_COLORS[task],
                line_width=2,
                width=500,
                height=350,
            )
            scatter = task_df.hvplot.scatter(
                x="step", y=metric_name, color=TASK_COLORS[task], size=50
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
                hline = hv.HLine(baseline_vals[task]).opts(
                    color=TASK_COLORS[task], line_dash="dashed", line_width=1, alpha=0.4
                )
                combined = combined * hline
            overlay = combined if overlay is None else overlay * combined
        if overlay is not None:
            overlay = overlay.opts(
                title=cfg_label,
                xlabel="Step",
                ylabel=metric_name,
                show_grid=True,
                legend_position="bottom_right" if not is_lower else "top_right",
                fontsize={"title": 11, "labels": 10, "ticks": 9, "legend": 8},
            )
            plots.append(overlay)
    if len(plots) == 0:
        return None
    layout = hv.Layout(plots).cols(3)
    layout = layout.opts(
        hv.opts.Layout(
            title=f"TextGrad {metric_name} per task ({direction}) — ★ = best step"
        )
    )
    return layout


def make_round_comparison_dashboard(agg_df, metric_name):
    """All R1-R10 rounds on one plot per metric, showing aggregate progression."""
    agg_key = AGG_METRIC_MAP.get(metric_name)
    if agg_key is None:
        return None
    mdf = agg_df[agg_df["metric"] == metric_name].copy()
    if len(mdf) == 0:
        return None
    is_lower = metric_name in LOWER_IS_BETTER
    direction = "lower=better" if is_lower else "higher=better"
    overlay = None
    for round_name in ROUND_AGGREGATE_DATA.keys():
        rdf = mdf[mdf["round"] == round_name].sort_values("step")
        if len(rdf) == 0:
            continue
        color = ROUND_COLORS.get(round_name, "#888888")
        lw = 3 if "(best)" in round_name else 1.5
        line = rdf.hvplot.line(
            x="step",
            y="value",
            label=round_name,
            color=color,
            line_width=lw,
            width=900,
            height=500,
        )
        scatter = rdf.hvplot.scatter(
            x="step",
            y="value",
            color=color,
            size=50 if "(best)" not in round_name else 100,
        )
        combined = line * scatter
        overlay = combined if overlay is None else overlay * combined
    if overlay is None:
        return None
    overlay = overlay.opts(
        title=f"TextGrad R1-R10: {metric_name} ({direction})",
        xlabel="Optimization Step",
        ylabel=f"Avg {metric_name}",
        show_grid=True,
        legend_position="right",
        fontsize={"title": 13, "labels": 11, "ticks": 10, "legend": 9},
    )
    return overlay


print("=" * 80)
print("TextGrad Hyperparameter Tuning — Plot Generator")
print("=" * 80)

print("\n1. Computing per-task metrics from final run parquets...")
per_task_df = compute_per_task_metrics_from_parquet()
print(f"   Rows: {len(per_task_df)}, Configs: {per_task_df['config'].nunique()}")

print("\n2. Building aggregate round comparison data...")
agg_df = build_aggregate_dataframe()
print(f"   Rows: {len(agg_df)}, Rounds: {agg_df['round'].nunique()}")

print("\n3. Generating per-task dashboards (final runs)...")
for metric_name in METRIC_NAMES:
    layout = make_per_task_dashboard(per_task_df, metric_name)
    if layout is None:
        print(f"   SKIP {metric_name}: no data")
        continue
    out_path = os.path.join(OUT_DIR, f"textgrad_pertask_{metric_name.lower()}")
    renderer = hv.renderer("bokeh")
    renderer.save(layout, out_path)
    print(f"   Saved: {out_path}.html")

print("\n4. Generating round comparison dashboards (R1-R10)...")
for metric_name in AGG_METRIC_MAP.keys():
    plot = make_round_comparison_dashboard(agg_df, metric_name)
    if plot is None:
        print(f"   SKIP {metric_name}: no aggregate data")
        continue
    out_path = os.path.join(OUT_DIR, f"textgrad_rounds_{metric_name.lower()}")
    renderer = hv.renderer("bokeh")
    renderer.save(plot, out_path)
    print(f"   Saved: {out_path}.html")

csv_path = os.path.join(OUT_DIR, "textgrad_per_task_metrics.csv")
per_task_df.to_csv(csv_path, index=False)
agg_csv = os.path.join(OUT_DIR, "textgrad_round_aggregates.csv")
agg_df.to_csv(agg_csv, index=False)
print(f"\nCSV data: {csv_path}")
print(f"          {agg_csv}")
print(f"\nAll plots saved to: {OUT_DIR}/")
print("Open any .html file in a browser to interact.")
