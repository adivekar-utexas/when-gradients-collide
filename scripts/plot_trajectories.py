"""Plot per-task metric progression for TextGrad mode x validation_metric experiments.

Creates interactive HTML dashboards (one per dataset per metric) using Panel + HoloViews:
- pn.Row/Column nesting: tight shared axis labels via HTML panes
- 4-column GridSpec: one subplot per config (mode x val_metric)
- Each subplot: per-task line+scatter, HV indicator, star best step, dashed baseline
- Single shared legend bar at bottom, horizontally centered
- No toolbars, no per-subplot legends, no per-subplot axis labels
- ggplot2-style font (system-ui / San Francisco / Segoe UI)
"""

import os, re
import sys
from typing import Dict, List, Optional, Set, Tuple


import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas  # noqa: F401
import panel as pn
from pymoo.indicators.hv import HV as HVIndicator
import moocore
from bokeh.models import Range1d, NumeralTickFormatter, BasicTicker

hv.extension("bokeh")
pn.extension()

from when_gradients_collide.metrics import Metric

BASE: str = os.path.join(os.path.dirname(__file__), "e2e_outputs", "unified", "4")
print(BASE)
OUT_DIR: str = os.path.join(BASE, "textgrad_mode_plots")
os.makedirs(OUT_DIR, exist_ok=True)

LOWER_IS_BETTER: Set[str] = {"MAE", "LCE"}

METRIC_NAMES: List[str] = [
    "Accuracy",
    "MAE",
    "SpearmanCorrelation",
    "KendallTau",
    "OffByOne",
    "OffByTwo",
    "OffByThree",
    "F1",
]

THEORETICAL_WORST: Dict[str, float] = {
    "Accuracy": 0.0,
    "F1": 0.0,
    "SpearmanCorrelation": -1.0,
    "KendallTau": -1.0,
    "OffByOne": 0.0,
    "OffByTwo": 0.0,
    "OffByThree": 0.0,
}
DATASET_MAE_WORST: Dict[str, float] = {"SummEval": 4.0, "BRIGHTER": 3.0}

COLOR_PALETTES: Dict[str, Dict[str, Dict[str, str]]] = {
    "alternating_light": {
        "SummEval": {
            "fluency": "#8fd7d7",
            "relevance": "#ff8ca1",
            "coherence": "#bdd373",
            "consistency": "#ffcd8e",
        },
        # "BRIGHTER": {
        #     "anger": "#ff8ca1",
        #     "fear": "#8fd7d7",
        #     "joy": "#bdd373",
        #     "sadness": "#c4b7ea",
        #     "surprise": "#ffcd8e",
        # },
    },
    "tableau_medium": {
        "SummEval": {
            "fluency": "#729ece",
            "relevance": "#ff9e4a",
            "coherence": "#67bf5c",
            "consistency": "#ed665d",
        },
        # "BRIGHTER": {
        #     "anger": "#ed665d",
        #     "fear": "#729ece",
        #     "joy": "#ff9e4a",
        #     "sadness": "#ad8bc9",
        #     "surprise": "#67bf5c",
        # },
    },
    "material_design": {
        "SummEval": {
            "fluency": "#2196F3",
            "relevance": "#FF9800",
            "coherence": "#4CAF50",
            "consistency": "#E91E63",
        },
        # "BRIGHTER": {
        #     "anger": "#E91E63",
        #     "fear": "#9C27B0",
        #     "joy": "#FF9800",
        #     "sadness": "#2196F3",
        #     "surprise": "#4CAF50",
        # },
    },
}
SELECTED_PALETTE: str = "alternating_light"

DATASETS: List[Dict] = [
    {
        "name": "SummEval",
        "prefix": ["TextGrad-SummEval-29May2026"],
        "tasks": ["fluency", "relevance", "coherence", "consistency"],
        "task_colors": COLOR_PALETTES[SELECTED_PALETTE]["SummEval"],
        "max_step": 12,
    },
    # {
    #     "name": "BRIGHTER",
    #     "prefix": "TextGrad-BRIGHTER-02Apr2026",
    #     "tasks": ["anger", "fear", "joy", "sadness", "surprise"],
    #     "task_colors": COLOR_PALETTES[SELECTED_PALETTE]["BRIGHTER"],
    #     "max_step": 12,
    # },
]

MODE_ORDER: List[str] = ["CCC", "SCC", "SSC", "SSS"] # "SSS", , "CCC", "SSC",
VAL_ORDER: List[str] = [
    # "ob1",
    "mae",
    # "spearman",
    # "none",
]
SINGLE_TASK_MODE: str = "CCC"
SINGLE_TASK_LABEL_PREFIX: str = "Single"
CONFIGS: List[Tuple[str, Optional[str]]] = []
for val in VAL_ORDER:
    CONFIGS.append((f"{SINGLE_TASK_LABEL_PREFIX} val={val}", None))
    for mode in MODE_ORDER:
        CONFIGS.append((f"{mode} val={val}", f"{mode}-val={val}"))

BOKEH_FONT: str = "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
CSS_FONT: str = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
PLOT_W: int = 280
PLOT_H: int = 220
SHOW_HV: bool = True
SHOW_TASK_SYMBOLS: bool = False
GRID_COL_GAP: int = 20
GRID_ROW_GAP: int = 20
GRID_COLS: int = len(MODE_ORDER) + 1
GRID_ROWS: int = len(VAL_ORDER)
GRID_W: int = PLOT_W * GRID_COLS + (GRID_COLS - 1) * GRID_COL_GAP
GRID_H: int = PLOT_H * GRID_ROWS + (GRID_ROWS - 1) * GRID_ROW_GAP
YLABEL_W: int = 35
XLABEL_H: int = 50
LEGEND_W = PLOT_W           # exactly one subplot width
LEGEND_H = int(PLOT_H * 0.55)  # enough to fit multiple rows from ncols=3
X_PAD: float = 0.4

def _extract_run_num(run_dir: str) -> int:
    """
    run_dir is something like:
      ...\\TextGrad-...-run_3\\run
      .../TextGrad-...-run_3/run
    """
    # normalize: parent folder is "...-run_3"
    parent = os.path.basename(os.path.normpath(os.path.dirname(run_dir)))
    # m = re.search(r"-run_(\d+)$", parent)
    m = re.search(r"-run[_-]?(\d+)$", parent)
    if not m:
        raise ValueError(f"Could not parse run number from run_dir={run_dir} (parent={parent})")
    return int(m.group(1))


def make_style_hook(
    *,
    hv_range_min: Optional[float] = None,
    hv_range_max: Optional[float] = None,
    hv_tick_format: str = "0.0",
    bands: Optional[List[Dict]] = None,
):
    """Return a bokeh hook that styles the subplot and fixes the secondary y-axis.

    When ``hv_range_min`` / ``hv_range_max`` are provided the hook pins the
    ``extra_y_ranges['HV']`` to a fixed ``Range1d`` so all subplots in a grid
    share the same Hypervolume Indicator scale.  The hook also hides the y-grid
    (horizontal gridlines create visual confusion when two y-axes have different
    scales) while keeping the x-grid (vertical lines) for readability.
    ``hv_tick_format`` is a Numeral.js format string applied to the right y-axis
    so that tick labels have a uniform number of decimal places.
    ``bands`` is a list of dicts with keys ``steps``, ``mins``, ``maxs``,
    ``color`` to render as filled patches on the default y-range.
    """

    def _hook(plot, element) -> None:
        from bokeh.models import Patch
        p = plot.state
        for legend in p.legend:
            legend.visible = False
        p.toolbar_location = None
        p.title.text_font = BOKEH_FONT
        p.title.text_font_size = "18px"
        p.title.text_font_style = "normal"

        stray_keys: list = [k for k in p.extra_y_ranges if k != "HV"]
        for k in stray_keys:
            del p.extra_y_ranges[k]
        p.right = [
            ax
            for ax in p.right
            if getattr(ax, "y_range_name", None) == "HV"
            or not hasattr(ax, "y_range_name")
        ]

        if hv_range_min is not None and hv_range_max is not None:
            if "HV" in p.extra_y_ranges:
                p.extra_y_ranges["HV"] = Range1d(start=hv_range_min, end=hv_range_max)

        for g in p.grid:
            if g.dimension == 0:
                g.visible = False
            elif g.dimension == 1:
                if g.axis is not None and getattr(g.axis, "y_range_name", "default") == "default":
                    g.visible = True
                    g.grid_line_color = "#e5e5e5"
                    g.grid_line_alpha = 0.8
                    g.ticker = BasicTicker(desired_num_ticks=6)
                else:
                    g.visible = False

        for axis in p.xaxis + p.yaxis:
            axis.major_label_text_font = BOKEH_FONT
            axis.major_label_text_font_size = "11pt"
            axis.axis_label_text_font = BOKEH_FONT
            axis.axis_label_text_font_size = "13pt"
            axis.axis_label = ""

        for ax in p.yaxis:
            if ax.y_range_name == "HV":
                ax.formatter = NumeralTickFormatter(format=hv_tick_format)
                ax.ticker = BasicTicker(desired_num_ticks=5)
            elif ax.y_range_name == "default":
                ax.formatter = NumeralTickFormatter(format="0.00")
                ax.ticker = BasicTicker(desired_num_ticks=6)

        if bands is not None:
            from bokeh.models import ColumnDataSource
            for bd in bands:
                steps: list = bd["steps"]
                xs: list = list(steps) + list(reversed(steps))
                ys: list = list(bd["maxs"]) + list(reversed(bd["mins"]))
                source = ColumnDataSource(data=dict(x=xs, y=ys))
                patch = p.patch(
                    "x", "y", source=source,
                    fill_color=bd["color"], fill_alpha=0.15,
                    line_color=None, line_width=0,
                    y_range_name="default",
                    level="underlay",
                )

    return _hook


def legend_hook(plot, element) -> None:
    p = plot.state
    for attr in (
        "min_border_top",
        "min_border_bottom",
        "min_border_left",
        "min_border_right",
    ):
        setattr(p, attr, 0)
    p.outline_line_alpha = 0
    p.background_fill_alpha = 0
    p.border_fill_alpha = 0
    for legend in p.legend:
        legend.label_text_font = BOKEH_FONT
        legend.label_text_font_size = "12pt"
        legend.orientation = "horizontal"
        legend.location = "center"
        legend.glyph_width = 20
        legend.glyph_height = 20
        legend.spacing = 15
        legend.padding = 5
        legend.margin = 0


def _discover_run_dirs(base_dir: str, prefix: str, cfg_base: str) -> List[str]:
    """Find all run_N directories for a config base like 'SSC-val=mae'."""
    import glob
    pattern: str = os.path.join(base_dir, f"{prefix}-{cfg_base}-run_*", "run")
    return sorted(glob.glob(pattern))


def _load_parquet_rows(
    *,
    run_dir: str,
    cfg_label: str,
    run_id: int,
    tasks: List[str],
    max_step: int,
) -> List[Dict]:
    rows: List[Dict] = []
    for step in range(0, max_step + 1):
        pq_path: str = os.path.join(run_dir, f"eval_step_{step}.parquet")
        if not os.path.exists(pq_path):
            continue
        df: pd.DataFrame = pd.read_parquet(pq_path)
        for task in tasks:
            gt_col: str = f"gt_{task}"
            pred_col: str = f"pred_{task}"
            if gt_col not in df.columns or pred_col not in df.columns:
                continue
            gt = pd.to_numeric(df[gt_col], errors="coerce")
            pred = pd.to_numeric(df[pred_col], errors="coerce")
            valid = gt.notna() & pred.notna()
            y_true: List = gt[valid].tolist()
            y_pred: List = pred[valid].tolist()
            if len(y_true) == 0:
                continue
            row: Dict = {
                "config": cfg_label, "run": run_id,
                "step": step, "task": task,
            }
            for mname in METRIC_NAMES:
                try:
                    mcls = Metric.get_subclass(mname)
                    val = mcls.compute(y_true=y_true, y_pred=y_pred)
                    row[mname] = val
                except Exception:
                    row[mname] = None
            rows.append(row)
    return rows


def compute_all_per_task_metrics(
    *,
    prefix: str,
    tasks: List[str],
    max_step: int,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for cfg_label, cfg_base in CONFIGS:
        if cfg_base is not None:
            run_dirs: List[str] = _discover_run_dirs(BASE, prefix, cfg_base)
            if len(run_dirs) == 0:
                print(f"  SKIP {cfg_label}: no runs found for {prefix}-{cfg_base}-run_*")
                continue
            for rd in run_dirs:
                run_num: int = _extract_run_num(rd)
                rows.extend(_load_parquet_rows(
                    run_dir=rd, cfg_label=cfg_label, run_id=run_num,
                    tasks=tasks, max_step=max_step,
                ))
        else:
            val_metric: str = cfg_label.split("val=")[1]
            for task in tasks:
                st_dirs: List[str] = _discover_run_dirs(
                    BASE, prefix,
                    f"{SINGLE_TASK_MODE}-val={val_metric}-task={task}",
                )
                if len(st_dirs) == 0:
                    print(f"  SKIP {cfg_label}/{task}: no runs found")
                    continue
                for rd in st_dirs:
                    run_num = int(rd.split("-run_")[-1].split("/")[0])
                    rows.extend(_load_parquet_rows(
                        run_dir=rd, cfg_label=cfg_label, run_id=run_num,
                        tasks=[task], max_step=max_step,
                    ))
    return pd.DataFrame(rows)


def compute_hypervolume_per_step(
    *,
    cfg_df: pd.DataFrame,
    metric_name: str,
    tasks: List[str],
    is_lower_better: bool,
    dataset_name: str,
) -> pd.DataFrame:
    """Compute accumulated Pareto-set hypervolume indicator at each step.

    At each optimization step, the per-task metric vector is added to an
    archive.  The archive is filtered to its non-dominated subset, and the
    hypervolume of that subset is computed against the theoretical worst-case
    reference point.  Because adding a point can only grow or maintain the
    non-dominated region, the resulting HV series is monotonically
    non-decreasing — upward movement means progress, flat means stagnation.

    Uses ``moocore.filter_dominated`` + ``moocore.hypervolume`` with the
    ``maximise`` flag so no manual negation is needed.
    """
    steps_sorted: List[int] = sorted(cfg_df["step"].unique())
    if len(steps_sorted) == 0:
        return pd.DataFrame(columns=["step", "HV"])

    if metric_name == "MAE":
        worst: float = DATASET_MAE_WORST[dataset_name]
    elif metric_name in THEORETICAL_WORST:
        worst = THEORETICAL_WORST[metric_name]
    else:
        worst = 0.0

    num_tasks: int = len(tasks)
    ref_point: np.ndarray = np.full(num_tasks, worst)
    maximise: bool = not is_lower_better

    step_vectors: Dict[int, np.ndarray] = {}
    for step in steps_sorted:
        step_df: pd.DataFrame = cfg_df[cfg_df["step"] == step]
        vals: List[float] = []
        for task in tasks:
            tr: pd.DataFrame = step_df[step_df["task"] == task]
            if len(tr) > 0 and tr[metric_name].notna().any():
                vals.append(float(tr[metric_name].mean()))
            else:
                vals.append(np.nan)
        if not any(np.isnan(v) for v in vals):
            step_vectors[step] = np.array(vals)

    if len(step_vectors) < 2:
        return pd.DataFrame(columns=["step", "HV"])

    archive: List[np.ndarray] = []
    hv_rows: List[Dict] = []
    for step in steps_sorted:
        if step not in step_vectors:
            continue
        archive.append(step_vectors[step])
        points: np.ndarray = np.array(archive)
        nd: np.ndarray = moocore.filter_dominated(points, maximise=maximise)
        hv_val: float = float(moocore.hypervolume(nd, ref=ref_point, maximise=maximise))
        hv_rows.append({"step": step, "HV": hv_val})

    return pd.DataFrame(hv_rows)


def build_subplot(
    *,
    cfg_df: pd.DataFrame,
    cfg_label: str,
    metric_name: str,
    tasks: List[str],
    task_colors: Dict[str, str],
    is_lower_better: bool,
    dataset_name: str,
    max_step: int,
    show_hv: bool = True,
    hv_range_min: Optional[float] = None,
    hv_range_max: Optional[float] = None,
    hv_tick_format: str = "0.0",
) -> Optional[hv.Overlay]:
    if len(cfg_df) == 0:
        return None

    baseline_vals: Dict[str, float] = {}
    for task in tasks:
        s0: pd.DataFrame = cfg_df[(cfg_df["task"] == task) & (cfg_df["step"] == 0)]
        if len(s0) > 0 and s0[metric_name].notna().any():
            baseline_vals[task] = s0[metric_name].iloc[0]

    band_data: List[Dict] = []
    overlay: Optional[hv.Overlay] = None
    for task in tasks:
        task_raw: pd.DataFrame = (
            cfg_df[cfg_df["task"] == task]
            .dropna(subset=[metric_name])
            .copy()
        )
        if len(task_raw) == 0:
            continue

        task_agg: pd.DataFrame = (
            task_raw.groupby("step")[metric_name]
            .agg(["mean", "min", "max"])
            .reset_index()
            .rename(columns={"mean": metric_name, "min": "_min", "max": "_max"})
            .sort_values("step")
        )
        has_band: bool = (task_agg["_max"] - task_agg["_min"]).sum() > 1e-9

        if has_band:
            band_data.append({
                "steps": task_agg["step"].tolist(),
                "mins": task_agg["_min"].tolist(),
                "maxs": task_agg["_max"].tolist(),
                "color": task_colors[task],
            })

        best_step: Optional[int] = None
        best_val: Optional[float] = None
        if len(task_agg) > 0:
            best_idx = (
                task_agg[metric_name].idxmin()
                if is_lower_better
                else task_agg[metric_name].idxmax()
            )
            best_step = task_agg.loc[best_idx, "step"]
            best_val = task_agg.loc[best_idx, metric_name]

        line = task_agg.hvplot.line(
            x="step",
            y=metric_name,
            label=task,
            color=task_colors[task],
            line_width=1.5,
            frame_width=PLOT_W,
            height=PLOT_H,
        )
        combined = line

        if SHOW_TASK_SYMBOLS:
            scatter = task_agg.hvplot.scatter(
                x="step",
                y=metric_name,
                color=task_colors[task],
                size=70,
            )
            combined = combined * scatter

            if best_step is not None:
                star = pd.DataFrame(
                    [{"step": best_step, metric_name: best_val}]
                ).hvplot.scatter(
                    x="step",
                    y=metric_name,
                    color=task_colors[task],
                    size=200,
                    marker="star",
                )
                combined = combined * star

        overlay = combined if overlay is None else overlay * combined

    if overlay is None:
        return None

    avg_rows: List[Dict] = []
    run_avg: pd.DataFrame = (
        cfg_df.dropna(subset=[metric_name])
        .groupby(["step", "task"])[metric_name]
        .mean()
        .reset_index()
    )
    steps_in_data: Set[int] = set(run_avg["step"].unique())
    for step in sorted(steps_in_data):
        step_vals: List[float] = []
        for task in tasks:
            tv: pd.DataFrame = run_avg[
                (run_avg["task"] == task) & (run_avg["step"] == step)
            ]
            if len(tv) > 0:
                step_vals.append(float(tv[metric_name].iloc[0]))
        if len(step_vals) == len(tasks):
            avg_rows.append({"step": step, metric_name: float(np.mean(step_vals))})

    AVG_COLOR: str = "#536872"
    if len(avg_rows) > 0:
        avg_df: pd.DataFrame = pd.DataFrame(avg_rows).sort_values("step")
        overlay = overlay * avg_df.hvplot.line(
            x="step", y=metric_name, label="avg", color=AVG_COLOR,
            line_width=3.0, line_dash="solid",
        )
        overlay = overlay * avg_df.hvplot.scatter(
            x="step", y=metric_name, color=AVG_COLOR, size=80,
        )
        avg_best_idx = (
            avg_df[metric_name].idxmin() if is_lower_better
            else avg_df[metric_name].idxmax()
        )
        avg_best_step: int = avg_df.loc[avg_best_idx, "step"]
        avg_best_val: float = avg_df.loc[avg_best_idx, metric_name]
        overlay = overlay * pd.DataFrame(
            [{"step": avg_best_step, metric_name: avg_best_val}]
        ).hvplot.scatter(
            x="step", y=metric_name, color=AVG_COLOR,
            size=250, marker="star",
        )

    if show_hv:
        hv_df: pd.DataFrame = compute_hypervolume_per_step(
            cfg_df=cfg_df,
            metric_name=metric_name,
            tasks=tasks,
            is_lower_better=is_lower_better,
            dataset_name=dataset_name,
        )
        if len(hv_df) > 1:
            overlay = overlay * hv_df.hvplot.line(
                x="step",
                y="HV",
                label="HV indicator",
                color="black",
                line_width=3.5,
                ylabel="Hypervolume",
            )
            overlay = overlay * hv_df.hvplot.scatter(
                x="step", y="HV", color="black", size=130, marker="diamond"
            )

    overlay = overlay.opts(
        title=cfg_label,
        xlabel="",
        ylabel="",
        xlim=(-X_PAD, max_step + X_PAD),
        frame_width=PLOT_W,
        height=PLOT_H,
        show_grid=True,
        multi_y=show_hv,
        fontsize={"title": 18, "ticks": 11},
        hooks=[
            make_style_hook(
                hv_range_min=hv_range_min,
                hv_range_max=hv_range_max,
                hv_tick_format=hv_tick_format,
                bands=band_data if len(band_data) > 0 else None,
            )
        ],
    )
    return overlay


def make_legend_panel(
    *,
    tasks: List[str],
    task_colors: Dict[str, str],
    legend_width: int,
    include_hv: bool,
) -> hv.Overlay:
    legend_overlay: Optional[hv.Overlay] = None
    for task in tasks:
        c = pd.DataFrame({"x": [np.nan], "y": [np.nan]}).hvplot.line(
            x="x", y="y", label=task, color=task_colors[task], line_width=2.5,
        )
        legend_overlay = c if legend_overlay is None else legend_overlay * c
    avg_entry = pd.DataFrame({"x": [np.nan], "y": [np.nan]}).hvplot.scatter(
        x="x", y="y", label="avg", color="#536872", size=100,
    )
    legend_overlay = legend_overlay * avg_entry
    if include_hv:
        hv_entry = pd.DataFrame({"x": [np.nan], "y": [np.nan]}).hvplot.scatter(
            x="x", y="y", label="HV indicator", color="black", size=130, marker="diamond"
        )
        legend_overlay = legend_overlay * hv_entry
    star = pd.DataFrame({"x": [np.nan], "y": [np.nan]}).hvplot.scatter(
        x="x", y="y", label="best step", color="gray", size=140, marker="star"
    )
    legend_overlay = legend_overlay * star
    return legend_overlay.opts(
        height=LEGEND_H,
        width=legend_width,
        xaxis=None,
        yaxis=None,
        show_grid=False,
        toolbar=None,
        legend_position="top",
        legend_cols=9,
        title="",
        hooks=[legend_hook],
    )

def _hv_tick_format(hv_min: float, hv_max: float) -> str:
    """Choose a Numeral.js format string that gives uniform decimal places."""
    span: float = hv_max - hv_min
    if span <= 0:
        return "0.00"
    if span < 0.05:
        return "0.000"
    if span < 0.5:
        return "0.00"
    return "0.0"


def make_dashboard(
    *,
    all_df: pd.DataFrame,
    metric_name: str,
    tasks: List[str],
    task_colors: Dict[str, str],
    dataset_name: str,
    max_step: int,
) -> Optional[pn.Row]:
    is_lower_better: bool = metric_name in LOWER_IS_BETTER

    hv_range_min: Optional[float] = None
    hv_range_max: Optional[float] = None
    hv_fmt: str = "0.00"

    if SHOW_HV:
        all_hv_values: List[float] = []
        for cfg_label, cfg_suffix in CONFIGS:
            if cfg_label.startswith(SINGLE_TASK_LABEL_PREFIX):
                continue
            cfg_df: pd.DataFrame = all_df[all_df["config"] == cfg_label].copy()
            hv_df: pd.DataFrame = compute_hypervolume_per_step(
                cfg_df=cfg_df,
                metric_name=metric_name,
                tasks=tasks,
                is_lower_better=is_lower_better,
                dataset_name=dataset_name,
            )
            if len(hv_df) > 0:
                all_hv_values.extend(hv_df["HV"].tolist())

        if len(all_hv_values) > 0:
            hv_raw_min: float = float(np.min(all_hv_values))
            hv_raw_max: float = float(np.max(all_hv_values))
            hv_padding: float = (
                (hv_raw_max - hv_raw_min) * 0.05 if hv_raw_max > hv_raw_min else 0.1
            )
            hv_range_min = hv_raw_min - hv_padding
            hv_range_max = hv_raw_max + hv_padding
            hv_fmt = _hv_tick_format(hv_range_min, hv_range_max)

    subplots: List[hv.Overlay] = []
    for cfg_label, cfg_suffix in CONFIGS:
        is_single_task: bool = cfg_label.startswith(SINGLE_TASK_LABEL_PREFIX)
        subplot_show_hv: bool = SHOW_HV and (not is_single_task)
        sp: Optional[hv.Overlay] = build_subplot(
            cfg_df=all_df[all_df["config"] == cfg_label].copy(),
            cfg_label=cfg_label,
            metric_name=metric_name,
            tasks=tasks,
            task_colors=task_colors,
            is_lower_better=is_lower_better,
            dataset_name=dataset_name,
            max_step=max_step,
            show_hv=subplot_show_hv,
            hv_range_min=hv_range_min,
            hv_range_max=hv_range_max,
            hv_tick_format=hv_fmt,
        )
        if sp is not None:
            subplots.append(sp)

    if len(subplots) == 0:
        return None

    plot_rows: List[pn.Row] = []
    for row_idx, row_start in enumerate(range(0, len(subplots), GRID_COLS)):
        is_last_row: bool = row_start + GRID_COLS >= len(subplots)
        row_plots: List[pn.pane.HoloViews] = [
            pn.pane.HoloViews(sp, sizing_mode="fixed")
            for sp in subplots[row_start : row_start + GRID_COLS]
        ]
        plot_rows.append(
            pn.Row(
                *row_plots,
                sizing_mode="fixed",
                margin=(0, 0, 0 if is_last_row else GRID_ROW_GAP, 0),
            )
        )
    plot_area: pn.Column = pn.Column(
        *plot_rows,
        sizing_mode="fixed",
        margin=(0, 0, 0, 0),
    )

    plot_area_height: int = PLOT_H * GRID_ROWS + GRID_ROW_GAP * (GRID_ROWS - 1)

    ylabel_pane: pn.pane.HTML = pn.pane.HTML(
        f'<div style="writing-mode: vertical-rl; transform: rotate(180deg); '
        f"font-family: {CSS_FONT}; font-size: 20px; font-weight: normal; "
        f"text-align: center; display: flex; align-items: center; "
        f'justify-content: center; height: 100%; margin: 0; padding: 0;">'
        f"{metric_name}</div>",
        width=YLABEL_W,
        height=plot_area_height,
        sizing_mode="fixed",
        margin=(0, -10, 0, 0),
    )

    right_panes: List = []
    if SHOW_HV:
        hv_ylabel_pane: pn.pane.HTML = pn.pane.HTML(
            f'<div style="writing-mode: vertical-rl; '
            f"font-family: {CSS_FONT}; font-size: 20px; font-weight: normal; "
            f"text-align: center; display: flex; align-items: center; "
            f'justify-content: center; height: 100%; margin: 0; padding: 0;">'
            f"Hypervolume Indicator</div>",
            width=YLABEL_W,
            height=plot_area_height,
            sizing_mode="fixed",
            margin=(0, 0, 0, -10),
        )
        right_panes.append(hv_ylabel_pane)

    xlabel_pane: pn.pane.HTML = pn.pane.HTML(
        f'<div style="text-align: center; font-family: {CSS_FONT}; '
        f'font-size: 20px; font-weight: normal; margin: 0; padding: 0;">'
        f"Optimization Step</div>",
        height=26,
        sizing_mode="stretch_width",
        margin=(0, 0, 8, 0),
    )
    legend_pane: pn.pane.HoloViews = pn.pane.HoloViews(
        make_legend_panel(
            tasks=tasks, task_colors=task_colors,
            legend_width=GRID_W, include_hv=SHOW_HV,
        ),
        height=LEGEND_H,
        sizing_mode="stretch_width",
        margin=(0, 0, 0, 0),
    )

    center_col: pn.Column = pn.Column(
        plot_area,
        xlabel_pane,
        legend_pane,
        sizing_mode="fixed",
        margin=(0, 0, 0, 0),
    )
    return pn.Row(
        ylabel_pane,
        center_col,
        *right_panes,
        sizing_mode="fixed",
        margin=(0, 0, 0, 0),
    )


for ds in DATASETS:
    ds_name: str = ds["name"]
    prefixes: List[str] = ds["prefix"]
    tasks: List[str] = ds["tasks"]
    task_colors: Dict[str, str] = ds["task_colors"]
    max_step: int = ds["max_step"]

    print(f"\n{'=' * 70}")
    print(f"Dataset: {ds_name}")
    print(f"{'=' * 70}")
    print("Computing per-task metrics from parquet files...")

    # all_df: pd.DataFrame = compute_all_per_task_metrics(
    #     prefix=prefix,
    #     tasks=tasks,
    #     max_step=max_step,
    # )

    ### In case there are multiple prefixes -> 
    all_parts: List[pd.DataFrame] = []
    for prefix in prefixes:
        print(f"Computing per-task metrics from parquet files for prefix={prefix} ...")
        part_df: pd.DataFrame = compute_all_per_task_metrics(
            prefix=prefix,
            tasks=tasks,
            max_step=max_step,
        )
        if len(part_df) > 0:
            # optional: keep track of which run-batch it came from
            part_df["prefix"] = prefix
            all_parts.append(part_df)

    all_df: pd.DataFrame = (
        pd.concat(all_parts, ignore_index=True) if len(all_parts) > 0 else pd.DataFrame()
    )


    print(f"  Total rows: {len(all_df)}")
    print(f"  Configs found: {sorted(all_df['config'].unique())}")

    for metric_name in METRIC_NAMES:
        print(f"  Building dashboard for {metric_name}...")
        dashboard: Optional[pn.Row] = make_dashboard(
            all_df=all_df,
            metric_name=metric_name,
            tasks=tasks,
            task_colors=task_colors,
            dataset_name=ds_name,
            max_step=max_step,
        )
        if dashboard is None:
            print(f"    SKIP {metric_name}: no data")
            continue

        direction: str = (
            "lower=better" if metric_name in LOWER_IS_BETTER else "higher=better"
        )
        title: str = f"{ds_name} — {metric_name} ({direction})"

        out_path: str = os.path.join(
            OUT_DIR, f"textgrad_{ds_name.lower()}_{metric_name.lower()}_rows.html"
        )
        dashboard.save(out_path, title=title)
        print(f"    Saved: {out_path}")

    csv_path: str = os.path.join(OUT_DIR, f"textgrad_{ds_name.lower()}_all_metrics.csv")
    all_df.to_csv(csv_path, index=False)
    print(f"  Raw data CSV: {csv_path}")

print(f"\nAll plots saved to: {OUT_DIR}/")
print("Open any .html file in a browser to interact with the plots.")