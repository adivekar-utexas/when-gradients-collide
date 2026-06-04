"""Gradient Specificity bar chart — per-task vs all-task decomposition modes.

Reads the evaluated gradient specificity parquet and produces a single-panel
bar chart showing the sharp cliff between per-task modes (Single, SSS, SSC)
and all-task modes (SCC, CCC).
"""

import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from bokeh.io import output_file, save
from bokeh.models import (
    BasicTicker,
    ColumnDataSource,
    Label,
    NumeralTickFormatter,
    Span,
    Whisker,
)
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

BASE: str = os.path.join(os.path.dirname(__file__), "e2e_outputs", "unified")
PARQUET_PATH: str = os.path.join(BASE, "rq3_results", "gradient_specificity.parquet")
OUT_DIR: str = os.path.join(BASE, "textgrad_mode_plots")
os.makedirs(OUT_DIR, exist_ok=True)

BOKEH_FONT: str = "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
MODE_ORDER: List[str] = ["SSC", "SCC"] # , "CCC" "Single", "SSS", 
PER_TASK_MODES: List[str] = ["SSC"] # "Single", "SSS", 
ALL_TASK_MODES: List[str] = ["SCC"] # , "CCC"

COLOR_PER_TASK: str = "#8fd7d7"
COLOR_ALL_TASK: str = "#ff8ca1"

df = pd.read_parquet(PARQUET_PATH)

per_mode = (
    df.groupby("Mode")[["mean", "std"]]
      .mean()
      .reindex(MODE_ORDER)
)

means = per_mode["mean"].tolist()
stds  = per_mode["std"].tolist()
upper: List[float] = [m + s for m, s in zip(means, stds)]
lower: List[float] = [m - s for m, s in zip(means, stds)]
colors: List[str] = [
    COLOR_PER_TASK if m in PER_TASK_MODES else COLOR_ALL_TASK
    for m in MODE_ORDER
]

source: ColumnDataSource = ColumnDataSource(data=dict(
    modes=MODE_ORDER,
    means=means,
    upper=upper,
    lower=lower,
    colors=colors,
))

FRAME_W: int = 400
FRAME_H: int = 300

p = figure(
    x_range=MODE_ORDER,
    y_range=(0, 10.8),
    frame_width=FRAME_W,
    frame_height=FRAME_H,
    toolbar_location=None,
)

p.vbar(
    x="modes",
    top="means",
    width=0.6,
    source=source,
    color="colors",
    line_color="white",
    line_width=1.5,
)

p.add_layout(Whisker(
    source=source,
    base="modes",
    upper="upper",
    lower="lower",
    line_color="#444444",
    line_width=1.5,
    upper_head=None,
    lower_head=None,
))

per_task_mean: float = float(np.mean([m for m, mode in zip(means, MODE_ORDER) if mode in PER_TASK_MODES]))
all_task_mean: float = float(np.mean([m for m, mode in zip(means, MODE_ORDER) if mode in ALL_TASK_MODES]))
drop_pct: int = round((1 - all_task_mean / per_task_mean) * 100)

p.add_layout(Span(
    location=per_task_mean,
    dimension="width",
    line_color="#555555",
    line_dash="dashed",
    line_width=1.2,
    line_alpha=0.7,
))
p.add_layout(Span(
    location=all_task_mean,
    dimension="width",
    line_color="#555555",
    line_dash="dashed",
    line_width=1.2,
    line_alpha=0.7,
))

midpoint_y: float = (per_task_mean + all_task_mean) / 2
p.add_layout(Label(
    x=4.4,
    y=midpoint_y,
    text=f"{drop_pct}% drop",
    text_font=BOKEH_FONT,
    text_font_size="14pt",
    text_font_style="bold",
    text_color="#d63384",
    text_align="center",
    text_baseline="middle",
    x_units="data",
    y_units="data",
))

p.add_layout(Label(
    x=4.4,
    y=per_task_mean + 0.25,
    text=f"{per_task_mean:.1f}",
    text_font=BOKEH_FONT,
    text_font_size="11pt",
    text_color="#555555",
    text_align="center",
    text_baseline="bottom",
    x_units="data",
    y_units="data",
))
p.add_layout(Label(
    x=4.4,
    y=all_task_mean - 0.25,
    text=f"{all_task_mean:.1f}",
    text_font=BOKEH_FONT,
    text_font_size="11pt",
    text_color="#555555",
    text_align="center",
    text_baseline="top",
    x_units="data",
    y_units="data",
))

for axis in p.xaxis + p.yaxis:
    axis.major_label_text_font = BOKEH_FONT
    axis.major_label_text_font_size = "12pt"
    axis.axis_label_text_font = BOKEH_FONT
    axis.axis_label_text_font_size = "14pt"

p.yaxis.axis_label = "Gradient Specificity Score"
p.xaxis.axis_label = "Decomposition Mode"
p.yaxis.ticker = BasicTicker(desired_num_ticks=6)
p.yaxis.formatter = NumeralTickFormatter(format="0")

p.xgrid.visible = False
p.ygrid.grid_line_color = "#e5e5e5"
p.ygrid.grid_line_alpha = 0.8

p.outline_line_color = None
p.background_fill_color = "white"
p.border_fill_color = "white"

p.title.text = "Gradient Specificity by Decomposition Mode"
p.title.text_font = BOKEH_FONT
p.title.text_font_size = "14pt"
p.title.text_font_style = "normal"

out_path: str = os.path.join(OUT_DIR, "gradient_specificity.html")
output_file(out_path, title="Gradient Specificity")
save(p)
print(f"Saved: {out_path}")

print(f"\nBar values:")
for mode, mean, std in zip(MODE_ORDER, means, stds):
    print(f"  {mode:>8s}: {mean:.2f} ± {std:.2f}")
print(f"\nPer-task mean: {per_task_mean:.2f}")
print(f"All-task mean: {all_task_mean:.2f}")
print(f"Drop: {drop_pct}%")
