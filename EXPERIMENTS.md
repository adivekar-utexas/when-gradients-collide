# Reproducing "When Gradients Collide"

This guide walks through reproducing every numerical result in the ACL 2026 CustomNLP4U Workshop paper *"When Gradients Collide: Failure Modes of Multi-Objective Prompt Optimization for LLM Judges."* All experiments are fully scripted under `scripts/`. Each section below maps to a paper section, with the exact commands, expected outputs, and rough runtime.

## Overview

The paper studies what happens when a textual-gradient prompt optimizer (TextGrad or OPRO) is asked to improve a single LLM judge prompt that scores text on several quality dimensions at once. Two failure modes are identified:

1. **Gradient dilution** — When feedback from several tasks is concatenated into a single textual gradient, the LLM cannot tell which advice applies to which task. The gradient becomes a vague soup of cross-task suggestions rather than a focused, actionable critique. We measure this directly in Section 5.2 by LLM-judging the specificity of each gradient on a 1–10 scale and observe a 59% drop when moving from per-task to combined gradient computation.
2. **Instruction interference** — When per-task instructions optimized in isolation are stitched into a single multi-task prompt, they interact in ways that make the combined prompt worse than a naive baseline. The cherry-pick experiment in Section 5.4 shows test-set Spearman falling from 0.305 (per-task instructions, evaluated per-task) to 0.220 (the same instructions combined and evaluated jointly) — a degradation of 0.085 in absolute terms.

To study these failure modes systematically, the paper parameterizes every multi-objective optimization by three binary decisions: whether to compute **loss**, **gradient**, and **optimizer update** per-task or jointly. This yields four decomposition modes:

- **SSS** — per-task loss, per-task gradient, per-task optimizer (the most decomposed mode)
- **SSC** — per-task loss, per-task gradient, combined optimizer
- **SCC** — per-task loss, combined gradient, combined optimizer
- **CCC** — combined loss, combined gradient, combined optimizer (the most aggregated mode)

A single-task baseline (**ST**) is also reported as an approximate upper bound: the prompt is optimized on one task at a time and evaluated only on that task.

## Fast Track: Download Pre-computed Results

If you only want to inspect the numbers, skip the rerunning and download pre-computed results from HuggingFace:

```bash
python scripts/download_results.py --subset aggregates
```

This pulls the small CSV/parquet summaries (~1 MB) used to build Table 1 and the trajectory plot.

```bash
python scripts/download_results.py --subset full
```

This pulls the full per-step logs, validation gates, and gradient text for every run (~1.9 GB). Required to regenerate figures, recompute HVI trajectories, or run custom analyses.

Both commands print a progress bar and a file listing under `data/results/`.

## Section 5.1: Main Results Table (Table 1)

Run all decomposition modes and aggregate into the main table:

```bash
python scripts/run_decomposition_modes.py
```

This launches 12 TextGrad jobs in parallel per dataset: 4 modes (SSS, SSC, SCC, CCC) x 3 validation metrics (`off_by_one`, `mae`, `spearman_correlation`), 12 optimization steps each. On SummEval the tasks are fluency, relevance, coherence, and consistency. On BRIGHTER they are anger, fear, joy, sadness, and surprise.

```bash
python scripts/make_main_results_table.py
```

Reads per-step logs from `data/results/`, computes mean/std Spearman and OffByOne across 3 runs, along with HVI trajectories. The table is printed to stdout as markdown and saved to `figures/main_results.md`.

**Expected pattern:** SSS consistently outperforms the combined modes. The paper shows SSS > SSC > SCC > CCC, with the single-task (ST) baseline sitting above SSS as an approximate upper bound since each task is optimized on its own dedicated prompt.

**Runtime:** 4-6 hours per dataset with API calls on the default Qwen 3 model. The 12 jobs run in parallel via the Concurry worker pool.

## Section 5.2: Gradient Specificity (Figure 3)

Measure how focused each textual gradient is on its target task, then plot the results:

```bash
python scripts/eval_gradient_specificity.py
```

Samples gradients from SummEval runs across all four decomposition modes plus the single-task baseline. An LLM judge scores each gradient on a 1-10 specificity scale. Three runs per mode are evaluated for variance estimates. Results land in `data/results/gradient_specificity.parquet`.

```bash
python scripts/plot_gradient_specificity.py
```

Renders an interactive Bokeh bar chart of specificity by mode to `figures/gradient_specificity.html`.

**Expected numbers:** Per-task modes (Single, SSS, SSC) cluster around 9.0 mean specificity. Combined-gradient modes (SCC, CCC) drop sharply to ~3.7. The ~59% cliff between per-task and combined gradient computation is the paper's most direct measurement of gradient dilution.

**Runtime:** ~15-30 minutes for evaluation, plus a few seconds for the plot.

## Section 5.3: Feedback Adherence

Measure how well the optimizer revised instructions address the gradient it received:

```bash
python scripts/eval_feedback_adherence.py
```

Runs on SummEval across all modes (SSS, SSC, SCC, CCC, and Single), 3 runs each. An LLM judge scores whether the new instruction actually incorporates the feedback from the gradient on a 1-10 scale. Results land in `data/results/feedback_adherence.parquet`.

**Expected numbers:** 7.7-8.8 across all modes. High adherence even when the gradient itself is vague. This is a key finding: the optimizer faithfully follows whatever gradient it receives, which means the failure mode is upstream in the gradient, not downstream in the optimizer.

**Runtime:** ~15-30 minutes for evaluation.

## Section 5.4: Instruction Interference (Cherry-Pick) (Table 2)

Test whether per-task-optimized instructions degrade when combined:

```bash
python scripts/run_cherrypick.py
```

This experiment scans all single-task runs (3 runs x 13 steps per run) to find the best instruction for each task for a given metric, then stitches the four best per-task instructions into one combined multi-task prompt and evaluates it on the full test set. It produces 6 evaluation numbers: {Spearman, OffByOne, MAE} x {val=none, val=mae}.

**Expected numbers (Spearman):** Per-task-optimized instructions evaluated per-task yield 0.305 Spearman. The same four instructions stitched together and evaluated jointly yield 0.220. This degradation of 0.085 is the paper's most direct evidence of instruction interference: the per-task instructions are individually strong, but interact badly when combined into a single prompt.

**Runtime:** ~15-30 minutes with API calls.

## Section 5.5: Trajectories (Figure 2)

Visualize per-task Spearman across all 12 optimization steps for each mode:

```bash
python scripts/plot_trajectories.py
```

Renders an interactive HTML plot showing per-task Spearman correlation as a function of optimization step, faceted by decomposition mode. Hover over each line to see exact values; click legend entries to toggle modes. The plot is saved to `figures/trajectories.html`.

**Expected output:** A Bokeh figure with one panel per mode, showing four lines per panel (one per task). SSS and SSC show steady improvement across steps; SCC and CCC plateau early or even regress, consistent with gradient dilution and instruction interference setting in from step 2-3 onward.

**Runtime:** A few seconds, no API calls required (reads from pre-computed per-step logs).

## Expected Total Runtime

| Action | Time |
|---|---|
| Aggregates download | <1 minute |
| Full download | 5-10 minutes (1.9 GB) |
| Full TextGrad runs | 4-6 hours per dataset with API calls |
| Single experiment | 15-30 minutes |

The main bottleneck is the full TextGrad runs in `run_decomposition_modes.py`. Each dataset spawns 12 parallel jobs (4 modes x 3 validation metrics), each running 12 optimization steps. With Concurry's worker pool, wall-clock time is roughly the time of a single 12-step run (~90 minutes) multiplied by a few sequential rounds needed to stay within API rate limits.

The evaluation scripts (`eval_gradient_specificity.py`, `eval_feedback_adherence.py`, `run_cherrypick.py`) are comparatively fast: 15-30 minutes each, since they only score a small number of sampled gradients or run a single evaluation pass over the test set.

The plotting scripts (`plot_gradient_specificity.py`, `plot_trajectories.py`, `make_main_results_table.py`) take seconds and require no API calls.

## API Cost Estimates

Roughly $5-20 per dataset depending on model choice. The paper's default configuration uses Qwen 3 via OpenRouter, which is at the low end of that range. Switching to Claude Sonnet or GPT-4o will push costs toward the upper bound.

Cost breakdown by script:

| Script | Cost | Notes |
|---|---|---|
| `run_decomposition_modes.py` | $10-20 per dataset | 12 parallel jobs, 12 steps each, 4 LLM roles |
| `eval_gradient_specificity.py` | $1-3 | Scores ~60 gradients with one LLM judge call each |
| `eval_feedback_adherence.py` | $1-3 | Scores ~60 revised instructions |
| `run_cherrypick.py` | $2-5 | One full test-set evaluation pass |
| `plot_trajectories.py` | $0 | Reads local parquet, no API calls |
| `plot_gradient_specificity.py` | $0 | Reads local parquet, no API calls |
| `make_main_results_table.py` | $0 | Reads local parquet, no API calls |

The evaluation scripts (`eval_gradient_specificity.py`, `eval_feedback_adherence.py`, `run_cherrypick.py`) can also run on pre-downloaded results from `download_results.py --subset full`, in which case the only cost is the LLM judge calls they make (not the original optimization runs).

## Prerequisites

Before running any experiment, make sure you have:

1. Installed the package and its dependencies (see `INSTALL.md` for full instructions).
2. Downloaded and prepared both datasets (SummEval and BRIGHTER) into `data/`. The `INSTALL.md` step `python scripts/setup_datasets.py` handles this automatically.
3. Configured at least one API provider. Copy `.env.example` to `.env` and fill in the relevant keys. The default model (DeepSeek V3) uses OpenRouter; other options include OpenAI, Anthropic, DeepSeek, or a local/self-hosted endpoint.
4. Verified the setup runs cleanly with `pytest tests/`.

## Environment Notes

- **Python 3.12+** is required (tested on 3.12 and 3.13).
- **macOS and Linux** are supported. Windows has not been tested.
- Per-step logs are stored in `data/results/` as Parquet files. The full experiment tree for both datasets, all modes, and all runs is approximately 1.9 GB.
- The `scripts/run_decomposition_modes.py` script automatically detects available CPU cores and scales the worker pool accordingly. On a machine with fewer cores, the 12 jobs will queue rather than crash, at the cost of longer wall-clock time.