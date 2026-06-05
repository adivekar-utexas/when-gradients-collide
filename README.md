<div align="center">

# When Gradients Collide

### Failure Modes of Multi-Objective Prompt Optimization for LLM Judges

**Parth Darshan &nbsp;&middot;&nbsp; Abhishek Divekar**

*ACL 2026*

[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Results-ffc107.svg)](https://huggingface.co/datasets/adivekar/when-gradients-collide-results)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

</div>

<hr />

> **Abstract:** *LLM-as-a-judge pipelines increasingly rely on automatic prompt optimization, where an outer LLM reads textual feedback ("gradients") about an inner judge and rewrites the judge prompt. When the judge scores multiple quality dimensions at once, the optimizer must reconcile several per-task feedback signals into a single revision. We study this multi-objective prompt optimization problem and identify two recurring failure modes. **Gradient dilution**: concatenating feedback from several tasks into one textual gradient makes the consuming LLM unable to tell which advice applies to which task, dropping gradient task-focus by 59% (9.0 to 3.7 on a 1-to-10 scale). **Instruction interference**: per-task instructions optimized in isolation interfere when stitched into a single multi-task prompt, degrading test-set Spearman from 0.305 to 0.220. We characterize both failures across four decomposition modes (SSS, SSC, SCC, CCC) on SummEval and BRIGHTER, using TextGrad and OPRO.*

<hr />

## Overview

<p align="center">
  <img src="figures/overview.jpg" width="95%" alt="Overview of the four decomposition modes and the two failure modes." />
</p>

The judge prompt is optimized with a textual-gradient loop. Each multi-objective optimization is parameterized by three binary choices: compute per-task (**S**eparate) or combined (**C**ombined) **loss**, **gradient**, and **optimizer update**. The four resulting modes (SSS, SSC, SCC, CCC) span the space from fully decomposed to fully aggregated.

## Table of Contents

- [Key Findings](#key-findings)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducing the Paper](#reproducing-the-paper)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Citation](#citation)
- [License](#license)

## Key Findings

| Finding | Result | Where |
|---|---|---|
| **Gradient dilution** | Gradient task-focus drops **59%** (9.0 → 3.7 / 10) when per-task feedback is combined into one gradient | `eval_gradient_specificity.py` |
| **Instruction interference** | Best per-task instructions degrade from **0.305 → 0.220** Spearman when stitched into one prompt | `run_cherrypick.py` |
| **High adherence** | The optimizer faithfully follows the gradient it receives (7.7–8.8 / 10) — the failure is upstream | `eval_feedback_adherence.py` |
| **Mode ordering** | Performance follows **SSS > SSC > SCC > CCC**, below the single-task upper bound | `run_decomposition_modes.py` |

## Installation

See [INSTALL.md](INSTALL.md) for full instructions. In brief:

```bash
git clone https://github.com/adivekar-utexas/when-gradients-collide.git
cd when-gradients-collide

conda create -n wgc python=3.12 -y
conda activate wgc
pip install -e .
```

Then configure an API key:

```bash
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY (or another provider key)
```

## Quick Start

```bash
# Download pre-computed results (~1 MB) from HuggingFace
python scripts/download_results.py --subset aggregates

# Regenerate a paper figure (no API calls needed)
python scripts/plot_gradient_specificity.py
```

## Reproducing the Paper

We provide two tracks. See [EXPERIMENTS.md](EXPERIMENTS.md) for per-figure commands and expected numbers.

**Fast track** — download pre-computed results, then regenerate tables and figures (no API calls):

```bash
python scripts/download_results.py --subset aggregates
python scripts/make_main_results_table.py
python scripts/plot_gradient_specificity.py
python scripts/plot_trajectories.py
```

**Full track** — rerun every experiment from scratch (requires API keys, ~4–6 h/dataset):

| Step | Command | Paper artifact |
|---|---|---|
| Decomposition modes | `python scripts/run_decomposition_modes.py` | Table 1, Figure 2 |
| Gradient specificity | `python scripts/eval_gradient_specificity.py` | Figure 3 |
| Feedback adherence | `python scripts/eval_feedback_adherence.py` | Section 5.3 |
| Cherry-pick | `python scripts/run_cherrypick.py` | Table 2 |

## Repository Structure

```
when-gradients-collide/
├── data/                             # Datasets (SummEval, BRIGHTER)
├── figures/                          # Overview diagram + generated plots
├── scripts/                          # Experiment + analysis scripts
│   ├── download_results.py           #   Download pre-computed results from HuggingFace
│   ├── run_decomposition_modes.py    #   Run 4 modes (SSS/SSC/SCC/CCC)
│   ├── run_cherrypick.py             #   Cherry-pick best per-task instructions
│   ├── eval_gradient_specificity.py  #   Measure gradient task-focus
│   ├── eval_feedback_adherence.py    #   Measure optimizer adherence
│   ├── make_main_results_table.py    #   Generate the main results table
│   ├── plot_gradient_specificity.py  #   Specificity bar chart
│   └── plot_trajectories.py          #   Interactive trajectory plots
├── src/when_gradients_collide/       # Main package
│   ├── algorithm/                    #   TextGrad, OPRO
│   ├── expt/                         #   runner, dataset, setup_datasets
│   ├── experiment_config.py          #   Typed LLM config + presets
│   ├── config.py                     #   Global runtime configuration
│   └── ...                           #   metrics, prompt template, observability, etc.
├── tests/                            # Unit tests
├── INSTALL.md                        # Installation guide
└── EXPERIMENTS.md                    # Reproduction guide
```

## Configuration

LLM settings use a typed `ExperimentConfig` ([experiment_config.py](src/when_gradients_collide/experiment_config.py)) with built-in presets:

```python
from when_gradients_collide.experiment_config import ExperimentConfig, LLM_PRESETS

config = ExperimentConfig(
    llm=LLM_PRESETS["deepseek"],  # or LLM_PRESETS["claude"]
    dataset="SummEval",
    steps=12,
    batch_size=10,
)
```

Each preset defines per-role models (`task_model`, `optimizer_model`, `gradient_model`, `loss_model`), rate-limited endpoints, and load balancing. Override any field with `model_copy(update=...)` or build a fully custom `LLMConfig`.

## Citation

```bibtex
@inproceedings{darshan-divekar-2026-gradients,
    title     = {When Gradients Collide: Failure Modes of Multi-Objective
                 Prompt Optimization for {LLM} Judges},
    author    = {Darshan, Parth and Divekar, Abhishek},
    booktitle = {Proceedings of the 64th Annual Meeting of the Association
                 for Computational Linguistics (ACL 2026)},
    year      = {2026},
}
```

## License

Released under the [MIT License](LICENSE).
