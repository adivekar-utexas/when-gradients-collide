# When Gradients Collide

**Failure modes of multi-objective prompt optimization for LLM judges.**

This repository contains the code and data release for the ACL 2026 paper
*"When Gradients Collide: Failure Modes of Multi-Objective Prompt Optimization
for LLM Judges"*. It studies what happens when a textual-gradient prompt
optimizer (such as TextGrad or OPRO) is asked to improve a single LLM judge
prompt that scores text on several quality dimensions at once.

## Paper Summary

LLM-as-a-judge pipelines rely on automatic prompt optimization:
an outer LLM reads feedback ("gradients") about how an inner judge is doing
and rewrites the judge prompt to make it better. When the judge evaluates
multiple objectives at once (fluency, relevance, coherence, and consistency
for a summary; safety, accuracy, and helpfulness for a chat response), the
optimizer must combine several per-task feedback signals into a single
revision. This is the multi-objective prompt optimization problem.

The paper identifies two recurring failure modes in that setting. The first
is **gradient dilution**: when feedback from several tasks is concatenated
into a single textual gradient, the LLM that consumes the gradient cannot
tell which advice applies to which task, and the resulting gradient scores
substantially lower on task focus than per-task gradients. The second is
**instruction interference**: when per-task instructions that were
individually optimized in isolation are then combined into a single
multi-task prompt, they can interact in ways that make the combined prompt
worse than a naive single-prompt baseline.

To study these failure modes systematically, the paper parameterizes every
multi-objective optimization by three binary decisions: whether to compute
per-task losses (S) or a single combined loss (C), whether to compute
per-task gradients (S) or a single combined gradient (C), and whether to
feed per-task instructions to the optimizer (S) or a single combined
instruction (C). This gives four decomposition modes -- **SSS, SSC, SCC, CCC**
-- that span the space from fully per-task to fully combined.

Across 12 TextGrad runs per dataset on SummEval and BRIGHTER, the paper
finds that gradient specificity drops by roughly 59% when moving from
per-task gradient computation to combined gradient computation (from a mean
of about 9.0 to 3.7 on a 1-to-10 LLM-judged specificity scale). The
cherry-pick experiment -- taking the best per-task instruction found in any
single-task run, on any step, and stitching them into a combined prompt --
degrades test-set Spearman correlation from 0.305 (naive combined baseline)
to 0.220. Multi-objective optimization of a shared prompt never recovers
the performance of independently optimized per-task instructions.

## Repository Structure

```
when-gradients-collide/
├── data/                             # Datasets (SummEval, BRIGHTER)
│   ├── SummEval/
│   └── BRIGHTER/
├── figures/                          # Generated plots
├── scripts/                          # Analysis and experiment scripts
│   ├── download_results.py           # Download pre-computed results from HuggingFace
│   ├── eval_feedback_adherence.py    # Measure gradient adherence (RQ3)
│   ├── eval_gradient_specificity.py  # Measure gradient focus (RQ3)
│   ├── make_main_results_table.py    # Generate paper results table
│   ├── plot_gradient_specificity.py  # Plot specificity bar chart
│   ├── plot_trajectories.py          # Interactive trajectory visualization
│   ├── run_cherrypick.py             # Cherry-pick best single-task instructions
│   └── run_decomposition_modes.py    # Run 4 decomposition modes (SSS/SSC/SCC/CCC)
├── src/
│   └── when_gradients_collide/       # Main package
│       ├── algorithm/                # Algorithm implementations
│       │   ├── opro.py               # OPRO algorithm
│       │   └── textgrad.py           # TextGrad algorithm
│       ├── expt/                     # Experiment orchestration
│       │   ├── dataset.py            # Dataset definitions
│       │   ├── runner.py             # Main experiment runner
│       │   └── setup_datasets.py     # Dataset download/preprocessing
│       ├── experiment_config.py      # Typed LLM config with presets
│       ├── config.py                 # Global runtime configuration
│       ├── data_structures.py        # Core data types
│       ├── gradient_computer.py      # Gradient computation
│       ├── loss_computer.py          # Loss/feedback computation
│       ├── metrics.py                # Evaluation metrics
│       ├── observability.py          # Parquet-based logging
│       ├── prompt_algorithm.py       # Base algorithm class
│       ├── prompt_optimizer.py       # Prompt update logic
│       ├── prompt_template.py        # Multi-task prompt template
│       └── prompt_trajectory.py      # Trajectory tracking
├── tests/                            # Unit tests
├── INSTALL.md                        # Installation guide
├── EXPERIMENTS.md                    # Experiment reproduction guide
├── pyproject.toml                    # Package configuration
└── requirements.txt                  # Dependencies
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-username/when-gradients-collide.git
cd when-gradients-collide
conda create -n wgc python=3.12 -y && conda activate wgc
pip install -e .

# 2. Configure API key
cp .env.example .env
# Edit .env with your OpenRouter API key

# 3. Download pre-computed results
python scripts/download_results.py --subset aggregates

# 4. Reproduce a paper figure
python scripts/plot_gradient_specificity.py
```

## Two-Track Reproduction

**Fast track** (download results, generate figures):

```bash
python scripts/download_results.py --subset aggregates
python scripts/make_main_results_table.py
python scripts/plot_gradient_specificity.py
python scripts/plot_trajectories.py
```

**Full track** (run all experiments from scratch):

```bash
# Run decomposition modes (SSS, SSC, SCC, CCC) on SummEval
python scripts/run_decomposition_modes.py

# Cherry-pick best per-task instructions
python scripts/run_cherrypick.py

# Analyze gradient specificity and feedback adherence
python scripts/eval_gradient_specificity.py
python scripts/eval_feedback_adherence.py
```

See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed reproduction instructions with expected numbers.

## Algorithms

This repository implements two prompt optimization algorithms:

- **TextGrad** -- gradient-based prompt optimization using textual feedback from loss computation. Supports all four decomposition modes (SSS/SSC/SCC/CCC).
- **OPRO** -- optimization by prompting, which uses trajectory-based optimization with instruction-score pairs to guide prompt updates.

## Configuration

LLM configuration follows a typed `ExperimentConfig` pattern (see `src/when_gradients_collide/experiment_config.py`). Built-in presets:

```python
from when_gradients_collide.experiment_config import LLM_PRESETS, ExperimentConfig

config = ExperimentConfig(
    llm=LLM_PRESETS["deepseek"],  # or LLM_PRESETS["claude"]
    dataset="SummEval",
    steps=12,
    batch_size=10,
)
```

See `src/when_gradients_collide/experiment_config.py` for available presets.

## Citation

```bibtex
@inproceedings{darshan-divekar-2026-gradients,
    title     = {When Gradients Collide: Failure Modes of Multi-Objective
                 Prompt Optimization for {LLM} Judges},
    author    = {Darshan, Parth and Divekar, Abhishek},
    booktitle = {Proceedings of the 64th Annual Meeting of the Association
                 for Computational Linguistics (ACL 2026)},
    year      = {2026},
    address   = {Vienna, Austria},
    publisher = {Association for Computational Linguistics},
}
```

## License

[MIT](LICENSE)
