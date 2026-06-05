# Installation Guide

This guide walks you through installing the **when-gradients-collide** (WGC) research codebase, which reproduces the multi-objective prompt optimization experiments from "When Gradients Collide: Failure Modes of Multi-Objective Prompt Optimization for LLM Judges."

The PyPI distribution is `when-gradients-collide`; the importable Python module is `when_gradients_collide` (top-level sub-packages include `when_gradients_collide.algorithm`, `when_gradients_collide.expt`, and so on).

## 1. Prerequisites

Before you start, make sure the following are installed on your system:

- **Python 3.12 or newer** (the project pins `requires-python = ">=3.12"` in `pyproject.toml`).

  ```bash
  python --version
  # Python 3.12.x or higher
  ```

- **conda** (Miniconda, Mambaforge, or Miniforge). The package pulls in compiled dependencies (`pyarrow`, `fastparquet`, `numpy`, `pandas`) that are most reliable through a conda environment.

  ```bash
  conda --version
  ```

- **git**, for cloning the repository.

- **(Optional) An LLM API key.** The default configuration is wired for OpenRouter, but the framework speaks any OpenAI-compatible provider that LiteLLM supports (OpenAI, Anthropic, DeepSeek, local vLLM, etc.). See section 3.

- **Disk space**: ~3 GB for the conda environment, plus a few hundred MB for the parquet datasets.

## 2. Environment Setup

Create a dedicated conda environment for WGC:

```bash
# Clone the repository
git clone https://github.com/yourusername/when-gradients-collide.git
cd when-gradients-collide

# Create conda environment with Python 3.12
conda create -n wgc python=3.12 -y
conda activate wgc
```

## 3. Install Dependencies

Install the package in development mode with all dependencies:

```bash
# Install the package and runtime dependencies
pip install -e .

# Or install from requirements.txt if you prefer
pip install -r requirements.txt
```

For development (testing, linting, Jupyter):

```bash
# Install dev dependencies (pytest, ruff, ipykernel, etc.)
pip install -e ".[dev]"
```

**Key dependencies** (from `pyproject.toml`):
- `litellm` — unified LLM API (OpenRouter, OpenAI, Anthropic, etc.)
- `slowburn` — LLM worker pool with rate limiting
- `concurry` — distributed execution framework
- `datasets` — HuggingFace dataset loading
- `pyarrow`, `fastparquet` — parquet file support
- `hvplot`, `bokeh` — plotting for trajectory visualization

## 4. API Key Configuration

WGC uses environment variables for LLM API access. The framework supports multiple providers via LiteLLM.

### Step 1: Copy the environment template

```bash
cp .env.example .env
```

### Step 2: Edit `.env` with your API key

The `.env.example` file shows the expected format:

```ini
# Copy this file to .env and fill in your API keys.
OPENROUTER_API_KEY=sk-or-v1-your-key-here
# Optional: HuggingFace token for gated datasets (e.g. WildGuard).
# HF_TOKEN=hf_...
```

**Supported providers** (via the `experiment_config.py` `LLMConfig` classes):

- **OpenRouter** — set `OPENROUTER_API_KEY` (default; routes to multiple models)
- **OpenAI** — set `OPENAI_API_KEY`
- **Anthropic** — set `ANTHROPIC_API_KEY`
- **DeepSeek** — set `DEEPSEEK_API_KEY_0`, `DEEPSEEK_API_KEY_1`, etc.
- **Local/self-hosted** — configure `api_base` in your experiment config

The experiment configs in `src/when_gradients_collide/experiment_config.py` use `${ENV_VAR}` template syntax for API keys, so the actual key values are resolved at runtime from environment variables.

### Step 3 (Optional): HuggingFace token for gated datasets

The WildGuard dataset requires a HuggingFace token. Generate one at https://huggingface.co/settings/tokens, then add it to your `.env` file:

```ini
HF_TOKEN=hf_your_token_here
```

### Step 4: Verify your API key works

```bash
# Quick smoke test via the LLM worker
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API key loaded:', 'OPENROUTER_API_KEY' in os.environ)"
```

## 5. Dataset Preparation

WGC includes three benchmark datasets for prompt optimization experiments. After running `pip install -e .`, use the built-in setup script:

### Automatic download (recommended)

The package registers a console script `wgc-setup-datasets`:

```bash
# Setup all three datasets (SummEval, WildGuard, BRIGHTER)
wgc-setup-datasets

# Or setup specific datasets
wgc-setup-datasets --datasets SummEval
wgc-setup-datasets --datasets BRIGHTER
wgc-setup-datasets --datasets WildGuard
```

This downloads datasets from HuggingFace, preprocesses them, and saves parquet files under `data/`.

### Manual download

Alternatively, run the setup module directly:

```bash
python -m when_gradients_collide.expt.setup_datasets --datasets SummEval BRIGHTER
```

### Dataset details

| Dataset | Task | Train | Test | Source |
|---------|------|-------|------|--------|
| **SummEval** | Summary quality evaluation (fluency, relevance, coherence, consistency) | 160 | 480 | HuggingFace (public) |
| **WildGuard** | Safety classification (prompt safety, response safety, refusal detection) | varies | varies | HuggingFace (gated — needs `HF_TOKEN`) |
| **BRIGHTER** | Emotion intensity detection | varies | varies | HuggingFace (public) |

### Verify datasets are ready

```bash
# Check that parquet files exist in data/
ls -lh data/SummEval/*.parquet
ls -lh data/BRIGHTER/*.parquet
ls -lh data/WildGuard/*.parquet
```

You should see `*-train.parquet` and `*-test.parquet` files for each dataset.

## 6. Verification

After completing the setup, run these checks to confirm everything is installed correctly.

### Check Python imports

```bash
python -c "import when_gradients_collide; print('Package OK:', when_gradients_collide.__file__)"
```

Expected output shows the path inside `src/when_gradients_collide/`.

### Check sub-package imports

```bash
python -c "
from when_gradients_collide.algorithm import OPRO, TextGrad
from when_gradients_collide.config import wgc_config, temp_config
from when_gradients_collide.expt.dataset import SummEval, BRIGHTER, WildGuard
print('All sub-package imports OK')
"
```

### Run unit tests

The project ships with unit tests (no API calls required for the default set):

```bash
# Run all unit tests
pytest --tb=short -rf tests/

# Run a specific test file
pytest --tb=short -rf tests/test_metrics.py
```

The full test suite includes `unit` (fast, no network) and `integration` (real LLM calls, costs $) markers. To skip integration tests:

```bash
pytest --tb=short -rf tests/ -m "not integration"
```

### Check the CLI entry point

```bash
# Verify the console script was installed
which wgc-setup-datasets
wgc-setup-datasets --help
```

## 7. Troubleshooting

### Common Issues

**1. Python version mismatch**

```bash
python --version
# If below 3.12, create a new conda env with Python 3.12
conda create -n wgc python=3.12 -y
conda activate wgc
```

**2. `ModuleNotFoundError: No module named 'morphic'` or `'concurry'`**

These are custom packages on PyPI. Make sure your pip is up to date:

```bash
pip install --upgrade pip
pip install -e .
```

**3. `pyarrow` or `fastparquet` installation fails**

These are compiled C++ dependencies. Install via conda:

```bash
conda install pyarrow fastparquet -c conda-forge
```

**4. `litellm` version conflicts**

The project pins `litellm==1.*`. If you see import errors:

```bash
pip install litellm==1.65.1  # or any 1.x version
```

**5. `.env` file not loading**

The framework loads `.env` at runtime via `dotenv`. Make sure:

```bash
# Your .env is in the repository root (next to pyproject.toml)
ls -la .env
# Check the content
cat .env
```

**6. `slowburn` import errors on macOS**

`concurry[all]` and `slowburn[all]` pull in Ray for distributed execution. If Ray fails to install on your platform:

```bash
pip install concurry slowburn  # without [all] — no Ray dependency
```

**7. Tests fail with `OMNIROUTE_API_KEY not set`**

Integration tests require an API key. Set it in your `.env`:

```ini
OMNIROUTE_API_KEY=your_key_here
```

Or skip integration tests:

```bash
pytest --tb=short -rf tests/ -m "not integration"
```

## 8. Optional Dependencies

### Development and Analysis Tools

For development work (testing, linting, notebooks):

```bash
# Install the [dev] extra
pip install -e ".[dev]"
```

This adds:
- `pytest` (8.4.2) — testing framework
- `pytest-timeout` — prevents hanging tests
- `ipykernel` — Jupyter kernel support
- `ipywidgets` — interactive widgets
- `ruff` — fast Python linter

### Plotting and Visualization

The project includes trajectory visualization and analysis notebooks. Required packages are already in the main dependencies:

- `hvplot` — high-level plotting API
- `bokeh` — interactive web-based visualizations
- `pandas` — data manipulation
- `matplotlib` — additional plotting (transitive dependency)

### Distributed Execution (Ray)

For running experiments on distributed clusters:

```bash
# Install Ray for distributed execution
pip install concurry[all] slowburn[all]
```

This enables:
- Multi-node experiment execution
- Automatic worker pool management
- Distributed gradient computation

**Note:** Ray is optional for local development and single-machine experiments.

