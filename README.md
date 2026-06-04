# When Gradients Collide: Failure Modes in Multi-Criteria LLM Judge Prompt Optimization

This repository contains code and resources for our ACL 2026 **CustomNLP Workshop** paper:

**When Gradients Collide: Failure Modes in Multi-Criteria LLM Judge Prompt Optimization**  
Abhishek Divekar*, Parth Darshan* (*equal contribution*)

- arXiv: https://arxiv.org/abs/2605.26046

---

## Overview

Customizing an LLM-as-a-judge to a specific domain often requires optimizing a single prompt across multiple evaluation criteria (e.g., fluency, relevance, coherence, consistency). Textual-gradient methods automate prompt optimization by producing natural-language critiques and edits—but these “gradients” are not numeric vectors, so classical multi-task gradient conflict tools (PCGrad, MGDA) do not directly apply.

This project studies multi-criteria textual-gradient prompt optimization for LLM judges by varying how much cross-task information is shared across the loss, gradient, and optimizer stages.

We identify two separable failure modes:

1. **Optimization-time gradient dilution**: when the gradient LLM processes multiple criteria jointly, task-specific feedback becomes generic and less actionable.
2. **Inference-time instruction interference**: even independently strong per-task instructions can hurt when combined into a single multi-criteria prompt.

---

## Architecture

Each optimization step consists of four stages:

1. **Task model** predicts scores using the current prompt
2. **Loss LLM** critiques predictions against ground truth
3. **Gradient LLM** converts critiques into instruction edits (“textual gradients”)
4. **Optimizer LLM** rewrites the prompt instructions

A **decomposition mode** controls whether each pipeline stage operates **per-task (Separate, S)** or **jointly across all tasks (Combined, C)**.

### Decomposition modes (S/C across stages)

- **SINGLE-TASK**: optimize each criterion in an independent run
- **SSS**: loss, gradient, optimizer all separate per task
- **SSC**: loss+gradient separate; optimizer combined
- **SCC**: loss separate; gradient+optimizer combined
- **CCC**: all combined

---

## Architecture Diagram

![Decomposition modes and pipeline architecture](img/overview.jpg)

---

## Key Takeaways (High-level)

- In many multi-task configurations, optimization **does not improve over the initial generic prompt**.
- When the **gradient stage is combined** (SCC/CCC), **gradient specificity drops sharply**, indicating “dilution”.
- Even with oracle per-task instructions, **combining them into one prompt can degrade performance**, indicating “instruction interference”.

---

## Repository Structure

Update this section to match your repo:

- `src/` — core code (data loading, prompting, optimization loop).
- `tests/` — run scripts (training/optimization/eval) and tests for code.
- `expt/`  - dataset and experiment setups. 
---

## Setup

### Environment

    ```
    python -m venv .venv
    source .venv/bin/activate
    pip install uv
    uv pip install -r requirements.txt
    ```

### Model / API configuration

Create a `.env` file in the **project root** (the same folder that contains `src/`, `tests/`, etc.) and set your credentials and model names there.

Example:
    ```

    OPENROUTER_API_KEY=...
    AG0=sk-...
    AG1=sk-...
    AG2=sk-...
    AG3=sk-...
    AG4=sk-...
    AG5=sk-...
    ```
Note: AG0, AG1, .. AG5 are omniroute api keys
    
---

## Running Experiments

Example (adjust flags/paths to match your code):

    python tests/run_mae_runs_2_3.py

---

## Citation

    @article{divekar2026when,
      title={When Gradients Collide: Failure Modes in Multi-Criteria LLM Judge Prompt Optimization},
      author={Divekar, Abhishek and Darshan, Parth},
      journal={arXiv preprint arXiv:2605.26046},
      year={2026}
    }

---

## Contact

- Abhishek Divekar: adivekar@amazon.com
- Parth Darshan: b22cs040@iitj.ac.in