import asyncio
import json
import os
import random

# Disable litellm's background LoggingWorker entirely.
# litellm (<=1.78) has a GLOBAL_LOGGING_WORKER singleton with an asyncio.Queue
# that gets bound to the first event loop that touches it. When multiple
# AlgorithmRunner threads each spin up their own event loops (for SlowBurnLLM),
# the Queue raises "bound to a different event loop" from the second thread on.
# See: https://github.com/BerriAI/litellm/issues/17813
#      https://github.com/BerriAI/litellm/issues/14521
#
# Replacing the module-level GLOBAL_LOGGING_WORKER variable is NOT sufficient
# because litellm.utils and litellm.caching.caching_handler both do
# `from ... import GLOBAL_LOGGING_WORKER` at import time, holding their own
# stale reference. The only fix that reaches all call sites is to patch the
# CLASS METHOD so every instance (past and future) becomes a no-op.
import warnings
from typing import Any, Dict, List, Optional, Tuple

import litellm
from concurry import CallLimit, LimitSet, RateLimit, ResourceLimit, Worker, RateLimitAlgorithm
from concurry.core.limit.limit_set import BaseLimitSet
from morphic import validate
from morphic.typed import format_exception_msg
from slowburn import SlowBurnLLM, create_llm, CostLimit
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings(
    "ignore", message="coroutine .* was never awaited", category=RuntimeWarning
)

try:
    from litellm.litellm_core_utils.logging_worker import LoggingWorker

    def _noop_enqueue(self, coroutine=None, async_coroutine=None):
        coro = coroutine or async_coroutine
        if coro is not None:
            coro.close()

    LoggingWorker.ensure_initialized_and_enqueue = _noop_enqueue
    LoggingWorker.start = lambda self: None
    LoggingWorker.enqueue = _noop_enqueue
except (ImportError, AttributeError):
    pass

from when_gradients_collide.algorithm import GPO, OPRO, PE2, TextGrad
from when_gradients_collide.config import wgc_config
from when_gradients_collide.data_input import Dataset
from when_gradients_collide.data_structures import Task
from when_gradients_collide.prompt_template import PromptTemplate
from when_gradients_collide.task_predictor import parse_task_response


def parse_task_response_retry_until(result: str, **context) -> bool:
    """Retry until the task response is valid JSON."""
    try:
        parse_task_response(result)
        return True
    except Exception:
        return False
    
WINDOW_5H = 5 * 3600
WINDOW_1W = 7 * 86400


LLM_CONFIGS = {
    "DeepSeek": {
        "reasoning": False,
        "provider_order": {},

        # One shared API base for all DeepSeek OpenCode-Go endpoints
        "api_base": "https://opencode.ai/zen/go/v1",  # "https://adivekar-server.tailfcaae5.ts.net/1c32-018b-e83f/v1",
        "load_balancing": "RoundRobin",
        "endpoint_env_vars": ["AG0", "AG1", "AG2", "AG3", "AG4", "AG5"],

        # Per-role model configs (task/optimizer/loss/gradient)
        "task_model": {
            "name": "openai/deepseek-v4-flash",
            "max_calls_5h": 30000,
            "max_calls_1w": 78000,
            "budget_5h_usd": 7.0,
            "max_concurrent": 10,
            "input_cost_per_token": 0.14 / 1_000_000,
            "output_cost_per_token": 0.28 / 1_000_000,
        },
        "optimizer_model": {
            "name": "openai/deepseek-v4-pro",
            "max_calls_5h": 3000,
            "max_calls_1w": 8000,
            "budget_5h_usd": 3.0,
            "max_concurrent": 10,
            "input_cost_per_token": 0.435 / 1_000_000,
            "output_cost_per_token": 0.87 / 1_000_000,
        },
        "loss_model": {
            "name": "openai/deepseek-v4-pro",
            "max_calls_5h": 3000,
            "max_calls_1w": 8000,
            "budget_5h_usd": 3.0,
            "max_concurrent": 10,
            "input_cost_per_token": 0.435 / 1_000_000,
            "output_cost_per_token": 0.87 / 1_000_000,
        },
        "gradient_model": {
            "name": "openai/deepseek-v4-pro",
            "max_calls_5h": 3000,
            "max_calls_1w": 8000,
            "budget_5h_usd": 3.0,
            "max_concurrent": 10,
            "input_cost_per_token": 0.435 / 1_000_000,
            "output_cost_per_token": 0.87 / 1_000_000,
        },
    },

    "Custom": {
        "task_model": "openai/openrouter/qwen/qwen3-8b",
        "optimizer_model": "openai/openrouter/qwen/qwen3-235b-a22b-2507",
        "loss_model": "openai/openrouter/qwen/qwen3-235b-a22b-2507",
        "gradient_model": "openai/opencode-go/deepseek-v4-pro",
        "reasoning": False,
        "provider_order": {},
    },

    "claude4.6": {
        "reasoning": False,
        "provider_order": {},

        # One shared API base for all DeepSeek OpenCode-Go endpoints
        "api_base": "http://localhost:20128/v1",  # "https://adivekar-server.tailfcaae5.ts.net/1c32-018b-e83f/v1",
        "load_balancing": "RoundRobin",
        "endpoint_env_vars": ["AG0", "AG1", "AG2"],

        # Updated to the "current schema" (per-role dict objects)
        "task_model": {
            "name": "openai/openrouter/anthropic/claude-sonnet-4.6",
            "max_calls_5h": 2000,
            "max_calls_1w": 5000,
            "budget_5h_usd": 8.0,
            "max_concurrent": 10,
            "input_cost_per_token": 3.0 / 1_000_000,
            "output_cost_per_token": 15.0 / 1_000_000,
        },
        "optimizer_model": {
            "name": "openai/openrouter/anthropic/claude-sonnet-4.6",
            "max_calls_5h": 2000,
            "max_calls_1w": 5000,
            "budget_5h_usd": 8.0,
            "max_concurrent": 10,
            "input_cost_per_token": 3.0 / 1_000_000,
            "output_cost_per_token": 15.0 / 1_000_000,
        },
        "loss_model": {
            "name": "openai/openrouter/anthropic/claude-sonnet-4.6",
            "max_calls_5h": 2000,
            "max_calls_1w": 5000,
            "budget_5h_usd": 8.0,
            "max_concurrent": 10,
            "input_cost_per_token": 3.0 / 1_000_000,
            "output_cost_per_token": 15.0 / 1_000_000,
        },
        "gradient_model": {
            "name": "openai/openrouter/anthropic/claude-sonnet-4.6",
            "max_calls_5h": 2000,
            "max_calls_1w": 5000,
            "budget_5h_usd": 8.0,
            "max_concurrent": 10,
            "input_cost_per_token": 3.0 / 1_000_000,
            "output_cost_per_token": 15.0 / 1_000_000,
        },
    },
}

REASONING_EXTRA_TOKENS = 2000
QWEN_NO_THINK_SUFFIX = "\n/no_think"

def _require_env(name: str) -> str:
    v = os.getenv(name)
    # print(f"Fetched env name: {name}")
    if not v:
        print(f"Missing env var: {name}")
        raise ValueError(f"Missing env var: {name}")
    return v

def _endpoint_with_limits(*, endpoint_id, api_key, max_calls_5h, max_calls_1w, budget_5h_usd, max_concurrent):
    return {
        "endpoint_id": endpoint_id,                       # label for cost reports
        "api_key": api_key,                        # this key only on this endpoint
        # Per-endpoint limits replace the global cascade for any slot they set.
        "limits": dict(
            # Two RateLimits on the same slot -> both windows enforced.
            requests=[
                RateLimit(key="requests", capacity=max_calls_5h,
                          window=WINDOW_5H, algorithm=RateLimitAlgorithm.GCRA),
                RateLimit(key="requests", capacity=max_calls_1w,
                          window=WINDOW_1W, algorithm=RateLimitAlgorithm.GCRA),
            ],
            # Dollar budget reset every 5 hours.
            budget=[
                CostLimit(budget_usd=budget_5h_usd, window=WINDOW_5H,
                          algorithm=RateLimitAlgorithm.GCRA),
            ],
            # Cap on simultaneously in-flight requests on this key.
            concurrency=max_concurrent,
        ),
    }

def build_balancing_pool_from_config(
    *,
    preset_cfg: dict,
    role: str,  # "task_model" | "optimizer_model" | "loss_model" | "gradient_model"
    max_tokens: int,
    timeout: float,
    temperature: float,
):
    """
    preset_cfg: e.g. LLM_CONFIGS["DeepSeek"]
    role: one of "task_model" | "optimizer_model" | "loss_model" | "gradient_model"
    """
    model_cfg = preset_cfg[role]
    model_name = model_cfg["name"]

    endpoints = [
        _endpoint_with_limits(
            endpoint_id=env_name,
            api_key=_require_env(env_name),
            max_calls_5h=model_cfg["max_calls_5h"],
            max_calls_1w=model_cfg["max_calls_1w"],
            budget_5h_usd=model_cfg["budget_5h_usd"],
            max_concurrent=model_cfg["max_concurrent"],
        )
        for env_name in preset_cfg["endpoint_env_vars"]
    ]

    litellm_params={
            "extra_body": {"thinking": {"type": "disabled"}},
        }

    return create_llm(
        model=model_name,
        api_base=preset_cfg["api_base"],
        endpoints=endpoints,
        load_balancing="RoundRobin",
        litellm_params=litellm_params,
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,

        # IMPORTANT: don't error if SlowBurn doesn't know token pricing for this model
        # on_pricing_unavailable="warn",
    )

def _detect_reasoning_family(model_name: str) -> Optional[str]:
    """Detect the reasoning parameter family from the model name."""
    if "/qwen/" in model_name:
        return "qwen"
    if "/openai/" in model_name:
        return "openai"
    if "/anthropic/" in model_name:
        return "anthropic"
    return None

# def _build_litellm_params(
#     *,
#     llm: str,
#     providers: Optional[List[str]],
#     model_name: str,
#     reasoning: bool,
# ) -> Dict[str, Any]:
#     """
#     Build litellm_params.

#     - Uses LLM_CONFIGS[llm]["api_base"] if present (e.g., OpenCode-Go endpoint).
#     - Falls back to OMNIROUTE_API_BASE otherwise.
#     - Disables thinking via extra_body.
#     """
#     cfg = LLM_CONFIGS.get(llm, {})
#     api_base = cfg.get("api_base") or os.getenv("OMNIROUTE_API_BASE")

#     return {
#         "api_base": api_base,
#         "extra_body": {"thinking": {"type": "disabled"}},
#     }


def _get_prompt_suffix(*, model_name: str, reasoning: bool) -> str:
    """Return a prompt suffix to append to every user message for this worker.

    For Qwen models with reasoning disabled, returns the ``/no_think`` token
    as a belt-and-suspenders complement to the API-level ``enable_thinking: false``.
    Some OpenRouter providers ignore the API param; the in-message token is
    always respected by the Qwen chat template.
    """
    if not reasoning and _detect_reasoning_family(model_name) == "qwen":
        return QWEN_NO_THINK_SUFFIX
    return ""


def _stamp_prompt_suffix(llm: SlowBurnLLM, *, model_name: str, reasoning: bool) -> None:
    """Set ``_prompt_suffix`` on a SlowBurnLLM worker instance.

    Pipeline components read this via ``get_prompt_suffix(llm_pool)`` and
    append it to every user message before calling ``call_llm_batch``.
    """
    suffix = _get_prompt_suffix(model_name=model_name, reasoning=reasoning)
    object.__setattr__(llm, "_prompt_suffix", suffix)


def get_prompt_suffix(llm_pool: Any) -> str:
    """Read the prompt suffix stored on an LLM worker (empty string if none).

    Re-exported from ``when_gradients_collide.llm_utils`` for convenience.
    """
    from when_gradients_collide.llm_utils import get_prompt_suffix as _get

    return _get(llm_pool)


def _build_retry_config(*, cfg: Any) -> Dict[str, Any]:
    """Build retry config dict shared by all LLM factory functions.

    Args:
        cfg: WgcDefaults instance.

    Returns:
        Dict of retry-related kwargs for SlowBurnLLM.options().
    """
    return dict(
        num_retries={"call_llm": cfg.num_retries, "*": 0},
        retry_wait={"call_llm": cfg.retry_wait, "*": 1},
        retry_algorithm={"call_llm": cfg.retry_algorithm, "*": "Exponential"},
        retry_jitter={"call_llm": cfg.retry_jitter, "*": 0},
        retry_on={
            "call_llm": [
                ValueError,
                asyncio.TimeoutError,
                litellm.Timeout,
                litellm.APIError,
                litellm.APIConnectionError,
                litellm.BadRequestError,
                litellm.InternalServerError,
                litellm.RateLimitError,
                litellm.ServiceUnavailableError,
            ],
            "*": [],
        },
    )


# @validate
# def create_shared_limits() -> BaseLimitSet:
#     """Create shared LimitSet for all LLM workers.

#     Reads capacities from wgc_config.defaults at call time.

#     Returns:
#         LimitSet configured for rate limiting across all LLM workers.
#     """
#     cfg = wgc_config.defaults
#     return LimitSet(
#         limits=[
#             ResourceLimit(key="parallel_calls", capacity=cfg.max_parallel_calls),
#             CallLimit(window_seconds=60, capacity=cfg.max_rpm),
#             RateLimit(
#                 key="input_tokens", window_seconds=60, capacity=cfg.max_input_tpm
#             ),
#             RateLimit(
#                 key="output_tokens", window_seconds=60, capacity=cfg.max_output_tpm
#             ),
#         ],
#         mode="asyncio",
#         shared=True,
#     )

@validate
def create_task_llm(
    *,
    llm: str,
    reasoning: bool = False,
    temperature: Optional[float] = None,
) -> Any:
    if llm not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM: {llm}. Options: {list(LLM_CONFIGS.keys())}")
    
    cfg = wgc_config.defaults
    max_tokens = cfg.task_llm_max_tokens
    timeout = cfg.task_llm_timeout
    resolved_temperature = temperature if temperature is not None else cfg.task_llm_temperature

    config = LLM_CONFIGS[llm]
    model_name = config["task_model"]["name"]
    reasoning = config.get("reasoning", False)

    print(f"Task: temp->{resolved_temperature} | max_tokens: {max_tokens} | timeout: {timeout}")

    pool = build_balancing_pool_from_config(
        preset_cfg=config, 
        role="task_model",
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=resolved_temperature
    )

    try:
        _stamp_prompt_suffix(pool, model_name=model_name, reasoning=reasoning)
    except Exception:
        pass

    return pool

@validate
def create_optimizer_llm(
    *,
    llm: str,
    temperature: Optional[float] = None,
) -> Any:
    if llm not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM: {llm}. Options: {list(LLM_CONFIGS.keys())}")
    
    cfg = wgc_config.defaults
    max_tokens = cfg.optimizer_llm_max_tokens
    timeout = cfg.optimizer_llm_timeout
    resolved_temperature = temperature if temperature is not None else cfg.optimizer_llm_temperature

    config = LLM_CONFIGS[llm]
    model_name = config["optimizer_model"]["name"]
    reasoning = config.get("reasoning", False)

    print(f"Optimizer: temp->{resolved_temperature} | max_tokens: {max_tokens} | timeout: {timeout}")

    pool = build_balancing_pool_from_config(
        preset_cfg=config, 
        role="optimizer_model",
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=resolved_temperature,
    )

    try:
        _stamp_prompt_suffix(pool, model_name=model_name, reasoning=reasoning)
    except Exception:
        pass

    return pool

@validate
def create_gradient_llm(
    *,
    llm: str,
    temperature: Optional[float] = None,
) -> Any:
    if llm not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM: {llm}. Options: {list(LLM_CONFIGS.keys())}")
    
    cfg = wgc_config.defaults
    max_tokens = cfg.gradient_llm_max_tokens
    timeout = cfg.gradient_llm_timeout
    resolved_temperature = temperature if temperature is not None else cfg.gradient_llm_temperature

    config = LLM_CONFIGS[llm]
    model_name = config["gradient_model"]["name"]
    reasoning = config.get("reasoning", False)

    print(f"Gradient: temp->{resolved_temperature} | max_tokens: {max_tokens} | timeout: {timeout}")

    pool = build_balancing_pool_from_config(
        preset_cfg=config,
        role="gradient_model",
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=resolved_temperature,
    )

    try:
        _stamp_prompt_suffix(pool, model_name=model_name, reasoning=reasoning)
    except Exception:
        pass

    return pool

@validate
def create_loss_llm(
    *,
    llm: str,
    temperature: Optional[float] = None,
) -> Any:
    if llm not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM: {llm}. Options: {list(LLM_CONFIGS.keys())}")

    config = LLM_CONFIGS[llm]
    model_name = config["loss_model"]["name"]
    reasoning = config.get("reasoning", False)

    cfg = wgc_config.defaults
    max_tokens = cfg.loss_llm_max_tokens
    timeout = cfg.loss_llm_timeout
    resolved_temperature = temperature if temperature is not None else cfg.loss_llm_temperature

    print(f"Loss: temp->{resolved_temperature} | max_tokens: {max_tokens} | timeout: {timeout}")

    pool = build_balancing_pool_from_config(
        preset_cfg=config,
        role="loss_model",
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=resolved_temperature,
    )

    try:
        _stamp_prompt_suffix(pool, model_name=model_name, reasoning=reasoning)
    except Exception:
        pass

    return pool

# Dataset configurations


@validate
def build_prompt_skeleton(
    *,
    dataset: Dataset,
    tasks: List[Task],
) -> str:
    """Build prompt skeleton dynamically from the Dataset object.

    The skeleton is the frozen portion of the task prompt that does NOT
    change during optimization.  It contains:
    1. The evaluation directive (what to evaluate)
    2. General output constraints (no reasoning)

    The per-task ``## Output format`` section is NOT part of the skeleton.
    It lives in ``task_output_formats`` on ``PromptTemplate`` and is rendered
    dynamically by ``render_instructions()``, which supports per-task
    filtering via ``task_filter`` to prevent cross-task leakage in
    TextGrad's ``separate_tasks`` mode.

    The mutable per-task instructions are also NOT part of the skeleton;
    they are appended by ``PromptTemplate.render_instructions()``.

    Args:
        dataset: Dataset instance.
        tasks: List of tasks to include (order is preserved).

    Returns:
        Prompt skeleton string (without ``## Output format`` section).
    """
    from when_gradients_collide.task_output_spec import collective_output_noun

    task_output_specs: Dict[str, Any] = dataset.task_output_specs
    evaluation_directive: str = dataset.evaluation_directive

    if len(evaluation_directive) == 0:
        if len(dataset.prompt_prefix) > 0:
            evaluation_directive = dataset.prompt_prefix
        else:
            raise ValueError(
                f"Dataset {dataset.dataset_name!r} has empty evaluation_directive. "
                f"Define it as a ClassVar on the Dataset subclass."
            )

    if len(task_output_specs) == 0:
        task_output_formats: Dict[str, str] = dataset.task_output_formats
        if len(task_output_formats) == 0:
            raise ValueError(
                f"Dataset {dataset.dataset_name!r} has empty task_output_specs "
                f"and task_output_formats. Define task_output_specs."
            )
        for task in tasks:
            if task.task_name not in task_output_formats:
                raise ValueError(
                    f"Task '{task.task_name}' not found in task_output_formats "
                    f"for dataset '{dataset.dataset_name}'"
                )
        output_noun_plural: str = "values"
    else:
        for task in tasks:
            if task.task_name not in task_output_specs:
                raise ValueError(
                    f"Task '{task.task_name}' not found in task_output_specs "
                    f"for dataset '{dataset.dataset_name}'"
                )
        output_noun_plural = collective_output_noun(
            {t.task_name: task_output_specs[t.task_name] for t in tasks}
        )

    skeleton: str = (
        f"{evaluation_directive.strip()}"
        f"\nUse the Instructions below to perform your evaluation. "
        f"Output a JSON with the requested {output_noun_plural}. "
        f"Do NOT include reasoning or explanations.\n"
    )

    return skeleton


def _build_task_output_formats(
    *,
    dataset: Dataset,
    tasks: List[Task],
) -> Dict[str, str]:
    """Build per-task output format specs for ``PromptTemplate.task_output_formats``.

    Reads from ``dataset.task_output_specs`` (preferred) or falls back to
    ``dataset.task_output_formats`` (a plain string dict).

    Args:
        dataset: Dataset instance.
        tasks: List of tasks to include (order is preserved).

    Returns:
        Dict mapping task_name to its format string (e.g. ``"1|2|3|4|5"``).
    """
    task_output_specs: Dict[str, Any] = dataset.task_output_specs
    if len(task_output_specs) > 0:
        result: Dict[str, str] = {}
        for task in tasks:
            if task.task_name not in task_output_specs:
                raise ValueError(
                    f"Task '{task.task_name}' not found in task_output_specs "
                    f"for dataset '{dataset.dataset_name}'"
                )
            result[task.task_name] = task_output_specs[task.task_name].format_str
        return result

    task_output_formats: Dict[str, str] = dataset.task_output_formats
    if len(task_output_formats) == 0:
        raise ValueError(
            f"Dataset {dataset.dataset_name!r} has empty task_output_specs "
            f"and task_output_formats. Define task_output_specs."
        )
    result = {}
    for task in tasks:
        if task.task_name not in task_output_formats:
            raise ValueError(
                f"Task '{task.task_name}' not found in task_output_formats "
                f"for dataset '{dataset.dataset_name}'"
            )
        result[task.task_name] = task_output_formats[task.task_name]
    return result


@validate
def get_initial_prompt(
    *,
    dataset: Dataset,
    tasks: List[Task],
) -> PromptTemplate:
    """Get initial prompt for a dataset with specified tasks.

    Args:
        dataset: Dataset instance.
        tasks: List of tasks to include in the prompt.

    Returns:
        PromptTemplate configured for the specified tasks, with
        ``task_output_formats`` populated for per-task filtering support.
    """
    skeleton: str = build_prompt_skeleton(
        dataset=dataset,
        tasks=tasks,
    )
    task_output_formats: Dict[str, str] = _build_task_output_formats(
        dataset=dataset,
        tasks=tasks,
    )
    return PromptTemplate(
        skeleton=skeleton,
        instruction={t.task_name: t.task_instruction for t in tasks},
        tasks=tasks,
        input_col_labels=dataset.input_col_labels,
        task_output_formats=task_output_formats,
    )


@validate
def get_task_losses(
    *, dataset: Dataset, tasks: Optional[List[Task]] = None
) -> Dict[str, str]:
    """Get task losses from the Dataset object.

    Args:
        dataset: Dataset instance (carries task_losses).
        tasks: Optional list of tasks to filter losses for.

    Returns:
        Dict mapping task names to loss function names.
    """
    all_losses = dataset.task_losses
    if len(all_losses) == 0:
        raise ValueError(
            f"Dataset {dataset.dataset_name!r} has empty task_losses. "
            f"Define it as a ClassVar on the Dataset subclass."
        )
    if tasks is not None:
        task_names = {t.task_name for t in tasks}
        return {k: v for k, v in all_losses.items() if k in task_names}
    return all_losses


@validate
def get_single_task_prompt(
    *,
    task: Task,
    dataset: Dataset,
) -> PromptTemplate:
    """Get initial prompt for a single task.

    Args:
        task: The task to create a prompt for.
        dataset: Dataset instance.

    Returns:
        PromptTemplate configured for this single task.
    """
    skeleton = build_prompt_skeleton(
        dataset=dataset,
        tasks=[task],
    )
    return PromptTemplate(
        skeleton=skeleton,
        instruction={task.task_name: task.task_instruction},
        tasks=[task],
        input_col_labels=dataset.input_col_labels,
    )


@validate
def find_last_prompt(output_dir: str) -> Tuple[Optional[int], Optional[str]]:
    """Find the latest saved prompt in an output directory."""
    prompts_dir = os.path.join(output_dir, "prompts")
    if not os.path.exists(prompts_dir):
        return None, None

    for i in range(100, -1, -1):
        prompt_path = os.path.join(prompts_dir, f"step_{i}_new.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r") as f:
                return i, f.read()
    return None, None


@validate
def check_run_status(output_dir: str) -> Dict[str, Any]:
    """Check the status of a run based on its output files."""
    summary_path = os.path.join(output_dir, "run_summary.json")

    if not os.path.exists(output_dir):
        return {"status": "not_found", "error_step": None, "last_prompt_step": None}

    result = {"status": "incomplete", "error_step": None, "last_prompt_step": None}

    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

        if "completed_at" in summary:
            result["status"] = "completed"
            return result

        if "error_step" in summary:
            result["status"] = "error"
            result["error_step"] = summary["error_step"]

    last_step, _ = find_last_prompt(output_dir)
    result["last_prompt_step"] = last_step

    return result


@validate
def resume_failed_runs(
    futures: Dict[str, Any],
    experiments: List[Dict[str, Any]],
    runner_pool: Any,
    run_name: str,
) -> Dict[str, Any]:
    """Check inactive runs and re-submit failed ones from their error step."""
    resumed_futures = {}

    for exp in experiments:
        exp_key = f"{exp['dataset_name']}_{exp['algorithm']}_{exp['llm']}"
        output_dir = exp.get("output_dir")

        if not output_dir:
            continue

        status = check_run_status(output_dir)

        if status["status"] == "error":
            error_step = status["error_step"]
            last_prompt_step, prompt_content = find_last_prompt(output_dir)

            resume_step = (
                error_step if error_step is not None else (last_prompt_step or 0)
            )

            print(f"[RESUME] Re-Submitting {exp_key} from step {resume_step}")

            future = runner_pool.run(
                run_name=run_name,
                dataset=exp["dataset"],
                output_dir=output_dir,
                algo_name=exp["algorithm"],
                llm=exp["llm"],
                steps=exp["steps"],
                api_key=exp["api_key"],
                batch_size=exp["batch_size"],
                eval_every=exp["eval_every"],
                verbosity=1,
                start_step=resume_step,
                resume_prompt=prompt_content,
            )
            resumed_futures[exp_key] = future

    return resumed_futures


class AlgorithmRunner(Worker):
    """Worker for running algorithms in parallel.

    Each AlgorithmRunner instance creates a shared LimitSet that is used by all
    LLM workers it instantiates, ensuring proper rate limiting across all LLM calls.

    Algorithm-specific parameters (e.g. ``k``, ``validation_metric``,
    ``trajectory_strategy``, ``pe2_batch_size``) are passed through via
    ``**algo_params`` and forwarded directly to the algorithm constructor.
    The runner does not hard-code any algorithm hyperparameters.
    """

    # @validate omitted: Worker methods are serialized for Ray remote execution.
    # validate_call closure is not picklable across process boundaries.
    def run(
        self,
        *,
        dataset: Dataset,
        algo_name: str,
        api_key: str,
        steps: int,
        batch_size: int,
        eval_every: int,
        run_name: str = "run1",
        llm: str = "llama3.1",
        verbosity: int = 1,
        start_step: int = 0,
        resume_prompt: Optional[str] = None,
        output_dir: Optional[str] = None,
        **algo_params,
    ) -> Dict[str, Any]:
        """Run algorithm and return results.

        Args:
            dataset: Dataset to run on.
            algo_name: Algorithm name ("gpo", "opro", "textgrad", "pe2").
            api_key: API key for LLM service.
            steps: Number of training steps.
            batch_size: Batch size for training.
            eval_every: Evaluate every N steps.
            run_name: Name for this run.
            llm: LLM family to use.
            verbosity: Logging verbosity (0=silent, 1=default, 2=detailed, 3=debug).
            start_step: Resume from this step.
            resume_prompt: Resume from this prompt content.
            output_dir: Output directory (auto-generated if None).
            **algo_params: Algorithm-specific hyperparameters passed directly
                to the algorithm constructor.  Examples:
                - GPO: ``k=5``, ``trajectory_strategy="relevance"``
                - OPRO: ``k=20``, ``num_candidates=8``
                - TextGrad: ``validation_metric="accuracy"``,
                  ``validation_gate_samples=50``
                - PE2: ``pe2_batch_size=2``, ``max_prompt_tokens=100``
                Any parameter accepted by the algorithm's constructor can
                be passed here.  Unknown keys will be rejected by Pydantic
                validation at construction time.
        """
        print(
            f"[AlgorithmRunner] Starting {algo_name} on {dataset.dataset_name} "
            f"(run: {run_name}, llm: {llm})"
        )

        try:
            tasks = dataset.tasks

            initial_prompt = get_initial_prompt(
                dataset=dataset,
                tasks=tasks,
            )
            task_losses = get_task_losses(dataset=dataset, tasks=tasks)


            # Resolve the algorithm class so we can read its temperature defaults
            # BEFORE creating LLM workers.  If the caller passed explicit
            # temperature overrides in algo_params, those take priority over
            # the class-level defaults.
            algo_cls_map = {
                "gpo": GPO,
                "opro": OPRO,
                "textgrad": TextGrad,
                "pe2": PE2,
            }
            if algo_name not in algo_cls_map:
                raise ValueError(
                    f"Unknown algorithm: {algo_name!r}. "
                    f"Must be 'gpo', 'textgrad', 'opro', or 'pe2'."
                )
            algo_cls = algo_cls_map[algo_name]

            def _resolve_temperature(
                field_name: str,
            ) -> Optional[float]:
                """Read temperature from algo_params (explicit override) or class default."""
                if field_name in algo_params:
                    return algo_params[field_name]
                field_info = algo_cls.model_fields.get(field_name)
                if field_info is not None:
                    return field_info.default
                return None

            task_llm = create_task_llm(
                llm=llm,
                temperature=_resolve_temperature("task_llm_temperature"),
            )
            optimizer_llm = create_optimizer_llm(
                llm=llm,
                temperature=_resolve_temperature("optimizer_llm_temperature"),
            )
            gradient_llm = create_gradient_llm(
                llm=llm,
                temperature=_resolve_temperature("gradient_llm_temperature"),
            )
            loss_llm = create_loss_llm(
                llm=llm,
                temperature=_resolve_temperature("loss_llm_temperature"),
            )

            common_params: Dict[str, Any] = {
                "tasks": tasks,
                "steps": steps,
                "batch_size": batch_size,
                "eval_every": eval_every,
                "name": f"{dataset.dataset_name}_{algo_name}_{run_name}",
                "verbosity": verbosity,
            }

            if algo_name == "gpo":
                algo = algo_cls(
                    task_llm=task_llm,
                    optimizer_llm=optimizer_llm,
                    task_losses=task_losses,
                    **{
                        **common_params,
                        **algo_params,
                    },
                )
            elif algo_name == "textgrad":
                algo = algo_cls(
                    task_llm=task_llm,
                    gradient_llm=gradient_llm,
                    optimizer_llm=optimizer_llm,
                    loss_llm=loss_llm,
                    **{
                        **common_params,
                        **algo_params,
                    },
                )
            elif algo_name == "opro":
                algo = algo_cls(
                    task_llm=task_llm,
                    optimizer_llm=optimizer_llm,
                    task_losses=task_losses,
                    **{
                        **common_params,
                        **algo_params,
                    },
                )
            elif algo_name == "pe2":
                algo = algo_cls(
                    task_llm=task_llm,
                    gradient_llm=optimizer_llm,
                    optimizer_llm=optimizer_llm,
                    task_losses=task_losses,
                    **{
                        **common_params,
                        **algo_params,
                    },
                )
            else:
                raise ValueError(
                    f"Unknown algorithm: {algo_name!r}. "
                    f"Must be 'gpo', 'textgrad', 'opro', or 'pe2'."
                )

            results = algo.train(
                dataset=dataset,
                initial_prompt=initial_prompt,
                output_dir=output_dir,
                start_step=start_step,
            )

            print(f"[AlgorithmRunner] Completed {algo_name} on {dataset.dataset_name}")
            print("#" * 80)
            return {
                "status": "success",
                "dataset": dataset.dataset_name,
                "algorithm": algo_name,
                "run_name": run_name,
                "llm": llm,
                "steps": steps,
                "batch_size": batch_size,
                "eval_every": eval_every,
                "results": results,
                **{
                    **common_params,
                    **algo_params,
                },
            }
        except Exception as e:
            print(
                f"[AlgorithmRunner] Failed {algo_name} on {dataset.dataset_name}:\n"
                f"{format_exception_msg(e)}"
            )
            print("#" * 80)
            return {
                "status": "error",
                "dataset": dataset.dataset_name,
                "algorithm": algo_name,
                "run_name": run_name,
                "llm": llm,
                "steps": steps,
                "batch_size": batch_size,
                "eval_every": eval_every,
                "error": format_exception_msg(e),
                **{
                    **common_params,
                    **algo_params,
                },
            }
