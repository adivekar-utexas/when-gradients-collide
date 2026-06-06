"""Experiment configuration for prompt optimization.

All LLM configurations are loaded from JSON files on disk (via TypedPath),
not hardcoded in Python.  See ``expt/configs/`` for examples.
"""

from typing import Any, Dict, Optional

from morphic import Typed
from pydantic import Field


class ModelConfig(Typed):
    """Per-role LLM configuration."""

    name: str = Field(description="LiteLLM model string")
    max_tokens: int = Field(default=4096, description="Max output tokens")
    temperature: float = Field(default=0.1, description="Sampling temperature")
    timeout: float = Field(default=60.0, description="Request timeout (seconds)")
    reasoning: bool = Field(default=False, description="Enable reasoning mode")


class EndpointConfig(Typed):
    """API endpoint with rate limits."""

    endpoint_id: str = Field(description="Endpoint identifier")
    api_key: str = Field(description="API key (supports ${ENV_VAR} template)")
    max_calls_5h: int = Field(default=1000, description="Max calls per 5 hours")
    max_calls_1w: int = Field(default=5000, description="Max calls per week")
    budget_5h_usd: float = Field(default=10.0, description="Budget per 5 hours in USD")
    max_concurrent: int = Field(default=5, description="Max concurrent requests")


class LLMConfig(Typed):
    """Complete LLM configuration — loadable from a JSON file via TypedPath."""

    task_model: ModelConfig = Field(description="Task LLM configuration")
    optimizer_model: ModelConfig = Field(description="Optimizer LLM configuration")
    gradient_model: ModelConfig = Field(description="Gradient LLM configuration")
    loss_model: ModelConfig = Field(description="Loss LLM configuration")
    endpoints: Dict[str, EndpointConfig] = Field(
        default_factory=dict,
        description="Named endpoints for load balancing",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="OpenAI-compatible API base URL",
    )
    load_balancing: str = Field(default="RoundRobin", description="Load balancing strategy")
    endpoint_env_vars: list = Field(
        default_factory=list,
        description="Environment variable names holding API keys",
    )


class ExperimentConfig(Typed):
    """Top-level experiment configuration.

    The ``llm`` field can be either an inline LLMConfig (dict or instance)
    or a string path to a JSON file.  ``load_config()`` resolves the path
    automatically.  See ``expt/configs/`` for examples.
    """

    llm: LLMConfig = Field(description="LLM configuration")
    dataset: str = Field(description="Dataset name (SummEval, BRIGHTER, etc.)")
    output_dir: str = Field(default="results", description="Output directory")
    steps: int = Field(default=10, description="Number of optimization steps")
    batch_size: int = Field(default=10, description="Batch size for evaluation")
    eval_every: int = Field(default=1, description="Evaluate every N steps")
    seed: int = Field(default=42, description="Random seed")
    checkpoint_every: int = Field(default=1, description="Checkpoint every N steps")
    verbosity: int = Field(default=1, description="Logging verbosity (0-3)")


def load_config(config_path: str) -> ExperimentConfig:
    """Load an experiment config from a JSON file, resolving relative paths.

    The ``llm`` field in the experiment JSON can be either:
    - a dict (inline LLM config), or
    - a string path to another JSON file (relative to the experiment config's
      directory, or absolute).

    Returns:
        A validated ExperimentConfig with the LLMConfig resolved and nested.
    """
    import json
    from pathlib import Path

    config_path_obj = Path(config_path).resolve()
    if not config_path_obj.is_file():
        raise FileNotFoundError(
            f"Experiment config file not found: {config_path_obj}"
        )

    with open(config_path_obj, "r") as f:
        raw: Dict[str, Any] = json.load(f)

    base_dir: Path = config_path_obj.parent

    llm_value: Any = raw.get("llm")
    if isinstance(llm_value, str):
        llm_path = Path(llm_value)
        if not llm_path.is_absolute():
            llm_path = (base_dir / llm_path).resolve()
        if not llm_path.is_file():
            raise FileNotFoundError(
                f"LLM config file not found: {llm_path} "
                f"(referenced by experiment config {config_path_obj})"
            )
        with open(llm_path, "r") as f:
            llm_data: Dict[str, Any] = json.load(f)
        raw["llm"] = llm_data
    elif not isinstance(llm_value, dict):
        raise ValueError(
            f"Experiment config 'llm' field must be a dict or path string; "
            f"got {type(llm_value).__name__}"
        )

    return ExperimentConfig(**raw)
