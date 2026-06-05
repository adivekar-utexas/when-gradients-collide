"""Experiment configuration for prompt optimization.

Inspired by pymc_model_selection/experiment_config.py pattern.
"""

from typing import Any, Dict, Optional
from pydantic import Field
from morphic import Typed


class ModelConfig(Typed):
    """LLM model configuration."""

    name: str = Field(description="LiteLLM model string")
    max_tokens: int = Field(default=4096, description="Max output tokens")
    temperature: float = Field(default=0.1, description="Sampling temperature")
    timeout: float = Field(default=60.0, description="Request timeout in seconds")
    reasoning: bool = Field(default=False, description="Enable reasoning mode")


class EndpointConfig(Typed):
    """API endpoint configuration with rate limits."""

    endpoint_id: str = Field(description="Endpoint identifier")
    api_key: str = Field(description="API key for this endpoint")
    max_calls_5h: int = Field(default=1000, description="Max calls per 5 hours")
    max_calls_1w: int = Field(default=5000, description="Max calls per week")
    budget_5h_usd: float = Field(default=10.0, description="Budget per 5 hours in USD")
    max_concurrent: int = Field(default=5, description="Max concurrent requests")


class LLMConfig(Typed):
    """Complete LLM configuration."""

    task_model: ModelConfig = Field(description="Task LLM configuration")
    optimizer_model: ModelConfig = Field(description="Optimizer LLM configuration")
    gradient_model: ModelConfig = Field(description="Gradient LLM configuration")
    loss_model: ModelConfig = Field(description="Loss LLM configuration")
    endpoints: Dict[str, EndpointConfig] = Field(
        default_factory=dict,
        description="Named endpoints for load balancing"
    )
    api_base: Optional[str] = Field(
        default=None,
        description="OpenAI-compatible API base URL (e.g. http://localhost:20128/v1)"
    )
    load_balancing: str = Field(
        default="RoundRobin",
        description="Load balancing strategy across endpoints"
    )
    endpoint_env_vars: list = Field(
        default_factory=list,
        description="Environment variable names holding API keys for endpoints"
    )


class ExperimentConfig(Typed):
    """Experiment configuration."""

    llm: LLMConfig = Field(description="LLM configuration")
    dataset: str = Field(description="Dataset name (SummEval, BRIGHTER, etc.)")
    output_dir: str = Field(default="results", description="Output directory")
    steps: int = Field(default=10, description="Number of optimization steps")
    batch_size: int = Field(default=10, description="Batch size for evaluation")
    eval_every: int = Field(default=1, description="Evaluate every N steps")
    seed: int = Field(default=42, description="Random seed")
    checkpoint_every: int = Field(default=1, description="Checkpoint every N steps")
    verbosity: int = Field(default=1, description="Logging verbosity (0-3)")


# ============================================================================
# Built-in LLM Presets
# ============================================================================

def deepseek_preset() -> LLMConfig:
    """DeepSeek preset: V4 Flash for tasks, V4 Pro for optimizer/gradient/loss."""
    return LLMConfig(
        task_model=ModelConfig(
            name="openai/deepseek-v4-flash",
            max_tokens=4096,
            temperature=0.1,
            timeout=60.0,
            reasoning=False,
        ),
        optimizer_model=ModelConfig(
            name="openai/deepseek-v4-pro",
            max_tokens=16384,
            temperature=0.7,
            timeout=600.0,
            reasoning=False,
        ),
        gradient_model=ModelConfig(
            name="openai/deepseek-v4-pro",
            max_tokens=8192,
            temperature=0.7,
            timeout=600.0,
            reasoning=False,
        ),
        loss_model=ModelConfig(
            name="openai/deepseek-v4-pro",
            max_tokens=1024,
            temperature=0.7,
            timeout=60.0,
            reasoning=False,
        ),
        endpoints={
            "endpoint_0": EndpointConfig(
                endpoint_id="endpoint_0",
                api_key="${DEEPSEEK_API_KEY_0}",
                max_calls_5h=5000,
                max_calls_1w=20000,
                budget_5h_usd=50.0,
                max_concurrent=10,
            ),
            "endpoint_1": EndpointConfig(
                endpoint_id="endpoint_1",
                api_key="${DEEPSEEK_API_KEY_1}",
                max_calls_5h=5000,
                max_calls_1w=20000,
                budget_5h_usd=50.0,
                max_concurrent=10,
            ),
            "endpoint_2": EndpointConfig(
                endpoint_id="endpoint_2",
                api_key="${DEEPSEEK_API_KEY_2}",
                max_calls_5h=5000,
                max_calls_1w=20000,
                budget_5h_usd=50.0,
                max_concurrent=10,
            ),
        },
        api_base=None,
        load_balancing="RoundRobin",
        endpoint_env_vars=["DEEPSEEK_API_KEY_0", "DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2"],
    )


def claude_preset() -> LLMConfig:
    """Claude Sonnet 4.6 preset via local proxy."""
    return LLMConfig(
        task_model=ModelConfig(
            name="openai/openrouter/anthropic/claude-sonnet-4.6",
            max_tokens=4096,
            temperature=0.1,
            timeout=60.0,
            reasoning=False,
        ),
        optimizer_model=ModelConfig(
            name="openai/openrouter/anthropic/claude-sonnet-4.6",
            max_tokens=16384,
            temperature=0.7,
            timeout=600.0,
            reasoning=False,
        ),
        gradient_model=ModelConfig(
            name="openai/openrouter/anthropic/claude-sonnet-4.6",
            max_tokens=8192,
            temperature=0.7,
            timeout=600.0,
            reasoning=False,
        ),
        loss_model=ModelConfig(
            name="openai/openrouter/anthropic/claude-sonnet-4.6",
            max_tokens=1024,
            temperature=0.7,
            timeout=60.0,
            reasoning=False,
        ),
        endpoints={
            "endpoint_0": EndpointConfig(
                endpoint_id="endpoint_0",
                api_key="${OPENROUTER_API_KEY_0}",
                max_calls_5h=2000,
                max_calls_1w=8000,
                budget_5h_usd=100.0,
                max_concurrent=5,
            ),
        },
        api_base="http://localhost:20128/v1",
        load_balancing="RoundRobin",
        endpoint_env_vars=["OPENROUTER_API_KEY_0"],
    )


# Registry of built-in presets
LLM_PRESETS: Dict[str, LLMConfig] = {
    "deepseek": deepseek_preset(),
    "claude": claude_preset(),
}
