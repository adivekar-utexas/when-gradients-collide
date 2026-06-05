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


class PromptMOOConfig(Typed):
    """Experiment configuration."""
    llm: LLMConfig = Field(description="LLM configuration")
    dataset: str = Field(description="Dataset name (SummEval, BRIGHTER, etc.)")
    output_dir: str = Field(default="results", description="Output directory")
    steps: int = Field(default=10, description="Number of optimization steps")
    batch_size: int = Field(default=10, description="Batch size for evaluation")
    eval_every: int = Field(default=1, description="Evaluate every N steps")
    seed: int = Field(default=42, description="Random seed")
    checkpoint_every: int = Field(default=1, description="Checkpoint every N steps")
