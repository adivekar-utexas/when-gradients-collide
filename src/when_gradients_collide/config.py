"""
Global configuration for WGC.

Provides a single source of truth for all tunable defaults (timeouts, temperatures,
max_tokens, rate limits, retry config, etc.).

Pattern follows SlowBurn/Concurry: a mutable singleton ``wgc_config``
whose ``defaults`` field can be mutated at runtime or scoped via ``temp_config()``.
"""

from contextlib import contextmanager
from typing import Any, Generator

from morphic import MutableTyped
from pydantic import ConfigDict, Field, confloat, conint


class WgcDefaults(MutableTyped):
    """All tunable defaults for WGC, in one place.

    Call sites read from ``wgc_config.defaults.<field>`` at call time.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Batch-level timeout for .result() calls in pipeline components
    batch_invocation_timeout: confloat(gt=0.0) = 3600.0

    # Per-LLM-role timeouts (used in create_*_llm factory functions)
    task_llm_timeout: confloat(gt=0.0) = 60.0
    optimizer_llm_timeout: confloat(gt=0.0) = 600.0
    gradient_llm_timeout: confloat(gt=0.0) = 600.0
    loss_llm_timeout: confloat(gt=0.0) = 60.0

    # Per-LLM-role max_tokens
    task_llm_max_tokens: conint(ge=1) = 256
    optimizer_llm_max_tokens: conint(ge=1) = 16384
    gradient_llm_max_tokens: conint(ge=1) = 8192
    loss_llm_max_tokens: conint(ge=1) = 1024

    # Temperature defaults per role (fallback when the algorithm class does
    # not specify a preference, i.e. leaves its temperature field as None).
    # Values match the paper: task=0.1, optimizer/gradient/loss=0.7.
    task_llm_temperature: confloat(ge=0.0, le=2.0) = 0.1
    optimizer_llm_temperature: confloat(ge=0.0, le=2.0) = 0.7
    gradient_llm_temperature: confloat(ge=0.0, le=2.0) = 0.7
    loss_llm_temperature: confloat(ge=0.0, le=2.0) = 0.7

    # Shared rate limits (across all 4 LLM workers per experiment)
    max_parallel_calls: conint(ge=1) = 20
    max_rpm: conint(ge=1) = 1_000
    max_input_tpm: conint(ge=1) = 10_000_000
    max_output_tpm: conint(ge=1) = 1_000_000

    # Retry
    num_retries: conint(ge=0) = 10
    retry_algorithm: str = "Fibonacci"
    retry_wait: confloat(ge=0.0) = 2.0
    retry_jitter: confloat(ge=0.0) = 0.7

    # Training loop
    substep_delay: confloat(ge=0.0) = 1.5

    # Verbosity
    verbosity: conint(ge=0) = 1


class WgcConfig(MutableTyped):
    """Top-level config object. The global singleton is ``wgc_config``."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    defaults: WgcDefaults = Field(default_factory=WgcDefaults)

    def reset_to_defaults(self) -> None:
        """Restore all defaults to their original values."""
        object.__setattr__(self, "defaults", WgcDefaults())


wgc_config = WgcConfig()


@contextmanager
def temp_config(**overrides: Any) -> Generator[WgcConfig, None, None]:
    """Temporarily override config defaults within a ``with`` block.

    All keyword arguments must be valid field names on ``WgcDefaults``.
    On exit (normal or exception), the original values are restored.

    Usage::

        from when_gradients_collide.config import temp_config

        with temp_config(batch_invocation_timeout=300, verbosity=0):
            results = algo.train(dataset, initial_prompt)

    Raises:
        ValueError: If any key is not a recognized WgcDefaults field.
    """
    defaults = wgc_config.defaults
    valid_fields = set(WgcDefaults.model_fields.keys())
    unknown = set(overrides.keys()) - valid_fields
    if len(unknown) > 0:
        raise ValueError(
            f"Unknown config key(s): {sorted(unknown)}. Valid: {sorted(valid_fields)}"
        )
    saved = {k: getattr(defaults, k) for k in overrides}
    try:
        for key, value in overrides.items():
            setattr(defaults, key, value)
        yield wgc_config
    finally:
        for key, value in saved.items():
            setattr(defaults, key, value)
