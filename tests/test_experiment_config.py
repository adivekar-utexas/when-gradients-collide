"""Tests for experiment configuration system."""

import pytest
from when_gradients_collide.experiment_config import (
    ExperimentConfig,
    LLMConfig,
    LLM_PRESETS,
    deepseek_preset,
    claude_preset,
)


class TestLLMConfig:
    """Test LLMConfig class."""

    def test_model_configs_required(self):
        """Test that model configs are required."""
        with pytest.raises(ValueError, match="task_model"):
            LLMConfig(
                optimizer_model={"name": "gpt-4"},
                gradient_model={"name": "gpt-4"},
                loss_model={"name": "gpt-4"},
            )

    def test_minimal_config(self):
        """Test creating minimal LLMConfig."""
        config = LLMConfig(
            task_model={"name": "gpt-4"},
            optimizer_model={"name": "gpt-4"},
            gradient_model={"name": "gpt-4"},
            loss_model={"name": "gpt-4"},
        )
        assert config.task_model.name == "gpt-4"
        assert config.endpoints == {}
        assert config.api_base is None


class TestLLMPresets:
    """Test built-in LLM presets."""

    def test_deepseek_preset(self):
        """Test DeepSeek preset configuration."""
        preset = deepseek_preset()
        assert preset.task_model.name == "openai/deepseek-v4-flash"
        assert preset.optimizer_model.name == "openai/deepseek-v4-pro"
        assert preset.gradient_model.name == "openai/deepseek-v4-pro"
        assert preset.loss_model.name == "openai/deepseek-v4-pro"
        assert len(preset.endpoints) == 3

    def test_claude_preset(self):
        """Test Claude preset configuration."""
        preset = claude_preset()
        assert "claude-sonnet-4.6" in preset.task_model.name
        assert "claude-sonnet-4.6" in preset.optimizer_model.name
        assert len(preset.endpoints) == 1
        assert preset.api_base == "http://localhost:20128/v1"

    def test_presets_registry(self):
        """Test that presets are registered in LLM_PRESETS."""
        assert "deepseek" in LLM_PRESETS
        assert "claude" in LLM_PRESETS
        assert isinstance(LLM_PRESETS["deepseek"], LLMConfig)
        assert isinstance(LLM_PRESETS["claude"], LLMConfig)


class TestExperimentConfig:
    """Test ExperimentConfig class."""

    def test_minimal_config(self):
        """Test creating minimal ExperimentConfig."""
        config = ExperimentConfig(
            llm=deepseek_preset(),
            dataset="SummEval",
        )
        assert config.dataset == "SummEval"
        assert config.steps == 10
        assert config.batch_size == 10
        assert config.output_dir == "results"
        assert config.seed == 42

    def test_custom_values(self):
        """Test creating ExperimentConfig with custom values."""
        config = ExperimentConfig(
            llm=deepseek_preset(),
            dataset="BRIGHTER",
            steps=20,
            batch_size=32,
            output_dir="custom_results",
            seed=123,
        )
        assert config.dataset == "BRIGHTER"
        assert config.steps == 20
        assert config.batch_size == 32
        assert config.output_dir == "custom_results"
        assert config.seed == 123

    def test_from_preset(self):
        """Test creating ExperimentConfig from preset name."""
        config = ExperimentConfig(
            llm=LLM_PRESETS["deepseek"],
            dataset="SummEval",
        )
        assert config.llm.task_model.name == "openai/deepseek-v4-flash"

    def test_validation_dataset_required(self):
        """Test that dataset is required."""
        with pytest.raises(ValueError, match="dataset"):
            ExperimentConfig(llm=deepseek_preset())

    def test_validation_llm_required(self):
        """Test that llm is required."""
        with pytest.raises(ValueError, match="llm"):
            ExperimentConfig(dataset="SummEval")


class TestConfigurationWorkflow:
    """Test typical configuration workflows."""

    def test_override_preset_values(self):
        """Test overriding preset values via model_copy."""
        preset = deepseek_preset()
        # Override task model temperature via model_copy
        overridden_task = preset.task_model.model_copy(update={"temperature": 0.3})
        assert overridden_task.temperature == 0.3
        assert overridden_task.name == "openai/deepseek-v4-flash"

        # Override optimizer timeout via model_copy
        overridden_opt = preset.optimizer_model.model_copy(update={"timeout": 900.0})
        assert overridden_opt.timeout == 900.0

    def test_create_custom_llm_config(self):
        """Test creating custom LLMConfig."""
        config = LLMConfig(
            task_model={
                "name": "gpt-4o",
                "max_tokens": 8192,
                "temperature": 0.2,
            },
            optimizer_model={"name": "gpt-4o"},
            gradient_model={"name": "gpt-4o"},
            loss_model={"name": "gpt-4o"},
            endpoints={
                "custom_endpoint": {
                    "endpoint_id": "custom_endpoint",
                    "api_key": "test-key",
                    "max_calls_5h": 1000,
                }
            },
            api_base="https://api.openai.com/v1",
        )
        assert config.task_model.name == "gpt-4o"
        assert config.task_model.max_tokens == 8192
        assert len(config.endpoints) == 1
        assert config.api_base == "https://api.openai.com/v1"
