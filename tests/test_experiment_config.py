import json
import pytest
import tempfile
from pathlib import Path

from when_gradients_collide.experiment_config import (
    ExperimentConfig,
    LLMConfig,
    ModelConfig,
    EndpointConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sample_model_dict(**overrides):
    d = {
        "name": "openai/gpt-4o-mini",
        "max_tokens": 4096,
        "temperature": 0.1,
        "timeout": 60.0,
        "reasoning": False,
    }
    d.update(overrides)
    return d


def _sample_llm_dict(**overrides):
    d = {
        "task_model": _sample_model_dict(),
        "optimizer_model": _sample_model_dict(name="openai/gpt-4o"),
        "gradient_model": _sample_model_dict(name="openai/gpt-4o"),
        "loss_model": _sample_model_dict(name="openai/gpt-4o-mini"),
    }
    d.update(overrides)
    return d


def _sample_experiment_dict(**overrides):
    d = {
        "llm": _sample_llm_dict(),
        "dataset": "SummEval",
    }
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# TestModelConfig
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestModelConfig:
    def test_create_with_name_only(self):
        m = ModelConfig(name="openai/gpt-4o-mini")
        assert m.name == "openai/gpt-4o-mini"

    def test_defaults(self):
        m = ModelConfig(name="openai/gpt-4o-mini")
        assert m.max_tokens == 4096
        assert m.temperature == 0.1
        assert m.timeout == 60.0
        assert m.reasoning is False

    def test_override_all_fields(self):
        m = ModelConfig(
            name="anthropic/claude-3-5-sonnet",
            max_tokens=8192,
            temperature=0.7,
            timeout=120.0,
            reasoning=True,
        )
        assert m.name == "anthropic/claude-3-5-sonnet"
        assert m.max_tokens == 8192
        assert m.temperature == 0.7
        assert m.timeout == 120.0
        assert m.reasoning is True

    def test_missing_name_raises(self):
        with pytest.raises(Exception):
            ModelConfig()  # type: ignore[call-arg]

    def test_wrong_type_max_tokens_raises(self):
        with pytest.raises(Exception):
            ModelConfig(name="openai/gpt-4o-mini", max_tokens="not-an-int")  # type: ignore[arg-type]

    def test_wrong_type_temperature_raises(self):
        with pytest.raises(Exception):
            ModelConfig(name="openai/gpt-4o-mini", temperature="hot")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestEndpointConfig
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestEndpointConfig:
    def test_create_minimal(self):
        ep = EndpointConfig(endpoint_id="ep-1", api_key="sk-test")
        assert ep.endpoint_id == "ep-1"
        assert ep.api_key == "sk-test"

    def test_defaults(self):
        ep = EndpointConfig(endpoint_id="ep-1", api_key="sk-test")
        assert ep.max_calls_5h == 1000
        assert ep.max_calls_1w == 5000
        assert ep.budget_5h_usd == 10.0
        assert ep.max_concurrent == 5

    def test_override_all_fields(self):
        ep = EndpointConfig(
            endpoint_id="ep-2",
            api_key="${MY_API_KEY}",
            max_calls_5h=200,
            max_calls_1w=1000,
            budget_5h_usd=25.0,
            max_concurrent=20,
        )
        assert ep.endpoint_id == "ep-2"
        assert ep.api_key == "${MY_API_KEY}"
        assert ep.max_calls_5h == 200
        assert ep.max_calls_1w == 1000
        assert ep.budget_5h_usd == 25.0
        assert ep.max_concurrent == 20

    def test_env_var_template_preserved(self):
        ep = EndpointConfig(endpoint_id="ep-1", api_key="${OMNIROUTE_API_KEY}")
        assert ep.api_key == "${OMNIROUTE_API_KEY}"

    def test_missing_endpoint_id_raises(self):
        with pytest.raises(Exception):
            EndpointConfig(api_key="sk-test")  # type: ignore[call-arg]

    def test_missing_api_key_raises(self):
        with pytest.raises(Exception):
            EndpointConfig(endpoint_id="ep-1")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TestLLMConfig
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestLLMConfig:
    def test_create_minimal(self):
        llm = LLMConfig(
            task_model=_sample_model_dict(),
            optimizer_model=_sample_model_dict(name="openai/gpt-4o"),
            gradient_model=_sample_model_dict(name="openai/gpt-4o"),
            loss_model=_sample_model_dict(name="openai/gpt-4o-mini"),
        )
        assert llm.task_model.name == "openai/gpt-4o-mini"
        assert llm.optimizer_model.name == "openai/gpt-4o"
        assert llm.gradient_model.name == "openai/gpt-4o"
        assert llm.loss_model.name == "openai/gpt-4o-mini"

    def test_defaults(self):
        llm = LLMConfig(**_sample_llm_dict())
        assert llm.endpoints == {}
        assert llm.api_base is None
        assert llm.load_balancing == "RoundRobin"
        assert llm.endpoint_env_vars == []

    def test_with_endpoints(self):
        llm = LLMConfig(
            **_sample_llm_dict(
                endpoints={
                    "ep1": {"endpoint_id": "ep-1", "api_key": "sk-1"},
                    "ep2": {"endpoint_id": "ep-2", "api_key": "sk-2"},
                }
            )
        )
        assert len(llm.endpoints) == 2
        assert llm.endpoints["ep1"].endpoint_id == "ep-1"
        assert llm.endpoints["ep2"].api_key == "sk-2"

    def test_with_api_base(self):
        llm = LLMConfig(**_sample_llm_dict(api_base="http://localhost:8000/v1"))
        assert llm.api_base == "http://localhost:8000/v1"

    def test_missing_task_model_raises(self):
        d = _sample_llm_dict()
        del d["task_model"]
        with pytest.raises(Exception):
            LLMConfig(**d)

    def test_missing_optimizer_model_raises(self):
        d = _sample_llm_dict()
        del d["optimizer_model"]
        with pytest.raises(Exception):
            LLMConfig(**d)


# ---------------------------------------------------------------------------
# TestExperimentConfig
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestExperimentConfig:
    def test_create_minimal(self):
        cfg = ExperimentConfig(
            llm=LLMConfig(**_sample_llm_dict()),
            dataset="SummEval",
        )
        assert cfg.dataset == "SummEval"
        assert isinstance(cfg.llm, LLMConfig)

    def test_defaults(self):
        cfg = ExperimentConfig(
            llm=LLMConfig(**_sample_llm_dict()),
            dataset="SummEval",
        )
        assert cfg.output_dir == "results"
        assert cfg.steps == 10
        assert cfg.batch_size == 10
        assert cfg.eval_every == 1
        assert cfg.seed == 42
        assert cfg.checkpoint_every == 1
        assert cfg.verbosity == 1

    def test_override_all_fields(self):
        cfg = ExperimentConfig(
            llm=LLMConfig(**_sample_llm_dict()),
            dataset="BRIGHTER",
            output_dir="/tmp/out",
            steps=50,
            batch_size=5,
            eval_every=5,
            seed=123,
            checkpoint_every=10,
            verbosity=3,
        )
        assert cfg.dataset == "BRIGHTER"
        assert cfg.output_dir == "/tmp/out"
        assert cfg.steps == 50
        assert cfg.batch_size == 5
        assert cfg.eval_every == 5
        assert cfg.seed == 123
        assert cfg.checkpoint_every == 10
        assert cfg.verbosity == 3

    def test_llm_from_dict_coerced(self):
        exp_dict = _sample_experiment_dict()
        cfg = ExperimentConfig(**exp_dict)
        assert isinstance(cfg.llm, LLMConfig)
        assert cfg.llm.task_model.name == "openai/gpt-4o-mini"

    def test_missing_llm_raises(self):
        with pytest.raises(Exception):
            ExperimentConfig(dataset="SummEval")  # type: ignore[call-arg]

    def test_missing_dataset_raises(self):
        with pytest.raises(Exception):
            ExperimentConfig(llm=LLMConfig(**_sample_llm_dict()))  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TestLoadConfig
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestLoadConfig:
    def test_load_with_inline_llm_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "exp.json"
            with open(cfg_path, "w") as f:
                json.dump(_sample_experiment_dict(), f)

            cfg = load_config(str(cfg_path))
            assert isinstance(cfg, ExperimentConfig)
            assert cfg.dataset == "SummEval"
            assert isinstance(cfg.llm, LLMConfig)
            assert cfg.llm.task_model.name == "openai/gpt-4o-mini"
            assert cfg.llm.optimizer_model.name == "openai/gpt-4o"

    def test_load_with_inline_llm_preserves_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "exp.json"
            exp_dict = _sample_experiment_dict(
                llm=_sample_llm_dict(
                    endpoints={
                        "ep1": {"endpoint_id": "ep-1", "api_key": "sk-1"},
                    }
                )
            )
            with open(cfg_path, "w") as f:
                json.dump(exp_dict, f)

            cfg = load_config(str(cfg_path))
            assert len(cfg.llm.endpoints) == 1
            assert cfg.llm.endpoints["ep1"].endpoint_id == "ep-1"

    def test_load_with_llm_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            llm_path = tmp_path / "llm.json"
            with open(llm_path, "w") as f:
                json.dump(_sample_llm_dict(), f)

            exp_path = tmp_path / "exp.json"
            with open(exp_path, "w") as f:
                json.dump(
                    {
                        "llm": "./llm.json",
                        "dataset": "SummEval",
                    },
                    f,
                )

            cfg = load_config(str(exp_path))
            assert isinstance(cfg.llm, LLMConfig)
            assert cfg.llm.task_model.name == "openai/gpt-4o-mini"
            assert cfg.llm.optimizer_model.name == "openai/gpt-4o"
            assert cfg.dataset == "SummEval"

    def test_load_with_llm_bare_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            llm_path = tmp_path / "llm.json"
            with open(llm_path, "w") as f:
                json.dump(_sample_llm_dict(), f)

            exp_path = tmp_path / "exp.json"
            with open(exp_path, "w") as f:
                json.dump(
                    {
                        "llm": "llm.json",
                        "dataset": "SummEval",
                    },
                    f,
                )

            cfg = load_config(str(exp_path))
            assert isinstance(cfg.llm, LLMConfig)
            assert cfg.llm.task_model.name == "openai/gpt-4o-mini"

    def test_load_with_llm_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            llm_path = tmp_path / "llm.json"
            with open(llm_path, "w") as f:
                json.dump(_sample_llm_dict(), f)

            other_dir = tmp_path / "other"
            other_dir.mkdir()
            exp_path = other_dir / "exp.json"
            with open(exp_path, "w") as f:
                json.dump(
                    {
                        "llm": str(llm_path),
                        "dataset": "SummEval",
                    },
                    f,
                )

            cfg = load_config(str(exp_path))
            assert isinstance(cfg.llm, LLMConfig)
            assert cfg.llm.task_model.name == "openai/gpt-4o-mini"

    def test_load_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does_not_exist.json"
            with pytest.raises(FileNotFoundError):
                load_config(str(missing))

    def test_load_missing_llm_path_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "exp.json"
            with open(cfg_path, "w") as f:
                json.dump(
                    {
                        "llm": "./nonexistent_llm.json",
                        "dataset": "SummEval",
                    },
                    f,
                )

            with pytest.raises(FileNotFoundError):
                load_config(str(cfg_path))

    def test_load_invalid_llm_type_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "exp.json"
            with open(cfg_path, "w") as f:
                json.dump(
                    {
                        "llm": 12345,
                        "dataset": "SummEval",
                    },
                    f,
                )

            with pytest.raises(ValueError):
                load_config(str(cfg_path))

    def test_load_invalid_llm_none_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "exp.json"
            with open(cfg_path, "w") as f:
                json.dump(
                    {
                        "llm": None,
                        "dataset": "SummEval",
                    },
                    f,
                )

            with pytest.raises(ValueError):
                load_config(str(cfg_path))

    def test_load_invalid_llm_list_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "exp.json"
            with open(cfg_path, "w") as f:
                json.dump(
                    {
                        "llm": ["not", "a", "dict"],
                        "dataset": "SummEval",
                    },
                    f,
                )

            with pytest.raises(ValueError):
                load_config(str(cfg_path))
