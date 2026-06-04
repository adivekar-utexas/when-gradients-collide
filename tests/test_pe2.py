"""
Unit tests for PE2 pipeline components: PE2LossComputer, PE2GradientComputer,
PE2Optimizer.

Tests validate the actual construction of inputs (prompts sent to LLMs)
and outputs (parsed results) at each stage, without making real LLM calls.
"""

import json

import pytest

from prompt_moo.data_structures import (
    DatasetSample,
    NumericFeedback,
    PredictionResult,
    Task,
    TextGradient,
)
from prompt_moo.algorithm.pe2 import PE2GradientComputer
from prompt_moo.algorithm.pe2 import PE2LossComputer
from prompt_moo.algorithm.pe2 import PE2Optimizer
from prompt_moo.prompt_template import PromptTemplate

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FLUENCY = Task(
    task_name="fluency",
    task_description="Evaluate fluency and readability",
    task_instruction="Rate from 1 to 5.",
    gt_col="fluency",
)
COHERENCE = Task(
    task_name="coherence",
    task_description="Evaluate logical structure",
    task_instruction="Rate from 1 to 5.",
    gt_col="coherence",
)
TASKS = [FLUENCY, COHERENCE]

SKELETON = (
    "Evaluate the summary. Output JSON with the requested task scores. "
    "Do NOT include reasoning or explanations.\n"
    'Output format: {"fluency": <1-5>, "coherence": <1-5>}\n'
)


def _make_template() -> PromptTemplate:
    return PromptTemplate(
        skeleton=SKELETON,
        instruction={
            "fluency": FLUENCY.task_instruction,
            "coherence": COHERENCE.task_instruction,
        },
        tasks=TASKS,
        input_col_labels={"text": "text", "summary": "summary"},
    )


def _make_predictions_and_gts():
    """Create 4 samples: 2 failures, 2 successes."""
    preds = [
        PredictionResult(
            sample_id="s0",
            prompt="test_prompt",
            task_outputs={"fluency": 3, "coherence": 2},
            raw_response='{"fluency": 3, "coherence": 2}',
        ),
        PredictionResult(
            sample_id="s1",
            prompt="test_prompt",
            task_outputs={"fluency": 5, "coherence": 5},
            raw_response='{"fluency": 5, "coherence": 5}',
        ),
        PredictionResult(
            sample_id="s2",
            prompt="test_prompt",
            task_outputs={"fluency": 2, "coherence": 4},
            raw_response='{"fluency": 2, "coherence": 4}',
        ),
        PredictionResult(
            sample_id="s3",
            prompt="test_prompt",
            task_outputs={"fluency": 4, "coherence": 3},
            raw_response='{"fluency": 4, "coherence": 3}',
        ),
    ]
    gts = [
        DatasetSample(
            sample_id="s0",
            inputs={"text": "The cat sat on the mat.", "summary": "Cat on mat."},
            ground_truths={"fluency": 4, "coherence": 3},
        ),
        DatasetSample(
            sample_id="s1",
            inputs={"text": "The dog ran fast.", "summary": "Fast dog."},
            ground_truths={"fluency": 5, "coherence": 5},
        ),
        DatasetSample(
            sample_id="s2",
            inputs={"text": "Rain fell all day.", "summary": "Rainy day."},
            ground_truths={"fluency": 4, "coherence": 4},
        ),
        DatasetSample(
            sample_id="s3",
            inputs={"text": "Stars filled the sky.", "summary": "Starry sky."},
            ground_truths={"fluency": 4, "coherence": 3},
        ),
    ]
    return preds, gts


# ===========================================================================
# PE2LossComputer
# ===========================================================================
@pytest.mark.unit
class TestPE2LossComputer:
    """PE2LossComputer produces numeric-only feedback, no LLM calls."""

    def test_registry_lookup(self):
        from prompt_moo.loss_computer import LossComputer

        instance = LossComputer.of("pe2")
        assert instance.__class__.__name__ == "PE2LossComputer"

    def test_returns_numeric_feedback_only(self):
        preds, gts = _make_predictions_and_gts()
        comp = PE2LossComputer()
        result = comp.compute(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            prompt_template=_make_template(),
            llm_pool=None,
            loss_batch_size=4,
            verbosity=0,
            loss_functions={
                "fluency": {"metric": "accuracy"},
                "coherence": {"metric": "accuracy"},
            },
        )
        assert FLUENCY in result
        assert COHERENCE in result
        for task, feedbacks in result.items():
            for fb in feedbacks:
                assert isinstance(fb, NumericFeedback)

    def test_accuracy_values_correct(self):
        """s0: flu pred=3, gt=4 (wrong); s1: flu pred=5, gt=5 (right);
        s2: flu pred=2, gt=4 (wrong); s3: flu pred=4, gt=4 (right).
        Accuracy = 2/4 = 0.5
        """
        preds, gts = _make_predictions_and_gts()
        comp = PE2LossComputer()
        result = comp.compute(
            predictions=preds,
            ground_truths=gts,
            tasks=[FLUENCY],
            prompt_template=_make_template(),
            llm_pool=None,
            loss_batch_size=4,
            verbosity=0,
            loss_functions={"fluency": {"metric": "accuracy"}},
        )
        flu_feedbacks = result[FLUENCY]
        assert len(flu_feedbacks) == 1
        assert flu_feedbacks[0].value == pytest.approx(0.5)

    def test_requires_loss_functions_kwarg(self):
        preds, gts = _make_predictions_and_gts()
        comp = PE2LossComputer()
        with pytest.raises(ValueError, match="loss_functions"):
            comp.compute(
                predictions=preds,
                ground_truths=gts,
                tasks=TASKS,
                prompt_template=_make_template(),
                llm_pool=None,
                loss_batch_size=4,
            )

    def test_no_llm_calls(self):
        """PE2LossComputer must work with llm_pool=None."""
        preds, gts = _make_predictions_and_gts()
        comp = PE2LossComputer()
        result = comp.compute(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            prompt_template=_make_template(),
            llm_pool=None,
            loss_batch_size=2,
            verbosity=0,
            loss_functions={
                "fluency": {"metric": "accuracy"},
                "coherence": {"metric": "accuracy"},
            },
        )
        assert len(result) == 2


# ===========================================================================
# PE2GradientComputer — prompt construction
# ===========================================================================
@pytest.mark.unit
class TestPE2GradientComputerPromptConstruction:
    """Tests for the Step 1 prompt built by PE2GradientComputer.

    These tests call the internal _build_pe2_step1_prompt and
    _format_examples methods directly (no LLM calls).
    """

    def test_registry_lookup(self):
        from prompt_moo.gradient_computer import GradientComputer

        instance = GradientComputer.of("pe2")
        assert instance.__class__.__name__ == "PE2GradientComputer"

    def test_step1_prompt_has_system_message(self):
        comp = PE2GradientComputer()
        preds, gts = _make_predictions_and_gts()
        prompt = comp._build_pe2_step1_prompt(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            prompt_template=_make_template(),
            full_template="{{prompt}}\n{{input}}",
            batch_size_shown=2,
            input_col_labels={"text": "text", "summary": "summary"},
        )
        assert "[System] You are a helpful assistant." in prompt

    def test_step1_prompt_has_two_step_task_description(self):
        comp = PE2GradientComputer()
        preds, gts = _make_predictions_and_gts()
        prompt = comp._build_pe2_step1_prompt(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            prompt_template=_make_template(),
            full_template="{{prompt}}\n{{input}}",
            batch_size_shown=2,
            input_col_labels={"text": "text", "summary": "summary"},
        )
        assert "## Step 1" in prompt
        assert "## Step 2" in prompt
        assert "two main steps" in prompt

    def test_step1_prompt_has_scripted_acknowledgment(self):
        comp = PE2GradientComputer()
        preds, gts = _make_predictions_and_gts()
        prompt = comp._build_pe2_step1_prompt(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            prompt_template=_make_template(),
            full_template="{{prompt}}\n{{input}}",
            batch_size_shown=2,
            input_col_labels={"text": "text", "summary": "summary"},
        )
        assert "Sure, I'd be happy to help" in prompt

    def test_step1_prompt_has_current_prompt_section(self):
        comp = PE2GradientComputer()
        preds, gts = _make_predictions_and_gts()
        tmpl = _make_template()
        prompt = comp._build_pe2_step1_prompt(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            prompt_template=tmpl,
            full_template="{{prompt}}\n{{input}}",
            batch_size_shown=2,
            input_col_labels={"text": "text", "summary": "summary"},
        )
        assert "## Prompt" in prompt
        assert "Evaluate the summary" in prompt

    def test_step1_prompt_has_full_template_section(self):
        comp = PE2GradientComputer()
        preds, gts = _make_predictions_and_gts()
        prompt = comp._build_pe2_step1_prompt(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            prompt_template=_make_template(),
            full_template="{{prompt}}\n{{input}}",
            batch_size_shown=2,
            input_col_labels={"text": "text", "summary": "summary"},
        )
        assert "## Full Template" in prompt
        assert "{{prompt}}" in prompt
        assert "{{input}}" in prompt

    def test_step1_prompt_has_five_reasoning_questions(self):
        comp = PE2GradientComputer()
        preds, gts = _make_predictions_and_gts()
        prompt = comp._build_pe2_step1_prompt(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            prompt_template=_make_template(),
            full_template="{{prompt}}\n{{input}}",
            batch_size_shown=2,
            input_col_labels={"text": "text", "summary": "summary"},
        )
        assert "Is the output correct compared to the label" in prompt
        assert "Is the output correctly following the given prompt" in prompt
        assert "Is the prompt correctly describing the task" in prompt
        assert "is it necessary to edit the prompt" in prompt
        assert "actionable suggestions to edit the prompt" in prompt

    def test_step1_prompt_has_ground_truth_absoluteness_statement(self):
        comp = PE2GradientComputer()
        preds, gts = _make_predictions_and_gts()
        prompt = comp._build_pe2_step1_prompt(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            prompt_template=_make_template(),
            full_template="{{prompt}}\n{{input}}",
            batch_size_shown=2,
            input_col_labels={"text": "text", "summary": "summary"},
        )
        assert "ground-truth labels are __absolutely correct__" in prompt

    def test_step1_prompt_includes_example_count(self):
        comp = PE2GradientComputer()
        preds, gts = _make_predictions_and_gts()
        prompt = comp._build_pe2_step1_prompt(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            prompt_template=_make_template(),
            full_template="{{prompt}}\n{{input}}",
            batch_size_shown=3,
            input_col_labels={"text": "text", "summary": "summary"},
        )
        assert "3 example(s)" in prompt

    def test_step1_prompt_contains_examples_section(self):
        comp = PE2GradientComputer()
        preds, gts = _make_predictions_and_gts()
        prompt = comp._build_pe2_step1_prompt(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            prompt_template=_make_template(),
            full_template="{{prompt}}\n{{input}}",
            batch_size_shown=2,
            input_col_labels={"text": "text", "summary": "summary"},
        )
        assert "## Examples" in prompt
        assert "### Example 1" in prompt
        assert "### Example 2" in prompt


# ===========================================================================
# PE2GradientComputer — hard negative sampling
# ===========================================================================
@pytest.mark.unit
class TestPE2HardNegativeSampling:
    """Tests for _format_examples: failure prioritization."""

    def test_failures_prioritized_over_successes(self):
        """With max_examples=2, both selected should be failures (s0 and s2)."""
        comp = PE2GradientComputer()
        preds, gts = _make_predictions_and_gts()
        text = comp._format_examples(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            max_examples=2,
            input_col_labels={"text": "text", "summary": "summary"},
        )
        assert "### Example 1" in text
        assert "### Example 2" in text
        assert "cat sat" in text.lower() or "rain fell" in text.lower()

    def test_successes_fill_when_not_enough_failures(self):
        """With max_examples=4 and only 2 failures, successes fill the rest."""
        comp = PE2GradientComputer()
        preds, gts = _make_predictions_and_gts()
        text = comp._format_examples(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            max_examples=4,
            input_col_labels={"text": "text", "summary": "summary"},
        )
        assert "### Example 1" in text
        assert "### Example 4" in text

    def test_all_correct_uses_successes(self):
        """If all predictions are correct, successes are used."""
        comp = PE2GradientComputer()
        preds = [
            PredictionResult(
                sample_id="s0",
                prompt="test_prompt",
                task_outputs={"fluency": 4},
                raw_response='{"fluency": 4}',
            ),
        ]
        gts = [
            DatasetSample(
                sample_id="s0",
                inputs={"text": "hello"},
                ground_truths={"fluency": 4},
            ),
        ]
        text = comp._format_examples(
            predictions=preds,
            ground_truths=gts,
            tasks=[FLUENCY],
            max_examples=2,
            input_col_labels={"text": "text"},
        )
        assert "### Example 1" in text

    def test_example_contains_input_output_label(self):
        """Each example must have Input, Output, and Label lines."""
        comp = PE2GradientComputer()
        preds, gts = _make_predictions_and_gts()
        text = comp._format_examples(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            max_examples=1,
            input_col_labels={"text": "text", "summary": "summary"},
        )
        assert "Input:" in text
        assert "Output:" in text
        assert "Label:" in text

    def test_output_shows_task_predictions(self):
        """Output line should contain task_name=value pairs."""
        comp = PE2GradientComputer()
        preds, gts = _make_predictions_and_gts()
        text = comp._format_examples(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            max_examples=1,
            input_col_labels={"text": "text", "summary": "summary"},
        )
        assert "fluency=" in text
        assert "coherence=" in text

    def test_label_shows_ground_truths(self):
        """Label line should contain task_name=gt_value pairs."""
        comp = PE2GradientComputer()
        preds, gts = _make_predictions_and_gts()
        text = comp._format_examples(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            max_examples=1,
            input_col_labels={"text": "text", "summary": "summary"},
        )
        lines = text.strip().split("\n")
        label_lines = [l for l in lines if l.startswith("Label:")]
        assert len(label_lines) >= 1
        label_line = label_lines[0]
        assert "fluency=" in label_line
        assert "coherence=" in label_line

    def test_long_input_not_truncated(self):
        """Long inputs are passed through in full (no truncation)."""
        comp = PE2GradientComputer()
        long_text = "x" * 500
        preds = [
            PredictionResult(
                sample_id="s0",
                prompt="test_prompt",
                task_outputs={"fluency": 3},
                raw_response='{"fluency": 3}',
            ),
        ]
        gts = [
            DatasetSample(
                sample_id="s0",
                inputs={"text": long_text},
                ground_truths={"fluency": 4},
            ),
        ]
        text = comp._format_examples(
            predictions=preds,
            ground_truths=gts,
            tasks=[FLUENCY],
            max_examples=1,
            input_col_labels={"text": "text"},
        )
        assert long_text in text


# ===========================================================================
# PE2Optimizer — meta-prompt construction
# ===========================================================================
@pytest.mark.unit
class TestPE2OptimizerMetaPrompt:
    """Tests for PE2Optimizer.create_meta_prompt construction."""

    def test_registry_lookup(self):
        from prompt_moo.prompt_optimizer import PromptOptimizer

        instance = PromptOptimizer.of("pe2")
        assert instance.__class__.__name__ == "PE2Optimizer"

    def _build_meta_prompt(self, step_size=None, max_prompt_tokens=50):
        opt = PE2Optimizer()
        tmpl = _make_template()
        gradients = {
            FLUENCY: [
                TextGradient(
                    task_name="all_tasks",
                    gradient_text="The model overpredicts fluency.",
                    based_on_feedbacks=[],
                    gradient_prompt="step1 prompt here",
                )
            ],
            COHERENCE: [
                TextGradient(
                    task_name="all_tasks",
                    gradient_text="The model overpredicts fluency.",
                    based_on_feedbacks=[],
                    gradient_prompt="step1 prompt here",
                )
            ],
        }
        conversation = {
            "step1_prompt": "[System] You are a helpful assistant.\n[User] ...step1...",
            "step1_reasoning": "### Example 1\nThe output is wrong because...",
        }
        kwargs = {
            "pe2_conversation": conversation,
            "max_prompt_tokens": max_prompt_tokens,
            "use_textual_feedback": False,
        }
        if step_size is not None:
            kwargs["step_size"] = step_size
        return opt.create_meta_prompt(
            gradients=gradients,
            current_prompt=tmpl,
            tasks=TASKS,
            **kwargs,
        )

    def test_meta_prompt_contains_step1_conversation(self):
        mp = self._build_meta_prompt()
        assert "[System] You are a helpful assistant." in mp
        assert "...step1..." in mp

    def test_meta_prompt_contains_step1_reasoning(self):
        mp = self._build_meta_prompt()
        assert "The output is wrong because" in mp

    def test_meta_prompt_contains_step2_instruction(self):
        mp = self._build_meta_prompt()
        assert "Step 1 and help with Step 2" in mp

    def test_meta_prompt_contains_current_prompt_repeated(self):
        mp = self._build_meta_prompt()
        assert "## Current Prompt" in mp
        assert "Evaluate the summary" in mp

    def test_meta_prompt_contains_json_format_instruction(self):
        mp = self._build_meta_prompt()
        assert '"instructions"' in mp
        assert '"fluency"' in mp
        assert '"coherence"' in mp

    def test_meta_prompt_contains_max_tokens_constraint(self):
        mp = self._build_meta_prompt(max_prompt_tokens=75)
        assert "less than 75 words" in mp

    def test_meta_prompt_no_step_size_by_default(self):
        mp = self._build_meta_prompt()
        assert "allowed to change up to" not in mp

    def test_meta_prompt_with_step_size(self):
        mp = self._build_meta_prompt(step_size=10)
        assert "allowed to change up to 10 words" in mp

    def test_meta_prompt_role_markers(self):
        """The conversation continuation must have [Assistant] and [User] markers."""
        mp = self._build_meta_prompt()
        assert "[Assistant]" in mp
        assert "[User]" in mp

    def test_meta_prompt_ends_with_no_extra_text(self):
        mp = self._build_meta_prompt()
        assert "Do not include any text outside the JSON" in mp


# ===========================================================================
# PE2Optimizer — response parsing
# ===========================================================================
@pytest.mark.unit
class TestPE2OptimizerParsing:
    """Tests for PE2Optimizer.parse_meta_prompt_response."""

    def test_parse_clean_json(self):
        opt = PE2Optimizer()
        response = json.dumps(
            {
                "instructions": {
                    "fluency": "Improved fluency instruction",
                    "coherence": "Improved coherence instruction",
                }
            }
        )
        result = opt.parse_meta_prompt_response(response=response, tasks=TASKS)
        assert result["fluency"] == "Improved fluency instruction"
        assert result["coherence"] == "Improved coherence instruction"

    def test_parse_json_with_surrounding_text(self):
        opt = PE2Optimizer()
        response = (
            "Here is the improved prompt:\n"
            '{"instructions": {"fluency": "new flu", "coherence": "new coh"}}\n'
            "Done."
        )
        result = opt.parse_meta_prompt_response(response=response, tasks=TASKS)
        assert result["fluency"] == "new flu"
        assert result["coherence"] == "new coh"

    def test_parse_missing_task_falls_back(self):
        """If the LLM omits a task, fallback to the task's original instruction."""
        opt = PE2Optimizer()
        response = json.dumps(
            {
                "instructions": {
                    "fluency": "new fluency",
                }
            }
        )
        result = opt.parse_meta_prompt_response(response=response, tasks=TASKS)
        assert result["fluency"] == "new fluency"
        assert result["coherence"] == COHERENCE.task_instruction

    def test_parse_flat_dict_format(self):
        """PE2Optimizer accepts flat dict (no "instructions" wrapper)."""
        opt = PE2Optimizer()
        response = json.dumps(
            {
                "fluency": "flat flu",
                "coherence": "flat coh",
            }
        )
        result = opt.parse_meta_prompt_response(response=response, tasks=TASKS)
        assert result["fluency"] == "flat flu"

    def test_parse_no_json_raises(self):
        opt = PE2Optimizer()
        with pytest.raises(ValueError, match="No JSON"):
            opt.parse_meta_prompt_response(response="No JSON here", tasks=TASKS)

    def test_parse_invalid_json_raises(self):
        opt = PE2Optimizer()
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            opt.parse_meta_prompt_response(
                response="{this is not valid json}", tasks=TASKS
            )

    def test_parse_double_braces_normalized(self):
        opt = PE2Optimizer()
        response = '{{"instructions": {{"fluency": "fixed", "coherence": "fixed"}}}}'
        result = opt.parse_meta_prompt_response(response=response, tasks=TASKS)
        assert result["fluency"] == "fixed"


# ===========================================================================
# End-to-end prompt flow: gradient prompt -> optimizer prompt continuity
# ===========================================================================
@pytest.mark.unit
class TestPE2ConversationContinuity:
    """The optimizer meta-prompt must start with the gradient's Step 1 prompt
    and include the reasoning, forming one continuous conversation."""

    def test_optimizer_prompt_starts_with_step1(self):
        comp = PE2GradientComputer()
        preds, gts = _make_predictions_and_gts()
        tmpl = _make_template()

        step1_prompt = comp._build_pe2_step1_prompt(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            prompt_template=tmpl,
            full_template="{{prompt}}\n{{input}}",
            batch_size_shown=2,
            input_col_labels={"text": "text", "summary": "summary"},
        )

        fake_reasoning = "### Example 1\nOutput is wrong because fluency=3 but label=4."

        opt = PE2Optimizer()
        meta_prompt = opt.create_meta_prompt(
            gradients={t: [] for t in TASKS},
            current_prompt=tmpl,
            tasks=TASKS,
            pe2_conversation={
                "step1_prompt": step1_prompt,
                "step1_reasoning": fake_reasoning,
            },
            max_prompt_tokens=50,
        )

        assert meta_prompt.startswith(step1_prompt)

    def test_reasoning_appears_before_step2(self):
        comp = PE2GradientComputer()
        preds, gts = _make_predictions_and_gts()
        tmpl = _make_template()

        step1_prompt = comp._build_pe2_step1_prompt(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            prompt_template=tmpl,
            full_template="{{prompt}}\n{{input}}",
            batch_size_shown=2,
            input_col_labels={"text": "text", "summary": "summary"},
        )

        fake_reasoning = "UNIQUE_REASONING_MARKER_XYZ"

        opt = PE2Optimizer()
        meta_prompt = opt.create_meta_prompt(
            gradients={t: [] for t in TASKS},
            current_prompt=tmpl,
            tasks=TASKS,
            pe2_conversation={
                "step1_prompt": step1_prompt,
                "step1_reasoning": fake_reasoning,
            },
            max_prompt_tokens=50,
        )

        reasoning_pos = meta_prompt.index("UNIQUE_REASONING_MARKER_XYZ")
        step2_pos = meta_prompt.index("help with Step 2")
        assert reasoning_pos < step2_pos

    def test_five_questions_in_final_optimizer_prompt(self):
        """The five PE2 reasoning questions from Step 1 should be present
        in the optimizer meta-prompt (because it includes the full Step 1)."""
        comp = PE2GradientComputer()
        preds, gts = _make_predictions_and_gts()
        tmpl = _make_template()

        step1_prompt = comp._build_pe2_step1_prompt(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            prompt_template=tmpl,
            full_template="{{prompt}}\n{{input}}",
            batch_size_shown=2,
            input_col_labels={"text": "text", "summary": "summary"},
        )

        opt = PE2Optimizer()
        meta_prompt = opt.create_meta_prompt(
            gradients={t: [] for t in TASKS},
            current_prompt=tmpl,
            tasks=TASKS,
            pe2_conversation={
                "step1_prompt": step1_prompt,
                "step1_reasoning": "some reasoning",
            },
            max_prompt_tokens=50,
        )

        assert "Is the output correct compared to the label" in meta_prompt
        assert "absolutely correct" in meta_prompt


# ===========================================================================
# Golden-value tests: exact expected inputs and outputs
# ===========================================================================


def _make_minimal_data():
    """Single failure example for deterministic golden-value tests."""
    preds = [
        PredictionResult(
            sample_id="g0",
            prompt="test_prompt",
            task_outputs={"fluency": 3},
            raw_response='{"fluency": 3}',
        ),
    ]
    gts = [
        DatasetSample(
            sample_id="g0",
            inputs={"text": "Hello world."},
            ground_truths={"fluency": 4},
        ),
    ]
    return preds, gts


SINGLE_TASK = [FLUENCY]

SINGLE_SKELETON = 'Evaluate the summary.\nOutput format: {"fluency": <1-5>}\n'


def _make_single_template():
    return PromptTemplate(
        skeleton=SINGLE_SKELETON,
        instruction={"fluency": FLUENCY.task_instruction},
        tasks=SINGLE_TASK,
        input_col_labels={"text": "text"},
    )


@pytest.mark.unit
class TestPE2GoldenExampleFormatting:
    """Assert exact string output of _format_examples for a known single-sample input."""

    def test_single_failure_exact_output(self):
        comp = PE2GradientComputer()
        preds, gts = _make_minimal_data()
        text = comp._format_examples(
            predictions=preds,
            ground_truths=gts,
            tasks=SINGLE_TASK,
            max_examples=1,
            input_col_labels={"text": "text"},
        )
        expected = (
            "### Example 1\n"
            "Input: text: Hello world.\n"
            "Output: fluency=3\n"
            "Label: fluency=4\n"
        )
        assert text == expected

    def test_two_examples_first_failure_then_success(self):
        """With 1 failure + 1 success and max_examples=2, failure comes first."""
        comp = PE2GradientComputer()
        preds = [
            PredictionResult(
                sample_id="a",
                prompt="test_prompt",
                task_outputs={"fluency": 3},
                raw_response='{"fluency": 3}',
            ),
            PredictionResult(
                sample_id="b",
                prompt="test_prompt",
                task_outputs={"fluency": 5},
                raw_response='{"fluency": 5}',
            ),
        ]
        gts = [
            DatasetSample(
                sample_id="a",
                inputs={"text": "Bad."},
                ground_truths={"fluency": 5},
            ),
            DatasetSample(
                sample_id="b",
                inputs={"text": "Good."},
                ground_truths={"fluency": 5},
            ),
        ]
        text = comp._format_examples(
            predictions=preds,
            ground_truths=gts,
            tasks=SINGLE_TASK,
            max_examples=2,
            input_col_labels={"text": "text"},
        )
        expected = (
            "### Example 1\n"
            "Input: text: Bad.\n"
            "Output: fluency=3\n"
            "Label: fluency=5\n"
            "\n"
            "### Example 2\n"
            "Input: text: Good.\n"
            "Output: fluency=5\n"
            "Label: fluency=5\n"
        )
        assert text == expected

    def test_multi_task_example_exact_output(self):
        comp = PE2GradientComputer()
        preds = [
            PredictionResult(
                sample_id="m0",
                prompt="test_prompt",
                task_outputs={"fluency": 2, "coherence": 1},
                raw_response='{"fluency": 2, "coherence": 1}',
            ),
        ]
        gts = [
            DatasetSample(
                sample_id="m0",
                inputs={"text": "Test."},
                ground_truths={"fluency": 4, "coherence": 3},
            ),
        ]
        text = comp._format_examples(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            max_examples=1,
            input_col_labels={"text": "text"},
        )
        expected = (
            "### Example 1\n"
            "Input: text: Test.\n"
            "Output: fluency=2, coherence=1\n"
            "Label: fluency=4, coherence=3\n"
        )
        assert text == expected


@pytest.mark.unit
class TestPE2GoldenLossOutput:
    """Assert exact loss computer output for known inputs."""

    def test_single_task_all_wrong(self):
        """All predictions wrong -> accuracy = 0.0"""
        comp = PE2LossComputer()
        preds = [
            PredictionResult(
                sample_id="x",
                prompt="test_prompt",
                task_outputs={"fluency": 1},
                raw_response='{"fluency": 1}',
            ),
            PredictionResult(
                sample_id="y",
                prompt="test_prompt",
                task_outputs={"fluency": 2},
                raw_response='{"fluency": 2}',
            ),
        ]
        gts = [
            DatasetSample(sample_id="x", inputs={}, ground_truths={"fluency": 5}),
            DatasetSample(sample_id="y", inputs={}, ground_truths={"fluency": 5}),
        ]
        result = comp.compute(
            predictions=preds,
            ground_truths=gts,
            tasks=[FLUENCY],
            prompt_template=_make_single_template(),
            llm_pool=None,
            loss_batch_size=10,
            verbosity=0,
            loss_functions={"fluency": {"metric": "accuracy"}},
        )
        assert len(result[FLUENCY]) == 1
        fb = result[FLUENCY][0]
        assert fb.task_name == "fluency"
        assert fb.metric_name == "accuracy"
        assert fb.value == pytest.approx(0.0)
        assert fb.optimization_direction == "maximize"
        assert set(fb.aggregated_from_samples) == {"x", "y"}

    def test_single_task_all_correct(self):
        comp = PE2LossComputer()
        preds = [
            PredictionResult(
                sample_id="x",
                prompt="test_prompt",
                task_outputs={"fluency": 5},
                raw_response='{"fluency": 5}',
            ),
        ]
        gts = [
            DatasetSample(sample_id="x", inputs={}, ground_truths={"fluency": 5}),
        ]
        result = comp.compute(
            predictions=preds,
            ground_truths=gts,
            tasks=[FLUENCY],
            prompt_template=_make_single_template(),
            llm_pool=None,
            loss_batch_size=10,
            verbosity=0,
            loss_functions={"fluency": {"metric": "accuracy"}},
        )
        assert result[FLUENCY][0].value == pytest.approx(1.0)

    def test_multi_task_independent_accuracy(self):
        """Each task's accuracy is computed independently."""
        comp = PE2LossComputer()
        preds = [
            PredictionResult(
                sample_id="x",
                prompt="test_prompt",
                task_outputs={"fluency": 5, "coherence": 1},
                raw_response="{}",
            ),
        ]
        gts = [
            DatasetSample(
                sample_id="x", inputs={}, ground_truths={"fluency": 5, "coherence": 3}
            ),
        ]
        result = comp.compute(
            predictions=preds,
            ground_truths=gts,
            tasks=TASKS,
            prompt_template=_make_template(),
            llm_pool=None,
            loss_batch_size=10,
            verbosity=0,
            loss_functions={
                "fluency": {"metric": "accuracy"},
                "coherence": {"metric": "accuracy"},
            },
        )
        assert result[FLUENCY][0].value == pytest.approx(1.0)
        assert result[COHERENCE][0].value == pytest.approx(0.0)


@pytest.mark.unit
class TestPE2GoldenOptimizerOutput:
    """Assert exact parsed output of PE2Optimizer for known LLM responses."""

    def test_exact_parsed_instructions(self):
        opt = PE2Optimizer()
        response = '{"instructions": {"fluency": "Be strict on grammar.", "coherence": "Check logical flow."}}'
        result = opt.parse_meta_prompt_response(response=response, tasks=TASKS)
        assert result == {
            "fluency": "Be strict on grammar.",
            "coherence": "Check logical flow.",
        }

    def test_missing_task_fallback_value(self):
        opt = PE2Optimizer()
        response = '{"instructions": {"fluency": "New instruction."}}'
        result = opt.parse_meta_prompt_response(response=response, tasks=TASKS)
        assert result == {
            "fluency": "New instruction.",
            "coherence": "Rate from 1 to 5.",
        }


@pytest.mark.unit
class TestPE2GoldenStep1PromptStructure:
    """Assert the exact top-level structure of the Step 1 prompt
    by checking ordered sections appear in the correct sequence."""

    def test_section_ordering(self):
        comp = PE2GradientComputer()
        preds, gts = _make_minimal_data()
        prompt = comp._build_pe2_step1_prompt(
            predictions=preds,
            ground_truths=gts,
            tasks=SINGLE_TASK,
            prompt_template=_make_single_template(),
            full_template="{{prompt}}\n{{input}}",
            batch_size_shown=1,
            input_col_labels={"text": "text"},
        )
        sections = [
            "[System] You are a helpful assistant.",
            "[User]",
            "## Step 1",
            "## Step 2",
            "[Assistant]",
            "Sure, I'd be happy to help",
            "[User]",
            "## Prompt",
            "## Full Template",
            "## Examples",
            "### Example 1",
            "## Instructions",
            "ground-truth labels are __absolutely correct__",
            "Is the output correct compared to the label",
            "Is the output correctly following the given prompt",
            "Is the prompt correctly describing the task",
            "is it necessary to edit the prompt",
            "actionable suggestions to edit the prompt",
        ]
        prev_pos = -1
        for section in sections:
            pos = prompt.find(section, prev_pos + 1)
            assert pos > prev_pos, (
                f"Section {section!r} not found after position {prev_pos}.\n"
                f"Prompt excerpt around expected location:\n"
                f"{prompt[max(0, prev_pos) : prev_pos + 200]}"
            )
            prev_pos = pos

    def test_prompt_embeds_actual_data(self):
        """The Step 1 prompt must contain the actual input text and pred/gt values."""
        comp = PE2GradientComputer()
        preds, gts = _make_minimal_data()
        prompt = comp._build_pe2_step1_prompt(
            predictions=preds,
            ground_truths=gts,
            tasks=SINGLE_TASK,
            prompt_template=_make_single_template(),
            full_template="{{prompt}}\n{{input}}",
            batch_size_shown=1,
            input_col_labels={"text": "text"},
        )
        assert "Hello world." in prompt
        assert "fluency=3" in prompt
        assert "fluency=4" in prompt


@pytest.mark.unit
class TestPE2GoldenMetaPromptStructure:
    """Assert the exact structure of the full optimizer meta-prompt
    (Step 1 prompt + reasoning + Step 2 instructions)."""

    def test_meta_prompt_section_ordering(self):
        comp = PE2GradientComputer()
        preds, gts = _make_minimal_data()
        tmpl = _make_single_template()
        step1_prompt = comp._build_pe2_step1_prompt(
            predictions=preds,
            ground_truths=gts,
            tasks=SINGLE_TASK,
            prompt_template=tmpl,
            full_template="{{prompt}}\n{{input}}",
            batch_size_shown=1,
            input_col_labels={"text": "text"},
        )
        reasoning = "The output fluency=3 does not match label fluency=4."

        opt = PE2Optimizer()
        meta = opt.create_meta_prompt(
            gradients={FLUENCY: []},
            current_prompt=tmpl,
            tasks=SINGLE_TASK,
            pe2_conversation={
                "step1_prompt": step1_prompt,
                "step1_reasoning": reasoning,
            },
            max_prompt_tokens=50,
        )
        sections = [
            "[System] You are a helpful assistant.",
            "## Prompt",
            "## Full Template",
            "## Examples",
            "## Instructions",
            "absolutely correct",
            "[Assistant]",
            "fluency=3 does not match label fluency=4",
            "[User]",
            "help with Step 2",
            "## Current Prompt",
            "less than 50 words",
            '"instructions"',
            "Do not include any text outside the JSON",
        ]
        prev_pos = -1
        for section in sections:
            pos = meta.find(section, prev_pos + 1)
            assert pos > prev_pos, (
                f"Section {section!r} not found after position {prev_pos}."
            )
            prev_pos = pos


# ===========================================================================
# E2E test: LossComputer -> GradientComputer -> Optimizer with mock LLM
# ===========================================================================


class _MockFuture:
    """Mimics a Concurry Future for test purposes."""

    def __init__(self, value):
        self._value = value

    def result(self, timeout=None):
        return self._value


class _MockLLMPool:
    """Mock LLM pool that records calls and returns canned responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._call_idx = 0
        self.recorded_prompts = []

    def call_llm_batch(self, prompts, verbosity=0, validator=None):
        self.recorded_prompts.append(prompts)
        resp = self._responses[self._call_idx]
        self._call_idx += 1
        if validator is not None:
            return _MockFuture([validator(r) for r in resp])
        return _MockFuture(resp)


@pytest.mark.unit
class TestPE2EndToEnd:
    """End-to-end test: wire PE2LossComputer -> PE2GradientComputer ->
    PE2Optimizer with a mock LLM and verify every intermediate artifact."""

    def test_full_pipeline(self):
        preds, gts = _make_minimal_data()
        tmpl = _make_single_template()

        # ---- Stage 1: Loss ----
        loss_comp = PE2LossComputer()
        loss_functions = {"fluency": {"metric": "accuracy"}}
        feedbacks = loss_comp.compute(
            predictions=preds,
            ground_truths=gts,
            tasks=SINGLE_TASK,
            prompt_template=tmpl,
            llm_pool=None,
            loss_batch_size=10,
            verbosity=0,
            loss_functions=loss_functions,
        )
        assert FLUENCY in feedbacks
        assert feedbacks[FLUENCY][0].value == pytest.approx(0.0)

        # ---- Stage 2: Gradient (with mock LLM) ----
        mock_reasoning = (
            "### Example 1\n"
            "Output: fluency=3\n"
            "Label: fluency=4\n"
            "Is the output correct: No, predicted 3 but label is 4.\n"
            "Is the prompt correctly describing the task: "
            "Partially, it says 'Rate fluency' but lacks detail.\n"
            "Is it necessary to edit: Yes.\n"
            "Suggestions: Add guidance about grammar and readability."
        )
        gradient_llm = _MockLLMPool(responses=[[mock_reasoning]])

        pe2_conversation = {}
        grad_comp = PE2GradientComputer()
        gradients = grad_comp.compute(
            feedbacks=feedbacks,
            prompt_template=tmpl,
            tasks=SINGLE_TASK,
            llm_pool=gradient_llm,
            gradient_batch_size=1,
            verbosity=0,
            predictions=preds,
            ground_truths=gts,
            pe2_conversation=pe2_conversation,
            pe2_batch_size=1,
        )

        assert FLUENCY in gradients
        assert len(gradients[FLUENCY]) == 1
        assert gradients[FLUENCY][0].gradient_text == mock_reasoning
        assert gradients[FLUENCY][0].task_name == "all_tasks"

        assert "step1_prompt" in pe2_conversation
        assert "step1_reasoning" in pe2_conversation
        assert pe2_conversation["step1_reasoning"] == mock_reasoning

        sent_to_gradient_llm = gradient_llm.recorded_prompts[0][0]
        assert "[System] You are a helpful assistant." in sent_to_gradient_llm
        assert "Hello world." in sent_to_gradient_llm
        assert "fluency=3" in sent_to_gradient_llm
        assert "fluency=4" in sent_to_gradient_llm
        assert "absolutely correct" in sent_to_gradient_llm

        # ---- Stage 3: Optimizer (with mock LLM) ----
        optimizer_json = json.dumps(
            {
                "instructions": {
                    "fluency": "Rate fluency 1-5. Consider grammar, clarity, readability."
                }
            }
        )
        optimizer_llm = _MockLLMPool(responses=[[optimizer_json]])

        opt = PE2Optimizer()
        optimizer_result = opt.optimize(
            gradients=gradients,
            current_prompt=tmpl,
            tasks=SINGLE_TASK,
            llm_pool=optimizer_llm,
            verbosity=0,
            pe2_conversation=pe2_conversation,
            max_prompt_tokens=50,
        )

        assert optimizer_result.new_prompt is not None
        new_instructions = optimizer_result.new_prompt.instruction
        assert new_instructions["fluency"] == (
            "Rate fluency 1-5. Consider grammar, clarity, readability."
        )

        sent_to_optimizer = optimizer_llm.recorded_prompts[0][0]
        assert mock_reasoning in sent_to_optimizer
        assert "help with Step 2" in sent_to_optimizer
        assert "## Current Prompt" in sent_to_optimizer
        assert "Hello world." in sent_to_optimizer

    def test_conversation_is_single_continuous_thread(self):
        """The optimizer prompt must contain the ENTIRE Step 1 prompt
        (not just a summary), proving it's one continuous conversation."""
        preds, gts = _make_minimal_data()
        tmpl = _make_single_template()

        loss_comp = PE2LossComputer()
        feedbacks = loss_comp.compute(
            predictions=preds,
            ground_truths=gts,
            tasks=SINGLE_TASK,
            prompt_template=tmpl,
            llm_pool=None,
            loss_batch_size=10,
            verbosity=0,
            loss_functions={"fluency": {"metric": "accuracy"}},
        )

        mock_reasoning = "Mock reasoning output."
        gradient_llm = _MockLLMPool(responses=[[mock_reasoning]])
        pe2_conversation = {}

        grad_comp = PE2GradientComputer()
        grad_comp.compute(
            feedbacks=feedbacks,
            prompt_template=tmpl,
            tasks=SINGLE_TASK,
            llm_pool=gradient_llm,
            gradient_batch_size=1,
            verbosity=0,
            predictions=preds,
            ground_truths=gts,
            pe2_conversation=pe2_conversation,
            pe2_batch_size=1,
        )

        step1_prompt_sent_to_llm = gradient_llm.recorded_prompts[0][0]

        optimizer_llm = _MockLLMPool(
            responses=[[json.dumps({"instructions": {"fluency": "new instruction"}})]]
        )
        opt = PE2Optimizer()
        opt.optimize(
            gradients={FLUENCY: []},
            current_prompt=tmpl,
            tasks=SINGLE_TASK,
            llm_pool=optimizer_llm,
            verbosity=0,
            pe2_conversation=pe2_conversation,
            max_prompt_tokens=50,
        )

        optimizer_prompt = optimizer_llm.recorded_prompts[0][0]
        assert optimizer_prompt.startswith(step1_prompt_sent_to_llm), (
            "The optimizer prompt must start with the exact same Step 1 "
            "prompt that was sent to the gradient LLM, proving one "
            "continuous conversation."
        )
