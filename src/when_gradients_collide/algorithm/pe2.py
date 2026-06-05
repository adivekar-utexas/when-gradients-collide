"""PE2 algorithm, optimizer, gradient computer, and loss computer.

PE2 (Ye et al., ICLR 2024) uses a single "proposal model" in a multi-turn
conversation to both analyze errors (Step 1: structured five-question reasoning)
and generate an improved prompt (Step 2: refinement).

In WGC's 4-step pipeline, this is approximated by:
1. PE2GradientComputer builds the full Step 1 prompt, calls the gradient LLM,
   and returns the conversation state (step1_prompt + step1_reasoning) via
   ``pe2_conversation``.
2. PE2Optimizer concatenates Step 1 conversation + the model's reasoning +
   Step 2 instructions into a single meta-prompt for the optimizer LLM.
"""

import json
from typing import Any, ClassVar, Dict, List, Optional, Union

from morphic import validate
from morphic.typed import format_exception_msg
from pydantic import Field, PrivateAttr

from ..config import wgc_config
from ..data_structures import (
    Batch,
    DatasetSample,
    NumericFeedback,
    PredictionResult,
    Task,
    TextGradient,
    TextualFeedback,
)
from ..gradient_computer import GradientComputer
from ..llm_utils import apply_prompt_suffix
from ..loss_computer import LossComputer, TaskLevelLossComputer
from ..prompt_algorithm import PromptAlgorithm
from ..prompt_optimizer import PromptOptimizer
from ..prompt_template import PromptTemplate
from ..prompt_trajectory import PromptTrajectory
from ..types import strip_smart_quotes


class PE2(PromptAlgorithm):
    """PE2 algorithm implementation (Ye et al., ICLR 2024).

    PE2 uses a single "proposal model" in a multi-turn conversation to
    both analyze errors (Step 1: structured five-question reasoning) and
    generate an improved prompt (Step 2: refinement).

    The gradient LLM and optimizer LLM should be the same model instance
    to approximate the paper's single-conversation architecture.

    Attributes:
        pe2_batch_size: Number of examples shown in the PE2 Step 1 prompt
            (the paper's "batch size" hyperparameter, default 2).
        max_prompt_tokens: Maximum word count for each task instruction
            (the paper's prompt length constraint, default 50).
        step_size: Maximum word edits allowed per instruction.
            ``None`` means no constraint (the paper's default).
    """

    aliases = ["pe2"]

    loss_computer: Dict[str, Any] = Field(default_factory=lambda: {"name": "pe2"})
    gradient_computer: Dict[str, Any] = Field(default_factory=lambda: {"name": "pe2"})
    prompt_optimizer: Dict[str, Any] = Field(default_factory=lambda: {"name": "pe2"})

    task_losses: Dict[str, str] = Field(default_factory=dict)

    pe2_batch_size: int = 2
    max_prompt_tokens: int = 50
    step_size: Optional[int] = None

    # Paper: task LLM uses low temperature for deterministic scoring;
    # the proposal model (gradient + optimizer) uses higher temperature.
    task_llm_temperature: Optional[float] = Field(default=0.1, ge=0.0, le=2.0)
    optimizer_llm_temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    gradient_llm_temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)

    _pe2_conversation: Optional[Dict[str, str]] = PrivateAttr(default=None)
    _previous_instructions: Optional[Dict[str, str]] = PrivateAttr(default=None)
    _last_predictions: Optional[List[PredictionResult]] = PrivateAttr(default=None)
    _last_ground_truths: Optional[List[DatasetSample]] = PrivateAttr(default=None)

    @classmethod
    def pre_initialize(cls, data: dict) -> None:
        """Default loss_batch_size and gradient_batch_size to batch_size.

        PE2 evaluates the entire training batch as one unit for loss
        computation. The gradient step uses its own ``pe2_batch_size``
        to select how many examples the proposal model sees.
        """
        batch_size = data.get("batch_size")
        if batch_size is not None:
            if data.get("loss_batch_size") is None:
                data["loss_batch_size"] = batch_size
            if data.get("gradient_batch_size") is None:
                data["gradient_batch_size"] = batch_size

    def _get_algorithm_context(
        self, step: int, batch: Optional[Batch] = None
    ) -> Dict[str, Any]:
        """Build the algorithm context dict passed to all pipeline components.

        Validates that all tasks have loss configurations, builds the
        PE2 conversation state, and assembles all PE2-specific parameters
        as explicit keys (not hidden in **kwargs).

        Args:
            step: Current optimization step number.
            batch: Current training batch (passed to optimizer for step-size scheduling).

        Returns:
            Dict with keys: loss_functions, pe2_conversation, pe2_batch_size,
            max_prompt_tokens, step_size, predictions, ground_truths, batch.

        Raises:
            ValueError: If any task is missing from ``task_losses``.
        """
        missing_tasks = [
            t.task_name for t in self.tasks if t.task_name not in self.task_losses
        ]
        if len(missing_tasks) > 0:
            raise ValueError(
                f"Tasks missing from task_losses: {missing_tasks}. "
                f"All tasks must have a loss metric configured."
            )

        loss_functions = {
            task.task_name: self._build_loss_fn_config(
                task_name=task.task_name, use_textual=False
            )
            for task in self.tasks
        }

        if self._pe2_conversation is None:
            self._pe2_conversation = {}

        context: Dict[str, Any] = {
            "algorithm": "pe2",
            "step": step,
            "loss_functions": loss_functions,
            "pe2_conversation": self._pe2_conversation,
            "pe2_batch_size": self.pe2_batch_size,
            "max_prompt_tokens": self.max_prompt_tokens,
            "step_size": self.step_size,
        }

        if self._last_predictions is not None:
            context["predictions"] = self._last_predictions
        if self._last_ground_truths is not None:
            context["ground_truths"] = self._last_ground_truths
        if batch is not None:
            context["batch"] = batch

        return context

    # -- Hooks --

    def _after_predict(
        self,
        *,
        step: int,
        predictions: List,
        batch: Batch,
    ) -> None:
        """Store predictions and ground truths for the gradient/optimizer steps."""
        self._last_predictions = predictions
        self._last_ground_truths = batch.samples
        self._pe2_conversation = {}

    def _build_loss_context(self, *, step: int) -> Dict[str, Any]:
        """Build context for the loss computer (excludes predictions/ground_truths)."""
        context = self._get_algorithm_context(step)
        context.pop("predictions", None)
        context.pop("ground_truths", None)
        return context

    def _build_gradient_context(self, *, step: int) -> Dict[str, Any]:
        """Build context for the gradient computer (includes predictions/ground_truths)."""
        return self._get_algorithm_context(step)

    def _build_optimizer_context(self, *, step: int, batch: Batch) -> Dict[str, Any]:
        """Build context for the optimizer (includes batch for step-size scheduling)."""
        return self._get_algorithm_context(step=step, batch=batch)

    def _build_run_config(
        self, *, initial_prompt: PromptTemplate, start_step: int
    ) -> Dict[str, Any]:
        """Extend run config with PE2-specific hyperparameters."""
        config = super()._build_run_config(
            initial_prompt=initial_prompt,
            start_step=start_step,
        )
        config["pe2_batch_size"] = self.pe2_batch_size
        config["max_prompt_tokens"] = self.max_prompt_tokens
        config["step_size"] = self.step_size
        return config

    def _update_state(
        self,
        step: int,
        feedbacks: Dict,
        gradients: Dict,
        current_prompt: PromptTemplate,
        new_prompt: PromptTemplate,
        all_candidates: Optional[List[PromptTemplate]] = None,
        all_candidate_scores: Optional[List[Dict[Task, List[NumericFeedback]]]] = None,
    ) -> None:
        """Update PE2 state after a step: save new instructions, clear per-step state."""
        self._previous_instructions = new_prompt.instruction
        self._last_predictions = None
        self._last_ground_truths = None
        self._pe2_conversation = None

    def _serialize_algorithm_context(self, *, step: int) -> Dict[str, Any]:
        """Serialize algorithm context for observability logging."""
        return {
            "algorithm": "pe2",
            "step": step,
            "pe2_batch_size": self.pe2_batch_size,
            "max_prompt_tokens": self.max_prompt_tokens,
            "step_size": self.step_size,
        }

    def _serialize_algorithm_state(self) -> Dict[str, Any]:
        """Serialize algorithm state for observability logging."""
        return {"previous_instructions": self._previous_instructions}


class PE2Optimizer(PromptOptimizer):
    """PE2-specific Prompt Optimizer: continues the multi-turn conversation
    from the PE2GradientComputer's Step 1 reasoning into Step 2 (prompt
    refinement).

    PE2's key architectural property is that the "gradient" (structured
    reasoning) and "update" (new prompt) happen in a single multi-turn
    conversation with the same model.  In WGC's modular pipeline
    we approximate this by concatenating the full Step 1 conversation
    (including the model's own reasoning output) with the Step 2
    instructions into a single prompt.
    """

    aliases: ClassVar[List[str]] = ["pe2"]

    @validate
    def create_meta_prompt(
        self,
        *,
        gradients: Dict[Task, List[TextGradient]],
        current_prompt: PromptTemplate,
        tasks: List[Task],
        trajectory: Optional[PromptTrajectory] = None,
        batch: Optional[Batch] = None,
        task_demonstrations: Optional[List[DatasetSample]] = None,
        input_col_labels: Optional[Dict[str, str]] = None,
        predictions: Optional[List[PredictionResult]] = None,
        ground_truths: Optional[List[DatasetSample]] = None,
        optimizer_task_strategy: Optional[str] = None,
        pe2_conversation: Optional[Dict[str, str]] = None,
        max_prompt_tokens: int,
        step_size: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Build the PE2 Step 2 meta-prompt by continuing the Step 1 conversation.

        Concatenates the full Step 1 conversation prefix (system + task
        description + acknowledgment + data + five-question reasoning) with
        the model's Step 1 response, then appends the Step 2 refinement
        instructions.

        Args:
            gradients: Per-task gradients (unused by PE2; the reasoning is in
                ``pe2_conversation``).
            current_prompt: Current prompt template.
            tasks: List of tasks to generate instructions for.
            pe2_conversation: Dict with ``step1_prompt`` (the full Step 1 prompt
                sent to the gradient LLM) and ``step1_reasoning`` (the gradient
                LLM's response).  Required.
            max_prompt_tokens: Maximum word count for each task instruction.
                Defaults to 50.
            step_size: Maximum word edits allowed per instruction.
                ``None`` means no constraint.
            **kwargs: Absorbed for interface compatibility.

        Returns:
            The complete meta-prompt string.

        Raises:
            ValueError: If ``pe2_conversation`` is None or missing required keys.
        """
        step1_prompt: str = (
            pe2_conversation.get("step1_prompt", "")
            if pe2_conversation is not None
            else ""
        )
        step1_reasoning: str = (
            pe2_conversation.get("step1_reasoning", "")
            if pe2_conversation is not None
            else ""
        )

        current_prompt_text: str = current_prompt.render_instructions()

        step_size_line: str = ""
        if step_size is not None:
            step_size_line = f"* You are allowed to change up to {step_size} words in the original prompt.\n"

        example_instruction: str = ",\n".join(
            [
                f'        "{t.task_name}": "improved instruction for {t.task_name}"'
                for t in tasks
            ]
        )

        meta_prompt: str = f"""{step1_prompt}

[Assistant]
{step1_reasoning}

[User]
Now please carefully review your reasoning in Step 1 and help with Step 2: refining the prompt.

## Current Prompt
{current_prompt_text}

## Instructions
{step_size_line}* The total length of each task instruction should be less than {max_prompt_tokens} words.
* Please help edit the per-task instructions so that the updated prompt will not fail on these examples anymore.
* Return ONLY a valid JSON object mapping each task name to its improved instruction.

The JSON structure should be:
{{
  "instructions": {{
{example_instruction}
  }}
}}

Do not include any text outside the JSON."""

        return meta_prompt

    @validate
    def parse_meta_prompt_response(
        self,
        *,
        response: str,
        tasks: List[Task],
        **kwargs: Any,
    ) -> Dict[str, str]:
        """Parse the optimizer LLM response into a task -> instruction dict.

        Handles JSON with ``"instructions"`` wrapper, flat dicts, code fences,
        and escaped double-braces.

        Args:
            response: Raw LLM response text.
            tasks: List of tasks (for fallback to original instructions).

        Returns:
            Dict mapping each task name to its instruction string.

        Raises:
            ValueError: If no valid JSON found or parsing fails.
        """
        cleaned: str = response.strip()

        start: int = cleaned.find("{")
        end: int = cleaned.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError(
                f"[PE2Optimizer] No JSON object found in LLM output:\n{response}"
            )

        json_str: str = cleaned[start:end].replace("\n", " ").strip()

        try:
            response_json: Dict[str, Any] = json.loads(json_str)
        except json.JSONDecodeError:
            json_str_fallback: str = strip_smart_quotes(
                json_str.replace("{{", "{").replace("}}", "}")
            )
            try:
                response_json = json.loads(json_str_fallback)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                raise ValueError(
                    f"[PE2Optimizer] Failed to parse JSON: {format_exception_msg(e)}.\n"
                    f"Response:\n{response}\nExtracted string:\n{json_str}"
                )

        if "instructions" in response_json and isinstance(
            response_json["instructions"], dict
        ):
            candidate: Dict[str, str] = response_json["instructions"]
        elif isinstance(response_json, dict) and all(
            isinstance(v, str) for v in response_json.values()
        ):
            candidate: Dict[str, str] = response_json
        else:
            raise ValueError(
                f"[PE2Optimizer] Invalid response format: expected dict of "
                f'string values or {{"instructions": {{...}}}}, '
                f"got: {type(response_json).__name__} with keys {list(response_json.keys()) if isinstance(response_json, dict) else 'N/A'}"
            )

        return {
            task.task_name: candidate[task.task_name]
            if task.task_name in candidate
            else task.task_instruction
            for task in tasks
        }


class PE2GradientComputer(GradientComputer):
    """PE2 Gradient Computer: builds the multi-turn Step 1 conversation from
    the PE2 meta-prompt and generates structured per-example reasoning.

    PE2 does NOT use separate loss feedback.  Instead, it shows the proposal
    model the raw predictions and ground truths alongside the five structured
    reasoning questions.  The "gradient" is the model's own structured analysis.

    The conversation state (``step1_prompt`` and ``step1_reasoning``) is written
    into the ``pe2_conversation`` dict so the PE2Optimizer can continue the
    conversation for Step 2.
    """

    aliases = ["pe2"]

    # @validate omitted: pe2_conversation is a mutable dict mutated in-place
    # by this method (writes step1_prompt and step1_reasoning). Pydantic's
    # validate_call does not preserve object identity for dict arguments,
    # breaking the mutable pass-through contract.
    def compute(
        self,
        feedbacks: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]],
        prompt_template: PromptTemplate,
        tasks: List[Task],
        llm_pool: Any,  # LLMPool protocol; see types.py
        gradient_batch_size: int,
        verbosity: int = 1,
        *,
        use_textual_feedback: bool = False,
        predictions: Optional[List[PredictionResult]] = None,
        ground_truths: Optional[List[DatasetSample]] = None,
        input_col_labels: Optional[Dict[str, str]] = None,
        gradient_task_strategy: Optional[str] = None,
        pe2_conversation: Optional[Dict[str, str]] = None,
        pe2_batch_size: int,
        **kwargs: Any,
    ) -> Dict[Task, List[TextGradient]]:
        """Build the PE2 Step 1 prompt, call the gradient LLM, and store
        the conversation state for the optimizer.

        Args:
            feedbacks: Per-task feedback from the loss computer (unused by PE2;
                PE2 shows raw predictions/ground_truths instead).
            prompt_template: Current prompt template.
            tasks: List of tasks.
            llm_pool: Gradient LLM pool.
            gradient_batch_size: Batch size (unused by PE2; uses ``pe2_batch_size``).
            verbosity: Logging verbosity level.
            predictions: Forward-pass predictions from the task LLM.  Required.
            ground_truths: Ground truth samples from the training batch.  Required.
            input_col_labels: Mapping of column names to display labels.
                Used for human-readable input formatting in the examples section.
            pe2_conversation: Mutable dict where this method writes
                ``step1_prompt`` and ``step1_reasoning``.  Required.
            pe2_batch_size: Number of examples to show in the Step 1 prompt.
                Defaults to 2.

        Returns:
            Dict mapping each Task to a single-element list containing the
            combined gradient (the model's structured reasoning).

        Raises:
            ValueError: If ``predictions``, ``ground_truths``, or
                ``pe2_conversation`` is None.
        """
        if predictions is None:
            raise ValueError(
                "PE2GradientComputer requires 'predictions' to build "
                "Step 1 examples with model outputs."
            )
        if ground_truths is None:
            raise ValueError(
                "PE2GradientComputer requires 'ground_truths' to build "
                "Step 1 examples with ground-truth labels."
            )
        if pe2_conversation is None:
            raise ValueError(
                "PE2GradientComputer requires 'pe2_conversation' dict to store "
                "Step 1 conversation state for the optimizer."
            )

        resolved_col_labels: Dict[str, str] = (
            input_col_labels
            if input_col_labels is not None
            else prompt_template.input_col_labels
        )

        step1_prompt: str = self._build_pe2_step1_prompt(
            predictions=predictions,
            ground_truths=ground_truths,
            tasks=tasks,
            prompt_template=prompt_template,
            full_template="{{prompt}}\n{{input}}",
            batch_size_shown=pe2_batch_size,
            input_col_labels=resolved_col_labels,
        )

        prompts: List[str] = apply_prompt_suffix([step1_prompt], llm_pool)
        responses: List[str] = llm_pool.call_llm_batch(
            prompts=prompts, verbosity=verbosity
        ).result(timeout=wgc_config.defaults.batch_invocation_timeout)

        reasoning_text: str = responses[0] if len(responses) > 0 else ""

        pe2_conversation["step1_prompt"] = step1_prompt
        pe2_conversation["step1_reasoning"] = reasoning_text

        combined_gradient: TextGradient = TextGradient(
            task_name="all_tasks",
            gradient_text=reasoning_text,
            based_on_feedbacks=[],
            gradient_prompt=step1_prompt,
        )

        result: Dict[Task, List[TextGradient]] = {}
        for task in tasks:
            result[task] = [combined_gradient]
        return result

    def _build_pe2_step1_prompt(
        self,
        *,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        tasks: List[Task],
        prompt_template: PromptTemplate,
        full_template: str,
        batch_size_shown: int,
        input_col_labels: Dict[str, str],
    ) -> str:
        """Build the PE2 Step 1 prompt: the full multi-turn conversation
        serialized as a single string (system + task description +
        acknowledgment + data with five-question template).

        The prompt structure follows the PE2 paper's Appendix (lines 871-929):
        Turn 0: System message
        Turn 1: Two-step task description (upfront)
        Turn 2: Scripted acknowledgment
        Turn 3: Data turn (prompt, full template, examples, five-question instructions)

        Args:
            predictions: Forward-pass predictions from the task LLM.
            ground_truths: Ground truth samples.
            tasks: List of tasks.
            prompt_template: Current prompt template.
            full_template: Template showing how prompt + input are joined.
            batch_size_shown: Number of examples to include.
            input_col_labels: Mapping of column names to display labels.

        Returns:
            The complete Step 1 prompt string.
        """
        current_prompt_text: str = prompt_template.render_instructions()

        examples_text: str = self._format_examples(
            predictions=predictions,
            ground_truths=ground_truths,
            tasks=tasks,
            max_examples=batch_size_shown,
            input_col_labels=input_col_labels,
        )

        num_examples: int = min(batch_size_shown, len(predictions))

        prompt: str = f"""[System] You are a helpful assistant.

[User]
A prompt is a text paragraph that outlines the expected actions and instructs the model to generate a specific output. This prompt is concatenated with the input text, and the model then creates the required output.

In our collaboration, we'll work together to refine a prompt. The process consists of two main steps:

## Step 1
I will provide you with the current prompt, how the prompt is concatenated with the input text (i.e., "full template"), along with {num_examples} example(s) that are associated with this prompt. Each example contains the input, the final answer produced by the model, and the ground-truth label to the input. Your task is to analyze the examples, determining whether the existing prompt is describing the task reflected by these examples precisely, and suggest changes to the prompt.

## Step 2
Next, you will carefully review your reasoning in step 1, integrate the insights to craft a new, optimized prompt. Some extra instructions will be provided too.

[Assistant]
Sure, I'd be happy to help you with this prompt engineering problem. Please provide me with the current prompt and the examples you have.

[User]
## Prompt
{current_prompt_text}

## Full Template
This describes how the prompt of interest is concatenated with the input text. The prompt may appear before the input text, or after the input text. Optionally the full template may contain other template information.
```
{full_template}
```

## Examples
{examples_text}

## Instructions
For some of these examples, the output does not match with the label. This may be due to the prompt being misleading or not describing the task precisely.

Please examine the example(s) carefully. Note that the ground-truth labels are __absolutely correct__, but the prompts (task descriptions) may be incorrect and need modification. For each example, provide reasoning according to the following template:

### Example <id>
Input: <input summary>
Output: <model output>
Label: <ground-truth label>
Is the output correct compared to the label: <yes or no, and your reasoning>
Is the output correctly following the given prompt: <yes or no, and your reasoning>
Is the prompt correctly describing the task shown by the input-label pair: <yes or no, and your reasoning>
To output the correct label, is it necessary to edit the prompt: <yes or no, and your reasoning>
If yes, provide detailed analysis and actionable suggestions to edit the prompt: <analysis and suggestions>"""

        return prompt

    @staticmethod
    def _format_examples(
        *,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        tasks: List[Task],
        max_examples: int,
        input_col_labels: Dict[str, str],
    ) -> str:
        """Format prediction/ground-truth pairs for the PE2 examples section.

        Prioritizes failure examples (hard negative sampling) following
        the default PE2 configuration (paper lines 386-388).

        Args:
            predictions: Forward-pass predictions.
            ground_truths: Ground truth samples.
            tasks: List of tasks.
            max_examples: Maximum number of examples to include.
            input_col_labels: Mapping of raw column names to display labels.

        Returns:
            Formatted examples string with Input/Output/Label per example.
        """
        failures: List[tuple] = []
        successes: List[tuple] = []

        for pred, gt in zip(predictions, ground_truths):
            is_failure: bool = False
            for task in tasks:
                pred_val = pred.task_outputs.get(task.task_name)
                gt_val = gt.ground_truths.get(task.task_name)
                if pred_val is not None and gt_val is not None and pred_val != gt_val:
                    is_failure = True
                    break
            if is_failure:
                failures.append((pred, gt))
            else:
                successes.append((pred, gt))

        selected: List[tuple] = failures[:max_examples]
        if len(selected) < max_examples:
            selected.extend(successes[: max_examples - len(selected)])

        lines: List[str] = []
        for idx, (pred, gt) in enumerate(selected, 1):
            lines.append(f"### Example {idx}")

            input_parts: List[str] = []
            for col, val in gt.inputs.items():
                if col not in input_col_labels:
                    raise ValueError(
                        f"PE2GradientComputer: column {col!r} has no label in "
                        f"input_col_labels. Available: {list(input_col_labels.keys())}."
                    )
                label: str = input_col_labels[col]
                input_parts.append(f"{label}: {val}")
            lines.append("Input: " + "; ".join(input_parts))

            output_parts: List[str] = []
            for task in tasks:
                if task.task_name not in pred.task_outputs:
                    raise ValueError(
                        f"PE2GradientComputer: task {task.task_name!r} not found in "
                        f"pred.task_outputs. Available: {list(pred.task_outputs.keys())}."
                    )
                val = pred.task_outputs[task.task_name]
                output_parts.append(f"{task.task_name}={val}")
            lines.append("Output: " + ", ".join(output_parts))

            label_parts: List[str] = []
            for task in tasks:
                if task.task_name not in gt.ground_truths:
                    raise ValueError(
                        f"PE2GradientComputer: task {task.task_name!r} not found in "
                        f"gt.ground_truths. Available: {list(gt.ground_truths.keys())}."
                    )
                val = gt.ground_truths[task.task_name]
                label_parts.append(f"{task.task_name}={val}")
            lines.append("Label: " + ", ".join(label_parts))
            lines.append("")

        return "\n".join(lines)


class PE2LossComputer(TaskLevelLossComputer):
    """PE2-specific Loss Computer: numeric losses only, no textual feedback.

    PE2 does not use a separate loss LLM.  Instead, the proposal model
    sees raw predictions and ground truths directly in the gradient step.
    This loss computer computes numeric metric purely for
    trajectory tracking and observability; the actual "feedback" consumed
    by PE2's gradient step is the raw prediction data, not these numbers.

    Inherits from ``TaskLevelLossComputer`` to reuse its batching and
    metric computation infrastructure.
    """

    aliases = ["pe2"]

    @validate
    def compute(
        self,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        tasks: List[Task],
        prompt_template: PromptTemplate,
        llm_pool: Optional[Any],  # LLMPool protocol; see types.py
        loss_batch_size: int,
        verbosity: int = 1,
        *,
        loss_functions: Optional[Dict[str, Dict[str, Any]]] = None,
        input_col_labels: Optional[Dict[str, str]] = None,
        loss_task_strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[Task, List[Union[NumericFeedback, TextualFeedback]]]:
        """Compute only numeric losses for PE2 (no LLM calls).

        PE2 uses the proposal model to inspect raw predictions directly.
        No loss LLM is used.

        Args:
            predictions: Prediction results from the task LLM.
            ground_truths: Ground truth samples from the training batch.
            tasks: Tasks to compute losses for.
            prompt_template: Current prompt template (unused by PE2).
            llm_pool: Should be None (PE2 has no loss LLM).
            loss_batch_size: Batch size for splitting predictions.
            verbosity: Logging verbosity level.
            loss_functions: Per-task loss config dicts.  Required.

        Returns:
            Dict mapping each Task to a list of NumericFeedback objects.

        Raises:
            ValueError: If loss_functions is None.
        """
        if loss_functions is None:
            raise ValueError(
                f"{self.__class__.__name__}.compute() requires 'loss_functions'"
            )
        result: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]] = {}

        for task in tasks:
            if task.task_name not in loss_functions:
                continue
            loss_fn_config: Dict[str, Any] = loss_functions[task.task_name]
            task_batches = LossComputer.batch_predictions(
                predictions=predictions,
                ground_truths=ground_truths,
                task=task,
                batch_size=loss_batch_size,
            )
            feedbacks: List[Union[NumericFeedback, TextualFeedback]] = []
            for pred_batch, gt_batch in task_batches:
                numeric = self._compute_numeric_loss(
                    predictions=pred_batch,
                    ground_truths=gt_batch,
                    task=task,
                    loss_fn_config=loss_fn_config,
                )
                if numeric is not None:
                    feedbacks.append(numeric)
            result[task] = feedbacks
        return result
