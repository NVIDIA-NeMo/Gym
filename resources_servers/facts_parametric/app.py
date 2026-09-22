# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FACTS Parametric resources server.

Closed-book factoid QA from the FACTS Benchmark Suite (Cheng et al., 2025, arXiv:2512.10791, section 4). The
official protocol grades every policy answer three times with Gemini 2.5 Pro and averages the grades. This server
reproduces the published reference implementation (Kaggle starter notebook
``yulongt/facts-parametric-benchmark-starter-code``, version 10):

- the grader prompt is the starter's ``GRADER_TEMPLATE`` byte for byte (``prompts/grader_template.txt``);
- three independent grader calls are issued per response with ``seed`` 0, 1, 2 and provider-default sampling;
- each grader reply is mapped to a label with the starter's ``extract_classification`` substring rule
  (``INCORRECT`` > ``MISTAKE`` > ``CORRECT`` > ``NOT_ATTEMPTED`` > ``UNKNOWN``).

``reward`` is the fraction of the three grades that are ``correct`` (the paper averages the sampled grades). The
starter's stricter all-three-correct score is kept as ``all_correct``. The four grade labels, a diagnostic
final-line parse, and every grader receipt are returned so aggregate metrics can separate model outcome from
grader validity.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import call_judge
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.facts_parametric.verifier_fixture import create_fixture_server


GRADER_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "grader_template.txt"
GRADER_TEMPLATE_SHA256 = "9d6a61f9ce3b875b5f97d25305abe6c073911f97e98297780483dd84fd9b548c"  # pragma: allowlist secret
GRADER_VARIANT = "kaggle-starter-yulongt-facts-parametric-benchmark-starter-code-v10"
LABELS = ("correct", "incorrect", "not_attempted", "unknown")
LabelParser = Literal["starter", "output_line"]

_OUTPUT_LINE = re.compile(r"Output:\s*\[?\s*(CORRECT|MISTAKE|INCORRECT|UNKNOWN|NOT_ATTEMPTED)\s*\]?", re.IGNORECASE)
_OUTPUT_LINE_LABELS = {
    "CORRECT": "correct",
    "MISTAKE": "incorrect",
    "INCORRECT": "incorrect",
    "UNKNOWN": "unknown",
    "NOT_ATTEMPTED": "not_attempted",
}


def load_grader_template() -> str:
    template = GRADER_TEMPLATE_PATH.read_text(encoding="utf-8")
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if digest != GRADER_TEMPLATE_SHA256:
        raise RuntimeError(f"grader template drifted from the pinned starter prompt: sha256 {digest}")
    return template


def extract_text_from_response(response: NeMoGymResponse) -> str:
    """Return the last assistant message text; reasoning items are split off by the model server."""
    for output in reversed(response.output):
        if getattr(output, "type", None) != "message" or getattr(output, "role", None) != "assistant":
            continue
        content = getattr(output, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts = [item.text for item in content if isinstance(getattr(item, "text", None), str)]
            if texts:
                return "\n".join(texts).strip()
    return ""


def starter_classification(judgment: str) -> str:
    """The starter notebook's ``extract_classification``, verbatim order of substring checks."""
    judgment = judgment.strip()
    if "INCORRECT" in judgment:
        return "incorrect"
    if "MISTAKE" in judgment:
        return "incorrect"
    if "CORRECT" in judgment:
        return "correct"
    if "NOT_ATTEMPTED" in judgment:
        return "not_attempted"
    return "unknown"


def output_line_classification(judgment: str) -> str:
    """Diagnostic parse of the grader's mandated closing line ``Output: [LABEL]`` (last occurrence wins)."""
    matches = _OUTPUT_LINE.findall(judgment)
    if not matches:
        return "unparsed"
    return _OUTPUT_LINE_LABELS[matches[-1].upper()]


def build_grader_prompt(template: str, *, question: str, expected_answer: str, prediction: str) -> str:
    return template.format(question=question, gold_answer=expected_answer, prediction=prediction)


class FACTSParametricConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[]),
        description=(
            "Grader request overrides (temperature, top_p, max_output_tokens). Unset fields are omitted so the "
            "grader provider uses its defaults, as the official starter does."
        ),
    )
    judge_samples: int = Field(default=3, ge=1, description="Independent grades per response (official: 3).")
    judge_seeds: Optional[List[int]] = Field(
        default=None, description="Seed per grade; defaults to 0..judge_samples-1 like the starter's seed=i."
    )
    label_parser: LabelParser = Field(
        default="starter",
        description="Which parse feeds reward: the starter's substring rule (official) or the closing Output line.",
    )
    grader_template_path: str = Field(default=str(GRADER_TEMPLATE_PATH))


class FACTSParametricRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    id: Optional[Union[int, str]] = None
    question: Optional[str] = None
    expected_answer: Optional[str] = None
    topic: Optional[str] = None
    source_url: Optional[str] = None


class FACTSParametricVerifyRequest(FACTSParametricRunRequest, BaseVerifyRequest):
    pass


class FACTSParametricVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    generation: str = ""
    generation_empty: bool = False
    generation_truncated: bool = False
    judge_labels: List[str] = Field(default_factory=list, description="Canonical label per grade (label_parser).")
    judge_labels_starter: List[str] = Field(default_factory=list)
    judge_labels_output_line: List[str] = Field(default_factory=list)
    judge_parse_agreement: float = 1.0
    judge_empty_samples: int = 0
    judge_receipts: List[Dict[str, Any]] = Field(default_factory=list)
    judge_prompt_sha256: str = ""
    grader_variant: str = GRADER_VARIANT
    is_correct: float = 0.0
    is_incorrect: float = 0.0
    is_not_attempted: float = 0.0
    is_unknown: float = 0.0
    all_correct: float = 0.0


class FACTSParametricResourcesServer(SimpleResourcesServer):
    config: FACTSParametricConfig

    def model_post_init(self, context: Any) -> None:
        path = Path(self.config.grader_template_path)
        self._grader_template = (
            load_grader_template() if path == GRADER_TEMPLATE_PATH else path.read_text(encoding="utf-8")
        )
        seeds = self.config.judge_seeds
        if seeds is None:
            seeds = list(range(self.config.judge_samples))
        if len(seeds) != self.config.judge_samples:
            raise ValueError("judge_seeds must have exactly judge_samples entries")
        self._judge_seeds = list(seeds)
        return super().model_post_init(context)

    def _judge_params(self, prompt: str, seed: int) -> NeMoGymChatCompletionCreateParamsNonStreaming:
        base = self.config.judge_responses_create_params
        params: Dict[str, Any] = {"messages": [{"role": "user", "content": prompt}], "seed": seed}
        if base.temperature is not None:
            params["temperature"] = base.temperature
        if base.top_p is not None:
            params["top_p"] = base.top_p
        if base.max_output_tokens is not None:
            params["max_tokens"] = base.max_output_tokens
        return NeMoGymChatCompletionCreateParamsNonStreaming(**params)

    async def _grade_once(self, prompt: str, sample: int, seed: int) -> Dict[str, Any]:
        completion = await call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/chat/completions",
            json=self._judge_params(prompt, seed),
            response_model=NeMoGymChatCompletion,
        )
        choice = completion.choices[0] if completion.choices else None
        content = (choice.message.content if choice and choice.message else None) or ""
        return {
            "sample": sample,
            "seed": seed,
            "judge_model": completion.model,
            "response_id": completion.id,
            "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
            "usage": completion.usage.model_dump(mode="json") if completion.usage else None,
            "content": content,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "label_starter": starter_classification(content),
            "label_output_line": output_line_classification(content),
        }

    async def verify(self, body: FACTSParametricVerifyRequest) -> FACTSParametricVerifyResponse:
        generation = extract_text_from_response(body.response)
        incomplete = body.response.incomplete_details
        truncated = bool(incomplete and getattr(incomplete, "reason", None) == "max_output_tokens")
        prompt = build_grader_prompt(
            self._grader_template,
            question=body.question or "",
            expected_answer=body.expected_answer or "",
            prediction=generation,
        )
        receipts = list(
            await asyncio.gather(
                *(self._grade_once(prompt, sample, seed) for sample, seed in enumerate(self._judge_seeds))
            )
        )
        labels_starter = [receipt["label_starter"] for receipt in receipts]
        labels_output_line = [receipt["label_output_line"] for receipt in receipts]
        if self.config.label_parser == "starter":
            labels = labels_starter
        else:
            labels = [label if label in LABELS else "unknown" for label in labels_output_line]
        count = len(labels)
        fractions = {label: sum(1 for item in labels if item == label) / count for label in LABELS}
        agreement = sum(1 for a, b in zip(labels_starter, labels_output_line) if a == b) / count
        return FACTSParametricVerifyResponse(
            **body.model_dump(),
            reward=fractions["correct"],
            generation=generation,
            generation_empty=not generation,
            generation_truncated=truncated,
            judge_labels=labels,
            judge_labels_starter=labels_starter,
            judge_labels_output_line=labels_output_line,
            judge_parse_agreement=agreement,
            judge_empty_samples=sum(1 for receipt in receipts if not receipt["content"]),
            judge_receipts=receipts,
            judge_prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            is_correct=fractions["correct"],
            is_incorrect=fractions["incorrect"],
            is_not_attempted=fractions["not_attempted"],
            is_unknown=fractions["unknown"],
            all_correct=1.0 if count and all(label == "correct" for label in labels) else 0.0,
        )

    @staticmethod
    def _bootstrap_ci(values: List[float], *, resamples: int = 2000, seed: int = 20251211) -> tuple[float, float]:
        if len(values) < 2:
            return (float("nan"), float("nan"))
        rng = random.Random(seed)
        count = len(values)
        means = sorted(sum(rng.choice(values) for _ in range(count)) / count for _ in range(resamples))
        return (means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1])

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        rollouts = [rollout for task in tasks for rollout in task]
        metrics: Dict[str, Any] = {}
        if not rollouts:
            return metrics

        def _pooled(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
            grades = [label for row in rows for label in row.get("judge_labels", [])]
            total = len(grades)
            counts = {label: sum(1 for grade in grades if grade == label) for label in LABELS}
            accuracy = counts["correct"] / total if total else 0.0
            hedging = counts["not_attempted"] / total if total else 0.0
            attempted = total - counts["not_attempted"]
            attempted_accuracy = counts["correct"] / attempted if attempted else 0.0
            f1 = (
                2 * accuracy * attempted_accuracy / (accuracy + attempted_accuracy)
                if accuracy + attempted_accuracy
                else 0.0
            )
            return {
                "accuracy": accuracy,
                "hedging_rate": hedging,
                "mistake_rate": counts["incorrect"] / total if total else 0.0,
                "unknown_rate": counts["unknown"] / total if total else 0.0,
                "attempted_accuracy": attempted_accuracy,
                "f1": f1,
                "strict_all_correct_rate": sum(row.get("all_correct", 0.0) for row in rows) / len(rows),
                "num_rollouts": len(rows),
                "num_grades": total,
                "num_correct_grades": counts["correct"],
                "num_incorrect_grades": counts["incorrect"],
                "num_not_attempted_grades": counts["not_attempted"],
                "num_unknown_grades": counts["unknown"],
            }

        metrics.update(_pooled(rollouts))
        rewards = [float(row.get("reward", 0.0)) for row in rollouts]
        low, high = self._bootstrap_ci(rewards)
        metrics["accuracy_ci95_low"] = low
        metrics["accuracy_ci95_high"] = high
        total_grades = metrics["num_grades"]
        empty_samples = sum(int(row.get("judge_empty_samples", 0)) for row in rollouts)
        metrics["judge_empty_grade_rate"] = empty_samples / total_grades if total_grades else 0.0
        metrics["judge_valid_rate"] = 1.0 - metrics["judge_empty_grade_rate"]
        metrics["judge_parse_agreement_rate"] = sum(
            float(row.get("judge_parse_agreement", 1.0)) for row in rollouts
        ) / len(rollouts)
        metrics["generation_empty_rate"] = sum(1 for row in rollouts if row.get("generation_empty")) / len(rollouts)
        metrics["generation_truncated_rate"] = sum(1 for row in rollouts if row.get("generation_truncated")) / len(
            rollouts
        )
        by_topic: Dict[str, List[Dict[str, Any]]] = {}
        for row in rollouts:
            by_topic.setdefault(str(row.get("topic") or "unknown"), []).append(row)
        for topic, rows in sorted(by_topic.items()):
            pooled = _pooled(rows)
            metrics[f"accuracy/topic/{topic}"] = pooled["accuracy"]
            metrics[f"hedging_rate/topic/{topic}"] = pooled["hedging_rate"]
            metrics[f"num_rollouts/topic/{topic}"] = pooled["num_rollouts"]
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        keys = (
            "accuracy",
            "accuracy_ci95_low",
            "accuracy_ci95_high",
            "hedging_rate",
            "attempted_accuracy",
            "f1",
            "strict_all_correct_rate",
            "judge_valid_rate",
            "generation_truncated_rate",
            "mean/input_tokens",
            "mean/output_tokens",
        )
        return {key: agent_metrics[key] for key in keys if key in agent_metrics}


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_fixture_server,
    request_model=FACTSParametricVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    FACTSParametricResourcesServer.run_webserver()
