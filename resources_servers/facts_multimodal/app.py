# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FACTS Multimodal resources server."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, ClassVar

import yaml
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
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics


_DEFAULT_COVERAGE_PROMPT_PATH = str(Path(__file__).parent / "prompts" / "coverage.yaml")
_DEFAULT_FACTUALITY_PROMPT_PATH = str(Path(__file__).parent / "prompts" / "factuality.yaml")
_THINKING_BLOCK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", flags=re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*?\}", flags=re.DOTALL)


def extract_text_from_response(response: NeMoGymResponse) -> str:
    """Return the final assistant message, excluding hidden reasoning blocks."""
    for output in reversed(response.output):
        if getattr(output, "type", None) != "message" or getattr(output, "role", None) != "assistant":
            continue
        content = getattr(output, "content", None)
        if isinstance(content, str):
            return _THINKING_BLOCK_RE.sub("", content).strip()
        if isinstance(content, list):
            texts = [item.text for item in content if isinstance(getattr(item, "text", None), str)]
            if texts:
                return _THINKING_BLOCK_RE.sub("", "\n".join(texts)).strip()
    return ""


def _extract_coverage_score(judge_text: str, essential_fact_count: int) -> float | None:
    if essential_fact_count == 0:
        return 1.0
    for match in _JSON_OBJECT_RE.finditer(judge_text):
        try:
            verdict = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(verdict, dict):
            continue
        fact_verdicts = [
            value.lower()
            for key, value in verdict.items()
            if isinstance(key, str) and re.fullmatch(r"Fact \d+", key, flags=re.IGNORECASE) and isinstance(value, str)
        ]
        if len(fact_verdicts) != essential_fact_count or any(value not in {"yes", "no"} for value in fact_verdicts):
            continue
        return fact_verdicts.count("yes") / essential_fact_count
    return None


def parse_coverage_score(judge_text: str, essential_fact_count: int) -> float:
    """Compute coverage from per-fact verdicts; malformed or omitted facts fail closed."""
    score = _extract_coverage_score(judge_text, essential_fact_count)
    return score if score is not None else 0.0


def _extract_factuality_verdict(judge_text: str) -> bool | None:
    match = re.search(
        r"FINAL VERDICT:\s*(NO|HAS) CLEAR CONTRADICTION(?:\(S\))?\.?\s*$",
        judge_text.strip(),
        flags=re.IGNORECASE,
    )
    return None if match is None else match.group(1).upper() == "NO"


def parse_factuality_verdict(judge_text: str) -> bool:
    """Return whether a notebook-style factuality verdict finds no contradiction."""
    return bool(_extract_factuality_verdict(judge_text))


def format_rubric_items(rubric_items: list[dict[str, Any]], *, essential_only: bool) -> str:
    """Format structured release rubrics in the notebook's tagged fact-list form."""
    formatted = []
    for index, item in enumerate(rubric_items, start=1):
        tags = [str(tag) for tag in item.get("tags", [])]
        importance = next((tag for tag in tags if tag in {"essential", "non-essential", "optional"}), "")
        if essential_only and importance != "essential":
            continue
        fact_type = next((tag for tag in tags if tag not in {"essential", "non-essential", "optional"}), "Sentence")
        formatted.append(
            f"Fact # <ID> {index} </ID> <FACT> {item.get('fact', '')} </FACT> "
            f"<TYPE> {fact_type} </TYPE> <IMPORTANCE> {importance} </IMPORTANCE>"
        )
    return "\n".join(formatted)


class FACTSMultimodalConfig(BaseResourcesServerConfig):
    """Configuration for the multimodal FACTS rubric judge."""

    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[]),
        description="Optional judge generation controls; unset values use the provider defaults.",
    )
    coverage_prompt_path: str = _DEFAULT_COVERAGE_PROMPT_PATH
    factuality_prompt_path: str = _DEFAULT_FACTUALITY_PROMPT_PATH
    use_base64_images: bool = Field(
        default=True,
        description="Use prepared data URLs instead of source image URLs for policy and factuality judge requests.",
    )


class FACTSMultimodalRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    id: str = ""
    prompt: str = ""
    image_url: str = ""
    image_path: str = ""
    image_mime_type: str = ""
    image_data_url: str = ""
    rubric_items: list[dict[str, Any]] = Field(default_factory=list)


class FACTSMultimodalVerifyRequest(FACTSMultimodalRunRequest, BaseVerifyRequest):
    pass


class FACTSMultimodalVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    generation: str = ""
    coverage_judge_output: str = ""
    factuality_judge_output: str = ""
    coverage: float = 0.0
    factuality: float = 0.0
    accuracy: float = 0.0
    response_status: str = ""
    model_incomplete: bool = False
    empty_generation: bool = False
    coverage_parse_failed: bool = False
    factuality_parse_failed: bool = False


class FACTSMultimodalServer(SimpleResourcesServer):
    """Score image-question answers against FACTS human-authored rubrics."""

    config: FACTSMultimodalConfig

    def model_post_init(self, context: object) -> None:
        coverage_prompt_data = yaml.safe_load(Path(self.config.coverage_prompt_path).read_text(encoding="utf-8"))
        factuality_prompt_data = yaml.safe_load(Path(self.config.factuality_prompt_path).read_text(encoding="utf-8"))
        self._coverage_prompt_template: str = coverage_prompt_data["user"]
        self._factuality_prompt_template: str = factuality_prompt_data["user"]
        super().model_post_init(context)

    async def _call_judge(self, judge_prompt: str, image_url: str | None = None) -> str:
        content: list[dict[str, str]] = [{"type": "input_text", "text": judge_prompt}]
        if image_url is not None:
            if not (image_url.startswith("data:image/") or image_url.startswith(("https://", "http://"))):
                raise JudgeError("FACTS Multimodal task is missing its configured image URL")
            content.append({"type": "input_image", "image_url": image_url, "detail": "auto"})
        params = self.config.judge_responses_create_params.model_copy(deep=True)
        params.input = [
            NeMoGymEasyInputMessage(
                role="user",
                content=content,
            )
        ]
        judge_response = await call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=params,
            response_model=NeMoGymResponse,
        )
        text = extract_text_from_response(judge_response)
        if not text:
            raise JudgeError("empty FACTS Multimodal judge response")
        return text

    async def _judge_coverage(self, prompt: str, fact_count: int) -> tuple[str, float, bool]:
        """Retry malformed coverage verdicts without changing valid all-No outcomes."""
        judge_text = ""
        for _ in range(3):
            judge_text = await self._call_judge(prompt)
            if (score := _extract_coverage_score(judge_text, fact_count)) is not None:
                return judge_text, score, False
        return judge_text, 0.0, True

    async def _judge_factuality(self, prompt: str, image_url: str) -> tuple[str, bool, bool]:
        """Retry until the image-aware judge emits one of the two required verdicts."""
        judge_text = ""
        for _ in range(3):
            judge_text = await self._call_judge(prompt, image_url)
            if (verdict := _extract_factuality_verdict(judge_text)) is not None:
                return judge_text, verdict, False
        return judge_text, False, True

    @staticmethod
    def _score_fn(result: dict[str, float]) -> dict[str, float]:
        return {
            "accuracy": result.get("accuracy", 0.0),
            "coverage": result.get("coverage", 0.0),
            "factuality": result.get("factuality", 0.0),
        }

    def compute_metrics(self, tasks: list[list[dict]]) -> dict:
        metrics, _, _, _ = compute_pass_majority_metrics(tasks, score_fn=self._score_fn)
        rows = [rollout for task_rollouts in tasks for rollout in task_rollouts]
        if rows:
            metrics.update(
                {
                    "response_incomplete_rate": sum(bool(row.get("model_incomplete")) for row in rows) / len(rows),
                    "empty_generation_rate": sum(bool(row.get("empty_generation")) for row in rows) / len(rows),
                    "coverage_parse_failure_rate": sum(bool(row.get("coverage_parse_failed")) for row in rows)
                    / len(rows),
                    "factuality_parse_failure_rate": sum(bool(row.get("factuality_parse_failed")) for row in rows)
                    / len(rows),
                }
            )
        return metrics

    def get_key_metrics(self, agent_metrics: dict) -> dict:
        return highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]")

    async def verify(self, body: FACTSMultimodalVerifyRequest) -> FACTSMultimodalVerifyResponse:
        generation = extract_text_from_response(body.response)
        raw_status = body.response.status or ""
        response_status = str(getattr(raw_status, "value", raw_status))
        model_incomplete = response_status == "incomplete"
        if not generation:
            return FACTSMultimodalVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                generation="",
                response_status=response_status,
                model_incomplete=model_incomplete,
                empty_generation=True,
            )
        essential_rubrics = format_rubric_items(body.rubric_items, essential_only=True)
        factuality_rubrics = format_rubric_items(body.rubric_items, essential_only=False)
        coverage_prompt = self._coverage_prompt_template.format(
            prompt=body.prompt,
            rubric_items=essential_rubrics,
            generation=generation,
        )
        factuality_prompt = self._factuality_prompt_template.format(
            prompt=body.prompt,
            rubric_items=factuality_rubrics,
            generation=generation,
        )
        factuality_image_url = body.image_data_url if self.config.use_base64_images else body.image_url
        essential_fact_count = sum(
            "essential" in [str(tag) for tag in item.get("tags", [])] for item in body.rubric_items
        )
        coverage_result, factuality_result = await asyncio.gather(
            self._judge_coverage(coverage_prompt, essential_fact_count),
            self._judge_factuality(factuality_prompt, factuality_image_url),
        )
        coverage_judge_output, coverage, coverage_parse_failed = coverage_result
        factuality_judge_output, factuality_verdict, factuality_parse_failed = factuality_result
        factuality = float(factuality_verdict)
        accuracy = float(coverage > 0.5 and factuality)
        return FACTSMultimodalVerifyResponse(
            **body.model_dump(),
            reward=accuracy,
            generation=generation,
            coverage_judge_output=coverage_judge_output,
            factuality_judge_output=factuality_judge_output,
            coverage=coverage,
            factuality=factuality,
            accuracy=accuracy,
            response_status=response_status,
            model_incomplete=model_incomplete,
            empty_generation=False,
            coverage_parse_failed=coverage_parse_failed,
            factuality_parse_failed=factuality_parse_failed,
        )


if __name__ == "__main__":
    FACTSMultimodalServer.run_webserver()
