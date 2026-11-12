# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UI generation resources server: an LLM judge grades the code blocks of a generated UI answer.

The judge reads the task as the policy saw it (the user prompt, or the whole role-labelled conversation when
there is more than one message or request-level instructions), the task's reference answer (a guideline,
never shown to the policy) and the policy's final answer, and returns one JSON verdict that scores ``prompt_adherence`` and ``quality``
from 0 to 4. Only the code inside fenced blocks is judged; nothing is executed or rendered.

A verdict counts only when the judge finished (``finish_reason == "stop"``) and its JSON matches the
rubric's schema exactly. Anything else is a judge failure: the row is masked and keeps its answer for
re-judging, and is never scored as 0 for the policy.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, ClassVar, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming, NeMoGymResponse
from nemo_gym.server_runtime import (
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    ReverifyMode,
    SimpleResourcesServer,
    observe_llm_judge_grading,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import JudgeError, VerifierFixture, call_judge, record_parsed_judge_verdict
from resources_servers.ui_generation.verifier_fixture import create_fixture_server, invoke_with_fixture_judge


SCHEMA_VERSION = "ui-code-judge-v1"
VERDICT_KEYS = frozenset(
    {
        "schema_version",
        "prompt_adherence",
        "quality",
        "critical_issues",
        "reference_differences",
        "uncertainty",
        "execution_verified",
    }
)
MAX_SCORE = 4
# A fenced block in a judge reply: an optional (any-case) json tag, the body, then the closing fence on its own line.
_FENCE_RE = re.compile(r"```[ \t]*(?:json)?[ \t]*\n(.*?)\n[ \t]*```", re.DOTALL | re.IGNORECASE)
_LEADING_THINK_RE = re.compile(r"^\s*<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL)
_DEFAULT_JUDGE_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge.yaml"

RewardScore = Literal["quality", "prompt_adherence", "mean", "min"]


class UiGenerationResourcesServerConfig(BaseResourcesServerConfig):
    # verify() reads only the request: the prompt, the saved policy answer and verifier_metadata.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    judge_model_server: ModelServerRef
    judge_chat_completions_create_params: NeMoGymChatCompletionCreateParamsNonStreaming = Field(
        description="Sampling settings of each judge request; its messages are replaced by the rubric and the case."
    )
    judge_prompt_path: str = Field(
        default=str(_DEFAULT_JUDGE_PROMPT_PATH),
        description="YAML file whose 'system' key holds the judge rubric, sent as the system message.",
    )
    judge_max_concurrency: int = Field(default=64, ge=1, description="Judge requests in flight per server.")
    judge_max_attempts: int = Field(
        default=4,
        ge=1,
        description="Attempts per judge request for a 408/429/5xx reply, a timeout or a lost connection.",
    )
    reward_score: RewardScore = Field(
        default="quality",
        description=(
            "Which verdict score becomes the reward, divided by 4: quality (default), prompt_adherence, the mean "
            "of both, or the min of both."
        ),
    )


class UiGenerationVerifierMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    reference_answer: str = Field(description="The task's reference answer; the judge's guideline.")
    category: Optional[str] = Field(
        default=None, description="Optional task label (for example the target stack); metrics report it."
    )


class UiGenerationVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: UiGenerationVerifierMetadata


class UiGenerationVerifyResponse(BaseMultiRewardVerifyResponse):
    model_config = ConfigDict(extra="allow")

    prompt_adherence_score: int
    quality_score: int
    judge_verdict: Optional[dict[str, Any]] = None
    judge_reasoning: Optional[str] = None


def validate_verdict(obj: object) -> Optional[str]:
    """Return why ``obj`` is not a rubric verdict, or None when it matches the schema exactly."""
    if not isinstance(obj, dict) or set(obj) != VERDICT_KEYS:
        found = sorted(obj) if isinstance(obj, dict) else type(obj).__name__
        return f"keys {found} != {sorted(VERDICT_KEYS)}"
    if obj["schema_version"] != SCHEMA_VERSION:
        return f"schema_version {obj['schema_version']!r}"
    for key in ("prompt_adherence", "quality"):
        part = obj[key]
        if not isinstance(part, dict) or set(part) != {"score", "rationale"}:
            return f"{key} must have exactly score and rationale"
        if type(part["score"]) is not int or not 0 <= part["score"] <= MAX_SCORE:
            return f"{key}.score {part['score']!r} is not an integer in 0..{MAX_SCORE}"
        if not isinstance(part["rationale"], str):
            return f"{key}.rationale is not a string"
    for key in ("critical_issues", "uncertainty"):
        if not isinstance(obj[key], list) or not all(isinstance(x, str) for x in obj[key]):
            return f"{key} must be a list of strings"
    diffs = obj["reference_differences"]
    if not isinstance(diffs, list) or not all(
        isinstance(d, dict)
        and set(d) == {"difference", "material_to_prompt"}
        and isinstance(d["difference"], str)
        and isinstance(d["material_to_prompt"], bool)
        for d in diffs
    ):
        return "reference_differences must be a list of {difference: str, material_to_prompt: bool}"
    if obj["execution_verified"] is not False:
        return "execution_verified must be false"
    return None


def parse_judge_completion(completion: NeMoGymChatCompletion) -> tuple[dict[str, Any], Optional[str]]:
    """Return (verdict, judge reasoning) from a judge reply; raise ``JudgeError`` when it holds no valid verdict."""
    if not completion.choices:
        raise JudgeError("malformed_response: the judge reply has no choices")
    choice = completion.choices[0]
    if choice.finish_reason != "stop":
        raise JudgeError(f"finish_{choice.finish_reason}: the judge did not finish its verdict")
    content = choice.message.content or ""
    reasoning = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None)
    # A model server with a reasoning parser returns the judge's reasoning as a leading <think> block.
    think = _LEADING_THINK_RE.match(content)
    if think:
        reasoning = reasoning or think.group(1).strip()
        content = content[think.end() :]
    verdict = _verdict_json(content.strip())
    if error := validate_verdict(verdict):
        raise JudgeError(f"invalid_schema: {error}")
    return verdict, reasoning


def _verdict_json(text: str) -> Any:
    """The reply's JSON: the whole text, else the first fenced block that parses (prose around it is ignored)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        for fenced in _FENCE_RE.finditer(text):
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                continue
        raise JudgeError(f"invalid_json: {error}") from error


def _text_of(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text") or "" for part in content if isinstance(part, dict))
    return ""


def judge_prompt(body: BaseVerifyRequest) -> str | list[dict[str, str]]:
    """The task as the policy saw it, for the judge's ``prompt`` key.

    A task that is one user message (or a plain string input) and no request-level ``instructions`` is that
    text. Anything else is the whole conversation in order, as ``{"role", "content"}`` messages, so the judge
    sees every constraint and every piece of earlier code the policy saw: ``instructions`` first as a
    ``system`` message (where the model server puts them), then each message with its role. The reference
    answer is never part of it.
    """
    params = body.responses_create_params
    params_input = params.input
    items = [{"role": "user", "content": params_input}] if isinstance(params_input, str) else params_input
    messages = [
        {"role": m["role"], "content": _text_of(m.get("content"))}
        for m in (m.model_dump() if isinstance(m, BaseModel) else m for m in items)
        if isinstance(m, dict) and isinstance(m.get("role"), str)
    ]
    if params.instructions:
        messages.insert(0, {"role": "system", "content": params.instructions})
    if len(messages) == 1 and messages[0]["role"] == "user":
        return messages[0]["content"]
    return messages


def final_answer_text(response: NeMoGymResponse) -> str:
    """The policy's final answer: its last assistant message, without leading reasoning blocks.

    Only complete ``<think>`` blocks at the very start are removed; think-tag text anywhere else (for example
    a string literal in the submitted code) is part of the answer and reaches the judge unchanged.
    """
    for item in reversed(response.output):
        if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
            text = "".join(getattr(part, "text", None) or "" for part in item.content)
            while think := _LEADING_THINK_RE.match(text):
                text = text[think.end() :]
            return text.strip()
    return ""


def response_is_truncated(response: NeMoGymResponse) -> bool:
    return response.status == "incomplete" or response.incomplete_details is not None


class UiGenerationResourcesServer(SimpleResourcesServer):
    config: UiGenerationResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        prompt = yaml.safe_load(Path(self.config.judge_prompt_path).read_text())
        self._judge_rubric: str = prompt["system"]
        self._judge_semaphore = asyncio.Semaphore(self.config.judge_max_concurrency)
        return super().model_post_init(context)

    def judge_request(
        self, prompt: str | list[dict[str, str]], reference: str, candidate: str
    ) -> NeMoGymChatCompletionCreateParamsNonStreaming:
        """The rubric as system message and the case as one JSON user message, with the keys the rubric names."""
        case = {"prompt": prompt, "original_guideline": reference, "candidate_answer": candidate}
        params = self.config.judge_chat_completions_create_params.model_copy(deep=True)
        params.messages = [
            {"role": "system", "content": self._judge_rubric},
            {"role": "user", "content": json.dumps(case, ensure_ascii=False)},
        ]
        return params

    def reward_from_scores(self, prompt_adherence: int, quality: int) -> float:
        score = {
            "quality": quality,
            "prompt_adherence": prompt_adherence,
            "mean": (prompt_adherence + quality) / 2,
            "min": min(prompt_adherence, quality),
        }[self.config.reward_score]
        return score / MAX_SCORE

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Mean reward and scores per task ``category``, for rows that carry one."""
        values: dict[tuple[str, str], list[float]] = defaultdict(list)
        for rollout in (rollout for task in tasks for rollout in task):
            category = (rollout.get("verifier_metadata") or {}).get("category")
            if category is None:
                continue
            for key in ("reward", "prompt_adherence_score", "quality_score"):
                if isinstance(rollout.get(key), (int, float)):
                    values[category, key].append(rollout[key])
        return {f"{category}/mean/{key}": fmean(xs) for (category, key), xs in sorted(values.items())}

    def _scored(
        self, body: UiGenerationVerifyRequest, prompt_adherence: int, quality: int, **fields: Any
    ) -> UiGenerationVerifyResponse:
        return UiGenerationVerifyResponse(
            **body.model_dump(),
            reward=self.reward_from_scores(prompt_adherence, quality),
            reward_components={"prompt_adherence": prompt_adherence / MAX_SCORE, "quality": quality / MAX_SCORE},
            prompt_adherence_score=prompt_adherence,
            quality_score=quality,
            **fields,
        )

    async def verify(self, body: UiGenerationVerifyRequest) -> UiGenerationVerifyResponse:
        # A cut-off or empty answer has no complete code to judge; it is the policy's outcome and scores 0
        # without a judge call.
        if response_is_truncated(body.response):
            return self._scored(body, 0, 0, failure_reason="policy_response_truncated")
        candidate = final_answer_text(body.response)
        if not candidate:
            return self._scored(body, 0, 0, failure_reason="empty_final_answer")

        request = self.judge_request(judge_prompt(body), body.verifier_metadata.reference_answer, candidate)

        async def grade() -> UiGenerationVerifyResponse:
            async with self._judge_semaphore:
                completion = await call_judge(
                    self.server_client,
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/chat/completions",
                    json=request,
                    response_model=NeMoGymChatCompletion,
                    max_attempts=self.config.judge_max_attempts,
                )
            verdict, reasoning = parse_judge_completion(completion)
            record_parsed_judge_verdict()
            return self._scored(
                body,
                verdict["prompt_adherence"]["score"],
                verdict["quality"]["score"],
                judge_verdict=verdict,
                judge_reasoning=reasoning,
            )

        return await observe_llm_judge_grading(grade)


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_fixture_server,
    request_model=UiGenerationVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
    invoke=invoke_with_fixture_judge,
)


if __name__ == "__main__":
    UiGenerationResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = UiGenerationResourcesServer.run_webserver()  # noqa: F401
