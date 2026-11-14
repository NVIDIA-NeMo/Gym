# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic-MME task-only scoring; process judgments are deliberately not rewards."""

import asyncio
import json
import math
import re
from pathlib import Path
from typing import Any, ClassVar, Optional

from pydantic import ConfigDict, Field, ValidationError

from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymChatCompletion
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import JudgeError, VerifierFixture, call_judge
from resources_servers.agentic_mme.task_data import GoldenAnswer
from resources_servers.agentic_mme.verifier_fixture import create_server


def judge_tolerance(metadata: dict[str, Any], golden: GoldenAnswer) -> float:
    """The numeric tolerance shown to the judge: the task's answer_tolerance (the author's final answer
    check), else a numeric golden answer's own tolerance."""
    tolerance = metadata.get("answer_tolerance")
    if isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool) and math.isfinite(tolerance):
        return max(float(tolerance), 0.0)
    return golden.tolerance if golden.match_type == "numeric" else 0.0


_REASONING_TAG = re.compile(r"<(/?)(think|thinking)>", re.I)


def strip_closed_reasoning(text: str) -> str:
    """Remove each <think>...</think> (or <thinking>) block in one linear pass.

    Same result as re.sub(r"<(think|thinking)>.*?</\\1>", "", text, flags=re.S | re.I), which rescans the
    rest of the text for every unclosed opening tag.
    """
    # Unicode regex backreferences fold İ to i but keep ı distinct.
    tags = [
        (m.start(), m.end(), m.group(1) == "/", m.group(2).replace("İ", "i").lower())
        for m in _REASONING_TAG.finditer(text)
    ]
    next_close: dict[str, list[int]] = {name: [] for _, _, _, name in tags}
    for index, (_, _, closing, name) in enumerate(tags):
        if closing:
            next_close[name].append(index)
    cursors = {name: 0 for name in next_close}
    kept, position = [], 0
    for index, (start, end, closing, name) in enumerate(tags):
        closes = next_close[name]
        while cursors[name] < len(closes) and closes[cursors[name]] <= index:
            cursors[name] += 1
        if not closing and start >= position and cursors[name] < len(closes):
            kept.append(text[position:start])
            position = tags[closes[cursors[name]]][1]
    kept.append(text[position:])
    return "".join(kept)


def extract_answer(text: str) -> str:
    """Extract the public atomic answer format, excluding private reasoning.

    With several <answer> tags the last one wins: a model that quotes the format before answering
    (e.g. <answer>YOUR_SHORT_ANSWER</answer>) is scored on the answer it gives afterwards.
    """
    text = strip_closed_reasoning(text)
    text = re.split(r"</(?:think|thinking)>", text, flags=re.I)[-1]
    text = re.split(r"<(?:think|thinking)>", text, flags=re.I)[0]
    answers = re.findall(r"<answer>(.*?)(?:</answer>|$)", text, re.S | re.I)
    return (answers[-1] if answers else text).strip()


def matches_answer(answer: str, golden: GoldenAnswer) -> bool:
    """Match the released evaluator's exact/contains/numeric task semantics.

    The paper says normalized exact match, but the released dataset/evaluator
    default to contains. Do not silently substitute Gym's soft numeric grader.
    """
    target = str(golden.value).strip()
    if not answer:
        return False
    if golden.match_type == "exact":
        return answer == target
    if golden.match_type == "numeric":
        try:
            actual, expected = float(answer), float(target)
        except ValueError:
            return False
        return math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= golden.tolerance
    numeric = target.replace(".", "").replace("-", "").replace("+", "").isdigit()
    if numeric and len(target) <= 2:
        return re.search(r"\b" + re.escape(target) + r"\b", answer) is not None
    return target.lower() in answer.lower()


JUDGE_SYSTEM = """You are a strict grader. You decide ONE thing: whether the model's FINAL \
answer matches the reference answer.

Judge only the final answer the model commits to (normally inside <answer>...</answer>), not its \
reasoning. Differences that do NOT matter: units, casing, whitespace, LaTeX wrappers, \
spelled-out vs numeric digits, ordering where the question does not ask for an order \
(e.g. groups of matching items), trailing punctuation, extra prose around the answer.

These make it WRONG: a different number (outside the stated tolerance, when one is given), \
a different option letter, a different coordinate, a different colour, a different count, \
a different set of items, or several conflicting final answers.

You are NOT solving the problem and you cannot see any image. If the model gives no final \
answer, or you cannot tell what it asserts, answer "unsure".

Reply with ONLY a JSON object:
{"verdict": "equivalent" | "different" | "unsure", "confidence": 0.0-1.0, "reason": "<12 words"}
("equivalent" = the final answer is correct.)"""

JUDGE_USER = """Question:
{question}

Reference answer (gold):
{expected}
{tolerance}
What the extractor pulled out of the model's response:
{extracted}

The model's final response text (may be truncated):
<<<
{text}
>>>

Is the model's final answer correct?"""


def parse_verdict(content: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", content or "", re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    verdict = str(data.get("verdict", "unsure")).lower().strip()
    if verdict not in ("equivalent", "different", "unsure"):
        verdict = "unsure"
    return {"verdict": verdict, "reason": str(data.get("reason", ""))[:200]}


def question_text(params: Any) -> str:
    """Text of the last user message; images are not sent to the judge."""
    items = params.input if not isinstance(params.input, str) else [{"role": "user", "content": params.input}]
    for item in reversed(list(items)):
        data = item if isinstance(item, dict) else item.model_dump()
        if data.get("role") != "user":
            continue
        content = data.get("content")
        if isinstance(content, str):
            return content
        return "\n".join(part.get("text", "") for part in content or [] if part.get("type") == "input_text")
    return ""


class AgenticMMEConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    # When set, an LLM judge decides correctness and string match is kept only as a diagnostic.
    judge_model_server: Optional[ModelServerRef] = None
    judge_model: str = ""
    judge_temperature: float = 0.0
    judge_max_tokens: int = 1024
    # Per worker: with num_workers > 1, each worker allows this many judge calls at once.
    judge_max_concurrency: int = Field(default=32, ge=1)
    judge_max_text_chars: int = Field(default=8000, ge=1)


class AgenticMMEVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, Any] = Field(default_factory=dict)


class AgenticMMEVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    extracted_answer: str = ""
    failure_reason: str | None = None
    evaluation_track: str = "task_only"
    string_match_reward: float | None = None
    judge_verdict: str | None = None
    judge_reason: str | None = None


class AgenticMMEServer(SimpleResourcesServer):
    config: AgenticMMEConfig
    _judge_slots: asyncio.Semaphore | None = None

    async def judge(
        self, question: str, expected: str, extracted: str, text: str, tolerance: float = 0.0
    ) -> dict[str, Any]:
        if self._judge_slots is None:
            self._judge_slots = asyncio.Semaphore(self.config.judge_max_concurrency)
        prompt = JUDGE_USER.format(
            question=question,
            expected=expected,
            tolerance=f"Tolerance: a number within ±{tolerance:g} of the reference is correct.\n" if tolerance else "",
            extracted=extracted or "(nothing)",
            text=text[-self.config.judge_max_text_chars :],
        )
        params = {
            "model": self.config.judge_model,
            "messages": [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
            "temperature": self.config.judge_temperature,
            "max_tokens": self.config.judge_max_tokens,
        }
        async with self._judge_slots:
            completion = await call_judge(
                self.server_client,
                server_name=self.config.judge_model_server.name,
                url_path="/v1/chat/completions",
                json=params,
                response_model=NeMoGymChatCompletion,
            )
        if not completion.choices:
            raise JudgeError("judge returned no choices")
        return parse_verdict(completion.choices[0].message.content or "")

    async def verify(self, body: AgenticMMEVerifyRequest) -> AgenticMMEVerifyResponse:
        result = AgenticMMEVerifyResponse(**body.model_dump(), reward=0.0)
        try:
            golden = GoldenAnswer.model_validate(body.verifier_metadata.get("golden_answer"))
        except ValidationError:
            result.failure_reason = "invalid_golden_answer"
            return result
        # Only the terminal turn may answer; never score an earlier tool-turn message.
        terminal = []
        for item in body.response.output:
            if item.type in ("function_call", "function_call_output"):
                terminal = []
            elif item.type == "message" and item.role == "assistant":
                terminal = [part.text for part in item.content if part.type == "output_text"]
        if body.response.incomplete_details:
            # A cut-off episode scores zero in both metrics, keeping the same sample denominator.
            result.string_match_reward = 0.0
            result.failure_reason = "incomplete_response"
            return result
        final_text = "\n".join(terminal)
        result.extracted_answer = extract_answer(final_text)
        result.string_match_reward = float(matches_answer(result.extracted_answer, golden))
        result.reward = result.string_match_reward
        if self.config.judge_model_server is not None and result.extracted_answer:
            # A failed judge call raises JudgeError; the judge_failsafe wrapper around /verify turns it
            # into a masked judge_failed row in the failures sidecar, regraded later, not a wrong answer.
            judged = await self.judge(
                question_text(body.responses_create_params),
                str(golden.value),
                result.extracted_answer,
                final_text,
                tolerance=judge_tolerance(body.verifier_metadata, golden),
            )
            result.judge_verdict = judged["verdict"]
            result.judge_reason = judged["reason"]
            result.reward = float(judged["verdict"] == "equivalent")
        if not result.reward:
            result.failure_reason = "incorrect_answer" if result.extracted_answer else "missing_answer"
        return result


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_server,
    request_model=AgenticMMEVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    AgenticMMEServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = AgenticMMEServer.run_webserver()  # noqa: F401
