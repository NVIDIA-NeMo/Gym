# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness of a multi-turn visual-coding answer, decided by a GPT judge over the final answer.

The grading of Python_call_gen's opencode harness (`format_judge --mode correctness`):
- the response text is every assistant text part joined, and the judge sees its last
  `judge_max_text_chars` characters;
- `extracted` is string_match's final_answer extraction (last "answer:", else last \\boxed{});
- correct iff the verdict is "equivalent"; "different", "unsure" and judge errors score 0.
string_match's grade of `extracted` is kept as a diagnostic, not the reward.
"""

import asyncio
import json
import re
from typing import Any, ClassVar, Optional

from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import NeMoGymChatCompletion
from resources_servers.string_match.app import (
    _extract_answer,
    _extract_last_assistant_text,
    _grade_string_match,
)


JUDGE_SYSTEM = """You are a strict grader. You decide ONE thing: whether the model's FINAL answer matches the reference answer.

Judge only the final answer the model commits to (normally the last \\boxed{...}), not its reasoning. Differences that do NOT matter: units, casing, whitespace, LaTeX wrappers, spelled-out vs numeric digits, ordering where the question does not ask for an order (e.g. groups of matching items), trailing punctuation, extra prose around the answer.

These make it WRONG: a different number, a different option letter, a different coordinate, a different colour, a different count, a different set of items, or several conflicting final answers.

You are NOT solving the problem and you cannot see any image. If the model gives no final answer, or you cannot tell what it asserts, answer "unsure".

Reply with ONLY a JSON object:
{"verdict": "equivalent" | "different" | "unsure", "confidence": 0.0-1.0, "reason": "<12 words"}
("equivalent" = the final answer is correct.)"""

JUDGE_USER = """Question:
{question}

Reference answer (gold):
{expected}

What the extractor pulled out of the model's response:
{extracted}

The model's final response text (may be truncated):
<<<
{text}
>>>

Is the model's answer the same as the reference, differing only in formatting?"""


def parse_verdict(content: str) -> dict[str, Any]:
    """The first JSON object in the reply; anything without a valid verdict is "unsure"."""
    match = re.search(r"\{.*\}", content or "", re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    verdict = str(data.get("verdict", "unsure")).lower().strip() if isinstance(data, dict) else "unsure"
    if verdict not in ("equivalent", "different", "unsure"):
        verdict = "unsure"
    reason = str(data.get("reason", "")) if isinstance(data, dict) else ""
    return {"verdict": verdict, "reason": reason[:200]}


class VisualCodingOpenCodeConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    judge_model_server: ModelServerRef
    judge_model: str
    judge_temperature: float = 0.0
    judge_top_p: float = 1.0
    judge_max_tokens: int = 1024
    judge_max_concurrency: int = Field(default=32, ge=1)
    judge_max_question_chars: int = Field(default=2000, ge=1)
    judge_max_text_chars: int = Field(default=4000, ge=1)


class VisualCodingOpenCodeVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    expected_answer: str
    question: str = ""


class VisualCodingOpenCodeVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    extracted_answer: Optional[str] = None
    string_match_reward: float = 0.0
    judge_verdict: Optional[str] = None
    judge_reason: Optional[str] = None
    failure_reason: Optional[str] = None


class VisualCodingOpenCodeServer(SimpleResourcesServer):
    config: VisualCodingOpenCodeConfig
    _judge_slots: Optional[asyncio.Semaphore] = None

    async def judge(self, question: str, expected: str, extracted: Optional[str], text: str) -> dict[str, Any]:
        if self._judge_slots is None:
            self._judge_slots = asyncio.Semaphore(self.config.judge_max_concurrency)
        prompt = JUDGE_USER.format(
            question=question[: self.config.judge_max_question_chars],
            expected=expected,
            extracted=extracted if extracted is not None else "<nothing extracted>",
            text=text[-self.config.judge_max_text_chars :],
        )
        params = {
            "model": self.config.judge_model,
            "messages": [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
            "temperature": self.config.judge_temperature,
            "top_p": self.config.judge_top_p,
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
        return parse_verdict(completion.choices[0].message.content or "")

    async def verify(self, body: VisualCodingOpenCodeVerifyRequest) -> VisualCodingOpenCodeVerifyResponse:
        result = VisualCodingOpenCodeVerifyResponse(**body.model_dump(), reward=0.0)
        text = _extract_last_assistant_text(body)
        if not text:
            # A timed-out or failed run leaves no answer: wrong, with no judge call.
            result.failure_reason = "no_response"
            return result
        result.extracted_answer = _extract_answer(text, "final_answer")
        if result.extracted_answer is not None:
            result.string_match_reward = _grade_string_match(body.expected_answer, result.extracted_answer)
        try:
            judged = await self.judge(body.question, body.expected_answer, result.extracted_answer, text)
        except JudgeError as exc:
            result.failure_reason = "judge_error"
            result.judge_reason = str(exc)[:500]
            return result
        result.judge_verdict = judged["verdict"]
        result.judge_reason = judged["reason"]
        result.reward = float(judged["verdict"] == "equivalent")
        if not result.reward:
            result.failure_reason = "incorrect_answer"
        return result


if __name__ == "__main__":
    VisualCodingOpenCodeServer.run_webserver()
