# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BigFinanceBench tools and verifier for NeMo Gym.

The five tools come from a commit-pinned, tools-only BigFinanceBench package.
The verifier sends its single judge call through a Gym model server instead of
calling LiteLLM directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, ClassVar, Literal, Optional

from big_finance_harness.tools import (
    EdgarSearchTool,
    FetchUrlTool,
    FinalAnswerTool,
    PythonExecTool,
    WebSearchTool,
)
from big_finance_harness.tools.web_search import _SerpApiBackend, _TavilyBackend
from fastapi import Body, FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from starlette.requests import Request

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


logger = logging.getLogger(__name__)

_MAX_TRACE_CHARS = 150_000
_TOOL_RESULT_CAP = 4_000
_TOOL_ARGS_CAP = 1_500

JUDGE_SYSTEM = """\
You are an impartial grader for a financial-research agent benchmark. You evaluate whether
the agent satisfied each step of an analyst rubric and whether its final answer matches
the reference answer. You are strict but fair: a rubric line is satisfied only if the
trace contains positive evidence for it.

Return only the JSON object specified by the response schema. Do not add commentary.
"""


class BigFinanceResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    serp_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    sec_edgar_user_agent: Optional[str] = None
    web_search_timeout_s: float = 30.0
    edgar_timeout_s: float = 20.0
    fetch_timeout_s: float = 30.0
    python_exec_timeout_s: float = 5.0
    judge_model_server: Optional[ModelServerRef] = None
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    judge_call_timeout_s: Optional[float] = 1800.0
    reward_mode: Literal["final_answer", "rubric_points"] = "final_answer"


class RubricLine(BaseModel):
    text: str
    points: int = Field(default=1, ge=1)


class BigFinanceRunRequest(BaseRunRequest):
    id: str = ""
    query: str = ""
    reference_answer: str = ""
    rubric: list[RubricLine] = Field(default_factory=list)
    evaluation_only: Literal[True] = True
    do_not_train: Literal[True] = True
    benchmark_canary: Optional[str] = None
    sources: list[str] = Field(default_factory=list)


class BigFinanceVerifyRequest(BigFinanceRunRequest, BaseVerifyRequest):
    pass


class RubricVerdict(BaseModel):
    index: int
    text: str
    points: int
    satisfied: bool
    explanation: str = ""


class BigFinanceVerifyResponse(BaseVerifyResponse):
    final_answer: Optional[str] = None
    reference_answer: str = ""
    final_answer_correct: bool = False
    rubric_verdicts: list[RubricVerdict] = Field(default_factory=list)
    rubric_points_earned: int = 0
    rubric_points_possible: int = 0
    rubric_points_fraction: float = 0.0
    rubric_lines_earned: int = 0
    rubric_lines_possible: int = 0
    judge_error: Optional[str] = None
    judge_text: Optional[str] = None


def _message_text(item: Any) -> str:
    chunks: list[str] = []
    if getattr(item, "type", None) != "message":
        return ""
    for part in getattr(item, "content", []) or []:
        text = getattr(part, "text", None)
        if isinstance(text, str):
            chunks.append(text)
    return "".join(chunks)


def _responses_output_text(response: Any) -> str:
    """Extract judge text without validating unrelated response metadata."""
    if not isinstance(response, dict):
        return ""
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    chunks: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return "".join(chunks)


def extract_final_answer(response: NeMoGymResponse) -> Optional[str]:
    """Prefer the last valid final_answer call, then the last assistant prose."""
    prose: Optional[str] = None
    for item in reversed(response.output):
        if getattr(item, "type", None) == "function_call" and getattr(item, "name", None) == "final_answer":
            try:
                args = json.loads(getattr(item, "arguments", "{}"))
            except (TypeError, json.JSONDecodeError):
                args = {}
            answer = args.get("answer") if isinstance(args, dict) else None
            if isinstance(answer, str) and answer.strip():
                return answer
        text = _message_text(item)
        if prose is None and text.strip():
            prose = text
    return prose


def format_trace(response: NeMoGymResponse) -> str:
    """Port the upstream trace rendering and size caps to Responses API output."""
    lines: list[str] = []
    step = -1
    in_model_batch = False
    for item in response.output:
        item_type = getattr(item, "type", None)
        is_assistant_message = item_type == "message" and getattr(item, "role", None) == "assistant"
        is_model_item = item_type in {"reasoning", "function_call"} or is_assistant_message
        if is_model_item and not in_model_batch:
            step += 1
            lines.append(f"=== step {step} ===")
            in_model_batch = True

        if is_assistant_message:
            text = _message_text(item)
            if text:
                lines.append(f"assistant: {text}")
        elif item_type == "function_call":
            args = getattr(item, "arguments", "") or ""
            if len(args) > _TOOL_ARGS_CAP:
                args = args[:_TOOL_ARGS_CAP] + "..."
            lines.append(f"tool_call {getattr(item, 'name', '')}({args})")
        elif item_type == "function_call_output":
            content = str(getattr(item, "output", "") or "")
            if len(content) > _TOOL_RESULT_CAP:
                content = content[:_TOOL_RESULT_CAP] + "..."
            lines.append(f"tool_result: {content}")
            in_model_batch = False
        elif item_type == "message":
            # A harness-injected user nudge separates two model turns.
            in_model_batch = False
    text = "\n".join(lines)
    if len(text) <= _MAX_TRACE_CHARS:
        return text
    half = _MAX_TRACE_CHARS // 2
    return f"{text[:half]}\n\n... [trace truncated for length] ...\n\n{text[-half:]}"


def _judge_prompt(body: BigFinanceVerifyRequest, answer: Optional[str], trace: str) -> str:
    rubric = "\n".join(f"{i}. {line.text}" for i, line in enumerate(body.rubric, 1))
    return f"""\
QUESTION:
{body.query}

REFERENCE ANSWER:
{body.reference_answer}

RUBRIC (one line per analyst step; numbered):
{rubric}

AGENT'S FINAL ANSWER:
{answer or "[no final answer was produced]"}

AGENT'S TRACE (assistant text, tool calls, tool results):
{trace}

For each rubric line, return a boolean indicating whether the trace and final answer
together evidence that the line was satisfied. Also return a boolean indicating whether
the final answer matches the reference answer (numerically equivalent values count as
matching; minor formatting differences are acceptable; sign and units must match).
"""


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["final_answer_correct", "rubric"],
        "properties": {
            "final_answer_correct": {"type": "boolean"},
            "rubric": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["index", "satisfied", "explanation"],
                    "properties": {
                        "index": {"type": "integer"},
                        "satisfied": {"type": "boolean"},
                        "explanation": {"type": "string"},
                    },
                },
            },
        },
    }


def _parse_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("judge response did not contain a JSON object")


class BigFinanceResourcesServer(SimpleResourcesServer):
    config: BigFinanceResourcesServerConfig
    _tools: dict[str, Any]

    def model_post_init(self, context: Any) -> None:
        backend = None
        if self.config.serp_api_key:
            backend = _SerpApiBackend(self.config.serp_api_key, self.config.web_search_timeout_s)
        elif self.config.tavily_api_key:
            backend = _TavilyBackend(self.config.tavily_api_key, self.config.web_search_timeout_s)

        self._tools: dict[str, Any] = {
            "web_search": WebSearchTool(backend=backend, timeout_s=self.config.web_search_timeout_s),
            "edgar_search": (
                EdgarSearchTool(
                    user_agent=self.config.sec_edgar_user_agent,
                    timeout_s=self.config.edgar_timeout_s,
                )
                if self.config.sec_edgar_user_agent
                else None
            ),
            "fetch_url": FetchUrlTool(
                timeout_s=self.config.fetch_timeout_s,
                sec_user_agent=self.config.sec_edgar_user_agent,
            ),
            "python_exec": PythonExecTool(timeout_s=self.config.python_exec_timeout_s),
            "final_answer": FinalAnswerTool(),
        }

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        return await super().seed_session(body)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        for name in self._tools:
            app.post(f"/{name}")(self._handler(name))
        return app

    def _handler(self, name: str):
        async def handler(body: dict = Body(default={})):
            tool = self._tools[name]
            if tool is None:
                return PlainTextResponse(
                    json.dumps({"error": f"Tool '{name}' is not configured."}),
                    media_type="application/json",
                )
            try:
                # The shared agent records the HTTP response body verbatim as the
                # function_call_output. Returning the upstream string directly keeps
                # model-visible observations identical to the standalone harness.
                return PlainTextResponse(await tool.run(body if isinstance(body, dict) else {}))
            except Exception as exc:  # noqa: BLE001 - tool errors are observations, not HTTP 500s
                return PlainTextResponse(
                    json.dumps({"error": f"{type(exc).__name__}: {exc}"}),
                    media_type="application/json",
                )

        return handler

    async def _judge(self, body: BigFinanceVerifyRequest, answer: Optional[str]) -> tuple[dict[str, Any], str]:
        if self.config.judge_model_server is None:
            raise RuntimeError("judge_model_server is not configured")
        params = (
            self.config.judge_responses_create_params or NeMoGymResponseCreateParamsNonStreaming(input=[])
        ).model_copy(deep=True)
        params.input = [
            NeMoGymEasyInputMessage(role="system", content=JUDGE_SYSTEM),
            NeMoGymEasyInputMessage(role="user", content=_judge_prompt(body, answer, format_trace(body.response))),
        ]
        params.text = {
            "format": {
                "type": "json_schema",
                "name": "rubric_grading",
                "strict": True,
                "schema": _response_schema(),
            }
        }
        response = await asyncio.wait_for(
            self.server_client.post(
                server_name=self.config.judge_model_server.name,
                url_path="/v1/responses",
                json=params,
            ),
            timeout=self.config.judge_call_timeout_s,
        )
        await raise_for_status(response)
        text = _responses_output_text(await get_response_json(response))
        return _parse_json_object(text), text

    async def verify(self, request: Request, body: BigFinanceVerifyRequest) -> BigFinanceVerifyResponse:
        answer = extract_final_answer(body.response)
        possible = sum(line.points for line in body.rubric)
        try:
            grade, judge_text = await self._judge(body, answer)
            by_index = {entry.get("index"): entry for entry in grade.get("rubric", []) if isinstance(entry, dict)}
            verdicts: list[RubricVerdict] = []
            for index, line in enumerate(body.rubric, 1):
                entry = by_index.get(index, {})
                verdicts.append(
                    RubricVerdict(
                        index=index,
                        text=line.text,
                        points=line.points,
                        satisfied=entry.get("satisfied") is True,
                        explanation=str(entry.get("explanation", "missing")),
                    )
                )
            earned = sum(v.points for v in verdicts if v.satisfied)
            final_correct = grade.get("final_answer_correct") is True
            fraction = earned / possible if possible else 0.0
            reward = float(final_correct) if self.config.reward_mode == "final_answer" else fraction
            return BigFinanceVerifyResponse(
                **body.model_dump(),
                reward=reward,
                final_answer=answer,
                final_answer_correct=final_correct,
                rubric_verdicts=verdicts,
                rubric_points_earned=earned,
                rubric_points_possible=possible,
                rubric_points_fraction=fraction,
                rubric_lines_earned=sum(v.satisfied for v in verdicts),
                rubric_lines_possible=len(verdicts),
                judge_text=judge_text,
            )
        except Exception as exc:  # noqa: BLE001 - preserve rollout and expose judge failure
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("BigFinance judge failed")
            return BigFinanceVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                final_answer=answer,
                rubric_points_possible=possible,
                rubric_lines_possible=len(body.rubric),
                judge_error=error,
                failure_reason=error,
            )


if __name__ == "__main__":
    BigFinanceResourcesServer.run_webserver()
