# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model server for typed decision engines (``/v1/systemone`` or ``/v1/decisions``).

A decision engine does not generate text: it takes a ``state`` and typed ``questions`` and returns one answer per
question with a probability for every option. This server lets Gym agents call one through ``/v1/responses``:

- the last user message must be the JSON request ``{"state": ..., "questions": {...}}``; other messages (an LLM
  answer-format instruction, for example) are not sent to the engine;
- the reply is one assistant message whose text is ``{"di_engine": {"status", "response", "error", "wall_ms"}}``,
  with ``status`` one of ``ok``, ``unsupported`` (the engine declared a capacity limit) or ``error`` (the call failed
  after its retries). Resources servers read the probabilities from ``response``.

``/v1/systemone`` is called with ``{"model", "state", "questions", **extra_body}``, the request the Decision Index
kit's ``http`` engine sends. ``/v1/decisions`` (the OpenAI Decisions request shape) is also supported: the state goes
in as one ``input_json`` part, ``choice`` maps to ``choice`` and ``noul`` to ``predicate``.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

import aiohttp
from fastapi import HTTPException
from pydantic import Field

from nemo_gym.client.http_client import request
from nemo_gym.core import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.models import BaseResponsesAPIModelConfig, Body, SimpleResponsesAPIModel


ENVELOPE_KEY = "di_engine"
# Messages in a 400/413/422 reply that mean "this request is beyond the engine's declared capacity". Such a request
# is recorded as unsupported (it counts as unanswered), as the Decision Index kit's http engine does.
CAPACITY_MARKERS = (
    "options per choice",
    "a choice needs at least two options",
    "a score takes 2 to 10 levels",
    "the canvas holds",
    "maximum context length",
    "maximum model length",
    "longer than the maximum model length",
    "context window",
    "too many tokens",
)


class DecisionEngineModelConfig(BaseResponsesAPIModelConfig):
    base_url: str = Field(description="Engine root URL, without the endpoint path.")
    api_key: Optional[str] = Field(default=None, description="Bearer token; set it from the environment in YAML.")
    model: str = Field(default="default", description="Model id sent in the request body.")
    endpoint: Literal["systemone", "decisions"] = "systemone"
    extra_body: Dict[str, Any] = Field(default_factory=dict, description="Extra request fields, e.g. packing.")
    max_concurrent_requests: int = Field(default=32, ge=1)
    request_timeout_seconds: float = Field(default=600.0, gt=0)
    max_attempts: int = Field(default=4, ge=1, description="Attempts per request on transport errors and 5xx/429.")
    backoff_seconds: float = Field(default=2.0, ge=0)


def decision_request(body: NeMoGymResponseCreateParamsNonStreaming) -> Dict[str, Any]:
    """The ``{state, questions}`` request carried by the last user message."""
    messages = body.input if isinstance(body.input, list) else [{"role": "user", "content": body.input}]
    for message in reversed(messages):
        item = message if isinstance(message, dict) else message.model_dump()
        if item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        try:
            payload = json.loads(content)
        except (TypeError, ValueError):
            break
        if isinstance(payload, dict) and isinstance(payload.get("questions"), dict) and "state" in payload:
            return {"state": payload["state"], "questions": payload["questions"]}
        break
    raise HTTPException(
        status_code=400,
        detail='decision_engine_model needs the last user message to be the JSON request {"state", "questions"}',
    )


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def to_decisions_request(model: str, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
    """A /v1/systemone request in the OpenAI Decisions shape."""
    items = []
    for name, q in questions.items():
        if q["type"] == "choice":
            items.append(
                {
                    "type": "choice",
                    "name": name,
                    "instructions": _text(q.get("instructions", "")),
                    "choices": [{"value": k, "description": _text(v)} for k, v in q["criteria"].items()],
                }
            )
        elif q["type"] == "noul":
            items.append({"type": "predicate", "name": name, "instructions": _text(q.get("instructions", ""))})
        else:
            raise ValueError(f"question type {q['type']!r} has no /v1/decisions mapping")
    source = (
        state if isinstance(state, str) else [{"role": "user", "content": [{"type": "input_json", "data": state}]}]
    )
    return {"model": model, "input": source, "questions": items}


def from_decisions_response(raw: Dict[str, Any]) -> Dict[str, Any]:
    """An OpenAI Decisions response as /v1/systemone answers."""
    answers = {}
    for a in raw.get("answers", []):
        if a["type"] == "predicate":
            answers[a["name"]] = {"type": "noul", "noul": a["probability"]}
        elif a["type"] == "choice":
            probabilities = {p["value"]: p["probability"] for p in a["probabilities"]}
            answers[a["name"]] = {"type": "choice", "choice": a["choice"], "probabilities": probabilities}
    return {"model": raw.get("model"), "answers": answers, "usage": raw.get("usage")}


def usage_of(input_tokens: Optional[int], envelope: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Responses-API usage for one engine read.

    Input tokens are what the engine reports (``usage.input_tokens``); without that report the usage is left unknown.
    A decision engine reads one answer position per question and generates no text, so output tokens are the number
    of answers returned (0 when the read failed).
    """
    if not isinstance(input_tokens, int):
        return None
    answers = len((envelope.get("response") or {}).get("answers") or {})
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": answers,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + answers,
    }


class DecisionEngineModel(SimpleResponsesAPIModel):
    config: DecisionEngineModelConfig

    def model_post_init(self, context: Any) -> None:
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        return super().model_post_init(context)

    def _mints_response_ids(self) -> bool:
        return True

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = "Bearer " + self.config.api_key
        return headers

    async def _post(self, path: str, payload: Dict[str, Any]) -> Tuple[int, str]:
        response = await request(
            method="POST",
            url=self.config.base_url.rstrip("/") + path,
            headers=self._headers(),
            json=payload,
            _max_num_tries=1,
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_seconds),
        )
        return response.status, await response.text()

    async def call_engine(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        """One engine read with retries. Returns the envelope body (never raises for engine-side failures)."""
        if self.config.endpoint == "decisions":
            path, payload = (
                "/v1/decisions",
                {**to_decisions_request(self.config.model, state, questions), **self.config.extra_body},
            )
        else:
            path, payload = (
                "/v1/systemone",
                {"model": self.config.model, "state": state, "questions": questions, **self.config.extra_body},
            )
        error: Optional[str] = None
        start = time.perf_counter()
        for attempt in range(self.config.max_attempts):
            if attempt:
                await asyncio.sleep(self.config.backoff_seconds * 2 ** (attempt - 1))
            try:
                async with self._semaphore:
                    t = time.perf_counter()
                    status, text = await self._post(path, payload)
                    wall_ms = 1000 * (time.perf_counter() - t)
            except (asyncio.TimeoutError, aiohttp.ClientError, OSError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                continue
            if status in (400, 413, 422) and any(marker in text for marker in CAPACITY_MARKERS):
                return {"status": "unsupported", "error": text[:2000], "wall_ms": wall_ms}
            if status == 429 or status >= 500:
                error = f"HTTP {status}: {text[:500]}"
                continue
            if status >= 400:
                return {"status": "error", "error": f"HTTP {status}: {text[:2000]}", "wall_ms": wall_ms}
            try:
                raw = json.loads(text)
            except ValueError:
                return {"status": "error", "error": f"non-JSON reply: {text[:500]}", "wall_ms": wall_ms}
            if self.config.endpoint == "decisions":
                raw = from_decisions_response(raw)
            response = {k: v for k, v in raw.items() if k != "evaluation_trace"}
            return {"status": "ok", "response": response, "wall_ms": wall_ms}
        return {"status": "error", "error": error, "wall_ms": 1000 * (time.perf_counter() - start)}

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        payload = decision_request(body)
        envelope = await self.call_engine(payload["state"], payload["questions"])
        text = json.dumps({ENVELOPE_KEY: envelope}, ensure_ascii=False, separators=(",", ":"))
        engine_usage = (envelope.get("response") or {}).get("usage") or {}
        input_tokens = engine_usage.get("input_tokens")
        output: List[Dict[str, Any]] = [
            {
                "id": f"msg_{uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ]
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time.time()),
            model=self.config.model,
            object="response",
            output=output,
            tool_choice="none",
            parallel_tool_calls=False,
            tools=[],
            usage=usage_of(input_tokens, envelope),
        )

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        raise HTTPException(status_code=400, detail="decision_engine_model serves /v1/responses only")


if __name__ == "__main__":
    DecisionEngineModel.run_webserver()
