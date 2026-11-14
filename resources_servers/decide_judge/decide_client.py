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
"""Async client for Nemotron Decide, a decision model that returns option probabilities in one forward pass.

Nemotron Decide does not generate text. A request carries a JSON ``state`` and typed questions; the answer to every
question is a probability per option. Two request shapes are supported and give the same read:

- ``systemone``: ``POST {base_url}/v1/systemone`` with ``{"model", "state", "questions": {name: {...}}}``.
  Yes/no questions are ``noul`` (answer ``noul`` = p(yes)), choice questions are ``choice`` with ``criteria``.
- ``decisions``: ``POST {base_url}/v1/decisions`` (OpenAI Decisions API shape). The state is sent as a single
  ``input_json`` part, which the server reads as the same JSON state; yes/no questions are ``predicate``.

The API key is read from the environment variable named by ``api_key_env`` at call time and never stored in config.
All HTTP goes through NeMo Gym's shared aiohttp client.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal, Optional

from aiohttp import ClientError, ClientTimeout
from pydantic import BaseModel, Field, field_validator


DEFAULT_BASE_URL = "http://localhost:8770"
RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class NemotronDecideEndpointConfig(BaseModel):
    """Where and how to reach Nemotron Decide."""

    base_url: str = Field(
        default=DEFAULT_BASE_URL,
        description=(
            "Server root without the /v1/... route: a locally served model (default "
            f"{DEFAULT_BASE_URL}) or any compatible endpoint."
        ),
    )
    api: Literal["systemone", "decisions"] = "systemone"
    model: str = "nemotron-decide"
    api_key_env: Optional[str] = Field(
        default="DECIDE_API_KEY",
        description="Name of the environment variable holding the bearer token. Empty or null sends no auth header.",
    )
    timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0, description="Extra attempts on 408/425/429/5xx and connection errors.")
    retry_backoff_s: float = Field(default=0.5, ge=0)
    max_concurrency: int = Field(default=32, ge=1)

    @field_validator("base_url")
    @classmethod
    def _strip_slash(cls, value: str) -> str:
        return value.rstrip("/")


class DecideQuestion(BaseModel):
    """A normalized question: ``yes_no`` or ``choice`` over named options."""

    name: str
    kind: Literal["yes_no", "choice"]
    instructions: str
    options: dict[str, str] = Field(default_factory=dict, description="choice only: option value -> description")


class DecideAnswer(BaseModel):
    name: str
    kind: Literal["yes_no", "choice"]
    probabilities: dict[str, float]
    choice: str
    confidence: float

    @property
    def p_yes(self) -> float:
        return self.probabilities["yes"]


@dataclass
class DecideResult:
    answers: dict[str, DecideAnswer]
    latency_ms: float
    usage: dict[str, Any] = field(default_factory=dict)
    server_metrics: dict[str, Any] = field(default_factory=dict)


class NemotronDecideError(RuntimeError):
    """The endpoint could not produce a read (transport, auth, rate limit, or malformed reply)."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


def _yes_no(p_yes: float) -> tuple[dict[str, float], str, float]:
    p_yes = min(1.0, max(0.0, float(p_yes)))
    probs = {"yes": p_yes, "no": 1.0 - p_yes}
    choice = "yes" if p_yes >= 0.5 else "no"
    return probs, choice, probs[choice]


class NemotronDecideClient:
    def __init__(self, config: NemotronDecideEndpointConfig):
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    @property
    def url(self) -> str:
        return f"{self.config.base_url}/v1/{self.config.api}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            key = os.environ.get(self.config.api_key_env)
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return headers

    # ----- request / response conversion -----

    def build_payload(self, state: Any, questions: list[DecideQuestion]) -> dict[str, Any]:
        names = [q.name for q in questions]
        if not questions:
            raise ValueError("at least one question is required")
        if len(set(names)) != len(names):
            raise ValueError(f"question names must be unique: {names}")
        for q in questions:
            if q.kind == "choice" and len(q.options) < 2:
                raise ValueError(f"choice question {q.name!r} needs at least two options")
        if self.config.api == "systemone":
            out: dict[str, Any] = {}
            for q in questions:
                if q.kind == "yes_no":
                    out[q.name] = {"type": "noul", "instructions": q.instructions}
                else:
                    out[q.name] = {"type": "choice", "instructions": q.instructions, "criteria": dict(q.options)}
            return {"model": self.config.model, "state": state, "questions": out}
        decision_questions = []
        for q in questions:
            if q.kind == "yes_no":
                decision_questions.append({"type": "predicate", "name": q.name, "instructions": q.instructions})
            else:
                decision_questions.append(
                    {
                        "type": "choice",
                        "name": q.name,
                        "instructions": q.instructions,
                        "choices": [{"value": v, "description": d} for v, d in q.options.items()],
                    }
                )
        return {
            "model": self.config.model,
            "input": [{"role": "user", "content": [{"type": "input_json", "data": state}]}],
            "questions": decision_questions,
        }

    def parse_answers(self, body: dict[str, Any], questions: list[DecideQuestion]) -> dict[str, DecideAnswer]:
        by_name = {q.name: q for q in questions}
        raw = body.get("answers")
        if self.config.api == "systemone":
            if not isinstance(raw, dict):
                raise NemotronDecideError(f"systemone reply has no answers object: {str(body)[:300]}")
            items = raw.items()
        else:
            if not isinstance(raw, list):
                raise NemotronDecideError(f"decisions reply has no answers list: {str(body)[:300]}")
            items = [(a.get("name"), a) for a in raw]
        answers: dict[str, DecideAnswer] = {}
        for name, a in items:
            q = by_name.get(name)
            if q is None:
                continue
            if q.kind == "yes_no":
                p = a.get("noul") if self.config.api == "systemone" else a.get("probability")
                if p is None:
                    raise NemotronDecideError(f"yes/no answer {name!r} has no probability: {a}")
                probs, choice, confidence = _yes_no(p)
            else:
                if self.config.api == "systemone":
                    probs = {str(k): float(v) for k, v in (a.get("probabilities") or {}).items()}
                else:
                    probs = {str(p["value"]): float(p["probability"]) for p in a.get("probabilities") or []}
                if set(probs) != set(q.options):
                    raise NemotronDecideError(f"choice answer {name!r} options {sorted(probs)} != {sorted(q.options)}")
                choice = max(probs, key=probs.get)
                confidence = probs[choice]
            answers[name] = DecideAnswer(
                name=name, kind=q.kind, probabilities=probs, choice=choice, confidence=confidence
            )
        missing = [q.name for q in questions if q.name not in answers]
        if missing:
            raise NemotronDecideError(f"reply is missing answers for {missing}")
        return answers

    # ----- transport -----

    async def _post(self, payload: dict[str, Any]) -> tuple[int, Any]:  # pragma: no cover - exercised live
        from nemo_gym.client import request

        response = await request(
            method="POST",
            url=self.url,
            json=payload,
            headers=self._headers(),
            timeout=ClientTimeout(total=self.config.timeout_s),
            _max_num_tries=1,
        )
        text = await response.text()
        try:
            return response.status, json.loads(text)
        except json.JSONDecodeError:
            return response.status, text

    async def decide(self, state: Any, questions: list[DecideQuestion]) -> DecideResult:
        payload = self.build_payload(state, questions)
        last_error: Optional[NemotronDecideError] = None
        async with self._semaphore:
            for attempt in range(self.config.max_retries + 1):
                if attempt:
                    await asyncio.sleep(self.config.retry_backoff_s * 2 ** min(attempt - 1, 4))
                started = perf_counter()
                try:
                    status, body = await self._post(payload)
                except (ClientError, asyncio.TimeoutError, OSError) as error:
                    last_error = NemotronDecideError(f"{type(error).__name__}: {error}")
                    continue
                latency_ms = (perf_counter() - started) * 1000
                if status == 200 and isinstance(body, dict):
                    return DecideResult(
                        answers=self.parse_answers(body, questions),
                        latency_ms=latency_ms,
                        usage=body.get("usage") if isinstance(body.get("usage"), dict) else {},
                        server_metrics=body.get("metrics") if isinstance(body.get("metrics"), dict) else {},
                    )
                last_error = NemotronDecideError(f"HTTP {status}: {str(body)[:500]}", status=status)
                if status not in RETRYABLE_STATUSES:
                    break
        assert last_error is not None
        raise last_error


# ----- rendering Responses API items into a compact JSON state -----


def _as_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json", exclude_none=True)
    return {"type": "unknown", "value": str(item)}


def clip(text: str, max_chars: int) -> str:
    """Keep the head and tail of long text so both the start and the end of a tool output stay visible."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return f"{text[:head]} …[{len(text) - max_chars} chars omitted]… {text[-tail:]}"


def content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            part = _as_dict(part)
            if isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif part.get("type") == "refusal" and isinstance(part.get("refusal"), str):
                parts.append(part["refusal"])
        return "\n".join(parts)
    return str(content)


def parse_arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def render_items(items: list[Any], max_item_chars: int = 4000) -> list[dict[str, Any]]:
    """Responses API items -> a list of compact steps. Reasoning items are dropped (no hidden thoughts in the state)."""
    names_by_call: dict[str, str] = {}
    steps: list[dict[str, Any]] = []
    for raw in items:
        item = _as_dict(raw)
        kind = item.get("type") or ("message" if "role" in item else None)
        if kind == "message":
            text = content_text(item.get("content"))
            steps.append({"role": item.get("role", "user"), "text": clip(text, max_item_chars)})
        elif kind == "function_call":
            names_by_call[item.get("call_id", "")] = item.get("name", "")
            arguments = parse_arguments(item.get("arguments"))
            if isinstance(arguments, str):
                arguments = clip(arguments, max_item_chars)
            steps.append({"role": "assistant", "tool_call": {"name": item.get("name"), "arguments": arguments}})
        elif kind == "function_call_output":
            output = item.get("output")
            text = output if isinstance(output, str) else content_text(output)
            steps.append(
                {
                    "role": "tool",
                    "tool": names_by_call.get(item.get("call_id", ""), "unknown"),
                    "output": clip(text, max_item_chars),
                }
            )
    return steps


def user_and_system_text(items: list[Any]) -> tuple[str, str]:
    """The user's own words and the system/developer instructions, kept apart from tool data (trust boundary)."""
    user, system = [], []
    for raw in items:
        item = _as_dict(raw)
        kind = item.get("type") or ("message" if "role" in item else None)
        if kind != "message":
            continue
        role = item.get("role")
        if role == "user":
            user.append(content_text(item.get("content")))
        elif role in ("system", "developer"):
            system.append(content_text(item.get("content")))
    return "\n\n".join(t for t in user if t), "\n\n".join(t for t in system if t)
