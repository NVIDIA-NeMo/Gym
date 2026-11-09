# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness of a multi-turn visual-coding answer, decided by a GPT judge over the final answer.

Grading:
- the response text is every assistant text part joined, and the judge sees its last
  `judge_max_text_chars` characters;
- `extracted` is string_match's final_answer extraction (last "answer:", else last \\boxed{});
- correct iff the verdict is "equivalent"; "different" and "unsure" score 0;
- a judge failure (a failed call, or a reply without a valid verdict) is not scored: the base class's
  judge_failsafe returns the row masked (judge_failed).
string_match's grade of `extracted` is kept as a diagnostic, not the reward.

With ``sandbox_provider`` set, each episode's seed lends the harness a sandbox for its tools, and the episode's close
stops it. ``sandbox_network_access: model_only`` lets that sandbox reach only ``sandbox_model_server``, so the model
cannot look answers up. Grading never reads the sandbox.
"""

import asyncio
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

from fastapi import FastAPI, Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import ModelServerRef
from nemo_gym.core import EpisodeId, NeMoGymChatCompletion, TaskId
from nemo_gym.sandbox import SandboxConfig
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    LentSandboxes,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFences,
    ResourcesSessionLocks,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.verification import JudgeError, VerifierFixture, call_judge
from resources_servers.string_match.app import (
    _extract_answer,
    _extract_last_assistant_text,
    _grade_string_match,
)
from resources_servers.visual_coding_opencode.task_data import TaskData
from resources_servers.visual_coding_opencode.verifier_fixture import (
    create_visual_coding_opencode_server,
    invoke_visual_coding_opencode,
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
    """The verdict and reason from the first JSON object in the reply.

    A reply without a JSON object or without a valid verdict is a judge failure, not an "unsure" verdict: it raises
    JudgeError, so the row is masked instead of scored 0. "unsure" counts only when the judge says it.
    """
    match = re.search(r"\{.*\}", content or "", re.S)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        raise JudgeError(f"judge reply has no JSON object: {(content or '')[:200]!r}")
    verdict = str(data.get("verdict", "")).lower().strip()
    if verdict not in ("equivalent", "different", "unsure"):
        raise JudgeError(f"judge reply has no valid verdict: {(content or '')[:200]!r}")
    return {"verdict": verdict, "reason": str(data.get("reason", ""))[:200]}


class VisualCodingOpenCodeSandboxConfig(SandboxConfig):
    # An OCI image built from sandbox_image/Dockerfile: OpenCode, Python 3 and the image toolkit.
    image: Optional[str] = None


class VisualCodingOpenCodeConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    # With sandbox_provider set, each episode's seed lends the harness a sandbox, and its close stops it. Unset keeps
    # the server stateless, and the harness brings its own sandbox.
    sandbox_provider: Optional[str] = None
    sandbox_config: VisualCodingOpenCodeSandboxConfig = Field(default_factory=VisualCodingOpenCodeSandboxConfig)
    # The working directory of the lent sandbox. The image must create it (WORKDIR in sandbox_image/Dockerfile).
    sandbox_workdir: str = "/workspace"
    # model_only: the lent sandbox reaches only sandbox_model_server's host. inherit keeps the provider's network.
    sandbox_network_access: Literal["inherit", "model_only"] = "inherit"
    sandbox_model_server: Optional[ModelServerRef] = None
    judge_model_server: ModelServerRef
    judge_model: str
    judge_temperature: float = 0.0
    judge_top_p: float = 1.0
    judge_max_tokens: int = 1024
    judge_max_concurrency: int = Field(default=32, ge=1)
    judge_max_question_chars: int = Field(default=2000, ge=1)
    judge_max_text_chars: int = Field(default=4000, ge=1)


class VisualCodingOpenCodeVerifyRequest(TaskData, BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


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
    # The episode and task each open session was seeded for, keyed by resources_session_id.
    _sessions: dict[str, tuple[EpisodeId, TaskId]] = PrivateAttr(default_factory=dict)
    _locks: ResourcesSessionLocks = PrivateAttr()
    _fences: ResourcesSessionFences[EpisodeId] = PrivateAttr()
    _lent_sandboxes: LentSandboxes | None = PrivateAttr(default=None)

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._locks = ResourcesSessionLocks()
        self._fences = ResourcesSessionFences(self.config.session_close_retry_window_seconds)
        if self.config.sandbox_provider is not None:
            if not self.config.sandbox_config.image:
                raise ValueError(
                    "Set sandbox_config.image to an image built from resources_servers/visual_coding_opencode/sandbox_image"
                )
            model_server = self.config.sandbox_model_server
            self._lent_sandboxes = LentSandboxes(
                self.config.sandbox_provider,
                self.config.sandbox_config,
                image=self.config.sandbox_config.image,
                workdir=self.config.sandbox_workdir,
                network_access=self.config.sandbox_network_access,
                model_server=model_server.name if model_server is not None else None,
            )
        elif self.config.sandbox_network_access != "inherit":
            raise ValueError("sandbox_network_access applies to a lent sandbox; set sandbox_provider")

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as maybe_state:
                    yield maybe_state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        return app

    async def seed_session(
        self, request: Request, body: ResourcesSeedSessionRequest | BaseSeedSessionRequest
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        """Lend the episode a sandbox when the server lends one; otherwise keep no state."""
        if not isinstance(body, ResourcesSeedSessionRequest) or self._lent_sandboxes is None:
            return await super().seed_session(body)
        session_id = body.resources_session_id
        async with self._locks.locked(session_id):
            self._fences.raise_if_fenced(session_id)
            seeded = self._sessions.get(session_id)
            if seeded is not None and seeded != (body.episode_id, body.task_id):
                raise ResourcesSessionConflictError("resources_session_id is already bound to another episode or task")
            # A repeated seed gets the sandbox it already has.
            sandbox_access = await self._lent_sandboxes.lend(session_id, body.task_id)
            self._sessions[session_id] = (body.episode_id, body.task_id)
            request.session[SESSION_ID_KEY] = session_id
            return ResourcesSeedSessionResponse(resources_session_id=session_id, sandbox_access=sandbox_access)

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        """Stop the episode's sandbox. Safe to repeat, and for a session never seeded."""
        session_id = body.resources_session_id
        async with self._locks.locked(session_id):
            closed_episode = self._fences.get(session_id)
            if closed_episode is not None and closed_episode != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the closed resources session")
            seeded = self._sessions.get(session_id)
            if seeded is not None and seeded[0] != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the seeded resources session")
            # A failed stop raises before the fence, so the caller can close again.
            if self._lent_sandboxes is not None:
                await self._lent_sandboxes.release(session_id)
            self._sessions.pop(session_id, None)
            self._fences.fence(session_id, body.episode_id)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def shutdown(self) -> None:
        if self._lent_sandboxes is not None:
            await self._lent_sandboxes.release_all()
        self._sessions.clear()

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
        """Score the final answer with the judge.

        A judge failure (a failed call, or a reply without a valid verdict) is not caught here. The base class's
        judge_failsafe returns the row masked (mask_sample, failure_kind judge_failed), so RL does not train on it
        as a wrong answer.
        """
        result = VisualCodingOpenCodeVerifyResponse(**body.model_dump(), reward=0.0)
        text = _extract_last_assistant_text(body)
        if not text:
            # A timed-out or failed run leaves no answer: wrong, with no judge call.
            result.failure_reason = "no_response"
            return result
        result.extracted_answer = _extract_answer(text, "final_answer")
        if result.extracted_answer is not None:
            result.string_match_reward = _grade_string_match(body.expected_answer, result.extracted_answer)
        judged = await self.judge(body.question, body.expected_answer, result.extracted_answer, text)
        result.judge_verdict = judged["verdict"]
        result.judge_reason = judged["reason"]
        result.reward = float(judged["verdict"] == "equivalent")
        if not result.reward:
            result.failure_reason = "incorrect_answer"
        return result


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_visual_coding_opencode_server,
    request_model=VisualCodingOpenCodeVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
    invoke=invoke_visual_coding_opencode,
)


if __name__ == "__main__":
    VisualCodingOpenCodeServer.run_webserver()
