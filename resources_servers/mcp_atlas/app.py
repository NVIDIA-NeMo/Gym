# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP-Atlas: multi-step tool use against the real MCP servers of the upstream agent-environment image.

Each rollout gets its own sandbox that serves only the task's MCP servers. The model sees the task's
``ENABLED_TOOLS`` and every call is proxied to the sandbox's ``/call-tool``. The verifier judges the final
answer claim by claim (see ``verifier.py``); ``reward`` is 1 when claim coverage reaches the pass threshold.

When the environment rather than the model fails (the sandbox does not start, a task tool is missing, a
service answers with an auth, quota, TLS or server error), the rollout still runs but its verify response
sets ``mask_sample`` with a ``mcp_atlas:<kind>`` ``failure_kind``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

from aiohttp import ClientError
from fastapi import FastAPI, Request
from pydantic import ConfigDict, Field, PositiveFloat, PositiveInt, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ClosedResourcesSessionRecords,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.episode_types import EpisodeId
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import NeMoGymChatCompletion
from nemo_gym.server_utils import SESSION_ID_KEY


if __package__:
    from .sandbox import AtlasSandbox, SandboxError, SandboxPool, call_sandbox_tool
    from .verifier import coverage, final_answer, judge_request, parse_verdict
else:
    # Gym starts a resources server from its own directory, so sibling modules import at top level.
    from sandbox import AtlasSandbox, SandboxError, SandboxPool, call_sandbox_tool
    from verifier import coverage, final_answer, judge_request, parse_verdict


logger = logging.getLogger(__name__)

# The credentials of the 16 key-gated MCP servers, by the names the upstream image reads.
SERVICE_ENV_NAMES = (
    "AIRTABLE_API_KEY",
    "ALCHEMY_API_KEY",
    "BRAVE_API_KEY",
    "E2B_API_KEY",
    "EXA_API_KEY",
    "GITHUB_TOKEN",
    "GOOGLE_MAPS_API_KEY",
    "GOOGLE_CLIENT_ID",
    "GOOGLE_CLIENT_SECRET",
    "GOOGLE_REFRESH_TOKEN",
    "LARA_ACCESS_KEY_ID",
    "LARA_ACCESS_KEY_SECRET",
    "NPS_API_KEY",
    "MONGODB_CONNECTION_STRING",
    "NOTION_TOKEN",
    "OXYLABS_USERNAME",
    "OXYLABS_PASSWORD",
    "SLACK_MCP_XOXC_TOKEN",
    "SLACK_MCP_XOXD_TOKEN",
    "TWELVE_DATA_API_KEY",
    "WEATHER_API_KEY",
)

# The image's uvx and npx servers install packages when they start. These user configs pin that resolution to
# the 1.2.7 image date: later releases (mcp 2.x) break several servers. They are files, not variables, because
# MCP stdio clients pass only HOME, PATH and a few more variables to the servers they start.
DEFAULT_SANDBOX_FILES = {
    "/root/.config/uv/uv.toml": 'exclude-newer = "2026-07-17T00:00:00Z"\ncache-dir = "/tmp/uv-cache"\n',
    "/root/.npmrc": "before=2026-07-17\ncache=/tmp/npm-cache\nupdate-notifier=false\n",
}

# Tool errors that name the service or the network, not the model's request.
SERVICE_ERROR_PATTERN = re.compile(
    r"\b(?:401|402|403|429|502|503|504)\b|unauthori[sz]ed|forbidden|api[ _-]?key|rate[ _-]?limit|"
    r"too many requests|quota|payment required|credits|certificate|\bssl\b|\btls\b|"
    r"connection (?:refused|reset|closed|aborted)|timed out|timeout|name or service not known|"
    r"name resolution|getaddrinfo|econnrefused|enotfound|etimedout|econnreset|"
    r"client failed to connect|session was closed",
    re.IGNORECASE,
)
EMPTY_RESULT = "success"
# Failure kinds, most severe first. The first group means the rollout had no working environment and is
# always masked; the second is masked when mask_service_errors is set.
HARD_FAILURES = ("session_lost", "sandbox_start_failed", "sandbox_unreachable", "tools_missing")
SOFT_FAILURES = ("service_error", "tool_timeout")


class MCPAtlasResourcesServerConfig(BaseResourcesServerConfig):
    # Path to the agent-environment SIF; one sandbox per rollout.
    sandbox_image: str = ""
    # An already running agent-environment (for example `docker run -p 1984:1984`); shared by every
    # rollout, so stateful servers leak between tasks. For development only.
    sandbox_base_url: str = ""
    sandbox_provider: dict[str, Any] = Field(
        default_factory=lambda: {
            "apptainer": {
                "create": {
                    "start_timeout_s": 600,
                    "extra_start_args": ["--userns", "--writable-tmpfs", "--no-home", "--cleanenv"],
                    "apply_resource_limits": False,
                },
                # --cleanenv keeps this server's own environment (judge and cluster keys) out of the sandbox,
                # where the model's shell and code tools could read it.
                "exec": {
                    "fakeroot_for_root": False,
                    "default_timeout_s": 180,
                    "concurrency": 64,
                    "extra_exec_args": ["--cleanenv"],
                },
            }
        }
    )
    sandbox_env: dict[str, str] = Field(default_factory=dict)
    sandbox_files: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_SANDBOX_FILES))
    service_env_names: list[str] = Field(default_factory=lambda: list(SERVICE_ENV_NAMES))
    sandbox_port_range: tuple[int, int] = (27000, 28999)
    sandbox_max_concurrency: PositiveInt = 32
    sandbox_startup_timeout_s: PositiveFloat = 600
    sandbox_ttl_s: PositiveFloat = 3600
    sandbox_scratch_root: Optional[str] = None
    tool_call_timeout_s: PositiveFloat = 60
    max_tool_calls: PositiveInt = 100
    judge_model_server: ModelServerRef
    judge_max_tokens: Optional[PositiveInt] = None
    judge_max_concurrency: PositiveInt = 64
    # Judge calls per claim until one gives a readable verdict; upstream also makes 8.
    judge_max_attempts: PositiveInt = 8
    pass_threshold: float = 0.75
    mask_service_errors: bool = True
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class MCPAtlasSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: dict[str, Any]


class MCPAtlasVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: dict[str, Any]


class MCPAtlasVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    coverage: float
    passed_at_0_50: bool
    n_claims: int
    verdicts: list[dict[str, Any]]
    tool_calls: int
    service_errors: list[str]


@dataclass
class _Rollout:
    task_id: str
    tools: frozenset[str]
    box: Optional[AtlasSandbox] = None
    tool_calls: int = 0
    # (kind, detail) of every environment failure seen in this rollout.
    failures: list[tuple[str, str]] = field(default_factory=list)


def tool_output_text(body: Any) -> str:
    """The text the model sees for a successful call: the text blocks, or "success" when there are none."""
    if not isinstance(body, list):
        return json.dumps(body)
    parts = []
    for block in body:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        else:
            parts.append(json.dumps(block))
    text = "\n".join(parts)
    return text if text.strip() else EMPTY_RESULT


def saved_environment(body: MCPAtlasVerifyRequest) -> Optional[tuple[list[tuple[str, str]], int]]:
    """The environment failures and tool-call count a saved rollout row records, or None for a live request.

    ``gym eval reverify`` posts the saved row without a session, so its first verify's ``service_errors`` and
    ``tool_calls`` take the place of the live rollout state; the answer is then judged again.
    """
    extra = body.model_extra or {}
    errors, tool_calls = extra.get("service_errors"), extra.get("tool_calls")
    if not isinstance(errors, list) or not isinstance(tool_calls, int):
        return None
    failures = []
    for entry in errors:
        kind, _, detail = str(entry).partition(": ")
        failures.append((kind, detail))
    return failures, tool_calls


class MCPAtlasResourcesServer(SimpleResourcesServer):
    config: MCPAtlasResourcesServerConfig
    _rollouts: dict[str, _Rollout] = PrivateAttr(default_factory=dict)
    _pool: Optional[SandboxPool] = PrivateAttr(default=None)
    _judge_slots: asyncio.Semaphore = PrivateAttr()
    _reaper: Optional[asyncio.Task] = PrivateAttr(default=None)
    _closed_sessions: ClosedResourcesSessionRecords[EpisodeId] = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("mcp_atlas keeps one sandbox per rollout in-process and requires num_workers=1")
        if bool(self.config.sandbox_image) == bool(self.config.sandbox_base_url):
            raise ValueError("set exactly one of sandbox_image or sandbox_base_url")
        self._judge_slots = asyncio.Semaphore(self.config.judge_max_concurrency)
        self._closed_sessions = ClosedResourcesSessionRecords(self.config.session_close_retry_window_seconds)
        if self.config.sandbox_image:
            if not os.path.isfile(self.config.sandbox_image):
                raise ValueError(f"sandbox_image is not a file: {self.config.sandbox_image}")
            services = {name: os.environ[name] for name in self.config.service_env_names if os.environ.get(name)}
            logger.info("MCP-Atlas sandboxes get %d of %d service credentials", len(services), len(SERVICE_ENV_NAMES))
            self._pool = SandboxPool(
                image=self.config.sandbox_image,
                sandbox_provider=self.config.sandbox_provider,
                sandbox_env={**self.config.sandbox_env, **services},
                sandbox_files=self.config.sandbox_files,
                port_range=self.config.sandbox_port_range,
                max_concurrency=self.config.sandbox_max_concurrency,
                startup_timeout_s=self.config.sandbox_startup_timeout_s,
                ttl_s=self.config.sandbox_ttl_s,
                scratch_root=self.config.sandbox_scratch_root,
            )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.router.add_event_handler("startup", self._start_reaper)
        app.router.add_event_handler("shutdown", self._shutdown)
        # Registered last, so the protocol routes above match first.
        app.post("/{tool_name}")(self.call_tool)
        return app

    async def _start_reaper(self) -> None:
        if self._pool is not None and self._reaper is None:
            self._reaper = asyncio.create_task(self._reap_forever())

    async def _reap_forever(self) -> None:
        while True:
            await asyncio.sleep(60)
            try:
                await self._pool.reap_expired()
            except Exception:
                logger.exception("MCP-Atlas sandbox reaper failed; retrying in 60s")

    async def _shutdown(self) -> None:
        if self._reaper is not None:
            self._reaper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper
        if self._pool is not None:
            try:
                await self._pool.close()
            except SandboxError as exc:
                logger.error("MCP-Atlas shutdown left sandboxes for the provider's ttl_s to reclaim: %s", exc)

    async def seed_session(self, request: Request, body: MCPAtlasSeedSessionRequest) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        if session_id in self._closed_sessions:
            raise ValueError(f"MCP-Atlas session {session_id} is already closed")
        meta = body.verifier_metadata
        rollout = _Rollout(task_id=str(meta.get("task_id", "")), tools=frozenset(meta.get("enabled_tools", [])))
        if self._pool is None:
            names = {name: name for name in rollout.tools}
            rollout.box = AtlasSandbox(base_url=self.config.sandbox_base_url, tool_names=names, missing_tools=[])
        else:
            try:
                rollout.box = await self._pool.start(
                    session_id, servers=list(meta.get("servers", [])), tools=sorted(rollout.tools)
                )
            except SandboxError as exc:
                # Not raised: a failed seed would end the whole collection run. The rollout runs without
                # tools and its verify response is masked.
                logger.warning("MCP-Atlas sandbox start failed for task %s: %s", rollout.task_id, exc)
                rollout.failures.append(("sandbox_start_failed", str(exc)[:2000]))
            else:
                if rollout.box.missing_tools:
                    rollout.failures.append(("tools_missing", ", ".join(rollout.box.missing_tools)))
        self._rollouts[session_id] = rollout
        return BaseSeedSessionResponse()

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        """Stop the session's sandbox and drop its rollout, so an unfinished rollout frees its capacity slot.

        Closing twice, or closing a session that was never seeded, succeeds. When the sandbox does not stop,
        SandboxError is raised and the close is not recorded, so the caller can retry it.
        """
        # A /run close names its run in the body and this server's session in the cookie; a typed close names
        # the same session in both.
        session_ids = {body.resources_session_id, request.session.get(SESSION_ID_KEY)} - {None}
        if self._pool is not None:
            for session_id in session_ids:
                await self._pool.stop(session_id)
        for session_id in session_ids:
            self._rollouts.pop(session_id, None)
            self._closed_sessions.remember(session_id, body.episode_id)
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    async def call_tool(self, request: Request, tool_name: str, body: dict[str, Any]) -> str:
        """Proxy one tool call; every failure becomes a model-visible tool output, never an HTTP error."""
        rollout = self._rollouts.get(request.session.get(SESSION_ID_KEY))
        if rollout is None:
            return "Error: no MCP-Atlas session; seed_session was not called"
        if tool_name not in rollout.tools:
            return f"Error: tool '{tool_name}' is not available for this task"
        if rollout.tool_calls >= self.config.max_tool_calls:
            return f"Error: the limit of {self.config.max_tool_calls} tool calls is reached; give your final answer"
        rollout.tool_calls += 1
        if rollout.box is None:
            return "Error: the MCP environment is unavailable"
        if tool_name == "met-museum_get-museum-object":
            # Upstream's harness does the same: image blocks do not fit a text tool output.
            body = {**body, "returnImage": False}
        try:
            status, payload = await call_sandbox_tool(
                rollout.box, tool_name, body, timeout_s=self.config.tool_call_timeout_s
            )
        except asyncio.TimeoutError:
            rollout.failures.append(("tool_timeout", tool_name))
            return f"Tool call timed out after {self.config.tool_call_timeout_s:g}s"
        except (ClientError, OSError, SandboxError) as exc:
            rollout.failures.append(("sandbox_unreachable", f"{tool_name}: {type(exc).__name__}: {exc}"))
            return f"Error: tool call failed: {type(exc).__name__}"
        if status == 200:
            text = tool_output_text(payload)
            # Some servers report a failed upstream call as a short successful result.
            if len(text) < 500 and "error" in text.lower() and SERVICE_ERROR_PATTERN.search(text):
                rollout.failures.append(("service_error", f"{tool_name}: {text}"))
            return text
        # The sandbox reports every tool error as an HTTP error; the model sees its body, as upstream.
        text = payload if isinstance(payload, str) else json.dumps(payload)
        if status != 500 or SERVICE_ERROR_PATTERN.search(text):
            rollout.failures.append(("service_error", f"{tool_name}: HTTP {status}: {text[:500]}"))
        return text

    async def _judge(self, claim: str, answer: str) -> dict[str, Any]:
        """Judge one claim. Raises JudgeError when no attempt gives a readable verdict.

        Upstream scores such a claim 0; here judge_failsafe masks the rollout, because the measurement failed.
        """
        params = judge_request(claim=claim, answer=answer, max_tokens=self.config.judge_max_tokens)
        content = None
        for _attempt in range(self.config.judge_max_attempts):
            async with self._judge_slots:
                completion = await call_judge(
                    self.server_client,
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/chat/completions",
                    json=params,
                    response_model=NeMoGymChatCompletion,
                )
            content = completion.choices[0].message.content if completion.choices else None
            verdict = parse_verdict(claim, content)
            if verdict is not None:
                return verdict
        raise JudgeError(
            f"no readable judge verdict after {self.config.judge_max_attempts} attempts: {(content or '')[:500]}"
        )

    async def verify(self, request: Request, body: MCPAtlasVerifyRequest) -> MCPAtlasVerifyResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        rollout = self._rollouts.get(session_id)
        if self._pool is not None and session_id is not None:
            try:
                await self._pool.stop(session_id)
            except SandboxError as exc:
                # The pool keeps the sandbox, and the reaper stops it again; the measurement is unaffected.
                logger.warning("MCP-Atlas sandbox of session %s did not stop at verify: %s", session_id, exc)
        try:
            claims = [c for c in body.verifier_metadata.get("claims", []) if isinstance(c, str) and c]
            answer = final_answer(body.response)
            if rollout is not None:
                failures, tool_calls = list(rollout.failures), rollout.tool_calls
            else:
                failures, tool_calls = saved_environment(body) or ([("session_lost", "no session at verify")], 0)
            kinds = {kind for kind, _ in failures}
            sandbox_down = bool(kinds & {"sandbox_start_failed", "session_lost"})
            # Upstream scores an empty or "ERROR:" answer 0 without calling the judge.
            judged = claims and answer.strip() and not answer.startswith("ERROR:") and not sandbox_down
            verdicts = list(await asyncio.gather(*(self._judge(c, answer) for c in claims))) if judged else []
            score = coverage(verdicts, len(claims)) if judged else 0.0
            masked = bool(kinds & set(HARD_FAILURES)) or (self.config.mask_service_errors and bool(kinds))
            worst = next((kind for kind in HARD_FAILURES + SOFT_FAILURES if kind in kinds), None)
            # A reverified row carries the fields set below from its first verify; they are computed again.
            computed = set(MCPAtlasVerifyResponse.model_fields) - set(MCPAtlasVerifyRequest.model_fields)
            fields = body.model_dump(exclude=computed)
            return MCPAtlasVerifyResponse(
                **fields,
                reward=1.0 if score >= self.config.pass_threshold else 0.0,
                coverage=score,
                passed_at_0_50=score >= 0.5,
                n_claims=len(claims),
                verdicts=verdicts,
                tool_calls=tool_calls,
                service_errors=[f"{kind}: {detail}" for kind, detail in failures],
                mask_sample=masked,
                failure_kind=f"mcp_atlas:{worst}" if worst else None,
                failure_reason="; ".join(f"{kind}: {detail}" for kind, detail in failures)[:2000] or None,
            )
        finally:
            self._rollouts.pop(session_id, None)


if __name__ == "__main__":
    MCPAtlasResourcesServer.run_webserver()
