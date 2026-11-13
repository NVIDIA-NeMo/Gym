# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP-Atlas: multi-step tool use against the real MCP servers of the upstream agent-environment image.

Each rollout gets its own sandbox that serves only the task's MCP servers. The model sees the task's
``ENABLED_TOOLS`` and every call is proxied to the sandbox's ``/call-tool``. The verifier judges the final
answer claim by claim (see ``verifier.py``); ``reward`` is 1 when claim coverage reaches the pass threshold.

Invalid credential configuration is rejected separately from scoring. When the environment rather than the
model fails (the sandbox does not start, a task tool is missing, or a service has quota/TLS/server errors),
the rollout still runs but its verify response
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
from pathlib import Path
from typing import Any, ClassVar, Optional

from aiohttp import ClientError
from fastapi import FastAPI, HTTPException, Request
from pydantic import ConfigDict, Field, PositiveFloat, PositiveInt, PrivateAttr

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import ModelServerRef
from nemo_gym.core import EpisodeId, NeMoGymChatCompletion
from nemo_gym.sandbox import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSessionFences,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.verification import JudgeError, call_judge


if __package__:
    from .authentication import is_authentication_failure, tool_error_diagnostic, tool_server_name
    from .credentials import (
        SERVICE_ENV_NAMES,
        CredentialConfigurationError,
        configured_dataset_servers,
        enabled_servers,
        task_environment,
        validate_workload_environment,
    )
    from .sandbox import AtlasSandbox, SandboxError, SandboxPool, call_sandbox_tool
    from .verifier import coverage, final_answer, judge_request, parse_verdict
else:
    # Gym starts a resources server from its own directory, so sibling modules import at top level.
    from authentication import is_authentication_failure, tool_error_diagnostic, tool_server_name
    from credentials import (
        SERVICE_ENV_NAMES,
        CredentialConfigurationError,
        configured_dataset_servers,
        enabled_servers,
        task_environment,
        validate_workload_environment,
    )
    from sandbox import AtlasSandbox, SandboxError, SandboxPool, call_sandbox_tool
    from verifier import coverage, final_answer, judge_request, parse_verdict


logger = logging.getLogger(__name__)

# The image's uvx and npx servers install packages when they start. These user configs pin that resolution to
# the 1.2.7 image date: later releases (mcp 2.x) break several servers. They are files, not variables, because
# MCP stdio clients pass only HOME, PATH and a few more variables to the servers they start.
DEFAULT_SANDBOX_FILES = {
    "/root/.config/uv/uv.toml": 'exclude-newer = "2026-07-17T00:00:00Z"\n',
    "/root/.npmrc": "before=2026-07-17\nupdate-notifier=false\n",
}
# Apptainer's image paths may be read-only; its development sandboxes need writable scratch caches.
APPTAINER_SANDBOX_FILES = {
    "/root/.config/uv/uv.toml": 'exclude-newer = "2026-07-17T00:00:00Z"\ncache-dir = "/tmp/uv-cache"\n',
    "/root/.npmrc": "before=2026-07-17\ncache=/tmp/npm-cache\nupdate-notifier=false\n",
}

# Require an HTTP status or a diagnostic at the start of an error, rather than matching words/numbers
# in model arguments (for example an object ID of 502 or a query containing "credits").
SERVICE_HTTP_STATUSES = {401, 402, 403, 429, 502, 503, 504}
SERVICE_ERROR_PATTERN = re.compile(
    r"^(?:HTTP(?:/\d(?:\.\d)?)?\s+(?:401|402|403|408|429|5\d\d)\b|"
    r"(?:HTTPError|ClientResponseError):\s*(?:401|402|403|408|429|5\d\d)\b|"
    r"unauthori[sz]ed\b|forbidden\b|authentication failed\b|(?:API )?rate[ _-]?limit exceeded\b|"
    r"You have exceeded a secondary rate limit\b|"
    r"too many requests\b|payment required\b|certificate verify failed\b|"
    r"(?:SSLError|SSLCertVerificationError|ConnectionError|ConnectTimeout|ReadTimeout):|"
    r"connection (?:refused|reset|closed|aborted)\b|name or service not known\b|"
    r"name resolution failed\b|getaddrinfo failed\b|econnrefused\b|enotfound\b|etimedout\b|econnreset\b|"
    r"client failed to connect\b|session was closed\b)",
    re.IGNORECASE,
)
EMPTY_RESULT = "success"
# Failure kinds, most severe first. The first group means the rollout had no working environment and is
# always masked; the second is masked when mask_service_errors is set.
HARD_FAILURES = ("session_lost", "sandbox_start_failed", "sandbox_unreachable", "tools_missing")
SOFT_FAILURES = ("service_error", "tool_timeout")
# These tools return model-written program/file outcomes. Their error text is not a service diagnostic.
MODEL_RESULT_SERVERS = frozenset(
    {
        "calculator",
        "cli-mcp-server",
        "desktop-commander",
        "filesystem",
        "git",
        "memory",
        "mcp-code-executor",
        "mcp-server-code-runner",
    }
)


def _is_service_error(status: int, text: str, tool_name: str) -> bool:
    if status in SERVICE_HTTP_STATUSES:
        return True
    error, wrapped = tool_error_diagnostic(text, tool_name)
    # The image wraps upstream exceptions in "Failed to call tool:"; only there does a leading status
    # have HTTP meaning. A number elsewhere in a tool's message is not a service failure.
    if wrapped and re.match(r"(?:401|402|403|429|502|503|504)\b", error):
        return True
    return (status >= 400 or wrapped) and SERVICE_ERROR_PATTERN.match(error) is not None


class MCPAtlasResourcesServerConfig(BaseResourcesServerConfig):
    model_config = ConfigDict(hide_input_in_errors=True)
    # OCI image for OpenSandbox, or a local SIF with explicit development mode.
    sandbox_image: str = ""
    # An already running agent-environment (for example `docker run -p 1984:1984`); shared by every
    # rollout, so stateful servers leak between tasks. For development only.
    sandbox_base_url: str = ""
    sandbox_provider: str | dict[str, Any] = Field(default="sandbox", repr=False)
    sandbox_development_mode: bool = False
    # OpenSandbox create options. The requested policy is not evidence of deployment enforcement.
    sandbox_options: dict[str, Any] = Field(default_factory=dict)
    sandbox_env: dict[str, str] = Field(default_factory=dict, repr=False)
    # None selects provider defaults; an explicit mapping replaces them, including an empty mapping.
    sandbox_files: Optional[dict[str, str]] = None
    service_env_names: list[str] = Field(default_factory=lambda: list(SERVICE_ENV_NAMES))
    sandbox_port_range: tuple[int, int] = (27000, 28999)
    sandbox_max_concurrency: PositiveInt = 32
    sandbox_startup_timeout_s: PositiveFloat = 600
    sandbox_ttl_s: PositiveFloat = 3600
    sandbox_scratch_root: Optional[str] = None
    # Give a tool response the upstream driver's 1800 s task budget plus transport headroom.
    # This is a bridge deadline, not a claim about every installed tool's execution limit.
    tool_call_timeout_s: PositiveFloat = 1860
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
    servers: frozenset[str]
    box: Optional[AtlasSandbox] = None
    tool_calls: int = 0
    # (kind, detail) of every environment failure seen in this rollout.
    failures: list[tuple[str, str]] = field(default_factory=list)
    configuration_error: Optional[str] = None


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
    model_config = ConfigDict(hide_input_in_errors=True)
    config: MCPAtlasResourcesServerConfig
    _rollouts: dict[str, _Rollout] = PrivateAttr(default_factory=dict)
    _pool: Optional[SandboxPool] = PrivateAttr(default=None)
    _judge_slots: asyncio.Semaphore = PrivateAttr()
    _reaper: Optional[asyncio.Task] = PrivateAttr(default=None)
    _closed_sessions: ResourcesSessionFences[EpisodeId] = PrivateAttr()
    _terminal_auth_sessions: ResourcesSessionFences[str] = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("mcp_atlas keeps one sandbox per rollout in-process and requires num_workers=1")
        if bool(self.config.sandbox_image) == bool(self.config.sandbox_base_url):
            raise ValueError("set exactly one of sandbox_image or sandbox_base_url")
        if self.config.sandbox_base_url and not self.config.sandbox_development_mode:
            raise ValueError("sandbox_base_url requires sandbox_development_mode=true; do not use it for scores")
        self._judge_slots = asyncio.Semaphore(self.config.judge_max_concurrency)
        self._closed_sessions = ResourcesSessionFences(self.config.session_close_retry_window_seconds)
        self._terminal_auth_sessions = ResourcesSessionFences(self.config.session_close_retry_window_seconds)
        if self.config.sandbox_image:
            provider = resolve_provider_config(self.config.sandbox_provider, self.server_client.global_config_dict)
            provider_name = next(iter(provider))
            if provider_name == "apptainer":
                if not self.config.sandbox_development_mode:
                    raise ValueError("Apptainer requires sandbox_development_mode=true; do not use it for scores")
                if not os.path.isfile(self.config.sandbox_image):
                    raise ValueError(f"sandbox_image is not a file: {self.config.sandbox_image}")
            elif provider_name == "opensandbox":
                if os.path.isabs(self.config.sandbox_image) or self.config.sandbox_image.endswith(".sif"):
                    raise ValueError("OpenSandbox sandbox_image must be an OCI image reference, not a local SIF")
                policy = self.config.sandbox_options.get("network_policy")
                if not isinstance(policy, dict) or policy.get("defaultAction") != "deny":
                    raise ValueError(
                        "OpenSandbox requires sandbox_options.network_policy with defaultAction=deny; "
                        "deployment enforcement and allowed external services must be validated before scoring"
                    )
                from opensandbox.models.sandboxes import NetworkPolicy

                encoded_policy = NetworkPolicy.model_validate(policy).model_dump(
                    by_alias=True, mode="json", exclude_unset=True
                )
                if encoded_policy != policy:
                    raise ValueError(
                        "network_policy contains fields the installed OpenSandbox SDK drops or changes; "
                        "use a policy the SDK preserves exactly"
                    )
            else:
                raise ValueError("MCP-Atlas supports OpenSandbox, or Apptainer in explicit development mode")
            sandbox_files = self.config.sandbox_files
            if sandbox_files is None:
                sandbox_files = APPTAINER_SANDBOX_FILES if provider_name == "apptainer" else DEFAULT_SANDBOX_FILES
            unknown_names = sorted(set(self.config.service_env_names) - set(SERVICE_ENV_NAMES))
            if unknown_names:
                raise CredentialConfigurationError(f"unknown service credential names: {', '.join(unknown_names)}")
            services = {name: os.environ[name] for name in self.config.service_env_names if os.environ.get(name)}
            environment = {**services, **self.config.sandbox_env}
            required = configured_dataset_servers(
                self.server_client.global_config_dict,
                resources_server=self.config.name,
                root=Path(__file__).resolve().parents[2],
            )
            validate_workload_environment(environment)
            if required.servers:
                task_environment(environment, sorted(required.servers))
            for path in required.deferred_inputs:
                logger.warning(
                    "MCP-Atlas credential preflight awaits prepared input %s; each task is checked before startup",
                    path,
                )
            self._pool = SandboxPool(
                image=self.config.sandbox_image,
                sandbox_provider=provider,
                sandbox_options=self.config.sandbox_options,
                sandbox_metadata=resolve_provider_metadata(
                    self.config.sandbox_provider, self.server_client.global_config_dict
                ),
                sandbox_env=environment,
                sandbox_files=dict(sandbox_files),
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
                logger.error(
                    "MCP-Atlas shutdown could not stop every sandbox; manual cleanup may be required "
                    "(Apptainer does not enforce ttl_s): %s",
                    exc,
                )
                raise

    async def seed_session(self, request: Request, body: MCPAtlasSeedSessionRequest) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        async with self.session_lock(session_id):
            if session_id in self._closed_sessions:
                raise ValueError(f"MCP-Atlas session {session_id} is already closed")
            previous = self._rollouts.get(session_id)
            if previous is not None and previous.configuration_error:
                raise ResourcesSeedRejectedError(previous.configuration_error)
            meta = body.verifier_metadata
            try:
                servers = enabled_servers(meta.get("servers"))
                tools = meta.get("enabled_tools")
                if not isinstance(tools, list) or not all(isinstance(tool, str) for tool in tools):
                    raise CredentialConfigurationError("enabled_tools must be a list of tool names")
            except CredentialConfigurationError as exc:
                raise ResourcesSeedRejectedError(str(exc)) from exc
            rollout = _Rollout(
                task_id=str(meta.get("task_id", "")), tools=frozenset(tools), servers=frozenset(servers)
            )
            if self._pool is None:
                names = {name: name for name in rollout.tools}
                rollout.box = AtlasSandbox(base_url=self.config.sandbox_base_url, tool_names=names, missing_tools=[])
            else:
                try:
                    rollout.box = await self._pool.start(session_id, servers=servers, tools=sorted(rollout.tools))
                except CredentialConfigurationError as exc:
                    raise ResourcesSeedRejectedError(str(exc)) from exc
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
        for session_id in sorted(session_ids):
            async with self.session_lock(session_id):
                if self._pool is not None:
                    await self._pool.stop(session_id)
                rollout = self._rollouts.pop(session_id, None)
                if rollout is not None and rollout.configuration_error:
                    self._terminal_auth_sessions.fence(session_id, rollout.configuration_error)
                self._closed_sessions.fence(session_id, body.episode_id)
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    async def retire_session_state(self, session_id: str) -> None:
        """Release a retired attempt, retaining ownership when sandbox teardown fails."""
        async with self.session_lock(session_id):
            if self._pool is not None:
                await self._pool.stop(session_id)
            self._rollouts.pop(session_id, None)

    async def call_tool(self, request: Request, tool_name: str, body: dict[str, Any]) -> str:
        """Proxy a tool call; reject configuration failures without producing a scored service-failure row."""
        session_id = request.session.get(SESSION_ID_KEY, "")
        async with self.session_lock(session_id):
            rollout = self._rollouts.get(request.session.get(SESSION_ID_KEY))
            if rollout is None:
                return "Error: no MCP-Atlas session; seed_session was not called"
            if rollout.configuration_error:
                raise HTTPException(422, rollout.configuration_error)
            if tool_name not in rollout.tools:
                return f"Error: tool '{tool_name}' is not available for this task"
            if rollout.tool_calls >= self.config.max_tool_calls:
                return (
                    f"Error: the limit of {self.config.max_tool_calls} tool calls is reached; give your final answer"
                )
            rollout.tool_calls += 1
            if rollout.box is None:
                return "Error: the MCP environment is unavailable"
            served_name = rollout.box.tool_names.get(tool_name, tool_name)
            if tool_name == "met-museum_get-museum-object":
                # Upstream's harness does the same: image blocks do not fit a text tool output.
                body = {**body, "returnImage": False}
            try:
                status, payload = await call_sandbox_tool(
                    rollout.box, tool_name, body, timeout_s=self.config.tool_call_timeout_s
                )
            except asyncio.TimeoutError:
                # This is an HTTP/provider deadline, not evidence that model-written code timed out.
                rollout.failures.append(("sandbox_unreachable", f"{tool_name}: transport timeout"))
                return "Error: sandbox transport timed out"
            except (ClientError, OSError, SandboxError) as exc:
                rollout.failures.append(("sandbox_unreachable", f"{tool_name}: {type(exc).__name__}: {exc}"))
                return f"Error: tool call failed: {type(exc).__name__}"
            if status == 200:
                # The pinned bridge has a separate HTTP error path for MCP is_error results. Successful
                # content can be arbitrary file/program text, including strings that look like auth errors.
                return tool_output_text(payload)
            # The sandbox reports every tool error as an HTTP error; the model sees its body, as upstream.
            text = payload if isinstance(payload, str) else json.dumps(payload)
            detail = payload.get("detail") if isinstance(payload, dict) else None
            diagnostic = detail if isinstance(detail, str) else text
            if is_authentication_failure(
                status, diagnostic, servers=rollout.servers, tool_name=tool_name, served_name=served_name
            ):
                rollout.configuration_error = (
                    f"MCP-Atlas authentication failed for {tool_name}; check credentials/scopes. "
                    "The environment is unavailable; give your final answer."
                )
                logger.error("%s", rollout.configuration_error)
                raise HTTPException(422, rollout.configuration_error)
            if tool_server_name(tool_name, rollout.servers) not in MODEL_RESULT_SERVERS and _is_service_error(
                status, diagnostic, served_name
            ):
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

    async def _judge_claims(self, claims: list[str], answer: str) -> list[dict[str, Any]]:
        tasks = [asyncio.create_task(self._judge(claim, answer)) for claim in claims]
        try:
            return list(await asyncio.gather(*tasks))
        finally:
            # A failed or cancelled row no longer needs its other claim requests or retry attempts.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def verify(self, request: Request, body: MCPAtlasVerifyRequest) -> MCPAtlasVerifyResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        async with self.session_lock(session_id or ""):
            closed = session_id is not None and session_id in self._closed_sessions
            terminal_error = self._terminal_auth_sessions.get(session_id) if session_id is not None else None
            if terminal_error is not None:
                raise HTTPException(422, terminal_error)
            rollout = self._rollouts.get(session_id)
            if self._pool is not None and session_id is not None:
                try:
                    await self._pool.stop(session_id)
                except SandboxError as exc:
                    # The pool keeps the sandbox, and the reaper stops it again; the measurement is unaffected.
                    logger.warning("MCP-Atlas sandbox of session %s did not stop at verify: %s", session_id, exc)
            try:
                if rollout is not None and rollout.configuration_error:
                    raise HTTPException(422, rollout.configuration_error)
                claims = [c for c in body.verifier_metadata.get("claims", []) if isinstance(c, str) and c]
                answer = final_answer(body.response)
                if rollout is not None:
                    failures, tool_calls = list(rollout.failures), rollout.tool_calls
                elif closed:
                    # A closed live cookie cannot be revived by injected saved-artifact fields. Ordinary
                    # close follows the framework lost-session contract; terminal auth has its own fence.
                    failures, tool_calls = [("session_lost", "session already closed")], 0
                else:
                    failures, tool_calls = saved_environment(body) or ([("session_lost", "no session at verify")], 0)
                body.__pydantic_extra__ = {
                    **(body.model_extra or {}),
                    "tool_calls": tool_calls,
                    "service_errors": [f"{kind}: {detail}" for kind, detail in failures],
                }
                kinds = {kind for kind, _ in failures}
                sandbox_down = bool(kinds & {"sandbox_start_failed", "session_lost"})
                # Upstream scores an empty or "ERROR:" answer 0 without calling the judge.
                judged = claims and answer.strip() and not answer.startswith("ERROR:") and not sandbox_down
                verdicts = await self._judge_claims(claims, answer) if judged else []
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
                # Keep a terminal configuration rejection until close, so a lost HTTP reply cannot turn
                # a repeated verify into a masked session_lost result or trigger another judge request.
                if rollout is None or not rollout.configuration_error:
                    self._rollouts.pop(session_id, None)


if __name__ == "__main__":
    MCPAtlasResourcesServer.run_webserver()
