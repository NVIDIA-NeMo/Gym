# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import shlex
import shutil
import signal
import sys
from asyncio import Semaphore
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from time import time
from typing import Any, Mapping, Optional
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field, JsonValue

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentOutcome,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import ModelServerRef, ResourcesServerRef, get_global_config_dict
from nemo_gym.core import (
    MODEL_REQUEST_ERROR,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.sandbox import (
    AsyncSandbox,
    ConnectableProvider,
    DirectSandboxConnection,
    SandboxSpec,
    create_provider,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse
from nemo_gym.trajectory import AgentInvocation, AgentObservationBundle, ObservationGap, SandboxObservation
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


LOG = logging.getLogger(__name__)


class SSARunError(RuntimeError):
    """The runner's own error result, with the HTTP status of the model request that ended the run, if any."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _saved_messages(path: Path) -> list[dict[str, Any]]:
    """The transcript SSA's TrajectoryHook saves every few messages, or none if it saved none."""
    try:
        messages = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return messages if isinstance(messages, list) else []


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_content_text(item) for item in content)
    if not isinstance(content, dict):
        return str(content)
    if "text" in content:
        return str(content["text"] or "")
    if "json" in content:
        return json.dumps(content["json"], ensure_ascii=False)
    return ""


def _extract_instruction(body_input) -> tuple[str, Optional[str]]:
    conversation = []
    system_messages = []
    for item in body_input:
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
        text = _content_text(content)
        if role in {"system", "developer"}:
            system_messages.append(text)
        elif role in {"user", "assistant"} and text:
            conversation.append((role, text))
    instruction = (
        conversation[0][1]
        if len(conversation) == 1
        else "\n\n".join(f"{role.title()}: {text}" for role, text in conversation)
    )
    return instruction, "\n\n".join(system_messages) or None


def _tool_output(block: dict[str, Any]) -> str:
    content = block.get("content") or []
    return "\n".join(text for item in content if (text := _content_text(item)))


def trajectory_to_output_items(messages: list[dict[str, Any]]) -> list[Any]:
    output_items: list[Any] = []
    item_index = 0
    for message in messages:
        role = message.get("role")
        content = message.get("content") or []
        if role == "assistant":
            for block in content:
                if "reasoningContent" in block:
                    reasoning = block["reasoningContent"].get("reasoningText") or {}
                    text = reasoning.get("text") or ""
                    if text:
                        output_items.append(
                            NeMoGymResponseReasoningItem(
                                id=f"reasoning-{item_index}",
                                summary=[NeMoGymSummary(text=text, type="summary_text")],
                                encrypted_content=reasoning.get("signature"),
                            )
                        )
                        item_index += 1
                elif "text" in block and block["text"]:
                    output_items.append(
                        NeMoGymResponseOutputMessage(
                            id=f"message-{item_index}",
                            content=[NeMoGymResponseOutputText(text=block["text"], annotations=[])],
                        )
                    )
                    item_index += 1
                elif "toolUse" in block:
                    tool = block["toolUse"]
                    call_id = str(tool.get("toolUseId") or f"call-{item_index}")
                    output_items.append(
                        NeMoGymResponseFunctionToolCall(
                            id=call_id,
                            call_id=call_id,
                            name=str(tool.get("name") or ""),
                            arguments=json.dumps(tool.get("input") or {}, ensure_ascii=False),
                            status="completed",
                        )
                    )
                    item_index += 1
        elif role == "user":
            for block in content:
                tool_result = block.get("toolResult")
                if not tool_result:
                    continue
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        call_id=str(tool_result.get("toolUseId") or ""),
                        output=_tool_output(tool_result),
                        status="completed" if tool_result.get("status") != "error" else "incomplete",
                    )
                )
    return output_items


class SimpleStrandsAgentConfig(BaseResponsesAPIAgentConfig):
    # Set only for the host /run path; agent sessions get the benchmark from the Environment Server.
    resources_server: Optional[ResourcesServerRef] = None
    model_server: ModelServerRef
    model: Optional[str] = None
    concurrency: int = 8
    max_turns: int = 100
    max_output_tokens: int = 131072
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    timeout: int = 1800
    model_timeout: int = 600
    shell_timeout: int = 120
    conversation_window: int = 300
    tools: list[str] = Field(default_factory=lambda: ["bash", "str_replace_editor"])
    prompt_tag: str = "swe_generic_v2"
    native_user_prompt: bool = False
    system_prompt: Optional[str] = None
    workspace_root: str = "outputs/simple_strands_agent/workspaces"
    keep_workspaces: bool = False
    # Agent sessions: SSA's tools run in the task sandbox the Resources Server supplies (or an agent-owned one).
    session_close_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    sandbox_provider: Optional[str] = None
    sandbox_config: dict[str, JsonValue] = Field(default_factory=dict)


@dataclass
class SimpleStrandsSession(AgentSessionState):
    """An episode's task sandbox, which the SSA runner connects to for every tool call."""

    sandbox: AsyncSandbox
    owns_sandbox: bool
    provider_name: str
    provider_config: dict[str, Any]
    workdir: str
    task: Optional[asyncio.Task] = None
    activation_request: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    observations: Optional[AgentObservationBundle] = None
    released: bool = False
    # The runner was killed or crashed, so a tool command may still run in the sandbox.
    interrupted: bool = False


class SimpleStrandsAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SimpleStrandsAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class SimpleStrandsAgent(SimpleResponsesAPIAgent):
    config: SimpleStrandsAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def _workspace(self) -> Path:
        root = Path(self.config.workspace_root).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        work_dir = root / f"ssa-{uuid4().hex[:8]}"
        work_dir.mkdir(parents=True)
        return work_dir

    async def _run_ssa(
        self, payload: dict[str, Any], host_dir: Path, sandbox: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        request_path = host_dir / "request.json"
        result_path = host_dir / "result.json"
        request_path.write_text(json.dumps(payload))
        runner = Path(__file__).with_name("ssa_runner.py")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(runner),
            str(request_path),
            str(result_path),
            *(["--sandbox-stdin"] if sandbox is not None else []),
            stdin=asyncio.subprocess.PIPE if sandbox is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stdin = json.dumps(sandbox).encode() if sandbox is not None else None
        communication = asyncio.create_task(process.communicate(stdin))
        try:
            stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), timeout=self.config.timeout)
        except asyncio.TimeoutError:
            await self._terminate_process(process, communication)
            raise TimeoutError(f"SSA timed out after {self.config.timeout}s") from None
        except asyncio.CancelledError:
            await self._terminate_process(process, communication)
            raise

        # A runner that fails writes its error result first, so the result says why.
        if process.returncode != 0 and not result_path.is_file():
            detail = stderr.decode(errors="replace")[-1000:]
            if stdout:
                LOG.debug("SSA stdout: %s", stdout.decode(errors="replace")[-1000:])
            raise RuntimeError(f"SSA failed: {detail}")
        if not result_path.is_file():
            raise RuntimeError("SSA did not produce a result")
        try:
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"SSA produced an invalid result: {error}") from error
        if not isinstance(result, dict):
            raise RuntimeError("SSA produced an invalid result: expected a JSON object")
        if result.get("error"):
            if stdout:
                LOG.debug("SSA stdout: %s", stdout.decode(errors="replace")[-1000:])
            raise SSARunError(f"SSA failed: {result['error']}", result.get("status_code"))
        return result

    @staticmethod
    async def _terminate_process(process, communication: asyncio.Task) -> None:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(asyncio.shield(communication), timeout=10)
        except asyncio.TimeoutError:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await communication

    def _payload(
        self, body: NeMoGymResponseCreateParamsNonStreaming, rollout_id: Optional[str], work_dir: str, output_dir: Path
    ) -> dict[str, Any]:
        """The runner request; work_dir is where SSA's tools run (a host workspace, or the sandbox's task workdir)."""
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]
        instruction, input_system = _extract_instruction(body.input)
        system_parts = [self.config.system_prompt, body.instructions, input_system]
        system_prompt = "\n\n".join(part for part in system_parts if part) or None
        model_name = self.config.model or str(body.model or self.config.model_server.name)
        reasoning_effort = (body.reasoning or {}).get("effort") or self.config.reasoning_effort
        return {
            "instruction": instruction,
            "system_prompt": system_prompt,
            "model": model_name,
            "model_base_url": self.resolve_model_base_url(self.config.model_server.name, rollout_id),
            "work_dir": work_dir,
            "output_dir": str(output_dir),
            "rollout_id": rollout_id or uuid4().hex,
            "max_turns": self.config.max_turns,
            "max_output_tokens": body.max_output_tokens or self.config.max_output_tokens,
            "temperature": body.temperature if body.temperature is not None else self.config.temperature,
            "reasoning_effort": reasoning_effort,
            "model_timeout": self.config.model_timeout,
            "shell_timeout": self.config.shell_timeout,
            "conversation_window": self.config.conversation_window,
            "tools": self.config.tools,
            "prompt_tag": self.config.prompt_tag,
            "native_user_prompt": self.config.native_user_prompt,
        }

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is not None:
            return await self._session_response(request, session_id, body)
        rollout_id = request.path_params.get("rollout_id") if request is not None else None
        work_dir = self._workspace()
        payload = self._payload(body, rollout_id, str(work_dir), work_dir / "output")
        try:
            result = await self._run_ssa(payload, work_dir)
        finally:
            if not self.config.keep_workspaces:
                shutil.rmtree(work_dir, ignore_errors=True)
        return self._response(body, result, payload["model"])

    def _response(
        self, body: NeMoGymResponseCreateParamsNonStreaming, result: dict[str, Any], model_name: str
    ) -> NeMoGymResponse:
        output_items = trajectory_to_output_items(result.get("messages") or [])
        if not any(getattr(item, "type", None) == "message" for item in output_items):
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"message-{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                )
            )
        usage = result.get("usage") or {}
        input_tokens = int(usage.get("inputTokens") or 0)
        output_tokens = int(usage.get("outputTokens") or 0)
        # Strands omits cacheReadInputTokens when the model reported no cache reads; that stays unknown.
        cache_read = usage.get("cacheReadInputTokens")
        cached_tokens = cache_read if type(cache_read) is int and cache_read >= 0 else None
        reasoning_tokens = int(usage.get("reasoningTokens") or 0)
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model_name,
            object="response",
            output=output_items,
            # SSA's iteration limit and the deadline end the run without an answer; the work so far is still graded.
            status="incomplete" if result.get("stop_reason") in ("max_iterations", "deadline") else "completed",
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached_tokens),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=reasoning_tokens),
                total_tokens=input_tokens + output_tokens,
            ),
        )

    def classify_outcome(self, native_result: Mapping[str, Any]) -> AgentOutcome:
        """A failed model request's HTTP status, which the runner reads from the exception; anything else unknown."""
        status_code = native_result.get("status_code")
        if isinstance(status_code, int) and 400 <= status_code <= 599:
            return AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=status_code)
        return AgentOutcome(failure_kind="unknown")

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> SimpleStrandsSession:
        """Connect to (or start) the task sandbox where SSA's tools will run; nothing is installed in it."""
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise AgentSeedRejectedError(
                "Simple Strands Agent uses its own tools; required tool grants are unsupported"
            )
        access = body.sandbox_access
        owns_sandbox = access is None
        if owns_sandbox:
            if not self.config.sandbox_provider:
                raise AgentSeedRejectedError(
                    "Simple Strands Agent requires sandbox_access or a configured sandbox_provider"
                )
            provider_ref = self.config.sandbox_provider
            workdir = self.config.sandbox_config.get("workdir") or "/app"
        else:
            if not isinstance(access.connection, DirectSandboxConnection):
                raise AgentSeedRejectedError("Simple Strands Agent requires direct SandboxAccess")
            if access.user is not None:
                raise AgentSeedRejectedError("Simple Strands Agent does not support a SandboxAccess user override")
            provider_ref = access.connection.provider_config_ref
            workdir = access.workdir
        path = PurePosixPath(workdir) if isinstance(workdir, str) else None
        if path is None or not path.is_absolute() or ".." in path.parts or path == PurePosixPath("/"):
            raise AgentSeedRejectedError("Simple Strands Agent task workdir must be an absolute path")
        global_config = get_global_config_dict()
        provider_config = resolve_provider_config(provider_ref, global_config)
        provider = create_provider(provider_config)
        if owns_sandbox:
            # The runner reconnects to the sandbox from its own process, which needs a connectable provider.
            if not isinstance(provider, ConnectableProvider):
                await provider.aclose()
                raise AgentSeedRejectedError(
                    f"Simple Strands Agent needs a connectable sandbox provider, not {provider.name}"
                )
            sandbox = AsyncSandbox(provider)
        else:
            try:
                sandbox = await AsyncSandbox.connect(access.connection.descriptor, provider=provider)
            except BaseException:
                await provider.aclose()
                raise
        state = SimpleStrandsSession(
            request=body,
            sandbox=sandbox,
            owns_sandbox=owns_sandbox,
            provider_name=provider.name,
            provider_config=provider_config,
            workdir=str(path),
        )
        try:
            if owns_sandbox:
                settings = dict(self.config.sandbox_config)
                metadata = settings.pop("metadata", None) or {}
                settings["metadata"] = {
                    **resolve_provider_metadata(provider_ref, global_config),
                    **metadata,
                    "agent_session_id": body.agent_session_id[:63],
                }
                settings["workdir"] = str(path)
                settings.setdefault("ttl_s", int(self.config.timeout + 600))
                await sandbox.start(SandboxSpec(**settings))
            # Fail at seed, not on the first tool call, when the workdir is missing or bash is not in the image.
            check = await sandbox.exec(
                f"bash -c {shlex.quote(f'test -d {shlex.quote(str(path))}')}", cwd="/", timeout_s=60
            )
            if check.return_code != 0 or check.error_type:
                raise RuntimeError(
                    f"The task sandbox has no bash or no workdir {path}: {check.stderr or check.stdout}"
                )
        except BaseException as error:
            try:
                await self._release(state)
            except BaseException:
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    async def _release(self, state: SimpleStrandsSession) -> None:
        """Stop the runner, then disconnect from a borrowed sandbox or stop an owned one."""
        if state.task is not None and not state.task.done():
            state.task.cancel()
        if state.task is not None:
            await asyncio.gather(state.task, return_exceptions=True)
        if not state.released:
            async with asyncio.timeout(self.config.session_close_timeout_seconds):
                await (state.sandbox.stop() if state.owns_sandbox else state.sandbox.disconnect())
            state.released = True

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, SimpleStrandsSession):
            raise TypeError("Expected SimpleStrandsSession")
        task = state.task
        # A runner stopped mid-run (cancelled, timed out or failed) can leave a tool command running in a borrowed
        # sandbox, and disconnect does not stop it. Only a runner that returned ran every command to its exit.
        interrupted = state.interrupted or (
            task is not None and (not task.done() or task.cancelled() or task.exception() is not None)
        )
        await self._release(state)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            cleanup_confirmed=state.owns_sandbox or not interrupted,
            agent_observations=state.observations
            or AgentObservationBundle(
                source="simple_strands", gaps=[ObservationGap(code="observation_capture_failed")]
            ),
        )

    def _session_observations(
        self,
        state: SimpleStrandsSession,
        *,
        status: str,
        error_type: Optional[str],
        output: list[Any],
        stop_reason: Optional[str] = None,
    ) -> AgentObservationBundle:
        conversation = []
        if state.activation_request is not None:
            instruction, _ = _extract_instruction(
                [NeMoGymEasyInputMessage(role="user", content=state.activation_request.input)]
                if isinstance(state.activation_request.input, str)
                else state.activation_request.input
            )
            conversation.append(NeMoGymEasyInputMessage(role="user", content=instruction))
        gaps = ["model_call_ownership_unavailable", "harness_tool_timing_unavailable"]
        gaps += ["optional_tool_grant_skipped" for _ in self.effective_tool_accesses(state.request)]
        return AgentObservationBundle(
            source="simple_strands",
            records=[
                AgentInvocation(
                    invocation_id=state.request.episode_id.capture_key,
                    status=status,
                    stop_reason=stop_reason,
                    error_type=error_type,
                    conversation=conversation + output,
                ),
                SandboxObservation(
                    role="agent",
                    provider=state.provider_name,
                    sandbox_id=state.sandbox.sandbox_id,
                    outcome="failed"
                    if status == "failed"
                    else "timeout"
                    if stop_reason == "deadline"
                    else "completed",
                    error_type=error_type,
                ),
            ],
            gaps=[ObservationGap(code=code) for code in sorted(set(gaps))],
        )

    async def _activate_session(
        self, state: SimpleStrandsSession, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        work_dir, failure = None, None
        try:
            work_dir = self._workspace()
            rollout_id = state.request.episode_id.capture_key
            payload = self._payload(body, rollout_id, state.workdir, work_dir / "output")
            access = state.request.sandbox_access
            sandbox = {
                "provider": state.provider_config,
                "descriptor": await state.sandbox.serialize(),
                "env": access.env if access else {},
            }
            try:
                async with self.sem:
                    result = await self._run_ssa(payload, work_dir, sandbox)
            except TimeoutError:
                state.interrupted = True
                # The deadline is graded like a timeout, on the transcript SSA saved so far.
                result = {
                    "messages": _saved_messages(work_dir / "output" / "trajectory.json"),
                    "stop_reason": "deadline",
                }
            except Exception as error:
                # A model HTTP error, a sandbox outage or a runner crash: a classified failure, not a bare 500.
                result, failure = {}, error
                state.interrupted = not isinstance(error, SSARunError)
        except BaseException as error:
            state.observations = self._session_observations(
                state, status="failed", error_type=type(error).__name__, output=[]
            )
            raise
        finally:
            if work_dir is not None and not self.config.keep_workspaces:
                shutil.rmtree(work_dir, ignore_errors=True)
        response = self._response(body, result, payload["model"])
        if failure is not None:
            LOG.warning("Simple Strands run failed: %s", failure)
            response = response.model_copy(
                update={"status": "failed", "error": {"code": "server_error", "message": str(failure)[-2000:]}}
            )
            state.observations = self._session_observations(
                state, status="failed", error_type=type(failure).__name__, output=[]
            )
            return self.report_outcome(response, {"status_code": getattr(failure, "status_code", None)})
        state.observations = self._session_observations(
            state,
            status=response.status,
            error_type=None,
            output=list(response.output),
            stop_reason=result.get("stop_reason"),
        )
        return response

    async def _session_response(
        self, request: Request, session_id: str, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        state = self._require_agent_session(session_id)
        if not isinstance(state, SimpleStrandsSession):
            raise TypeError("Expected SimpleStrandsSession")
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise HTTPException(409, "Simple Strands activation must use its seeded rollout route")
        if state.task is None:
            state.activation_request = body.model_copy(deep=True)
            state.task = asyncio.create_task(self._activate_session(state, state.activation_request))
        elif body != state.activation_request:
            raise HTTPException(409, "Simple Strands sessions support one activation; retries must repeat the request")
        return (await asyncio.shield(state.task)).model_copy(deep=True)

    async def run(self, request: Request, body: SimpleStrandsAgentRunRequest) -> SimpleStrandsAgentVerifyResponse:
        if self.config.resources_server is None:
            raise HTTPException(
                422, "Simple Strands /run needs a resources_server; agent sessions use an Environment Server"
            )
        async with self.sem:
            cookies = request.cookies
            seed_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_response)
            cookies = seed_response.cookies
            agent_response = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_response)
            cookies = agent_response.cookies
            response_json = await get_response_json(agent_response)
            if self.config.skip_verification:
                result = body.model_dump() | {
                    "response": response_json,
                    "reward": float(self.config.skip_verification_reward),
                    "verification_skipped": True,
                }
            else:
                verify_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/verify",
                    json=body.model_dump() | {"response": response_json},
                    cookies=cookies,
                )
                await raise_for_status(verify_response)
                result = await get_response_json(verify_response)
            response = NeMoGymResponse.model_validate(response_json)
            turns = sum(1 for item in response.output if getattr(item, "type", None) == "message")
            last = response.output[-1] if response.output else None
            return SimpleStrandsAgentVerifyResponse.model_validate(
                result
                | {
                    "turns_used": turns,
                    "finished_naturally": response.status == "completed" and getattr(last, "type", None) == "message",
                }
            )


if __name__ == "__main__":
    SimpleStrandsAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = SimpleStrandsAgent.run_webserver()  # noqa: F401
