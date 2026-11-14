# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run LAB tools in a sandbox created and cleaned up by the Harbor Environment Server."""

from __future__ import annotations

import asyncio
import json
import shlex
from dataclasses import dataclass
from time import time
from uuid import uuid4

from aiohttp import ClientResponseError
from fastapi import Body, HTTPException, Request
from pydantic import Field

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SandboxCommand,
    SandboxSession,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import ModelServerRef, get_global_config_dict
from nemo_gym.core import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    accumulate_response_usage,
)
from nemo_gym.sandbox import AsyncSandbox, create_provider, resolve_provider_config, upload_text
from nemo_gym.trajectory import AgentObservationBundle, ObservationGap
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.legal_agent_bench.lab_harbor.completion import CompletionState
from resources_servers.legal_agent_bench.lab_harbor.tools import get_all_tool_definitions


_TOOL_RUNNER = "/opt/legal-agent-bench/container_tool_runner.py"
_TOOL_ENV = {
    "VDR_DIR": "/workspace/vdr",
    "DOCUMENTS_DIR": "/workspace/vdr",
    "OUTPUT_DIR": "/workspace/output",
    "WORKSPACE_DIR": "/workspace",
    "SKILLS_DIR": "/workspace/skills",
}


class _AgentDeadlineExceeded(TimeoutError):
    """The agent budget expired and tool cleanup was confirmed."""


class LegalAgentBenchNativeAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    max_turns: int = Field(default=60, ge=1)
    agent_timeout_seconds: float = Field(default=10800, gt=0, allow_inf_nan=False)
    shell_timeout: float = Field(default=60, gt=0)
    model_timeout_seconds: float = Field(default=1800, gt=0)
    max_output_chars: int = Field(default=16384, ge=1)
    max_output_tokens: int = Field(default=65536, ge=1)


@dataclass
class LabNativeSession(AgentSessionState):
    session: SandboxSession[str]
    completion: CompletionState
    task: asyncio.Task[NeMoGymResponse] | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    tool_index: int = 0
    deadline: float | None = None


class LegalAgentBenchNativeAgent(SimpleResponsesAPIAgent):
    """Run LAB tools; the Harbor Environment Server owns the sandbox and handles grading."""

    config: LegalAgentBenchNativeAgentConfig

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> LabNativeSession:
        if body.continuation is not None:
            raise AgentSeedRejectedError("Native LAB supports one activation per episode")
        if body.sandbox_access is None:
            raise AgentSeedRejectedError("Native LAB requires a task sandbox owned by the Harbor Environment Server")
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise AgentSeedRejectedError(
                "Native LAB uses its own workspace tools; required tool grants are unsupported"
            )
        deadline = asyncio.get_running_loop().time() + min(
            self.config.agent_timeout_seconds, body.agent_timeout_sec or self.config.agent_timeout_seconds
        )
        access = body.sandbox_access
        provider_config = resolve_provider_config(access.connection.provider_config_ref, get_global_config_dict())
        provider = create_provider(provider_config)
        try:
            sandbox = await AsyncSandbox.connect(access.connection.descriptor, provider=provider)
        except BaseException:
            await provider.aclose()
            raise
        return LabNativeSession(
            request=body,
            session=SandboxSession(
                sandbox=sandbox,
                session_dir=f"/tmp/nemo-gym-lab-{uuid4().hex}",
                workdir=access.workdir,
                harness="legal_agent_bench_native",
                # Closing the agent stops its processes and disconnects; the environment server stops the sandbox.
                owns_sandbox=False,
            ),
            completion=CompletionState(),
            deadline=deadline,
        )

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, LabNativeSession):
            raise TypeError("Expected LAB native session")
        if state.task is not None and not state.task.done():
            state.task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(state.task), timeout=self.config.session_close_timeout_seconds)
            except asyncio.CancelledError:
                if not state.task.cancelled():
                    raise
        await state.session.close(timeout=self.config.session_close_timeout_seconds)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            cleanup_confirmed=state.session.cleanup_confirmed,
            agent_observations=AgentObservationBundle(
                source="legal_agent_bench_native_agent",
                gaps=[
                    ObservationGap(
                        code="observation_capture_failed",
                        detail="Evaluation-level adapter retains Responses/tool history; normalized invocation evidence is not emitted.",
                    )
                ],
            ),
        )

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(409, "Seed a LAB agent session before activation")
        state = self._require_agent_session(session_id)
        if not isinstance(state, LabNativeSession):
            raise TypeError("Expected LAB native session")
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise HTTPException(409, "Activation does not match the seeded rollout")
        if state.session.closing:
            raise HTTPException(409, "LAB agent session is closing")
        if state.activation_request is not None and state.activation_request != body:
            raise HTTPException(409, "Native LAB accepts only one activation per episode")
        if state.task is None:
            state.activation_request = body.model_copy(deep=True)
            model_path = self.url_path_for_request("/v1/responses", request)
            state.task = asyncio.create_task(self._run_loop(state, body, model_path, dict(request.cookies)))
        return await asyncio.shield(state.task)

    async def _run_loop(
        self,
        state: LabNativeSession,
        body: NeMoGymResponseCreateParamsNonStreaming,
        model_path: str,
        cookies: dict[str, str],
    ) -> NeMoGymResponse:
        messages = list(body.input) if isinstance(body.input, list) else [{"role": "user", "content": body.input}]
        messages = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            {
                "input": [
                    {
                        "role": "developer",
                        "content": "Use the LAB workspace tools. When complete, call finish with a summary and the deliverable paths under /workspace/output. Missing files are rejected twice, then accepted as incomplete evidence.",
                    },
                    *messages,
                ]
            }
        ).input
        tools = NeMoGymResponseCreateParamsNonStreaming(
            input=[], tools=[{"type": "function", "strict": False, **tool} for tool in get_all_tool_definitions()]
        ).tools
        output = []
        usage = None
        usage_missing = False
        finish_reason = "max_turns_exceeded"
        deadline = state.deadline or (asyncio.get_running_loop().time() + self.config.agent_timeout_seconds)
        model_response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=time(),
            model=body.model or self.config.model_server.name,
            object="response",
            output=[],
            status="incomplete",
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=body.parallel_tool_calls if body.parallel_tool_calls is not None else True,
        )
        for _turn in range(self.config.max_turns):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                finish_reason = "agent_timeout"
                break
            params = body.model_copy(
                update={
                    "input": messages,
                    "tools": tools,
                    "max_output_tokens": body.max_output_tokens
                    if body.max_output_tokens is not None
                    else self.config.max_output_tokens,
                }
            )
            try:
                async with asyncio.timeout(min(self.config.model_timeout_seconds, remaining)):
                    reply = await self.server_client.post(
                        server_name=self.config.model_server.name,
                        url_path=model_path,
                        json=params.model_dump(mode="json", exclude_unset=True),
                        cookies=cookies,
                    )
                    async with reply:
                        await raise_for_status(reply)
                        model_response = NeMoGymResponse.model_validate(await get_response_json(reply))
                        cookies.update({key: value.value for key, value in reply.cookies.items()})
            except TimeoutError:
                if asyncio.get_running_loop().time() < deadline:
                    raise
                usage_missing = True
                finish_reason = "agent_timeout"
                break
            except ClientResponseError as exc:
                content = getattr(exc, "response_content", b"")
                try:
                    payload = json.loads(content)
                except (ValueError, TypeError):
                    raise exc
                if (
                    exc.status not in (400, 413, 422)
                    or (payload.get("error") or {}).get("code") != "context_length_exceeded"
                ):
                    raise
                model_response = NeMoGymResponse(
                    id=f"resp_{uuid4().hex}",
                    created_at=time(),
                    model=body.model or self.config.model_server.name,
                    object="response",
                    output=[],
                    status="incomplete",
                    tools=tools,
                    tool_choice="auto",
                    parallel_tool_calls=body.parallel_tool_calls if body.parallel_tool_calls is not None else True,
                    metadata={"lab_context_limit": "true"},
                )
            output.extend(model_response.output)
            messages.extend(model_response.output)
            if model_response.usage is None:
                usage_missing = True
            else:
                previous = usage
                usage = accumulate_response_usage(usage, model_response.usage)
                if previous is not None:
                    writes = (
                        getattr(previous.input_tokens_details, "cache_write_tokens", None),
                        getattr(model_response.usage.input_tokens_details, "cache_write_tokens", None),
                    )
                    if any(value is not None for value in writes):
                        usage.input_tokens_details = usage.input_tokens_details.model_copy(
                            update={
                                "cache_write_tokens": sum(writes)
                                if all(type(value) is int for value in writes)
                                else None
                            }
                        )
            if model_response.status in {"failed", "incomplete"}:
                finish_reason = "model_incomplete" if model_response.status == "incomplete" else "model_error"
                if model_response.error is not None and model_response.error.code == "context_length_exceeded":
                    finish_reason = "model_incomplete"
                break
            calls = [item for item in model_response.output if item.type == "function_call"]
            if not calls:
                finish_reason = "no_tool_calls"
                break
            for call in calls:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    finish_reason = "agent_timeout"
                    break
                if state.completion.finished:
                    tool_output = "Task already finished; no further tools were executed."
                else:
                    try:
                        tool_output = await self._execute_tool(state, call.name, call.arguments, deadline=deadline)
                    except _AgentDeadlineExceeded:
                        tool_output = "Error: agent time limit reached; tool cleanup completed."
                        finish_reason = "agent_timeout"
                item = NeMoGymFunctionCallOutput(type="function_call_output", call_id=call.call_id, output=tool_output)
                messages.append(item)
                output.append(item)
                if finish_reason == "agent_timeout":
                    break
            if finish_reason == "agent_timeout" or asyncio.get_running_loop().time() >= deadline:
                finish_reason = "agent_timeout"
                break
            if state.completion.finished:
                finish_reason = "finish_tool"
                break
        clean = finish_reason in {"finish_tool", "no_tool_calls"}
        metadata = dict(model_response.metadata or {})
        metadata["lab_finish_reason"] = finish_reason
        if finish_reason == "model_error" and "ng_agent_outcome" not in metadata:
            metadata["ng_agent_outcome"] = '{"failure_kind":"unknown"}'
        if state.completion.summary:
            metadata["lab_finish_summary"] = state.completion.summary
        response = model_response.model_copy(
            update={
                "output": output,
                "usage": None if usage_missing else usage,
                "metadata": metadata,
                "status": "completed" if clean else "failed" if finish_reason == "model_error" else "incomplete",
            }
        )
        return response

    async def _execute_tool(self, state: LabNativeSession, name: str, arguments: str, *, deadline: float) -> str:
        try:
            parsed = json.loads(arguments)
        except (ValueError, TypeError) as exc:
            return f"Error: invalid JSON tool arguments: {exc}"
        if not isinstance(parsed, dict):
            return "Error: tool arguments must be a JSON object"
        names = {tool["name"] for tool in get_all_tool_definitions()}
        if name not in names:
            return f"Error: unknown tool {name}"
        if name == "bash" and (not isinstance(parsed.get("command"), str) or not parsed["command"]):
            return "Error: command is required"
        state.tool_index += 1
        root = f"/tmp/nemo-gym-lab-tool-{uuid4().hex}"

        async def stage() -> SandboxCommand:
            result = await state.session.sandbox.exec(f"mkdir -p {shlex.quote(root)}", timeout_s=60)
            if result.return_code != 0 or result.error_type is not None:
                raise RuntimeError("Could not stage LAB tool activation")
            payload = str(parsed["command"]) if name == "bash" else json.dumps(parsed)
            source = root + ("/command.sh" if name == "bash" else "/arguments.json")
            await upload_text(state.session.sandbox, path=source, text=payload)
            command = (
                f"exec /bin/bash -l {shlex.quote(source)}"
                if name == "bash"
                else (f"exec python3 {shlex.quote(_TOOL_RUNNER)} {shlex.quote(name)} < {shlex.quote(source)}")
            )
            return SandboxCommand(
                argv=["env", *(f"{key}={value}" for key, value in _TOOL_ENV.items()), "/bin/bash", "-c", command]
            )

        try:
            async with asyncio.timeout_at(deadline):
                await state.session.begin_activation(
                    session_dir=root, timeout=self.config.session_close_timeout_seconds
                )
            text = await state.session.execute(
                stage_activation=stage,
                collect=state.session.read_output_log,
                timeout=self.config.shell_timeout,
                deadline=deadline,
                close_timeout=self.config.session_close_timeout_seconds,
            )
        except Exception as error:
            # The shared supervisor cannot launch with less than one centisecond left.
            if isinstance(error, TimeoutError) and deadline - asyncio.get_running_loop().time() < 0.01:
                # The agent budget excludes bounded cleanup. Joining close also fences
                # staging that was still unwinding when the execution deadline expired.
                try:
                    await state.session.close(timeout=self.config.session_close_timeout_seconds)
                except Exception as cleanup_error:
                    raise RuntimeError("LAB tool deadline expired but session cleanup failed") from cleanup_error
                if (
                    not state.session.cleanup_confirmed
                    or (state.session.cleanup or {}).get("error")
                    or state.session.capture_error is not None
                ):
                    raise RuntimeError(
                        "LAB tool deadline expired without clean termination and artifact capture"
                    ) from error
                raise _AgentDeadlineExceeded("LAB agent time limit reached") from error
            receipt = state.session.cleanup
            if (
                not state.session.cleanup_confirmed
                or receipt is None
                or receipt.get("error")
                or state.session.capture_error
                or not isinstance(error, TimeoutError)
                or not receipt.get("timed_out")
            ):
                raise
            # Only a confirmed tool timeout is an ordinary tool observation.
            # Cleanup success does not turn provider failures into task outcomes.
            text = state.session.artifacts or ""
            text = f"Error: tool exited with {receipt.get('return_code')}; {text}"
        if name != "bash":
            try:
                data = json.loads(next(line for line in reversed(text.splitlines()) if line.strip()))
            except (ValueError, StopIteration):
                return f"Error: invalid container tool response: {text[: self.config.max_output_chars]}"
            if name == "finish":
                if "missing_deliverables" not in (data.get("metrics") or {}):
                    return f"Error: completion check failed: {data.get('result')}"
                return state.completion.finish(
                    summary=str(parsed.get("summary", "")), missing=data["metrics"]["missing_deliverables"]
                )
            text = str(data.get("result", ""))
        full_read = name == "read" and "limit" in parsed and parsed["limit"] in (None, 0)
        return text if full_read else text[: self.config.max_output_chars]


if __name__ == "__main__":
    LegalAgentBenchNativeAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = LegalAgentBenchNativeAgent.run_webserver()  # noqa: F401
