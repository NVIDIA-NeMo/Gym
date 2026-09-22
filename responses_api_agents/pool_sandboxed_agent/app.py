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

import json
import re
import sys
import tempfile
from pathlib import Path
from shlex import quote
from time import time
from traceback import format_exc
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ObservationGap,
    SandboxObservation,
)
from nemo_gym.sandbox import AsyncSandbox, create_provider
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    get_response_json,
    get_server_url,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
)


POOL_INSTALL_URL = "https://downloads.poolside.ai/pool/install.sh"
_FINISHED_RE = re.compile(r"pool run finished rc=(\d+)")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class PoolSandboxedAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    pool_version: str = "latest"
    remote_pool_binary_path: Optional[str] = None
    pool_model: str = "dummy_model"
    pool_max_context_window: int
    pool_extra_args: List[str] = Field(default_factory=list)
    pool_env: Dict[str, str] = Field(default_factory=dict)
    pool_agent_config: Dict[str, Any] = Field(default_factory=dict)

    sandbox_model_base_url: Optional[str] = None
    sandbox_provider: str
    sandbox_config: Dict[str, Any]
    sandbox_timeout: float

    debug: bool = False


class PoolSandboxedAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class PoolSandboxedAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class PoolSandboxedAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    pool_events_fpath: str
    pool_run_stdout: str
    pool_run_stderr: str
    pool_finished: bool
    pool_exit_code: Optional[int]
    pool_events_found: bool
    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


def parse_pool_events(events_text: str) -> tuple[List[NeMoGymResponseOutputItem], Dict[str, Any]]:
    output_items: List[NeMoGymResponseOutputItem] = []
    # toolCallResult events carry no id; pool reports results in call order.
    unanswered_call_ids: List[str] = []
    errors: List[str] = []

    def emit_tool_output(output: str) -> None:
        call_id = unanswered_call_ids.pop(0) if unanswered_call_ids else f"call-{uuid4().hex[:8]}"
        output_items.append(
            NeMoGymFunctionCallOutput(type="function_call_output", call_id=call_id, output=output, status="completed")
        )

    for line in events_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        match event:
            case {"type": "reasoning", "reasoning": str(think)} if think.strip():
                output_items.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rs_{uuid4().hex}",
                        summary=[NeMoGymSummary(text=think, type="summary_text")],
                        status="completed",
                    )
                )
            case {"type": "reasoning"} | {"type": "thought"}:
                # thought repeats the preceding reasoning event.
                pass
            case {"type": "assistantMessage", "message": str(text)} if text.strip():
                output_items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg-{len(output_items)}",
                        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )
            case {"type": "assistantMessage"}:
                pass
            case {"type": "toolCall"}:
                call_id = f"call-{uuid4().hex[:8]}"
                unanswered_call_ids.append(call_id)
                output_items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=json.dumps(event.get("args") or {}),
                        call_id=call_id,
                        name=str(event.get("name") or "tool"),
                        type="function_call",
                        id=call_id,
                        status="completed",
                    )
                )
            case {"type": "toolCallResult", "err": err}:
                emit_tool_output(f"[error] {err}")
            case {"type": "toolCallResult", "entries": list(entries)}:
                emit_tool_output("\n".join(entries))
            case {"type": "toolCallResult"}:
                emit_tool_output(str(event.get("result") or ""))
            case {"type": "error"} | {"error": _}:
                errors.append(str(event.get("error") or "unknown error"))
            case _:
                raise NotImplementedError(event)

    while unanswered_call_ids:
        emit_tool_output("")
    return output_items, ({"errors": errors} if errors else {})


class PoolSandboxedAgent(SimpleResponsesAPIAgent):
    config: PoolSandboxedAgentConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._sandbox_id_to_sandbox: Dict[str, AsyncSandbox] = dict()
        self._sandbox_id_to_run_result: Dict[str, Dict[str, Any]] = dict()

    async def _connect_sandbox(self, sandbox_id: Optional[str]) -> AsyncSandbox:
        if not sandbox_id:
            raise ValueError("pool_sandboxed_agent requires a sandbox_handle from the resources server")
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, get_global_config_dict()))
        return await AsyncSandbox.connect({"sandbox_id": sandbox_id}, provider=provider)

    async def _model_base_url(self, request: Request) -> str:
        base_url = self.config.sandbox_model_base_url or get_server_url(self.config.model_server.name)
        return self.base_url_for_run(base_url=base_url, body=await request.json()) + "/v1"

    def _install_command(self, home: str) -> str:
        if self.config.remote_pool_binary_path:
            return (
                f"mkdir -p {home}/bin && install -m 0755 {quote(self.config.remote_pool_binary_path)} {home}/bin/pool"
            )
        return (
            f'installer=$(mktemp) && curl -fsSL -o "$installer" {POOL_INSTALL_URL} '
            f'&& POOL_INSTALL_ACCEPT_EULA=1 POOL_INSTALL_DIR={home}/bin sh "$installer" {quote(self.config.pool_version)}'
        )

    def _pool_agent_config(self, base_url: str) -> Dict[str, Any]:
        # `pool exec --agent-config-file` does not merge with pool's defaults, so this mirrors them.
        context_window = self.config.pool_max_context_window
        config = {
            "max_steps": 0,
            "enabled_tools": [
                "todo_action",
                "read",
                "edit",
                "write",
                "get_diagnostics",
                "question",
                "shell",
                "shell_list",
                "shell_tail",
                "shell_status",
                "shell_kill",
                "shell_send",
                "shell_wait",
                "skill",
                "subagent",
                "switch_mode",
                "list_secrets",
                "exit",
            ],
            "http_timeout": "5m0s",
            "enable_no_tool_call_retry": False,
            "enable_truncation_recovery": True,
            "enable_multi_tool_calls": True,
            "enable_tool_call_loop_detection": True,
            "enable_todo_reminders": True,
            "parallel_tool_calls": {"tools": ["subagent"]},
            "tools": {
                "edit": {"fuzzy_match_threshold": 0.05},
                "read": {"max_lines": 500, "line_num_separator": "|◊|"},
                "write": {"line_num_separator": "|◊|"},
                "shell": {"foreground_timeout": "120s"},
            },
            "model": {
                "provider": {
                    "openai": {
                        "base_url": base_url,
                        "api_key": "dummy_key",  # pragma: allowlist secret
                        "model_id": self.config.pool_model,
                        "use_streaming": False,
                    }
                },
                "prompt": {"hidden_tool_names": ["exit"]},
                "max_completion_retries": 2,
                "max_completion_retries_transient": 3,
                "exit_tool_on_stop": True,
                "streaming_events_interval": "100ms",
            },
            "memory": {
                "compact": {
                    "TriggerCompressionTokenCount": int(context_window * 0.8),
                    "MaxSummarizeTokenCount": int(context_window * 0.7),
                    "MinRetainedSteps": 5,
                }
            },
        }
        return _deep_merge(config, self.config.pool_agent_config)

    def _pool_env(self, home: str) -> Dict[str, str]:
        return {
            "HOME": home,
            "XDG_CONFIG_HOME": f"{home}/config",
            "XDG_STATE_HOME": f"{home}/state",
            "XDG_DATA_HOME": f"{home}/data",
            # Any value makes pool skip its login bootstrap; the provider key lives in the agent config.
            "POOLSIDE_API_KEY": "dummy_key",  # pragma: allowlist secret
            **self.config.pool_env,
        }

    def _build_command(self, home: str) -> str:
        env_str = " ".join(f"{key}={quote(value)}" for key, value in self._pool_env(home).items())
        extra_args = " ".join(quote(arg) for arg in self.config.pool_extra_args)
        # pool exits 4 when it gives up on the task, so the run is not chained with &&.
        return f"""
        echo "Shell: $SHELL" \
        && mkdir -p {home} \
        && {self._install_command(home)} \
        && echo "Installed pool" \
        && {home}/bin/pool --version \
        && {{ {env_str} {home}/bin/pool exec -o json --sandbox disabled --unsafe-auto-allow \
            --agent-config-file {home}/agent_config.json -f {home}/prompt.txt {extra_args} \
            > {home}/events.jsonl 2> {home}/pool.stderr; \
            echo "pool run finished rc=$?"; }}
        """

    @staticmethod
    def _query_from_body(body: NeMoGymResponseCreateParamsNonStreaming) -> str:
        query = None
        for input_item in body.input:
            if input_item.role == "user":
                assert not query, body.input
                if isinstance(input_item.content, str):
                    query = input_item.content
                elif isinstance(input_item.content, list):
                    assert len(input_item.content) == 1, body.input
                    query = input_item.content[0]["text"]
        assert query, body.input
        return query

    def _agent_sandbox_observation(
        self, *, sandbox: AsyncSandbox, return_code: Any, error_type: Any, finished: bool
    ) -> SandboxObservation:
        handle = getattr(sandbox, "_handle", None)
        normalized_error = error_type.lower() if isinstance(error_type, str) else ""
        match (normalized_error, finished, return_code):
            case (err, _, _) if "timeout" in err:
                outcome = "timeout"
            case (err, _, _) if err:
                outcome = "sandbox_error"
            case (_, True, 0):
                outcome = "completed"
            case (_, _, int()):
                outcome = "failed"
            case _:
                outcome = "unknown"
        provider = getattr(handle, "provider_name", None)
        sandbox_id = getattr(handle, "sandbox_id", None)
        return SandboxObservation(
            role="agent",
            provider=provider if isinstance(provider, str) else None,
            sandbox_id=sandbox_id if isinstance(sandbox_id, str) else None,
            outcome=outcome,
            exit_code=return_code if not normalized_error and isinstance(return_code, int) else None,
            error_type=error_type if isinstance(error_type, str) else None,
        )

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        sandbox = self._sandbox_id_to_sandbox[request.cookies["sandbox_id"]]
        query = self._query_from_body(body)
        # Everything pool writes stays under `home`, outside the repo workdir the resources server diffs.
        home = f"/tmp/nemo-gym-pool-{uuid4().hex}"
        base_url = await self._model_base_url(request)

        await sandbox.exec(command=f"mkdir -p {home}")
        uploads = {"prompt.txt": query, "agent_config.json": json.dumps(self._pool_agent_config(base_url))}
        for remote_name, content in uploads.items():
            with tempfile.NamedTemporaryFile("w", suffix=remote_name, delete=False) as local_file:
                local_file.write(content)
            try:
                await sandbox.upload(local_file.name, f"{home}/{remote_name}")
            finally:
                Path(local_file.name).unlink(missing_ok=True)

        command = self._build_command(home)
        if self.config.debug:
            print(f"Running command:\n```bash\n{command}\n```\n", file=sys.stderr)

        run_error_type = None
        try:
            result = await sandbox.exec(command=command, timeout_s=self.config.sandbox_timeout)
        except Exception as exc:
            result = None
            run_error_type = type(exc).__name__
            print("pool exec hit error.", format_exc(), file=sys.stderr)

        result_stdout = (result.stdout if result else "") or ""
        result_stderr = (result.stderr if result else "") or ""
        if self.config.debug and result:
            print("pool install and run stdout:\n", result_stdout, file=sys.stderr)
            print("pool install and run stderr:\n", result_stderr, file=sys.stderr)

        pool_finished = False
        pool_exit_code: Optional[int] = None
        tail = result_stdout.rsplit("Shell: ", maxsplit=1)
        if len(tail) > 1 and (match := _FINISHED_RE.search(tail[1])):
            pool_finished = True
            pool_exit_code = int(match.group(1))

        results_dir: Path = Path(__file__).parent / "results" / request.session[SESSION_ID_KEY]
        results_dir.mkdir(parents=True, exist_ok=True)
        events_local_fpath = results_dir / "events.jsonl"
        for name in ("events.jsonl", "pool.stderr"):
            try:
                await sandbox.download(f"{home}/{name}", results_dir / name)
            except Exception:
                print(f"Failed to download {name}", format_exc(), file=sys.stderr)

        output: List[NeMoGymResponseOutputItem] = []
        parse_metadata: Dict[str, Any] = {}
        pool_events_found = events_local_fpath.exists() and events_local_fpath.stat().st_size > 0
        if pool_events_found:
            output, parse_metadata = parse_pool_events(events_local_fpath.read_text())

        rollout_id = getattr(request.state, "_ng_observation_invocation_id", None)
        observations = None
        if isinstance(rollout_id, str):
            sandbox_observation = self._agent_sandbox_observation(
                sandbox=sandbox,
                return_code=pool_exit_code if pool_finished else getattr(result, "return_code", None),
                error_type=getattr(result, "error_type", None) or run_error_type,
                finished=pool_finished,
            )
            status = {"completed": "completed", "failed": "failed", "sandbox_error": "failed", "timeout": "incomplete"}
            observations = AgentObservationBundle(
                source="pool",
                records=[
                    AgentInvocation(
                        invocation_id=rollout_id,
                        status=status.get(sandbox_observation.outcome, "unknown"),
                        error_type="; ".join(parse_metadata["errors"]) if parse_metadata.get("errors") else None,
                    ),
                    sandbox_observation,
                ],
                gaps=[
                    ObservationGap(code="model_call_ownership_unavailable"),
                    ObservationGap(code="sandbox_lifecycle_timing_unavailable"),
                ],
            )

        self._sandbox_id_to_run_result[request.cookies["sandbox_id"]] = {
            "pool_events_fpath": str(events_local_fpath) if pool_events_found else "",
            "pool_run_stdout": result_stdout,
            "pool_run_stderr": result_stderr,
            "pool_finished": pool_finished,
            "pool_exit_code": pool_exit_code,
            "pool_events_found": pool_events_found,
            "_ng_agent_observations": observations,
        }

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=body.model or self.config.model_server.name,
            object="response",
            output=output,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=None,
        )

    async def run(self, request: Request, body: PoolSandboxedAgentRunRequest) -> PoolSandboxedAgentVerifyResponse:
        cookies = request.cookies
        session_key = request.session[SESSION_ID_KEY]
        rollout_id = self.rollout_id_from_run(body)

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = cookies | seed_session_response.cookies

        seed_session_result = await seed_session_response.json()
        sandbox = await self._connect_sandbox(seed_session_result.get("sandbox_handle"))
        self._sandbox_id_to_sandbox[session_key] = sandbox
        cookies["sandbox_id"] = session_key
        request._cookies = cookies
        request.state._ng_observation_invocation_id = rollout_id

        try:
            response = await self.responses(request, body.responses_create_params)
        finally:
            del request.state._ng_observation_invocation_id
            run_result = self._sandbox_id_to_run_result.pop(session_key, {})
            observations = run_result.pop("_ng_agent_observations", None)

        verify_request = PoolSandboxedAgentVerifyRequest.model_validate(body.model_dump() | {"response": response})
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)

        try:
            await sandbox.stop()
        except Exception:
            print("Failed to stop sandbox", format_exc(), file=sys.stderr)
        self._sandbox_id_to_sandbox.pop(session_key, None)

        response_dict = await get_response_json(verify_response)
        response_dict |= run_result
        raw_verifier_observation = response_dict.pop("verifier_sandbox_observation", None)
        if rollout_id is not None:
            if observations is None:
                observations = AgentObservationBundle(
                    source="pool",
                    records=[AgentInvocation(invocation_id=rollout_id)],
                    gaps=[ObservationGap(code="observation_capture_failed")],
                )
            if raw_verifier_observation is not None:
                try:
                    verifier_observation = SandboxObservation.model_validate(raw_verifier_observation)
                    if verifier_observation.role != "verifier":
                        raise ValueError("resources server returned a non-verifier sandbox observation")
                    observations.records.append(verifier_observation)
                except Exception:
                    observations.gaps.append(ObservationGap(code="verifier_sandbox_observation_invalid"))
            else:
                observations.gaps.append(ObservationGap(code="verifier_sandbox_observation_unavailable"))
            response_dict["ng_agent_observations"] = observations.model_dump(mode="json")
        return PoolSandboxedAgentVerifyResponse.model_validate(response_dict)


if __name__ == "__main__":
    PoolSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = PoolSandboxedAgent.run_webserver()  # noqa: F401
