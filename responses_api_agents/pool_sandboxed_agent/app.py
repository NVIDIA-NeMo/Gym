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
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ObservationGap,
    SandboxObservation,
)
from nemo_gym.sandbox import AsyncSandbox, create_provider
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    get_response_json,
    get_server_url,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
)


POOL_INSTALL_URL = "https://downloads.poolside.ai/pool/install.sh"
_FINISHED_RE = re.compile(r"pool run finished rc=(\d+)")


class PoolSandboxedAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    # `latest` or a release such as 1.0.16; ignored when remote_pool_binary_path is set.
    pool_version: str = "latest"
    remote_pool_binary_path: Optional[str] = None
    # Model name pool sends to the Gym model proxy, which substitutes the configured policy model.
    pool_model: str = "dummy_model"
    pool_max_context_window: int
    pool_extra_args: List[str] = Field(default_factory=list)
    pool_env: Dict[str, str] = Field(default_factory=dict)

    # Overrides the Gym-side model server URL when it is not routable from inside the sandbox.
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
    """Convert `pool exec -o json` NLJSON output into Responses output items.

    Reasoning is buffered and prepended to the next assistant message inside <think> tags,
    matching the Claude Code and Codex agents. `thought` events duplicate `reasoning` and are
    skipped. Each toolCall is paired with the following toolCallResult.
    """
    output_items: List[NeMoGymResponseOutputItem] = []
    buffered_think: Optional[str] = None
    pending_call: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {"errors": []}

    def flush_pending_call(output: str) -> None:
        nonlocal pending_call
        if pending_call is None:
            return
        call_id = f"call-{uuid4().hex[:8]}"
        output_items.append(
            NeMoGymResponseFunctionToolCall(
                arguments=json.dumps(pending_call.get("args") or {}),
                call_id=call_id,
                name=str(pending_call.get("name") or "tool"),
                type="function_call",
                id=call_id,
                status="completed",
            )
        )
        output_items.append(
            NeMoGymFunctionCallOutput(type="function_call_output", call_id=call_id, output=output, status="completed")
        )
        pending_call = None

    def emit_message(text: str) -> None:
        nonlocal buffered_think
        if buffered_think:
            text = f"<think>\n{buffered_think}\n</think>\n\n{text}" if text else f"<think>\n{buffered_think}\n</think>"
            buffered_think = None
        output_items.append(
            NeMoGymResponseOutputMessage(
                id=f"msg-{len(output_items)}",
                content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                role="assistant",
                status="completed",
                type="message",
            )
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

        etype = event.get("type")
        if etype == "reasoning":
            think = event.get("reasoning") or ""
            if think.strip():
                buffered_think = f"{buffered_think}\n{think}" if buffered_think else think
        elif etype == "thought":
            continue
        elif etype == "assistantMessage":
            text = event.get("message") or ""
            if text.strip() or buffered_think:
                flush_pending_call("")
                emit_message(text)
        elif etype == "toolCall":
            flush_pending_call("")
            pending_call = event
        elif etype == "toolCallResult":
            if "err" in event:
                output = f"[error] {event['err']}"
            elif "entries" in event:
                output = "\n".join(event.get("entries") or [])
            else:
                output = str(event.get("result") or "")
            flush_pending_call(output)
        elif etype == "error" or "error" in event:
            metadata["errors"].append(str(event.get("error") or "unknown error"))
        else:
            raise NotImplementedError(event)

    flush_pending_call("")
    if buffered_think:
        emit_message("")
    if not metadata["errors"]:
        metadata.pop("errors")
    return output_items, metadata


class PoolSandboxedAgent(SimpleResponsesAPIAgent):
    config: PoolSandboxedAgentConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._sandbox_id_to_sandbox: Dict[str, AsyncSandbox] = dict()
        self._sandbox_id_to_run_result: Dict[str, Dict[str, Any]] = dict()

    async def _connect_sandbox(self, sandbox_id: Optional[str]) -> AsyncSandbox:
        if not sandbox_id:
            raise ValueError("pool_sandboxed_agent requires a sandbox_handle from the resources server")
        global_config_dict = get_global_config_dict()
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, global_config_dict))
        resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)
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

    def _pool_env(self, home: str, base_url: str) -> Dict[str, str]:
        # Everything pool writes (config, state, trajectories) stays under `home`, outside the
        # repo workdir: the resources server extracts the patch with `git diff` in the workdir.
        return {
            "HOME": home,
            "XDG_CONFIG_HOME": f"{home}/config",
            "XDG_STATE_HOME": f"{home}/state",
            "XDG_DATA_HOME": f"{home}/data",
            "POOLSIDE_STANDALONE_BASE_URL": base_url,
            "POOLSIDE_API_KEY": "dummy_key",  # pragma: allowlist secret
            "POOLSIDE_STANDALONE_MODEL": self.config.pool_model,
            "POOLSIDE_STANDALONE_CONTEXT_LENGTH": str(self.config.pool_max_context_window),
            **self.config.pool_env,
        }

    def _build_command(self, home: str, base_url: str) -> str:
        env_str = " ".join(f"{key}={quote(value)}" for key, value in self._pool_env(home, base_url).items())
        extra_args = " ".join(quote(arg) for arg in self.config.pool_extra_args)
        # `pool exec` exits 4 when the agent gives up on the task, so the run is not chained with &&.
        return f"""
        echo "Shell: $SHELL" \
        && mkdir -p {home} \
        && {self._install_command(home)} \
        && echo "Installed pool" \
        && {home}/bin/pool --version \
        && {{ {env_str} {home}/bin/pool exec -o json --sandbox disabled --unsafe-auto-allow \
            -f {home}/prompt.txt {extra_args} > {home}/events.jsonl 2> {home}/pool.stderr; \
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
        if "timeout" in normalized_error:
            outcome = "timeout"
        elif normalized_error:
            outcome = "sandbox_error"
        elif finished and return_code == 0:
            outcome = "completed"
        elif isinstance(return_code, int):
            outcome = "failed"
        else:
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
        home = f"/tmp/nemo-gym-pool-{uuid4().hex}"
        base_url = await self._model_base_url(request)

        await sandbox.exec(command=f"mkdir -p {home}")
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as prompt_file:
            prompt_file.write(query)
        try:
            await sandbox.upload(prompt_file.name, f"{home}/prompt.txt")
        finally:
            Path(prompt_file.name).unlink(missing_ok=True)

        command = self._build_command(home, base_url)
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
        for remote_name, local_name in (("events.jsonl", "events.jsonl"), ("pool.stderr", "pool.stderr")):
            try:
                await sandbox.download(f"{home}/{remote_name}", results_dir / local_name)
            except Exception:
                print(f"Failed to download {remote_name}", format_exc(), file=sys.stderr)

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
