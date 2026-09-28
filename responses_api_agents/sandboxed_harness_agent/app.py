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
"""Runs a coding-agent harness CLI inside the task's own sandbox.

The resources server seeds the task and hands back its sandbox. This agent attaches to it, runs
the harness's command once under `sandbox_timeout`, lets the harness collect its transcript,
calls /verify and stops the sandbox. It knows no harness: each harness is a subclass that
provides `_harness_command` and `_harness_collect` (see `opencode_sandboxed_agent`).
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from time import time
from traceback import format_exc
from typing import Any, ClassVar, Dict, List, Optional
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
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputItem,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ObservationGap,
    SandboxObservation,
    ToolCallObservation,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, create_provider
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    get_response_json,
    raise_for_status,
)


class SandboxedHarnessAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    # Sandbox config
    sandbox_provider: str
    sandbox_config: Dict[str, Any]
    sandbox_timeout: float

    debug: bool = False


class SandboxedHarnessAgentRunRequest(BaseRunRequest):
    # Allow for benchmark params to propagate properly
    model_config = ConfigDict(extra="allow")


class SandboxedHarnessAgentVerifyRequest(BaseVerifyRequest):
    # Allow for benchmark params to propagate properly
    model_config = ConfigDict(extra="allow")


class SandboxedHarnessAgentVerifyResponse(BaseVerifyResponse):
    # Allow for benchmark params to propagate properly
    model_config = ConfigDict(extra="allow")

    # Whether the harness printed its finished marker; the same field for every harness.
    harness_finished: bool
    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


@dataclass
class HarnessTranscript:
    """What a harness recovered from the sandbox after its command ran."""

    output: List[NeMoGymResponseOutputItem]
    usage: Optional[NeMoGymResponseUsage]
    # Local copy of the harness's own export, when one was found.
    results_fpath: Optional[Path]
    observations: Optional[AgentObservationBundle]


class SandboxedHarnessAgent(SimpleResponsesAPIAgent):
    config: SandboxedHarnessAgentConfig

    # Set by each harness subclass.
    harness_name: ClassVar[str]  # for logs, e.g. "OpenCode"
    harness_id: ClassVar[str]  # prefix of the harness's /run result fields and its observation source
    finished_marker: ClassVar[str]  # printed last by the command, only when the harness succeeded
    system_prompt: ClassVar[Optional[str]] = None  # prepended to responses_create_params.input in the result
    verify_response_class: ClassVar[type[SandboxedHarnessAgentVerifyResponse]] = SandboxedHarnessAgentVerifyResponse

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)

        self._sandbox_id_to_sandbox: Dict[str, AsyncSandbox] = dict()
        self._sandbox_id_to_run_result: Dict[str, Dict[str, Any]] = dict()

    async def _harness_command(self, request: Request, query: str, collect_observations: bool) -> tuple[str, Any]:
        """Return the shell command that installs and runs the harness on `query`, plus any state
        `_harness_collect` needs again.

        The command prints "Shell: $SHELL" first and `finished_marker` last, only when every step
        before it succeeded.
        """
        raise NotImplementedError

    async def _harness_collect(
        self,
        request: Request,
        sandbox: AsyncSandbox,
        state: Any,
        collect_observations: bool,
        observation_invocation_id: Optional[str],
    ) -> HarnessTranscript:
        """After the command ran, fetch the harness's transcript from the sandbox."""
        raise NotImplementedError

    async def _start_sandbox(self, sandbox_id: Optional[str] = None) -> AsyncSandbox:
        global_config_dict = get_global_config_dict()
        resolved_sandbox_provider = create_provider(
            resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        )
        provider_default_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)

        if sandbox_id:
            sandbox = await AsyncSandbox.connect({"sandbox_id": sandbox_id}, provider=resolved_sandbox_provider)
            return sandbox

        if self.config.debug:
            print("Creating new sandbox since one wasn't provided", file=sys.stderr)

        resources = SandboxResources.from_mapping(self.config.sandbox_config.get("resources", {}))
        # TODO @bxyu-nvidia: Refactor this after swapping to PTY as this should be set on the SWE Bench resources server side
        env = cpu_cap_env(resources.cpu) if self.config.sandbox_config.get("derive_cpu_env", True) else {}
        env |= dict(self.config.sandbox_config.get("env", {}))  # explicit keys win over the derived caps

        # TODO @bxyu-nvidia: Refactor this after Hemil's swap from Python dataclass to Pydantic BaseModel
        sandbox_spec = SandboxSpec(
            image="swebench/sweb.eval.x86_64.astropy_1776_astropy-12907",  # This is just the first SWE Bench Verified image for now
            ttl_s=self.config.sandbox_config.get("ttl_s", None),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s", None),
            workdir=None,  # Default to container's WORKDIR
            env=env,
            files=dict(),
            metadata=provider_default_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {
                "nemo_gym_agent": self.config.name,
            },
            resources=resources,
            entrypoint=None,
            provider_options=self.config.sandbox_config.get("provider_options", {}),
        )

        sandbox = AsyncSandbox(resolved_sandbox_provider)
        await sandbox.start(sandbox_spec)

        return sandbox

    def _agent_sandbox_observation(
        self,
        *,
        sandbox: AsyncSandbox,
        return_code: Any,
        error_type: Any,
        finished: bool,
    ) -> SandboxObservation:
        handle = getattr(sandbox, "_handle", None)
        handle_provider = getattr(handle, "provider_name", None)
        handle_sandbox_id = getattr(handle, "sandbox_id", None)
        normalized_error = error_type.lower() if isinstance(error_type, str) else ""
        if "timeout" in normalized_error:
            outcome = "timeout"
        elif normalized_error:
            outcome = "sandbox_error"
        elif return_code == 0 and finished:
            outcome = "completed"
        elif isinstance(return_code, int):
            outcome = "failed" if return_code != 0 else "unknown"
        else:
            outcome = "unknown"
        return SandboxObservation(
            role="agent",
            provider=handle_provider if isinstance(handle_provider, str) else None,
            sandbox_id=handle_sandbox_id if isinstance(handle_sandbox_id, str) else None,
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

        query = None
        # This can be modified to handle system/developer prompts too.
        for input_item in body.input:
            if input_item.role == "user":
                assert not query, body.input
                if isinstance(input_item.content, str):
                    query = input_item.content
                elif isinstance(input_item.content, list):
                    assert len(input_item.content) == 1, body.input
                    query = input_item.content[0]["text"]

        assert query, body.input

        observation_invocation_id = getattr(request.state, "_ng_observation_invocation_id", None)
        observation_invocation_id = observation_invocation_id if isinstance(observation_invocation_id, str) else None
        collect_observations = observation_invocation_id is not None
        command, harness_state = await self._harness_command(request, query, collect_observations)

        run_error_type = None
        try:
            result = await sandbox.exec(
                command=command,
                timeout_s=self.config.sandbox_timeout,
            )
        except Exception as exc:
            result = None
            run_error_type = type(exc).__name__
            print(f"{self.harness_name} exec hit error.", format_exc(), file=sys.stderr)

        if self.config.debug and result:
            print(f"{self.harness_name} install and run stdout:\n", result.stdout, file=sys.stderr)
            print(f"{self.harness_name} install and run stderr:\n", result.stderr, file=sys.stderr)

        transcript = await self._harness_collect(
            request, sandbox, harness_state, collect_observations, observation_invocation_id
        )
        observations = transcript.observations

        result_stdout = (result.stdout if result else "") or ""
        result_stderr = (result.stderr if result else "") or ""
        finished = False
        std_out_split = result_stdout.rsplit("Shell: ", maxsplit=1)
        if len(std_out_split) > 1:
            finished = self.finished_marker in std_out_split[1]

        if collect_observations and observations is not None:
            agent_sandbox_observation = self._agent_sandbox_observation(
                sandbox=sandbox,
                return_code=getattr(result, "return_code", None),
                error_type=getattr(result, "error_type", None) or run_error_type,
                finished=finished,
            )
            for record in observations.records:
                if isinstance(record, ToolCallObservation):
                    record.sandbox_id = agent_sandbox_observation.sandbox_id
                elif isinstance(record, AgentInvocation) and record.parent_invocation_id is None:
                    status = {
                        "completed": "completed",
                        "failed": "failed",
                        "sandbox_error": "failed",
                        "timeout": "incomplete",
                        "cancelled": "incomplete",
                    }.get(agent_sandbox_observation.outcome)
                    if status is not None:
                        record.status = status
            observations.records.append(agent_sandbox_observation)
            observations.gaps.append(ObservationGap(code="sandbox_lifecycle_timing_unavailable"))

        export_found = transcript.results_fpath is not None
        run_result = {
            f"{self.harness_id}_results_fpath": str(transcript.results_fpath) if export_found else "",
            f"{self.harness_id}_run_stdout": result_stdout,
            f"{self.harness_id}_run_stderr": result_stderr,
            f"{self.harness_id}_export_found": export_found,
            f"{self.harness_id}_finished": finished,
            "harness_finished": finished,
        }
        if collect_observations:
            run_result["_ng_agent_observations"] = observations
        self._sandbox_id_to_run_result[request.cookies["sandbox_id"]] = run_result

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=body.model or self.config.model_server.name,
            object="response",
            output=transcript.output,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=transcript.usage,
        )

    async def run(
        self, request: Request, body: SandboxedHarnessAgentRunRequest
    ) -> SandboxedHarnessAgentVerifyResponse:
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

        # @bxyu-nvidia: "sandbox_handle" comes from resources_servers/swebench/app.py
        # Once we graduate to use the sandbox server, this will be in a generic seed_session type that can be model validated.
        seed_session_result = await seed_session_response.json()
        sandbox = await self._start_sandbox(
            sandbox_id=seed_session_result.get("sandbox_handle"),
        )
        self._sandbox_id_to_sandbox[request.session[SESSION_ID_KEY]] = sandbox

        # Propagating the sandbox handle
        cookies["sandbox_id"] = session_key

        request._cookies = cookies
        request.state._ng_observation_invocation_id = rollout_id
        observations = None
        try:
            response = await self.responses(request, body.responses_create_params)
        finally:
            del request.state._ng_observation_invocation_id
            run_result = self._sandbox_id_to_run_result.get(session_key, {})
            observations = run_result.pop("_ng_agent_observations", None)

        verify_request = SandboxedHarnessAgentVerifyRequest.model_validate(body.model_dump() | {"response": response})

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
        run_result = self._sandbox_id_to_run_result.pop(session_key)
        response_dict |= run_result
        raw_verifier_sandbox_observation = response_dict.pop("verifier_sandbox_observation", None)
        if self.system_prompt is not None:
            response_dict["responses_create_params"]["input"].insert(
                0, {"content": self.system_prompt, "role": "system"}
            )

        if rollout_id is not None:
            if observations is None:
                observations = AgentObservationBundle(
                    source=self.harness_id,
                    records=[AgentInvocation(invocation_id=rollout_id)],
                    gaps=[ObservationGap(code="observation_capture_failed")],
                )
            if raw_verifier_sandbox_observation is not None:
                try:
                    verifier_observation = SandboxObservation.model_validate(raw_verifier_sandbox_observation)
                    if verifier_observation.role != "verifier":
                        raise ValueError("resources server returned a non-verifier sandbox observation")
                    observations.records.append(verifier_observation)
                except Exception:
                    observations.gaps.append(ObservationGap(code="verifier_sandbox_observation_invalid"))
            else:
                observations.gaps.append(ObservationGap(code="verifier_sandbox_observation_unavailable"))
            response_dict["ng_agent_observations"] = observations.model_dump(mode="json")
        return self.verify_response_class.model_validate(response_dict)
