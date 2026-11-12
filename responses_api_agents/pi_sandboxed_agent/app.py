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

"""Run the native Pi CLI in a provider-managed sandbox; parse and grade on the host."""

import json
import logging
from asyncio import Semaphore
from contextvars import ContextVar
from copy import deepcopy
from functools import cache
from hashlib import sha256
from pathlib import Path
from shlex import join, quote
from tempfile import TemporaryDirectory
from time import time
from typing import Any, ClassVar, Literal
from uuid import uuid4

from fastapi import Request
from openai.types.responses.response import IncompleteDetails
from pydantic import Field, field_validator

from nemo_gym.agents import (
    Body,
    SimpleResponsesAPIAgent,
    restricted_network_policy,
    sandbox_server_url,
    seed_mcp_servers,
    verify_agent_response,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import ModelServerRef, ResourcesServerRef, get_global_config_dict
from nemo_gym.core import (
    PROVIDER_IO_TIMEOUT,
    PROVIDER_REQUEST_FAILED,
    FailureBoundary,
    FailureReportRow,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NgFailureEvent,
    OutcomeAction,
    StageOutcomePolicyConfig,
    agent_route_failure,
    choose_stage_outcome,
    failure_report_for,
    observe_failure,
    sdk_error_details,
)
from nemo_gym.core.failure_kinds import PROVIDER_COMMAND_OUTCOME_UNKNOWN
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxResources,
    SandboxSpec,
    cpu_cap_env,
    create_provider,
    resolve_provider_config,
    resolve_provider_metadata,
    sandbox_failure,
)
from nemo_gym.trajectory import AgentInvocation, SandboxObservation, ToolCallObservation
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint, is_nemo_gym_fastapi_worker
from nemo_gym.utils.exception_inspection import error_to_http_status
from responses_api_agents.pi_agent.app import (
    MCP_SETUP_ERROR_EXIT_CODE,
    PiAgent,
    PiAgentConfig,
    PiAgentRunRequest,
    PiAgentVerifyResponse,
    parse_pi_events,
)


LOG = logging.getLogger(__name__)
_RUN: ContextVar[dict[str, Any] | None] = ContextVar("pi_sandboxed_run", default=None)

DEPRECATION_WARNING = (
    "DEPRECATED: pi_sandboxed_agent runs each episode through its own legacy /run path and will be removed after "
    "the environment-server migration. Use pi_agent with sandbox sessions under an environment server instead. "
    "The HLE search presets, which use tool_servers, have no session path yet and keep working here. "
    "See responses_api_agents/pi_agent/README.md#configure-and-run."
)


@cache
def _warn_deprecated_once() -> None:
    # Workers of a multi-worker server rebuild it after the main process has already logged this.
    if not is_nemo_gym_fastapi_worker():
        LOG.warning(DEPRECATION_WARNING)


class PiSandboxedAgentConfig(PiAgentConfig):
    """Pi CLI settings plus provider allocation and host-side tool services."""

    model_server: ModelServerRef
    sandbox_provider: str
    sandbox_config: dict[str, Any]
    network_access: Literal["inherit", "model_only", "model_and_tools"] = "inherit"
    tool_servers: list[ResourcesServerRef] = Field(default_factory=list)
    execution_failure_reward_zero: bool = False
    artifacts_dir: str = "results/pi_sandboxed"
    # What a provider failure of the agent's own command becomes. The default stops the rollout ungraded; a rule
    # may count a kind or status as a zero. Retrying is the collector's, not this agent's.
    outcome_policy: StageOutcomePolicyConfig = Field(default_factory=StageOutcomePolicyConfig)

    @field_validator("outcome_policy")
    @classmethod
    def require_supported_agent_actions(cls, policy: StageOutcomePolicyConfig) -> StageOutcomePolicyConfig:
        """Only the actions this agent can perform for a failure of its own command."""
        rules = policy.by_stage.get("agent", policy.default_policy)
        actions = {
            rules.default_action,
            *rules.by_failure_kind.values(),
            *rules.by_status_code.values(),
            *rules.by_failure_class.values(),
        }
        if actions - {OutcomeAction.FAIL, OutcomeAction.COUNT_AS_ZERO}:
            raise ValueError("Pi sandboxed agent-stage policy supports only fail and count_as_zero")
        return policy


class PiSandboxedAgentVerifyResponse(PiAgentVerifyResponse):
    pi_failed: bool = False
    pi_exit_code: int | None = None
    pi_error_type: str | None = None
    pi_results_dir: str = ""


class PiSandboxedAgent(PiAgent):
    config: PiSandboxedAgentConfig
    # The harness runs inside the task sandbox and calls the model server from there.
    calls_model_from_sandbox: ClassVar[bool] = True

    def model_post_init(self, __context: Any) -> None:
        # Pi is preinstalled in the sandbox image, never installed on the Gym host.
        SimpleResponsesAPIAgent.model_post_init(self, __context)
        _warn_deprecated_once()
        self.sem = Semaphore(self.config.concurrency)

    def _resolve_model_base_url(self, rollout_id: str | None = None) -> str:
        context = _RUN.get()
        if context is None:
            raise RuntimeError("Pi sandbox model routing requires a seeded /run request")
        return (
            self.base_url_for_run(
                sandbox_server_url(
                    self.config.model_server.name, require_reachable=self.config.network_access != "inherit"
                ),
                context["body"],
            ).rstrip("/")
            + "/v1"
        )

    async def _start_sandbox(self, seed: dict) -> AsyncSandbox:
        global_config = get_global_config_dict()
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, global_config))
        descriptor = seed.get("sandbox_descriptor")
        if not descriptor and seed.get("sandbox_handle"):
            descriptor = {"sandbox_id": seed["sandbox_handle"]}
        if descriptor:
            if self.config.network_access != "inherit":
                raise ValueError("Cannot verify network policy on a resource-owned sandbox")
            return await AsyncSandbox.connect(descriptor, provider=provider)
        options = deepcopy(self.config.sandbox_config)
        resources = SandboxResources.from_mapping(options.pop("resources", {}))
        env = cpu_cap_env(resources.cpu) if options.pop("derive_cpu_env", True) else {}
        env.update(options.pop("env", {}))
        metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config)
        metadata = metadata | options.pop("metadata", {}) | {"nemo_gym_agent": self.config.name}
        if self.config.network_access != "inherit":
            urls = [sandbox_server_url(self.config.model_server.name, require_reachable=True)]
            if self.config.network_access == "model_and_tools":
                if not self.config.tool_servers:
                    raise ValueError("model_and_tools requires tool_servers")
                urls.extend(sandbox_server_url(s.name, require_reachable=True) for s in self.config.tool_servers)
            options.setdefault("provider_options", {})["network_policy"] = restricted_network_policy(
                provider.name, urls
            )
        sandbox = AsyncSandbox(provider)
        await sandbox.start(SandboxSpec(resources=resources, env=env, metadata=metadata, **options))
        return sandbox

    async def _run_pi(
        self,
        instruction: str,
        system_prompt: str | None,
        *,
        rollout_id: str | None = None,
        collect_observations: bool = True,
        timeout: float | None = None,
    ) -> tuple[list[Any], dict[str, int], str, list[tuple[float, dict[str, Any]]]]:
        timeout = self.config.timeout if timeout is None else timeout
        context = _RUN.get()
        if context is None:
            raise RuntimeError("Pi sandbox execution requires a seeded /run request")
        sandbox = context["sandbox"]
        remote = "/tmp/nemo-gym-pi-" + uuid4().hex
        agent_home = remote + "/home/.pi/agent"
        files = {
            agent_home + "/models.json": json.dumps(self._build_models_config(rollout_id)),
            agent_home + "/settings.json": json.dumps({"compaction": {"enabled": self.config.auto_compaction}}),
        }
        files[remote + "/capture.py"] = Path(__file__).with_name("capture.py").read_text()
        env = dict(self.config.env) | {"HOME": remote + "/home", "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0"}
        cmd = [*self.config.command_parts, "--print", "--mode", "json", "--no-session"]
        extensions = []
        if self.config.bash_timeout is not None:
            env["NEMO_GYM_PI_BASH_TIMEOUT"] = str(self.config.bash_timeout)
            extensions.append("bash-timeout.mjs")
        if self.config.output_token_policy == "remaining_context":
            extensions.append("remaining-context.mjs")
        mcp = {name: server.model_dump() for name, server in self.config.mcp_servers.items()} | context["mcp"]
        if mcp:
            files[remote + "/mcp.json"] = json.dumps(mcp)
            env["NEMO_GYM_PI_MCP_CONFIG"] = remote + "/mcp.json"
            extensions.append("gym_mcp.mjs")
        for extension in extensions:
            files[remote + "/" + extension] = (Path(__file__).parent.parent / "pi_agent" / extension).read_text()
            cmd.extend(["--extension", remote + "/" + extension])
        cmd += ["--provider", "nemo", "--model", self.config.model]
        if self.config.thinking:
            cmd += ["--thinking", self.config.thinking]
        if system_prompt:
            cmd += ["--append-system-prompt", system_prompt]
        cmd += self.config.extra_args + [instruction]
        with TemporaryDirectory() as directory:
            for i, (target, contents) in enumerate(files.items()):
                local = Path(directory) / str(i)
                local.write_text(contents)
                local.chmod(0o600)
                await sandbox.upload(local, target)
        # Configuration failures are infrastructure errors, not zero-reward generations.
        setup = await sandbox.exec(command=f"chmod -R go-rwx {quote(remote)}", timeout_s=self.config.timeout)
        if setup.return_code != 0 or setup.error_type:
            raise RuntimeError("Unable to protect Pi session configuration")
        if self.config.pi_version:
            version = await sandbox.exec(
                command=join([*self.config.command_parts, "--version"]), timeout_s=self.config.timeout
            )
            if version.return_code != 0 or (version.stdout or "").strip() != self.config.pi_version:
                raise RuntimeError("Preinstalled Pi version does not match pi_version")
        # Persist the native stream in the sandbox too: provider stdout may be truncated.
        stdout_path, stderr_path = remote + "/stdout.jsonl", remote + "/stderr.log"
        result = None
        error_type = None
        try:
            result = await sandbox.exec(
                command=f"{join(['python3', remote + '/capture.py', remote + '/events.jsonl', *cmd])} > {quote(stdout_path)} 2> {quote(stderr_path)}",
                env=env,
                timeout_s=timeout,
            )
        except NgFailureEvent as exc:
            if exc.boundary != FailureBoundary.SANDBOX:
                raise
            provider_failure = exc
        except Exception as exc:
            # An exception, including a provider timeout, does not establish how Pi ended.
            provider_failure = self._untyped_exec_failure(sandbox, exc)
        else:
            provider_failure = self._unclassified_result_failure(sandbox, result)
        if provider_failure is not None:
            # Whether the rollout is graded is the configured policy's decision, not the adapter's.
            decision = choose_stage_outcome(
                failure_kind=provider_failure.failure_kind,
                http_status=provider_failure.http_status,
                stage="agent",
                config=self.config.outcome_policy,
            )
            failure = agent_route_failure(provider_failure, operation="agent.sandbox_exec", stage="agent")
            failure.__cause__ = provider_failure
            if decision.action != OutcomeAction.COUNT_AS_ZERO:
                raise failure
            observe_failure(failure)
            results_dir = Path(self.config.artifacts_dir) / sha256((rollout_id or uuid4().hex).encode()).hexdigest()
            results_dir.mkdir(parents=True, exist_ok=True)
            context["length_limited"] = False
            context["execution"] = {
                "pi_failed": True,
                "pi_exit_code": None,
                "pi_error_type": None,
                "pi_results_dir": str(results_dir),
                "failure_outcome_action": decision.action,
                "failure_outcome_attribution": decision.attribution,
                "failure_kind": provider_failure.failure_kind,
                "failure_reason": provider_failure.failure_reason,
                **FailureReportRow.fields(failure_report_for(failure)),
            }
            output, usage = parse_pi_events("")
            return output, usage, self.config.model, []
        root = Path(self.config.artifacts_dir) / sha256((rollout_id or uuid4().hex).encode()).hexdigest()
        root.mkdir(parents=True, exist_ok=True)
        # Export failures propagate; retrying such a request must not become a scored zero.
        await sandbox.download(stdout_path, root / "stdout.jsonl")
        await sandbox.download(stderr_path, root / "stderr.log")
        await sandbox.download(remote + "/events.jsonl", root / "events.jsonl")
        stdout = (root / "stdout.jsonl").read_text(errors="replace")
        error_type = error_type or getattr(result, "error_type", None)
        return_code = getattr(result, "return_code", None)
        if return_code == MCP_SETUP_ERROR_EXIT_CODE and mcp:
            raise RuntimeError("Required Gym MCP tools could not be initialized")
        events = []
        lines = (root / "events.jsonl").read_text().split("\n")
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # A killed capture process may leave its last write incomplete. Keep the
                # raw artifact, but only tolerate an unterminated tail on failed execution.
                if index != len(lines) - 1 or (not error_type and return_code == 0):
                    raise
                LOG.warning("Ignoring incomplete trailing Pi event after failed execution: %s", root / "events.jsonl")
        if error_type or return_code != 0:
            events.append(
                (
                    time(),
                    {
                        "type": "_ng_process_exit",
                        "return_code": return_code or 1,
                        "timed_out": "timeout" in (error_type or "").lower()
                        or (not error_type and return_code == 124),
                    },
                )
            )
        terminal_messages = next(
            (event.get("messages", []) for _, event in reversed(events) if event.get("type") == "agent_end"), []
        )
        stop_reason = next(
            (
                message.get("stopReason")
                for message in reversed(terminal_messages if isinstance(terminal_messages, list) else [])
                if isinstance(message, dict) and message.get("role") == "assistant"
            ),
            None,
        )
        context["length_limited"] = stop_reason == "length"
        context["execution"] = {
            # Scoring must not depend on whether observation collection is enabled.
            "pi_failed": bool(error_type or return_code != 0 or stop_reason in {"length", "error", "aborted"}),
            "pi_exit_code": return_code,
            "pi_error_type": error_type,
            "pi_results_dir": str(root),
        }
        output, usage = parse_pi_events(stdout)
        return output, usage, self.config.model, events

    @staticmethod
    def _sandbox_facts(sandbox: AsyncSandbox) -> dict[str, str | None]:
        handle = getattr(sandbox, "_handle", None)
        provider = getattr(handle, "provider_name", None)
        sandbox_id = getattr(handle, "sandbox_id", None)
        return {
            "provider": provider if isinstance(provider, str) else None,
            "sandbox_id": sandbox_id if isinstance(sandbox_id, str) else None,
        }

    @classmethod
    def _untyped_exec_failure(cls, sandbox: AsyncSandbox, error: Exception) -> NgFailureEvent:
        """The sandbox event for an exec that raised something other than a typed provider failure.

        A raised timeout is a provider request that timed out, whose remote outcome is unknown: it is not Pi's
        command deadline, which a provider returns as a result. Anything else is an unclassified request failure.
        """
        code = getattr(getattr(error, "error", None), "code", None)
        timed_out = isinstance(error, TimeoutError)
        return sandbox_failure(
            operation="exec",
            failure_kind=PROVIDER_IO_TIMEOUT if timed_out else PROVIDER_REQUEST_FAILED,
            reason="Sandbox exec request timed out; remote outcome is unknown" if timed_out else None,
            provider_status_reason=code if isinstance(code, str) and code else None,
            http_status=error_to_http_status(error),
            # The exception text may carry request or response content; record its type and status only.
            foreign_error=sdk_error_details(error, message=""),
            cause=error,
            **cls._sandbox_facts(sandbox),
        )

    @classmethod
    def _unclassified_result_failure(cls, sandbox: AsyncSandbox, result: Any) -> NgFailureEvent | None:
        """The sandbox event for a returned result that is neither Pi's outcome nor a native command deadline.

        A returned ``timeout`` (or exit 124) is a protocol result and stays gradable. A provider error type, or the
        ``-1`` sentinel, says the provider could not report how the command ended. Neither is an observed process
        exit, so no exit code is recorded, and the error type is the provider's reason, not an exception class.
        """
        error_type = getattr(result, "error_type", None)
        if result is None or error_type == "timeout" or not (error_type or getattr(result, "return_code", None) == -1):
            return None
        return sandbox_failure(
            operation="exec",
            failure_kind=PROVIDER_COMMAND_OUTCOME_UNKNOWN,
            reason="Sandbox exec returned a result that does not establish the command outcome",
            provider_status_reason=error_type if isinstance(error_type, str) and error_type else None,
            **cls._sandbox_facts(sandbox),
        )

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        raise ValueError("PiSandboxedAgent requires /run to seed tools and allocate its sandbox")

    async def run(self, request: Request, body: PiAgentRunRequest) -> PiSandboxedAgentVerifyResponse:
        async with self.sem:
            seeded = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=request.cookies,
            )
            await raise_for_status(seeded)
            cookies = request.cookies | seeded.cookies
            seed = await get_response_json(seeded)
            mcp = await seed_mcp_servers(
                self.server_client,
                self.config.tool_servers,
                body,
                cookies,
                timeout_s=self.config.timeout,
                require_reachable=self.config.network_access != "inherit",
            )
            sandbox = await self._start_sandbox(seed)
            rollout_id = self.rollout_id_from_run(body) or uuid4().hex
            context = {"sandbox": sandbox, "body": body, "mcp": mcp}
            token = _RUN.set(context)
            try:
                episode = await self._create_episode(body.responses_create_params, rollout_id=rollout_id)
                if context["length_limited"]:
                    episode.response.status = "incomplete"
                    episode.response.incomplete_details = IncompleteDetails(reason="max_output_tokens")
                observations = episode.observations
                observations.gaps = [g for g in observations.gaps if g.code != "no_sandbox_runtime"]
                execution = context["execution"]
                failed = execution["pi_failed"] or any(
                    isinstance(r, AgentInvocation) and r.status in {"failed", "incomplete"}
                    for r in observations.records
                )
                execution["pi_failed"] = failed
                timed_out = "timeout" in (execution["pi_error_type"] or "").lower() or (
                    not execution["pi_error_type"] and execution["pi_exit_code"] == 124
                )
                handle = sandbox._handle
                observations.records.append(
                    SandboxObservation(
                        role="agent",
                        sandbox_id=handle.sandbox_id,
                        provider=handle.provider_name,
                        outcome="timeout" if timed_out else "failed" if failed else "completed",
                        exit_code=execution["pi_exit_code"],
                        error_type=execution["pi_error_type"],
                    )
                )
                for record in observations.records:
                    if isinstance(record, ToolCallObservation):
                        record.sandbox_id = handle.sandbox_id
                metadata = execution | {"ng_agent_observations": observations.model_dump(mode="json")}
                payload = body.model_dump(mode="json") | {"response": episode.response.model_dump(mode="json")}
                root = Path(execution["pi_results_dir"])
                (root / "generation.json").write_text(json.dumps(payload | metadata))
                if "failure_outcome_action" in execution:
                    # The agent-stage policy already counted this failure as a zero. Apply it here: a second grading
                    # call on an empty answer could be masked or unavailable and would revoke the decision.
                    result = payload | {"reward": 0.0, "mask_sample": False}
                else:
                    result = await verify_agent_response(
                        self.server_client,
                        self.config.resources_server,
                        body,
                        episode.response,
                        cookies,
                        force_zero_reward=failed and self.config.execution_failure_reward_zero,
                    )
                return PiSandboxedAgentVerifyResponse.model_validate(
                    result
                    | metadata
                    | {
                        "turns_used": sum(getattr(i, "type", None) == "message" for i in episode.response.output),
                        "finished_naturally": not failed,
                    }
                )
            finally:
                _RUN.reset(token)
                # Connected resource-owned sandboxes remain alive through verification, just like OpenCode.
                try:
                    await sandbox.stop()
                except Exception:
                    LOG.exception("Failed to stop Pi sandbox")


if __name__ == "__main__":
    PiSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = PiSandboxedAgent.run_webserver()
