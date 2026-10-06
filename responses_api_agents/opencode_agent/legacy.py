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

import asyncio
import json
import sqlite3
import sys
from asyncio import Semaphore
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from shlex import quote
from time import time
from traceback import format_exc
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from anyio import CancelScope
from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field, FilePath

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
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
    TrajectoryRecord,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, create_provider
from nemo_gym.sandbox.agent_tools import (
    restricted_network_policy,
    sandbox_server_url,
    seed_mcp_servers,
    verify_agent_response,
)
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    raise_for_status,
)
from responses_api_agents.opencode_agent.artifacts import (
    opencode_export_usages,
    parse_opencode_export,
    parse_opencode_observations,
)
from responses_api_agents.opencode_agent.observability import scope_opencode_trajectory


_NATIVE_SESSION_KEY = "nemo_gym_opencode_native_session"
_ASSISTANT_MESSAGE_PLUGIN = Path(__file__).parents[1] / "opencode_sandboxed_agent" / "assistant_message_header.js"
_REMOTE_ASSISTANT_MESSAGE_PLUGIN = "/tmp/nemo-gym-opencode-assistant-message-header.js"


class LegacyOpenCodeAgentConfig(BaseResponsesAPIAgentConfig):
    """Configuration retained for unmigrated Resources consumers only."""

    resources_server: ResourcesServerRef | None = None
    model_server: ModelServerRef

    opencode_version: str = "1.17.11"
    remote_opencode_install_script_path: Optional[str] = None
    remote_opencode_binary_path: Optional[str] = None
    remote_opencode_musl_binary_path: Optional[str] = None
    local_ripgrep_binary_path: FilePath | None = None
    opencode_config: Dict[str, Any] = Field(default_factory=dict)
    opencode_max_context_window: int = 262144
    concurrency: int = Field(default=64, gt=0)
    preinstalled_opencode: bool = False
    execution_failure_reward_zero: bool = False
    output_token_policy: Literal["fixed", "remaining_context"] = "fixed"
    network_access: Literal["inherit", "model_only", "model_and_tools"] = "inherit"
    tool_servers: List[ResourcesServerRef] = Field(default_factory=list)
    artifacts_dir: Optional[str] = None
    opencode_model_call_timeout: Optional[int] = None

    # Sandbox config
    sandbox_provider: str = "sandbox"
    sandbox_config: Dict[str, Any] = Field(default_factory=dict)
    sandbox_timeout: float = 10800

    debug: bool = False


class LegacyOpenCodeAgentRunRequest(BaseRunRequest):
    # Allow for benchmark params to propagate properly
    model_config = ConfigDict(extra="allow")


def _build_remote_opencode_install_command(
    install_script_path: str,
    binary_path: str,
    musl_binary_path: str,
) -> str:
    """Build the invocation for the network-free, libc-aware cached installer."""
    return (
        f"bash {quote(install_script_path)} "
        f"--glibc-binary {quote(binary_path)} "
        f"--musl-binary {quote(musl_binary_path)}"
    )


def _extract_opencode_session_id(session_list_stdout: str) -> str:
    """Return the newest OpenCode session ID from ``session list`` JSON output."""
    sessions = json.loads(session_list_stdout)
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("OpenCode did not return any sessions")

    session_id = sessions[0].get("id") if isinstance(sessions[0], dict) else None
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("The newest OpenCode session does not have a valid ID")
    return session_id


def _read_opencode_child_messages(db_path: Path, root_session_id: str) -> list[dict[str, Any]]:
    """Read each descendant message once, excluding the separately exported root."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            """
            with recursive descendants(id) as (
                select id from session where id = ?
                union
                select session.id from session join descendants on session.parent_id = descendants.id
            )
            select message.data from message join descendants on message.session_id = descendants.id
            where message.session_id != ?
            order by message.time_created, message.id
            """,
            (root_session_id, root_session_id),
        ).fetchall()
    finally:
        con.close()
    return [{"info": json.loads(row[0])} for row in rows]


class LegacyOpenCodeAgentVerifyRequest(BaseVerifyRequest):
    # Allow for benchmark params to propagate properly
    model_config = ConfigDict(extra="allow")


class LegacyOpenCodeAgentVerifyResponse(BaseVerifyResponse):
    # Allow for benchmark params to propagate properly
    model_config = ConfigDict(extra="allow")

    opencode_results_fpath: str
    opencode_run_stdout: str
    opencode_run_stderr: str
    opencode_finished: bool
    opencode_export_found: bool
    opencode_exit_code: Optional[int] = None
    opencode_error_type: Optional[str] = None
    opencode_failed: bool = False
    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class LegacyOpenCodeAgent(SimpleResponsesAPIAgent):
    """Compatibility bridge for resources that use agent-owned /run orchestration."""

    ray_enabled = False
    config: LegacyOpenCodeAgentConfig

    def _native_session_marker(self, request: Request) -> str | None:
        try:
            session = request.session
        except (AssertionError, AttributeError):
            return None
        if not isinstance(session, Mapping) or _NATIVE_SESSION_KEY not in session:
            return None
        marker = session[_NATIVE_SESSION_KEY]
        if not isinstance(marker, str) or not marker:
            raise HTTPException(409, "Invalid native OpenCode session marker")
        return marker

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)

        self._sem = Semaphore(self.config.concurrency)
        self._sandbox_id_to_sandbox: Dict[str, AsyncSandbox] = dict()
        self._sandbox_id_to_run_result: Dict[str, Dict[str, Any]] = dict()

    async def _start_sandbox(self, sandbox_id: Optional[str] = None, workdir: Optional[str] = None) -> AsyncSandbox:
        global_config_dict = get_global_config_dict()
        resolved_sandbox_provider = create_provider(
            resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        )
        provider_default_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)

        if sandbox_id:
            if self.config.network_access != "inherit":
                raise ValueError("Cannot verify network policy on an externally supplied sandbox")
            sandbox = await AsyncSandbox.connect(
                {"sandbox_id": sandbox_id, "workdir": workdir}, provider=resolved_sandbox_provider
            )
            return sandbox

        if self.config.debug:
            print("Creating new sandbox since one wasn't provided", file=sys.stderr)

        resources = SandboxResources.from_mapping(self.config.sandbox_config.get("resources", {}))
        # TODO @bxyu-nvidia: Refactor this after swapping to PTY as this should be set on the SWE Bench resources server side
        env = cpu_cap_env(resources.cpu) if self.config.sandbox_config.get("derive_cpu_env", True) else {}
        env |= dict(self.config.sandbox_config.get("env", {}))  # explicit keys win over the derived caps

        # TODO @bxyu-nvidia: Refactor this after Hemil's swap from Python dataclass to Pydantic BaseModel
        sandbox_spec = SandboxSpec(
            image=self.config.sandbox_config.get("image", "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907"),
            ttl_s=self.config.sandbox_config.get("ttl_s", None),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s", None),
            workdir=self.config.sandbox_config.get("workdir"),
            env=env,
            files=dict(self.config.sandbox_config.get("files", {})),
            metadata=provider_default_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {
                "nemo_gym_agent": self.config.name,
            },
            resources=resources,
            entrypoint=self.config.sandbox_config.get("entrypoint"),
            provider_options=deepcopy(self.config.sandbox_config.get("provider_options", {})),
        )

        if self.config.network_access != "inherit":
            urls = [sandbox_server_url(self.config.model_server.name, require_reachable=True)]
            if self.config.network_access == "model_and_tools":
                if not self.config.tool_servers:
                    raise ValueError("model_and_tools requires tool_servers")
                urls.extend(
                    sandbox_server_url(server.name, require_reachable=True) for server in self.config.tool_servers
                )
            sandbox_spec.provider_options["network_policy"] = restricted_network_policy(
                resolved_sandbox_provider.name, urls
            )
        sandbox = AsyncSandbox(resolved_sandbox_provider)
        await sandbox.start(sandbox_spec)

        return sandbox

    def _runtime_plugins(self) -> list[str]:
        plugins = []
        if self.config.output_token_policy == "remaining_context":
            plugins.append("remaining-context.js")
        if self.config.tool_servers:
            plugins.append("required-mcp.js")
        return plugins

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
        if "timeout" in normalized_error or (not normalized_error and return_code == 124):
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

    async def _create_opencode_config(self, request: Request) -> Dict[str, Any]:
        base_url = (
            self.base_url_for_run(
                base_url=sandbox_server_url(
                    self.config.model_server.name, require_reachable=self.config.network_access != "inherit"
                ),
                body=await request.json(),
            )
            + "/v1"
        )
        config = {
            "model": "nemo_gym/dummy_model",
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "nemo_gym": {
                    # TODO @bxyu-nvidia: We should use @ai-sdk/openai here but there is some /v1/responses streaming error.
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {
                        "baseURL": base_url,
                        "apiKey": "dummy_key",  # pragma: allowlist secret
                        "chunkTimeout": int(self.config.sandbox_timeout * 1000),
                        "timeout": self.config.opencode_model_call_timeout
                        if self.config.opencode_model_call_timeout is not None
                        else False,  # milliseconds
                    },
                    "models": {
                        "dummy_model": {
                            "temperature": True,
                            "limit": {
                                "context": self.config.opencode_max_context_window,
                                "input": self.config.opencode_max_context_window,
                                # See the OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX flag below for more information.
                                "output": self.config.opencode_max_context_window,
                            },
                        },
                    },
                }
            },
        }

        def merge(base, override):
            for key, value in override.items():
                if isinstance(value, dict) and isinstance(base.get(key), dict):
                    merge(base[key], value)
                else:
                    base[key] = deepcopy(value)

        merge(config, self.config.opencode_config)
        config.setdefault("plugin", []).extend(f"file:///tmp/nemo-gym-{name}" for name in self._runtime_plugins())
        rollout_mcp = getattr(request.state, "_ng_opencode_mcp", None)
        if isinstance(rollout_mcp, dict):
            config.setdefault("mcp", {}).update(rollout_mcp)
        return config

    async def _seed_tool_servers(self, request: Request, body: LegacyOpenCodeAgentRunRequest) -> Dict[str, Any]:
        entries = await seed_mcp_servers(
            self.server_client,
            self.config.tool_servers,
            body,
            request.cookies,
            timeout_s=self.config.sandbox_timeout,
            require_reachable=self.config.network_access != "inherit",
        )
        return {name: {"type": "remote", **entry} for name, entry in entries.items()}

    def _opencode_export_to_usages(self, opencode_export: Dict[str, Any]) -> List[NeMoGymResponseUsage]:
        return opencode_export_usages(opencode_export)

    def _opencode_export_to_output_items(self, opencode_export: Dict[str, Any]) -> List[NeMoGymResponseOutputItem]:
        return parse_opencode_export(opencode_export)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        if self._native_session_marker(request) is not None:
            raise HTTPException(409, "Native OpenCode sessions cannot enter the legacy sandbox bridge")
        if self.config.tool_servers and not isinstance(getattr(request.state, "_ng_opencode_mcp", None), dict):
            raise ValueError("Configured tool servers require a seeded /run request")
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

        opencode_debug_str = ""
        if self.config.debug:
            opencode_debug_str = "--print-logs --log-level DEBUG"

        opencode_thinking_str = "--thinking"

        if self.config.preinstalled_opencode:
            install_str = f'test "$(opencode --version)" = {quote(self.config.opencode_version)}'
        elif self.config.remote_opencode_binary_path and self.config.remote_opencode_install_script_path:
            if self.config.remote_opencode_musl_binary_path:
                install_str = _build_remote_opencode_install_command(
                    install_script_path=self.config.remote_opencode_install_script_path,
                    binary_path=self.config.remote_opencode_binary_path,
                    musl_binary_path=self.config.remote_opencode_musl_binary_path,
                )
            else:
                install_str = (
                    f"bash {quote(self.config.remote_opencode_install_script_path)} "
                    f"--binary {quote(self.config.remote_opencode_binary_path)}"
                )
        else:
            print(
                "Downloading and installing OpenCode in the sandbox. Please consider mounting or uploading the appropriate OpenCode binary instead!",
                file=sys.stderr,
            )
            install_str = f"""installer=$(mktemp) && curl -fL -o "$installer" https://opencode.ai/install \
        && echo "Downloaded OpenCode installer to $installer" \
        && VERSION={self.config.opencode_version} bash "$installer\""""

        effective_config = await self._create_opencode_config(request)
        for name in self._runtime_plugins():
            await sandbox.upload(
                (Path(__file__).parents[1] / "opencode_sandboxed_agent" / name), f"/tmp/nemo-gym-{name}"
            )
        build_agent = effective_config.setdefault("agent", {}).setdefault("build", {})
        for name in ("temperature", "top_p"):
            value = getattr(body, name, None)
            if value is not None:
                build_agent[name] = value
        if self._model_call_capture_enabled():
            await sandbox.upload(_ASSISTANT_MESSAGE_PLUGIN, _REMOTE_ASSISTANT_MESSAGE_PLUGIN)
            effective_config["plugin"] = [
                *effective_config.get("plugin", []),
                f"file://{_REMOTE_ASSISTANT_MESSAGE_PLUGIN}",
            ]
        opencode_config_content = json.dumps(effective_config)
        observation_invocation_id = getattr(request.state, "_ng_observation_invocation_id", None)
        observation_invocation_id = observation_invocation_id if isinstance(observation_invocation_id, str) else None
        collect_observations = observation_invocation_id is not None
        xdg_home_str = ""
        remote_data_home = None
        if collect_observations:
            remote_data_home = f"/tmp/nemo-gym-opencode-{uuid4().hex}"
            xdg_home_str = f"XDG_DATA_HOME={remote_data_home}"

        # OpenCode's glob/grep tools otherwise download rg inside the sandbox.
        ripgrep_remote_path = None
        ripgrep_install_str = ""
        if self.config.local_ripgrep_binary_path is not None:
            ripgrep_remote_path = f"/tmp/nemo-gym-ripgrep-{uuid4().hex}"
            # Uploads may be root-owned: copy as the execution user; teardown removes the source.
            ripgrep_install_str = (
                '&& mkdir -p "$HOME/.opencode/bin" '
                f'&& cp {quote(ripgrep_remote_path)} "$HOME/.opencode/bin/rg" '
                '&& chmod 0755 "$HOME/.opencode/bin/rg" '
                '&& "$HOME/.opencode/bin/rg" --version'
            )

        # @bxyu-nvidia: Regarding `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=1000000000` below:
        # OpenCode defaults to 32k here https://github.com/anomalyco/opencode/blob/58a99916bb96edf5cf605dc03e1be1e4bacf9ff7/packages/opencode/src/provider/transform.ts#L21
        # and there is no way to set it to null.
        # Here, we set an exorbitantly high number that cannot ever be reached.
        # In future versions of OpenCode, this can be directly passed via maxOutputTokens in the limit config above https://github.com/anomalyco/opencode/blob/1b18a50418f730aca32630ccfcde850f2b5fc360/packages/opencode/src/provider/transform.ts#L1418
        command = f"""
        echo "Shell: $SHELL" \
        && {install_str} \
        {ripgrep_install_str} \
        && export PATH=$HOME/.opencode/bin:$PATH \
        && echo "Installed OpenCode" \
        && rm -f /tmp/nemo-gym-mcp-setup-error \
        && NEMO_GYM_REQUIRED_MCP_SERVERS={quote(json.dumps([s.name for s in self.config.tool_servers]))} OPENCODE_CONFIG_CONTENT={quote(opencode_config_content)} OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=1000000000 {xdg_home_str} \
            opencode run --title "NG dummy title" {opencode_debug_str} {opencode_thinking_str} -- {quote(query)} \
        && echo "OpenCode run finished"
        """

        if self.config.debug:
            print("Starting OpenCode (runtime configuration omitted to protect credentials)", file=sys.stderr)

        if ripgrep_remote_path is not None:
            await sandbox.upload(self.config.local_ripgrep_binary_path, ripgrep_remote_path)

        run_error_type = None
        try:
            result = await sandbox.exec(
                command=command,
                timeout_s=self.config.sandbox_timeout,
            )
        except Exception as exc:
            result = None
            run_error_type = type(exc).__name__
            print("OpenCode exec hit error.", format_exc(), file=sys.stderr)

        if self.config.debug and result:
            print("OpenCode install and run stdout:\n", result.stdout, file=sys.stderr)
            print("OpenCode install and run stderr:\n", result.stderr, file=sys.stderr)

        if self.config.tool_servers:
            mcp_check = await sandbox.exec(command="test ! -f /tmp/nemo-gym-mcp-setup-error", timeout_s=30)
            if mcp_check.return_code != 0 or mcp_check.error_type:
                raise RuntimeError("Required Gym MCP tools could not be initialized")

        export_fname = "export.json"
        # Kept outside the sandbox workdir on purpose: SWE-bench-style environments set the workdir
        # to the git repo, and resources servers extract the model patch with `git add -N . && git
        # diff`, which would sweep this transcript into the patch.
        export_remote_fpath = f"/tmp/opencode_{export_fname}"
        try:
            session_env = {"XDG_DATA_HOME": remote_data_home} if remote_data_home is not None else None
            session_list_result = await sandbox.exec(
                command="export PATH=$HOME/.opencode/bin:$PATH && opencode session list --format json",
                env=session_env,
                timeout_s=self.config.sandbox_timeout,
            )
            if session_list_result.return_code != 0:
                raise RuntimeError(f"Failed to list OpenCode sessions: {session_list_result}")
            session_id = _extract_opencode_session_id(session_list_result.stdout or "")
            export_result = await sandbox.exec(
                command=(
                    "export PATH=$HOME/.opencode/bin:$PATH "
                    f"&& opencode export {quote(session_id)} > {quote(export_remote_fpath)}"
                ),
                env=session_env,
                timeout_s=self.config.sandbox_timeout,
            )
        except Exception:
            raise RuntimeError("Failed to export OpenCode results") from None
        if export_result.return_code != 0 or export_result.error_type:
            raise RuntimeError("OpenCode export command failed")
        if self.config.debug and export_result:
            print("Export stdout:\n", export_result.stdout, file=sys.stderr)
            print("Export stderr:\n", export_result.stderr, file=sys.stderr)

        results_root = (
            Path(self.config.artifacts_dir) if self.config.artifacts_dir else Path(__file__).parent / "results"
        )
        results_dir = results_root / request.session[SESSION_ID_KEY]
        results_dir.mkdir(parents=True, exist_ok=True)
        results_local_fpath = results_dir / export_fname
        results_local_fpath.unlink(missing_ok=True)
        await sandbox.download(export_remote_fpath, results_local_fpath)

        observations = None
        trajectory = (
            TrajectoryRecord(task_id="", rollout_id=observation_invocation_id) if collect_observations else None
        )
        child_usages = []
        # Usage includes descendants even when detailed observation collection is disabled.
        if collect_observations or session_id is not None:
            snapshot_remote_fpath = (
                f"{remote_data_home}/opencode/nemo-gym-observations.db"
                if remote_data_home is not None
                else f"/tmp/nemo-gym-observations-{uuid4().hex}.db"
            )
            observations_local_fpath = results_dir / "opencode.db"
            observations_local_fpath.unlink(missing_ok=True)
            try:
                # Release channels and OPENCODE_DB can change the database filename.
                database_path_result = await sandbox.exec(
                    command="export PATH=$HOME/.opencode/bin:$PATH && opencode db path",
                    env=session_env,
                )
                observations_remote_fpath = (database_path_result.stdout or "").strip()
                if (
                    database_path_result.return_code != 0
                    or database_path_result.error_type is not None
                    or not observations_remote_fpath
                ):
                    raise RuntimeError(f"OpenCode database path lookup failed: {database_path_result.stderr}")
                snapshot_script = (
                    "import sqlite3,sys;"
                    "source=sqlite3.connect(f'file:{sys.argv[1]}?mode=ro',uri=True);"
                    "destination=sqlite3.connect(sys.argv[2]);"
                    "source.backup(destination);destination.close();source.close()"
                )
                snapshot_result = await sandbox.exec(
                    command=(
                        f"python3 -c {quote(snapshot_script)} "
                        f"{quote(observations_remote_fpath)} {quote(snapshot_remote_fpath)}"
                    ),
                    timeout_s=self.config.sandbox_timeout,
                )
                if snapshot_result.return_code != 0 or snapshot_result.error_type is not None:
                    raise RuntimeError(f"OpenCode database snapshot failed: {snapshot_result.stderr}")
                await sandbox.download(snapshot_remote_fpath, observations_local_fpath)
                if session_id is not None:
                    child_messages = await asyncio.to_thread(
                        _read_opencode_child_messages, observations_local_fpath, session_id
                    )
                    child_usages = self._opencode_export_to_usages({"messages": child_messages})
                if collect_observations:
                    observations = parse_opencode_observations(
                        observations_local_fpath,
                        observation_invocation_id,
                        trajectory,
                        model_ref=self.config.model_server,
                    )
            except Exception:
                print("Failed to capture OpenCode session usage or observations", format_exc(), file=sys.stderr)
                if collect_observations:
                    trajectory.gaps.append(ObservationGap(code="turns_unavailable"))
                    observations = AgentObservationBundle(
                        source="opencode",
                        records=[AgentInvocation(invocation_id=observation_invocation_id)],
                        gaps=[
                            ObservationGap(code="agent_artifact_unavailable"),
                            ObservationGap(code="agent_transcript_unavailable"),
                            ObservationGap(code="model_call_ownership_unavailable"),
                            ObservationGap(code="observation_capture_failed"),
                        ],
                    )
            finally:
                observations_local_fpath.unlink(missing_ok=True)

        opencode_export = dict()
        if results_local_fpath.exists():
            opencode_export = json.loads(results_local_fpath.read_text().strip() or "{}")

        if not opencode_export:
            raise RuntimeError("OpenCode export did not contain a transcript")
        output = []
        usage = None
        opencode_export_found = False
        if opencode_export:
            opencode_export_found = True
            # Assume only one input message. May change with a system/developer message later on.
            output = self._opencode_export_to_output_items(opencode_export)[1:]
            usage = NeMoGymResponseUsage.sum_from_list(
                [*self._opencode_export_to_usages(opencode_export), *child_usages]
            )

        result_stdout = (result.stdout if result else "") or ""
        result_stderr = (result.stderr if result else "") or ""
        opencode_finished = False
        std_out_split = result_stdout.rsplit("Shell: ", maxsplit=1)
        if len(std_out_split) > 1:
            opencode_finished = "OpenCode run finished" in std_out_split[1]

        assistant_infos = [
            message.get("info", {})
            for message in opencode_export.get("messages", [])
            if message.get("info", {}).get("role") == "assistant"
        ]
        length_limited = bool(assistant_infos and assistant_infos[-1].get("finish") == "length")
        terminal_error = assistant_infos[-1].get("error") if assistant_infos else None
        if terminal_error and not run_error_type:
            run_error_type = (
                terminal_error.get("name", "OpenCodeError") if isinstance(terminal_error, dict) else "OpenCodeError"
            )

        if collect_observations and observations is not None:
            agent_sandbox_observation = self._agent_sandbox_observation(
                sandbox=sandbox,
                return_code=getattr(result, "return_code", None),
                error_type=getattr(result, "error_type", None) or run_error_type,
                finished=opencode_finished,
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
                        record.status = "incomplete" if status == "completed" and length_limited else status
            observations.records.append(agent_sandbox_observation)
            observations.gaps.append(ObservationGap(code="sandbox_lifecycle_timing_unavailable"))

        run_result = {
            "opencode_failed": bool(run_error_type or getattr(result, "error_type", None))
            or getattr(result, "return_code", None) != 0
            or not opencode_finished
            or not opencode_export_found
            or length_limited,
            "opencode_exit_code": getattr(result, "return_code", None),
            "opencode_error_type": run_error_type or getattr(result, "error_type", None),
            "opencode_results_fpath": str(results_local_fpath) if opencode_export_found else "",
            "opencode_run_stdout": result_stdout,
            "opencode_run_stderr": result_stderr,
            "opencode_export_found": opencode_export_found,
            "opencode_finished": opencode_finished,
        }
        if collect_observations:
            run_result["_ng_agent_observations"] = observations
            run_result["_ng_trajectory"] = trajectory
        self._sandbox_id_to_run_result[request.cookies["sandbox_id"]] = run_result

        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=body.model or self.config.model_server.name,
            object="response",
            output=output,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=usage,
            status="incomplete" if length_limited else None,
            incomplete_details={"reason": "max_output_tokens"} if length_limited else None,
        )
        receipt = {
            "response": response.model_dump(mode="json"),
            "execution": {key: value for key, value in run_result.items() if not key.startswith("_ng_")},
        }
        pending = results_dir / "generation.json.partial"
        pending.write_text(json.dumps(receipt))
        pending.replace(results_dir / "generation.json")
        return response

    async def run(self, request: Request, body: LegacyOpenCodeAgentRunRequest) -> LegacyOpenCodeAgentVerifyResponse:
        async with self._sem:
            return await self._run(request, body)

    async def _run(self, request: Request, body: LegacyOpenCodeAgentRunRequest) -> LegacyOpenCodeAgentVerifyResponse:
        if self._native_session_marker(request) is not None:
            raise HTTPException(409, "Native OpenCode sessions must use EnvironmentServer /run")
        if self.config.resources_server is None:
            raise HTTPException(
                422, "Submit native episodes to EnvironmentServer /run; legacy /run requires resources_server"
            )
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

        request.state._ng_opencode_mcp = await self._seed_tool_servers(request, body)

        # @bxyu-nvidia: "sandbox_handle" comes from resources_servers/swebench/app.py
        # Once we graduate to use the sandbox server, this will be in a generic seed_session type that can be model validated.
        seed_session_result = await seed_session_response.json()
        sandbox = await self._start_sandbox(
            sandbox_id=seed_session_result.get("sandbox_handle"),
            workdir=seed_session_result.get("workdir"),
        )
        self._sandbox_id_to_sandbox[request.session[SESSION_ID_KEY]] = sandbox

        # Propagating the sandbox handle
        cookies["sandbox_id"] = session_key

        request._cookies = cookies
        request.state._ng_observation_invocation_id = rollout_id
        observations = None
        try:
            response = await self.responses(request, body.responses_create_params)
            run_result = self._sandbox_id_to_run_result.get(session_key, {}).copy()
            observations = run_result.pop("_ng_agent_observations", None)
            trajectory = run_result.pop("_ng_trajectory", None)
            response_dict = await verify_agent_response(
                self.server_client,
                self.config.resources_server,
                body,
                response,
                cookies,
                force_zero_reward=self.config.execution_failure_reward_zero
                and run_result.get("opencode_failed", False),
            )
        finally:
            del request.state._ng_observation_invocation_id
            del request.state._ng_opencode_mcp
            # A server cancels its handler when the caller disconnects, and OpenCode keeps running
            # in its pod regardless: only stopping the sandbox ends it. Shielded because the
            # cancellation is re-delivered at every await until the handler exits, so an unshielded
            # stop would itself be cancelled and leave the pod generating until its TTL.
            with CancelScope(shield=True):
                try:
                    await sandbox.stop()
                except Exception:
                    print("Failed to stop sandbox", format_exc(), file=sys.stderr)
                finally:
                    self._sandbox_id_to_sandbox.pop(session_key, None)
                    self._sandbox_id_to_run_result.pop(session_key, None)

        response_dict |= run_result
        if trajectory is not None:
            response_dict["ng_trajectory"] = scope_opencode_trajectory(trajectory, body, rollout_id)
        raw_verifier_sandbox_observation = response_dict.pop("verifier_sandbox_observation", None)
        if rollout_id is not None:
            if observations is None:
                observations = AgentObservationBundle(
                    source="opencode",
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
        return LegacyOpenCodeAgentVerifyResponse.model_validate(response_dict)
