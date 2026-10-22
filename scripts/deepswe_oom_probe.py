# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, no-inference DeepSWE agent OOM probe against one real OpenSandbox pod.

This is a manual integration probe, not a benchmark sample. It uses the resolved
Gym config for the candidate run, overrides only disposable-probe settings
(agent memory, TTL, ready timeout, create/command retries, and TTL renewal),
and replaces only the primary ``opencode run`` command after normal OpenCode
setup. The real provider must report the *same* sandbox as OOMKilled.
An allocator process exiting 137 while the pod survives is a failed probe.

Run only from the exact pinned Gym checkout, after checking the source and task
pins, cluster capacity, and that no DeepSWE writer or duplicate probe exists::

    python -m scripts.deepswe_oom_probe \
      --execute-real-oom-probe \
      --resolved-config /path/to/resolved-gym-config.yaml \
      --expected-config-sha256 <sha256-of-that-file> \
      --expected-gym-sha <40-character-commit> \
      --expected-probe-sha256 <sha256-of-this-script> \
      --tasks-dir /path/to/frozen-113-task-source \
      --expected-task-tree-sha256 <sha256-of-task-tree> \
      --task-id <one-canonical-task-id> \
      --expected-image <exact-task-image> \
      --artifacts-dir /new/empty/probe-directory

The config must contain the resolved ``sandbox``, ``deepswe_opencode_resources_server``,
``deepswe_opencode_sandboxed_agent``, and model-server blocks. This script never
starts a model server or sends a model request. Its local in-process transport
routes agent seed and verify requests to the real DeepSWE resources methods.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from fastapi import Request
from omegaconf import DictConfig, OmegaConf

import nemo_gym.global_config as global_config_module
from nemo_gym.failure_kinds import PROVIDER_OOM_KILLED
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.rollout_observability import NG_AGENT_SANDBOX_OBSERVATION_KEY, SandboxObservation
from nemo_gym.sandbox import AsyncSandbox, SandboxOOMKilledError
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.deepswe.app import (
    DeepSWEResourcesServer,
    DeepSWEResourcesServerConfig,
    DeepSWESeedSessionRequest,
    DeepSWEVerifyRequest,
)
from resources_servers.deepswe.task_store import DEEPSWE_SOURCE_REVISION, task_image
from responses_api_agents.opencode_sandboxed_agent.app import (
    OpenCodeSandboxedAgent,
    OpenCodeSandboxedAgentConfig,
    OpenCodeSandboxedAgentRunRequest,
    OpenCodeSandboxedAgentVerifyResponse,
)


RESOURCE_NAME = "deepswe_opencode_resources_server"
AGENT_NAME = "deepswe_opencode_sandboxed_agent"
AGENT_COMMAND_MARKER = 'opencode run --title "NG dummy title"'
SETUP_COMMAND_MARKER = 'echo "Installed OpenCode"'
PROBE_TTL_S = 300
PROBE_WATCHDOG_S = 90
PROBE_MEMORY_MIB = 2048


class ProbeNotValidated(RuntimeError):
    """The real provider did not prove a same-sandbox agent-execution OOM."""


@dataclass
class LocalResponse:
    payload: dict[str, Any]
    cookies: dict[str, str]
    ok: bool = True

    async def json(self) -> dict[str, Any]:
        return self.payload

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class LocalDeepSWETransport:
    """Transport boundary only; seed and verify are the real resources methods."""

    def __init__(self, resource: DeepSWEResourcesServer, request: Request) -> None:
        self.resource = resource
        self.request = request
        self.seeded_sandbox_id: str | None = None
        self.seeded_sandbox: AsyncSandbox | None = None
        self.seed_started_at: float | None = None
        self.verify_calls = 0

    async def post(
        self, *, server_name: str, url_path: str, json: dict[str, Any], cookies: dict[str, str]
    ) -> LocalResponse:
        if server_name != RESOURCE_NAME:
            raise ValueError(f"Unexpected resources server in OOM probe: {server_name!r}")
        if url_path == "/seed_session":
            if self.seeded_sandbox_id is not None:
                raise RuntimeError("OOM probe may seed only one sandbox")
            self.seed_started_at = monotonic()
            result = await self.resource.seed_session(self.request, DeepSWESeedSessionRequest.model_validate(json))
            self.seeded_sandbox_id = result.sandbox_handle
            session_id = self.request.session[SESSION_ID_KEY]
            self.seeded_sandbox = self.resource._agent_sessions[session_id].sandbox
            return LocalResponse(result.model_dump(mode="json"), dict(cookies))
        if url_path == "/verify":
            if self.seeded_sandbox_id is None or self.verify_calls:
                raise RuntimeError("OOM probe may verify exactly one seeded sandbox")
            result = await self.resource.verify(self.request, DeepSWEVerifyRequest.model_validate(json))
            self.verify_calls += 1
            return LocalResponse(result.model_dump(mode="json"), dict(cookies))
        raise ValueError(f"Unexpected resources route in OOM probe: {url_path!r}")


def allocator_command() -> str:
    """Touch at most 4 GiB in 16 MiB chunks inside the 2 GiB limited pod."""
    source = (
        "import time\n"
        "chunks = []\n"
        "for _ in range(256):\n"
        "    chunk = bytearray(16 * 1024 * 1024)\n"
        "    for offset in range(0, len(chunk), 4096):\n"
        "        chunk[offset] = 1\n"
        "    chunks.append(chunk)\n"
        "    time.sleep(0.02)\n"
        "time.sleep(5)\n"
    )
    return "exec python3 -c " + shlex.quote(source)


class AgentExecGate:
    """Delegate setup unchanged and replace exactly one primary agent exec."""

    def __init__(self, sandbox: AsyncSandbox, *, seeded_sandbox_id: str, seeded_at: float) -> None:
        if sandbox.sandbox_id != seeded_sandbox_id:
            raise ProbeNotValidated("Agent connected to a different sandbox than DeepSWE seeded")
        self._sandbox = sandbox
        self._real_exec = sandbox.exec
        self._seeded_at = seeded_at
        self.setup_completed = False
        self.agent_exec_started = False
        self.provider_oom: SandboxOOMKilledError | None = None

    @property
    def sandbox(self) -> AsyncSandbox:
        return self._sandbox

    async def exec(self, command: str, **kwargs: Any) -> Any:
        if not self.setup_completed:
            if (
                self.agent_exec_started
                or not command.strip().startswith('echo "Shell: $SHELL"')
                or SETUP_COMMAND_MARKER not in command
                or not command.strip().endswith("&& rm -f /tmp/nemo-gym-mcp-setup-error")
                or "opencode run" in command
            ):
                raise ProbeNotValidated("Unexpected command before OpenCode setup; refusing model inference")
            result = await self._real_exec(command, **kwargs)
            if result.return_code != 0 or result.error_type is not None:
                raise ProbeNotValidated("OpenCode setup failed before the OOM probe")
            self.setup_completed = True
            return result

        if self.agent_exec_started or AGENT_COMMAND_MARKER not in command:
            raise ProbeNotValidated("Unexpected post-setup command; refusing model inference")
        if monotonic() - self._seeded_at >= PROBE_TTL_S - PROBE_WATCHDOG_S:
            raise ProbeNotValidated("Too little sandbox TTL remains after setup for a bounded OOM probe")
        self.agent_exec_started = True
        try:
            result = await asyncio.wait_for(
                self._real_exec(allocator_command(), timeout_s=PROBE_WATCHDOG_S),
                timeout=PROBE_WATCHDOG_S,
            )
        except SandboxOOMKilledError as error:
            if error.sandbox_id != self._sandbox.sandbox_id:
                raise ProbeNotValidated("Provider OOM evidence names a different sandbox") from error
            self.provider_oom = error
            raise
        except TimeoutError as error:
            raise ProbeNotValidated("Agent exec watchdog expired without confirmed provider OOM") from error
        raise ProbeNotValidated(
            "Allocator returned without provider-confirmed sandbox OOM "
            f"(return_code={result.return_code}, error_type={result.error_type!r}); "
            "a child exit 137 is not validation"
        )


def assert_confirmed_oom_result(
    result: OpenCodeSandboxedAgentVerifyResponse, *, sandbox_id: str, gate: AgentExecGate, verify_calls: int
) -> None:
    """Require real agent and DeepSWE proof before accepting an OOM-zero result."""
    if not gate.setup_completed or not gate.agent_exec_started or gate.provider_oom is None:
        raise ProbeNotValidated("The agent exec did not produce real provider OOM evidence")
    if verify_calls != 1:
        raise ProbeNotValidated("DeepSWE did not verify exactly one seeded outcome")
    observation_json = (result.response.metadata or {}).get(NG_AGENT_SANDBOX_OBSERVATION_KEY)
    if not isinstance(observation_json, str):
        raise ProbeNotValidated("The agent did not forward an OOM observation")
    observation = SandboxObservation.model_validate_json(observation_json)
    if (
        observation.sandbox_id != sandbox_id
        or observation.role != "agent"
        or observation.phase != "agent_execution"
        or observation.outcome != "oom"
        or observation.error_type != SandboxOOMKilledError.__name__
        or result.failure_kind != PROVIDER_OOM_KILLED
        or result.reward != 0.0
        or result.mask_sample is not False
        or result.evaluation_completed is not False
        or result.model_patch_bytes != 0
        or result.verifier_exit_code is not None
        or result.opencode_error_type != SandboxOOMKilledError.__name__
    ):
        raise ProbeNotValidated("DeepSWE did not return one matched, unmasked agent-OOM zero without a verifier")


def load_probe_config(path: Path, *, expected_sha256: str) -> DictConfig:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Resolved Gym config SHA-256 differs from the reviewed file")
    config = OmegaConf.load(path)
    for block in ("sandbox", RESOURCE_NAME, AGENT_NAME):
        if block not in config:
            raise ValueError(f"Resolved Gym config lacks required block {block!r}")
    if set(config.sandbox.keys()) - {"opensandbox", "default_metadata"} or "opensandbox" not in config.sandbox:
        raise ValueError("OOM probe requires the real OpenSandbox provider")
    # The disposable probe may create one pod only. Keep the provider's ordinary
    # idempotent operation retries, but never retry a create or an exec command.
    config.sandbox.opensandbox.create.retries = 0
    config.sandbox.opensandbox.create.renew_interval_s = None
    config.sandbox.opensandbox.operations.command_retries = 0
    return config


def task_tree_sha256(tasks_dir: Path) -> str:
    """Hash every task-source path and its bytes in stable path order."""
    digest = hashlib.sha256()
    files = sorted(path for path in tasks_dir.rglob("*") if path.is_file() or path.is_symlink())
    if not files:
        raise ValueError("Frozen task source is empty")
    for path in files:
        if path.is_symlink():
            raise ValueError(f"Frozen task source contains a symlink: {path.relative_to(tasks_dir)}")
        relative = path.relative_to(tasks_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def build_probe_configs(
    global_config: DictConfig, *, tasks_dir: Path, artifacts_dir: Path
) -> tuple[DeepSWEResourcesServerConfig, OpenCodeSandboxedAgentConfig]:
    resource_data = OmegaConf.to_container(get_first_server_config_dict(global_config, RESOURCE_NAME), resolve=True)
    agent_data = OmegaConf.to_container(get_first_server_config_dict(global_config, AGENT_NAME), resolve=True)
    if not isinstance(resource_data, dict) or not isinstance(agent_data, dict):
        raise ValueError("Resolved DeepSWE server configs must be mappings")
    resource_config = DeepSWEResourcesServerConfig.model_validate(resource_data | {"name": RESOURCE_NAME})
    agent_config = OpenCodeSandboxedAgentConfig.model_validate(agent_data | {"name": AGENT_NAME})
    if not resource_config.score_agent_sandbox_oom_as_zero or not agent_config.report_agent_sandbox_oom:
        raise ValueError("Both confirmed agent OOM-zero flags must be enabled")
    if resource_config.is_verifying_golden_patch or agent_config.tool_servers:
        raise ValueError("Probe requires a non-golden DeepSWE run without external tool servers")
    if agent_config.network_access != "inherit":
        raise ValueError("The agent must connect to the seeded DeepSWE sandbox")
    if not agent_config.preinstalled_opencode and not (
        agent_config.remote_opencode_install_script_path and agent_config.remote_opencode_binary_path
    ):
        raise ValueError("Probe refuses a network-dependent OpenCode installer")

    resource_config.tasks_dir = tasks_dir
    resource_config.sandbox_config = dict(resource_config.sandbox_config) | {
        "ttl_s": PROBE_TTL_S,
        "ready_timeout_s": PROBE_WATCHDOG_S,
        "resources": {"memory_mib": PROBE_MEMORY_MIB},
    }
    resource_config.logs_dir = artifacts_dir / "verifier-logs"
    agent_config.artifacts_dir = str(artifacts_dir / "agent-artifacts")
    return resource_config, agent_config


def require_exact_checkout(expected_gym_sha: str) -> None:
    root = Path(__file__).resolve().parents[1]
    actual_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    if len(expected_gym_sha) != 40 or actual_sha != expected_gym_sha:
        raise ValueError(f"Gym checkout is {actual_sha}; reviewed pin is {expected_gym_sha}")
    production_paths = ["nemo_gym", "resources_servers", "responses_api_agents", "benchmarks"]
    diff = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", *production_paths],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if diff:
        raise ValueError("Real OOM probe requires unmodified production Gym source")


async def cleanup_probe_sandboxes(
    *, seeded: AsyncSandbox | None, connected: AsyncSandbox | None, expected_id: str | None
) -> dict[str, Any]:
    """Stop both clients with bounded waits and report provider-confirmed teardown."""
    errors: list[str] = []
    cleanup_started_at = monotonic()
    sandboxes = (("connected", connected), ("seeded", seeded))
    for label, sandbox in sandboxes:
        if sandbox is not None:
            if sandbox.sandbox_id != expected_id:
                errors.append(f"{label} sandbox ID differs from the seeded handle")
            if not sandbox._stopped:
                remaining_s = 25 - (monotonic() - cleanup_started_at)
                if remaining_s <= 0:
                    errors.append(f"{label} stop skipped: cleanup deadline expired")
                else:
                    try:
                        await asyncio.wait_for(sandbox.stop(), timeout=remaining_s)
                    except Exception as error:
                        errors.append(f"{label} stop failed: {type(error).__name__}: {error}")

    def stop_confirmed(sandbox: AsyncSandbox | None) -> bool:
        # stop() preserves the handle after provider close; disconnect() clears it.
        return bool(
            sandbox is not None
            and sandbox.sandbox_id == expected_id
            and sandbox._stopped
            and sandbox._handle is not None
        )

    seeded_stopped = stop_confirmed(seeded)
    connected_stopped = stop_confirmed(connected)
    return {
        "seeded_stop_confirmed": seeded_stopped,
        "connected_stop_confirmed": connected_stopped,
        "same_sandbox_stop_confirmed": seeded_stopped or connected_stopped,
        "errors": errors,
    }


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    require_exact_checkout(args.expected_gym_sha)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != args.expected_probe_sha256:
        raise ValueError("OOM probe script bytes differ from the reviewed SHA-256")
    if DEEPSWE_SOURCE_REVISION != args.expected_task_source_revision:
        raise ValueError("Gym DeepSWE source revision differs from the reviewed pin")
    global_config = load_probe_config(args.resolved_config, expected_sha256=args.expected_config_sha256)
    if not args.tasks_dir.is_absolute() or not args.tasks_dir.is_dir():
        raise ValueError("--tasks-dir must name an existing absolute frozen task source")
    if task_tree_sha256(args.tasks_dir) != args.expected_task_tree_sha256:
        raise ValueError("Frozen task-source bytes differ from the reviewed tree SHA-256")
    if args.artifacts_dir.exists():
        raise ValueError("--artifacts-dir must be a new path, never an existing run directory")
    args.artifacts_dir.mkdir(parents=True)
    resource_config, agent_config = build_probe_configs(
        global_config, tasks_dir=args.tasks_dir, artifacts_dir=args.artifacts_dir
    )
    old_global_config = global_config_module._GLOBAL_CONFIG_DICT
    global_config_module._GLOBAL_CONFIG_DICT = global_config
    resource: DeepSWEResourcesServer | None = None
    transport: LocalDeepSWETransport | None = None
    gate: AgentExecGate | None = None
    result_payload: dict[str, Any] | None = None
    failure: BaseException | None = None
    cleanup: dict[str, Any] = {}
    try:
        resource_client = MagicMock(spec=ServerClient)
        resource = DeepSWEResourcesServer(config=resource_config, server_client=resource_client)
        task = resource._task_store.get(args.task_id)
        if task_image(task) != args.expected_image:
            raise ValueError("Canonical task image differs from the reviewed image")
        request_body = OpenCodeSandboxedAgentRunRequest.model_validate(
            {
                "task_id": args.task_id,
                "image": args.expected_image,
                "verifier_metadata": {"task_id": args.task_id},
                "responses_create_params": {
                    "input": [{"role": "user", "content": "No model inference is permitted in this OOM probe."}]
                },
            }
        )
        request_json = json.dumps(request_body.model_dump(mode="json")).encode("utf-8")

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": request_json, "more_body": False}

        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/run",
                "headers": [],
                "session": {SESSION_ID_KEY: uuid4().hex},
            },
            receive=receive,
        )
        transport = LocalDeepSWETransport(resource, request)
        agent_client = MagicMock(spec=ServerClient)
        agent_client.global_config_dict = global_config
        agent_client.post = AsyncMock(side_effect=transport.post)
        agent = OpenCodeSandboxedAgent(config=agent_config, server_client=agent_client)
        original_start = agent._start_sandbox

        async def connect_seeded_sandbox(sandbox_id: str | None = None, workdir: str | None = None) -> AsyncSandbox:
            nonlocal gate
            if not sandbox_id or sandbox_id != transport.seeded_sandbox_id or transport.seed_started_at is None:
                raise ProbeNotValidated("OpenCode did not connect to the just-seeded sandbox")
            sandbox = await original_start(sandbox_id=sandbox_id, workdir=workdir)
            gate = AgentExecGate(sandbox, seeded_sandbox_id=sandbox_id, seeded_at=transport.seed_started_at)
            sandbox.exec = gate.exec
            return sandbox

        agent._start_sandbox = connect_seeded_sandbox
        try:
            result = await asyncio.wait_for(agent._run(request, request_body), timeout=PROBE_TTL_S - 30)
        except TimeoutError as error:
            raise ProbeNotValidated("Whole OOM probe exceeded the sandbox cleanup window") from error
        if gate is None or transport.seeded_sandbox_id is None:
            raise ProbeNotValidated("Probe did not seed and connect a sandbox")
        assert_confirmed_oom_result(
            result, sandbox_id=transport.seeded_sandbox_id, gate=gate, verify_calls=transport.verify_calls
        )
        result_payload = {
            "gym_sha": args.expected_gym_sha,
            "task_source_revision": DEEPSWE_SOURCE_REVISION,
            "task_id": args.task_id,
            "image": args.expected_image,
            "sandbox_id": transport.seeded_sandbox_id,
            "provider_oom_error": str(gate.provider_oom),
            "reward": result.reward,
            "failure_kind": result.failure_kind,
            "evaluation_completed": result.evaluation_completed,
            "mask_sample": result.mask_sample,
            "verifier_exit_code": result.verifier_exit_code,
            "model_patch_bytes": result.model_patch_bytes,
        }
    except BaseException as error:
        failure = error
    finally:
        try:
            seeded = transport.seeded_sandbox if transport is not None else None
            if seeded is None and resource is not None and resource._agent_sessions:
                seeded = next(iter(resource._agent_sessions.values())).sandbox
            cleanup = await cleanup_probe_sandboxes(
                seeded=seeded,
                connected=gate.sandbox if gate is not None else None,
                expected_id=transport.seeded_sandbox_id if transport is not None else None,
            )
            audit = {
                "status": (
                    "validated"
                    if (
                        failure is None
                        and result_payload is not None
                        and cleanup["same_sandbox_stop_confirmed"]
                        and not cleanup["errors"]
                    )
                    else "not_validated"
                ),
                "gym_sha": args.expected_gym_sha,
                "probe_sha256": args.expected_probe_sha256,
                "task_tree_sha256": args.expected_task_tree_sha256,
                "task_id": args.task_id,
                "sandbox_id": transport.seeded_sandbox_id if transport is not None else None,
                "probe_overrides": {
                    "agent_memory_mib": PROBE_MEMORY_MIB,
                    "ttl_s": PROBE_TTL_S,
                    "ready_timeout_s": PROBE_WATCHDOG_S,
                    "agent_exec_watchdog_s": PROBE_WATCHDOG_S,
                    "create_retries": 0,
                    "command_retries": 0,
                    "ttl_renewal": False,
                },
                "provider_oom_confirmed": gate is not None and gate.provider_oom is not None,
                "error_type": type(failure).__name__ if failure is not None else None,
                "error_message": str(failure)[:2000] if failure is not None else None,
                "cleanup": cleanup,
            }
            (args.artifacts_dir / "probe-audit.json").write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            if resource is not None:
                resource._agent_sessions.clear()
        finally:
            global_config_module._GLOBAL_CONFIG_DICT = old_global_config

    if failure is not None:
        raise failure.with_traceback(failure.__traceback__)
    if result_payload is None:
        raise ProbeNotValidated("OOM probe returned no validated result")
    if not cleanup["same_sandbox_stop_confirmed"]:
        raise ProbeNotValidated("No provider-confirmed same-sandbox teardown; inspect probe-audit.json and pod status")
    if cleanup["errors"]:
        raise ProbeNotValidated("Sandbox cleanup reported errors; inspect probe-audit.json and pod status")
    return result_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-real-oom-probe", action="store_true", help="Required opt-in to create one real pod")
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-gym-sha", required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    parser.add_argument("--expected-task-source-revision", required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--expected-task-tree-sha256", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--expected-image", required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute_real_oom_probe:
        parser.error("--execute-real-oom-probe is required; no sandbox was created")
    return args


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_probe(args))
    output_path = args.artifacts_dir / "probe-result.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Confirmed same-sandbox agent OOM-zero; diagnostic result: {output_path}")


if __name__ == "__main__":
    main()
