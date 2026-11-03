# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenClaw artifacts and request state over the shared sandbox lifecycle."""

import asyncio
import json
import logging
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.agent_utils.sandbox_session import (
    HARNESS_RUNNER_FILE,
    SandboxCommand,
    SandboxSession,
    shell_assignments,
)
from nemo_gym.agent_utils.supervisor_client import is_harness_termination_failure
from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import read_text, upload_text
from nemo_gym.trajectory import AgentObservationBundle
from nemo_gym.typed_exceptions import NgFailureEvent


LOG = logging.getLogger(__name__)


# POSIX sh run with positional parameters WORKDIR SESSION_ROOT RUNTIME_ROOT before uploading
# any adapter-owned files. Resolve symlinks and '..' on the sandbox filesystem, not on the agent-server
# host; an owned root that does not exist yet resolves through its nearest existing ancestor.
_SANDBOX_PATH_CHECK = """
[ -d "$1" ] || { echo "OpenClaw task workdir is missing or is not a directory: $1" >&2; exit 1; }
workdir=$(cd "$1" && pwd -P) || exit 1
shift
for owned in "$@"; do
    path=$owned rest=
    until [ -d "${path:-/}" ]; do rest=/${path##*/}$rest path=${path%/*}; done
    owned=$(cd "${path:-/}" && pwd -P) || exit 1
    owned=${owned%/}$rest
    # A path of / ends in its separator already; strip it so / contains every other path.
    case "${workdir%/}/" in "${owned%/}/"*) overlap=1 ;; *) overlap= ;; esac
    case "${owned%/}/" in "${workdir%/}/"*) overlap=1 ;; esac
    [ -z "$overlap" ] || { echo "OpenClaw task workdir overlaps adapter-owned storage: ${owned:-/}" >&2; exit 1; }
done
"""


class HarnessProcessInfo(BaseModel):
    """Optional OpenClaw shim identity, independent of supervisor cleanup."""

    model_config = ConfigDict(extra="forbid", strict=True)
    hostname: str
    pid: int


def parse_runtime_info(payload: object) -> HarnessProcessInfo | None:
    """Read OpenClaw diagnostics without failing an otherwise valid episode."""
    try:
        return HarnessProcessInfo.model_validate(payload)
    except ValidationError:
        LOG.warning("OpenClaw runtime metadata is missing or malformed")
        return None


@dataclass
class OpenClawSandboxSession(AgentSessionState):
    """OpenClaw request/output state; SandboxSession owns execution and teardown."""

    session: SandboxSession[dict[str, str]]
    runtime: str
    task: asyncio.Task[NeMoGymResponse] | None = None
    runtime_info: HarnessProcessInfo | None = None
    observations: AgentObservationBundle | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    # An MCP probe whose termination is not yet confirmed; close stops it before releasing the sandbox.
    probe: SandboxSession[dict[str, str]] | None = None

    async def upload_json(self, name: str, payload: JsonValue) -> None:
        """Stage adapter input using the shared text-transfer utility."""
        await upload_text(self.session.sandbox, path=f"{self.session.session_dir}/{name}", text=json.dumps(payload))

    async def read_text(self, name: str) -> str:
        """Read adapter output using file transfer."""
        return await read_text(self.session.sandbox, path=f"{self.session.session_dir}/{name}")

    async def close(self, timeout: float) -> None:
        """Stop an MCP probe, then capture and release before cancelling the HTTP activation."""
        if not self.session.launch_started:
            # Before the agent run launches, the activation may still be staging or about to start an MCP probe.
            # Join it first so no probe can start while release removes the session directory.
            await self._join_activation(timeout)
        if self.probe is not None:
            await self._stop_probe(timeout)
        await self.session.close(timeout=timeout)
        if self.task is not None and not self.task.done() and not self.task.cancelling():
            self.task.cancel()
        if self.task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=timeout)
            except asyncio.CancelledError:
                if not self.task.cancelled():
                    raise
            except Exception:
                if not self.task.done():
                    raise
                # The agent base replays the activation error independently of close.

    async def _join_activation(self, timeout: float) -> None:
        """Cancel an unfinished activation and wait for it; a task that outlives the timeout fails close."""
        if self.task is None or self.task.done():
            return
        if not self.task.cancelling():
            self.task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(self.task), timeout=timeout)
        except asyncio.CancelledError:
            if not self.task.cancelled():
                raise
        except Exception:
            if not self.task.done():
                raise
            # The agent base replays the activation error independently of close.

    async def _stop_probe(self, timeout: float) -> None:
        """Confirm an MCP probe stopped; unconfirmed termination fails close, which keeps the session for retry."""
        probe = self.probe
        assert probe is not None
        # Cancelling the activation unwinds the probe's execute, which stops it and records its receipt.
        await self._join_activation(timeout)
        if not probe.cleanup_confirmed:
            try:
                await probe.stop_harness(timeout=timeout)
            except NgFailureEvent as error:
                if not is_harness_termination_failure(error):
                    raise
                # A final negative receipt cannot be retried. Release the borrowed sandbox as the agent run's
                # session does, with cleanup reported unconfirmed; Resources tears the sandbox down.
                LOG.exception("OpenClaw MCP probe termination unconfirmed; releasing for Resources teardown")
                self.session.termination_error = error
                self.probe = None
                return
        if not probe.cleanup_confirmed:
            raise RuntimeError("OpenClaw MCP probe termination is unconfirmed")
        self.probe = None

    async def execute(self, payload: dict[str, JsonValue], *, timeout: float, close_timeout: float) -> dict[str, str]:
        """Run through the common lifecycle; the adapter validates terminal events."""
        artifacts = await self.session.execute(
            stage_activation=lambda: self.stage_activation(payload),
            collect=self.collect_artifacts,
            timeout=timeout,
            close_timeout=close_timeout,
        )
        if self.session.closing:
            raise asyncio.CancelledError
        return artifacts

    async def stage_activation(self, payload: dict[str, JsonValue]) -> SandboxCommand:
        """Stage the invocation and return the OpenClaw worker command."""
        directory = self.session.session_dir
        await self.upload_json("home/.openclaw/openclaw.json", payload["config"])
        await upload_text(self.session.sandbox, path=f"{directory}/prompt.txt", text=payload["prompt"])
        params = {
            "cwd": payload["cwd"],
            "command": payload["command"],
            "env": payload["env"],
            "stdin": "/dev/null",
            "stdout": f"{directory}/stdout.log",
            "stderr": f"{directory}/stderr.log",
        }
        await upload_text(self.session.sandbox, path=f"{directory}/params.sh", text=shell_assignments(params))
        return SandboxCommand(argv=["bash", f"{directory}/{HARNESS_RUNNER_FILE}", f"{directory}/params.sh"])

    async def collect_artifacts(self) -> dict[str, str]:
        """Retain transcript and diagnostics before releasing the task sandbox."""
        artifacts = {}
        session_id = self.session.session_dir.rsplit("/", 1)[-1]
        for name in ("stdout.log", "stderr.log", f"home/.openclaw/agents/main/sessions/{session_id}.jsonl"):
            try:
                artifacts[name] = await self.read_text(name)
            except Exception:
                LOG.warning("OpenClaw artifact unavailable: %s", name, exc_info=True)
        try:
            runtime = json.loads(await self.read_text("runtime.json"))
        except Exception:
            runtime = None
        self.runtime_info = parse_runtime_info(runtime)
        if "stdout.log" not in artifacts and not any(name.endswith(".jsonl") for name in artifacts):
            logs = await self.session.read_output_log()
            raise RuntimeError(f"OpenClaw sandbox runner returned no valid result: {logs}")
        return artifacts
