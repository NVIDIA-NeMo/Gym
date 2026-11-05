# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Droid artifacts and request state over the shared sandbox lifecycle."""

import asyncio
import json
import logging
from dataclasses import dataclass
from shlex import quote
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from nemo_gym.agents import HARNESS_RUNNER_FILE, AgentSessionState, SandboxCommand, SandboxSession, shell_assignments
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import read_text, upload_text
from nemo_gym.trajectory import AgentObservationBundle


LOG = logging.getLogger(__name__)
SESSION_ROOT = "/tmp/nemo-gym-droid-sessions"
RUNTIME_PREFIX = "/tmp/nemo-gym-droid-runtime-"


class HarnessProcessInfo(BaseModel):
    """Optional runner identity, independent of the supervisor cleanup receipt."""

    model_config = ConfigDict(extra="forbid", strict=True)
    hostname: str
    pid: int


def parse_runtime_info(payload: object) -> HarnessProcessInfo | None:
    """Read runner diagnostics without discarding an otherwise valid transcript."""
    try:
        return HarnessProcessInfo.model_validate(payload)
    except ValidationError:
        LOG.warning("Droid runtime metadata is missing or malformed")
        return None


@dataclass
class DroidArtifacts:
    """Native output copied before the sandbox connection is released."""

    events: str
    stderr: str = ""


@dataclass
class DroidSandboxSession(AgentSessionState):
    """Adapter state and capture hooks; SandboxSession owns supervision and release."""

    session: SandboxSession[DroidArtifacts]
    runtime: str
    task: asyncio.Task[NeMoGymResponse] | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    runtime_info: HarnessProcessInfo | None = None
    observations: AgentObservationBundle | None = None

    @property
    def factory_home(self) -> str:
        """Droid's private state directory (``FACTORY_HOME_OVERRIDE``), outside the task repo and home."""
        return self.session.session_dir + "/home"

    async def close(self, timeout: float) -> None:
        """Join captured-artifact parsing before returning observations, within a bound."""
        await self.session.close(timeout=timeout)
        # Captured artifacts are parsed by the activation after execute() returns.
        # Cancelling that step would discard evidence that close has already retained.
        if (
            self.session.artifacts is None
            and self.task is not None
            and not self.task.done()
            and not self.task.cancelling()
        ):
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
                # The agent route replays the activation error independently of close.

    async def execute(self, payload: dict[str, Any], *, timeout: float, close_timeout: float) -> DroidArtifacts:
        """Execute through the common lifecycle, retaining artifacts on failure."""
        artifacts = await self.session.execute(
            stage_activation=lambda: self.stage_activation(payload),
            collect=self.collect_artifacts,
            timeout=timeout,
            close_timeout=close_timeout,
        )
        if self.session.closing:
            raise asyncio.CancelledError
        return artifacts

    async def stage_activation(self, payload: dict[str, Any]) -> SandboxCommand:
        """Upload the settings, prompt and runner input, and return the runner command."""
        directory = self.session.session_dir
        sandbox = self.session.sandbox
        await upload_text(sandbox, path=f"{directory}/settings.json", text=json.dumps(payload["settings"]))
        await upload_text(sandbox, path=f"{directory}/prompt.txt", text=payload["prompt"])
        if payload["system"] is not None:
            await upload_text(sandbox, path=f"{directory}/system.txt", text=payload["system"])
        # Regular files avoid waiting for detached task services to close inherited pipes.
        params = {
            "cwd": self.session.workdir,
            "command": payload["command"],
            "env": payload["env"],
            "stdin": "/dev/null",
            "stdout": f"{directory}/events.jsonl",
            "stderr": f"{directory}/stderr.log",
        }
        await upload_text(sandbox, path=f"{directory}/params.sh", text=shell_assignments(params))
        return SandboxCommand(argv=["bash", f"{directory}/{HARNESS_RUNNER_FILE}", f"{directory}/params.sh"])

    async def collect_artifacts(self) -> DroidArtifacts:
        """Copy stdout, the stderr tail and runner identity after confirmed harness termination."""
        directory = self.session.session_dir
        if self.session.expired_before_launch:
            return DroidArtifacts(events="")  # A rollout stop arrived before launch, so Droid never ran.
        stderr = ""
        try:
            stderr = (await read_text(self.session.sandbox, path=f"{directory}/stderr.log"))[-16000:]
        except Exception:
            LOG.warning("Droid stderr log unavailable", exc_info=True)
        try:
            events = await read_text(self.session.sandbox, path=f"{directory}/events.jsonl")
        except Exception as error:
            if self.session.expired and await self._events_absent():
                # A rollout stop can end the runner after launch but before it opens Droid's stdout file.
                LOG.warning("Droid wrote no events before a rollout stop ended it; keeping an empty transcript")
                return DroidArtifacts(events="", stderr=stderr)
            logs = await self.session.read_output_log()
            raise RuntimeError(f"Droid runner returned no events: {(stderr or logs)[-16000:]}") from error
        try:
            runtime = json.loads(await read_text(self.session.sandbox, path=f"{directory}/runtime.json"))
        except Exception:
            runtime = None
        self.runtime_info = parse_runtime_info(runtime)
        return DroidArtifacts(events=events, stderr=stderr)

    async def _events_absent(self) -> bool:
        """Whether the session directory exists without events.jsonl; an unclear answer counts as present."""
        directory = self.session.session_dir
        try:
            probe = await self.session.sandbox.exec(
                f"if [ -e {quote(directory + '/events.jsonl')} ]; then echo present; "
                f"elif [ -d {quote(directory)} ]; then echo absent; fi",
                cwd="/",
                timeout_s=10,
            )
        except Exception:
            LOG.warning("Could not check for Droid events after a rollout stop", exc_info=True)
            return False
        return probe.return_code == 0 and not probe.error_type and (probe.stdout or "").strip() == "absent"
