# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenClaw artifacts and request state over the shared sandbox lifecycle."""

import asyncio
import json
import logging
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox.utils import read_text, upload_text


LOG = logging.getLogger(__name__)


# Executed by sandbox Python before uploading any adapter-owned files. Resolve
# symlinks and '..' on the sandbox filesystem, not on the agent-server host.
_SANDBOX_PATH_CHECK = """
from pathlib import Path
import sys
workdir, session_root, runtime_root = (Path(value).resolve() for value in sys.argv[1:])
if not workdir.is_dir():
    raise SystemExit("OpenClaw task workdir is missing or is not a directory: " + str(workdir))
for owned in (session_root, runtime_root):
    if workdir == owned or workdir in owned.parents or owned in workdir.parents:
        raise SystemExit("OpenClaw task workdir overlaps adapter-owned storage: " + str(owned))
"""


class HarnessProcessInfo(BaseModel):
    """Optional OpenClaw shim identity, independent of supervisor cleanup."""

    model_config = ConfigDict(extra="forbid", strict=True)
    hostname: str
    pid: int
    python: str | None = None


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

    async def upload_json(self, name: str, payload: JsonValue) -> None:
        """Stage adapter input using the shared text-transfer utility."""
        await upload_text(self.session.sandbox, path=f"{self.session.session_dir}/{name}", text=json.dumps(payload))

    async def read_text(self, name: str) -> str:
        """Read adapter output using file transfer."""
        return await read_text(self.session.sandbox, path=f"{self.session.session_dir}/{name}")

    async def close(self, timeout: float) -> None:
        """Capture and release before cancelling the HTTP activation."""
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
        await self.upload_json("home/.openclaw/openclaw.json", payload["config"])
        await self.upload_json("input.json", payload)
        return SandboxCommand(
            python="python3",
            argv=[
                "python3",
                "-I",
                f"{self.session.session_dir}/sandbox_runner.py",
                f"{self.session.session_dir}/input.json",
            ],
        )

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
