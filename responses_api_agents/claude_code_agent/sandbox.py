# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Claude Code artifacts and request state over the shared sandbox lifecycle."""

import asyncio
import io
import json
import logging
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from typing import Any

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.agent_utils.sandbox_session import HARNESS_RUNNER_FILE, SandboxCommand, SandboxSession, shell_assignments
from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.config import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle, ObservationGap
from nemo_gym.sandbox import read_text, upload_text
from nemo_gym.skills import stage_skills
from responses_api_agents.claude_code_agent.observability import extract_claude_code_observations


LOG = logging.getLogger(__name__)


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
        LOG.warning("Claude runtime metadata is missing or malformed")
        return None


@dataclass
class ClaudeCodeArtifacts:
    """Native output copied before the sandbox connection is released."""

    events: str
    transcripts: bytes | None = None


def parse_artifact_observations(
    artifacts: ClaudeCodeArtifacts,
    *,
    model_ref: ModelServerRef | None,
    status: str,
    error_type: str | None,
    run_metadata: dict[str, Any],
) -> AgentObservationBundle:
    """Extract native observations off the event loop after capture completes."""
    try:
        if artifacts.transcripts is None:
            raise ValueError("Claude native transcripts are unavailable")
        with tempfile.TemporaryDirectory(prefix="gym-claude-observations-") as temporary:
            path = Path(temporary)
            with tarfile.open(fileobj=io.BytesIO(artifacts.transcripts)) as archive:
                archive.extractall(path, filter="data")
            observations = extract_claude_code_observations(
                path,
                model_ref=model_ref,
                root_status=status,
                root_duration_ms=run_metadata.get("duration_ms"),
                root_error_type=error_type,
                compaction_attempts=run_metadata.get("compaction_attempts"),
            )
    except Exception:
        LOG.warning("Could not parse Claude native observations", exc_info=True)
        observations = AgentObservationBundle(
            source="claude_code", gaps=[ObservationGap(code="agent_transcript_unavailable")]
        )
    if model_ref is None:
        observations.gaps.append(ObservationGap(code="model_call_ownership_unavailable"))
    return observations


@dataclass
class ClaudeCodeSandboxSession(AgentSessionState):
    """Adapter state and capture hooks; SandboxSession owns supervision and release."""

    session: SandboxSession[ClaudeCodeArtifacts]
    runtime: str
    task: asyncio.Task[NeMoGymResponse] | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    runtime_info: HarnessProcessInfo | None = None
    observations: AgentObservationBundle | None = None
    stderr: str = ""

    async def upload_json(self, name: str, payload: JsonValue) -> None:
        """Stage runner input using the shared file-transfer utility."""
        await upload_text(self.session.sandbox, path=f"{self.session.session_dir}/{name}", text=json.dumps(payload))

    async def read_text(self, name: str) -> str:
        """Read a session artifact using file transfer."""
        return await read_text(self.session.sandbox, path=f"{self.session.session_dir}/{name}")

    async def stage_skills(self, path: str) -> None:
        """Upload a private copy of configured skills, including supporting files."""
        with tempfile.TemporaryDirectory(prefix="gym-claude-skills-") as temporary:
            skills = Path(temporary) / "skills"
            await asyncio.to_thread(stage_skills, path, skills)
            for source in sorted(skills.rglob("*")):
                if source.is_file():
                    target = f"{self.session.session_dir}/config/skills/{source.relative_to(skills).as_posix()}"
                    await self.session.sandbox.upload(source, target)

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

    async def execute(
        self, payload: dict[str, JsonValue], *, timeout: float, close_timeout: float
    ) -> ClaudeCodeArtifacts:
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

    async def stage_activation(self, payload: dict[str, JsonValue]) -> SandboxCommand:
        """Stage the invocation and return the runner command."""
        payload = dict(payload)
        await self.upload_json("config/settings.json", payload.pop("settings"))
        mcp_config = payload.pop("mcp_config")
        if mcp_config is not None:
            await self.upload_json("mcp.json", mcp_config)
        skills_path = payload.pop("skills_path")
        if skills_path:
            await self.stage_skills(skills_path)
        directory = self.session.session_dir
        # Regular files avoid waiting for detached task services to close inherited pipes.
        params = {
            "cwd": payload["cwd"],
            "command": payload["command"],
            "env": payload["env"],
            "stdin": "/dev/null",
            "stdout": f"{directory}/events.jsonl",
            "stderr": f"{directory}/stderr.log",
        }
        await upload_text(self.session.sandbox, path=f"{directory}/params.sh", text=shell_assignments(params))
        return SandboxCommand(argv=["bash", f"{directory}/{HARNESS_RUNNER_FILE}", f"{directory}/params.sh"])

    async def collect_artifacts(self) -> ClaudeCodeArtifacts:
        """Copy dedicated stdout and transcripts after confirmed harness termination."""
        try:
            self.stderr = (await self.read_text("stderr.log"))[-16000:]
        except Exception:
            LOG.warning("Claude stderr log unavailable", exc_info=True)
        try:
            events = await self.read_text("events.jsonl")
        except Exception as error:
            logs = await self.session.read_output_log()
            raise RuntimeError(f"Claude runner returned no events: {(self.stderr or logs)[-16000:]}") from error
        try:
            runtime = json.loads(await self.read_text("runtime.json"))
        except Exception:
            runtime = None
        self.runtime_info = parse_runtime_info(runtime)
        transcripts = None
        directory = self.session.session_dir
        try:
            result = await self.session.sandbox.exec(
                f"tar -czf {quote(directory + '/transcripts.tar.gz')} -C {quote(directory + '/config')} projects",
                timeout_s=30,
            )
            if result.return_code or result.error_type:
                raise RuntimeError("Claude native transcripts are unavailable")
            with tempfile.TemporaryDirectory(prefix="gym-claude-capture-") as temporary:
                local = Path(temporary) / "transcripts.tar.gz"
                await self.session.sandbox.download(directory + "/transcripts.tar.gz", local)
                transcripts = await asyncio.to_thread(local.read_bytes)
        except Exception:
            LOG.warning("Could not collect Claude native transcripts", exc_info=True)
        return ClaudeCodeArtifacts(events=events, transcripts=transcripts)
