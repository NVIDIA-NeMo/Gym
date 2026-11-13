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

from nemo_gym.agents import (
    HARNESS_RUNNER_FILE,
    HARNESS_RUNNER_SCRIPT,
    AgentSessionState,
    SandboxCommand,
    SandboxSession,
    shell_assignments,
    stage_skills,
)
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import read_text, upload_text
from nemo_gym.trajectory import AgentObservationBundle, ObservationGap
from responses_api_agents.claude_code_agent.observability import extract_claude_code_observations


LOG = logging.getLogger(__name__)


def with_final_newline(events: str) -> str:
    return events if not events or events.endswith("\n") else events + "\n"


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
    # Where to copy the stream-json output in the sandbox; the session directory is removed at close.
    transcript_path: str | None = None
    # CLAUDE_CONFIG_DIR, with the native conversation: fixed at the first activation, so a resume finds it.
    config_dir: str | None = None
    # Activations rearmed with ``begin_resume`` after the first one.
    resumes: int = 0
    # The stream-json events of the activations before the current one, held on the host for the response and the
    # transcript Gym writes.
    earlier_events: str = ""
    # Gym could not replace the verifier transcript, so the file at transcript_path may be the agent's.
    transcript_unavailable: bool = False

    def claude_config_dir(self) -> str:
        """Claude Code's config directory: the first activation's ``config``, kept across resumes."""
        if self.config_dir is None:
            self.config_dir = f"{self.session.session_dir}/config"
        return self.config_dir

    async def begin_resume(self, *, timeout: float) -> None:
        """Rearm the stopped harness in the same sandbox for a ``claude --resume`` of its conversation."""
        root = self.claude_config_dir().removesuffix("/config")
        directory = f"{root}/resume-{self.resumes + 1}"
        await self.session.begin_activation(session_dir=directory, timeout=timeout)
        result = await self.session.sandbox.exec(f"mkdir -p -- {quote(directory)}", cwd="/", timeout_s=timeout)
        if result.return_code or result.error_type:
            raise RuntimeError(f"Cannot prepare the Claude Code resume directory: {result.stderr}")
        await self.session.sandbox.upload(HARNESS_RUNNER_SCRIPT, f"{directory}/{HARNESS_RUNNER_FILE}")
        # Counted only once rearmed: a resume that could not start never ran.
        self.resumes += 1

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
        settings = payload.pop("settings")
        if not self.resumes:
            # A resume reads the settings the first activation put in the config directory it keeps.
            await self.upload_json("config/settings.json", settings)
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

    async def copy_transcript(self, path: str, events: str) -> None:
        """Write the conversation's events to ``path`` for a task verifier that reads the harness transcript.

        Gym writes the events it holds for every activation, replacing whatever the activation left at ``path``.
        When that fails, ``transcript_unavailable`` is set: the file at ``path`` may then be the agent's.
        """
        self.transcript_unavailable = True
        parent = quote(str(Path(path).parent))
        result = await self.session.sandbox.exec(f"mkdir -p {parent} && rm -rf -- {quote(path)}", timeout_s=60)
        if result.return_code or result.error_type:
            LOG.warning("Could not prepare the Claude transcript at %s: %s", path, (result.stderr or "")[-2000:])
            return
        try:
            await upload_text(self.session.sandbox, path=path, text=self.earlier_events + with_final_newline(events))
        except Exception:
            LOG.warning("Could not write the Claude transcript to %s", path, exc_info=True)
            return
        self.transcript_unavailable = False

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
        if self.transcript_path:
            await self.copy_transcript(self.transcript_path, events)
        try:
            runtime = json.loads(await self.read_text("runtime.json"))
        except Exception:
            runtime = None
        self.runtime_info = parse_runtime_info(runtime)
        transcripts = None
        directory = self.session.session_dir
        try:
            result = await self.session.sandbox.exec(
                f"tar -czf {quote(directory + '/transcripts.tar.gz')} -C {quote(self.claude_config_dir())} projects",
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
