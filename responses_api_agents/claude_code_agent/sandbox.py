# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Claude Code artifacts and request state over the shared sandbox lifecycle."""

import asyncio
import io
import json
import logging
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from time import time
from typing import Any

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.agents import (
    HARNESS_RUNNER_FILE,
    AgentSessionState,
    SandboxCommand,
    SandboxSession,
    shell_assignments,
    stage_skills,
)
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import read_text, upload_text
from nemo_gym.trajectory import (
    AgentInvocation,
    AgentObservationBundle,
    ObservationGap,
    TrajectoryRecord,
    TrajectoryTurn,
)
from responses_api_agents.claude_code_agent.observability import (
    ClaudeCodeTranscripts,
    extract_claude_code_observations,
    read_transcripts,
)


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


def read_artifact_transcripts(artifacts: ClaudeCodeArtifacts) -> ClaudeCodeTranscripts | None:
    """The run's transcripts from the captured archive, read once for token usage and observations; None without
    an archive or when it cannot be read."""
    if artifacts.transcripts is None:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="gym-claude-transcripts-") as temporary:
            path = Path(temporary)
            with tarfile.open(fileobj=io.BytesIO(artifacts.transcripts)) as archive:
                archive.extractall(path, filter="data")
            return read_transcripts(path)
    except Exception:
        LOG.warning("Could not read Claude native transcripts", exc_info=True)
        return None


def parse_artifact_observations(
    transcripts: ClaudeCodeTranscripts | None,
    *,
    model_ref: ModelServerRef | None,
    status: str,
    error_type: str | None,
    run_metadata: dict[str, Any],
) -> AgentObservationBundle:
    """Extract native observations off the event loop after capture completes."""
    try:
        if transcripts is None:
            raise ValueError("Claude native transcripts are unavailable")
        observations = extract_claude_code_observations(
            transcripts=transcripts,
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
    sandbox_id: str | None = None
    task: asyncio.Task[NeMoGymResponse] | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    runtime_info: HarnessProcessInfo | None = None
    observations: AgentObservationBundle | None = None
    stderr: str = ""
    # Multi-turn continuation: the activation running now, the Claude Code session the next one resumes, the input
    # system prompt every activation passes again, and the session's cumulative trajectory.
    activation_id: int | None = None
    harness_session_id: str | None = None
    input_system: str | None = None
    instructions_set: bool = False
    trajectory: TrajectoryRecord | None = None
    observations_so_far: AgentObservationBundle | None = None
    # Transcript records that earlier activations already reported.
    _seen_transcript_records: set[str] = field(default_factory=set, init=False, repr=False)

    @property
    def is_continuation(self) -> bool:
        """Whether the seed opted into multi-turn continuation."""
        return self.request.continuation is not None

    @property
    def config_dir(self) -> str:
        """Claude Code's ``CLAUDE_CONFIG_DIR``, which holds its saved sessions across activations."""
        return f"{self.session.root_dir}/config"

    @property
    def invocation_prefix(self) -> str:
        """Identity of the current activation, unique within the agent session."""
        assert self.activation_id is not None
        return f"{self.request.agent_session_id}:activation-{self.activation_id}"

    async def prepare_activation(self, activation_id: int, *, timeout: float) -> None:
        """Rearm the supervisor in a fresh directory; the saved Claude Code session stays in ``config_dir``."""
        assert self.is_continuation
        self.activation_id = activation_id
        self.runtime_info = None
        self.stderr = ""
        directory = f"{self.session.root_dir}/activation-{activation_id}"
        await self.session.begin_activation(session_dir=directory, timeout=timeout)
        result = await self.session.sandbox.exec(f"mkdir -p -- {quote(directory)}", cwd="/", timeout_s=timeout)
        if result.error_type == "timeout":
            raise TimeoutError("Claude activation directory preparation timed out")
        if result.return_code != 0 or result.error_type:
            raise RuntimeError(f"Cannot prepare Claude activation directory: {result.stderr}")

    def new_transcript_records(self, transcripts: ClaudeCodeTranscripts) -> ClaudeCodeTranscripts:
        """Keep the records that no earlier activation reported, and remember them.

        A resumed Claude Code session appends to the transcript it resumed, so an activation's evidence is the
        records added since the previous activation.
        """
        records = []
        for record in transcripts.records:
            key = record.get("uuid")
            key = key if isinstance(key, str) and key else json.dumps(record, sort_keys=True)
            if key not in self._seen_transcript_records:
                self._seen_transcript_records.add(key)
                records.append(record)
        return ClaudeCodeTranscripts(records=records, gaps=transcripts.gaps, files_read=transcripts.files_read)

    def record_activation(
        self,
        bundle: AgentObservationBundle,
        *,
        root_sessions: set[str],
        status: str,
        error_type: str | None,
        question: str,
        answer: str | None,
    ) -> AgentObservationBundle:
        """Add one activation's evidence to the session's observations and trajectory, and return it.

        Each activation of a resumed session reports the same Claude Code session id, so its root invocation is
        renamed ``<agent session>:activation-<N>:<Claude session>`` and tagged with the activation. An activation
        that left no transcript gets a root invocation with its outcome alone.
        """
        assert self.activation_id is not None
        renamed = {root: f"{self.invocation_prefix}:{root}" for root in root_sessions}

        def rename(value: object) -> object:
            if isinstance(value, dict):
                return {
                    key: renamed.get(item, item)
                    if key in {"invocation_id", "parent_invocation_id"} and isinstance(item, str)
                    else rename(item)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [rename(item) for item in value]
            return value

        bundle = AgentObservationBundle.model_validate(rename(bundle.model_dump(mode="json")))
        roots = [
            record
            for record in bundle.records
            if isinstance(record, AgentInvocation) and record.invocation_id in renamed.values()
        ]
        if not roots:
            invocation_id = self.invocation_prefix
            if self.harness_session_id is not None:
                invocation_id += f":{self.harness_session_id}"
            roots = [AgentInvocation(invocation_id=invocation_id, status=status, error_type=error_type)]
            bundle.records.insert(0, roots[0])
        for record in roots:
            record.activation_id = self.activation_id
            record.harness_session_id = self.harness_session_id

        previous = self.observations_so_far
        self.observations_so_far = AgentObservationBundle(
            source=bundle.source,
            records=[*(previous.records if previous else []), *bundle.records],
            gaps=[*(previous.gaps if previous else []), *bundle.gaps],
        )
        so_far = self.trajectory or TrajectoryRecord(
            task_id=self.request.task_id.task_id, rollout_id=self.request.episode_id.capture_key
        )
        root = roots[0]
        self.trajectory = so_far.model_copy(
            update={
                "invocations": [
                    *so_far.invocations,
                    *(record for record in bundle.records if isinstance(record, AgentInvocation)),
                ],
                "turns": [
                    *so_far.turns,
                    TrajectoryTurn(
                        invocation_id=root.invocation_id,
                        activation_id=self.activation_id,
                        task_id=so_far.task_id,
                        rollout_id=so_far.rollout_id,
                        turn_no=1,
                        timestamp=time(),
                        question=question,
                        answer=answer,
                        step_count=len(root.model_calls),
                        model_calls=list(root.model_calls),
                    ),
                ],
                "gaps": [*so_far.gaps, *bundle.gaps],
            }
        )
        return self.observations_so_far

    async def upload_json(self, name: str, payload: JsonValue) -> None:
        """Stage runner input at ``name``, an absolute path or one relative to the activation's directory."""
        path = name if name.startswith("/") else f"{self.session.session_dir}/{name}"
        await upload_text(self.session.sandbox, path=path, text=json.dumps(payload))

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
                    target = f"{self.config_dir}/skills/{source.relative_to(skills).as_posix()}"
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
        await self.upload_json(f"{self.config_dir}/settings.json", payload.pop("settings"))
        mcp_config = payload.pop("mcp_config")
        if mcp_config is not None:
            await self.upload_json("mcp.json", mcp_config)
        skills_path = payload.pop("skills_path")
        if skills_path:
            await self.stage_skills(skills_path)
        directory = self.session.session_dir
        # An optional prompt reaches the CLI on stdin; otherwise stdin is empty.
        stdin = "/dev/null"
        prompt = payload.pop("prompt", None)
        if prompt is not None:
            stdin = f"{directory}/prompt.txt"
            await upload_text(self.session.sandbox, path=stdin, text=prompt)
        # Regular files avoid waiting for detached task services to close inherited pipes.
        params = {
            "cwd": payload["cwd"],
            "command": payload["command"],
            "env": payload["env"],
            "stdin": stdin,
            "stdout": f"{directory}/events.jsonl",
            "stderr": f"{directory}/stderr.log",
        }
        await upload_text(self.session.sandbox, path=f"{directory}/params.sh", text=shell_assignments(params))
        # The seed uploaded the runner to the directory the session started in.
        runner = f"{self.session.root_dir}/{HARNESS_RUNNER_FILE}"
        return SandboxCommand(argv=["bash", runner, f"{directory}/params.sh"])

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
                f"tar -czf {quote(directory + '/transcripts.tar.gz')} -C {quote(self.config_dir)} projects",
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
