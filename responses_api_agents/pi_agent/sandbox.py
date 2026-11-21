# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pi artifacts and request state over the shared sandbox lifecycle."""

import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from time import monotonic
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, ValidationError

from nemo_gym.agents import (
    HARNESS_RUNNER_FILE,
    AgentSessionState,
    SandboxCommand,
    SandboxSession,
    shell_assignments,
)
from nemo_gym.core import MediaFile, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import read_text, stage_media, upload_text
from nemo_gym.trajectory import AgentInvocation, AgentObservationBundle, ObservationGap


LOG = logging.getLogger(__name__)


def interrupted_outcome(error: BaseException) -> tuple[str, str]:
    """The invocation status and error type of an activation that raised before Pi finished.

    A cancellation or a deadline stops work that may still be continued; any other error is a failure.
    """
    if isinstance(error, asyncio.CancelledError):
        return "incomplete", "cancelled"
    if isinstance(error, TimeoutError):
        return "incomplete", "timeout"
    return "failed", "server_error"


class HarnessProcessInfo(BaseModel):
    """Optional Pi shim identity, independent of supervisor cleanup."""

    model_config = ConfigDict(extra="forbid", strict=True)
    hostname: str
    pid: int


def parse_runtime_info(payload: object) -> HarnessProcessInfo | None:
    """Read Pi diagnostics without failing an otherwise valid episode."""
    try:
        return HarnessProcessInfo.model_validate(payload)
    except ValidationError:
        LOG.warning("Pi runtime metadata is missing or malformed")
        return None


@dataclass
class PiSandboxSession(AgentSessionState):
    """Pi request/output state; SandboxSession owns execution and teardown.

    ``directory`` holds what lasts for the whole session: the harness runner and Pi's saved conversation.
    In a continuation session each activation stages its configuration, extensions, prompt, supervisor and output
    in its own subdirectory, which is ``session.session_dir`` while it runs. A one-activation session uses
    ``directory`` for both.
    """

    session: SandboxSession[str]
    runtime: str
    # Defaults to the session directory the session starts in.
    directory: str = ""
    task: asyncio.Task[NeMoGymResponse] | None = None
    runtime_info: HarnessProcessInfo | None = None
    observations: AgentObservationBundle | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    # The ID Pi saves the conversation under, so each activation resumes it.
    harness_session_id: str = field(default_factory=lambda: str(uuid4()))
    activation_id: int | None = None
    # The first activation's system prompt. Pi does not save it, so every activation passes it again.
    system_instructions: str | None = None
    # Whether an activation attached prompt images. Pi saves them in its conversation, which later turns resend.
    has_prompt_images: bool = False
    # When the current activation started, so its invocation reports how long it ran. Set only in a continuation
    # session, whose activations each spend their own budget.
    activation_started: float | None = None
    # The request _validate_agent_activation validated, with the result, so the activation decodes its images once.
    validated_input: (
        tuple[
            NeMoGymResponseCreateParamsNonStreaming,
            NeMoGymResponseCreateParamsNonStreaming,
            list[MediaFile],
        ]
        | None
    ) = field(default=None, repr=False)
    _activation_observations: dict[int, AgentObservationBundle] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.directory = self.directory or self.session.session_dir

    @property
    def is_continuation(self) -> bool:
        """Whether the seed opted into multi-turn continuation."""
        return self.request.continuation is not None

    @property
    def invocation_id(self) -> str:
        """The current activation's identity in observations and in the model server's capture."""
        if not self.is_continuation:
            return self.request.episode_id.capture_key
        assert self.activation_id is not None
        return f"{self.request.agent_session_id}:activation-{self.activation_id}"

    @property
    def pi_session_directory(self) -> str:
        """Where Pi saves the conversation that later activations resume."""
        return f"{self.directory}/pi-sessions"

    async def prepare_activation(self, activation_id: int, *, timeout: float) -> None:
        """Rearm the supervised lifecycle in a fresh activation directory, keeping Pi's saved conversation."""
        assert self.is_continuation
        self.activation_id = activation_id
        self.runtime_info = None
        activation_directory = f"{self.directory}/activation-{activation_id}"
        await self.session.begin_activation(session_dir=activation_directory, timeout=timeout)
        # Uploads do not create directories, and each activation stages Pi's configuration in its own.
        result = await self.session.sandbox.exec(
            f"mkdir -p -- {quote(activation_directory + '/home/.pi/agent')}", cwd="/", timeout_s=timeout
        )
        if result.error_type == "timeout":
            raise TimeoutError("Pi activation directory preparation timed out")
        if result.return_code != 0 or result.error_type:
            raise RuntimeError(f"Cannot prepare Pi activation directory: {result.stderr}")

    def record_observations(self, observations: AgentObservationBundle) -> None:
        """Keep the current activation's observations; close reports every activation's in order."""
        self._activation_observations[self.activation_id or 1] = observations
        bundles = [self._activation_observations[key] for key in sorted(self._activation_observations)]
        self.observations = AgentObservationBundle(
            source="pi",
            records=[record for bundle in bundles for record in bundle.records],
            gaps=[gap for bundle in bundles for gap in bundle.gaps],
        )

    @property
    def activation_duration_ms(self) -> float | None:
        """How long the current continuation activation has run, or None outside a continuation session."""
        if self.activation_started is None:
            return None
        return (monotonic() - self.activation_started) * 1000

    def record_interrupted_activation(self, activation_id: int, error: BaseException) -> AgentObservationBundle:
        """Report an activation that failed before its run left observations, without replacing any it left.

        Returns the activation's observations, whichever reported them.
        """
        if activation_id in self._activation_observations:
            return self._activation_observations[activation_id]
        self.activation_id = activation_id
        status, error_type = interrupted_outcome(error)
        invocation = AgentInvocation(
            invocation_id=self.invocation_id,
            activation_id=activation_id,
            harness_session_id=self.harness_session_id,
            status=status,
            error_type=error_type,
            duration_ms=self.activation_duration_ms,
        )
        gap = ObservationGap(
            code="agent_activation_interrupted", invocation_id=invocation.invocation_id, detail=type(error).__name__
        )
        observations = AgentObservationBundle(source="pi", records=[invocation], gaps=[gap])
        self.record_observations(observations)
        return observations

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

    async def execute(
        self,
        *,
        command: list[str],
        env: dict[str, str],
        prompt: str,
        media: Sequence[MediaFile] = (),
        timeout: float,
        close_timeout: float,
        files: dict[str, str | Path] | None = None,
        deadline: float | None = None,
    ) -> str:
        """Run through the common lifecycle; the adapter validates terminal events.

        ``files`` and ``media`` are staged with the invocation; see ``stage_activation``.
        """
        return await self.session.execute(
            stage_activation=lambda: self.stage_activation(
                command=command, env=env, prompt=prompt, media=media, files=files
            ),
            collect=self.collect_artifacts,
            timeout=timeout,
            close_timeout=close_timeout,
            deadline=deadline,
        )

    async def stage_activation(
        self,
        *,
        command: list[str],
        env: dict[str, str],
        prompt: str,
        media: Sequence[MediaFile] = (),
        files: dict[str, str | Path] | None = None,
    ) -> SandboxCommand:
        """Stage the invocation and return the Pi worker command.

        ``files`` maps paths under the session directory to text or to a local source file, and ``media`` are prompt
        images. Both are uploaded here, inside the session's execute, so that close cancels and waits for the uploads
        before it releases the sandbox, and the activation deadline bounds them.
        """
        directory = self.session.session_dir
        for name, content in (files or {}).items():
            if isinstance(content, Path):
                await self.session.sandbox.upload(content, f"{directory}/{name}")
            else:
                await upload_text(self.session.sandbox, path=f"{directory}/{name}", text=content)
        if media:
            await stage_media(self.session.sandbox, media)
        await upload_text(self.session.sandbox, path=f"{directory}/prompt.txt", text=prompt)
        params = {
            "cwd": self.session.workdir,
            "command": command,
            "env": env,
            "stdin": f"{directory}/prompt.txt",
            "stdout": f"{directory}/stdout.log",
            "stderr": f"{directory}/stderr.log",
            "events": f"{directory}/events.jsonl",
        }
        await upload_text(self.session.sandbox, path=f"{directory}/params.sh", text=shell_assignments(params))
        return SandboxCommand(argv=["bash", f"{self.directory}/{HARNESS_RUNNER_FILE}", f"{directory}/params.sh"])

    async def collect_artifacts(self) -> str:
        """Keep partial events even when worker metadata is unavailable."""
        try:
            events = await self.read_text("events.jsonl")
        except Exception as error:
            logs = await self.session.read_output_log()
            raise RuntimeError(f"Pi sandbox runner returned no valid result: {logs}") from error
        try:
            runtime = await self.read_text("runtime.json")
        except Exception:
            runtime = None
        try:
            runtime_payload = json.loads(runtime) if runtime is not None else None
        except ValueError:
            runtime_payload = None
        self.runtime_info = parse_runtime_info(runtime_payload)
        return events
