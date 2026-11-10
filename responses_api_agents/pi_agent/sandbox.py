# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pi artifacts and request state over the shared sandbox lifecycle."""

import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from nemo_gym.agents import (
    HARNESS_RUNNER_FILE,
    AgentSessionState,
    SandboxCommand,
    SandboxSession,
    prefer_oom_kill_setup,
    shell_assignments,
)
from nemo_gym.core import MediaFile, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import read_text, stage_media, upload_text
from nemo_gym.trajectory import AgentObservationBundle


LOG = logging.getLogger(__name__)
# Streaming events repeat the whole partial message (or tool output) on every delta, so their volume
# grows quadratically with a turn's length. The adapter reads only completed messages, tool start/end
# and lifecycle events, so the runner leaves the streaming ones out of the event log. Pi prints each
# event as JSON with its type first.
STREAMING_EVENTS = ('{"type":?( )"message_update"*', '{"type":?( )"tool_execution_update"*')


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
    """Pi request/output state; SandboxSession owns execution and teardown."""

    session: SandboxSession[str]
    runtime: str
    task: asyncio.Task[NeMoGymResponse] | None = None
    runtime_info: HarnessProcessInfo | None = None
    observations: AgentObservationBundle | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None

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
        raise_oom_score_adj: bool = False,
        stderr: str | None = None,
        transcript: str | None = None,
    ) -> str:
        """Run through the common lifecycle; the adapter validates terminal events.

        ``files`` and ``media`` are staged with the invocation; see ``stage_activation``.
        Optional ``stderr`` and ``transcript`` paths (e.g. under a harness log directory) receive Pi's
        stderr and its JSON lines without the streaming events; otherwise stderr stays in the session
        directory.
        """
        return await self.session.execute(
            stage_activation=lambda: self.stage_activation(
                command=command,
                env=env,
                prompt=prompt,
                media=media,
                files=files,
                raise_oom_score_adj=raise_oom_score_adj,
                stderr=stderr,
            ),
            collect=lambda: self.collect_artifacts(transcript=transcript),
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
        raise_oom_score_adj: bool = False,
        stderr: str | None = None,
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
            "stderr": stderr or f"{directory}/stderr.log",
            "events": f"{directory}/events.jsonl",
            "events_exclude": STREAMING_EVENTS,
        }
        if raise_oom_score_adj:
            # Pi and its tools inherit the score; execd and the supervisor keep the sandbox's own.
            params["setup"] = prefer_oom_kill_setup()
        await upload_text(self.session.sandbox, path=f"{directory}/params.sh", text=shell_assignments(params))
        return SandboxCommand(argv=["bash", f"{directory}/{HARNESS_RUNNER_FILE}", f"{directory}/params.sh"])

    async def collect_artifacts(self, *, transcript: str | None = None) -> str:
        """Keep partial events even when worker metadata is unavailable; copy them to ``transcript`` if set."""
        try:
            events = await self.read_text("events.jsonl")
        except Exception as error:
            logs = await self.session.read_output_log()
            raise RuntimeError(f"Pi sandbox runner returned no valid result: {logs}") from error
        if transcript:
            try:
                await upload_text(self.session.sandbox, path=transcript, text=event_lines(events))
            except Exception:
                LOG.warning("Pi transcript could not be written to %s", transcript, exc_info=True)
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


def event_lines(events: str) -> str:
    """Pi's JSON lines as it printed them, from the runner's ``[seen_at, line]`` records."""
    lines = []
    for record in events.split("\n"):
        # The runner writes each line verbatim after the time it was seen.
        if record.startswith("[") and record.endswith("]") and ", " in record:
            lines.append(record[record.index(", ") + 2 : -1])
    return "".join(f"{line}\n" for line in lines)
