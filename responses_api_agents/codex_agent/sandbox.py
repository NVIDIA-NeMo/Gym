# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Codex artifacts and request state over the shared sandbox lifecycle."""

import asyncio
import json
import logging
import shlex
import time
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.agent_utils.sandbox_session import RUNNER_FILE, SandboxCommand, SandboxSession
from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox.utils import read_text, upload_text


LOG = logging.getLogger(__name__)
# The longest the optional session-rollout copy may take, and the part of the capture budget kept free after it.
_ARCHIVE_TIMEOUT_S = 120.0
_ARCHIVE_RESERVE_S = 1.0


class HarnessProcessInfo(BaseModel):
    """Optional Codex shim identity, independent of supervisor cleanup."""

    model_config = ConfigDict(extra="forbid", strict=True)
    hostname: str
    pid: int
    python: str | None = None


def parse_runtime_info(payload: object) -> HarnessProcessInfo | None:
    """Read Codex diagnostics without failing an otherwise valid episode."""
    try:
        return HarnessProcessInfo.model_validate(payload)
    except ValidationError:
        LOG.warning("Codex runtime metadata is missing or malformed")
        return None


@dataclass
class CodexSandboxSession(AgentSessionState):
    """Codex request/output state; SandboxSession owns execution and teardown."""

    session: SandboxSession[str]
    runtime: str
    logs_dir: str | None = None
    task: asyncio.Task[NeMoGymResponse] | None = None
    runtime_info: HarnessProcessInfo | None = None
    observations: AgentObservationBundle | None = None
    native_transcript: str | None = None
    model_attempts: list[dict[str, JsonValue]] | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    # The time SandboxSession allows artifact capture (its close timeout), so the optional copy fits inside it.
    capture_budget_s: float | None = None

    async def upload_json(self, name: str, payload: JsonValue) -> None:
        """Stage adapter input using the shared text-transfer utility."""
        await upload_text(self.session.sandbox, path=f"{self.session.session_dir}/{name}", text=json.dumps(payload))

    async def upload_text(self, name: str, text: str) -> None:
        """Stage Codex configuration using file transfer."""
        await upload_text(self.session.sandbox, path=f"{self.session.session_dir}/{name}", text=text)

    async def read_text(self, name: str) -> str:
        """Read adapter output using file transfer."""
        return await read_text(self.session.sandbox, path=f"{self.session.session_dir}/{name}")

    async def close(self, timeout: float) -> None:
        """Capture and release before cancelling the HTTP activation."""
        if self.capture_budget_s is None:
            self.capture_budget_s = timeout
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

    async def execute(self, payload: dict[str, JsonValue], *, timeout: float, close_timeout: float) -> str:
        """Run through the common lifecycle; the adapter validates terminal events."""
        self.capture_budget_s = close_timeout
        return await self.session.execute(
            stage_activation=lambda: self.stage_activation(payload),
            collect=self.collect_artifacts,
            timeout=timeout,
            close_timeout=close_timeout,
        )

    async def stage_activation(self, payload: dict[str, JsonValue]) -> SandboxCommand:
        """Stage the invocation and return the Codex worker command."""
        await self.upload_json("input.json", payload)
        return SandboxCommand(
            argv=[
                "python3",
                "-I",
                f"{self.session.session_dir}/{RUNNER_FILE}",
                f"{self.session.session_dir}/input.json",
            ],
            stdlib_runner=True,
        )

    async def copy_session_rollouts(self, deadline: float | None = None) -> None:
        """Copy Codex's session rollouts (CODEX_HOME/sessions) to logs_dir/sessions by ``deadline``
        (``time.monotonic()``); never fails collection."""
        if not self.logs_dir:
            return
        budget = _ARCHIVE_TIMEOUT_S if deadline is None else min(_ARCHIVE_TIMEOUT_S, deadline - time.monotonic())
        if budget <= 0:
            LOG.warning("Codex session rollouts not copied: no capture time left")
            return
        sessions = f"{self.session.session_dir}/home/.codex/sessions"
        target = f"{self.logs_dir}/sessions"
        command = (
            f"if [ -d {shlex.quote(sessions)} ]; then mkdir -p {shlex.quote(target)} && "
            f"cp -R {shlex.quote(sessions)}/. {shlex.quote(target)}/; fi"
        )
        try:
            async with asyncio.timeout(budget):
                result = await self.session.sandbox.exec(command, cwd="/", timeout_s=budget)
            if result.return_code != 0 or result.error_type:
                LOG.warning("Codex session rollouts not copied: %s", result.stderr or result.stdout)
        except Exception:
            LOG.exception("Codex session rollouts not copied")

    async def collect_artifacts(self) -> str:
        """Keep partial events even when worker metadata is unavailable."""
        started = time.monotonic()
        try:
            events = await self.read_text("events.jsonl")
        except Exception as error:
            logs = await self.session.read_output_log()
            raise RuntimeError(f"Codex sandbox runner returned no valid result: {logs}") from error
        try:
            runtime = await self.read_text("runtime.json")
        except Exception:
            runtime = None
        try:
            runtime_payload = json.loads(runtime) if runtime is not None else None
        except ValueError:
            runtime_payload = None
        self.runtime_info = parse_runtime_info(runtime_payload)
        try:
            snapshot = await self.session.sandbox.exec(
                f"python3 -I {shlex.quote(f'{self.session.session_dir}/{RUNNER_FILE}')} "
                f"--snapshot {shlex.quote(self.session.session_dir)}",
                timeout_s=10,
            )
            if snapshot.return_code != 0 or snapshot.error_type:
                raise RuntimeError("Codex native session snapshot failed")
            self.native_transcript = await self.read_text("native_session.jsonl")
        except Exception:
            LOG.warning("Codex native session transcript unavailable", exc_info=True)
        try:
            attempts = await self.read_text("model_attempts.jsonl")
            self.model_attempts = [json.loads(line) for line in attempts.splitlines()]
        except Exception:
            LOG.warning("Codex model attempt statuses unavailable", exc_info=True)
        # The optional copy goes last and ends before the capture budget does, so a slow copy cannot cost the events.
        if self.capture_budget_s is not None:
            await self.copy_session_rollouts(deadline=started + self.capture_budget_s - _ARCHIVE_RESERVE_S)
        else:
            await self.copy_session_rollouts()
        return events
