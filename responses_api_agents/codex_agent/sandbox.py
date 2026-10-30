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

from nemo_gym.agent_utils.sandbox_session import (
    HARNESS_RUNNER_FILE,
    SandboxCommand,
    SandboxSession,
    shell_assignments,
)
from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox.utils import read_text, upload_text


LOG = logging.getLogger(__name__)
# Uploaded beside the runner; runs Codex behind the 429 forwarder on the runtime's Node.
LAUNCHER_FILE = "sandbox_launcher.mjs"
# The longest the optional session-rollout copy may take, and the part of the capture budget kept free after it.
_ARCHIVE_TIMEOUT_S = 120.0
_ARCHIVE_RESERVE_S = 1.0
# Runner setup for raise_oom_score_adj: make the runner, and with it Codex and every tool, the memory-cgroup OOM
# killer's preferred victims. Raise oom_score_adj to 1000, or to 0 when the sandbox starts it below zero (a
# Guaranteed pod's -997), so the sandbox never becomes the node's first victim in a global OOM. Raising needs no
# privilege; a failure leaves the score unchanged. The supervisor and the exec daemon keep the sandbox's own score.
RAISE_OOM_SCORE_ADJ = """if read -r before 2>/dev/null </proc/self/oom_score_adj && [[ $before =~ ^-?[0-9]+$ ]]; then
    after=1000
    ((before < 0)) && after=0
    if echo "$after" 2>/dev/null >/proc/self/oom_score_adj; then
        echo "oom_score_adj: $before -> $after" >&2
    else
        echo "oom_score_adj: unchanged" >&2
    fi
else
    echo "oom_score_adj: unchanged" >&2
fi
"""


class HarnessProcessInfo(BaseModel):
    """Optional Codex shim identity, independent of supervisor cleanup."""

    model_config = ConfigDict(extra="forbid", strict=True)
    hostname: str
    pid: int


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
        """Stage the invocation and return the runner command that starts the Codex launcher."""
        directory = self.session.session_dir
        await self.upload_json("input.json", payload)
        await self.upload_text("prompt.txt", payload["prompt"])
        params = {
            "cwd": self.session.workdir,
            "command": [f"{self.runtime}/node/bin/node", f"{directory}/{LAUNCHER_FILE}", f"{directory}/input.json"],
            "stdin": f"{directory}/prompt.txt",
            "stdout": f"{directory}/stdout.log",
            "stderr": f"{directory}/stderr.log",
            "events": f"{directory}/events.jsonl",
        }
        if not payload.get("inherit_env"):
            # The launcher passes Codex an allowlisted environment; task Node options must not reach the launcher.
            # A Codex given the task's whole environment sees them, as a plain codex exec in the task would.
            params["unset_env"] = ["NODE_OPTIONS"]
        if payload.get("raise_oom_score_adj"):
            # The runner's stderr is the supervisor's output.log, which records the change.
            params["setup"] = RAISE_OOM_SCORE_ADJ
        await self.upload_text("params.sh", shell_assignments(params))
        return SandboxCommand(argv=["bash", f"{directory}/{HARNESS_RUNNER_FILE}", f"{directory}/params.sh"])

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
                "NODE_OPTIONS= "
                + shlex.join(
                    [
                        f"{self.runtime}/node/bin/node",
                        f"{self.session.session_dir}/{LAUNCHER_FILE}",
                        "--snapshot",
                        self.session.session_dir,
                    ]
                ),
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
