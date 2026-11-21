# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Codex artifacts and request state over the shared sandbox lifecycle."""

import asyncio
import errno
import hashlib
import json
import logging
import re
import shlex
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.agents import (
    HARNESS_RUNNER_FILE,
    AgentSessionState,
    CleanupReceipt,
    SandboxCommand,
    SandboxSession,
    shell_assignments,
)
from nemo_gym.core import MediaFile, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult, read_text, stage_media, upload_text
from nemo_gym.trajectory import AgentObservationBundle


LOG = logging.getLogger(__name__)
# Uploaded beside the runner; runs Codex behind the 429 forwarder on the runtime's Node.
LAUNCHER_FILE = "sandbox_launcher.mjs"
# The longest the optional session-rollout copy may take, and the part of the capture budget kept free after it.
_ARCHIVE_TIMEOUT_S = 120.0
_ARCHIVE_RESERVE_S = 1.0
# Session files kept for a failed activation: supervisor + harness diagnostics first.
FAILURE_ARTIFACT_FILES = (
    "output.log",
    "stderr.log",
    "cleanup.json",
    "runtime.json",
    "supervisor.pid",
    "stop.request",
    "events.jsonl",
)
FAILURE_ARTIFACT_MAX_BYTES = 16 * 1024 * 1024
# Suffix of the trimmed copy of a session file that preserve_failure_artifacts downloads.
FAILURE_ARTIFACT_TAIL_SUFFIX = ".gym-tail"
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


def _snapshot_failure_details(
    snapshot: SandboxExecResult | None, error: Exception, *, phase: str
) -> dict[str, JsonValue]:
    """Keep provider evidence without logging artifact contents or exception messages.

    A provider return code may be a runtime sentinel. Timeout evidence does not
    establish that the in-sandbox Node process was killed. Byte counts/hashes
    describe the available stderr string encoded as UTF-8 with replacement.
    """
    return_code = getattr(snapshot, "return_code", None)
    provider_error = getattr(snapshot, "error_type", None)
    stderr = getattr(snapshot, "stderr", None)
    encoded = str.encode(stderr, "utf-8", "replace") if isinstance(stderr, str) else None
    safe_provider_error = (
        provider_error
        if isinstance(provider_error, str) and provider_error in {"timeout", "sandbox", "user_switch_unavailable"}
        else None
    )
    known_codes = {
        "EACCES",
        "EIO",
        "EISDIR",
        "ELOOP",
        "EMFILE",
        "ENAMETOOLONG",
        "ENFILE",
        "ENOENT",
        "ENOMEM",
        "ENOSPC",
        "ENOTDIR",
        "ENOTSUP",
        "EPERM",
        "EROFS",
        "ETIMEDOUT",
        "EINVAL",
    }
    # Only recognized public Node/OS codes leave stderr; never paths or arbitrary text.
    scan = stderr[:65536] if isinstance(stderr, str) else ""
    codes = sorted(set(re.findall(r"(?m)^[ \t]+code:[ \t]*['\"]([A-Z0-9_]+)['\"]", scan)) & known_codes)
    known_numbers = {getattr(errno, code) for code in known_codes if hasattr(errno, code)}
    errnos = sorted(
        {
            int(value)
            for value in re.findall(r"(?m)^[ \t]+errno:[ \t]*(-?[0-9]{1,4})[ \t]*,?$", scan)
            if abs(int(value)) in known_numbers
        }
    )
    exception_errno = error.errno if isinstance(error, OSError) else None
    error_class = type(error).__name__
    return {
        "phase": phase if phase in {"exec", "read"} else "unknown",
        "provider_return_code": return_code if type(return_code) is int else None,
        "provider_error_type": safe_provider_error,
        "provider_error_type_sha256": (
            hashlib.sha256(str.encode(provider_error, "utf-8", "replace")).hexdigest()
            if isinstance(provider_error, str) and safe_provider_error is None
            else None
        ),
        "timeout_observed": isinstance(error, TimeoutError)
        or (isinstance(provider_error, str) and provider_error == "timeout"),
        "exception_class": error_class if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", error_class) else "Exception",
        "exception_errno": exception_errno if type(exception_errno) is int else None,
        "stderr_sha256": hashlib.sha256(encoded).hexdigest() if encoded is not None else None,
        "stderr_utf8_bytes": len(encoded) if encoded is not None else None,
        "stderr_errno_codes": codes,
        "stderr_errnos": errnos,
        "stderr_errno_scan_truncated": isinstance(stderr, str) and len(stderr) > 65536,
    }


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


def harness_kill_evidence(
    *,
    cleanup: CleanupReceipt | None,
    cleanup_receipt_missing: bool,
    launch_return_code: int | None,
) -> str | None:
    """Describe a harness process that was killed although nobody asked it to stop, else None.

    Only call this when the controller neither cancelled the activation nor hit the deadline:
    then the supervisor, the runner and Codex end only on their own, or because something in
    the sandbox (in practice an agent tool command, rarely the OOM killer) killed them. The
    receipt's return code is the runner's shell status: 128+N when it, or Codex under it,
    ended by signal N (N up to 64). This says the harness was killed, not by whom: 137 is
    also a memory-cgroup OOM kill and 134 Node's heap abort.

    A missing receipt counts only when the launch shell itself ended by a signal or its
    status is unknown; a launch that exited normally (0, 2, 127, ...) before the
    supervisor ran is a launch failure, not a kill.
    """
    if cleanup_receipt_missing:
        if launch_return_code is None or 128 < launch_return_code <= 128 + 64:
            return f"the process supervisor exited ({launch_return_code}) without writing its cleanup receipt"
        return None
    if cleanup is None or cleanup["timed_out"]:
        return None
    return_code = cleanup["return_code"]
    if return_code is not None and 128 < return_code <= 128 + 64:
        return f"the Codex runner or Codex was killed by signal {return_code - 128}"
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

    async def execute(
        self,
        payload: dict[str, JsonValue],
        *,
        timeout: float,
        close_timeout: float,
        files: dict[str, str] | None = None,
        media: Sequence[MediaFile] = (),
        deadline: float | None = None,
    ) -> str:
        """Run through the common lifecycle; the adapter validates terminal events.

        ``files`` and prompt ``media`` are staged with the invocation; the deadline bounds both staging and execution.
        """
        self.capture_budget_s = close_timeout
        return await self.session.execute(
            stage_activation=lambda: self.stage_activation(payload, files=files, media=media),
            collect=self.collect_artifacts,
            timeout=timeout,
            close_timeout=close_timeout,
            deadline=deadline,
        )

    async def stage_activation(
        self,
        payload: dict[str, JsonValue],
        *,
        files: dict[str, str] | None = None,
        media: Sequence[MediaFile] = (),
    ) -> SandboxCommand:
        """Stage the invocation and return the runner command that starts the Codex launcher.

        ``files`` maps paths under the session directory to text, kept out of ``input.json``. They are uploaded here,
        inside the session's execute, so that close cancels and waits for the uploads before it releases the
        sandbox.
        """
        directory = self.session.session_dir
        for name, text in (files or {}).items():
            await self.upload_text(name, text)
        if media:
            await stage_media(self.session.sandbox, media)
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

    async def _read_tail(self, name: str, max_bytes: int) -> str:
        """The last ``max_bytes`` of a session file, trimmed inside the sandbox before the download, so a huge log
        never reaches the Agent Server whole."""
        path = f"{self.session.session_dir}/{name}"
        tail = f"{name}{FAILURE_ARTIFACT_TAIL_SUFFIX}"
        result = await self.session.sandbox.exec(
            f"tail -c {int(max_bytes)} {shlex.quote(path)} > {shlex.quote(f'{self.session.session_dir}/{tail}')}",
            cwd="/",
            timeout_s=30,
        )
        if result.return_code != 0 or result.error_type:
            raise FileNotFoundError(f"{name}: {(result.stderr or '').strip()[:200] or result.error_type}")
        return await self.read_text(tail)

    async def preserve_failure_artifacts(self, root: str, detail: dict[str, JsonValue]) -> Path | None:
        """Copy the session's diagnostics to ``root/<episode>-<id>/`` before the sandbox is released.

        Best effort: a file that cannot be read is listed in failure.json, and an
        error here never replaces the activation's own failure.
        """
        try:
            key = "".join(c if c.isalnum() or c in "-_." else "_" for c in self.request.episode_id.capture_key)
            target = Path(root) / f"{key}-{uuid4().hex[:8]}"
            copied, unavailable = [], {}
            for name in FAILURE_ARTIFACT_FILES:
                try:
                    text = await self._read_tail(name, FAILURE_ARTIFACT_MAX_BYTES)
                except Exception as error:  # noqa: BLE001 - diagnostics only
                    unavailable[name] = f"{type(error).__name__}: {error}"[:300]
                    continue
                data = text.encode()
                if len(data) > FAILURE_ARTIFACT_MAX_BYTES:
                    data = data[-FAILURE_ARTIFACT_MAX_BYTES:]
                await asyncio.to_thread(_write_file, target / name, data)
                copied.append(name)
            summary = {
                **detail,
                "session_dir": self.session.session_dir,
                "saved_at": time.time(),
                "copied": copied,
                "unavailable": unavailable,
            }
            await asyncio.to_thread(_write_file, target / "failure.json", json.dumps(summary, indent=1).encode())
            LOG.warning("Codex failure artifacts preserved in %s", target)
            return target
        except Exception:  # noqa: BLE001 - diagnostics only
            LOG.warning("Could not preserve Codex failure artifacts", exc_info=True)
            return None

    async def collect_artifacts(self) -> str:
        """Keep partial events even when worker metadata is unavailable."""
        if self.session.expired_before_launch:
            return ""  # A rollout stop arrived before launch, so Codex never ran.
        started = time.monotonic()
        try:
            events = await self.read_text("events.jsonl")
        except Exception as error:
            if self.session.expired and await self._events_absent():
                # A rollout stop can end the worker after launch but before it opens its transcript.
                LOG.warning("Codex wrote no events before a rollout stop ended it; keeping an empty transcript")
                return ""
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
        snapshot = None
        phase = "exec"
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
            phase = "read"
            self.native_transcript = await self.read_text("native_session.jsonl")
        except Exception as error:
            LOG.warning(
                "Codex native session transcript unavailable: %s",
                json.dumps(_snapshot_failure_details(snapshot, error, phase=phase), sort_keys=True),
            )
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

    async def _events_absent(self) -> bool:
        """Whether the session directory exists without events.jsonl; an unclear answer counts as present."""
        directory = shlex.quote(self.session.session_dir)
        events = shlex.quote(f"{self.session.session_dir}/events.jsonl")
        try:
            probe = await self.session.sandbox.exec(
                f"if [ -e {events} ]; then echo present; elif [ -d {directory} ]; then echo absent; fi", timeout_s=10
            )
        except Exception:
            LOG.warning("Could not check for Codex events after a rollout stop", exc_info=True)
            return False
        return probe.return_code == 0 and not probe.error_type and probe.stdout.strip() == "absent"


def _write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
