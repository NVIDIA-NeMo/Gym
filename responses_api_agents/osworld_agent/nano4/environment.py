# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V4-owned execution feedback around an otherwise unchanged Gym DesktopEnv.

Wire schema/serialization/truncation authority:
35d35a7c38b356ff1549feda11572a4bceda4e49/convert_procua_sft.py

This module independently maps observed controller results into that schema.
The reference producer validates every resulting feedback block before use.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from typing import Any, Callable

from .agent import FEEDBACK_KEY
from .coordinate_compat import STEP1500_COORDINATES, extract_projection_audit
from .session import CURRENT_SESSION, Session


class CanonicalEnvironment:
    def __init__(self, backend: Any, session: Session, *, sleep: Callable[[float], None] = time.sleep) -> None:
        self.backend = backend
        self.session = session
        self.sleep = sleep
        self._active = False
        self._responses: list[Any] = []
        self._controller_error: Exception | None = None
        self._controller = backend.controller
        self._original_execute = backend.controller.execute_python_command

        def capture(command: str, *args: Any, **kwargs: Any) -> Any:
            try:
                result = self._original_execute(command, *args, **kwargs)
            except Exception as exc:
                if self._active:
                    self._controller_error = exc
                raise
            if self._active:
                self._responses.append(copy.deepcopy(result))
            return result

        backend.controller.execute_python_command = capture

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    def _payload(self, command: str) -> dict[str, Any]:
        stdout = ""
        stderr = ""
        message = ""
        returncode = None
        if self._controller_error is not None:
            error = self._controller_error
            import requests

            if isinstance(error, (TimeoutError, requests.Timeout)):
                status = "timeout"
            elif isinstance(error, requests.RequestException):
                status = "transport_error"
            else:
                status = "protocol_error"
            message = f"{type(error).__name__}: {error}"
        elif len(self._responses) != 1 or not isinstance(self._responses[0], dict):
            status = "protocol_error"
            message = f"Expected one complete controller response; observed {len(self._responses)}"
        else:
            result = self._responses[0]
            stdout = result.get("stdout", result.get("output", ""))
            stderr = result.get("stderr", result.get("error", ""))
            message = result.get("message", "")
            raw_code = result.get("returncode")
            valid_strings = all(isinstance(item, str) for item in (stdout, stderr, message))
            if type(raw_code) is int and valid_strings:
                returncode = raw_code
                status = "success" if returncode == 0 else "process_error"
            else:
                status = "protocol_error"
                stdout = stdout if isinstance(stdout, str) else repr(stdout)
                stderr = stderr if isinstance(stderr, str) else repr(stderr)
                message = "Controller response lacks an integer returncode or string output fields"
        command_hash = hashlib.sha256(command.encode()).hexdigest()
        signature = json.dumps([command_hash, status, returncode, stderr, message], separators=(",", ":"))
        if status == "success":
            self.session.previous_failure_signature = None
            self.session.consecutive_identical_failure_count = 0
        else:
            if signature == self.session.previous_failure_signature:
                self.session.consecutive_identical_failure_count += 1
            else:
                self.session.previous_failure_signature = signature
                self.session.consecutive_identical_failure_count = 1
        return {
            "schema_version": "command_execution_feedback.v1",
            "success": status == "success",
            "status": status,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "message": message,
            "model_command_sha256": command_hash,
            "truncated": False,
            "consecutive_identical_failure_count": self.session.consecutive_identical_failure_count,
        }

    def _feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        converter = self.session.reference.converter
        current = dict(payload)
        # Use the reference's serializer and UTF-8 cap primitives. Its strict
        # execution_feedback_text reader independently checks the resulting bytes.
        for field in ("stdout", "stderr", "message"):
            text = converter.render_execution_feedback_payload(current)
            if len(text.encode()) <= 4096:
                break
            current[field], _ = converter.cap_utf8(current[field], len(current[field].encode()) // 2)
            current["truncated"] = True
        else:
            text = converter.render_execution_feedback_payload(current)
            if len(text.encode()) > 4096:
                current.update(stdout="", stderr="", message="[truncated]", truncated=True)
                text = converter.render_execution_feedback_payload(current)
        return {
            "execution_feedback": payload,
            "execution_feedback_text": text,
            "execution_feedback_delivery": "delivered",
        }

    def step(self, action: str, pause: float = 0.5) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        pending = self.session.pending_action
        if pending is None:
            raise RuntimeError("V4 environment step has no corresponding parsed model action")
        permitted = {pending.desktop_action, f"exec({pending.desktop_action!r})"}
        if action not in permitted:
            raise ValueError("Desktop command differs from the task-local v4 parsed action")
        self._responses = []
        self._controller_error = None
        self._active = True
        try:
            if pending.name == "computer.wait":
                # The pinned argument-free schema advertises 20 seconds. Use
                # pause=0 for DesktopEnv's WAIT path to avoid its double pause.
                self.sleep(self.session.reference.lock["default_wait_seconds"])
                result = self.backend.step("WAIT", 0)
            else:
                try:
                    result = self.backend.step(action, pause)
                except Exception as exc:
                    if exc is not self._controller_error:
                        raise
                    # A command failure still requires a fresh observation. An
                    # observation failure propagates as an infrastructure error.
                    result = (self.backend._get_obs(), 0.0, False, {"command_exception": type(exc).__name__})
        finally:
            self._active = False
            self.session.pending_action = None
        observation, reward, done, info = result
        observation = dict(observation)
        info = dict(info or {})
        if pending.name == "exec_py":
            payload = self._payload(pending.model_source)
            if pending.coordinate_policy == STEP1500_COORDINATES:
                payload["stdout"], events = extract_projection_audit(payload["stdout"], pending.model_source)
                info["nemotron_v4_coordinate_projection"] = events
            feedback = self._feedback(payload)
            observation[FEEDBACK_KEY] = feedback
            info[FEEDBACK_KEY] = copy.deepcopy(feedback)
        else:
            observation["nemotron_v4_control_status"] = "success"
        return observation, reward, done, info

    def close(self) -> Any:
        try:
            return self.backend.close()
        finally:
            self._controller.execute_python_command = self._original_execute


def Nano4DesktopEnv(*args: Any, **kwargs: Any) -> CanonicalEnvironment:
    """Gym-compatible factory selected only by the independent v4 entrypoint."""
    session = CURRENT_SESSION.get()
    if "sandbox_provider" in kwargs:
        from responses_api_agents.osworld_agent.sandbox_desktop_env import SandboxDesktopEnv

        backend = SandboxDesktopEnv(*args, **kwargs)
    else:
        from desktop_env.desktop_env import DesktopEnv

        backend = DesktopEnv(*args, **kwargs)
    return CanonicalEnvironment(backend, session)
