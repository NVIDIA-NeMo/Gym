# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one Hermes conversation inside a Gym sandbox."""

from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4

from openai.types.chat import ChatCompletion


try:
    from .sandbox_observer import SandboxHermesObserver
except ImportError:
    from sandbox_observer import SandboxHermesObserver


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(path)


class FileModelRelay:
    """Exchange sequential model requests with the controlling agent server."""

    def __init__(self, exchange_dir: Path) -> None:
        self.exchange_dir = exchange_dir
        self.request_index = 0

    def call(self, api_kwargs: dict[str, Any]) -> ChatCompletion:
        request_id = self.request_index
        self.request_index += 1
        request_path = self.exchange_dir / f"model-request-{request_id}.json"
        response_path = self.exchange_dir / f"model-response-{request_id}.json"
        _write_atomic(request_path, api_kwargs)

        while not response_path.exists():
            time.sleep(0.1)

        payload = json.loads(response_path.read_text())
        response_path.unlink()
        if payload.get("error") is not None:
            raise RuntimeError(str(payload["error"]))
        return ChatCompletion.model_validate(payload["response"])


def _run(payload: dict[str, Any], exchange_dir: Path) -> dict[str, Any]:
    from run_agent import AIAgent

    hermes_home = exchange_dir / "hermes-home"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(payload["config_yaml"])
    os.environ["HERMES_HOME"] = str(hermes_home)
    os.environ["TERMINAL_ENV"] = "local"
    os.environ["TERMINAL_TIMEOUT"] = str(payload["terminal_timeout"])

    agent = AIAgent(
        base_url="http://nemo-gym-model-relay.invalid/v1",
        api_key="model-relay",
        model=payload["model"],
        use_streaming=False,
        temperature=payload["temperature"],
        insert_reasoning=True,
        max_iterations=payload["max_turns"],
        max_tokens=payload["max_tokens"],
        enabled_toolsets=payload["enabled_toolsets"],
        disabled_toolsets=payload["disabled_toolsets"],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        persist_session=False,
        save_trajectories=False,
    )
    relay = FileModelRelay(exchange_dir)
    agent._interruptible_api_call = relay.call
    observer = SandboxHermesObserver().instrument(agent)

    original_build_api_kwargs = agent._build_api_kwargs

    def build_api_kwargs(api_messages: list[dict[str, Any]]) -> dict[str, Any]:
        kwargs = original_build_api_kwargs(api_messages)
        if not payload["chat_template_kwargs_enabled"]:
            return kwargs
        chat_template_kwargs = kwargs.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})
        chat_template_kwargs.setdefault("enable_thinking", True)
        chat_template_kwargs["truncate_history_thinking"] = False
        return kwargs

    agent._build_api_kwargs = build_api_kwargs
    result = None
    error = None
    try:
        result = agent.run_conversation(
            payload["user_message"],
            payload["system_message"],
            payload["history"],
            task_id=payload["agent_session_id"],
        )
    except BaseException as exception:
        error = exception
        setattr(exception, "_sandbox_observations", observer.finish(result, error))
        raise
    return {
        "observations": observer.finish(result, error),
        "result": result,
        "runtime": {
            "hostname": os.uname().nodename,
            "pid": os.getpid(),
            "python": sys.executable,
        },
    }


def _run_worker(input_path: Path, output_path: Path) -> int:
    exchange_dir = input_path.parent
    try:
        output = _run(json.loads(input_path.read_text()), exchange_dir)
    except BaseException as error:
        output = {
            "error": str(error),
            "error_type": type(error).__name__,
            "observations": getattr(error, "_sandbox_observations", None),
            "traceback": traceback.format_exc(),
            "runtime": {
                "hostname": os.uname().nodename,
                "pid": os.getpid(),
                "python": sys.executable,
            },
        }
        _write_atomic(output_path, output)
        return 1

    _write_atomic(output_path, output)
    return 0


def _drain_children(timeout: float) -> None:
    """Kill and reap adopted descendants, including tools that start a new process group."""
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    deadline = time.monotonic() + timeout
    while True:
        for child in children.read_text().split():
            try:
                os.kill(int(child), signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Hermes descendants remain alive; verification must not proceed")
        time.sleep(0.01)


def _supervise(command: list[str], *, cleanup_timeout: float) -> dict[str, Any]:
    """Run Hermes as a child and acknowledge cleanup only after its descendants are gone."""
    process = None
    cleanup_confirmed = False
    error = None
    stopping = False

    def interrupt(*_: object) -> None:
        nonlocal stopping
        # Do not interrupt Popen between process creation and handle assignment.
        stopping = True

    signal.signal(signal.SIGTERM, interrupt)
    try:
        if sys.platform != "linux":
            raise RuntimeError("Native Hermes sessions require a Linux sandbox")
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), "Cannot establish Hermes child-subreaper boundary")
        process = subprocess.Popen(command, start_new_session=True)
        while process.poll() is None and not stopping:
            time.sleep(0.05)
    except Exception as exception:
        error = str(exception)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=cleanup_timeout)
                _drain_children(cleanup_timeout)
            cleanup_confirmed = True
        except Exception as exception:
            error = f"cleanup: {exception}"
    return {"cleanup_confirmed": cleanup_confirmed, "error": error}


def main() -> int:
    worker = len(sys.argv) == 4 and sys.argv[1] == "--worker"
    if not worker and len(sys.argv) != 3:
        print("usage: sandbox_runner.py [--worker] INPUT_JSON OUTPUT_JSON", file=sys.stderr)
        return 2
    input_path, output_path = map(Path, sys.argv[-2:])
    if worker:
        return _run_worker(input_path, output_path)
    payload = json.loads(input_path.read_text())
    receipt = _supervise(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(input_path), str(output_path)],
        cleanup_timeout=payload["cleanup_timeout"],
    )
    _write_atomic(input_path.parent / "cleanup.json", receipt)
    return 0 if receipt["cleanup_confirmed"] and receipt["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
