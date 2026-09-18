# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one Hermes conversation inside a Gym sandbox."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4

from openai.types.chat import ChatCompletion
from run_agent import AIAgent


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


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: sandbox_runner.py INPUT_JSON OUTPUT_JSON", file=sys.stderr)
        return 2

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
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


if __name__ == "__main__":
    raise SystemExit(main())
