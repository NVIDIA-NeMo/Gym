# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one Hermes conversation inside a Gym sandbox."""

from __future__ import annotations

import functools
import json
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4


try:
    from .model_kwargs import _model_api_kwargs
    from .sandbox_observer import SandboxHermesObserver
except ImportError:
    from model_kwargs import _model_api_kwargs
    from sandbox_observer import SandboxHermesObserver


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(path)


_MODEL_API_KEY = "gym"


def _use_model_server(base_url: str) -> None:
    """Point every model client Hermes builds in this process at the Gym Model Server.

    The root agent and its iteration-limit summary share one client, delegated children inherit the
    parent's base URL, and auxiliary clients such as context compression read ``OPENAI_BASE_URL``.
    """
    from run_agent import AIAgent

    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = _MODEL_API_KEY

    # The Model Server answers only whole responses, so no agent may stream, including the children Hermes builds.
    initialize = AIAgent.__init__

    @functools.wraps(initialize)
    def initialize_without_streaming(agent: AIAgent, *args: Any, **kwargs: Any) -> None:
        initialize(agent, *args, **{**kwargs, "use_streaming": False})

    AIAgent.__init__ = initialize_without_streaming


def _connect_mcp_servers(required: list[str]) -> None:
    """Connect the MCP servers config.yaml lists for this episode and register their tools.

    Hermes discovers MCP servers when ``run_agent`` is imported, which is before this episode's config exists.
    """
    from tools.mcp_tool import discover_mcp_tools, get_mcp_status

    discover_mcp_tools()
    connected = {server["name"] for server in get_mcp_status() if server["connected"]}
    missing = sorted(set(required) - connected)
    if missing:
        raise RuntimeError("Required MCP servers did not connect: " + ", ".join(missing))


def _run(payload: dict[str, Any], session_dir: Path, *, output_path: Path | None = None) -> dict[str, Any]:
    from run_agent import AIAgent

    hermes_home = session_dir / "hermes-home"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(payload["config_yaml"])
    os.environ["HERMES_HOME"] = str(hermes_home)
    os.environ["TERMINAL_ENV"] = "local"
    os.environ["TERMINAL_TIMEOUT"] = str(payload["terminal_timeout"])
    _use_model_server(payload["model_base_url"])
    if payload["mcp_servers"]:
        _connect_mcp_servers(payload["required_mcp_servers"])

    agent = AIAgent(
        base_url=payload["model_base_url"],
        api_key=_MODEL_API_KEY,
        model=payload["model"],
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
    observer = SandboxHermesObserver().instrument(agent)

    original_build_api_kwargs = agent._build_api_kwargs

    def build_api_kwargs(api_messages: list[dict[str, Any]]) -> dict[str, Any]:
        return _model_api_kwargs(
            original_build_api_kwargs(api_messages),
            preserve_reasoning_history=payload["chat_template_kwargs_enabled"],
            model_enable_thinking=payload.get("model_enable_thinking"),
        )

    agent._build_api_kwargs = build_api_kwargs
    result = None
    error = None
    timed_out = False
    runtime = {"hostname": os.uname().nodename, "pid": os.getpid(), "python": sys.executable}

    def progress() -> dict[str, Any]:
        return {
            "completed": False,
            "interrupted": True,
            "stop_reason": "wall_time",
            "messages": getattr(agent, "_session_messages", [])
            or [*payload["history"], {"role": "user", "content": payload["user_message"]}],
        }

    def interrupt(*_: object) -> None:
        nonlocal timed_out
        timed_out = True
        # Save first: a blocked tool or API call may not unwind before the hard cleanup.
        if output_path is not None:
            partial = progress()
            _write_atomic(
                output_path,
                {"result": partial, "observations": observer.finish(partial, None), "runtime": runtime},
            )
        agent.interrupt("sandbox timeout")

    previous_handler = signal.signal(signal.SIGTERM, interrupt)
    try:
        result = agent.run_conversation(
            payload["user_message"],
            payload["system_message"],
            payload["history"],
            task_id=payload["agent_session_id"],
        )
    except BaseException as exception:
        if timed_out:
            result = progress()
        else:
            error = exception
            setattr(exception, "_sandbox_observations", observer.finish(result, error))
            raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    if timed_out and not result.get("failed"):
        result = {**result, "completed": False, "interrupted": True, "stop_reason": "wall_time"}
    return {
        "observations": observer.finish(result, error),
        "result": result,
        "runtime": runtime,
    }


def _run_worker(input_path: Path, output_path: Path) -> int:
    exchange_dir = input_path.parent
    try:
        output = _run(json.loads(input_path.read_text()), exchange_dir, output_path=output_path)
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


def main() -> int:
    """Run the Hermes worker; the uploaded process supervisor owns its lifetime."""
    if len(sys.argv) != 3:
        print("usage: sandbox_runner.py INPUT_JSON OUTPUT_JSON", file=sys.stderr)
        return 2
    input_path, output_path = map(Path, sys.argv[-2:])
    return _run_worker(input_path, output_path)


if __name__ == "__main__":
    raise SystemExit(main())
