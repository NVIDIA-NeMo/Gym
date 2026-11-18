# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one Hermes conversation inside a Gym sandbox."""

from __future__ import annotations

import json
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(path)


_MODEL_API_KEY = "gym"
_COMPRESSION_CONFIG_KEY = "compression"


def _use_model_server(base_url: str, api_key: str = _MODEL_API_KEY) -> None:
    """Point every model client Hermes builds in this process at the Gym Model Server.

    The root agent and its iteration-limit summary share one client, delegated children inherit the
    parent's base URL. Compression's auxiliary client is configured in the episode's config.yaml.
    """
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = api_key


def _connect_mcp_servers(required: list[str]) -> None:
    """Refresh this episode's MCP tools and verify that every required server connected."""
    from tools.mcp_tool_discovery import discover_mcp_tools, get_mcp_status

    discover_mcp_tools()
    connected = {server["name"] for server in get_mcp_status() if server["connected"]}
    missing = sorted(set(required) - connected)
    if missing:
        raise RuntimeError("Required MCP servers did not connect: " + ", ".join(missing))


def _run(payload: dict[str, Any], session_dir: Path, *, output_path: Path | None = None) -> dict[str, Any]:
    # Hermes reads .env/config and caches state paths at import time. Configure its session home before any
    # harness imports, including sibling modules that may import Hermes transitively; preserve the task's HOME.
    hermes_home = session_dir / "hermes-home"
    hermes_home.mkdir(parents=True, exist_ok=True)
    import hermes_yaml as yaml

    api_key = payload.get("api_key", _MODEL_API_KEY)
    config = yaml.safe_load(payload["config_yaml"]) or {}
    config.setdefault("tools", {}).setdefault("tool_search", {})["enabled"] = "off"
    if any(
        isinstance(content, list)
        for content in [payload["user_message"], *(m.get("content") for m in payload["history"])]
    ):
        # Gym validated these input parts and the model route owns the capability decision.
        model_config = config.get("model")
        if not isinstance(model_config, dict):
            model_config = {"default": payload["model"]}
            config["model"] = model_config
        model_config["supports_vision"] = True
    config.setdefault("auxiliary", {})[_COMPRESSION_CONFIG_KEY] = {
        "provider": "custom",
        "base_url": payload.get("compression_model_base_url", payload["model_base_url"]),
        "api_key": api_key,
        "model": payload["model"],
    }
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(config))
    os.environ["HERMES_HOME"] = str(hermes_home)
    os.environ["TERMINAL_ENV"] = payload.get("terminal_backend", "local")
    os.environ["TERMINAL_TIMEOUT"] = str(payload["terminal_timeout"])
    _use_model_server(payload["model_base_url"], api_key)

    try:
        from .model_kwargs import configure_model_calls
        from .sandbox_observer import SESSION_TOKEN_COUNTERS, SandboxHermesObserver
    except ImportError:
        from model_kwargs import configure_model_calls
        from sandbox_observer import SESSION_TOKEN_COUNTERS, SandboxHermesObserver
    from run_agent import AIAgent

    if payload["mcp_servers"]:
        _connect_mcp_servers(payload["required_mcp_servers"])

    agent = AIAgent(
        base_url=payload["model_base_url"],
        api_key=api_key,
        model=payload["model"],
        provider="custom",
        api_mode="chat_completions",
        request_overrides={"temperature": payload["temperature"]} if payload["temperature"] is not None else {},
        reasoning_config={"enabled": True},
        skip_background_review=True,
        max_iterations=payload["max_turns"],
        max_tokens=payload["max_tokens"],
        enabled_toolsets=payload["enabled_toolsets"],
        disabled_toolsets=payload["disabled_toolsets"],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
    )
    configure_model_calls(agent, preserve_reasoning_history=payload["chat_template_kwargs_enabled"])
    observer = SandboxHermesObserver().instrument(agent, capture_summaries=True)
    result = None
    error = None
    stop_reason = None
    runtime = {"hostname": os.uname().nodename, "pid": os.getpid(), "python": sys.executable}

    def progress() -> dict[str, Any]:
        return {
            "completed": False,
            "interrupted": True,
            "stop_reason": stop_reason,
            "messages": getattr(agent, "_session_messages", [])
            or [*payload["history"], {"role": "user", "content": payload["user_message"]}],
        }

    def interrupt(*_: object) -> None:
        nonlocal stop_reason
        # Close publishes this marker before signalling the supervisor. A deadline does not.
        stop_request_path = payload.get("stop_request_path")
        stop_reason = "cancelled" if stop_request_path and Path(stop_request_path).exists() else "wall_time"
        # Save first: a blocked tool or API call may not unwind before the hard cleanup.
        if output_path is not None:
            partial = progress()
            _write_atomic(
                output_path,
                {"result": partial, "observations": observer.finish(partial, None), "runtime": runtime},
            )
        agent.interrupt("sandbox cancellation" if stop_reason == "cancelled" else "sandbox timeout")

    previous_handler = signal.signal(signal.SIGTERM, interrupt)
    try:
        result = agent.run_conversation(
            payload["user_message"],
            payload["system_message"],
            payload["history"],
            task_id=payload["agent_session_id"],
        )
        if getattr(agent, "_gym_iteration_limit_reached", False):
            result.setdefault("stop_reason", "max_iterations")
    except BaseException as exception:
        if stop_reason is not None:
            result = progress()
        else:
            error = exception
            setattr(exception, "_sandbox_observations", observer.finish(result, error))
            raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    if stop_reason is not None and not result.get("failed"):
        result = {**result, "completed": False, "interrupted": True, "stop_reason": stop_reason}
    # Hermes omits aggregates on early returns. Preserve its actual counters, including zero
    # before any completed call, without manufacturing usage absent from the runtime.
    result = dict(result)
    for field, counter in SESSION_TOKEN_COUNTERS.items():
        count = getattr(agent, counter, None)
        if field not in result and type(count) is int and count >= 0:
            result[field] = count
    return {
        "observations": observer.finish(result, error),
        "result": observer.restore_messages(result),
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
    if sys.argv[1] == "--host":
        # Isolate the result channel from third-party stdout and close it on tool exec.
        with os.fdopen(os.dup(sys.stdout.fileno()), "w") as result_pipe:
            os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
            try:
                result = _run(json.load(sys.stdin), Path(sys.argv[2]))
            except Exception as error:
                json.dump(
                    {
                        "error": str(error),
                        "error_type": type(error).__name__,
                        "observations": getattr(error, "_sandbox_observations", None),
                    },
                    result_pipe,
                )
                return 1
            json.dump(result, result_pipe)
        return 0
    input_path, output_path = map(Path, sys.argv[-2:])
    return _run_worker(input_path, output_path)


if __name__ == "__main__":
    raise SystemExit(main())
