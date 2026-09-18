# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone runner for NousResearch/hermes-agent@v2026.9.7. No Gym imports.

Launch with the dedicated runtime's ``python -I runner.py request.json`` from
outside the task repository. TERMINAL_CWD independently selects the tool cwd.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path
from shlex import quote
from threading import Lock


def progress_result(agent, n_input):
    # Hermes persists the assistant's tool call before executing it, but updates
    # _session_messages only after the tool returns. Keep that in-flight call on timeout.
    messages = getattr(agent, "_db_flush_scan_prefix", None) or getattr(agent, "_session_messages", [])
    return {
        "completed": False,
        "messages": messages,
        "api_calls": getattr(agent, "_api_call_count", 0),
        "n_input": n_input,
        "usage": {
            "input_tokens": agent.session_input_tokens,
            "output_tokens": agent.session_output_tokens,
            "cached_tokens": agent.session_cache_read_tokens,
            "reasoning_tokens": agent.session_reasoning_tokens,
        },
    }


def classify_stop(result, timed_out=False):
    # This pinned Hermes version reports reasoning-only truncation as a partial
    # result with an error string. It is a model budget stop, not a harness crash.
    has_model_output = bool((result.get("usage") or {}).get("output_tokens")) or any(
        m.get("role") == "assistant" for m in result.get("messages", [])[result.get("n_input", 0) :]
    )
    if timed_out and has_model_output:
        result["budget_exhausted"] = True
        result["stop_reason"] = "wall_time"
    elif (
        result.get("partial")
        and not result.get("failed")
        and str(result.get("error", "")).startswith(
            "Model used all output tokens on reasoning with none left for the response."
        )
    ):
        result["budget_exhausted"] = True
        result["stop_reason"] = "output_tokens"
    elif result.get("budget_exhausted") and not (
        result.get("failed") or result.get("error") or result.get("interrupted")
    ):
        result["stop_reason"] = "max_turns"
    if result.get("stop_reason"):
        result["completed"] = False
        result["failed"] = False
        result["stop_detail"] = result.pop("error", None)
    return result


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, default=str))
    temporary.replace(path)


def split_input(items):
    if isinstance(items, str):
        return items, [], None
    messages = []
    system = []
    for item in items:
        role = item.get("role")
        if role not in ("system", "developer", "user", "assistant") or item.get("tool_calls"):
            raise ValueError("Hermes sandbox input must contain text messages")
        content = item.get("content", "")
        if isinstance(content, list):
            if any(p.get("type") not in ("input_text", "output_text", "text") for p in content):
                raise ValueError("Hermes sandbox input currently supports text only")
            content = "\n".join(p["text"] for p in content)
        if role in ("system", "developer"):
            system.append(content)
        else:
            messages.append({"role": role, "content": content})
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Hermes sandbox input must end with a user message")
    return messages[-1]["content"], messages[:-1], "\n\n".join(system) or None


def run(params):
    import yaml

    runtime_manifest = json.loads((Path(sys.prefix) / "hermes-runtime.json").read_text())
    source = Path(sys.prefix) / "hermes-src"

    home = Path(params["run_dir"]) / "home"
    home.mkdir(parents=True, exist_ok=True)
    tools_bin = Path(sys.prefix) / "tools" / "bin"
    ripgrep_version = subprocess.check_output([str(tools_bin / "rg"), "--version"], text=True).splitlines()[0]
    # Hermes initializes a login shell, whose /etc/profile can discard the image's
    # toolchain PATH. Restore it after that profile, without adding runtime Python.
    tool_path = f"{tools_bin}:{os.environ['PATH']}"
    shell_init = home / "shell-init.sh"
    shell_init.write_text(f"export PATH={quote(tool_path)}\n")
    # Hermes reads config and caches at import time. These are per process/task.
    os.environ.update(
        HOME=str(home),
        HERMES_HOME=str(home / ".hermes"),
        TERMINAL_ENV="local",
        TERMINAL_CWD=params["workdir"],
        TERMINAL_TIMEOUT=str(params["terminal_timeout"]),
        HERMES_API_TIMEOUT=str(params["api_timeout"]),
        HERMES_API_CALL_STALE_TIMEOUT=str(params["api_timeout"]),
        HERMES_YOLO_MODE="1",
        PATH=tool_path,
    )
    hermes_home = Path(os.environ["HERMES_HOME"])
    hermes_home.mkdir()
    config = {
        "model": {
            "default": params["model"],
            "provider": "custom",
            "base_url": params["base_url"],
            "streaming": False,  # Gym's Chat Completions endpoint is non-streaming.
            "context_length": params.get("context_length"),
        },
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
        "toolsets": ["hermes-cli"],
        "agent": {"max_turns": params["max_turns"]},
        "delegation": {"max_iterations": 50},
        "compression": {"enabled": params["compression_enabled"], "threshold": 0.85},
        # Summarization uses the same model; its separate metadata lookup must
        # not lower the main context window back to the catalog fallback.
        "auxiliary": {"compression": {"context_length": params.get("context_length")}},
        "terminal": {
            "backend": "local",
            "cwd": params["workdir"],
            "timeout": params["terminal_timeout"],
            "shell_init_files": [str(shell_init)],
        },
        "checkpoints": {"enabled": False},
    }
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(config))

    from hermes_state import SessionDB
    from run_agent import AIAgent

    if Path(sys.modules["run_agent"].__file__).resolve().parent != source.resolve():
        raise RuntimeError("Hermes was imported from outside the pinned runtime")

    query, history, input_system = split_input(params["input"])
    request_overrides = {"temperature": params["temperature"]}
    if params["chat_template_kwargs"]:
        request_overrides["metadata"] = {"chat_template_kwargs": json.dumps(params["chat_template_kwargs"])}
    # Hermes saves tool calls before execution only when a session store is supplied.
    session_db = SessionDB()
    agent = AIAgent(
        base_url=params["base_url"],
        api_key="gym",  # The sandbox talks only to Gym's model proxy, never receives provider credentials.
        provider="custom",
        api_mode="chat_completions",
        model=params["model"],
        max_iterations=params["max_turns"],
        max_tokens=params["max_tokens"],
        request_overrides=request_overrides,
        reasoning_config={"enabled": True},
        enabled_toolsets=params["enabled_toolsets"],
        disabled_toolsets=params["disabled_toolsets"],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        save_trajectories=False,
        checkpoints_enabled=False,
        session_db=session_db,
    )
    n_input = len(history) + 1
    timed_out = False
    checkpoint_lock = Lock()

    def checkpoint(*_):
        with checkpoint_lock:  # Tool-start callbacks can run concurrently.
            write_json(Path(params["run_dir"]) / "progress.json", progress_result(agent, n_input))

    def on_timeout(*_):
        nonlocal timed_out
        timed_out = True
        # Save before cancellation: a blocked tool or API worker may not unwind
        # before the sandbox provider's SIGKILL grace period expires.
        write_json(Path(params["run_dir"]) / "result.json", classify_stop(progress_result(agent, n_input), True))
        agent.interrupt("sandbox timeout", hard_cancel=True)

    agent.step_callback = checkpoint
    agent.tool_start_callback = checkpoint
    signal.signal(signal.SIGTERM, on_timeout)
    try:
        result = agent.run_conversation(query, params["system_prompt"] or input_system, history)
    except BaseException as exc:
        result = progress_result(agent, n_input) | {
            "failed": True,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        traceback.print_exc()
    finally:
        session_db.close()
    result["budget_exhausted"] = (
        agent.iteration_budget.remaining <= 0 or result.get("api_calls", 0) >= params["max_turns"]
    )
    result["n_input"] = len(history) + 1
    result["usage"] = {
        "input_tokens": agent.session_input_tokens,
        "output_tokens": agent.session_output_tokens,
        "cached_tokens": agent.session_cache_read_tokens,
        "reasoning_tokens": agent.session_reasoning_tokens,
    }
    result["runtime"] = {
        "hermes_commit": runtime_manifest["hermes_commit"],
        "python": sys.version,
        "run_agent_path": sys.modules["run_agent"].__file__,
        "sys_path": sys.path,
        "cwd": os.getcwd(),
        "tool_cwd": os.environ["TERMINAL_CWD"],
        "tool_path": tool_path,
        "ripgrep_version": ripgrep_version,
        "context_length": agent.context_compressor.context_length,
        "compression_threshold": agent.context_compressor.threshold_tokens,
    }
    return classify_stop(result, timed_out)


def main():
    # Configure before importing Hermes so startup and turn logs reach captured stderr.
    logging.basicConfig(level=logging.INFO)
    params = json.loads(Path(sys.argv[1]).read_text())
    try:
        result = run(params)
    except BaseException as exc:
        result = {"completed": False, "failed": True, "error": str(exc), "error_type": type(exc).__name__}
        traceback.print_exc()
    write_json(Path(params["run_dir"]) / "result.json", result)
    return 1 if result.get("failed") or result.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
