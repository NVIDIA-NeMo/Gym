# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone runner for NousResearch/hermes-agent@v2026.8.31. No Gym imports.

Launch with the dedicated runtime's ``python -I runner.py request.json`` from
outside the task repository. TERMINAL_CWD independently selects the tool cwd.
"""

import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path


HERMES_COMMIT = "29112bef099274229cadff79cdff7bf7b99c4b77"


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
    if runtime_manifest["hermes_commit"] != HERMES_COMMIT:
        raise RuntimeError("Runtime must contain NousResearch/hermes-agent@v2026.8.31")
    source = Path(sys.prefix) / "hermes-src"
    commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={source}", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != HERMES_COMMIT:
        raise RuntimeError("Installed Hermes provenance does not match the required commit")

    home = Path(params["run_dir"]) / "home"
    home.mkdir(parents=True, exist_ok=True)
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
    )
    hermes_home = Path(os.environ["HERMES_HOME"])
    hermes_home.mkdir()
    config = {
        "model": {"default": params["model"], "provider": "custom", "base_url": params["base_url"]},
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
        "compression": {"enabled": params["compression_enabled"], "threshold": 0.85},
        "terminal": {"backend": "local", "cwd": params["workdir"], "timeout": params["terminal_timeout"]},
        "checkpoints": {"enabled": False},
    }
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(config))

    from run_agent import AIAgent

    if Path(sys.modules["run_agent"].__file__).resolve().parent != source.resolve():
        raise RuntimeError("Hermes was imported from outside the pinned runtime")

    query, history, input_system = split_input(params["input"])
    agent = AIAgent(
        base_url=params["base_url"],
        api_key="gym",  # The sandbox talks only to Gym's model proxy, never receives provider credentials.
        provider="custom",
        api_mode="chat_completions",
        model=params["model"],
        max_iterations=params["max_turns"],
        max_tokens=params["max_tokens"],
        request_overrides={"temperature": params["temperature"]},
        reasoning_config={"enabled": True},
        enabled_toolsets=params["enabled_toolsets"],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        save_trajectories=False,
        checkpoints_enabled=False,
    )
    # v2026.8.31 removed use_streaming and streams even without a consumer.
    # Gym's Chat Completions endpoint accepts only stream=false.
    agent._disable_streaming = True
    original_build = agent._build_api_kwargs

    def build(api_messages, tools_for_api=None):
        kwargs = original_build(api_messages, tools_for_api=tools_for_api)
        kwargs["stream"] = False
        if params["chat_template_kwargs"]:
            kwargs.setdefault("extra_body", {})["chat_template_kwargs"] = params["chat_template_kwargs"]
        return kwargs

    agent._build_api_kwargs = build
    signal.signal(signal.SIGTERM, lambda *_: agent.interrupt("sandbox timeout"))
    result = agent.run_conversation(query, params["system_prompt"] or input_system, history)
    result["n_input"] = len(history) + 1
    result["usage"] = {
        "input_tokens": agent.session_input_tokens,
        "output_tokens": agent.session_output_tokens,
        "cached_tokens": agent.session_cache_read_tokens,
        "reasoning_tokens": agent.session_reasoning_tokens,
    }
    result["runtime"] = {
        "hermes_commit": HERMES_COMMIT,
        "python": sys.version,
        "run_agent_path": sys.modules["run_agent"].__file__,
        "sys_path": sys.path,
        "cwd": os.getcwd(),
        "tool_cwd": os.environ["TERMINAL_CWD"],
    }
    return result


def main():
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
