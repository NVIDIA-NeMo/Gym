# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable task inventory, OpenCode configuration, and native score handling."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from saas_bench.loader import load_tasks
from saas_bench.opencode.prompts import SAAS_SYSTEM_PROMPT

HARNESS_VERSION = "2.1.1"


def inventory(
    root: Path,
    tasks_dir: str = "tasks",
    selector: str = "all",
    *,
    require_verifier: bool = True,
) -> dict[str, dict]:
    """Validate tasks and materialize a deterministic selected inventory."""
    import yaml

    root = root.resolve(strict=True)
    directory = (root / tasks_dir).resolve(strict=True)
    directory.relative_to(root)
    apps = yaml.safe_load((root / "saas_bench/apps.yaml").read_text())["apps"]
    tasks = {}
    for task in load_tasks(str(directory)):
        task_id = task["task_id"]
        if (
            task_id in tasks
            or not isinstance(task_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", task_id)
        ):
            raise ValueError(f"Duplicate or unsafe task ID: {task_id!r}")
        sites = task.get("meta", {}).get("meta_data", {}).get("sites", [])
        if not sites or set(sites) - set(apps):
            raise ValueError(f"{task_id}: empty or unknown app list {sites}")
        if require_verifier and not task["verify_py_path"]:
            raise ValueError(f"{task_id}: missing verify.py")
        inputs = []
        for item in task["meta"].get("multimodal_input", []):
            if item.get("file"):
                path = (root / item["file"]).resolve(strict=True)
                path.relative_to(root)
                inputs.append(str(path.relative_to(root)))
        task["input_paths"] = inputs
        if task["verify_py_path"]:
            task["verify_py_path"] = str(Path(task["verify_py_path"]).resolve())
        # Tie prepared rows to prompt, metadata, verifier and input bytes.
        digest = hashlib.sha256()
        for value in [task["description_md"], json.dumps(task["meta"], sort_keys=True)]:
            digest.update(value.encode())
        for path in (
            [Path(task["verify_py_path"])] if task["verify_py_path"] else []
        ) + [root / p for p in inputs]:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        task["source_sha256"] = digest.hexdigest()
        tasks[task_id] = task
    if not tasks:
        raise ValueError(f"No SaaS tasks found in {directory}")
    if selector == "all":
        return tasks
    selected = {}
    for token in selector.split():
        matches = {
            k: v
            for k, v in tasks.items()
            if k == token or v["category_id"].lower() == token.lower()
        }
        if not matches or set(matches) & set(selected):
            raise ValueError(f"Unknown or overlapping task selector: {token}")
        selected.update(matches)
    if not selected:
        raise ValueError("Empty task selection")
    return dict(sorted(selected.items()))


def dataset_rows(tasks: dict[str, dict]) -> list[dict]:
    """Export portable Gym rows; execution rebuilds URLs after app startup."""
    return [
        {
            "responses_create_params": {
                "input": [{"role": "user", "content": task["description_md"]}]
            },
            "verifier_metadata": {
                "task_id": task_id,
                "category_id": task["category_id"],
                "source_sha256": task["source_sha256"],
                "harness_version": HARNESS_VERSION,
            },
        }
        for task_id, task in tasks.items()
    ]


def opencode_config(
    *,
    endpoint: str,
    model: str,
    python: str,
    browser_config: Path,
    context_tokens: int,
    output_tokens: int,
    max_steps: int,
    files_dir: Path | None = None,
    input_files: list[str] | None = None,
    mcp_timeout_ms: int = 180000,
    refresh_socket: Path | None = None,
) -> dict:
    """Enable browser MCP and scoped native file/todo tools; route inference through Gym."""
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Model endpoint must be HTTP(S), without embedded credentials")
    if not 0 < output_tokens < context_tokens or max_steps < 1 or mcp_timeout_ms < 1000:
        raise ValueError("Invalid context/output limits or step budget")
    files = (files_dir or browser_config.parent / "files").resolve()
    # OpenCode 1.17.11 checks read/edit paths relative to its Git worktree.
    # The worker initializes that worktree at files.parent for each episode.
    inputs = {
        os.path.relpath(Path(path).resolve(), files.parent): "allow"
        for path in (input_files or [])
        if Path(path).suffix.lower() != ".pdf"
    }
    external = {
        str(Path(path).resolve().parent / "*"): "allow" for path in (input_files or [])
    }
    config = {
        "$schema": "https://opencode.ai/config.json",
        "model": f"nemo/{model}",
        "share": "disabled",
        "autoupdate": False,
        "snapshot": False,
        # Native default reservation caps at 20k; keep larger GPU output caps safe too.
        "compaction": {"reserved": output_tokens},
        "provider": {
            "nemo": {
                "npm": "@ai-sdk/openai-compatible",
                "options": {
                    "baseURL": endpoint.rstrip("/"),
                    "apiKey": "EMPTY",
                    "timeout": 600000,
                },
                "models": {
                    model: {
                        "name": model,
                        # OpenCode reserves the output budget itself. Subtracting
                        # it here reserves it twice and compacts too early.
                        "limit": {
                            "context": context_tokens,
                            "input": context_tokens,
                            "output": output_tokens,
                        },
                        "modalities": {"input": ["text", "image"], "output": ["text"]},
                    }
                },
            }
        },
        "agent": {
            "build": {"steps": max_steps, "prompt": SAAS_SYSTEM_PROMPT},
            "general": {"disable": True},
            "explore": {"disable": True},
        },
        "permission": {
            "*": "deny",
            "saas_browser_*": "allow",
            "doom_loop": "allow",
            "read": {"*": "deny", "files": "allow", "files/**": "allow", **inputs},
            "edit": {"*": "deny", "files/**": "allow"},
            "external_directory": {"*": "deny", **external},
            "todowrite": "allow",
        },
        "mcp": {
            "saas_browser": {
                "type": "local",
                "enabled": True,
                "command": [
                    python,
                    "-m",
                    "saas_bench.opencode.browser",
                    str(browser_config),
                ],
                "timeout": mcp_timeout_ms,
            }
        },
    }
    if refresh_socket is not None:
        config["plugin"] = [Path(__file__).with_name("visual_refresh.mjs").as_uri()]
    return config


def agent_environment(
    workspace: Path,
    source_root: Path,
    output_tokens: int,
    *,
    refresh_socket: Path | None = None,
) -> dict[str, str]:
    """Give CLI/MCP their runtime paths without inheriting verifier credentials."""
    allowed = {
        "PATH",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "NODE_EXTRA_CA_CERTS",
        "PLAYWRIGHT_BROWSERS_PATH",
        "TMPDIR",
    }
    env = {k: v for k, v in os.environ.items() if k in allowed}
    for name, suffix in [
        ("HOME", "home"),
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"),
        ("XDG_CACHE_HOME", "cache"),
    ]:
        path = workspace / suffix
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env["PYTHONPATH"] = str(source_root)
    env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(output_tokens)
    env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "true"
    for flag in (
        "OPENCODE_DISABLE_MODELS_FETCH",
        "OPENCODE_DISABLE_DEFAULT_PLUGINS",
        "OPENCODE_DISABLE_EXTERNAL_SKILLS",
        "OPENCODE_DISABLE_CLAUDE_CODE",
        "OPENCODE_DISABLE_LSP_DOWNLOAD",
        "OPENCODE_EXPERIMENTAL_DISABLE_FILEWATCHER",
    ):
        env[flag] = "true"
    env["OPENCODE_CONFIG"] = str(workspace / "opencode.json")
    if refresh_socket is not None:
        env["SAAS_BROWSER_REFRESH_SOCKET"] = str(refresh_socket)
    return env


def parse_events(text: str) -> tuple[list[dict], str]:
    """Keep native JSONL events, including tool results, and extract final text."""
    events, messages = [], []
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        events.append(event)
        if event.get("type") == "text":
            messages.append(event.get("part", {}).get("text", ""))
    return events, "\n".join(messages)


def context_overflow_recovered(events: list[dict]) -> bool:
    """Recognize native compaction + resumed completion after a context error.

    OpenCode 1.17.11 retains exit code 1 after a recovered context overflow.
    Only accept this when every error is ContextOverflowError and native events
    show a completed compaction iteration followed by a completed resumed turn.
    The original events and process exit code are never rewritten.
    """
    errors = [(i, e) for i, e in enumerate(events) if e.get("type") == "error"]
    if not errors or any(
        e.get("error", {}).get("name") != "ContextOverflowError" for _, e in errors
    ):
        return False
    last_error = errors[-1][0]
    finishes = [
        i
        for i, e in enumerate(events)
        if i > last_error and e.get("type") == "step_finish"
    ]
    if len(finishes) < 2 or events[finishes[-1]].get("part", {}).get("reason") not in {
        "stop",
        "length",
        "tool-calls",
    }:
        return False
    if events[finishes[0]].get("part", {}).get("reason") != "stop":
        return False
    return any(
        e.get("type") == "step_start" for e in events[finishes[0] + 1 : finishes[-1]]
    )


def response_items(events: list[dict]) -> list[dict]:
    """Translate native CLI events into Gym's typed message/tool transcript."""
    items = []
    for index, event in enumerate(events):
        part = event.get("part") or {}
        event_type = event.get("type")
        if event_type == "text" and part.get("text"):
            items.append(
                {
                    "id": part.get("id", f"msg_{index}"),
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": part["text"], "annotations": []}
                    ],
                }
            )
        elif event_type == "reasoning" and part.get("text"):
            items.append(
                {
                    "id": part.get("id", f"reasoning_{index}"),
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": part["text"]}],
                }
            )
        elif event_type == "tool_use":
            state = part.get("state") or {}
            call_id = part.get("callID", f"call_{index}")
            items.append(
                {
                    "type": "function_call",
                    "id": call_id,
                    "call_id": call_id,
                    "name": part.get("tool", "unknown"),
                    "arguments": json.dumps(state.get("input", {})),
                    "status": "completed"
                    if state.get("status") == "completed"
                    else "incomplete",
                }
            )
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": str(state.get("output", state.get("error", ""))),
                }
            )
    return items


def measured_score(verify: dict) -> float:
    """Never promote missing or failed verification to a measured zero."""
    score = verify.get("score")
    if verify.get("status") not in {"PASS", "FAIL"}:
        raise ValueError(f"Unscored verifier status: {verify.get('status')}")
    if (
        isinstance(score, bool)
        or not isinstance(score, (float, int))
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise ValueError(f"Invalid native score: {score!r}")
    return float(score)
