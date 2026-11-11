# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The configuration and command line of an OpenCode 2 run in a sandbox session.

The adapter writes OpenCode 2's own schema for what it owns (provider, model, plugins, MCP servers, image limits)
and passes the session's ``opencode_config`` keys (``permission``, ``tools``, ``compaction``, ``agent``) in the
OpenCode 1 form the agent configuration and ``configs/permissions/default.yaml`` use. OpenCode 2 reads that form
and migrates it: ``bash`` rules apply to ``shell``, ``task`` rules to ``subagent``, ``agent.build.steps`` stays the
build agent's step limit and ``agent.build.prompt`` its system prompt. ``attachment`` is renamed to ``media``.
"""

import copy
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any


PROVIDER = "nemo_gym"
# OpenCode 2's sandbox-session environment: OpenCode 1's isolation, and no hosted OpenCode models.
ISOLATION = {
    "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
    "OPENCODE_DISABLE_AUTOUPDATE": "true",
    "OPENCODE_DISABLE_MODELS_FETCH": "true",
    "OPENCODE_DISABLE_FREE_MODELS": "1",
}
# OpenCode 2 normalizes images within 2000 px on a side and 5 MiB of base64, and with auto_resize on re-encodes a
# large opaque PNG as JPEG even when it fits (packages/core/src/image/photon-normalize.ts). With auto_resize off and
# the largest limits its schema accepts, every image passes unchanged.
UNRESTRICTED_MEDIA = {
    "image": {"auto_resize": False, "max_width": 2**31 - 1, "max_height": 2**31 - 1, "max_base64_bytes": 2**31 - 1}
}


def mcp_server(url: str, headers: Mapping[str, str] | None, timeout_ms: int | None) -> dict[str, Any]:
    """An MCP grant as a direct remote MCP server; the model sees its tools as ``<grant>_<tool>``."""
    entry: dict[str, Any] = {
        "type": "remote",
        "url": url,
        # The grant's headers authenticate the session; a headless run must not start an OAuth flow.
        "oauth": False,
        # OpenCode 2 otherwise offers MCP tools only through its Code Mode `execute` tool.
        "codemode": False,
    }
    if headers:
        entry["headers"] = dict(headers)
    if timeout_ms is not None:
        entry["timeout"] = {"startup": timeout_ms, "catalog": timeout_ms, "execution": timeout_ms}
    return entry


def session_config(
    *,
    model_id: str,
    base_url: str,
    thinking: bool,
    context_window: int,
    max_output_tokens: int,
    image_input: bool,
    temperature: bool,
    opencode_config: Mapping[str, Any],
    mcp_servers: Mapping[str, Any],
    instructions_file: str | None,
    replace_prompt: str | None,
    plugin_directory: str,
    plugin_options: Mapping[str, Any],
    owner_plugins: Sequence[str],
) -> dict[str, Any]:
    """``OPENCODE_CONFIG_CONTENT`` for one OpenCode 2 run of a sandbox session."""
    model: dict[str, Any] = {
        "name": model_id,
        "limit": {"context": context_window, "input": context_window, "output": max_output_tokens},
        # The Gym Model Server returns and accepts reasoning in `reasoning_content`.
        "compatibility": {"reasoningField": "reasoning_content"},
        # OpenCode 2 assumes image input for a model it does not know; a text-only session declares text only, so
        # OpenCode replaces an image with a note instead of sending it.
        "capabilities": {"input": ["text", "image"] if image_input else ["text"], "output": ["text"]},
    }
    if model_id != "dummy_model":
        model["capabilities"]["reasoning"] = {"supported": thinking}
    if temperature:
        # OpenCode sends a temperature only to a model that declares it.
        model["capabilities"]["temperature"] = True
    config: dict[str, Any] = {
        "model": f"{PROVIDER}/{model_id}",
        "share": "disabled",
        "policies": [
            {"action": "provider.use", "resource": "*", "effect": "deny"},
            {"action": "provider.use", "resource": PROVIDER, "effect": "allow"},
        ],
        "providers": {
            PROVIDER: {
                "package": "aisdk:@ai-sdk/openai-compatible",
                "settings": {"baseURL": base_url, "apiKey": "dummy_key", "timeout": False},  # pragma: allowlist secret
                "models": {model_id: model},
            }
        },
        # OpenCode 2 also offers its Code Mode tool `execute`, through which the model reaches OpenCode's own tools
        # (sessions, worktrees, models). Denying `execute` would offer those tools directly instead, so the session
        # keeps OpenCode's default; a `permission` rule can deny it.
        "permission": copy.deepcopy(dict(opencode_config.get("permission") or {})),
        "plugins": [
            {"package": PurePosixPath(plugin_directory).as_uri(), "options": dict(plugin_options)},
            *(PurePosixPath(path).as_uri() for path in owner_plugins),
        ],
    }
    if image_input:
        # A session's opencode_config `attachment` replaces this, so a run on an endpoint with image limits can
        # restore one.
        config["media"] = copy.deepcopy(UNRESTRICTED_MEDIA)
    if "attachment" in opencode_config:
        config["media"] = copy.deepcopy(opencode_config["attachment"])
    for key in ("tools", "compaction"):
        if opencode_config.get(key):
            config[key] = copy.deepcopy(opencode_config[key])
    agent = copy.deepcopy(dict(opencode_config.get("agent") or {}))
    if replace_prompt is not None:
        agent.setdefault("build", {})["prompt"] = replace_prompt
    if agent:
        config["agent"] = agent
    if mcp_servers:
        config["mcp"] = {"servers": dict(mcp_servers)}
    if instructions_file is not None:
        config["instructions"] = [instructions_file]
    return config


def run_command(binary: str, *, session_id: str | None, thinking: bool) -> list[str]:
    """``opencode run`` for one activation; the prompt arrives on stdin.

    ``--standalone`` keeps every OpenCode process private to the run: without it, a background OpenCode service
    that an earlier command left running would serve the run with that service's configuration.
    """
    command = [binary, "run", "--standalone", "--format", "json", "--title", "NeMo Gym", "--agent", "build"]
    if thinking:
        command.append("--thinking")
    if session_id:
        command.extend(["--session", session_id])
    return command
