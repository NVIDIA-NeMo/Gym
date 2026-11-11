# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the pinned session harness inside the sandboxed adapter's existing /run flow."""

import re
import tomllib
from functools import cache
from pathlib import Path

import yaml
from pydantic import JsonValue

from nemo_gym.agents import SandboxPythonRuntime
from nemo_gym.sandbox import AsyncSandbox
from responses_api_agents.hermes_sandboxed_agent.runner import split_input


_AGENT_DIR = Path(__file__).parents[1] / "hermes_agent"


@cache
def _runtime() -> SandboxPythonRuntime:
    # Read the adapter's declared pin without installing or importing Hermes on this host.
    with (_AGENT_DIR / "pyproject.toml").open("rb") as source:
        dependencies = tomllib.load(source)["project"]["dependencies"]
    requirement = next(value for value in dependencies if value.startswith("hermes-agent@"))
    match = re.fullmatch(r"hermes-agent@ git\+https://github.com/([^@]+)@([0-9a-f]{40})", requirement)
    if match is None:
        raise ValueError("hermes_agent must declare an exact GitHub commit for the managed sandbox runtime")
    repository, commit = match.groups()
    return SandboxPythonRuntime(
        harness="hermes",
        requirements=(
            f"hermes-agent[mcp] @ https://github.com/{repository.removesuffix('.git')}/archive/{commit}.tar.gz",
            "mcp>=1.28.1,<2",
        ),
        # This host adapter does not install Hermes; its Rich 15 pin cannot
        # satisfy the pinned sandbox harness's Rich 14 dependency.
        overrides=("rich>=14.3.3,<15",),
        import_check=(
            "import os; os.environ['HERMES_HOME'] = os.path.join(os.environ['HOME'], '.hermes'); "
            "import run_agent; from mcp.client.streamable_http import streamablehttp_client"
        ),
    )


async def prepare_runtime(sandbox: AsyncSandbox, *, remote: str, timeout: float) -> str:
    """Reuse or install the declared Hermes runtime and upload its existing execution shim."""
    runtime = _runtime()
    if not await runtime.is_installed(sandbox):
        await runtime.install(sandbox, timeout_s=timeout)
    for name in ("sandbox_runner.py", "sandbox_observer.py", "model_kwargs.py"):
        await sandbox.upload(_AGENT_DIR / name, f"{remote}/{name}")
    return runtime.python


def runtime_payload(params: dict[str, JsonValue], *, run_id: str) -> dict[str, JsonValue]:
    """Map the sandboxed adapter's controls to the session shim without benchmark metadata."""
    template = params["chat_template_kwargs"]
    if not isinstance(template, dict) or set(template) - {"truncate_history_thinking"}:
        raise ValueError("Managed Hermes supports only truncate_history_thinking in chat_template_kwargs")
    user_message, history, input_system = split_input(params["input"])
    config = {
        "model": (
            {"default": params["model"], "context_length": params["context_length"]}
            if params["context_length"] is not None
            else params["model"]
        ),
        "provider": "auto",
        "toolsets": ["hermes-cli"],
        "agent": {"max_turns": params["max_turns"]},
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
        "compression": {"enabled": params["compression_enabled"], "threshold": 0.85},
        "terminal": {"backend": "local", "cwd": params["workdir"], "timeout": params["terminal_timeout"]},
        "delegation": {"max_iterations": 50},
        "checkpoints": {"enabled": False},
    }
    return {
        **{
            key: params[key]
            for key in (
                "model",
                "max_turns",
                "max_tokens",
                "temperature",
                "terminal_timeout",
                "enabled_toolsets",
                "disabled_toolsets",
            )
        },
        "agent_session_id": run_id,
        "model_base_url": params["base_url"],
        "chat_template_kwargs_enabled": not template.get("truncate_history_thinking", False),
        "config_yaml": yaml.safe_dump(config),
        "user_message": user_message,
        "history": history,
        "system_message": "\n\n".join(
            part
            for part in (params["system_prompt"], params.get("instructions"), input_system)
            if isinstance(part, str) and part
        )
        or None,
        "mcp_servers": [],
        "required_mcp_servers": [],
    }


def trajectory_result(output: dict[str, JsonValue], *, n_input: int) -> dict[str, JsonValue]:
    """Convert the shim's result envelope to the sandboxed adapter's trajectory fields."""
    if output.get("error") is not None:
        raise RuntimeError(f"Hermes sandbox runner failed: {output['error']}")
    native = output.get("result")
    if not isinstance(native, dict):
        raise ValueError("Hermes sandbox runner did not return a result object")
    result = {**native, "n_input": n_input}
    prompt_tokens, completion_tokens = native.get("prompt_tokens"), native.get("completion_tokens")
    if type(prompt_tokens) is int and type(completion_tokens) is int:
        result["usage"] = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "cached_tokens": native.get("cache_read_tokens"),
            "reasoning_tokens": native.get("reasoning_tokens"),
        }
    else:
        result["usage"] = None
    error = str(native.get("error") or "")
    if (
        native.get("stop_reason") in {"wall_time", "max_iterations"}
        or error == "First response truncated due to output length limit"
        or (
            native.get("partial")
            and (
                error == "Response truncated due to output length limit"
                or error.startswith("Request payload too large")
            )
        )
    ):
        result.update(completed=False, failed=False, budget_exhausted=True)
        result["stop_detail"] = result.pop("error", None)
        if not result.get("stop_reason"):
            result["stop_reason"] = "output_tokens" if "output length" in error else "context_limit"
    return result
