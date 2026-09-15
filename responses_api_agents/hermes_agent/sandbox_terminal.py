# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route the Hermes terminal tool to borrowed Gym sandboxes."""

import asyncio
import json
from dataclasses import dataclass
from threading import Lock
from typing import Any, Mapping

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers import create_provider


TERMINAL_SCHEMA = {
    "name": "terminal",
    "description": "Execute a foreground shell command in the task sandbox.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "minimum": 1},
            "workdir": {"type": "string"},
        },
        "required": ["command"],
        "additionalProperties": False,
    },
}


@dataclass
class _Binding:
    sandbox: AsyncSandbox
    loop: asyncio.AbstractEventLoop
    workdir: str


_BINDINGS: dict[str, _Binding] = {}
_BINDINGS_LOCK = Lock()
_REGISTERED = False


async def bind_sandbox(
    agent_session_id: str,
    *,
    descriptor: Mapping[str, Any],
    provider_config: Mapping[str, Any],
    workdir: str,
) -> None:
    """Connect one Hermes task identifier to one borrowed sandbox."""
    provider = create_provider(provider_config)
    try:
        sandbox = await AsyncSandbox.connect(descriptor, provider=provider)
    except BaseException:
        await provider.aclose()
        raise
    with _BINDINGS_LOCK:
        duplicate = agent_session_id in _BINDINGS
        if not duplicate:
            _BINDINGS[agent_session_id] = _Binding(
                sandbox=sandbox,
                loop=asyncio.get_running_loop(),
                workdir=workdir,
            )
    if duplicate:
        await sandbox.disconnect()
        raise ValueError(f"Sandbox already bound for agent session {agent_session_id}")


async def unbind_sandbox(agent_session_id: str) -> None:
    """Release borrower-side clients without stopping the owner sandbox."""
    with _BINDINGS_LOCK:
        binding = _BINDINGS.pop(agent_session_id, None)
    if binding is None:
        return
    await binding.sandbox.disconnect()


def _terminal_handler(arguments: dict[str, Any], **kwargs: Any) -> str:
    agent_session_id = kwargs.get("task_id")
    if not isinstance(agent_session_id, str):
        raise RuntimeError("Hermes terminal calls require an agent session")
    unsupported = set(arguments) - {"command", "timeout", "workdir"}
    if unsupported:
        raise ValueError(f"Unsupported sandbox terminal arguments: {sorted(unsupported)}")
    command = arguments.get("command")
    if not isinstance(command, str) or not command:
        raise ValueError("terminal command must be a non-empty string")
    with _BINDINGS_LOCK:
        binding = _BINDINGS.get(agent_session_id)
    if binding is None:
        raise RuntimeError(f"No borrowed sandbox for agent session {agent_session_id}")
    workdir = arguments.get("workdir", binding.workdir)
    timeout = arguments.get("timeout", 180)
    future = asyncio.run_coroutine_threadsafe(
        binding.sandbox.exec(command, cwd=workdir, timeout_s=timeout),
        binding.loop,
    )
    result = future.result(timeout=float(timeout) + 30)
    return json.dumps(
        {
            "output": result.stdout or "",
            "stderr": result.stderr or "",
            "exit_code": result.return_code,
            "error": result.error_type,
        }
    )


def register_sandbox_terminal() -> None:
    """Replace Hermes's terminal handler with the fail-closed Gym router."""
    global _REGISTERED
    if _REGISTERED:
        return
    from tools.registry import registry

    registry.register(
        name="terminal",
        toolset="terminal",
        schema=TERMINAL_SCHEMA,
        handler=_terminal_handler,
        description=TERMINAL_SCHEMA["description"],
        emoji="💻",
    )
    _REGISTERED = True
