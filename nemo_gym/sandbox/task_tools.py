# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expose sandbox-local task tools to terminal-capable harnesses."""

import json
from pathlib import Path
from shlex import quote
from typing import Protocol

from nemo_gym.sandbox import AsyncSandbox


class TaskToolContext(Protocol):
    session_id: str
    user: str | int | None
    workdir: str | None
    setup_timeout_sec: float
    skills_dir: str | None
    mcp_servers: list[dict]


async def prepare_task_tools(sandbox: AsyncSandbox, context: TaskToolContext, directory: Path) -> str:
    """Install sandbox-local MCP access and return instructions for any terminal harness."""
    directory.mkdir(parents=True, exist_ok=True)
    instruction = ""
    if context.skills_dir:
        instruction += f"\nTask skills are in {context.skills_dir}. Read the relevant SKILL.md files.\n"
    if context.mcp_servers:
        (directory / "mcp.json").write_text(json.dumps(context.mcp_servers))
        remote = f"/tmp/{context.session_id}-mcp"
        command = f"python3 -m venv {remote} && {remote}/bin/pip -q install mcp==1.29.0 httpx-aiohttp==0.2.0"
        result = await sandbox.exec(
            command, user=context.user, cwd=context.workdir, timeout_s=context.setup_timeout_sec
        )
        if result.return_code:
            raise RuntimeError(f"Task MCP client setup failed: {result.stderr}")
        await sandbox.upload(Path(__file__).with_name("mcp_client.py"), remote + "/client.py")
        await sandbox.upload(directory / "mcp.json", remote + "/servers.json")
        cli = f"{remote}/bin/python {remote}/client.py"
        daemon = f"echo $$ >> /tmp/{context.session_id}.pids; exec {cli} serve"
        started = await sandbox.exec(
            "bash -c "
            + quote(
                f"setsid --fork bash -c {quote(daemon)} > {remote}/server.log 2>&1 < /dev/null; "
                f"for i in $(seq 1 60); do [ -S {remote}/server.sock ] && exit 0; sleep 1; done; "
                f"cat {remote}/server.log; exit 1"
            ),
            user=context.user,
            cwd=context.workdir,
            timeout_s=65,
        )
        if started.return_code:
            raise RuntimeError(f"Task MCP session setup failed: {started.stdout}")
        listed = await sandbox.exec(cli + " list", user=context.user, cwd=context.workdir, timeout_s=60)
        if listed.return_code:
            raise RuntimeError(f"Task MCP discovery failed: {listed.stderr}")
        instruction += (
            f"\nTask MCP tools (JSON schemas): {listed.stdout}\nCall with: {cli} call SERVER TOOL 'JSON_ARGUMENTS'.\n"
        )
    return instruction
