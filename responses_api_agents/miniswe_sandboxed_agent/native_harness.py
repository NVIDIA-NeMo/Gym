# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native-session cleanup around the standard mini-SWE CLI transport."""

import json
from pathlib import Path
from shlex import quote

from responses_api_agents.miniswe_sandboxed_agent.harness import MiniSWEHarness


class NativeMiniSWEHarness(MiniSWEHarness):
    """Require a subreaper receipt before a borrowed sandbox may be verified."""

    launch_attempted = False
    cleanup_confirmed = False
    disposed = False

    async def setup(self) -> None:
        self.launch_attempted = False
        self.cleanup_confirmed = False
        self.disposed = False
        servers = self.context.mcp_servers
        self.context.mcp_servers = []
        try:
            await super().setup()
        finally:
            self.context.mcp_servers = servers
        remote = self.remote_directory
        await self.sandbox.upload(Path(__file__).with_name("sandbox_runner.py"), remote + "/supervisor.py")
        if servers:
            client = remote + "/mcp"
            result = await self.sandbox.exec(
                f"{remote}/uv --no-config venv {client} --python {remote}/python/bin/python3 && "
                f"{remote}/uv --no-config pip install --python {client}/bin/python mcp==1.29.0 httpx-aiohttp==0.2.0",
                user=self.context.user,
                timeout_s=self.context.setup_timeout_sec,
            )
            if result.return_code:
                raise RuntimeError(f"MCP runtime install failed ({result.return_code}): {result.stderr}")
            config = self.directory / "mcp.json"
            config.write_text(json.dumps(servers))
            await self.sandbox.upload(config, client + "/servers.json")
            await self.sandbox.upload(Path(__file__).with_name("mcp_client.py"), client + "/client.py")
            self.extra_instruction += (
                f"\nList task MCP tools: {client}/bin/python {client}/client.py list. "
                f"Call them with {client}/bin/python {client}/client.py call SERVER TOOL 'JSON_ARGUMENTS'.\n"
            )

    def runner_command(self, command: str) -> str:
        self.launch_attempted = True
        remote = self.remote_directory
        return (
            f"mkdir -p {remote}/home {remote}/cache && "
            f"HOME={remote}/home XDG_CACHE_HOME={remote}/cache "
            f"setsid --fork --wait {remote}/venv/bin/python {remote}/supervisor.py {remote} {quote(command)}"
        )

    async def close(self) -> None:
        """Fail closed if launch or descendant cleanup cannot be established."""
        if self.cleanup_confirmed or not self.launch_attempted:
            return
        remote = self.remote_directory
        result = await self.sandbox.exec(
            f"touch {remote}/stop; "
            f"for i in $(seq 1 100); do [ -f {remote}/cleanup.json ] && break; sleep 0.1; done; "
            f"test -f {remote}/cleanup.json && cat {remote}/cleanup.json",
            user=self.context.user,
            timeout_s=15,
        )
        if result.return_code:
            raise RuntimeError(f"mini-SWE cleanup unconfirmed ({result.return_code}): {result.stderr}")
        evidence = json.loads(result.stdout)
        if evidence.get("status") != "stopped" or evidence.get("remaining_pids") != []:
            raise RuntimeError(f"mini-SWE descendant cleanup failed: {evidence}")
        (self.directory / "cleanup.json").write_text(json.dumps(evidence, indent=2))
        await self.sandbox.download(remote + "/runtime.json", self.directory / "runtime.json")
        self.cleanup_confirmed = True

    async def dispose(self) -> None:
        """Remove adapter files only after confirmed process cleanup."""
        if self.disposed:
            return
        if self.launch_attempted and not self.cleanup_confirmed:
            raise RuntimeError("Cannot remove runtime files before confirmed cleanup")
        remote = quote(self.remote_directory)
        registry = quote("/tmp/" + self.context.session_id + ".pids")
        result = await self.sandbox.exec(
            f"rm -rf -- {remote} && rm -f -- {registry} && test ! -e {remote} && test ! -e {registry}",
            user=self.context.user,
            timeout_s=30,
        )
        if result.return_code:
            raise RuntimeError(f"mini-SWE session file cleanup failed ({result.return_code}): {result.stderr}")
        self.disposed = True
