# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Annotated, Any, Dict

from httpx import RemoteProtocolError
from mcp.server.fastmcp import FastMCP
from omegaconf import OmegaConf
from pydantic import Field

from ...code_execution.sandbox import get_sandbox
from ..tool_providers import MCPClientTool
from ..utils import add_config_args, load_mcp_config

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    output_dict: Dict[str, str]
    session_id: Any | None  # uuid


mcp = FastMCP(name="python_tool")

# Initialized from config in main()
sandbox = None

# TODO: how should we control timeout in description?
description = (
    "Call this function to execute Python code in a stateful Jupyter notebook environment. "
    "Python will respond with the output of the execution or time out after 120.0 seconds."
)


@mcp.tool(name="stateful_python_code_exec", description=description)
async def stateful_python_code_exec(
    code: Annotated[str, Field(description="Code to execute")],
    session_id: Annotated[str | None, Field(description="Session id for session persistence")] = None,
    timeout: Annotated[float, Field(description="Time in seconds to allow the job to run")] = 10,
) -> ExecutionResult:
    language = "ipython"
    try:
        output_dict, session_id = await sandbox.execute_code(
            code, language=language, timeout=timeout, session_id=session_id
        )
    except RemoteProtocolError:
        output_dict = {"process_status": "fail", "stdout": "", "stderr": "Error connecting to sandbox"}
        session_id = None

    return {"output_dict": output_dict, "session_id": session_id}


def main():
    parser = argparse.ArgumentParser(description="MCP server for executing Python code in a sandbox")
    add_config_args(parser)
    args = parser.parse_args()

    try:
        cfg = load_mcp_config(
            config=args.config,
            config_dir=args.config_dir,
            config_name=args.config_name,
        )
    except ValueError as e:
        logger.warning(f"{e} Falling back to default local sandbox config.")
        cfg = OmegaConf.create({"sandbox": {"sandbox_type": "local"}})

    global sandbox
    sandbox_cfg = OmegaConf.to_container(cfg.sandbox, resolve=True)
    sandbox = get_sandbox(**sandbox_cfg)
    # Initialize and run the server
    mcp.run(transport="stdio")


# ==============================
# Module-based tool implementation
# ==============================


class PythonTool(MCPClientTool):
    def __init__(self) -> None:
        super().__init__()
        # Defaults for stdio Python MCP using explicit client class
        self.apply_config_updates(
            {
                "client": "vendored.nemo_skills.mcp.clients.MCPStdioClient",
                "client_params": {
                    "command": "python",
                    "args": ["-m", "vendored.nemo_skills.mcp.servers.python_tool"],
                },
                # hide args from schemas and sanitize at runtime
                "hide_args": {"stateful_python_code_exec": ["session_id", "timeout"]},
                # use explicit Hydra connector built from full context by default
                "init_hook": "hydra",
                # execution-specific default
                "exec_timeout_s": 10,
            }
        )
        self.requests_to_sessions = defaultdict(lambda: None)

    async def execute(self, tool_name: str, arguments: Dict[str, Any], extra_args: Dict[str, Any] | None = None):
        # Ensure timeout is sent via extra_args (post-sanitize), not in main arguments
        arguments = dict(arguments)
        # TODO: error handling?
        request_id = extra_args.pop("request_id")
        merged_extra = dict(extra_args or {})
        merged_extra.setdefault("timeout", self._config.get("exec_timeout_s", 10))
        merged_extra["session_id"] = self.requests_to_sessions[request_id]
        result = await self._client.call_tool(tool=tool_name, args=arguments, extra_args=merged_extra)
        self.requests_to_sessions[request_id] = result["session_id"]
        output = f"{result['output_dict']['stdout']}{result['output_dict']['stderr']}"
        if output.endswith("\n"):  # there is always a trailing newline, removing it
            output = output[:-1]
        return output

    async def shutdown(self) -> None:
        return None


if __name__ == "__main__":
    main()
