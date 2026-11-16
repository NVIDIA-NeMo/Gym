#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the pinned OpenHands CodeAct controller for one agent-session activation inside the task sandbox.

Usage: ``<runtime python> -E -s run_openhands.py <input.json>``. The input names the workdir, the prompt, the model
endpoint and its headers, the limits, the MCP grants and the output paths; the agent server writes it. The runner
writes ``result.json`` (agent state, last error, whether the agent finished) and keeps its completion, model-request
and compaction logs in the trace directory. It exits 0 once a result is written, whatever the agent's outcome, and
non-zero when the controller could not run.
"""

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _install_mcp_grants(servers: list[dict[str, Any]], *, finish_tool_names: list[str], finished: list[str]) -> None:
    """Send each grant's headers on its MCP connections, name its tools after the grant, and require it to connect.

    OpenHands 0.62 sends only a bearer key to an HTTP MCP server, offers a server's tools under their bare names,
    and skips a server it cannot connect to, so the runner patches its client:
    - Gym's grants authenticate with their own headers;
    - each tool is offered as ``<grant>_<tool>``, so a task tool such as ``finish`` cannot shadow CodeAct's own;
    - a required grant that does not connect fails the run before the first model call;
    - a successful call to a tool named in ``finish_tool_names`` is appended to ``finished``.
    """
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport
    from openhands.mcp import client as mcp_client
    from openhands.mcp import utils as mcp_utils

    headers_by_url = {server["url"]: dict(server.get("headers") or {}) for server in servers}
    prefix_by_url = {server["url"]: server["name"].replace(".", "_") + "_" for server in servers}
    # Per client: the name offered to the model -> the server's own tool name.
    server_names: dict[int, dict[str, str]] = {}
    required = {server["url"] for server in servers if server.get("required")}
    # MCPClient is a pydantic model without extra fields, so the URL each client connected to is kept here.
    connected_url: dict[int, str] = {}

    async def connect_http(self, server, conversation_id=None, timeout: float = 30.0):
        if server.url not in headers_by_url:
            raise ValueError(f"MCP server {server.url} is not one of this session's grants")
        transport = StreamableHttpTransport(url=server.url, headers=headers_by_url[server.url] or None)
        self.client = Client(transport, timeout=timeout)
        await self._initialize_and_list_tools()
        prefix = prefix_by_url[server.url]
        names = {}
        for tool in self.tools:
            names[prefix + tool.name] = tool.name
            tool.name = prefix + tool.name
        self.tool_map = {tool.name: tool for tool in self.tools}
        server_names[id(self)] = names
        connected_url[id(self)] = server.url

    original_call_tool = mcp_client.MCPClient.call_tool

    async def call_tool(self, tool_name, args):
        name = server_names.get(id(self), {}).get(tool_name)
        if name is None:
            return await original_call_tool(self, tool_name, args)
        async with self.client:
            call = self.client.call_tool_mcp(name=name, arguments=args)
            result = await (asyncio.wait_for(call, timeout=self.server_timeout) if self.server_timeout else call)
        if name in finish_tool_names and not result.isError:
            finished.append(name)
        return result

    original_create = mcp_utils.create_mcp_clients

    async def create_mcp_clients(sse_servers, shttp_servers, conversation_id=None, stdio_servers=None):
        clients = await original_create(sse_servers, shttp_servers, conversation_id, stdio_servers)
        connected = {connected_url.get(id(client)) for client in clients}
        missing = sorted(required - connected)
        if missing:
            raise RuntimeError(f"Required MCP servers did not connect: {missing}")
        return clients

    mcp_client.MCPClient.connect_http = connect_http
    mcp_client.MCPClient.call_tool = call_tool
    mcp_utils.create_mcp_clients = create_mcp_clients


def _end_after_task_finish(finished: list[str]) -> None:
    """End the CodeAct run at its next step once a task finish tool has succeeded."""
    from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
    from openhands.events.action import AgentFinishAction

    original_step = CodeActAgent.step

    async def step(self, state):
        if finished:
            return AgentFinishAction(final_thought=f"Submitted with {finished[-1]}.")
        return await original_step(self, state)

    CodeActAgent.step = step


async def _check_required_mcp(servers: list[dict[str, Any]]) -> None:
    """Connect to every granted server once before the first model call; raises if a required one fails."""
    from openhands.core.config.mcp_config import MCPSHTTPServerConfig
    from openhands.mcp import utils as mcp_utils

    configs = [MCPSHTTPServerConfig(url=server["url"], timeout=server["timeout_s"]) for server in servers]
    clients = await mcp_utils.create_mcp_clients([], configs)
    for client in clients:
        try:
            await client.client.close()
        except Exception:
            pass


async def main(spec: dict[str, Any]) -> dict[str, Any]:
    trace = Path(spec["trace_dir"])
    workdir = Path(spec["workdir"])
    trace.mkdir(parents=True, exist_ok=True)
    (trace / "completions").mkdir(exist_ok=True)
    os.environ["NEMO_GYM_METRICS_FPATH"] = str(trace / "metrics.json")
    (trace / "metrics.json").write_text("{}")

    from direct_model_client import use_direct_model_client
    from openhands.core.config import AgentConfig, LLMConfig, OpenHandsConfig
    from openhands.core.config.condenser_config import LLMSummarizingCondenserConfig, NoOpCondenserConfig
    from openhands.core.config.mcp_config import MCPConfig, MCPSHTTPServerConfig
    from openhands.core.main import auto_continue_response, run_controller
    from openhands.core.schema import AgentState
    from openhands.events.action import MessageAction

    config = OpenHandsConfig(
        default_agent="CodeActAgent",
        runtime="cli",
        workspace_base=str(workdir),
        workspace_mount_path_in_sandbox=str(workdir),
        run_as_openhands=False,
        file_store="local",
        file_store_path=str(trace / "sessions"),
        # OpenHands requires an int; sys.maxsize leaves the activation deadline as the only bound.
        max_iterations=sys.maxsize if spec["max_iterations"] is None else spec["max_iterations"],
        save_trajectory_path=str(trace / "events.json"),
        init_git_in_empty_workspace=False,
    )
    model = spec["model"]
    llm_config = LLMConfig(
        model=model["name"],
        api_key="EMPTY",
        custom_llm_provider="openai",
        base_url=model["base_url"],
        native_tool_calling=True,
        temperature=model["temperature"],
        top_p=model["top_p"],
        max_input_tokens=model["max_context"],
        max_output_tokens=model["max_output_tokens"],
        # The direct model client owns retries (model.read_retries); the Gym model server retries transient errors.
        num_retries=0,
        timeout=model["read_timeout_s"],
        log_completions=True,
        log_completions_folder=str(trace / "completions"),
        caching_prompt=False,
    )
    config.set_llm_config(llm_config)
    compaction = spec.get("context_summarization") or {}
    condenser = NoOpCondenserConfig()
    if compaction:
        # A placeholder that compaction_condenser.install routes to the cutoff-triggered condenser.
        condenser = LLMSummarizingCondenserConfig(llm_config=llm_config, keep_first=1, max_size=4)
    servers = spec.get("mcp_servers") or []
    finish_tool_names = list(spec.get("finish_tool_names") or [])
    finished: list[str] = []
    if servers:
        _install_mcp_grants(servers, finish_tool_names=finish_tool_names, finished=finished)
    if finish_tool_names:
        _end_after_task_finish(finished)
        config.mcp = MCPConfig(
            shttp_servers=[MCPSHTTPServerConfig(url=server["url"], timeout=server["timeout_s"]) for server in servers]
        )
    config.set_agent_config(
        AgentConfig(
            enable_browsing=False,
            enable_jupyter=False,
            enable_mcp=bool(servers),
            # With task finish tools, CodeAct's own finish would end the run without submitting the work.
            enable_finish=not finish_tool_names,
            enable_plan_mode=False,
            enable_prompt_extensions=False,
            enable_history_truncation=False,
            condenser=condenser,
            runtime="cli",
        )
    )
    # Shell commands use the sandbox's own tools; the controller has its own venv.
    os.environ["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    if servers:
        await _check_required_mcp(servers)
    async with use_direct_model_client(
        read_timeout_s=model["read_timeout_s"],
        read_retries=model["read_retries"],
        timing_path=trace / "model-requests.jsonl",
        failed_request_dir=trace / "failed-model-requests",
        headers=model.get("headers") or {},
    ):
        if compaction:
            import compaction_condenser

            compaction_condenser.install(
                cutoff=float(compaction["cutoff"]),
                max_context=int(model["max_context"]),
                max_event_chars=int(compaction["max_event_chars"]),
                log_path=str(trace / "compactions.jsonl"),
            )
        state = await run_controller(
            config=config,
            initial_user_action=MessageAction(content=spec["prompt"]),
            sid=spec.get("session_id") or "openhands-codeact",
            fake_user_response_fn=auto_continue_response,
            headless_mode=True,
        )
    return {
        "agent_state": str(state.agent_state.value) if state else None,
        "last_error": str(state.last_error) if state and state.last_error else None,
        "finished": bool(state and state.agent_state == AgentState.FINISHED and not state.last_error),
    }


def run(input_path: Path) -> int:
    spec = json.loads(input_path.read_text())
    result_path = Path(spec["result_path"])
    try:
        result = asyncio.run(main(spec))
    except BaseException as error:
        trace = Path(spec["trace_dir"])
        trace.mkdir(parents=True, exist_ok=True)
        (trace / "controller-error.log").write_text(traceback.format_exc())
        result_path.write_text(
            json.dumps({"agent_state": "error", "finished": False, "last_error": f"{type(error).__name__}: {error}"})
            + "\n"
        )
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return 1
    result_path.write_text(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_openhands.py <input.json>")
    sys.exit(run(Path(sys.argv[1])))
