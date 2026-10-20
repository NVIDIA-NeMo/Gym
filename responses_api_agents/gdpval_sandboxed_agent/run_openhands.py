#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the pinned OpenHands CodeAct controller inside the GDPVal OCI sandbox."""

import argparse
import asyncio
import json
import sys
import os
import traceback
from pathlib import Path


async def main(args):
    config_data = json.loads(args.config.read_text())
    trace = args.trace.resolve()
    workspace = args.workspace.resolve()
    os.environ["NEMO_GYM_METRICS_FPATH"] = str(trace / "metrics.json")
    trace.mkdir(parents=True, exist_ok=True)
    (trace / "metrics.json").write_text("{}")
    (trace / "completions").mkdir(exist_ok=True)

    from direct_model_client import use_direct_model_client
    from openhands.core.config import AgentConfig, LLMConfig, OpenHandsConfig
    from openhands.core.config.mcp_config import MCPConfig, MCPStdioServerConfig
    from openhands.core.config.condenser_config import LLMSummarizingCondenserConfig, NoOpCondenserConfig
    from openhands.core.main import auto_continue_response, run_controller
    from openhands.core.schema import AgentState
    from openhands.events.action import MessageAction

    config = OpenHandsConfig(
        default_agent="CodeActAgent",
        runtime="cli",
        workspace_base=str(workspace),
        workspace_mount_path_in_sandbox=str(workspace),
        run_as_openhands=False,
        file_store="local",
        file_store_path=str(trace / "sessions"),
        # OpenHands requires int (None fails validation). sys.maxsize is an
        # unreachable iteration ceiling within the external execution timeout.
        max_iterations=(sys.maxsize if config_data["max_turns"] is None else config_data["max_turns"]),
        save_trajectory_path=str(trace / "events.json"),
        init_git_in_empty_workspace=False,
    )
    llm_config = LLMConfig(
        model=config_data["policy_model"],
        api_key="EMPTY",
        custom_llm_provider="openai",
        base_url=config_data["policy_proxy_url"],
        native_tool_calling=True,
        temperature=1.0,
        top_p=1.0,
        max_input_tokens=config_data["max_context"],
        max_output_tokens=config_data["max_output_tokens"],
        num_retries=2,
        timeout=180,
        log_completions=True,
        log_completions_folder=str(trace / "completions"),
        caching_prompt=False,
    )
    config.set_llm_config(llm_config)
    # Context compaction (training with NeMo RL multi-trace "compaction" mode): the GDPVal
    # condenser summarizes at a fraction of max_context and records its summary call like
    # any other policy call. The LLMSummarizingCondenserConfig is a placeholder that
    # compaction_condenser.install routes to that condenser (keep_first/max_size unused).
    compaction = config_data.get("context_summarization") or {}
    compaction_cutoff = compaction.get("cutoff")
    condenser = NoOpCondenserConfig()
    if compaction_cutoff is not None:
        condenser = LLMSummarizingCondenserConfig(llm_config=llm_config, keep_first=1, max_size=4)
    web_tools = config_data.get("web_tools") or {}
    web_tools_enabled = bool(web_tools.get("enabled"))
    if web_tools_enabled:
        # web_search / fetch_web_page as a stdio MCP server in this sandbox (eval parity).
        # OpenHands' own default stdio servers stay off (no search_api_key => no npx tavily).
        config.mcp = MCPConfig(
            stdio_servers=[
                MCPStdioServerConfig(
                    name="gdpval-web",
                    command=sys.executable,
                    args=["/runner/web_tools_mcp.py"],
                    env={
                        "GDPVAL_WEB_KEYS": ",".join(web_tools.get("keys") or []),
                        "GDPVAL_WEB_MAX_SWEEPS": str(web_tools.get("max_sweeps") or 1),
                    },
                )
            ]
        )
    config.set_agent_config(
        AgentConfig(
            enable_browsing=False,
            enable_jupyter=False,
            enable_mcp=web_tools_enabled,
            enable_plan_mode=False,
            enable_prompt_extensions=False,
            enable_history_truncation=False,
            condenser=condenser,
            runtime="cli",
        )
    )
    # Shell commands use the sandbox's Python/document tools; the controller has its own venv.
    os.environ["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    async with use_direct_model_client(
        read_timeout_s=config_data.get("model_read_timeout_s", 1800),
        read_retries=int(config_data.get("model_read_retries", 0)),
        timing_path=trace / "model-requests.jsonl",
        failed_request_dir=trace / "failed-model-requests",
    ):
        if compaction_cutoff is not None:
            import compaction_condenser

            compaction_condenser.install(
                cutoff=float(compaction_cutoff),
                max_context=int(config_data["max_context"]),
                max_event_chars=int(compaction.get("max_event_chars") or 10_000),
                log_path=str(trace / "compactions.jsonl"),
            )
        state = await run_controller(
            config=config,
            initial_user_action=MessageAction(
                content=(workspace / "prompt.txt").read_text()
            ),
            sid="gdpval-debug",
            fake_user_response_fn=auto_continue_response,
            headless_mode=True,
        )
    status = {
        "agent_state": str(state.agent_state.value) if state else None,
        "last_error": str(state.last_error) if state and state.last_error else None,
        "finished": bool(
            state and state.agent_state == AgentState.FINISHED and not state.last_error
        ),
    }
    (trace / "agent-status.json").write_text(json.dumps(status, indent=2) + "\n")
    if not status["finished"]:
        raise RuntimeError(
            f"OpenHands did not finish successfully: {status['agent_state']}"
        )


async def run_with_diagnostics(args):
    try:
        await main(args)
    except BaseException as error:
        args.trace.mkdir(parents=True, exist_ok=True)
        (args.trace / "controller-error.log").write_text(traceback.format_exc())
        status = args.trace / "agent-status.json"
        if not status.exists():
            status.write_text(
                json.dumps(
                    {
                        "agent_state": "error",
                        "finished": False,
                        "last_error": f"{type(error).__name__}: {error}",
                    },
                    indent=2,
                )
                + "\n"
            )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("/control/agent-config.json")
    )
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--trace", type=Path, default=Path("/trace"))
    asyncio.run(run_with_diagnostics(parser.parse_args()))
