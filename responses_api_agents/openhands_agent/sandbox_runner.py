# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one OpenHands Software Agent SDK conversation inside the task sandbox.

The adapter uploads this file without its ``.py`` suffix and runs it with the private Python runtime that
seed installed, never the task image's Python. It reads one JSON input, appends every conversation event to
a JSON-lines file as it happens, and writes a JSON result when the run ends. The process supervisor sends
SIGTERM at the deadline and on close; the result is then written with ``status: terminated`` before the
terminal session is closed, so partial work is kept.

Only the standard library and the pinned OpenHands packages are imported, and they only inside ``main``.
"""

import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path


RESULT_VERSION = 1
# Events that become the tool message answering a tool call. OpenHands builds that message from the tool's output
# plus its own notes (exit code, working directory, no-output timeouts, truncation, rejections), so the runner
# records the message's text with the event.
TOOL_RESULT_EVENTS = ("ObservationEvent", "AgentErrorEvent", "UserRejectObservation")


class Terminated(BaseException):
    """Raised in the main thread by SIGTERM, so cleanup runs on the way out of the conversation."""


def _on_sigterm(signum: int, frame: object) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise Terminated()


def describe_error(error: BaseException) -> list[dict[str, object]]:
    """The error and the errors it was raised from, outermost first.

    Only explicit causes (``raise ... from``) are followed: OpenHands raises its own error class from the
    provider's error, but an error raised while another one was being handled was not caused by it. Each entry
    keeps the class and any HTTP status code a provider error carries. The error's text can hold a provider's
    response body, so it goes only to stderr, with the traceback.
    """
    chain: list[dict[str, object]] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        status = getattr(current, "status_code", None)
        chain.append(
            {
                "type": type(current).__name__,
                "module": type(current).__module__,
                "status_code": status if isinstance(status, int) and not isinstance(status, bool) else None,
            }
        )
        current = current.__cause__
    return chain


def write_inputrc(state_dir: str) -> str:
    """Write a readline file that keeps the task's own settings and turns bracketed paste off.

    OpenHands 1.59.0's subprocess terminal leaves the bracketed-paste escape codes that bash 5.1 and later
    print around each command in the tool output the model reads.
    """
    lines = []
    # Readline reads INPUTRC instead of ~/.inputrc or /etc/inputrc, so include the one it would have read.
    for candidate in (os.path.expanduser("~/.inputrc"), "/etc/inputrc"):
        if os.path.isfile(candidate):
            lines.append(f"$include {candidate}")
            break
    lines.append("set enable-bracketed-paste off")
    path = f"{state_dir}/inputrc"
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def mcp_config(servers: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    """OpenHands MCP server settings for the episode's grants, by grant name."""
    return {
        name: {
            "url": server["url"],
            "transport": "http",
            "headers": dict(server.get("headers") or {}),
            "timeout": server.get("timeout"),
        }
        for name, server in servers.items()
    }


def check_required_mcp(servers: dict[str, dict[str, object]]) -> None:
    """Connect to every required grant once; OpenHands itself skips a server it cannot reach, with a warning."""
    required = mcp_config({name: server for name, server in servers.items() if server.get("required")})
    if not required:
        return
    from openhands.sdk.mcp import MCPServer, create_mcp_tools

    for name, server in required.items():
        try:
            with create_mcp_tools({name: MCPServer(**server)}, timeout=60) as client:
                if not client.tools:
                    raise RuntimeError("it offers no tools")
        except Exception as error:
            raise RuntimeError(f"Required MCP server {name} did not connect: {error}") from error


def finished_by_task_tool(event: object, finish_tool_names: list[str]) -> bool:
    """Whether ``event`` is a successful result of one of the task's finish tools."""
    return (
        getattr(event, "kind", None) == "ObservationEvent"
        and getattr(event, "tool_name", None) in finish_tool_names
        and not getattr(getattr(event, "observation", None), "is_error", True)
    )


def _write_json(path: str, payload: dict[str, object]) -> None:
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(temporary, path)


def main(argv: list[str]) -> int:
    config = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    signal.signal(signal.SIGTERM, _on_sigterm)
    # OpenHands keeps per-user state under ~/.openhands; move it into the session directory so the task's home
    # stays as the task left it. Tool shells keep the task's HOME and XDG directories.
    os.environ["OH_PERSISTENCE_DIR"] = config["state_dir"]
    os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"

    result: dict[str, object] = {
        "version": RESULT_VERSION,
        "status": "error",
        "execution_status": None,
        "error": None,
        "stats": None,
        "openhands_version": None,
        "started_at": time.time(),
        "ended_at": None,
    }
    lock = threading.Lock()
    events = open(config["events_path"], "a", encoding="utf-8")

    def record(event: object) -> None:
        payload = event.model_dump_json()  # type: ignore[attr-defined]
        if getattr(event, "kind", None) in TOOL_RESULT_EVENTS:
            content = event.to_llm_message().content  # type: ignore[attr-defined]
            text = "".join(part.text for part in content if isinstance(getattr(part, "text", None), str))
            payload = json.dumps({**json.loads(payload), "tool_message_text": text})
        # OpenHands timestamps are naive local times; record when each event was seen as epoch seconds.
        line = f"[{time.time()!r}, {payload}]\n"
        with lock:
            events.write(line)
            events.flush()

    conversation = None
    finish_tool_names = list(config.get("finish_tool_names") or [])

    def end_on_task_finish(event: object) -> None:
        # A successful task finish ends the run, as OpenHands' own finish would.
        if conversation is not None and finished_by_task_tool(event, finish_tool_names):
            from openhands.sdk.conversation.state import ConversationExecutionStatus

            conversation.state.execution_status = ConversationExecutionStatus.FINISHED

    exit_code = 1
    try:
        import openhands.sdk
        from openhands.sdk import LLM, Agent, AgentContext, Conversation, Tool
        from openhands.tools.file_editor import FileEditorTool
        from openhands.tools.preset.default import get_default_condenser
        from openhands.tools.task_tracker import TaskTrackerTool
        from openhands.tools.terminal import TerminalTool

        result["openhands_version"] = openhands.sdk.__version__
        llm = LLM(**config["llm"])
        # The subprocess terminal keeps every tool shell a child of this process. The tmux backend would start a
        # tmux server that daemonizes out of the supervised process group and outlives the run.
        # Bash history goes to the session directory instead of the task's home.
        terminal_env = {
            "HISTFILE": f"{config['state_dir']}/bash_history",
            "INPUTRC": write_inputrc(config["state_dir"]),
        }
        terminal = {"terminal_type": "subprocess", "env": terminal_env}
        tools = [
            Tool(name=TerminalTool.name, params=terminal),
            Tool(name=FileEditorTool.name),
            Tool(name=TaskTrackerTool.name),
        ]
        # OpenHands' default condenser summarizes with a copy of the agent's LLM, so it uses the same endpoint.
        condenser = (
            get_default_condenser(llm.model_copy(update={"usage_id": "condenser"})) if config["condenser"] else None
        )
        servers = config.get("mcp_servers") or {}
        check_required_mcp(servers)
        agent = Agent(
            llm=llm,
            tools=tools,
            mcp_config=mcp_config(servers),
            # With task finish tools, OpenHands' own finish would end the run without submitting the work.
            include_default_tools=["ThinkTool"] if finish_tool_names else ["FinishTool", "ThinkTool"],
            agent_context=AgentContext(system_message_suffix=config.get("system_message_suffix")),
            system_prompt_kwargs={"cli_mode": True},
            condenser=condenser,
        )
        conversation = Conversation(
            agent=agent,
            workspace=config["workdir"],
            persistence_dir=config["state_dir"],
            callbacks=[record, end_on_task_finish],
            max_iteration_per_run=config["max_iterations"],
            visualizer=None,
            delete_on_close=False,
        )
        conversation.send_message(config["prompt"])
        conversation.run()
        result["status"] = "returned"
        exit_code = 0
    except Terminated:
        result["status"] = "terminated"
        exit_code = 128 + signal.SIGTERM
    except Exception as error:
        result["error"] = describe_error(error)
        traceback.print_exception(error, file=sys.stderr)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        result["ended_at"] = time.time()
        if conversation is not None:
            try:
                result["execution_status"] = str(conversation.state.execution_status.value)
                result["stats"] = conversation.conversation_stats.model_dump(mode="json")
            except Exception as error:
                result["stats_error"] = describe_error(error)
        _write_json(config["result_path"], result)
        with lock:
            events.close()
        if conversation is not None:
            try:
                # Closes the tool executors, which stops the terminal's shell and anything it still runs.
                conversation.close()
            except Exception as error:
                print(f"OpenHands conversation close failed: {error!r}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
