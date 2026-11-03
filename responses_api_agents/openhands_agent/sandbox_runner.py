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
from pathlib import Path


RESULT_VERSION = 1


class Terminated(BaseException):
    """Raised in the main thread by SIGTERM, so cleanup runs on the way out of the conversation."""


def _on_sigterm(signum: int, frame: object) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise Terminated()


def describe_error(error: BaseException) -> list[dict[str, object]]:
    """The exception chain, outermost first, with any HTTP status code a provider error carries."""
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
                "message": str(current)[-4000:],
                "status_code": status if isinstance(status, int) and not isinstance(status, bool) else None,
            }
        )
        current = current.__cause__ or current.__context__
    return chain


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
        # OpenHands timestamps are naive local times; record when each event was seen as epoch seconds.
        line = f"[{time.time()!r}, {event.model_dump_json()}]\n"  # type: ignore[attr-defined]
        with lock:
            events.write(line)
            events.flush()

    conversation = None
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
        terminal = {"terminal_type": "subprocess", "env": {"HISTFILE": f"{config['state_dir']}/bash_history"}}
        tools = [
            Tool(name=TerminalTool.name, params=terminal),
            Tool(name=FileEditorTool.name),
            Tool(name=TaskTrackerTool.name),
        ]
        # OpenHands' default condenser summarizes with a copy of the agent's LLM, so it uses the same endpoint.
        condenser = (
            get_default_condenser(llm.model_copy(update={"usage_id": "condenser"})) if config["condenser"] else None
        )
        agent = Agent(
            llm=llm,
            tools=tools,
            agent_context=AgentContext(system_message_suffix=config.get("system_message_suffix")),
            system_prompt_kwargs={"cli_mode": True},
            condenser=condenser,
        )
        conversation = Conversation(
            agent=agent,
            workspace=config["workdir"],
            persistence_dir=config["state_dir"],
            callbacks=[record],
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
