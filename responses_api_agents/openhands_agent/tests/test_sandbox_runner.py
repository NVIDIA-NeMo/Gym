# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the sandbox runner as a process against a stand-in for the OpenHands SDK."""

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from responses_api_agents.openhands_agent.sandbox import RUNNER_SOURCE


# Records what the runner built and lets each test choose how the conversation ends.
STUB = """
import json, os, time

__version__ = "stub-version"
LOG = os.environ["STUB_LOG"]

def note(**entry):
    with open(LOG, "a") as handle:
        handle.write(json.dumps(entry) + "\\n")

# Some OpenHands modules read OH_PERSISTENCE_DIR once, at import.
note(imported_with=os.environ.get("OH_PERSISTENCE_DIR"))

class Event:
    def __init__(self, kind, **fields):
        self.kind = kind
        self.fields = {"kind": kind, **fields}
    def model_dump_json(self):
        return json.dumps(self.fields)

class _Part:
    def __init__(self, **fields):
        self.__dict__.update(fields)

class _Message:
    # The tool message OpenHands sends the model: the output plus its own notes; an image part has no text.
    content = [_Part(text="out"), _Part(image_urls=["data:x"]), _Part(text="\\n[Command finished with exit code 0]")]

class Observation(Event):
    def to_llm_message(self):
        return _Message()

class LLM:
    def __init__(self, **settings):
        self.settings = settings
        note(llm=settings)
    def model_copy(self, update):
        return LLM(**{**self.settings, **update})

class AgentContext:
    def __init__(self, **fields):
        note(context=fields)

class Agent:
    def __init__(self, **fields):
        note(
            agent={
                "tools": [tool.spec for tool in fields["tools"]],
                "condenser": fields["condenser"] is not None,
                "mcp_config": fields["mcp_config"],
                "include_default_tools": fields["include_default_tools"],
            }
        )

class Tool:
    def __init__(self, name, params=None):
        self.spec = {"name": name, "params": params or {}}

class _Status:
    value = "running"

class _TaskObservation:
    def __init__(self, is_error):
        self.is_error = is_error

class _TaskResult:
    kind = "ObservationEvent"
    tool_name = "finish"
    def __init__(self, is_error):
        self.observation = _TaskObservation(is_error)
    def model_dump_json(self):
        return json.dumps({"kind": self.kind, "tool_name": self.tool_name})
    def to_llm_message(self):
        return _Message()

class _State:
    execution_status = _Status()

class _Stats:
    def model_dump(self, mode):
        return {"usage_to_metrics": {}}

class Conversation:
    def __init__(self, agent, workspace, persistence_dir, callbacks, max_iteration_per_run, visualizer, delete_on_close):
        note(conversation={"workspace": workspace, "persistence_dir": persistence_dir, "max": max_iteration_per_run})
        note(environ={name: os.environ.get(name) for name in ("OH_PERSISTENCE_DIR", "OPENHANDS_SUPPRESS_BANNER")})
        self.callbacks = callbacks
        self.state = _State()
        self.conversation_stats = _Stats()
    def send_message(self, text):
        for callback in self.callbacks:
            callback(Event("MessageEvent", source="user", text=text))
    def run(self):
        mode = os.environ["STUB_MODE"]
        for callback in self.callbacks:
            callback(Event("ActionEvent", tool_call_id="c1", llm_response_id="r1"))
        if mode == "finish":
            for callback in self.callbacks:
                callback(Observation("ObservationEvent", tool_call_id="c1", observation={"content": "out"}))
            _State.execution_status.value = "finished"
        elif mode == "task_finish":
            # The task's finish tool (over MCP) answers; a failed call first must not end the run.
            for is_error in (True, False):
                result = _TaskResult(is_error)
                for callback in self.callbacks:
                    callback(result)
                note(status_after_task_finish=str(getattr(self.state.execution_status, "value", self.state.execution_status)))
        elif mode == "error":
            try:
                error = RuntimeError("upstream said no")
                error.status_code = 503
                raise error
            except RuntimeError as cause:
                raise ValueError("conversation failed") from cause
        elif mode == "crash":
            # A local error raised while a provider's HTTP 400 was being handled, not from it.
            try:
                error = RuntimeError("bad request body")
                error.status_code = 400
                raise error
            except RuntimeError:
                raise KeyError("local crash")
        else:
            note(running=True)
            while True:
                time.sleep(0.05)
    def close(self):
        note(closed=True)
"""


# Records each server probe; STUB_MCP_DOWN names a server that does not connect.
MCP_STUB = """
import contextlib, json, os

class MCPServer:
    # create_mcp_tools takes only these, not plain settings.
    def __init__(self, **settings):
        self.settings = settings

@contextlib.contextmanager
def create_mcp_tools(config, timeout):
    (name,) = config
    assert all(isinstance(server, MCPServer) for server in config.values())
    with open(os.environ["STUB_LOG"], "a") as handle:
        handle.write(json.dumps({"mcp_probe": {n: s.settings for n, s in config.items()}, "timeout": timeout}) + "\\n")
    if name == os.environ.get("STUB_MCP_DOWN"):
        raise ConnectionError("refused")
    class Client:
        tools = [type("Tool", (), {"name": name})() for name in json.loads(os.environ.get("STUB_MCP_TOOLS", '["finish"]'))]
    yield Client()
"""


@pytest.fixture
def stub(tmp_path: Path) -> Path:
    root = tmp_path / "stub"
    modules = {
        "openhands/__init__.py": "",
        "openhands/sdk/__init__.py": STUB,
        "openhands/tools/__init__.py": "",
        "openhands/tools/file_editor.py": "class FileEditorTool:\n    name = 'file_editor'\n",
        "openhands/tools/task_tracker.py": "class TaskTrackerTool:\n    name = 'task_tracker'\n",
        "openhands/tools/terminal.py": "class TerminalTool:\n    name = 'terminal'\n",
        "openhands/tools/preset/__init__.py": "",
        "openhands/tools/preset/default.py": "def get_default_condenser(llm):\n    return llm\n",
        "openhands/sdk/conversation/__init__.py": "",
        "openhands/sdk/conversation/state.py": (
            "import enum\nclass ConversationExecutionStatus(str, enum.Enum):\n    FINISHED = 'finished'\n"
        ),
        "openhands/sdk/mcp/__init__.py": MCP_STUB,
    }
    for name, source in modules.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(source)
    return root


def start(tmp_path: Path, stub: Path, mode: str, **extra: object) -> tuple[subprocess.Popen, Path]:
    session = tmp_path / "session"
    (session / "state").mkdir(parents=True)
    (tmp_path / "work").mkdir()
    runner_input = {
        "workdir": str(tmp_path / "work"),
        "prompt": "Build it",
        "system_message_suffix": "Extra",
        "llm": {"model": "openai/m", "base_url": "http://model/ng-rollout/r/v1"},
        "condenser": True,
        "max_iterations": 7,
        "state_dir": str(session / "state"),
        "events_path": str(session / "events.jsonl"),
        "result_path": str(session / "result.json"),
        **{key: value for key, value in extra.items() if key not in {"mcp_down", "mcp_tools"}},
    }
    (session / "input.json").write_text(json.dumps(runner_input))
    env = {"PYTHONPATH": str(stub), "STUB_MODE": mode, "STUB_LOG": str(tmp_path / "stub.log"), "PATH": "/usr/bin:/bin"}
    if extra.get("mcp_down"):
        env["STUB_MCP_DOWN"] = str(extra["mcp_down"])
    if "mcp_tools" in extra:
        env["STUB_MCP_TOOLS"] = json.dumps(extra["mcp_tools"])
    with (tmp_path / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(
            [sys.executable, str(RUNNER_SOURCE), str(session / "input.json")], env=env, stderr=stderr
        )
    return process, session


def notes(tmp_path: Path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "stub.log").read_text().splitlines()]


def events(session: Path) -> list[dict]:
    return [json.loads(line) for line in (session / "events.jsonl").read_text().splitlines()]


def test_finished_run_records_events_result_and_the_harness_configuration(tmp_path, stub):
    process, session = start(tmp_path, stub, "finish")
    assert process.wait(timeout=60) == 0
    result = json.loads((session / "result.json").read_text())
    assert (result["status"], result["execution_status"], result["openhands_version"]) == (
        "returned",
        "finished",
        "stub-version",
    )
    recorded = events(session)
    assert [event["kind"] for _, event in recorded] == ["MessageEvent", "ActionEvent", "ObservationEvent"]
    assert all(isinstance(seen_at, float) for seen_at, _ in recorded)
    # A tool result keeps OpenHands' own JSON and adds the text of the tool message the model read.
    assert recorded[2][1] == {
        "kind": "ObservationEvent",
        "tool_call_id": "c1",
        "observation": {"content": "out"},
        "tool_message_text": "out\n[Command finished with exit code 0]",
    }
    assert "tool_message_text" not in recorded[1][1]
    log = notes(tmp_path)
    terminal = next(entry["agent"]["tools"][0] for entry in log if "agent" in entry)
    # The subprocess terminal keeps tool shells in the supervised process tree; history stays in the session.
    assert terminal == {
        "name": "terminal",
        "params": {
            "terminal_type": "subprocess",
            "env": {"HISTFILE": str(session / "state/bash_history"), "INPUTRC": str(session / "state/inputrc")},
        },
    }
    assert (session / "state/inputrc").read_text().endswith("set enable-bracketed-paste off\n")
    condenser_llm = [entry["llm"] for entry in log if "llm" in entry][1]
    assert condenser_llm["base_url"] == "http://model/ng-rollout/r/v1" and condenser_llm["usage_id"] == "condenser"
    assert {"closed": True} in log


def test_runner_hands_openhands_the_session_settings(tmp_path, stub):
    process, session = start(tmp_path, stub, "finish")
    assert process.wait(timeout=60) == 0
    log = notes(tmp_path)
    workspace = {"workspace": str(tmp_path / "work"), "persistence_dir": str(session / "state"), "max": 7}
    assert {"conversation": workspace} in log
    assert {"context": {"system_message_suffix": "Extra"}} in log
    # OpenHands' per-user state goes to the session, not the task's home, from the first import on.
    assert {"environ": {"OH_PERSISTENCE_DIR": str(session / "state"), "OPENHANDS_SUPPRESS_BANNER": "1"}} in log
    assert {"imported_with": str(session / "state")} in log
    assert [entry["llm"] for entry in log if "llm" in entry][0] == {
        "model": "openai/m",
        "base_url": "http://model/ng-rollout/r/v1",
    }
    result = json.loads((session / "result.json").read_text())
    assert result["ended_at"] >= result["started_at"]


def test_error_records_the_explicit_cause_chain_with_its_status(tmp_path, stub):
    process, session = start(tmp_path, stub, "error")
    assert process.wait(timeout=60) == 1
    result = json.loads((session / "result.json").read_text())
    assert result["status"] == "error"
    # Each error keeps its class and status; its text goes only to stderr, with the traceback.
    assert result["error"] == [
        {"type": "ValueError", "module": "builtins", "status_code": None},
        {"type": "RuntimeError", "module": "builtins", "status_code": 503},
    ]
    assert "upstream said no" in (tmp_path / "stderr.log").read_text()


def test_an_error_raised_while_handling_another_does_not_take_its_status(tmp_path, stub):
    process, session = start(tmp_path, stub, "crash")
    assert process.wait(timeout=60) == 1
    result = json.loads((session / "result.json").read_text())
    assert result["error"] == [{"type": "KeyError", "module": "builtins", "status_code": None}]


def test_sigterm_keeps_partial_events_and_closes_the_terminal(tmp_path, stub):
    process, session = start(tmp_path, stub, "hang")
    deadline = time.monotonic() + 60
    while not (tmp_path / "stub.log").exists() or {"running": True} not in notes(tmp_path):
        assert time.monotonic() < deadline and process.poll() is None
        time.sleep(0.05)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=60) == 128 + signal.SIGTERM
    result = json.loads((session / "result.json").read_text())
    assert result["status"] == "terminated"
    assert [event["kind"] for _, event in events(session)] == ["MessageEvent", "ActionEvent"]
    assert {"closed": True} in notes(tmp_path)


def test_inputrc_keeps_the_task_readline_file_and_turns_bracketed_paste_off(tmp_path, monkeypatch):
    from responses_api_agents.openhands_agent.sandbox_runner import write_inputrc

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    (home / ".inputrc").write_text("set bell-style none\n")
    path = write_inputrc(str(tmp_path))
    assert Path(path).read_text() == f"$include {home / '.inputrc'}\nset enable-bracketed-paste off\n"


GRANTS = {
    "gdpval_resources_server": {
        "url": "http://resources:8080/mcp",
        "headers": {"x-nemo-gym-mcp-session": "token"},
        "required": True,
        "timeout": 300,
    },
    "search": {"url": "http://search:8080/mcp", "headers": {}, "required": False, "timeout": 300},
}


def test_mcp_grants_reach_openhands_and_required_ones_are_probed_first(tmp_path, stub):
    process, _ = start(tmp_path, stub, "finish", mcp_servers=GRANTS)
    assert process.wait(timeout=60) == 0
    log = notes(tmp_path)
    agent = next(entry["agent"] for entry in log if "agent" in entry)
    assert agent["mcp_config"]["gdpval_resources_server"] == {
        "url": "http://resources:8080/mcp",
        "transport": "http",
        "headers": {"x-nemo-gym-mcp-session": "token"},
        "timeout": 300,
    }
    assert set(agent["mcp_config"]) == {"gdpval_resources_server", "search"}
    # Without finish tools, OpenHands keeps its own finish.
    assert agent["include_default_tools"] == ["FinishTool", "ThinkTool"]
    probes = [entry["mcp_probe"] for entry in log if "mcp_probe" in entry]
    assert [list(probe) for probe in probes] == [["gdpval_resources_server"]]


def test_a_required_mcp_server_that_does_not_connect_fails_before_the_conversation(tmp_path, stub):
    process, session = start(tmp_path, stub, "finish", mcp_servers=GRANTS, mcp_down="gdpval_resources_server")
    assert process.wait(timeout=60) == 1
    result = json.loads((session / "result.json").read_text())
    assert result["status"] == "error"
    assert not any("conversation" in entry for entry in notes(tmp_path))


@pytest.mark.parametrize(
    "servers,finish_tool_names,mcp_down,mcp_tools",
    [
        ({}, ["finish"], None, ["finish"]),
        (GRANTS, ["finsh"], None, ["finish"]),
        ({"search": GRANTS["search"]}, ["finish"], "search", ["finish"]),
        ({"search": GRANTS["search"]}, ["finish"], None, []),
    ],
    ids=["no-grants", "unknown-name", "optional-grant-down", "optional-grant-empty"],
)
def test_missing_task_finish_fails_before_the_conversation(
    tmp_path, stub, servers, finish_tool_names, mcp_down, mcp_tools
):
    process, session = start(
        tmp_path,
        stub,
        "finish",
        mcp_servers=servers,
        finish_tool_names=finish_tool_names,
        mcp_down=mcp_down,
        mcp_tools=mcp_tools,
    )
    assert process.wait(timeout=60) == 1
    assert json.loads((session / "result.json").read_text())["status"] == "error"
    assert "No MCP grant offers a tool in finish_tool_names" in (tmp_path / "stderr.log").read_text()
    assert not any("conversation" in entry for entry in notes(tmp_path))


def test_a_required_mcp_server_with_no_tools_fails_before_the_conversation(tmp_path, stub):
    process, session = start(tmp_path, stub, "finish", mcp_servers=GRANTS, mcp_tools=[])
    assert process.wait(timeout=60) == 1
    assert json.loads((session / "result.json").read_text())["status"] == "error"
    assert (
        "Required MCP server gdpval_resources_server did not connect: it offers no tools"
        in (tmp_path / "stderr.log").read_text()
    )
    assert not any("conversation" in entry for entry in notes(tmp_path))


@pytest.mark.parametrize("required", [False, True])
def test_task_finish_from_a_reachable_grant_allows_an_unreachable_optional_grant(tmp_path, stub, required):
    servers = {**GRANTS, "gdpval_resources_server": {**GRANTS["gdpval_resources_server"], "required": required}}
    process, session = start(
        tmp_path,
        stub,
        "task_finish",
        mcp_servers=servers,
        finish_tool_names=["finish", "abandon_task_finish"],
        mcp_down="search",
    )
    assert process.wait(timeout=60) == 0
    log = notes(tmp_path)
    probes = [entry["mcp_probe"] for entry in log if "mcp_probe" in entry]
    assert [list(probe) for probe in probes] == [["gdpval_resources_server"], ["search"]]
    assert json.loads((session / "result.json").read_text())["execution_status"] == "finished"


def test_a_successful_task_finish_ends_the_run_and_hides_openhands_finish(tmp_path, stub):
    process, _ = start(tmp_path, stub, "task_finish", mcp_servers=GRANTS, finish_tool_names=["finish"])
    assert process.wait(timeout=60) == 0
    log = notes(tmp_path)
    agent = next(entry["agent"] for entry in log if "agent" in entry)
    assert agent["include_default_tools"] == ["ThinkTool"]
    # The failed call leaves the run going; the successful one marks it finished.
    assert [entry["status_after_task_finish"] for entry in log if "status_after_task_finish" in entry] == [
        "running",
        "finished",
    ]
