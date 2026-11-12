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

__version__ = "1.59.0"
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
        note(agent={"tools": [tool.spec for tool in fields["tools"]], "condenser": fields["condenser"] is not None})

class Tool:
    def __init__(self, name, params=None):
        self.spec = {"name": name, "params": params or {}}

class _Status:
    value = "running"

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
    }
    for name, source in modules.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(source)
    return root


def start(tmp_path: Path, stub: Path, mode: str) -> tuple[subprocess.Popen, Path]:
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
    }
    (session / "input.json").write_text(json.dumps(runner_input))
    env = {"PYTHONPATH": str(stub), "STUB_MODE": mode, "STUB_LOG": str(tmp_path / "stub.log"), "PATH": "/usr/bin:/bin"}
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
        "1.59.0",
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
