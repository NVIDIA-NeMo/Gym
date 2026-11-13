# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Check FM01 oracle evidence reading without installing Gym or starting Docker."""

import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType

import pytest


OUTPUT = "NeMo Gym v0.7.0\nPython 3.13.16 (/workspace/.venv/bin/python)\n"
SESSION_FIXTURE = Path(__file__).with_name("fixtures") / "fm01-codex-version-session.jsonl"


@pytest.fixture
def verifier() -> ModuleType:
    root = Path(__file__).resolve().parents[3]
    path = root / "environments/pare/gym-fm-01/tests/verify.py"
    spec = importlib.util.spec_from_file_location("pare_fm01_verifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def oracle_trajectory() -> dict:
    return {
        "agent": {"model_name": "oracle"},
        "steps": [{"step_id": 2, "source": "agent", "message": OUTPUT}],
    }


def test_oracle_message_identifies_installation_without_exit_file(verifier: ModuleType) -> None:
    checks = list(verifier.find_version_checks(oracle_trajectory()))
    assert len(checks) == 1
    assert checks[0]["python"] == "/workspace/.venv/bin/python"
    assert checks[0]["gym_version"] == "0.7.0"
    assert checks[0]["evidence"] == {
        "step_id": 2,
        "source": "oracle",
    }


def test_regular_agent_message_is_not_oracle_evidence(verifier: ModuleType) -> None:
    trajectory = oracle_trajectory()
    trajectory["agent"]["model_name"] = "coding-model"
    assert list(verifier.find_version_checks(trajectory)) == []


@pytest.mark.parametrize("output", ["NeMo Gym v0.7.0\n", "Python 3.13.16 (/workspace/.venv/bin/python)\n"])
def test_oracle_requires_both_version_and_python_path(verifier: ModuleType, output: str) -> None:
    trajectory = oracle_trajectory()
    trajectory["steps"][0]["message"] = output
    assert list(verifier.find_version_checks(trajectory)) == []


@pytest.mark.parametrize("installed", [False, True])
def test_oracle_output_still_requires_independent_installation_check(
    verifier: ModuleType, monkeypatch: pytest.MonkeyPatch, installed: bool
) -> None:
    inspected = []

    def inspect(candidate: dict) -> dict:
        inspected.append(candidate)
        return {"passed": installed, "python_version": "3.13.16", "version_output": OUTPUT}

    monkeypatch.setattr(verifier, "check_installation", inspect)
    result = verifier.verify_installation(oracle_trajectory())
    assert result["passed"] is installed
    assert len(inspected) == 1
    assert inspected[0]["python"] == "/workspace/.venv/bin/python"
    assert result["checks"][0]["installation"]["passed"] is installed


@pytest.mark.parametrize("exit_code", [0, 1])
def test_codex_terminal_observation_still_identifies_installation(verifier: ModuleType, exit_code: int) -> None:
    trajectory = {
        "agent": {"model_name": "coding-model"},
        "steps": [
            {
                "step_id": 3,
                "source": "agent",
                "tool_calls": [{"tool_call_id": "call-1", "function_name": "exec_command"}],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "call-1",
                            "content": f"Process exited with code {exit_code}\nOutput:\n" + OUTPUT,
                        }
                    ]
                },
            }
        ],
    }
    checks = list(verifier.find_version_checks(trajectory))
    assert len(checks) == 1
    assert checks[0]["python"] == "/workspace/.venv/bin/python"
    assert checks[0]["evidence"]["tool_call_id"] == "call-1"
    assert checks[0]["evidence"]["exit_code"] == exit_code


def session_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "type": "response_item",
        "payload": {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(arguments)},
    }


def session_result(call_id: str, output: str) -> dict:
    return {
        "type": "response_item",
        "payload": {"type": "function_call_output", "call_id": call_id, "output": output},
    }


def write_session(directory: Path, records: list[dict], name: str = "rollout-test.jsonl") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def test_recorded_codex_version_output_is_read_with_completion(verifier: ModuleType, tmp_path: Path) -> None:
    """Replay real command/poll records, not a fabricated success envelope."""
    path = tmp_path / "rollout-recorded.jsonl"
    shutil.copyfile(SESSION_FIXTURE, path)
    checks = list(verifier.find_session_version_checks(tmp_path))
    assert len(checks) == 1
    assert checks[0]["gym_version"] == "0.7.0"
    assert checks[0]["python"] == "/workspace/.venv/bin/python"
    evidence = checks[0]["evidence"]
    assert evidence["source"] == "codex_session"
    assert evidence["session_file"] == path.name
    assert evidence["tool_call_id"] == "call_L3r0U3InxHBF8HamazXTGarj"
    assert evidence["exit_code"] == 0
    assert evidence["observations"][0]["record_line"] == 4


@pytest.mark.parametrize("installed", [False, True])
def test_session_reader_still_requires_independent_installation_check(
    verifier: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, installed: bool
) -> None:
    shutil.copyfile(SESSION_FIXTURE, tmp_path / "rollout-recorded.jsonl")
    inspected = []

    def inspect(candidate: dict) -> dict:
        inspected.append(candidate)
        return {"passed": installed, "python_version": "3.13.14", "version_output": OUTPUT}

    monkeypatch.setattr(verifier, "check_installation", inspect)
    trajectory = {"agent": {"model_name": "coding-model"}, "steps": []}
    result = verifier.verify_installation(trajectory, sessions_dir=tmp_path)
    assert result["passed"] is installed
    assert len(inspected) == 1
    assert inspected[0]["evidence"]["source"] == "codex_session"


@pytest.mark.parametrize("exit_code", [0, 1, -9])
def test_session_reader_records_exit_status_without_requiring_zero(
    verifier: ModuleType, tmp_path: Path, exit_code: int
) -> None:
    write_session(
        tmp_path,
        [
            session_call("command", "exec_command", {}),
            session_result("command", f"Process exited with code {exit_code}\nOutput:\n" + OUTPUT),
        ],
    )
    checks = list(verifier.find_session_version_checks(tmp_path))
    assert len(checks) == 1
    assert checks[0]["evidence"]["exit_code"] == exit_code


def test_session_reader_follows_matching_poll_to_completion(verifier: ModuleType, tmp_path: Path) -> None:
    write_session(
        tmp_path,
        [
            session_call("command", "exec_command", {}),
            session_result("command", "Process running with session ID 42\nOutput:\n" + OUTPUT),
            session_call("other", "write_stdin", {"session_id": 99}),
            session_result("other", "Process exited with code 7\nOutput:\nunrelated\n"),
            session_call("poll", "write_stdin", {"session_id": 42}),
            session_result("poll", "Process exited with code 1\nOutput:\nfinished\n"),
        ],
    )
    checks = list(verifier.find_session_version_checks(tmp_path))
    assert len(checks) == 1
    assert checks[0]["evidence"]["exit_code"] == 1
    assert [ref["source_call_id"] for ref in checks[0]["evidence"]["observations"]] == ["command", "poll"]


@pytest.mark.parametrize("header", ["Process running with session ID 42", "No completion recorded"])
def test_session_reader_rejects_version_without_completion(verifier: ModuleType, tmp_path: Path, header: str) -> None:
    write_session(
        tmp_path,
        [
            session_call("command", "exec_command", {}),
            session_result("command", header + "\nOutput:\n" + OUTPUT + "Process exited with code 0\n"),
        ],
    )
    with pytest.raises(verifier.EvidenceError, match="no recorded completion"):
        list(verifier.find_session_version_checks(tmp_path))


def test_completion_from_another_session_file_is_not_used(verifier: ModuleType, tmp_path: Path) -> None:
    write_session(
        tmp_path,
        [
            session_call("command", "exec_command", {}),
            session_result("command", "Process running with session ID 42\nOutput:\n" + OUTPUT),
        ],
        "rollout-first.jsonl",
    )
    write_session(
        tmp_path,
        [
            session_call("poll", "write_stdin", {"session_id": 42}),
            session_result("poll", "Process exited with code 0\nOutput:\nfinished\n"),
        ],
        "rollout-second.jsonl",
    )
    with pytest.raises(verifier.EvidenceError, match="no recorded completion"):
        list(verifier.find_session_version_checks(tmp_path))


@pytest.mark.parametrize("output", [OUTPUT, "Process exited with code 0\nOutput:\nNeMo Gym v0.7.0\n"])
def test_session_reader_requires_terminal_envelope_and_python_path(
    verifier: ModuleType, tmp_path: Path, output: str
) -> None:
    write_session(tmp_path, [session_call("command", "exec_command", {}), session_result("command", output)])
    assert list(verifier.find_session_version_checks(tmp_path)) == []


def test_session_reader_ignores_messages_nonterminal_and_unmatched_results(
    verifier: ModuleType, tmp_path: Path
) -> None:
    output = "Process exited with code 0\nOutput:\n" + OUTPUT
    write_session(
        tmp_path,
        [
            {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": OUTPUT}},
            session_call("search", "web_search", {}),
            session_result("search", output),
            session_result("unknown", output),
        ],
    )
    assert list(verifier.find_session_version_checks(tmp_path)) == []


def test_reference_validation_does_not_consume_codex_sessions(verifier: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "rollout-invalid.jsonl").write_text("not valid JSON\n")
    trajectory = oracle_trajectory()
    trajectory["steps"] = []
    result = verifier.verify_installation(trajectory, sessions_dir=tmp_path)
    assert not result["passed"]
    assert result["checks"] == []


def test_unreadable_session_is_reported(verifier: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "rollout-invalid.jsonl").write_text("not valid JSON\n")
    with pytest.raises(verifier.EvidenceError, match="line 1"):
        list(verifier.find_session_version_checks(tmp_path))


@pytest.mark.parametrize("retention_error", [False, True])
def test_main_retains_session_evidence_without_making_copying_a_grading_requirement(
    verifier: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retention_error: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = tmp_path / "agent"
    sessions = agent / "sessions" / "2026/11/06"
    sessions.mkdir(parents=True)
    shutil.copyfile(SESSION_FIXTURE, sessions / "rollout-recorded.jsonl")
    (agent / "trajectory.json").write_text(json.dumps({"agent": {"model_name": "codex"}, "steps": []}))
    logs = tmp_path / "verifier"
    paths = {
        "/logs/agent/trajectory.json": agent / "trajectory.json",
        "/logs/agent/sessions": agent / "sessions",
        "/logs/verifier": logs,
    }
    monkeypatch.setattr(verifier, "Path", lambda path: paths.get(path, Path(path)))
    monkeypatch.setattr(
        verifier,
        "check_installation",
        lambda candidate: {
            "passed": True,
            "python_version": "3.13.14",
            "version_output": OUTPUT,
        },
    )
    if retention_error:

        def refuse_copy(source: Path, destination: Path) -> None:
            raise OSError("diagnostic destination unavailable")

        monkeypatch.setattr(verifier.shutil, "copyfile", refuse_copy)
    verifier.main()
    assert (logs / "reward.txt").read_text() == "1\n"
    assert (logs / "gym-version.txt").read_text() == OUTPUT
    evidence = json.loads((logs / "evidence.json").read_text())
    assert evidence["version_check"]["source"] == "codex_session"
    assert evidence["version_check"]["session_file"] == "2026/11/06/rollout-recorded.jsonl"
    if retention_error:
        assert "Could not retain Codex session" in capsys.readouterr().err
    else:
        assert (logs / "codex-sessions/2026/11/06/rollout-recorded.jsonl").read_bytes() == SESSION_FIXTURE.read_bytes()
