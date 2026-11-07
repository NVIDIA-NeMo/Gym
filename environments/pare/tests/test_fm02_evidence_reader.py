# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Check reference-log discovery without changing FM02's grading or calling its judge."""

import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType

import pytest


FIXTURE = Path(__file__).with_name("fixtures") / "fm02-oracle-trajectory.json"
EXPECTED = {"verification_contract": {}, "task_request": "Carlos request", "gym_agent": {"harness": "codex_agent"}}
LOG_OUTPUT = "Finished rollout collection!\nSaved rollout: /workspace/results/carlos_rollouts.jsonl\n"


@pytest.fixture
def reader() -> ModuleType:
    root = Path(__file__).resolve().parents[3]
    path = root / "environments/pare/gym-fm-02/tests/read_execution.py"
    spec = importlib.util.spec_from_file_location("pare_fm02_evidence_reader", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "oracle.txt").write_text(LOG_OUTPUT)
    solution = tmp_path / "solution" / "solve.sh"
    solution.parent.mkdir()
    solution.write_text("#!/bin/bash\ngym eval run --no-serve\n")
    return workspace, agent, solution


def test_saved_oracle_trajectory_keeps_reference_log_available(
    reader: ModuleType, paths: tuple[Path, Path, Path]
) -> None:
    workspace, agent, solution = paths
    trajectory = agent / "trajectory.json"
    shutil.copyfile(FIXTURE, trajectory)
    # The full oracle.txt was not retained on the host. Reconstruct its content
    # for this reader test from the actual output tail saved in the trajectory.
    saved = json.loads(trajectory.read_text())
    output = next(step["message"] for step in saved["steps"] if step.get("source") == "agent")
    (agent / "oracle.txt").write_text(output)
    log_text = (agent / "oracle.txt").read_text()

    evidence = reader.EvidenceFiles(EXPECTED, workspace=workspace, agent_root=agent, solution=solution)
    files = {item["path"]: item for item in evidence.catalog}
    assert files[str(trajectory)]["kind"] == "public_trace"
    assert files[str(agent / "oracle.txt")]["kind"] == "oracle"
    assert files[str(solution)]["kind"] == "oracle"
    assert not (agent / "exit-code.txt").exists()
    log_id = files[str(agent / "oracle.txt")]["id"]
    assert evidence.read(log_id, 0, len(log_text))["text"] == log_text
    assert "Finished rollout collection!" in output
    assert "Saved rollout: /workspace/results/carlos_rollouts.jsonl" in output
    initial = evidence.initial(100_000)
    assert any(item["file_id"] == log_id and item["text"] == log_text for item in initial["contents"])
    # The original filtering is preserved; assistant messages are not promoted
    # into command records, and the log alone does not create a scored result.
    assert "Finished rollout collection!" not in evidence.text(files[str(trajectory)]["id"])
    assert evidence.candidates == []


@pytest.mark.parametrize(
    ("name", "record"),
    [
        ("trajectory.json", {"agent": {"model_name": "oracle"}, "steps": [{"step_id": 1}]}),
        ("codex.txt", {"type": "thread.started", "thread_id": "thread-1"}),
        ("sessions/session.jsonl", {"type": "session_meta", "payload": {"id": "session-1"}}),
    ],
)
def test_identity_records_do_not_suppress_reference_logs(
    reader: ModuleType, paths: tuple[Path, Path, Path], name: str, record: dict
) -> None:
    workspace, agent, solution = paths
    trace = agent / name
    trace.parent.mkdir(exist_ok=True)
    trace.write_text(json.dumps(record))
    evidence = reader.EvidenceFiles(EXPECTED, workspace=workspace, agent_root=agent, solution=solution)
    assert {item["path"] for item in evidence.catalog} == {str(trace), str(agent / "oracle.txt"), str(solution)}


@pytest.mark.parametrize(
    ("name", "record"),
    [
        ("trajectory.json", {"steps": [{"step_id": 2, "tool_calls": [{"function_name": "exec_command"}]}]}),
        (
            "trajectory.json",
            {"steps": [{"step_id": 2, "observation": {"results": [{"content": "Process exited with code 7"}]}}]},
        ),
        ("codex.txt", {"type": "item.completed", "item": {"type": "command_execution", "exit_code": 7}}),
        ("codex.txt", {"type": "item.completed", "item": {"type": "mcp_tool_call", "status": "failed"}}),
        *[
            ("sessions/session.jsonl", {"type": "response_item", "payload": {"type": kind, "call_id": "call-1"}})
            for kind in ("function_call", "function_call_output", "custom_tool_call", "custom_tool_call_output")
        ],
    ],
)
def test_execution_records_keep_existing_discovery_behavior(
    reader: ModuleType, paths: tuple[Path, Path, Path], name: str, record: dict
) -> None:
    workspace, agent, solution = paths
    trace = agent / name
    trace.parent.mkdir(exist_ok=True)
    trace.write_text(json.dumps(record))
    evidence = reader.EvidenceFiles(EXPECTED, workspace=workspace, agent_root=agent, solution=solution)
    assert [item["path"] for item in evidence.catalog] == [str(trace)]
    assert evidence.text(evidence.catalog[0]["id"]) == reader.public_trace(trace)


def test_metadata_does_not_hide_execution_in_another_file(reader: ModuleType, paths: tuple[Path, Path, Path]) -> None:
    workspace, agent, solution = paths
    (agent / "trajectory.json").write_text(
        json.dumps({"steps": [{"tool_calls": [{"function_name": "exec_command"}]}]})
    )
    (agent / "codex.txt").write_text(json.dumps({"type": "thread.started", "thread_id": "thread-1"}))
    evidence = reader.EvidenceFiles(EXPECTED, workspace=workspace, agent_root=agent, solution=solution)
    assert {item["path"] for item in evidence.catalog} == {str(agent / "trajectory.json"), str(agent / "codex.txt")}


def test_missing_reference_files_are_not_invented(reader: ModuleType, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = tmp_path / "agent"
    agent.mkdir()
    trajectory = agent / "trajectory.json"
    shutil.copyfile(FIXTURE, trajectory)
    evidence = reader.EvidenceFiles(
        EXPECTED, workspace=workspace, agent_root=agent, solution=tmp_path / "absent-solve.sh"
    )
    assert [item["path"] for item in evidence.catalog] == [str(trajectory)]
    assert evidence.candidates == []
