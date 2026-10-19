# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from responses_api_agents.claweval_agent import core


def task_package(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "tasks"
    task = root / "G_video_qa_deadbeef"
    (task / "fixtures").mkdir(parents=True)
    (task / "grader_reference").mkdir()
    (task / "fixtures/video.mp4").write_bytes(b"video")
    (task / ".grade_artifacts.py").write_text("print('grade')\n")
    (task / ".grade_contract.json").write_text("{}\n")
    (task / "grader_reference/00.png").write_bytes(b"png")
    (task / "grader.py").write_text("class Placeholder:\n    pass\n")
    definition = {
        "task_id": task.name,
        "task_name": "fixture",
        "category": "video_qa",
        "prompt": {"text": "fresh prompt"},
        "sandbox_files": ["fixtures/video.mp4"],
        "sandbox_grader_files": [".grade_artifacts.py", ".grade_contract.json"],
        "local_grader_files": ["grader_reference/00.png"],
        "env_snapshot_files": ["/workspace/output.txt"],
        "env_snapshot_commands": ["python /workspace/.grade_artifacts.py"],
        "environment": {"timeout_seconds": 1200, "env_snapshot_timeout": 30},
    }
    (task / "task.yaml").write_text(yaml.safe_dump(definition, sort_keys=False))
    claw_source = tmp_path / "claw"
    (claw_source / "src/claw_eval").mkdir(parents=True)
    (claw_source / "src/claw_eval/__init__.py").write_text("")
    return root, claw_source


def test_prompt_only_dataset_drops_recorded_assistant_trace(tmp_path: Path) -> None:
    root, _ = task_package(tmp_path)
    tasks = core.inventory({"original_100": root})
    sft = tmp_path / "sft.jsonl"
    record = {
        "id": "original_100__G_video_qa_deadbeef_trial1",
        "conversations": [
            {"from": "system", "value": "OpenCode system"},
            {"from": "user", "value": "fresh prompt"},
            {"from": "assistant", "value": "gold answer must not leak"},
            {"from": "tool", "value": "private tool result"},
        ],
    }
    sft.write_text(json.dumps(record) + "\n")
    rows = core.dataset_rows(sft, tasks)
    assert rows == [
        {
            "responses_create_params": {"input": [{"role": "user", "content": "fresh prompt"}]},
            "verifier_metadata": {
                "source_id": record["id"],
                "task_id": "G_video_qa_deadbeef",
                "task_variant": "original_100",
                "family": "video_qa",
                "prompt_sha256": hashlib.sha256(b"fresh prompt").hexdigest(),
                "task_package_sha256": tasks["original_100::G_video_qa_deadbeef"]["task_package_sha256"],
            },
        }
    ]
    assert "gold answer" not in json.dumps(rows)
    assert "private tool result" not in json.dumps(rows)


def test_stage_keeps_private_grader_out_of_agent_workdir(tmp_path: Path) -> None:
    root, claw_source = task_package(tmp_path)
    task = core.inventory({"original_100": root})["original_100::G_video_qa_deadbeef"]
    staged = core.stage_harbor_task(task, "variant prompt", tmp_path / "staged", claw_eval_source=claw_source)
    assert (staged / "steps/solve/instruction.md").read_text() == "variant prompt"
    assert (staged / "steps/solve/workdir/fixtures/video.mp4").read_bytes() == b"video"
    assert (staged / "steps/solve/workdir/.claw_tools/mcp_server.py").is_file()
    assert "claw_ReadMedia" in (staged / "steps/solve/workdir/AGENTS.md").read_text()
    assert not (staged / "steps/solve/workdir/.grade_artifacts.py").exists()
    assert (staged / "steps/solve/tests/.grade_artifacts.py").is_file()
    assert (staged / "steps/solve/tests/judge_transport.py").read_bytes() == Path(core.__file__).with_name(
        "judge_transport.py"
    ).read_bytes()
    assert not (staged / "steps/solve/workdir/judge_transport.py").exists()
    private_task = yaml.safe_load((staged / "steps/solve/tests/task.yaml").read_text())
    assert private_task["prompt"]["text"] == "variant prompt"
    manifest = json.loads((staged / "staging_manifest.json").read_text())
    assert manifest["private_grader_material"].startswith("steps/solve/tests")


def test_source_id_validation() -> None:
    assert core.source_id_task_id("brief_1000__B_video_chart_123_trial7") == "B_video_chart_123"
    with pytest.raises(ValueError):
        core.source_id_task_id("G_video_chart_123")
    with pytest.raises(ValueError):
        core.source_id_task_id("other__T1_trial1")


def test_result_requires_native_reward_and_accepts_zero(tmp_path: Path) -> None:
    trial = tmp_path / "jobs/trial/G_video_qa_deadbeef/steps/solve"
    (trial / "agent").mkdir(parents=True)
    (trial / "verifier").mkdir()
    (trial.parent.parent / "result.json").write_text(
        json.dumps({"task_name": "claweval/G_video_qa_deadbeef", "step_results": []})
    )
    (trial / "agent/trajectory.json").write_text(json.dumps({"steps": [{"source": "agent"}]}))
    (trial / "verifier/reward.json").write_text(
        json.dumps({"reward": 0.0, "completion": 0.0, "robustness": 0.0, "safety": 1.0, "passed": False})
    )
    result = core.read_result(tmp_path, {"task_id": "G_video_qa_deadbeef"})
    assert result["reward"] == 0.0 and result["status"] == "OK"


def test_media_mcp_lists_tools_and_returns_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from responses_api_agents.claweval_agent import mcp_server

    monkeypatch.setattr(mcp_server, "WORKSPACE", tmp_path)
    monkeypatch.setattr(mcp_server, "STATE_DIR", tmp_path / ".claw_tools")
    mcp_server.STATE_DIR.mkdir()
    image = tmp_path / "frame.png"
    image.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c6360606060000000050001a5f645400000000049454e44ae426082"
        )
    )
    server = mcp_server.MediaServer()
    listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert [tool["name"] for tool in listed["result"]["tools"]] == [
        "ReadMedia",
        "BrowserScreenshot",
    ]
    called = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "ReadMedia", "arguments": {"path": str(image)}},
        }
    )
    assert called["result"]["isError"] is False
    assert called["result"]["content"][1]["type"] == "image"
    assert json.loads((mcp_server.STATE_DIR / "metrics.json").read_text())["media_frames"] == 1


def test_copy_or_link_copies_foreign_owned_fixture(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    destination = tmp_path / "staged" / "video.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(core.os, "geteuid", lambda: source.stat().st_uid + 1)

    core._copy_or_link(source, destination)

    assert destination.read_bytes() == b"video"
    assert destination.stat().st_ino != source.stat().st_ino
