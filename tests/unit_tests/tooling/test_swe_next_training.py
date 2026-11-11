# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selected rows retain prompts, hidden tests, ordering and full update budgets."""

import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from scripts import retarget_swe_training_recipe as retarget
from scripts.prepare_swe_single_agent_turn_recipe import OLD_AGENTS, ROOT
from scripts.swe_rows_to_harbor import convert


@pytest.mark.parametrize("route", ["old-run", "single-agent-turn", "harbor"])
@pytest.mark.parametrize("harnesses", [["opencode"], list(OLD_AGENTS)])
def test_retarget_preserves_rows_and_runs_twenty_complete_epochs(tmp_path, monkeypatch, route, harnesses):
    source = tmp_path / "parent"
    source.mkdir()
    config = OmegaConf.load(ROOT / "examples/training/swe_single_agent_turn.yaml")
    config.policy.model_name = "original-checkpoint"
    config.grpo.max_num_steps = 20
    config.grpo.max_num_epochs = 5
    config.grpo.num_generations_per_prompt = 4
    config.logger.log_dir = str(source / "logs")
    config.checkpointing.checkpoint_dir = str(source / "checkpoints")
    OmegaConf.save(config, source / "config.yaml")
    (source / "provenance.json").write_text(
        json.dumps(
            {
                "route": route,
                "harnesses": harnesses,
                "model": {"path": "original-checkpoint"},
                "files_sha256": {"config.yaml": "old"},
                "tasks": ["old"],
            }
        )
    )
    rows = [
        {
            "instance_id": f"task-{i}",
            "image_ref": "image@sha256:" + "a" * 64,
            "workdir": "/testbed",
            "patch": "hidden-gold",
            "test_patch": "hidden-test",
            "expected_output_json": '{"test": "PASSED"}',
            "responses_create_params": {"input": [{"role": "user", "content": f"original prompt {i}"}]},
        }
        for i in range(8)
    ]
    data = tmp_path / "tasks.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"dataset":"selected"}')
    if route == "harbor":
        folder = tmp_path / "loaded"
        tasks = [
            SimpleNamespace(
                task_id=convert.task_folder_name("swe_next", row["instance_id"]),
                path=folder / row["instance_id"],
                task_data={"instance_id": row["instance_id"]},
                digest="b" * 64,
                instruction=row["responses_create_params"]["input"][0]["content"],
                turns=[],
                config=SimpleNamespace(agent=SimpleNamespace(timeout_sec=480)),
            )
            for row in rows
        ]
        monkeypatch.setattr(
            retarget, "prepare_target", lambda *a, **kw: SimpleNamespace(tasks=tasks, folder=folder, skipped=[])
        )
    out = tmp_path / "fresh"
    receipt = retarget.retarget(
        prepared=source,
        output=out,
        task_rows=data,
        dataset_manifest=manifest,
        source_server="swe_next",
        wandb_project="selected-project",
        wandb_entity="team",
    )
    assert (out / "source.jsonl").read_bytes() == data.read_bytes()
    assert receipt["epochs"] == 20 and receipt["episodes"] == 640 * len(harnesses)
    actual = OmegaConf.load(out / "config.yaml")
    assert actual.policy.model_name == "original-checkpoint"
    assert actual.grpo.num_prompts_per_step == 8 * len(harnesses)
    assert actual.policy.train_global_batch_size == 32 * len(harnesses)
    assert actual.logger.wandb.project == "selected-project" and actual.logger.wandb.mode == "online"
    records = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    assert len(records) == 8 * len(harnesses)
    for record, row in zip(records, [r for r in rows for _ in harnesses]):
        if route == "old-run":
            assert record["instance_id"] == row["instance_id"]
            params = record["responses_create_params"]
        else:
            assert row["instance_id"] in record["task_id"]["task_id"]
            params = record["task_input"]["responses_create_params"]
        assert [(m["role"], m["content"]) for m in params["input"]] == [
            (m["role"], m["content"]) for m in row["responses_create_params"]["input"]
        ]
        assert "hidden-gold" not in json.dumps(params) and "hidden-test" not in json.dumps(params)
    if route != "harbor":
        assert "swe_next" in actual.env.nemo_gym.swe_resources.resources_servers
        assert "log_parser_cache_dir" not in actual.env.nemo_gym.swe_resources.resources_servers.swe_next
    else:
        assert "swe_resources" not in actual.env.nemo_gym
        task = out / "harbor" / convert.task_folder_name("swe_next", "task-0")
        assert (task / "instruction.md").read_text() == "original prompt 0"
        grading = json.loads((task / "tests/nemo_gym/task.json").read_text())
        assert grading["source"] == "swe_next"
        assert grading["row"]["expected_output_json"] == rows[0]["expected_output_json"]
        assert "patch" not in grading["row"]
        setup = (task / "environment/setup.sh").read_text()
        assert setup.index("rm -rf r2e_tests") < setup.index("git init")
        assert setup.startswith("#!/bin/bash\nset -e\n")
        assert "seed-base" not in setup
        assert (task / "tests/nemo_gym/resources_servers/swe_next/verification.py").read_bytes() == (
            ROOT / "resources_servers/swe_next/verification.py"
        ).read_bytes()


@pytest.mark.parametrize(
    "status,reward,completed", [("PASSED", 1.0, True), ("FAILED", 0.0, True), ("timeout", 0.0, False)]
)
def test_swe_next_harbor_grader_uses_expected_statuses(tmp_path, monkeypatch, status, reward, completed):
    from tests.unit_tests.tooling.test_swe_rows_to_harbor import _load_grader

    row = {
        "instance_id": "example",
        "image_ref": "image",
        "workdir": "/testbed",
        "patch": "gold",
        "test_patch": "hidden tests",
        "expected_output_json": '{"test_one": "PASSED"}',
        "responses_create_params": {"input": [{"role": "user", "content": "original task"}]},
    }
    task = tmp_path / "task"
    convert.write_task(convert.SWENextAdapter().plan(row), task, agent_timeout_sec=480, metadata={})
    grader = _load_grader(task)

    async def execute(self, command, **kwargs):
        if command.startswith("ln -sfn"):
            return SimpleNamespace(return_code=0, error_type=None)
        return SimpleNamespace(
            stdout="___NEMO_GYM_SWE_NEXT_TEST_BEGIN___\n"
            + status
            + " r2e_tests/test.py::test_one\n___NEMO_GYM_SWE_NEXT_TEST_END___\n",
            stderr="",
            return_code=None if status == "timeout" else 0,
            error_type="timeout" if status == "timeout" else None,
        )

    monkeypatch.setattr(grader.LocalSandbox, "exec", execute)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "model_patch.diff").write_text("agent patch")
    logs = tmp_path / "logs"
    assert grader.main(["--root", str(tmp_path / "root"), "--logs", str(logs), "--artifacts", str(artifacts)]) == 0
    result = json.loads((logs / "reward.json").read_text())
    assert result["reward"] == reward and bool(result["evaluation_completed"]) is completed
    assert (tmp_path / "root/tmp/nemo_gym_test_patch.diff").read_text() == "hidden tests"
