# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selected rows retain prompts, hidden tests, ordering and full update budgets."""

import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from scripts import retarget_swe_training_recipe as retarget
from scripts.prepare_swe_single_agent_turn_recipe import OLD_AGENTS, ROOT


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
            "harbor_task_id": f"existing-task-{i}",
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
                task_id=row["harbor_task_id"],
                path=folder / row["instance_id"],
                task_data={"instance_id": row["instance_id"]},
                digest="b" * 64,
                instruction="Prepared Harbor instruction: " + row["instance_id"],
                turns=[],
                config=SimpleNamespace(agent=SimpleNamespace(timeout_sec=480)),
            )
            for row in rows
        ]

        def prepare_existing(target, **kwargs):
            assert target == str(folder)
            return SimpleNamespace(tasks=tasks, folder=folder, skipped=[])

        monkeypatch.setattr(retarget, "prepare_target", prepare_existing)
    out = tmp_path / "fresh"
    receipt = retarget.retarget(
        prepared=source,
        output=out,
        task_rows=data,
        dataset_manifest=manifest,
        source_server="swe_next",
        wandb_project="selected-project",
        wandb_entity="team",
        harbor_tasks=folder if route == "harbor" else None,
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
        expected = (
            [("user", "Prepared Harbor instruction: " + row["instance_id"])]
            if route == "harbor"
            else [(m["role"], m["content"]) for m in row["responses_create_params"]["input"]]
        )
        assert [(m["role"], m["content"]) for m in params["input"]] == expected
        assert "hidden-gold" not in json.dumps(params) and "hidden-test" not in json.dumps(params)
    if route != "harbor":
        assert "swe_next" in actual.env.nemo_gym.swe_resources.resources_servers
        assert "log_parser_cache_dir" not in actual.env.nemo_gym.swe_resources.resources_servers.swe_next
    else:
        assert "swe_resources" not in actual.env.nemo_gym
        assert not (out / "harbor").exists()
        assert json.loads((out / "provenance.json").read_text())["harbor_tasks"] == str(folder)


@pytest.mark.parametrize("invalid", ["missing-directory", "missing-id", "duplicate-id", "unknown-id", "skipped"])
def test_harbor_requires_matching_existing_tasks(tmp_path, monkeypatch, invalid):
    source = tmp_path / "parent"
    source.mkdir()
    config = OmegaConf.load(ROOT / "examples/training/swe_single_agent_turn.yaml")
    OmegaConf.save(config, source / "config.yaml")
    (source / "provenance.json").write_text(
        json.dumps({"route": "harbor", "harnesses": ["opencode"], "files_sha256": {}, "model": {}})
    )
    rows = [{"instance_id": "one", "harbor_task_id": "one"}, {"instance_id": "two", "harbor_task_id": "two"}]
    if invalid == "missing-id":
        rows[0].pop("harbor_task_id")
    elif invalid == "duplicate-id":
        rows[1]["harbor_task_id"] = "one"
    elif invalid == "unknown-id":
        rows[1]["harbor_task_id"] = "other"
    data = tmp_path / "tasks.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(
        retarget,
        "prepare_target",
        lambda *a, **kw: SimpleNamespace(
            tasks=[SimpleNamespace(task_id="one"), SimpleNamespace(task_id="two")],
            skipped={"bad": "unloadable"} if invalid == "skipped" else {},
        ),
    )
    with pytest.raises(ValueError, match="--harbor-tasks|harbor_task_id"):
        retarget.retarget(
            prepared=source,
            output=tmp_path / "out",
            task_rows=data,
            dataset_manifest=manifest,
            source_server="swe_next",
            wandb_project="test",
            wandb_entity="test",
            harbor_tasks=None if invalid == "missing-directory" else tmp_path / "harbor",
        )
    assert not (tmp_path / "out" / "train.jsonl").exists()
