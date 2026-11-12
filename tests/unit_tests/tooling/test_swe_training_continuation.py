# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from omegaconf import OmegaConf
from scripts import prepare_swe_single_agent_turn_recipe as recipe
from scripts.swe_training_continuation import prepare_continuation


@pytest.fixture
def original(tmp_path):
    prepared = tmp_path / "original"
    prepared.mkdir()
    config = {
        "data": {"train": {"data_path": str(prepared / "train.jsonl"), "repeat": 1}},
        "checkpointing": {"checkpoint_dir": str(prepared / "checkpoints"), "save_period": 10, "load_optim": True},
        "grpo": {"max_num_steps": 20, "num_generations_per_prompt": 4},
        "policy": {"model_name": "/fixed/model", "train_global_batch_size": 32},
        "async_rl": {"max_inflight_prompts": 8},
        "logger": {
            "log_dir": str(prepared / "logs"),
            "wandb": {"project": "project", "entity": "entity", "name": "logical"},
        },
        "env": {
            "nemo_gym": {
                "cache_dir": str(prepared / "runtime-cache"),
                "results_dir": str(prepared / "results"),
                "resource": {
                    "verification_log_dir": str(prepared / "verifier"),
                    "log_parser_cache_dir": "/fixed/parser",
                },
                "agents": [
                    {"artifacts_dir": str(prepared / "harness-artifacts"), "settings": str(prepared / "settings.json")}
                ],
            }
        },
    }
    OmegaConf.save(OmegaConf.create(config), prepared / "config.yaml")
    (prepared / "settings.json").write_text("{}")
    (prepared / "source-mount.json").write_text('{"source":"/fixed/source"}')
    return prepared, config


@pytest.mark.parametrize(
    "route,harnesses",
    [
        ("single-agent-turn", ["claude_code"]),
        ("harbor", ["claude_code"]),
        ("harbor", ["opencode", "hermes", "pi", "codex", "claude_code"]),
    ],
)
def test_canonical_command_keeps_original_identity_and_assets(
    original, tmp_path, monkeypatch, capsys, route, harnesses
):
    prepared, config = original
    rows = [
        {"task_id": task, "agent_ref": harness, "task_dir": str(prepared / "harbor" / str(task))}
        for task in range(8)
        for harness in harnesses
    ]
    frozen = "".join(json.dumps(row) + "\n" for row in rows)
    (prepared / "train.jsonl").write_text(frozen)
    (prepared / "harbor").mkdir()
    (prepared / "harbor" / "private-grader").write_text("frozen grader")
    provenance = {"route": route, "harnesses": harnesses, "files_sha256": {"train.jsonl": "recorded-input-digest"}}
    (prepared / "provenance.json").write_text(json.dumps(provenance))
    output = tmp_path / "attempt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "recipe",
            "continue",
            "--prepared",
            str(prepared),
            "--output-dir",
            str(output),
            "--wandb-id",
            "same-run",
            "--prior-job-id",
            "123",
        ],
    )
    recipe.main()
    receipt = json.loads(capsys.readouterr().out)
    actual = OmegaConf.to_container(OmegaConf.load(output / "config.yaml"), resolve=False)
    for key in ("data", "checkpointing", "grpo", "policy", "async_rl"):
        assert actual[key] == config[key]
    assert actual["logger"]["wandb"] == {
        **config["logger"]["wandb"],
        "id": "same-run",
        "resume": "must",
        "mode": "online",
    }
    assert actual["logger"]["log_dir"] == str(output / "logs")
    env = actual["env"]["nemo_gym"]
    assert env["results_dir"] == str(output / "results")
    assert env["cache_dir"] == str(output / "runtime-cache")
    assert env["resource"]["verification_log_dir"] == str(output / "verifier")
    assert env["resource"]["log_parser_cache_dir"] == "/fixed/parser"
    assert env["agents"][0]["settings"] == str(prepared / "settings.json")
    assert env["agents"][0]["artifacts_dir"] == str(output / "harness-artifacts")
    assert (prepared / "train.jsonl").read_text() == frozen
    assert (prepared / "harbor" / "private-grader").read_text() == "frozen grader"
    assert not (output / "train.jsonl").exists() and not (output / "harbor").exists()
    assert (output / "source-mount.json").read_bytes() == (prepared / "source-mount.json").read_bytes()
    assert not (prepared / "checkpoints").exists()  # No checkpoint inspection or synthetic checkpoint required.
    assert not receipt["training_started"] and not receipt["terminal_state_verified"]
    saved = json.loads((output / "provenance.json").read_text())
    assert saved["inherited_preparation_files_sha256"] == provenance["files_sha256"]
    assert set(saved["files_sha256"]) == {"config.yaml", "source-mount.json"}
    with pytest.raises(FileExistsError):
        prepare_continuation(prepared, output, wandb_id="same-run", prior_job_id="123")


def test_rejects_reusing_a_failed_continuation_as_the_original(original, tmp_path):
    prepared, _ = original
    (prepared / "provenance.json").write_text('{"continuation": {}}')
    with pytest.raises(ValueError, match="original checkpoint-producing"):
        prepare_continuation(prepared, tmp_path / "next", wandb_id="same-run", prior_job_id="123")
    assert not (tmp_path / "next").exists()
