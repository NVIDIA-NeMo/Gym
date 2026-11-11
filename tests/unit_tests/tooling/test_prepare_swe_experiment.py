# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from scripts.prepare_swe_experiment import prepare_experiment


@pytest.fixture
def prepared(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    config = {
        "policy": {
            "model_name": "/original/bf77",
            "train_global_batch_size": 32,
            "generation": {"temperature": 1.0, "max_new_tokens": 1024},
            "optimizer": {"lr": 3e-6},
        },
        "data": {"train": {"data_path": str(original / "train.jsonl")}, "shuffle": False},
        "grpo": {"seed": 42},
        "async_rl": {"max_inflight_prompts": 16},
        "checkpointing": {"checkpoint_dir": str(original / "checkpoints")},
        "logger": {
            "log_dir": str(original / "logs"),
            "wandb": {"entity": "team", "project": "study", "id": "old", "resume": "must"},
        },
        "env": {
            "nemo_gym": {
                "cache_dir": str(original / "cache"),
                "pi_sandboxed_agent": {"responses_api_agents": {"pi_sandboxed_agent": {}}},
                "tasksets": {"frozen": "/frozen/tasks"},
            }
        },
    }
    template = original / "config.yaml"
    OmegaConf.save(OmegaConf.create(config), template)
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": {"task_id": str(i), "taskset": "swe_next_pi"},
                    "task_input": {"hidden": str(i)},
                }
            )
            + "\n"
            for i in range(32)
        )
    )
    return template, inputs, config


def test_training_preserves_original_policy_assets_and_distinct_task_schedule(tmp_path, prepared):
    template, inputs, before = prepared
    frozen = inputs.read_bytes(), template.read_bytes()
    out = tmp_path / "train"
    receipt = prepare_experiment(template, inputs, out, mode="train", policy="mixed")
    config = OmegaConf.to_container(OmegaConf.load(out / "config.yaml"))
    assert config["grpo"] == {
        "seed": 42,
        "max_num_steps": 100,
        "max_num_epochs": 50,
        "num_prompts_per_step": 16,
        "num_generations_per_prompt": 4,
    }
    assert config["policy"] == before["policy"] | {"train_global_batch_size": 64}
    assert config["data"]["train"]["data_path"] == str(inputs)
    assert config["env"]["nemo_gym"]["tasksets"] == before["env"]["nemo_gym"]["tasksets"]
    assert config["env"]["nemo_gym"]["pi_sandboxed_agent"]["responses_api_agents"]["pi_sandboxed_agent"][
        "artifacts_dir"
    ].startswith(str(out))
    assert config["checkpointing"]["save_period"] == 25
    assert receipt["expected_episodes"] == 6400
    assert "id" not in config["logger"]["wandb"] and "resume" not in config["logger"]["wandb"]
    assert (inputs.read_bytes(), template.read_bytes()) == frozen


def test_checkpoint_evaluation_expands_trials_and_loads_only_named_weights(tmp_path, prepared):
    template, inputs, before = prepared
    weights = tmp_path / "not-probed/checkpoints/step_25/policy/weights"
    out = tmp_path / "eval"
    receipt = prepare_experiment(
        template, inputs, out, mode="evaluate", policy="candidate", harness="pi", weights=weights, step=25
    )
    config = OmegaConf.to_container(OmegaConf.load(out / "config.yaml"))
    assert config["policy"]["pretrained_checkpoint"] == {"path": str(weights), "format": "megatron_bridge"}
    assert config["policy"]["generation"] == before["policy"]["generation"]
    assert not config["checkpointing"]["enabled"]
    assert config["checkpointing"]["checkpoint_dir"] == str(out / "checkpoints")
    assert config["env"]["nemo_gym"]["trajectory_collection_raw_results"] is True
    rows = [json.loads(line) for line in (out / "evaluation.jsonl").read_text().splitlines()]
    assert len(rows) == 96 and rows[0] == rows[1] == rows[2] and rows[2] != rows[3]
    assert receipt["expected_optimizer_updates"] == 0 and receipt["expected_episodes"] == 96
    assert json.loads((out / "EVALUATION.json").read_text()) == receipt
    assert not weights.exists()


def test_original_evaluation_and_existing_output_protection(tmp_path, prepared):
    template, inputs, _ = prepared
    out = tmp_path / "eval"
    prepare_experiment(template, inputs, out, mode="evaluate", policy="original", harness="pi")
    saved = (out / "config.yaml").read_bytes()
    assert "pretrained_checkpoint" not in OmegaConf.load(out / "config.yaml").policy
    with pytest.raises(FileExistsError):
        prepare_experiment(template, inputs, out, mode="evaluate", policy="original", harness="pi")
    assert (out / "config.yaml").read_bytes() == saved


@pytest.mark.parametrize("change", ["duplicate", "partial_batch", "weights_without_step"])
def test_invalid_population_or_checkpoint_identity_rejected(tmp_path, prepared, change):
    template, inputs, _ = prepared
    kwargs = {}
    if change == "duplicate":
        inputs.write_text(inputs.read_text() + inputs.read_text().splitlines()[0] + "\n")
    elif change == "partial_batch":
        kwargs["tasks_per_step"] = 17
    else:
        kwargs["weights"] = Path("/not-probed/weights")
    with pytest.raises(ValueError):
        prepare_experiment(template, inputs, tmp_path / "out", mode="train", policy="mixed", **kwargs)
    assert not (tmp_path / "out").exists()
