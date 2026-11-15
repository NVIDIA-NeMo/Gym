# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark path interpolation must survive structural server validation."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.config.config_types import maybe_get_server_instance_config


@pytest.mark.parametrize("reference", ["${dataset_path}", "${oc.env:BENCHMARK_TEST_PATH}"])
def test_benchmark_dataset_paths_resolve_in_the_original_config(tmp_path, monkeypatch, reference):
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.setenv("BENCHMARK_TEST_PATH", str(dataset))
    config = OmegaConf.create(
        dict(
            dataset_path=str(dataset),
            prepare_path=str(tmp_path / "prepare.py"),
            agent=dict(
                responses_api_agents=dict(
                    example=dict(
                        entrypoint="app.py",
                        datasets=[
                            dict(
                                name="benchmark",
                                type="benchmark",
                                jsonl_fpath=reference,
                                prepare_script="${prepare_path}",
                                prompt_config=None,
                            )
                        ],
                    )
                )
            ),
        )
    )
    instance, error = maybe_get_server_instance_config("agent", config.agent)
    assert error is None
    prepared = instance.datasets[0]
    assert prepared.jsonl_fpath == dataset
    assert prepared.jsonl_fpath.read_text() == "{}\n"
    assert prepared.prepare_script == tmp_path / "prepare.py"
    assert prepared.prompt_config is None
    # Inspection must not replace the original interpolation with a stale resolved path.
    assert (
        OmegaConf.to_container(config, resolve=False)["agent"]["responses_api_agents"]["example"]["datasets"][0][
            "jsonl_fpath"
        ]
        == reference
    )
    other = tmp_path / "other.jsonl"
    other.write_text('{"changed": true}\n')
    if reference == "${dataset_path}":
        config.dataset_path = str(other)
    else:
        monkeypatch.setenv("BENCHMARK_TEST_PATH", str(other))
    refreshed, error = maybe_get_server_instance_config("agent", config.agent)
    assert error is None
    assert refreshed.datasets[0].jsonl_fpath == other


def test_literal_benchmark_paths_keep_path_api():
    config = OmegaConf.create(
        dict(
            agent=dict(
                responses_api_agents=dict(
                    example=dict(
                        entrypoint="app.py",
                        datasets=[dict(name="benchmark", type="benchmark", jsonl_fpath="data/input.jsonl")],
                    )
                )
            )
        )
    )
    instance, error = maybe_get_server_instance_config("agent", config.agent)
    assert error is None
    assert instance.datasets[0].jsonl_fpath == Path("data/input.jsonl")


@pytest.mark.parametrize("field", ["jsonl_fpath", "prepare_script", "prompt_config", "prepare_project"])
def test_unused_benchmark_paths_may_keep_an_unset_environment_variable(monkeypatch, tmp_path, field):
    monkeypatch.delenv("UNSET_BENCHMARK_PATH", raising=False)
    reference = "${oc.env:UNSET_BENCHMARK_PATH}"
    dataset = dict(name="benchmark", type="benchmark", jsonl_fpath="data/input.jsonl")
    dataset[field] = reference
    config = OmegaConf.create(
        dict(agent=dict(responses_api_agents=dict(example=dict(entrypoint="app.py", datasets=[dataset]))))
    )
    instance, error = maybe_get_server_instance_config("agent", config.agent)
    assert error is None
    assert str(getattr(instance.datasets[0], field)) == reference
    path = tmp_path / "provided-later"
    monkeypatch.setenv("UNSET_BENCHMARK_PATH", str(path))
    instance, error = maybe_get_server_instance_config("agent", config.agent)
    assert error is None
    assert getattr(instance.datasets[0], field) == path


def test_vision2code_eval_config_loads_without_unused_task_dataset(monkeypatch):
    monkeypatch.delenv("VISION2CODE_EVAL_TASKS", raising=False)
    for name in ("SQSH", "POLICY_URL", "POLICY_MODEL", "JUDGE_URL", "JUDGE_MODEL"):
        monkeypatch.setenv("VISION2CODE_" + name, "/unused")
    config = OmegaConf.merge(
        OmegaConf.load("resources_servers/vision2code/configs/vision2code.yaml"),
        OmegaConf.load("resources_servers/vision2code/configs/vision2code_evals.yaml"),
    )
    instance, error = maybe_get_server_instance_config("vision2code", config.vision2code)
    assert error is None
    assert str(instance.datasets[0].jsonl_fpath) == "${oc.env:VISION2CODE_EVAL_TASKS}"
    assert (
        OmegaConf.to_container(config, resolve=False)["vision2code"]["resources_servers"]["vision2code"]["datasets"][
            0
        ]["jsonl_fpath"]
        == "${oc.env:VISION2CODE_EVAL_TASKS}"
    )
