# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from pytest import raises

from nemo_gym.dataset_metrics import DatasetMetricsHookError, load_dataset_metrics_hook


def test_load_dataset_metrics_hook_returns_none_when_module_is_absent(tmp_path: Path) -> None:
    assert load_dataset_metrics_hook(tmp_path) is None


def test_load_dataset_metrics_hook_loads_owner_export(tmp_path: Path) -> None:
    (tmp_path / "dataset_metrics.py").write_text(
        'def compute_task_metrics(task_input):\n    return {"Categories": task_input["category"]}\n'
    )

    hook = load_dataset_metrics_hook(tmp_path)

    assert hook is not None
    assert hook({"category": "reasoning"}) == {"Categories": "reasoning"}


def test_load_dataset_metrics_hook_rejects_missing_export(tmp_path: Path) -> None:
    (tmp_path / "dataset_metrics.py").write_text("OTHER_EXPORT = True\n")

    with raises(DatasetMetricsHookError, match="does not export"):
        load_dataset_metrics_hook(tmp_path)


def test_load_dataset_metrics_hook_rejects_import_failure(tmp_path: Path) -> None:
    (tmp_path / "dataset_metrics.py").write_text("raise RuntimeError('broken hook')\n")

    with raises(DatasetMetricsHookError, match="Failed to import.*broken hook"):
        load_dataset_metrics_hook(tmp_path)


def test_load_dataset_metrics_hook_rejects_non_callable_export(tmp_path: Path) -> None:
    (tmp_path / "dataset_metrics.py").write_text("compute_task_metrics = 42\n")

    with raises(DatasetMetricsHookError, match="not callable"):
        load_dataset_metrics_hook(tmp_path)


def test_load_dataset_metrics_hook_is_cached_per_server(tmp_path: Path) -> None:
    server_dir = tmp_path / "resources_servers" / "example"
    server_dir.mkdir(parents=True)
    (server_dir / "dataset_metrics.py").write_text(
        'def compute_task_metrics(task_input):\n    return {"Categories": task_input["category"]}\n'
    )

    first = load_dataset_metrics_hook(server_dir)
    second = load_dataset_metrics_hook(server_dir)

    assert first is second


def test_load_dataset_metrics_hook_module_key_includes_server_type(tmp_path: Path) -> None:
    resources = tmp_path / "resources_servers" / "example"
    agent = tmp_path / "responses_api_agents" / "example"
    resources.mkdir(parents=True)
    agent.mkdir(parents=True)
    (resources / "dataset_metrics.py").write_text(
        'OWNER = "resources"\ndef compute_task_metrics(task_input):\n    return {}\n'
    )
    (agent / "dataset_metrics.py").write_text(
        'OWNER = "agent"\ndef compute_task_metrics(task_input):\n    return {}\n'
    )

    resources_hook = load_dataset_metrics_hook(resources)
    agent_hook = load_dataset_metrics_hook(agent)

    assert resources_hook is not None and resources_hook.__module__.endswith(".resources_servers.example")
    assert agent_hook is not None and agent_hook.__module__.endswith(".responses_api_agents.example")
