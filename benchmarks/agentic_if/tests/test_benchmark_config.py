# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.environment.manifest import load_manifest, resolve_manifest_config_path
from nemo_gym.environment.validation import _mirror_differences, _resolve_manifest_composition


BENCHMARK_DIR = Path(__file__).parents[1]


def test_manifest_mirrors_the_resolved_config() -> None:
    path = BENCHMARK_DIR / "manifest.yaml"
    manifest = load_manifest(path)
    composition = _resolve_manifest_composition(
        resolve_manifest_config_path(path, manifest), dataset_owner=manifest.dataset_owner
    )

    assert _mirror_differences(manifest, composition) == {}


def test_the_benchmark_routes_to_opencode_through_single_agent_turn() -> None:
    benchmark = BenchmarkConfig.from_config_path(BENCHMARK_DIR / "config.yaml")

    assert benchmark.name == "agentic_if"
    assert benchmark.dataset.taskset == "agentic_if:benchmark"
    assert benchmark.environment_server == "agentic_if_environment_server"
    assert benchmark.agent_name == "agentic_if_opencode_agent"
