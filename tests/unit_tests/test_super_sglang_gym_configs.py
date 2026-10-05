# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep Gym evaluation settings consistent across SGLang topology recipes."""

from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parents[2] / "benchmarks/nemotron_3.5_super/sglang_configs"


def _benchmark_without_experiment_name(path: Path) -> dict[str, object]:
    config = yaml.safe_load(path.read_text())
    assert isinstance(config, dict), f"{path.name}: expected a YAML mapping"
    benchmark = config.get("benchmark")
    assert isinstance(benchmark, dict), f"{path.name}: missing benchmark mapping"
    assert benchmark.get("type") == "custom", f"{path.name}: expected a custom Gym benchmark"
    command = benchmark.get("command")
    assert isinstance(command, str) and command.strip(), f"{path.name}: missing benchmark command"
    env = benchmark.get("env")
    assert isinstance(env, dict), f"{path.name}: missing benchmark environment"
    experiment_name = env.pop("EXPERIMENT_NAME", None)
    assert isinstance(experiment_name, str) and experiment_name.strip(), f"{path.name}: missing EXPERIMENT_NAME"
    return benchmark


def test_gym_benchmark_sections_match() -> None:
    benchmarks = {}
    # Discover topology recipes automatically, excluding explicit serving-only recipes.
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        config = yaml.safe_load(path.read_text())
        if isinstance(config, dict) and config.get("benchmark") == {"type": "manual"}:
            continue
        benchmarks[path] = _benchmark_without_experiment_name(path)

    assert len(benchmarks) >= 2, "Expected at least two Gym recipes to check benchmark consistency"
    # Any existing recipe can be the reference; topology names may change or disappear.
    (reference_path, reference), *others = benchmarks.items()
    for path, benchmark in others:
        assert benchmark == reference, (
            f"{path.name}: benchmark differs from {reference_path.name}; only benchmark.env.EXPERIMENT_NAME may differ"
        )
