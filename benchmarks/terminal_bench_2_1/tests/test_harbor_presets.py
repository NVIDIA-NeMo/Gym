# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Terminal-Bench 2.1 presets that run on the Harbor environment server, and the recipes that stay on the relay."""

import argparse
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import nemo_gym.cli.eval as cli_eval
from nemo_gym.config.global_config import GlobalConfigDictParser
from nemo_gym.data.tasks.harbor.prepare import PreparedTaskset, build_run, harbor_benchmark
from nemo_gym.data.tasks.harbor.task import load_task
from tests.unit_tests.data.tasks.harbor.harbor_build_helpers import PULL_DOCKERFILE, write_task


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
ROOT = BENCHMARK_DIR.parents[1]


@pytest.mark.parametrize(
    ("preset", "harness"),
    [("terminus_2", "terminus_2_sandboxed_agent"), ("opencode", "opencode_agent")],
)
def test_preset_runs_the_entry_with_its_harness(preset: str, harness: str) -> None:
    benchmark = harbor_benchmark(BENCHMARK_DIR / f"{preset}.yaml")
    assert benchmark.target == "terminal_bench_2_1"
    assert (benchmark.agent.instance_name, benchmark.agent.impl_name) == (harness, harness)


def _merged_num_repeats(tmp_path: Path, preset: str, agent_type: str | None, overrides: list[str]) -> int:
    """The run config's ``num_repeats`` for ``preset``, from its config files merged as Gym merges them."""
    args = argparse.Namespace(target=None, agent=None, agent_type=agent_type)
    agent, _, _ = cli_eval._benchmark_agent(args, [], harbor_benchmark(BENCHMARK_DIR / f"{preset}.yaml"))
    folder = tmp_path / "ds"
    tasks = [load_task(write_task(folder / "a", dockerfile=PULL_DOCKERFILE))]
    prepared = PreparedTaskset("ds", folder, tasks, tmp_path / "out" / "tasks.jsonl", tmp_path / "out")
    prepared.output_dir.mkdir(parents=True)
    _, tokens = build_run(prepared, agent, sandbox=None, overrides=list(overrides))
    paths = [
        json.loads(entry) if entry.startswith('"') else entry
        for token in tokens
        if token.startswith("+config_paths=[")
        for entry in token[len("+config_paths=[") : -1].split(",")
    ]
    _, configs = GlobalConfigDictParser().load_extra_config_paths(paths)
    # The command-line overrides `build_run` passes on apply after every config file.
    dotlist = [token.lstrip("+") for token in tokens if not token.startswith("+config_paths=")]
    return OmegaConf.merge(*configs, OmegaConf.from_dotlist(dotlist)).num_repeats


@pytest.mark.parametrize("preset", ["terminus_2", "opencode"])
@pytest.mark.parametrize("agent_type", [None, "pi_agent"])
def test_preset_keeps_eight_repeats_over_the_entrys_default(tmp_path: Path, preset: str, agent_type) -> None:
    """The run config's `num_repeats` is the entry's default (1); the preset's 8 follows it, for its own harness and
    a swapped-in one, and the command line still wins."""
    assert _merged_num_repeats(tmp_path / "default", preset, agent_type, []) == 8
    assert _merged_num_repeats(tmp_path / "override", preset, agent_type, ["+num_repeats=1"]) == 1


@pytest.mark.parametrize(
    "config",
    [
        "benchmarks/terminal_bench_2_1/terminus_2_legacy_agent.yaml",
        "benchmarks/terminal_bench_2_1/opencode_legacy_agent.yaml",
        "benchmarks/terminal_bench_2_1/inkling_small.yaml",
        "benchmarks/nemotron_3.5_super/eval_container_config.yaml",
        "benchmarks/nemotron_3.5_super/eval_container_config_with_staged.yaml",
    ],
)
def test_relay_recipes_still_compose_their_own_servers(config: str) -> None:
    """These include no Harbor-backed preset, so `--benchmark`/`--config` keep composing the resources server."""
    assert harbor_benchmark(ROOT / config) is None
