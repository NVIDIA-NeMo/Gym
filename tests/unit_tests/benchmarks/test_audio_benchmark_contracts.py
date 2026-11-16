# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declaration and profile contracts that every audio benchmark (``benchmarks/<name>``) must satisfy.

The audio benchmarks are discovered from ``benchmarks/*/manifest.yaml``: every benchmark whose resources server is one of
the audio servers is checked, so adding a benchmark needs no change here. Benchmark-specific values (licenses, request
limits, metric types) are asserted in each ``test_<name>_benchmark.py``.
"""

import importlib
from pathlib import Path

import pytest
import yaml

from benchmarks import audio_manifest
from nemo_gym.environment.benchmarks import BenchmarkConfig, _benchmark_config_name, _is_benchmark_config
from nemo_gym.environment.manifest import load_manifest
from resources_servers.audio_eval.score import evaluator_settings, resolve


pytestmark = pytest.mark.usefixtures("mock_port_allocation")

ROOT = Path(__file__).resolve().parents[3]
AUDIO_SERVERS = ("audio_eval", "asr_with_pc", "voicebench")


def discover() -> dict[str, str]:
    """``{benchmark name: resources server}`` of every audio benchmark that declares a manifest."""
    found = {}
    for path in sorted((ROOT / "benchmarks").glob("*/manifest.yaml")):
        server = (yaml.safe_load(path.read_text()) or {}).get("resources_server")
        if server in AUDIO_SERVERS:
            found[path.parent.name] = server
    return found


AUDIO_BENCHMARKS = discover()
WITH_PROFILES = sorted(name for name in AUDIO_BENCHMARKS if (ROOT / "benchmarks" / name / "profiles.json").exists())


def test_the_audio_benchmark_names_are_unique_directories() -> None:
    assert all((ROOT / "benchmarks" / name / "config.yaml").is_file() for name in AUDIO_BENCHMARKS)


@pytest.mark.parametrize("name", sorted(AUDIO_BENCHMARKS))
def test_declaration_is_discovered_and_consistent(name: str) -> None:
    directory = ROOT / "benchmarks" / name
    config_path = directory / "config.yaml"
    assert _is_benchmark_config(config_path)
    assert _benchmark_config_name(config_path.relative_to(ROOT / "benchmarks")) == name

    # ``policy_base_url`` and the other policy settings are runtime values: a config that interpolates them only
    # validates non-strictly.
    strict = "${policy_base_url}" not in config_path.read_text()
    benchmark = BenchmarkConfig.from_config_path(config_path, strict=strict)
    dataset = benchmark.dataset
    assert benchmark.name == name and benchmark.num_repeats == 1
    assert dataset.type == "benchmark"
    assert dataset.prepare_script == Path(f"benchmarks/{name}/prepare.py")

    manifest = load_manifest(directory / "manifest.yaml")
    assert manifest.kind == "benchmark"
    assert len(manifest.datasets) == 1
    assert manifest.datasets[0].jsonl_fpath == str(dataset.jsonl_fpath)
    assert manifest.datasets[0].num_repeats == 1

    if AUDIO_BENCHMARKS[name] == "audio_eval":
        environment = yaml.safe_load(config_path.read_text())[f"{name}_environment_server"]
        assert environment["_inherit_from"] == "audio_eval_environment_server"

    # A ``prepare()`` module that declares ``OUTPUT_FPATH`` must write where the benchmark reads. A module may import a
    # scoring library that only its resources server installs; its output path is then checked where that library is
    # available.
    try:
        output = getattr(importlib.import_module(f"benchmarks.{name}.prepare"), "OUTPUT_FPATH", None)
    except ImportError:
        return
    if output is not None:
        assert Path(dataset.jsonl_fpath) == Path("benchmarks") / name / "data" / output.name
        assert Path(dataset.jsonl_fpath).absolute() == output.absolute()


@pytest.mark.parametrize("name", WITH_PROFILES)
def test_profile_contracts_resolve_and_have_no_unread_keys(name: str) -> None:
    profiles = audio_manifest.load_profiles(ROOT / "benchmarks" / name / "profiles.json")["profiles"]
    assert profiles
    for profile, contract in profiles.items():
        judged = "judge_prompt_template" in contract
        assert set(contract) == {"prompt_override", "request", "skills"} | (
            {"judge_prompt_template"} if judged else set()
        )
        # A judged profile may be scored by the judge alone, with no evaluator arguments.
        assert {"METRICS_TYPE"} <= set(contract["skills"]) <= {"EVAL_ARGS", "METRICS_TYPE"}, profile
        assert judged or "EVAL_ARGS" in contract["skills"], profile
        assert isinstance(audio_manifest.request_controls(contract)["max_output_tokens"], int), profile


@pytest.mark.parametrize("name", WITH_PROFILES)
def test_profile_scorers_are_found_by_name(name: str) -> None:
    """The evaluator and metrics a profile names exist as modules (a missing third-party library is not an error here)."""
    profiles = audio_manifest.load_profiles(ROOT / "benchmarks" / name / "profiles.json")["profiles"]
    for profile, contract in profiles.items():
        evaluator, _ = evaluator_settings(contract)
        for kind, scorer in (("evaluators", evaluator), ("metrics", contract["skills"]["METRICS_TYPE"])):
            if scorer is None:
                continue
            try:
                resolve(kind, scorer)
            except ImportError as error:
                assert not (error.name or "").startswith("resources_servers"), (profile, kind, scorer, error)
