# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check preset identity, composition, variant preservation and override boundaries."""

import importlib.util
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
module_spec = importlib.util.spec_from_file_location("model_presets", ROOT / "nemo_gym/model_presets.py")
presets = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(presets)


@pytest.fixture
def case(tmp_path):
    path = tmp_path / "preset.json"
    data = {
        "schema_version": 1,
        "name": "example-model",
        "checkpoint": {"repository": "example/model", "revision": "b" * 40},
        "runtime": {"source": "https://example.test/runtime", "sha256": "c" * 64},
        "service": {
            "type": "vllm",
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "number_of_instances": 1,
            "extra_args": "",
            "pre_command": "",
            "env": {},
            "health_check": {},
        },
        "run": {"responses_create_params.temperature": 0.5, "responses_create_params.max_output_tokens": 128},
        "sources": [{"url": "https://example.test/spec", "sha256": "d" * 64, "description": "Synthetic fixture"}],
    }
    path.write_text(json.dumps(data))
    return (
        path,
        {
            "checkpoint": "/shared/weights",
            "checkpoint_repository": "example/model",
            "checkpoint_revision": "b" * 40,
            "container": "/shared/runtime.sqsh",
            "runtime_sha256": "c" * 64,
            "cache": "/shared/cache",
            "served_model_name": "example-model",
        },
        {"repository": "https://example.test/presets", "revision": "a" * 40, "path": "preset.json"},
    )


def render(case, **kwargs):
    path, bindings, source = case
    return presets.render_preset(path, bindings=bindings, benchmarks=["example_task"], source=source, **kwargs)


def test_native_fragment_keeps_model_and_benchmark_separate(case):
    result = render(case)
    composed = OmegaConf.to_container(OmegaConf.create(result), resolve=True)
    assert composed["services"]["policy"]["pre_command"] == result["services"]["policy"]["pre_command"]
    assert "compute" not in result
    assert "prepare" not in result["driver"]["benchmarks"]["example_task"]
    assert result["driver"]["policy_model_type"] == "vllm_model"
    assert result["services"]["policy"]["mounts"] == ["/shared/weights:/checkpoint:ro", "/shared/cache:/cache"]
    provenance = json.loads(result["driver"]["env"]["GYM_MODEL_PRESET_PROVENANCE"].removeprefix("lit:"))
    assert provenance["revision"] == "a" * 40
    assert provenance["checkpoint"]["revision"] == case[1]["checkpoint_revision"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_repository", "unrelated-model"),
        ("checkpoint_revision", "e" * 40),
        ("runtime_sha256", "b" * 64),
    ],
)
def test_incompatible_assets_are_rejected(case, field, value):
    case[1][field] = value
    with pytest.raises(ValueError, match="mismatch"):
        render(case)


def test_mutable_source_reference_is_rejected(case):
    case[2]["revision"] = "main"
    with pytest.raises(ValueError, match="immutable"):
        render(case)


def test_malformed_artifact_revision_is_rejected(case, tmp_path):
    preset = presets.read_json(case[0])
    preset["checkpoint"]["revision"] = "sha256:unversioned"
    path = tmp_path / "preset.json"
    path.write_text(json.dumps(preset))
    with pytest.raises(ValueError, match="immutable"):
        render((path, case[1], case[2]))


def test_cluster_scratch_mount_is_preserved(case):
    case[1]["mounts"] = ["/raid/scratch:/raid/scratch"]
    result = render(case)
    assert result["services"]["policy"]["mounts"] == [
        "/shared/weights:/checkpoint:ro",
        "/shared/cache:/cache",
        "/raid/scratch:/raid/scratch",
    ]


def test_cluster_mount_source_can_be_bound_after_rendering(case):
    case[1]["mounts"] = ["${paths.scratch}:/raid/scratch"]
    result = render(case)
    assert result["services"]["policy"]["mounts"][-1] == case[1]["mounts"][0]


@pytest.mark.parametrize(
    "mounts",
    [
        ["/elsewhere:/checkpoint"],
        ["/elsewhere:/cache/hidden"],
        ["/one:/scratch", "/two:/scratch"],
        ["relative:/scratch"],
        ["${source}/suffix:/scratch"],
        ["/source:${target}"],
    ],
)
def test_invalid_cluster_mounts_are_rejected(case, mounts):
    case[1]["mounts"] = mounts
    with pytest.raises(ValueError, match="[Mm]ount"):
        render(case)


def test_explicit_override_records_reason(case):
    field = "run.responses_create_params.temperature"
    result = render(case, overrides={field: {"value": 0.0, "reason": "A deliberately selected greedy comparison"}})
    assert result["driver"]["benchmarks"]["example_task"]["run"]["responses_create_params.temperature"] == 0.0
    provenance = json.loads(result["driver"]["env"]["GYM_MODEL_PRESET_PROVENANCE"].removeprefix("lit:"))
    assert provenance["overrides"][field]["reason"]


@pytest.mark.parametrize(
    "change",
    [
        {"run.responses_create_params.temperature": {"value": 0.0}},
        {"run.responses_create_params.temperature": {"value": 0.0, "reason": ""}},
        {"run.responses_create_params.temperature": {"value": "0", "reason": "Wrong type"}},
        {"run.responses_create_params.input": {"value": [], "reason": "Injected prompt"}},
    ],
)
def test_unexplained_or_unrecognized_override_is_rejected(case, change):
    with pytest.raises(ValueError):
        render(case, overrides=change)


def test_duplicate_json_keys_cannot_hide_settings(tmp_path):
    path = tmp_path / "preset.json"
    path.write_text('{"service": {}, "service": {"extra_args": "hidden"}}')
    with pytest.raises(ValueError, match="Duplicate"):
        presets.read_json(path)


def test_fragment_validates_against_submit_schema(case):
    from nemo_gym.orchestration.api import DriverConfig, VllmServiceConfig

    result = render(case)
    driver = DriverConfig.model_validate(result["driver"])
    service = VllmServiceConfig.model_validate(result["services"]["policy"])
    assert driver.policy_model == "policy"
    assert driver.benchmarks["example_task"].run["responses_create_params.max_output_tokens"] == 128
    assert service.model == "/checkpoint"
    assert service.tensor_parallel_size == 1


def test_cli_runs_without_site_packages(case, tmp_path):
    import subprocess
    import sys

    path, bindings, source = case
    bindings_path, source_path = tmp_path / "bindings.json", tmp_path / "source.json"
    bindings_path.write_text(json.dumps(bindings))
    source_path.write_text(json.dumps(source))
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "nemo_gym/model_presets.py"),
            "--preset",
            str(path),
            "--bindings",
            str(bindings_path),
            "--source",
            str(source_path),
            "--benchmark",
            "example_task",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout) == render(case)
