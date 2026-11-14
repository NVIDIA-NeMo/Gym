# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest
import yaml

from nemo_gym.environment.onboarding import verify_environment
from nemo_gym.environment.registry import EnvironmentCatalogEntry
from nemo_gym.environment.validation import validate_environment
from nemo_gym.verification import exercise_verifier_fixture
from resources_servers.opencode_multimodal_benchmarks.app import VERIFIER_FIXTURE


@pytest.mark.asyncio
@pytest.mark.parametrize("fixture", [VERIFIER_FIXTURE])
async def test_catalog_verifier_contract(fixture):
    await exercise_verifier_fixture(fixture, reward_range=(0, 1), determinism="unknown")


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["vstar", "hrbench8k", "vtcbench", "zerobench"])
async def test_prepared_catalog_validation_and_verifier_need_no_services(benchmark, tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    catalog = root / "benchmarks/opencode_multimodal_benchmarks" / benchmark
    directory = tmp_path / "benchmarks/opencode_multimodal_benchmarks" / benchmark
    directory.mkdir(parents=True)
    manifest = yaml.safe_load((catalog / "manifest.yaml").read_text())
    dataset = manifest["datasets"][0]
    data_path = directory / "prepared.jsonl"
    names = {"vstar": "VStarBench", "hrbench8k": "HRBench8K", "vtcbench": "VTCBench", "zerobench": "ZeroBench"}
    data_path.write_text(
        json.dumps(
            {
                "responses_create_params": {"input": [{"role": "user", "content": "Select A or B."}]},
                "verifier_metadata": {
                    "benchmark": names[benchmark],
                    "index": "0",
                    "question": "Select A or B.",
                    "answer": "A",
                    "options": {"A": "cat", "B": "dog"},
                    "cycle_category": "0",
                    "expected_samples": 1,
                },
            }
        )
        + "\n"
    )
    dataset["jsonl_fpath"] = str(data_path)
    dataset["prepare_script"] = str(root / dataset["prepare_script"])
    manifest_path = directory / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest))
    config_path = directory / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "config_paths": [str(catalog / "config.yaml")],
                "opencode_multimodal_benchmarks": {
                    "resources_servers": {
                        "opencode_multimodal_benchmarks": {
                            "datasets": [dataset | {"taskset": "visual_tasks"}],
                        }
                    }
                },
            }
        )
    )
    for key in ("OPENSANDBOX_API_KEY", "OPENSANDBOX_DOMAIN", "GYM_JUDGE_API_KEY", "GYM_JUDGE_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    validation = validate_environment(manifest_path)
    assert validation.datasets[0].rows == 1
    report = await verify_environment(
        EnvironmentCatalogEntry(
            name=manifest["name"],
            kind="benchmark",
            path=directory,
            config_path=config_path,
            manifest_path=manifest_path,
        )
    )
    assert report.resources_server == "opencode_multimodal_benchmarks"
    assert report.cases
