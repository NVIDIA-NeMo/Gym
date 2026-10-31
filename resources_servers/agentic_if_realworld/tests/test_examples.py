# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public CI examples are five independent synthetic tasks with real collected rollouts."""

import json
from pathlib import Path

from nemo_gym.cli.env import TestConfig as GymTestConfig
from nemo_gym.cli.env import _validate_data_single
from nemo_gym.harness_capabilities.checker import EvidenceScope, inspect_record
from resources_servers.agentic_if_realworld.benchmark.core import Dataset, judge
from resources_servers.agentic_if_realworld.benchmark.synthetic import EXAMPLE_LITERALS, create, review


def test_public_synthetic_examples_and_rollouts(tmp_path):
    create(tmp_path / "data", examples=True)
    dataset = Dataset(tmp_path / "data", synthetic=True)
    directory = Path(__file__).parents[1] / "data"
    examples = [json.loads(line) for line in (directory / "example.jsonl").read_text().splitlines()]
    rollouts = [json.loads(line) for line in (directory / "example_rollouts.jsonl").read_text().splitlines()]
    metrics = json.loads((directory / "example_metrics.json").read_text())
    assert metrics["jsonl_fpath"] == "resources_servers/agentic_if_realworld/data/example.jsonl"
    assert not Path(metrics["jsonl_fpath"]).is_absolute()
    assert examples == dataset.prepared()
    assert len({judge.sha(dataset.request(identifier)) for identifier in dataset.inputs}) == 5
    assert {row["task_id"] for row in rollouts} == set(EXAMPLE_LITERALS)
    assert len(rollouts) == 5
    verdicts = []
    for row in rollouts:
        identifier = row["task_id"]
        assert [
            {key: item[key] for key in ["role", "content"]} for item in row["responses_create_params"]["input"]
        ] == dataset.prepared_row(identifier)["responses_create_params"]["input"]
        text = row["response"]["output"][0]["content"][0]["text"]
        verdict = "pass" if text == EXAMPLE_LITERALS[identifier] else "fail"
        assert row["scoring_status"] == "scored" and row["item_verdict"] == verdict
        assert row["reward"] == float(verdict == "pass") and row["mask_sample"] is False
        assert row["judge_response_id"] and row["judge_usage"]["total_tokens"] > 0
        assert not inspect_record(row, scope=EvidenceScope(tools=False))["findings"]
        verdicts.append(verdict)
    assert verdicts.count("pass") == 3 and verdicts.count("fail") == 2
    _validate_data_single(
        GymTestConfig(entrypoint="resources_servers/agentic_if_realworld", should_validate_data=True)
    )


def test_default_synthetic_fixture_hash_is_unchanged(tmp_path):
    create(tmp_path)
    dataset = Dataset(tmp_path, synthetic=True)
    assert dataset.manifest_sha256 == "dc339a843cad5ba2ce95e2fae73fcd0529d8c9977bc272274403023fdd8a648e"
    response = {"status": "complete", "finish_reason": "stop", "message": {"role": "assistant", "content": "OK."}}
    payload = dataset.payload(
        "synthetic",
        response,
        {
            "id": "synthetic",
            "input_sha256": judge.sha(dataset.request("synthetic")),
            "response_sha256": judge.sha(response),
            "benchmark_manifest_sha256": dataset.manifest_sha256,
        },
    )
    assert judge.validate_review(review(payload), payload)["item_verdict"] == "pass"
