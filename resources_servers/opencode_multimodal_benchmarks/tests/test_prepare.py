# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib
import json

import pytest

from benchmarks.opencode_multimodal_benchmarks.prepare import task_row
from responses_api_agents.opencode_multimodal_agent.app import OpenCodeMultimodalRunRequest


def test_prepared_image_request_validates_against_native_agent():
    row = task_row({"prompt": "What color?", "images": ["/tmp/image.png"], "answer": "A"})
    parsed = OpenCodeMultimodalRunRequest.model_validate(row)
    content = parsed.responses_create_params.model_dump()["input"][0]["content"]
    assert content[1]["image_url"] == "/tmp/image.png"
    assert content[1]["detail"] == "auto"
    assert parsed.verifier_metadata["answer"] == "A"
    assert "answer" not in parsed.responses_create_params.model_dump()


@pytest.mark.parametrize("limit,count", [(None, 3), ("None", 3), ("null", 3), (2, 2), ("2", 2)])
def test_native_launcher_limit_preserves_dataset(monkeypatch, tmp_path, limit, count):
    module = importlib.import_module("benchmarks.opencode_multimodal_benchmarks.prepare")
    monkeypatch.chdir(tmp_path)
    tasks = [{"index": str(i), "prompt": "Question", "images": [], "answer": "A"} for i in range(3)]
    monkeypatch.setattr(module, "prepare_tasks", lambda *args, **kwargs: (tasks, {}))
    monkeypatch.setattr(module, "validate_mcq_tasks", lambda *args: None)
    output = module.prepare("vstar", "unused.tsv", limit=limit)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == count
    assert [r["verifier_metadata"]["index"] for r in rows] == [str(i) for i in range(count)]
    assert all(r["verifier_metadata"]["expected_samples"] == count for r in rows)
    assert all("answer" not in r["responses_create_params"] for r in rows)


@pytest.mark.parametrize("limit", [0, -1, "0", "invalid", 1.5, True])
def test_invalid_limit_fails_before_materializing_images(monkeypatch, limit):
    module = importlib.import_module("benchmarks.opencode_multimodal_benchmarks.prepare")

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("Invalid limit must be rejected before loading the dataset")

    monkeypatch.setattr(module, "prepare_tasks", unexpected_preparation)
    with pytest.raises(ValueError):
        module.prepare("vstar", "unused.tsv", limit=limit)
