# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from benchmarks.agentic_mme import prepare as prepare_module


def _release(root: Path, count: int) -> None:
    (root / "json").mkdir(parents=True)
    for index in range(count):
        (root / "json" / f"{index:03d}.json").write_text(json.dumps({"task_id": str(index)}))


def test_failed_conversion_leaves_no_prepared_file(tmp_path, monkeypatch) -> None:
    # An existing benchmark JSONL counts as prepared, so a crash must never leave a partial one behind.
    _release(tmp_path / "release", prepare_module.EXPECTED_TASKS)
    output = tmp_path / "data" / "agentic_mme_benchmark.jsonl"
    calls = []

    def convert(task: dict, root: Path) -> dict:
        calls.append(task["task_id"])
        if len(calls) == 10:
            raise FileNotFoundError("missing image")
        return {"task": task["task_id"]}

    monkeypatch.setattr(prepare_module, "dataset_root", lambda: tmp_path / "release")
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", output)
    monkeypatch.setattr(prepare_module, "convert", convert)
    with pytest.raises(FileNotFoundError):
        prepare_module.prepare()
    assert not output.exists()

    calls.clear()
    monkeypatch.setattr(prepare_module, "convert", lambda task, root: {"task": task["task_id"]})
    assert prepare_module.prepare() == output
    assert len(output.read_text().splitlines()) == prepare_module.EXPECTED_TASKS


def test_incomplete_release_is_refused(tmp_path, monkeypatch) -> None:
    _release(tmp_path / "release", 3)
    monkeypatch.setattr(prepare_module, "dataset_root", lambda: tmp_path / "release")
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", tmp_path / "data" / "agentic_mme_benchmark.jsonl")
    with pytest.raises(ValueError, match="expected 418 task files"):
        prepare_module.prepare()


def test_convert_keeps_images_in_order_and_gold_out_of_the_prompt(tmp_path) -> None:
    from PIL import Image

    (tmp_path / "images").mkdir()
    Image.new("RGB", (8, 4), "red").save(tmp_path / "images" / "a.png")
    # The release names some files .png that ship as .jpg; resolve_image must find them.
    Image.new("RGB", (4, 8), "blue").save(tmp_path / "images" / "b.jpg")
    task = {
        "task_id": "L2-0007",
        "meta": {"level": 2, "domain": "finance"},
        "input": {"image": ["images/a.png", "images/b.png"], "prompt": "How much was it sold for?"},
        "golden_answer": {"value": "44.6 million"},
        "process_evaluation": {
            "checkpoints": [{"answer_check": {"match_type": "exact_match", "value": "secret-checkpoint-value"}}],
            "efficiency": {"reference_tool_calls": 2},
        },
    }
    row = prepare_module.convert(task, tmp_path)
    system, user = row["responses_create_params"]["input"]
    assert system["role"] == "system" and user["role"] == "user"
    parts = user["content"]
    assert [p["type"] for p in parts] == ["input_text", "input_image", "input_text", "input_image", "input_text"]
    assert [parts[0]["text"], parts[2]["text"], parts[4]["text"]] == [
        "Image 0.",
        "Image 1.",
        "How much was it sold for?",
    ]
    assert parts[1]["image_url"].startswith("data:image/png;base64,")
    assert parts[3]["image_url"].startswith("data:image/jpeg;base64,")
    meta = row["verifier_metadata"]
    assert meta["task_id"] == "L2-0007" and meta["level"] == 2 and meta["domain"] == "finance"
    assert meta["golden_answer"] == {"value": "44.6 million"}
    assert meta["process_evaluation"]["efficiency"]["reference_tool_calls"] == 2
    prompt = json.dumps(row["responses_create_params"])
    assert "44.6 million" not in prompt and "secret-checkpoint-value" not in prompt
