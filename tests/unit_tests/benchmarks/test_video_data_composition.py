# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose canonical task rows without translating their instruction or media fields."""

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest
from examples.configs.video_grpo import prepare_three_source_data as module

from nemo_gym.data.tasks.harbor import load_task, materialize_task
from tests.unit_tests.data.tasks.harbor.test_harbor_tasks import HELLO_TOML, write_task


def _inputs(tmp_path: Path, *, flat_avb: bool = False) -> dict[str, object]:
    frames = tmp_path / "frames/video"
    frames.mkdir(parents=True)
    for ordinal in [2, 1]:
        (frames / f"frame-{ordinal}-of-2-2fps.jpg").write_bytes(b"fixture frame; no decoding")
    prompt = {
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "images/frame.png",
                        "_is_video_frame": True,
                        "_video_frame_index": 0,
                        "_video_fps": 1.0,
                    },
                    {"type": "input_text", "text": "Keep the literal instruction."},
                ],
            }
        ],
        "max_output_tokens": 8192,
    }
    avb = {
        "task_source": "agentic_vbench_train",
        "task_id": {"taskset": "avb-opencode", "task_id": "literal-task"},
        "task_input": {
            "responses_create_params": prompt,
            "task_data": {"_ng_digest": "unchanged", "ground_truth": "keep"},
        },
    }
    if flat_avb:
        avb = {"task_source": "agentic_vbench_agent", "responses_create_params": prompt}
    claw = {
        "task_source": "claweval_agent",
        "responses_create_params": {"input": "Claw instruction"},
        "verifier_metadata": {"private_fixture": "unchanged"},
    }
    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(b"".join((json.dumps(row, indent=None) + "\n").encode() for row in [avb, claw]))
    output = io.StringIO()
    fields = [
        "index",
        "video",
        "video_path",
        "question",
        "candidates",
        "answer",
        "duration",
        "domain",
        "sub_category",
        "task_type",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerow(
        {
            "index": "1",
            "video": "video",
            "video_path": "./video/video.mp4",
            "question": "Choose.",
            "candidates": "['first', 'second']",
            "answer": "B",
            "duration": "short",
            "domain": "fixture",
            "sub_category": "fixture",
            "task_type": "fixture",
        }
    )
    annotations = tmp_path / "video.tsv"
    annotations.write_text(output.getvalue())
    return {
        "agentic_jsonl": rows,
        "expected_agentic_sha256": hashlib.sha256(rows.read_bytes()).hexdigest(),
        "videomme_tsv": annotations,
        "expected_videomme_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
        "frame_root": frames.parent,
        "output_root": tmp_path / "out",
    }


def test_stock_task_rows_and_frame_flags_remain_byte_exact(tmp_path: Path) -> None:
    options = _inputs(tmp_path)
    source = options["agentic_jsonl"]
    assert isinstance(source, Path)
    before = source.read_bytes().splitlines(keepends=True)
    manifest = module.prepare(**options)
    output = options["output_root"]
    assert isinstance(output, Path)
    combined = (output / "three_source_train.jsonl").read_bytes().splitlines(keepends=True)
    assert [line for line in combined if json.loads(line)["task_source"] != module.VIDEO_AGENT] == before
    assert manifest["rows"] == 3
    assert manifest["task_source_counts"] == {"agentic_vbench_train": 1, "claweval_agent": 1, module.VIDEO_AGENT: 1}
    video = json.loads((output / "videomme_train.jsonl").read_text())
    content = video["responses_create_params"]["input"][0]["content"]
    assert [Path(item["image_url"].removeprefix("file://")).name for item in content[:-1]] == [
        "frame-1-of-2-2fps.jpg",
        "frame-2-of-2-2fps.jpg",
    ]
    assert video["answer"] == "B" and video["responses_create_params"]["max_output_tokens"] == 8192


def test_flat_avb_is_not_translated_or_written(tmp_path: Path) -> None:
    options = _inputs(tmp_path, flat_avb=True)
    with pytest.raises(ValueError, match="stock materialized Harbor rows"):
        module.prepare(**options)
    output = options["output_root"]
    assert isinstance(output, Path)
    assert not output.exists()


def test_stock_single_step_turns_are_preserved_without_a_top_level_copy(tmp_path: Path) -> None:
    options = _inputs(tmp_path)
    folder = write_task(tmp_path / "stock-task", toml=HELLO_TOML + '\n[[steps]]\nname = "solve"\n')
    step = folder / "steps/solve"
    (step / "tests").mkdir(parents=True)
    (step / "instruction.md").write_text("Keep this stock step instruction.\n")
    (step / "tests/test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    task = load_task(folder)
    stock = materialize_task(task, "avb-opencode").model_dump(mode="json", exclude_unset=True)
    assert "responses_create_params" not in stock["task_input"]
    assert [turn["name"] for turn in stock["task_input"]["turns"]] == ["solve"]
    source = options["agentic_jsonl"]
    assert isinstance(source, Path)
    original = source.read_bytes().splitlines(keepends=True)
    source.write_bytes((json.dumps(stock, separators=(",", ":")) + "\n").encode() + original[1])
    options["expected_agentic_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = source.read_bytes().splitlines(keepends=True)

    module.prepare(**options)

    output = options["output_root"]
    assert isinstance(output, Path)
    combined = (output / "three_source_train.jsonl").read_bytes().splitlines(keepends=True)
    assert [line for line in combined if json.loads(line).get("task_source") != module.VIDEO_AGENT] == expected
