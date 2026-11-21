# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from benchmarks.aga_visual_tasks.prepare import native_row


ROOT = Path(__file__).resolve().parents[3]


def original(image_root: Path) -> dict:
    Image.new("RGBA", (2, 3), (11, 22, 33, 44)).save(image_root / "target.png")
    return {
        "source_task": "opencv_visual_question_answering",
        "expected_answer": "A",
        "question": "Question",
        "id": "original-id",
        "image_tools_base_agent_ref": {"type": "responses_api_agents", "name": "mcqa_simple_agent"},
        "responses_create_params": {
            "parallel_tool_calls": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Question /images/target.png"},
                        {"type": "input_image", "image_url": "/images/target.png", "detail": "auto"},
                    ],
                }
            ],
        },
    }


def test_rejects_wrong_training_subset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Agent-X"):
        native_row(original(tmp_path), "agent_x", tmp_path)


def test_image_path_note_preserves_public_mapping_and_original_content(tmp_path: Path) -> None:
    row = original(tmp_path)
    Image.new("RGB", (4, 2), (55, 66, 77)).save(tmp_path / "another.png")
    content = row["responses_create_params"]["input"][0]["content"]
    content.append({"type": "input_image", "image_url": "/images/another.png", "detail": "auto"})
    row["responses_create_params"]["input"].insert(0, {"role": "system", "content": "Original system instruction"})
    row["expected_answer"] = "PRIVATE_GOLD_MUST_NOT_REACH_POLICY"
    row["metadata"] = {"private_grader_detail": "PRIVATE_VERIFIER_MUST_NOT_REACH_POLICY"}
    before = copy.deepcopy(row)
    baseline = native_row(row, "code_assisted_vqa", tmp_path)
    prepared = native_row(row, "code_assisted_vqa", tmp_path, include_image_paths=True)
    assert row == before
    note = prepared["responses_create_params"]["input"][-1]["content"].pop()
    assert note == {
        "type": "input_text",
        "text": "Original input image files, in first appearance order:\n"
        "1. /images/target.png\n2. /images/another.png",
    }
    assert prepared == baseline
    assert prepared["source_images"] == {
        f"/images/{name}": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("target.png", "another.png")
    }
    assert list(prepared["source_images"]) == ["/images/target.png", "/images/another.png"]
    public_input = json.dumps(prepared["responses_create_params"]) + json.dumps(note)
    assert row["expected_answer"] not in public_input
    assert row["metadata"]["private_grader_detail"] not in public_input
