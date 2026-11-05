# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset

from benchmarks.vials.prepare import HF_CONFIG, HF_REPO, HF_REVISION, HF_SPLIT, prepare
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


def test_prepare_multimodal_inputs(tmp_path: Path) -> None:
    """All panels reach model input in order; reference answers remain verifier-only."""
    source_rows = []
    for task_id, payloads, lower, upper in (
        ("single", [b"first-image-bytes"], None, None),
        ("multi", [b"second-image-bytes", b"third-image-bytes"], None, None),
        ("bounded", [b"fourth-image-bytes"], 11.0, 13.0),
    ):
        source_rows.append(
            {
                "id": task_id,
                "question": "What is shown in A &amp; B?",
                "answer": "REFERENCE_ONLY_SENTINEL",
                "images": [{"bytes": payload, "path": None} for payload in payloads],
                "domain": "test",
                "metadata": {
                    "image_filenames": [f"panel_{index}.png" for index in range(len(payloads))],
                    "secondary_tags": [],
                    "answer_lower_bound": lower,
                    "answer_upper_bound": upper,
                },
            }
        )
    source = Dataset.from_list(source_rows)
    with patch("datasets.load_dataset", return_value=source) as load:
        output = prepare(output=tmp_path / "vials.jsonl")
        load.assert_called_once_with(HF_REPO, HF_CONFIG, split=HF_SPLIT, revision=HF_REVISION)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 3
    for raw, row in zip(source_rows, rows, strict=True):
        params = row["responses_create_params"]
        NeMoGymResponseCreateParamsNonStreaming.model_validate(params)
        assert "REFERENCE_ONLY_SENTINEL" not in json.dumps(params)
        assert row["expected_answer"] == "REFERENCE_ONLY_SENTINEL"
        blocks = params["input"][1]["content"]
        assert len(blocks) == len(raw["images"]) + 1
        assert blocks[-1]["type"] == "input_text"
        assert blocks[-1]["text"].startswith("What is shown in A & B?\n\n")
        assert "ANSWER: <final answer>" in blocks[-1]["text"]
        for block, image in zip(blocks[:-1], raw["images"], strict=True):
            assert block["type"] == "input_image"
            assert block["detail"] == "auto"
            assert block["image_url"].startswith("data:image/png;base64,")
            assert base64.b64decode(block["image_url"].split(",", 1)[1], validate=True) == image["bytes"]
        assert [Path(path).read_bytes() for path in row["image_paths"]] == [image["bytes"] for image in raw["images"]]
    assert rows[2]["answer_lower_bound"] == 11.0
    assert rows[2]["answer_upper_bound"] == 13.0
    examples = [json.loads(line) for line in (tmp_path / "example.jsonl").read_text().splitlines()]
    assert examples == rows
