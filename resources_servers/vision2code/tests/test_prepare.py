# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import base64
import hashlib
import json

import pytest
from PIL import Image

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.vision2code.prepare import prepare


def test_manifest(tmp_path):
    image = tmp_path / "reference.png"
    Image.new("RGB", (60, 80)).save(image)
    manifest = tmp_path / "manifest.jsonl"
    names = ["ChemVQA-2K", "DocVQA", "ChartQA", "EEE-Bench"]
    manifest.write_text("".join(json.dumps({"id": n, "rubric": n, "image_path": image.name}) + "\n" for n in names))
    output = tmp_path / "tasks.jsonl"
    assert prepare(manifest, output) == 4
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    for row in rows:
        metadata = row["verifier_metadata"]
        assert (metadata["width"], metadata["height"]) == (60, 80)
        assert metadata["sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
        assert metadata["image_path"] == str(image)
        params = row["responses_create_params"]
        NeMoGymResponseCreateParamsNonStreaming.model_validate(params)
        assert params["tools"] == []
        content = params["input"][1]["content"]
        assert base64.b64decode(content[0]["image_url"].split(",")[1]) == image.read_bytes()
        assert "60 pixels wide and 80 pixels tall" in content[1]["text"]
        assert "/workspace/output.png" in content[1]["text"]


@pytest.mark.parametrize("bad", ["unknown", "ChemVQA-2K-spacing-v1", "duplicate", "corrupt"])
def test_invalid_manifest_is_atomic(tmp_path, bad):
    image = tmp_path / "reference.png"
    Image.new("RGB", (60, 80)).save(image)
    item = {"id": "sample", "rubric": "ChartQA", "image_path": image.name}
    if bad == "corrupt":
        image.write_text("not an image")
    elif bad != "duplicate":
        item["rubric"] = bad
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text((json.dumps(item) + "\n") * (2 if bad == "duplicate" else 1))
    output = tmp_path / "tasks.jsonl"
    output.write_text("previous")
    with pytest.raises((ValueError, OSError)):
        prepare(manifest, output)
    assert output.read_text() == "previous"
