# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import base64
import hashlib
import json
from typing import get_args

import pytest
from PIL import Image
from pydantic import TypeAdapter

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.data import validate_jsonl_rows
from resources_servers.vision2code.evaluation import RubricName
from resources_servers.vision2code.prepare import prepare
from resources_servers.vision2code.task_data import TaskData


def test_manifest(tmp_path):
    image = tmp_path / "reference.png"
    Image.new("RGB", (60, 80)).save(image)
    manifest = tmp_path / "manifest.jsonl"
    names = get_args(RubricName)
    manifest.write_text("".join(json.dumps({"id": n, "rubric": n, "image_path": image.name}) + "\n" for n in names))
    output = tmp_path / "tasks.jsonl"
    assert prepare(manifest, output) == 19
    report = validate_jsonl_rows("vision2code", TypeAdapter(TaskData), str(output), output.read_text().splitlines())
    assert report.clean, report.summary()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert {row["verifier_metadata"]["rubric"] for row in rows} == set(names)
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
        assert content[1]["text"] == "Generate Python code to recreate this image. Return code only."


@pytest.mark.parametrize("bad", ["unknown", "strict-category", "geometry", "graph_algorithms", "duplicate", "corrupt"])
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


@pytest.mark.parametrize("rendering,libraries", [(False, True), (True, False), (True, True)])
def test_optional_task_instructions(tmp_path, rendering, libraries):
    Image.new("RGB", (60, 80)).save(tmp_path / "reference.png")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "x", "rubric": "DocVQA-v2", "image_path": "reference.png"}))
    output = tmp_path / "tasks.jsonl"
    prepare(manifest, output, include_rendering_instructions=rendering, include_library_instructions=libraries)
    row = json.loads(output.read_text())
    text = row["responses_create_params"]["input"][1]["content"][1]["text"]
    assert ("60 pixels wide and 80 pixels tall" in text) == rendering
    assert ("/workspace/output.png" in text) == rendering
    assert ("no network" in text) == rendering
    assert ("RDKit" in text and "ReportLab" in text) == libraries
    assert row["verifier_metadata"]["rubric"] == "DocVQA-v2"


@pytest.mark.parametrize("rendering,libraries", [(False, False), (True, False), (False, True), (True, True)])
def test_canvas_dimensions_preserve_optional_instructions(tmp_path, rendering, libraries):
    Image.new("RGB", (60, 80)).save(tmp_path / "reference.png")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "x", "rubric": "generic", "image_path": "reference.png"}))
    output = tmp_path / "tasks.jsonl"
    prepare(
        manifest,
        output,
        include_canvas_dimensions=True,
        include_rendering_instructions=rendering,
        include_library_instructions=libraries,
    )
    text = json.loads(output.read_text())["responses_create_params"]["input"][1]["content"][1]["text"]
    base = "Generate Python code to recreate this image. Return code only. Target canvas: 60 x 80 pixels."
    assert text.startswith(base)
    assert ("/workspace/output.png" in text) == rendering
    assert ("Available libraries:" in text) == libraries
    if not rendering and not libraries:
        assert text == base
