# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify preset prompt preparation without inference or Slurm."""

import json
import os
from pathlib import Path

import pytest
from PIL import Image

from resources_servers.vision2code import prepare


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "resources_servers/vision2code/configs"


@pytest.mark.parametrize("preset,dimensions", [("vision2code_evals", False), ("vision2code_feedback", True)])
def test_preset_prompt_options_need_no_credentials(monkeypatch, preset, dimensions):
    for name in tuple(os.environ):
        if name.startswith("VISION2CODE_"):
            monkeypatch.delenv(name)
    options = prepare.load_preparation_config(CONFIGS / f"{preset}.yaml")
    assert options.include_canvas_dimensions is dimensions
    assert not options.include_rendering_instructions and not options.include_library_instructions


@pytest.fixture
def example(tmp_path):
    image = tmp_path / "reference.png"
    Image.new("RGB", (80, 60), "white").save(image)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "sample", "image_path": str(image), "rubric": "ChartQA"}))
    return manifest


def test_cli_flags_override_preset_without_removing_existing_additions(example, tmp_path, monkeypatch):
    output = tmp_path / "tasks.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare",
            str(example),
            str(output),
            "--config",
            str(CONFIGS / "vision2code_feedback.yaml"),
            "--no-include-canvas-dimensions",
            "--include-library-instructions",
            "--include-rendering-instructions",
        ],
    )
    prepare.main()
    prompt = json.loads(output.read_text())["responses_create_params"]["input"][1]["content"][1]["text"]
    assert "Target canvas:" not in prompt
    assert "80 pixels wide and 60 pixels tall" in prompt
    assert "Available libraries:" in prompt and "ReportLab" in prompt
