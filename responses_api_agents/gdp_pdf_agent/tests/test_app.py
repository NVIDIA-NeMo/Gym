# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from responses_api_agents.gdp_pdf_agent.app import (
    GdpPdfAgentConfig,
    _resolve_under,
    materialize_document,
)
from responses_api_agents.simple_agent.app import SimpleAgentRunRequest


def _config(**overrides) -> GdpPdfAgentConfig:
    values = {
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "",
        "name": "gdp_pdf_agent",
        "resources_server": ResourcesServerRef(type="resources_servers", name="gdp_pdf"),
        "model_server": ModelServerRef(type="responses_api_models", name="policy"),
        "documents_base_dir": ".",
        "source_dpi": 150,
        "image_dpi": 75,
        "pages_per_image": 2,
        "max_images": 1,
    }
    values.update(overrides)
    return GdpPdfAgentConfig(**values)


def _write_document(root: Path) -> None:
    pages_dir = root / "document" / "pages"
    pages_dir.mkdir(parents=True)
    for page_number, color in ((1, "red"), (2, "blue"), (3, "green")):
        Image.new("RGB", (100, 120), color).save(pages_dir / f"page_{page_number:04d}.png")
    manifest = {
        "source_dpi": 150,
        "page_count": 3,
        "pages": [
            {"page_number": page_number, "text": f"text {page_number}", "image": f"pages/page_{page_number:04d}.png"}
            for page_number in (1, 2, 3)
        ],
    }
    (root / "document" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_materializes_full_text_and_configured_image_fallback(tmp_path: Path) -> None:
    _write_document(tmp_path)
    row = {
        "responses_create_params": {"input": "seed"},
        "verifier_metadata": {
            "task_prompt": "Do the analysis.",
            "document_manifest": "document/manifest.json",
        },
    }

    materialized = materialize_document(row, tmp_path, _config())
    SimpleAgentRunRequest.model_validate(materialized)

    params = materialized["responses_create_params"]
    assert params["tools"] == []
    assert params["parallel_tool_calls"] is False
    content = params["input"][0]["content"]
    image_blocks = [block for block in content if block["type"] == "input_image"]
    assert len(image_blocks) == 1
    assert "Images cover only pages 1-2" in content[0]["text"]
    assert all(f"<page {number}>\ntext {number}" in content[-1]["text"] for number in (1, 2, 3))

    encoded = image_blocks[0]["image_url"].split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        assert image.size == (100, 84)


def test_text_only_profile_still_includes_every_page(tmp_path: Path) -> None:
    _write_document(tmp_path)
    for image_path in (tmp_path / "document" / "pages").iterdir():
        image_path.unlink()
    row = {
        "responses_create_params": {"input": "seed"},
        "verifier_metadata": {
            "task_prompt": "Do the analysis.",
            "document_manifest": "document/manifest.json",
        },
    }

    materialized = materialize_document(row, tmp_path, _config(include_page_images=False))
    content = materialized["responses_create_params"]["input"][0]["content"]

    assert not any(block["type"] == "input_image" for block in content)
    assert "complete extracted text" in content[0]["text"]
    assert "text 3" in content[-1]["text"]


def test_document_paths_cannot_escape_base(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        _resolve_under(tmp_path, "../secret.pdf")


def test_rejects_upscaling() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        _config(source_dpi=100, image_dpi=120)


def test_rejects_mismatched_source_dpi(tmp_path: Path) -> None:
    _write_document(tmp_path)
    row = {
        "responses_create_params": {"input": "seed"},
        "verifier_metadata": {
            "task_prompt": "Do the analysis.",
            "document_manifest": "document/manifest.json",
        },
    }

    with pytest.raises(ValueError, match="prepared at 150 DPI"):
        materialize_document(row, tmp_path, _config(source_dpi=100, image_dpi=100))
