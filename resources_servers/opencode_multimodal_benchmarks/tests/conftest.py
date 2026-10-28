# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest
from PIL import Image

from benchmarks.opencode_multimodal_benchmarks import agentx_prepare
from benchmarks.opencode_multimodal_benchmarks.agentx_source import reference_identity
from benchmarks.opencode_multimodal_benchmarks.canonical import fingerprint


@pytest.fixture
def agentx_bundle(tmp_path, monkeypatch):
    """Synthetic full bundle; mutations must fail even with an updated caller checksum."""
    media = tmp_path / "media"
    media.mkdir()
    image = media / "original.png"
    Image.new("RGB", (2, 2), "red").save(image)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    target = image.with_name(digest + ".png")
    image.rename(target)
    canonical = agentx_prepare.manifest("AgentX")
    rows = []
    for identifier in canonical["tasks"]:
        task = {
            "index": identifier,
            "benchmark": "AgentX_GoalAcc",
            "split": "goal_acc_non_generative",
            "question": "What color?",
            "answer": "red",
            "GT_json": '{"value": "red", "justification": "visible color"}',
            "query_type": "objective",
            "query_type_raw": "Objective",
            "modality": "Image",
            "source_media_sha256": {"original.png": digest},
        }
        canonical["tasks"][identifier] = {
            "reference_sha256": fingerprint(reference_identity(task)),
            "modality": "Image",
            "media": [["original.png", digest]],
        }
        rows.append(
            {
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "What color?"},
                                {"type": "input_image", "image_url": target.name},
                            ],
                        }
                    ]
                },
                "verifier_metadata": task,
            }
        )
    monkeypatch.setattr(agentx_prepare, "manifest", lambda _: canonical)
    source = tmp_path / "agentx.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return source, media, target, rows
