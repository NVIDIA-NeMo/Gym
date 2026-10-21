# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest

from benchmarks.opencode_multimodal_benchmarks.agentx_prepare import prepare
from benchmarks.opencode_multimodal_benchmarks.prepare import prepare as prepare_tsv


def test_portable_agentx_preserves_references_and_rejects_corrupt_media(tmp_path, monkeypatch, agentx_bundle):
    monkeypatch.chdir(tmp_path)
    source, media, image, _ = agentx_bundle
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = prepare(str(source), digest, limit=1, media_root=str(media))
    row = json.loads(output.read_text())
    assert row["verifier_metadata"]["expected_samples"] == 1
    assert row["responses_create_params"]["input"][0]["content"][1]["image_url"] == str(image)
    assert "visible color" not in json.dumps(row["responses_create_params"])
    image.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="image checksum"):
        prepare(str(source), digest, media_root=str(media))


def test_tsv_checksum_checked_before_materialization(tmp_path):
    source = tmp_path / "wrong.tsv"
    source.write_text("wrong dataset")
    with pytest.raises(ValueError, match="source dataset checksum mismatch"):
        prepare_tsv("vstar", source, sha256="0" * 64)
