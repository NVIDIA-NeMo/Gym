# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
import json
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from benchmarks.opencode_multimodal_benchmarks import zerobench_prepare
from resources_servers.opencode_multimodal_benchmarks.app import VisualBenchmarksServer
from resources_servers.opencode_multimodal_benchmarks.tests.test_app import request, server  # noqa: F401
from resources_servers.opencode_multimodal_benchmarks.zerobench import ANSWER_INSTRUCTION, score


@pytest.mark.parametrize(
    "prediction,answer,expected",
    [
        ("The answer is {Cat}.", "cat", ("cat", True)),
        ("{cat} then {dog}", "cat", ("dog", False)),
        ("cat", "cat", ("", False)),
        ("{ cat }", "cat", (" cat ", False)),
        ("{cat.}", "cat", ("cat.", False)),
        ("{cat\ndog}", "cat\ndog", ("", False)),
        ("{1.0}", "1", ("1.0", False)),
        (r"\boxed{42}", "42", ("42", True)),
    ],
)
def test_public_scoring(prediction, answer, expected):
    assert score(prediction, answer) == expected


@pytest.mark.asyncio
async def test_verify_and_complete_coverage(server, monkeypatch):
    judge = AsyncMock(side_effect=AssertionError("ZeroBench must not call a judge"))
    monkeypatch.setattr(VisualBenchmarksServer, "_judge", judge)
    result = await server.verify(request("ZeroBench", "<think>{wrong}</think>{cat}", "cat"))
    assert result.reward == 1
    assert result.scorer == "zerobench-public-exact-v1"
    assert result.extracted == "cat"
    first = result.model_dump()
    first["verifier_metadata"].update(expected_samples=2)
    metrics = server.compute_metrics([[first]])
    assert "accuracy" not in metrics and metrics["ZeroBench/complete"] is False
    second = (await server.verify(request("ZeroBench", "", "cat"))).model_dump()
    second["verifier_metadata"].update(expected_samples=2, index="2")
    assert server.compute_metrics([[first], [second]])["accuracy"] == 0.5
    judge.assert_not_awaited()


def test_prepare_images_and_reference_isolation(tmp_path, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    image = io.BytesIO()
    Image.new("RGB", (4, 3), "red").save(image, format="PNG")
    raw = image.getvalue()
    rows = [
        {
            "question_id": str(i),
            "question_text": "Which color?",
            "question_answer": "secret-reference",
            "question_images_decoded": [{"bytes": raw, "path": "original.png"}],
        }
        for i in range(1, 101)
    ]

    class Table:
        def to_pylist(self):
            return rows

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(zerobench_prepare, "sha256_file", lambda _: zerobench_prepare.SOURCE_SHA256)
    monkeypatch.setattr(pq, "read_table", lambda _: Table())
    out = zerobench_prepare.prepare("source.parquet", limit=2)
    prepared = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(prepared) == 2
    assert prepared[0]["verifier_metadata"]["answer"] == "secret-reference"
    policy = prepared[0]["responses_create_params"]
    assert "secret-reference" not in json.dumps(policy)
    content = policy["input"][0]["content"]
    assert content[0]["text"] == "Which color?" + ANSWER_INSTRUCTION
    assert __import__("pathlib").Path(content[1]["image_url"]).read_bytes() == raw
    rows.pop()
    with pytest.raises(ValueError, match="question ID"):
        zerobench_prepare.prepare("source.parquet")


def test_reject_changed_source(tmp_path):
    pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "modified.parquet"
    source.write_bytes(b"not the canonical release")
    with pytest.raises(ValueError, match="checksum"):
        zerobench_prepare.prepare(str(source))
