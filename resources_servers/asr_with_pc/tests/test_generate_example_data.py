# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The example-data generator writes one valid self-contained row per sample transcript."""

import json
import sys
from pathlib import Path

from resources_servers.asr_with_pc import generate_example_data as gen


def test_main_writes_one_example_row_per_transcript(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "nested" / "example.jsonl"
    monkeypatch.setattr(sys, "argv", ["generate_example_data", "--out", str(out)])

    gen.main()

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [row["expected_answer"] for row in rows] == gen.SAMPLE_TRANSCRIPTS
    assert [row["sample_id"] for row in rows] == [f"example-{i:02d}" for i in range(len(rows))]
    assert all(row["split"] == "example" for row in rows)
    assert all(
        row["responses_create_params"]["metadata"]["audio_data"].startswith("data:audio/wav;base64,") for row in rows
    )
