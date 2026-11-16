# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The committed example rows are the generator's output and are valid for the server."""

import base64
import io
import json
import wave
from pathlib import Path

import resources_servers.audio_eval.create_examples as examples
from resources_servers.audio_eval.profiles import load_profiles, source_hash
from resources_servers.audio_eval.task_data import TaskData


def _rows(path: Path) -> list[dict]:
    """Rows of ``path`` without ``task_source``, the key ``gym dataset collate`` adds to the committed example."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [{key: value for key, value in row.items() if key != "task_source"} for row in rows]


def test_committed_example_rows_match_the_generator() -> None:
    expected = [examples.make_row(index, text) for index, text in enumerate(examples.TRANSCRIPTS, 1)]
    assert _rows(examples.DATA_DIR / "example.jsonl") == expected


def test_example_rows_are_valid_task_data_for_the_example_profile() -> None:
    profiles = load_profiles(examples.DATA_DIR / "example_profiles.json")["profiles"]
    for line in (examples.DATA_DIR / "example.jsonl").read_text().splitlines():
        row = TaskData.model_validate(json.loads(line))
        assert row.benchmark_profile in profiles
        assert row.ng_source_row_sha256 == source_hash(row.ng_source_row)


def test_example_audio_is_a_short_silent_wav_data_uri() -> None:
    uri = examples.silent_wav_data_uri(duration_seconds=0.1, sample_rate=16000)
    prefix = "data:audio/wav;base64,"
    assert uri.startswith(prefix)
    with wave.open(io.BytesIO(base64.b64decode(uri[len(prefix) :]))) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, 16000, 1600)
        assert set(wav.readframes(wav.getnframes())) == {0}


def test_main_writes_the_example_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(examples, "DATA_DIR", tmp_path / "data")
    examples.main()
    assert _rows(tmp_path / "data" / "example.jsonl") == _rows(
        Path(examples.__file__).parent / "data" / "example.jsonl"
    )
