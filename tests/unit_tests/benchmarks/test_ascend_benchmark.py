# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the ASCEND benchmark: declaration, discovery, row building, audio writing, prepare()."""

import json
import wave
from pathlib import Path

import pytest

from benchmarks import audio_manifest
from benchmarks.ascend import prepare as ascend
from nemo_gym.environment.benchmarks import BenchmarkConfig
from nemo_gym.environment.manifest import load_manifest
from tests.unit_tests.benchmarks.audio_benchmark_helpers import assert_capped_and_remapped_rows, sine_wav_bytes


USES_AUDIO_LIBRARIES = True  # prepare() writes WAV files with soundfile, which the core environment lacks

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "ascend"


def _row(sample_id: str, *, duration: float = 0.5, transcription: str = "你好 hello") -> dict:
    return {
        "id": sample_id,
        "path": f"waves/{sample_id}.wav",
        "audio": {"bytes": sine_wav_bytes(), "path": f"waves/{sample_id}.wav"},
        "transcription": transcription,
        "duration": duration,
        "language": "mixed",
        "original_speaker_id": 3,
        "session_id": 2,
        "topic": "sports",
    }


def _write_parquet(path: Path, rows: list[dict]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    return path


def test_declaration_specifics() -> None:
    dataset = BenchmarkConfig.from_config_path(BENCHMARK / "config.yaml").dataset
    manifest = load_manifest(BENCHMARK / "manifest.yaml")
    assert "soundfile" in dataset.prepare_dependencies
    assert manifest.canonical_split == "test" and manifest.prompt_source == "prepared" and manifest.experimental
    assert manifest.licensing == "CC-BY-SA-4.0" and manifest.domain == "asr"


def test_profile_contract_values() -> None:
    contract = audio_manifest.load_profiles(ascend.PROFILES)["profiles"]["ascend"]
    assert audio_manifest.request_controls(contract)["max_output_tokens"] == 256
    assert "++eval_type=ascend" in contract["skills"]["EVAL_ARGS"]
    assert "++eval_config.cleanup_generation_loops=true" in contract["skills"]["EVAL_ARGS"].split()
    assert contract["skills"]["METRICS_TYPE"] == "ascend"


def test_format_source_row_keeps_the_released_fields() -> None:
    source = ascend.format_source_row(_row("a/b:7", duration=1.25), 4, "/mnt/audio")
    assert source["id"] == "a_b_7"
    assert source["audio_path"] == "/mnt/audio/a_b_7.wav"
    assert source["expected_answer"] == "你好 hello"
    assert source["duration"] == 1.25 and source["task_type"] == "Code-Switched-ASR"
    assert source["subset_for_metrics"] == "test"
    assert source["extra_fields"] == {
        "language": "mixed",
        "speaker_id": 3,
        "session_id": 2,
        "topic": "sports",
        "source_repo": "CAiRE/ASCEND",
        "source_revision": ascend.HF_REVISION,
        "source_split": "test",
    }
    assert source["messages"] == [
        {"role": "user", "content": ascend.PROMPT, "audio": {"path": "/mnt/audio/a_b_7.wav", "duration": 1.25}}
    ]

    missing_id = _row("x") | {"id": None}
    assert ascend.format_source_row(missing_id, 4, "/mnt")["id"] == "ascend-test-4"
    with pytest.raises(ValueError, match="no duration"):
        ascend.format_source_row(_row("z", duration=0.0), 0, "/mnt")


def test_request_row_replaces_the_prompt_and_keeps_the_source() -> None:
    contract = audio_manifest.load_profiles(ascend.PROFILES)["profiles"]["ascend"]
    source = ascend.format_source_row(_row("0"), 0, "/mnt/audio")
    row = audio_manifest.build_row(source, profile="ascend", contract=contract, index=1)
    request = row["responses_create_params"]
    assert [m["role"] for m in request["input"]] == ["user"]
    assert request["input"][0]["content"].startswith("Transcribe the audio word for word")
    assert request["metadata"]["audio_path"] == "/mnt/audio/0.wav"
    assert row["sample_id"] == "ascend:1" and row["ng_source_row"] == source
    assert row["ng_source_row_sha256"] == audio_manifest.source_hash(source)


def test_read_rows_checks_the_census_and_caps(tmp_path: Path, monkeypatch) -> None:
    path = _write_parquet(tmp_path / "test.parquet", [_row(f"{i:05d}") for i in range(5)])
    monkeypatch.setattr(ascend, "EXPECTED_ROWS", 5)
    assert [row["id"] for row in ascend.read_rows(path)] == [f"{i:05d}" for i in range(5)]
    assert [row["id"] for row in ascend.read_rows(path, 3)] == ["00000", "00001", "00002"]
    monkeypatch.setattr(ascend, "EXPECTED_ROWS", 6)
    with pytest.raises(ValueError, match="Expected 6 ASCEND test rows, found 5"):
        list(ascend.read_rows(path))


def test_download_authenticates_the_parquet(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "test.parquet"
    path.write_bytes(b"released bytes")
    monkeypatch.setattr(ascend, "hf_hub_download", lambda *args, **kwargs: str(path))
    with pytest.raises(ValueError, match="size mismatch"):
        ascend.download_test_parquet()
    monkeypatch.setattr(ascend, "TEST_PARQUET_BYTES", path.stat().st_size)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ascend.download_test_parquet()
    monkeypatch.setattr(ascend, "TEST_PARQUET_SHA256", audio_manifest.sha256_file(path))
    assert ascend.download_test_parquet() == path


def test_write_audio_averages_channels_and_check_audio_rejects_bad_files(tmp_path: Path) -> None:
    target = tmp_path / "clip.wav"
    ascend.write_audio({"bytes": sine_wav_bytes(8000, 0.25, channels=2)}, target)
    with wave.open(str(target), "rb") as stream:
        assert (stream.getnchannels(), stream.getframerate(), stream.getnframes()) == (1, 8000, 2000)
        # The two channels are a tone and its inverse, so their average is silence.
        assert set(stream.readframes(2000)) == {0}
    assert not list(tmp_path.glob("*.tmp"))
    ascend.check_audio(target)

    with pytest.raises(ValueError, match="missing or not a WAV"):
        ascend.check_audio(tmp_path / "nope.wav")
    with wave.open(str(tmp_path / "empty.wav"), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
    with pytest.raises(ValueError, match="empty"):
        ascend.check_audio(tmp_path / "empty.wav")


@pytest.fixture
def offline_prepare(tmp_path: Path, monkeypatch) -> Path:
    """Run prepare() against a synthetic parquet, writing under tmp_path."""
    parquet = _write_parquet(tmp_path / "test.parquet", [_row(f"{i:05d}") for i in range(3)])
    monkeypatch.setattr(ascend, "download_test_parquet", lambda: parquet)
    monkeypatch.setattr(ascend, "EXPECTED_ROWS", 3)
    data_dir = tmp_path / "data"
    monkeypatch.setattr(ascend, "DATA_DIR", data_dir)
    monkeypatch.setattr(ascend, "OUTPUT_FPATH", data_dir / "ascend_benchmark.jsonl")
    return data_dir


def test_prepare_writes_the_benchmark_jsonl_with_existing_audio(offline_prepare: Path) -> None:
    output = ascend.prepare()
    assert output == offline_prepare / "ascend_benchmark.jsonl"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == ["ascend:1", "ascend:2", "ascend:3"]
    assert {row["benchmark_profile"] for row in rows} == {"ascend"}

    first = rows[0]
    audio_path = Path(first["responses_create_params"]["metadata"]["audio_path"])
    assert audio_path == (offline_prepare / "audio" / "00000.wav").resolve() and audio_path.is_file()
    assert first["ng_source_row"]["messages"][0]["audio"]["path"] == str(audio_path)
    assert first["ng_source_row_sha256"] == audio_manifest.source_hash(first["ng_source_row"])
    assert first["task_type"] == "Code-Switched-ASR" and first["expected_answer"] == "你好 hello"

    assert ascend.is_prepared_data_current(output)
    assert not ascend.is_prepared_data_current(offline_prepare / "other.jsonl")


def test_prepare_audio_root_and_row_cap_override(offline_prepare: Path) -> None:
    output = ascend.prepare(audio_root="/data/ascend", max_samples_per_profile=2)
    assert_capped_and_remapped_rows(ascend, output, rows=2, audio_path="/data/ascend/audio/00000.wav")
    assert sorted(p.name for p in (offline_prepare / "audio").iterdir()) == ["00000.wav", "00001.wav"]


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "3"])
def test_prepare_rejects_an_invalid_row_cap_before_any_download(bad, monkeypatch) -> None:
    def fail():
        raise AssertionError("the download must not start")

    monkeypatch.setattr(ascend, "download_test_parquet", fail)
    with pytest.raises(ValueError, match="positive integer"):
        ascend.prepare(max_samples_per_profile=bad)
