# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the HALAS benchmark: declaration, discovery, row building, audio extraction, prepare()."""

import csv
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from benchmarks import audio_manifest
from benchmarks.halas import prepare as halas
from nemo_gym.environment.benchmarks import BenchmarkConfig
from nemo_gym.environment.manifest import load_manifest
from resources_servers.audio_eval.score import evaluator_settings, resolve
from tests.unit_tests.benchmarks.audio_benchmark_helpers import assert_capped_and_remapped_rows, sine_wav_bytes


USES_AUDIO_LIBRARIES = True
ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "halas"

LABEL_COLUMNS = ("sys_a_label", "sys_b_label", "sys_c_label")


def _csv_row(audio_id: str, *, corrected: str = "", original: str = "Hello, world.", labels=("", "", "")) -> dict:
    row = {
        "audio_id": audio_id,
        "audio_duration": "1.5",
        "e22_reference_text": original,
        "corrected_reference_text": corrected,
        "split": "test",
    }
    row.update(dict(zip(LABEL_COLUMNS, labels)))
    return row


def _write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_declaration_specifics() -> None:
    dataset = BenchmarkConfig.from_config_path(BENCHMARK / "config.yaml").dataset
    manifest = load_manifest(BENCHMARK / "manifest.yaml")
    assert "soundfile" in dataset.prepare_dependencies
    assert manifest.canonical_split == "test" and manifest.prompt_source == "prepared" and manifest.experimental


def test_profile_contract_values() -> None:
    contract = audio_manifest.load_profiles(halas.PROFILES)["profiles"]["halas"]
    assert audio_manifest.request_controls(contract)["max_output_tokens"] == 256
    assert "++eval_type=halas" in contract["skills"]["EVAL_ARGS"]


def test_profile_contract_resolves_through_the_server() -> None:
    contract = audio_manifest.load_profiles(halas.PROFILES)["profiles"]["halas"]
    assert evaluator_settings(contract) == ("halas", {"cleanup_generation_loops": True})
    assert resolve("evaluators", "halas").__name__ == "HALASEvaluator"
    assert resolve("metrics", contract["skills"]["METRICS_TYPE"]).__name__ == "HALASMetrics"


def test_load_rows_checks_census_and_keeps_the_first_rows(tmp_path: Path, monkeypatch) -> None:
    rows = [_csv_row("0_a.wav"), _csv_row("1_b.wav"), _csv_row("1_b.wav")]
    path = _write_csv(tmp_path / "test.csv", rows)
    monkeypatch.setattr(halas, "hf_hub_download", lambda *args, **kwargs: str(path))
    monkeypatch.setattr(halas, "EXPECTED_ROWS", 3)
    monkeypatch.setattr(halas, "EXPECTED_DUPLICATE_AUDIO_COUNTS", {"1_b.wav": 2})

    loaded = halas.load_halas_rows(2)
    assert [row["audio_id"] for row in loaded] == ["0_a.wav", "1_b.wav"]
    assert [row["source_row_index"] for row in loaded] == [0, 1]

    monkeypatch.setattr(halas, "EXPECTED_DUPLICATE_AUDIO_COUNTS", {})
    with pytest.raises(ValueError, match="census drift"):
        halas.load_halas_rows()
    monkeypatch.setattr(halas, "EXPECTED_ROWS", 4)
    with pytest.raises(ValueError, match="row-count drift"):
        halas.load_halas_rows()


def test_reference_falls_back_and_labels_are_counted() -> None:
    labels = ("No hallucination", "Hallucination or looping", '{"choices":["Hallucination or looping"]}')
    row = _csv_row("3_c.wav", corrected="", original=" Original text. ", labels=labels) | {"source_row_index": 7}
    source = halas.format_source_row(row, "/mnt/audio")
    assert source["expected_answer"] == "Original text." == source["expected_answer_cased"]
    assert source["id"] == "halas-test-0007-3_c.wav"
    assert source["audio_filepath"] == "/mnt/audio/3_c.wav"
    assert source["messages"][0]["audio"] == {"path": "/mnt/audio/3_c.wav", "duration": 1.5}
    assert source["halas_released_system_hallucination_count"] == 2
    assert source["halas_released_system_no_hallucination_count"] == 1

    corrected = halas.format_source_row(_csv_row("4_d.wav", corrected="Fixed.") | {"source_row_index": 0}, "/x")
    assert corrected["expected_answer"] == "Fixed."
    counts = halas._model_label_counts({"a_label": "Language switch", "b_label": "", "c": "x"})
    assert counts == {"hallucination_or_looping": 0, "no_hallucination": 1}


def test_canonical_samples_resample_to_16k_mono() -> None:
    samples = halas.canonical_samples(sine_wav_bytes(8000))
    assert samples.dtype == np.float32 and samples.shape == (8000,)
    assert np.isfinite(samples).all()
    assert halas.canonical_samples(sine_wav_bytes(16000)).shape == (8000,)
    with pytest.raises(ValueError, match="too short"):
        halas.canonical_samples(sine_wav_bytes(16000, seconds=0.001))


def test_write_wav_header(tmp_path: Path) -> None:
    halas.write_wav(tmp_path / "x.wav", np.array([0.0, 1.5, -1.5] * 200, dtype=np.float32))
    with wave.open(str(tmp_path / "x.wav")) as stream:
        assert (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) == (1, 2, 16000)
        frames = np.frombuffer(stream.readframes(3), dtype=np.int16)
    assert frames.tolist() == [0, 32767, -32767]  # clipped
    assert not (tmp_path / "x.wav.tmp").exists()


class _FakeHfFileSystem:
    """Serves local parquet files under the Hugging Face path layout the loader globs."""

    def __init__(self, shards: list[Path]) -> None:
        self.shards = shards
        self.opened: list[str] = []

    def glob(self, pattern: str) -> list[str]:
        assert f"{halas.EARNINGS22_REPO_ID}@{halas.EARNINGS22_REVISION}/chunked/test-*.parquet" in pattern
        return [str(shard) for shard in self.shards]

    def open(self, path: str, mode: str):
        self.opened.append(path)
        return open(path, mode)


def _write_shard(path: Path, clips: list[tuple[str, str]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "segment_id": [segment for segment, _ in clips],
            "file_id": [file for _, file in clips],
            "audio": [{"bytes": sine_wav_bytes(16000), "path": None} for _ in clips],
        }
    )
    pq.write_table(table, path, row_group_size=2)


def test_extract_audio_reads_only_wanted_clips_and_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    first, second = tmp_path / "s0.parquet", tmp_path / "s1.parquet"
    _write_shard(first, [("0", "A"), ("1", "A"), ("2", "A"), ("3", "A")])
    _write_shard(second, [("0", "B"), ("1", "B")])
    filesystem = _FakeHfFileSystem([first, second])
    monkeypatch.setattr(halas, "HfFileSystem", lambda: filesystem)

    audio_dir = tmp_path / "audio"
    halas.extract_audio({"2_A.wav", "0_B.wav"}, audio_dir)
    assert sorted(path.name for path in audio_dir.iterdir()) == ["0_B.wav", "2_A.wav"]
    halas.check_audio({"2_A.wav", "0_B.wav"}, audio_dir)

    filesystem.opened.clear()
    halas.extract_audio({"2_A.wav", "0_B.wav"}, audio_dir)
    assert filesystem.opened == [], "existing clips must not be fetched again"

    with pytest.raises(FileNotFoundError, match="missing from Earnings22"):
        halas.extract_audio({"9_Z.wav"}, audio_dir)


def test_check_audio_rejects_a_missing_or_wrong_format_clip(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing or not a WAV"):
        halas.check_audio({"nope.wav"}, tmp_path)
    (tmp_path / "bad.wav").write_bytes(sine_wav_bytes(8000))
    with pytest.raises(ValueError, match="contract failed"):
        halas.check_audio({"bad.wav"}, tmp_path)


@pytest.fixture
def offline_prepare(tmp_path: Path, monkeypatch) -> Path:
    """Run prepare() against synthetic CSV rows and a fake Earnings-22 shard, writing under tmp_path."""
    rows = [_csv_row(f"{i}_{'A' if i < 2 else 'B'}.wav", labels=("No hallucination", "", "")) for i in range(3)]
    csv_path = _write_csv(tmp_path / "test.csv", rows)
    shard = tmp_path / "shard.parquet"
    _write_shard(shard, [("0", "A"), ("1", "A"), ("2", "B")])
    monkeypatch.setattr(halas, "hf_hub_download", lambda *args, **kwargs: str(csv_path))
    monkeypatch.setattr(halas, "HfFileSystem", lambda: _FakeHfFileSystem([shard]))
    monkeypatch.setattr(halas, "EXPECTED_ROWS", 3)
    monkeypatch.setattr(halas, "EXPECTED_DUPLICATE_AUDIO_COUNTS", {})
    data_dir = tmp_path / "data"
    monkeypatch.setattr(halas, "DATA_DIR", data_dir)
    monkeypatch.setattr(halas, "OUTPUT_FPATH", data_dir / "halas_benchmark.jsonl")
    return data_dir


def test_prepare_writes_one_combined_jsonl_with_existing_audio(offline_prepare: Path) -> None:
    output = halas.prepare()
    assert output == offline_prepare / "halas_benchmark.jsonl"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == ["halas:1", "halas:2", "halas:3"]
    assert {row["benchmark_profile"] for row in rows} == {"halas"}

    first = rows[0]
    audio_path = Path(first["responses_create_params"]["metadata"]["audio_path"])
    assert audio_path == (offline_prepare / "audio" / "0_A.wav").resolve() and audio_path.is_file()
    assert first["responses_create_params"]["input"][0]["content"].startswith("Audio transcription.")
    assert first["ng_source_row"]["messages"][0]["audio"]["path"] == str(audio_path)
    assert first["ng_source_row_sha256"] == audio_manifest.source_hash(first["ng_source_row"])
    assert first["task_type"] == "HALAS-ASR" and first["subset_for_metrics"] == "test"

    assert halas.is_prepared_data_current(output)
    assert not halas.is_prepared_data_current(offline_prepare / "other.jsonl")


def test_prepare_audio_root_and_row_cap_override(offline_prepare: Path) -> None:
    output = halas.prepare(audio_root="/data/halas", max_samples_per_profile=2)
    assert_capped_and_remapped_rows(halas, output, rows=2, audio_path="/data/halas/audio/0_A.wav")
    assert (offline_prepare / "audio" / "0_A.wav").is_file(), "audio is written locally whatever the row prefix"
