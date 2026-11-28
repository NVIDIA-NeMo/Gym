# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the Coval STT v1 benchmark: declaration, discovery, manifest census, row building, prepare()."""

import hashlib
import json
import wave
from decimal import Decimal
from pathlib import Path

import pytest

from benchmarks import audio_manifest
from benchmarks.coval_stt_v1 import prepare as coval
from nemo_gym.environment.manifest import load_manifest
from tests.unit_tests.benchmarks.audio_benchmark_helpers import assert_capped_and_remapped_rows


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "coval_stt_v1"


def test_declaration_specifics() -> None:
    manifest = load_manifest(BENCHMARK / "manifest.yaml")
    assert manifest.canonical_split == "test" and manifest.prompt_source == "prepared" and manifest.experimental
    assert manifest.licensing == "CC-BY-4.0" and manifest.domain == "asr"


def test_profile_contract_values() -> None:
    contract = audio_manifest.load_profiles(coval.PROFILES)["profiles"]["coval-stt-v1"]
    assert audio_manifest.request_controls(contract)["max_output_tokens"] == 256
    assert contract["request"] == "greedy"
    assert "CovalSTTV1Evaluator" in contract["skills"]["EVAL_ARGS"]
    assert "++eval_config.cleanup_generation_loops=true" in contract["skills"]["EVAL_ARGS"].split()
    assert contract["skills"]["METRICS_TYPE"].endswith("::CovalSTTV1Metrics")


def _write_wav(path: Path, seconds: float, rate: int = coval.SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\x01\x00" * int(seconds * rate))


@pytest.fixture
def offline_source(tmp_path: Path, monkeypatch) -> Path:
    """A synthetic three-utterance Coval source served from local files, with the census constants adapted."""
    source = tmp_path / "source"
    items = []
    for number, (speaker, seconds) in enumerate([("1", 2.0), ("1", 3.0), ("2", 2.5)], 1):
        wav = source / "audio" / f"{number:04d}.wav"
        _write_wav(wav, seconds)
        items.append(
            {
                "path": f"audio/{number:04d}.wav",
                "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                "transcript": f"TRANSCRIPT {number}",
                "duration_sec": seconds,
                "speaker_id": speaker,
                "chapter_id": "9",
                "utterance_id": f"{speaker}-9-{number:04d}",
                "speech_end_offset_ms": seconds * 1000,
            }
        )
    manifest = source / "stt-v1.json"
    manifest.write_text(
        json.dumps({"id": "stt-v1", "version": "1.0.0", "license": "CC-BY-4.0", "items": items}), encoding="utf-8"
    )
    monkeypatch.setattr(coval, "SOURCE_MANIFEST_URL", manifest.as_uri())
    monkeypatch.setattr(coval, "SOURCE_MANIFEST_SHA256", audio_manifest.sha256_file(manifest))
    monkeypatch.setattr(coval, "SOURCE_OBJECT_ROOT", source.as_uri())
    monkeypatch.setattr(coval, "EXPECTED_ROWS", 3)
    monkeypatch.setattr(coval, "EXPECTED_SPEAKERS", 2)
    monkeypatch.setattr(coval, "EXPECTED_DURATION_SECONDS", Decimal("7.5"))
    data_dir = tmp_path / "data"
    monkeypatch.setattr(coval, "DATA_DIR", data_dir)
    monkeypatch.setattr(coval, "OUTPUT_FPATH", data_dir / "coval_stt_v1_benchmark.jsonl")
    return manifest


def test_load_items_checks_the_census(offline_source: Path, monkeypatch) -> None:
    assert [item["utterance_id"] for item in coval.load_items()] == ["1-9-0001", "1-9-0002", "2-9-0003"]

    monkeypatch.setattr(coval, "EXPECTED_ROWS", 4)
    with pytest.raises(ValueError, match="row-count drift"):
        coval.load_items()
    monkeypatch.setattr(coval, "EXPECTED_ROWS", 3)
    monkeypatch.setattr(coval, "EXPECTED_SPEAKERS", 3)
    with pytest.raises(ValueError, match="speaker-count drift"):
        coval.load_items()
    monkeypatch.setattr(coval, "EXPECTED_SPEAKERS", 2)
    monkeypatch.setattr(coval, "EXPECTED_DURATION_SECONDS", Decimal("8"))
    with pytest.raises(ValueError, match="duration drift"):
        coval.load_items()
    monkeypatch.setattr(coval, "SOURCE_MANIFEST_SHA256", "0" * 64)
    (coval.DATA_DIR / "source" / "stt-v1.json").unlink()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        coval.load_items()


def test_format_source_row_keeps_the_validated_schema() -> None:
    item = {
        "path": "audio/0007.wav",
        "sha256": "ab" * 32,
        "transcript": "HELLO WORLD",
        "duration_sec": 4.5,
        "speaker_id": "11",
        "chapter_id": "22",
        "utterance_id": "11-22-0003",
    }
    row = coval.format_source_row(item, "/mnt/coval")
    assert row["id"] == "11-22-0003" and row["expected_answer"] == "HELLO WORLD"
    assert row["audio_filepath"] == "/mnt/coval/audio/0007.wav"
    assert row["messages"] == [
        {
            "role": "user",
            "content": "Transcribe the following:",
            "audio": {"path": row["audio_filepath"], "duration": 4.5},
        }
    ]
    assert row["task_type"] == "Coval-STT-v1-ASR" and "subset_for_metrics" not in row
    assert row["extra_fields"]["source_audio_sha256"] == "ab" * 32
    assert row["extra_fields"]["source_revision"] == coval.SOURCE_REVISION


def test_download_audio_rejects_a_wrong_hash_or_wav_contract(offline_source: Path, tmp_path: Path) -> None:
    items = coval.load_items()
    coval.download_audio(items, tmp_path / "audio_ok")

    tampered = [dict(items[0], sha256="0" * 64)]
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        coval.download_audio(tampered, tmp_path / "audio_bad")

    wrong_rate = tmp_path / "wrong" / "audio" / "0001.wav"
    _write_wav(wrong_rate, 2.0, rate=8000)
    item = dict(items[0], sha256=audio_manifest.sha256_file(wrong_rate))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(coval, "SOURCE_OBJECT_ROOT", (tmp_path / "wrong").as_uri())
        with pytest.raises(ValueError, match="audio contract failed"):
            coval.download_audio([item], tmp_path / "audio_rate")


def test_prepare_writes_one_combined_jsonl_with_existing_audio(offline_source: Path) -> None:
    output = coval.prepare()
    assert output == coval.DATA_DIR / "coval_stt_v1_benchmark.jsonl"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == ["coval-stt-v1:1", "coval-stt-v1:2", "coval-stt-v1:3"]
    assert {row["benchmark_profile"] for row in rows} == {"coval-stt-v1"}

    first = rows[0]
    audio_path = Path(first["responses_create_params"]["metadata"]["audio_path"])
    assert audio_path == (coval.DATA_DIR / "audio" / "0001.wav").resolve() and audio_path.is_file()
    assert first["responses_create_params"]["input"][0]["content"].startswith("Audio transcription.")
    assert first["ng_source_row"]["messages"][0]["audio"]["path"] == str(audio_path)
    assert first["ng_source_row_sha256"] == audio_manifest.source_hash(first["ng_source_row"])
    assert first["expected_answer"] == "TRANSCRIPT 1" and first["speaker_id"] == "1"

    assert coval.is_prepared_data_current(output)
    assert not coval.is_prepared_data_current(coval.DATA_DIR / "other.jsonl")


def test_prepare_audio_root_and_row_cap_override(offline_source: Path) -> None:
    output = coval.prepare(audio_root="/data/coval_stt_v1", max_samples_per_profile=2)
    assert_capped_and_remapped_rows(coval, output, rows=2, audio_path="/data/coval_stt_v1/audio/0001.wav")
    assert (coval.DATA_DIR / "audio" / "0001.wav").is_file(), "audio is written locally whatever the row prefix"
    assert not (coval.DATA_DIR / "audio" / "0003.wav").exists(), "only the capped utterances are fetched"


@pytest.mark.parametrize("cap", [0, -1])
def test_prepare_rejects_a_nonpositive_row_cap_before_downloading(offline_source: Path, cap: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        coval.prepare(max_samples_per_profile=cap)
    assert not (coval.DATA_DIR / "audio").exists()
