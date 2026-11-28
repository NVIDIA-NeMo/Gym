# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the VoiceArena Monsoon benchmark: declaration, discovery, row building, audio, prepare()."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from benchmarks import audio_manifest
from benchmarks.voicearena_monsoon import prepare as monsoon
from nemo_gym.environment.benchmarks import BenchmarkConfig
from nemo_gym.environment.manifest import load_manifest
from tests.unit_tests.benchmarks.audio_benchmark_helpers import assert_capped_and_remapped_rows, sine_wav_bytes


USES_AUDIO_LIBRARIES = True
ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "voicearena_monsoon"
EN, HI = "voicearena-monsoon.en-in.public", "voicearena-monsoon.hi-in.public"


def test_declaration_specifics() -> None:
    dataset = BenchmarkConfig.from_config_path(BENCHMARK / "config.yaml").dataset
    manifest = load_manifest(BENCHMARK / "manifest.yaml")
    assert "soundfile" in dataset.prepare_dependencies
    assert manifest.canonical_split == "test" and manifest.prompt_source == "prepared" and manifest.experimental
    assert manifest.licensing == "CC-BY-4.0"


def test_profile_contracts_resolve_and_cover_the_sources() -> None:
    profiles = audio_manifest.load_profiles(monsoon.PROFILES)["profiles"]
    assert set(profiles) == set(monsoon.SOURCES) == {EN, HI}
    for contract in profiles.values():
        assert audio_manifest.request_controls(contract)["max_output_tokens"] == 256
    assert "Latin script" in profiles[EN]["prompt_override"]["user_prompt"]
    assert "Devanagari" in profiles[HI]["prompt_override"]["user_prompt"]


def _source(language: str, field: str, rows: int, speakers: int) -> monsoon.Source:
    return monsoon.Source(language, language.replace("-", "_"), f"org/{language}", "rev", rows, speakers, field)


def _write_shards(directory: Path, source: monsoon.Source, records: list[dict]) -> list[Path]:
    """Two parquet shards (two rows per row group) in the layout of the Hugging Face datasets."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    shards = []
    for number, chunk in enumerate([records[:2], records[2:]]):
        table = pa.table(
            {
                "audio": [{"bytes": sine_wav_bytes(16000 if r["id"] != "c3" else 8000), "path": None} for r in chunk],
                "id": [r["id"] for r in chunk],
                source.reference_field: [r[source.reference_field] for r in chunk],
                "speaker_id": [r["speaker_id"] for r in chunk],
                "language": [source.source_language] * len(chunk),
                "audio_length_s": [0.5] * len(chunk),
            }
        )
        shard = directory / f"{source.language}-{number}.parquet"
        pq.write_table(table, shard, row_group_size=1)
        shards.append(shard)
    return shards


class _FakeHfFileSystem:
    """Serves local parquet files under the Hugging Face path layout the loader globs."""

    def __init__(self, shards: dict[str, list[Path]]) -> None:
        self.shards = shards
        self.opened: list[str] = []

    def glob(self, pattern: str) -> list[str]:
        repo = pattern.split("datasets/")[1].split("@")[0]
        assert pattern.endswith("@rev/data/test-*.parquet")
        return [str(shard) for shard in self.shards[repo]]

    def open(self, path: str, mode: str):
        self.opened.append(path)
        return open(path, mode)


@pytest.fixture
def offline_prepare(tmp_path: Path, monkeypatch) -> tuple[Path, _FakeHfFileSystem]:
    """Run prepare() against synthetic en-IN (text) and hi-IN (lattice) shards, writing under tmp_path."""
    en = _source("en-in", "text", 3, 2)
    hi = _source("hi-in", "lattice", 3, 3)
    en_records = [{"id": f"c{i}", "text": f" text {i} ", "speaker_id": 1 + i % 2} for i in range(1, 4)]
    hi_records = [{"id": f"h{i}", "lattice": [[f"w{i}", f"v{i}"], ["x"]], "speaker_id": i} for i in range(1, 4)]
    filesystem = _FakeHfFileSystem(
        {en.repo: _write_shards(tmp_path, en, en_records), hi.repo: _write_shards(tmp_path, hi, hi_records)}
    )
    monkeypatch.setattr(monsoon, "HfFileSystem", lambda: filesystem)
    monkeypatch.setattr(monsoon, "SOURCES", {EN: en, HI: hi})
    data_dir = tmp_path / "data"
    monkeypatch.setattr(monsoon, "DATA_DIR", data_dir)
    monkeypatch.setattr(monsoon, "OUTPUT_FPATH", data_dir / "voicearena_monsoon_benchmark.jsonl")
    return data_dir, filesystem


def test_load_rows_checks_the_census_and_keeps_the_first_rows(offline_prepare, monkeypatch) -> None:
    source = monsoon.SOURCES[EN]
    rows = monsoon.load_rows(source)
    assert [row["id"] for row in rows] == ["c1", "c2", "c3"]
    assert [row["id"] for row in monsoon.load_rows(source, 2)] == ["c1", "c2"]
    assert monsoon.load_rows(monsoon.SOURCES[HI])[0]["lattice"] == [["w1", "v1"], ["x"]]

    with pytest.raises(ValueError, match="row-count drift"):
        monsoon.load_rows(source._replace(expected_rows=4))
    with pytest.raises(ValueError, match="speaker-count drift"):
        monsoon.load_rows(source._replace(expected_speakers=3))
    with pytest.raises(ValueError, match="unexpected id or language"):
        monsoon.load_rows(source._replace(source_language="other"))


def test_lattice_is_stripped_and_validated() -> None:
    assert monsoon._check_lattice([[" a ", "b"], ["c"]], 1) == [["a", "b"], ["c"]]
    for bad in ([], [[]], [["a", " "]], [["a", None]]):
        with pytest.raises(ValueError, match="lattice"):
            monsoon._check_lattice(bad, 1)


def test_format_source_row_keeps_the_validated_schema() -> None:
    en = _source("en-in", "text", 1, 1)
    row = {"id": "c1", "text": " Hello there ", "speaker_id": 7, "audio_length_s": 1.5}
    source = monsoon.format_source_row(row, EN, en, "/mnt")
    assert source["id"] == "voicearena-monsoon-en-in.public-c1"
    assert source["expected_answer"] == "Hello there" and "reference_lattice" not in source
    assert source["audio_filepath"] == "/mnt/en-in/public/audio/c1.wav"
    assert source["messages"] == [
        {
            "role": "user",
            "content": "Transcribe the following:",
            "audio": {"path": source["audio_filepath"], "duration": 1.5},
        }
    ]
    assert source["monsoon_language"] == "en-in" and "subset_for_metrics" not in source
    assert source["extra_fields"]["source_license"] == "CC-BY-4.0" and source["extra_fields"]["source_id"] == "c1"

    hi = _source("hi-in", "lattice", 1, 1)
    lattice = [["इम्प्रूव", "improve"], ["है", "hai"]]
    source = monsoon.format_source_row(
        {"id": "h1", "lattice": lattice, "speaker_id": 7, "audio_length_s": 2.0}, HI, hi, "/m"
    )
    assert source["expected_answer"] == "इम्प्रूव है" and source["reference_lattice"] == lattice


def test_canonical_samples_resample_to_16k_mono() -> None:
    assert monsoon.canonical_samples(sine_wav_bytes(16000)).shape == (8000,)
    samples = monsoon.canonical_samples(sine_wav_bytes(8000))
    assert samples.dtype == np.float32 and samples.shape == (8000,) and np.isfinite(samples).all()
    with pytest.raises(ValueError, match="invalid"):
        monsoon.canonical_samples(sine_wav_bytes(16000, seconds=0.001))
    with pytest.raises(ValueError, match="libsndfile"):
        monsoon.canonical_samples(b"not audio")


def test_write_wav_header(tmp_path: Path) -> None:
    monsoon.write_wav(tmp_path / "x.wav", np.zeros(800, dtype=np.float32))
    with wave.open(str(tmp_path / "x.wav")) as stream:
        assert (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()) == (
            1,
            2,
            16000,
            800,
        )
    assert not (tmp_path / "x.wav.tmp").exists()


def test_check_audio_rejects_a_missing_or_wrong_clip(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing or not a WAV"):
        monsoon.check_audio([{"id": "nope", "audio_length_s": 0.5}], tmp_path)
    (tmp_path / "bad.wav").write_bytes(sine_wav_bytes(8000))
    with pytest.raises(ValueError, match="contract failed"):
        monsoon.check_audio([{"id": "bad", "audio_length_s": 0.5}], tmp_path)
    monsoon.write_wav(tmp_path / "short.wav", np.zeros(8000, dtype=np.float32))
    with pytest.raises(ValueError, match="duration drift"):
        monsoon.check_audio([{"id": "short", "audio_length_s": 2.0}], tmp_path)


def test_extract_audio_reads_only_wanted_clips_and_is_idempotent(offline_prepare, tmp_path: Path) -> None:
    _, filesystem = offline_prepare
    source = monsoon.SOURCES[EN]
    rows = monsoon.load_rows(source)
    audio_dir = tmp_path / "clips"
    monsoon.extract_audio(source, [rows[2]], audio_dir)
    assert [path.name for path in audio_dir.iterdir()] == ["c3.wav"]  # the 8 kHz clip, resampled
    monsoon.check_audio([rows[2]], audio_dir)

    filesystem.opened.clear()
    monsoon.extract_audio(source, [rows[2]], audio_dir)
    assert filesystem.opened == [], "existing clips must not be fetched again"

    with pytest.raises(FileNotFoundError, match="missing from"):
        monsoon.extract_audio(source, [{"id": "zzz"}], audio_dir)


def test_prepare_writes_one_combined_jsonl_with_existing_audio(offline_prepare) -> None:
    data_dir, _ = offline_prepare
    output = monsoon.prepare()
    assert output == data_dir / "voicearena_monsoon_benchmark.jsonl"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == [f"{EN}:{i}" for i in (1, 2, 3)] + [f"{HI}:{i}" for i in (1, 2, 3)]
    assert [row["benchmark_profile"] for row in rows] == [EN] * 3 + [HI] * 3

    first, hindi = rows[0], rows[3]
    audio_path = Path(first["responses_create_params"]["metadata"]["audio_path"])
    assert audio_path == (data_dir / "en-in" / "public" / "audio" / "c1.wav").resolve() and audio_path.is_file()
    assert first["responses_create_params"]["input"][0]["content"].endswith("Use Latin script only.")
    assert first["expected_answer"] == "text 1"
    assert first["ng_source_row"]["messages"][0]["audio"]["path"] == str(audio_path)
    assert first["ng_source_row_sha256"] == audio_manifest.source_hash(first["ng_source_row"])
    assert hindi["reference_lattice"] == [["w1", "v1"], ["x"]] and hindi["expected_answer"] == "w1 x"
    assert Path(hindi["responses_create_params"]["metadata"]["audio_path"]).is_file()

    assert monsoon.is_prepared_data_current(output)
    assert not monsoon.is_prepared_data_current(data_dir / "other.jsonl")


def test_prepare_audio_root_and_row_cap_override(offline_prepare) -> None:
    data_dir, _ = offline_prepare
    output = monsoon.prepare(audio_root="/data/voicearena-monsoon", max_samples_per_profile=1)
    rows = assert_capped_and_remapped_rows(
        monsoon, output, rows=2, audio_path="/data/voicearena-monsoon/hi-in/public/audio/h1.wav", index=1
    )
    assert [row["benchmark_profile"] for row in rows] == [EN, HI]
    assert (data_dir / "hi-in" / "public" / "audio" / "h1.wav").is_file(), "audio is written locally"
    assert not (data_dir / "hi-in" / "public" / "audio" / "h2.wav").exists(), "only the capped clips are fetched"


@pytest.mark.parametrize("cap", [0, -1, 1.5, True, "2"])
def test_prepare_rejects_an_invalid_row_cap_before_any_download(offline_prepare, cap) -> None:
    _, filesystem = offline_prepare
    with pytest.raises(ValueError, match="positive integer"):
        monsoon.prepare(max_samples_per_profile=cap)
    assert filesystem.opened == []
