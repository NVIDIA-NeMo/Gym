# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the MMAR benchmark: declaration, discovery, row building, audio extraction, prepare()."""

import io
import json
import tarfile
from pathlib import Path

import pytest

from benchmarks import audio_manifest
from benchmarks.mmar import prepare as mmar
from nemo_gym.environment.manifest import load_manifest
from tests.unit_tests.benchmarks.audio_benchmark_helpers import assert_capped_and_remapped_rows


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "mmar"


def _meta_row(name: str, *, modality: str = "speech", timestamp: str = "00:00:18,00:00:48") -> dict:
    return {
        "id": name,
        "audio_path": f"./audio/{name}.wav",
        "question": "How many pizzas did the woman order",
        "choices": ["10", "2", "20", "12"],
        "answer": "20",
        "modality": modality,
        "category": "Semantic Layer",
        "sub-category": "Content Analysis",
        "language": "en",
        "source": "youtube",
        "url": "https://example.com/" + name,
        "timestamp": timestamp,
    }


class _FakeHfFileSystem:
    """Serves one local tar.gz under the Hugging Face path the loader opens."""

    def __init__(self, archive: Path) -> None:
        self.archive = archive
        self.opened: list[str] = []

    def open(self, path: str, mode: str):
        assert path == f"datasets/{mmar.HF_REPO_ID}@{mmar.HF_REVISION}/{mmar.AUDIO_ARCHIVE_FILENAME}"
        self.opened.append(path)
        return self.archive.open(mode)


def _write_archive(path: Path, names: list[str]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name in names:
            payload = f"audio:{name}".encode()
            member = tarfile.TarInfo(f"mmar-audio/{name}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return path


def test_declaration_specifics() -> None:
    manifest = load_manifest(BENCHMARK / "manifest.yaml")
    assert manifest.canonical_split == "test" and manifest.prompt_source == "prepared" and manifest.experimental


def test_profile_contract_values() -> None:
    contract = audio_manifest.load_profiles(mmar.PROFILES)["profiles"]["mmar"]
    assert contract["skills"]["METRICS_TYPE"] == "set_token_mcq"
    assert contract["skills"]["EVAL_ARGS"] == "++eval_type=set-token-mcq"
    controls = audio_manifest.request_controls(contract)
    assert controls["max_output_tokens"] == 20480 and controls["temperature"] == 0.6


def test_modality_constants_match_the_released_census() -> None:
    assert sum(mmar.EXPECTED_MODALITY_COUNTS.values()) == 1000 and len(mmar.EXPECTED_MODALITY_COUNTS) == 7
    assert mmar.SPEECH_MODALITIES == ("speech",) and mmar.EXPECTED_MODALITY_COUNTS["speech"] == 294


def test_select_modalities_defaults_to_all_and_accepts_lists_and_strings() -> None:
    assert mmar.select_modalities(None) == tuple(mmar.EXPECTED_MODALITY_COUNTS)
    assert mmar.select_modalities("speech, music,speech") == ("speech", "music")
    assert mmar.select_modalities(["mix-sound-speech"]) == ("mix-sound-speech",)
    for bad in ("voice", "speech,voice", "", []):
        with pytest.raises(ValueError, match="Empty or unknown"):
            mmar.select_modalities(bad)


@pytest.fixture
def small_census(monkeypatch) -> dict:
    counts = {"speech": 3, "music": 1, "mix-sound-speech": 1}
    monkeypatch.setattr(mmar, "EXPECTED_MODALITY_COUNTS", counts)
    return counts


def _small_meta() -> list[dict]:
    return [
        _meta_row("a"),
        _meta_row("m", modality="music"),
        _meta_row("b"),
        _meta_row("x", modality="mix-sound-speech"),
        _meta_row("c"),
    ]


def test_load_rows_defaults_to_all_rows_and_narrows_by_modality(tmp_path: Path, monkeypatch, small_census) -> None:
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(_small_meta()))
    monkeypatch.setattr(mmar, "hf_hub_download", lambda *args, **kwargs: str(path))

    assert [row["id"] for row in mmar.load_mmar_rows()] == ["a", "m", "b", "x", "c"]
    assert [row["id"] for row in mmar.load_mmar_rows(mmar.SPEECH_MODALITIES)] == ["a", "b", "c"]
    assert [row["id"] for row in mmar.load_mmar_rows(("music", "mix-sound-speech"))] == ["m", "x"]
    assert [row["id"] for row in mmar.load_mmar_rows(max_samples=2)] == ["a", "m"]

    small_census["speech"] = 4
    with pytest.raises(ValueError, match="census drift"):
        mmar.load_mmar_rows()

    path.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(TypeError, match="JSON list"):
        mmar.load_mmar_rows()


def test_format_source_row_keeps_official_fields_and_builds_the_prompt() -> None:
    source = mmar.format_source_row(_meta_row("clip"), "/data/mmar/audio")
    audio_path = "/data/mmar/audio/clip.wav"
    assert source["audio_path"] == audio_path == source["audio_filepath"]
    assert source["source_audio_path"] == "./audio/clip.wav"
    assert source["expected_answer"] == "20" and source["answer"] == "20"
    assert source["subset_for_metrics"] == "Semantic Layer" and source["sub-category"] == "Content Analysis"
    assert source["dataset_revision"] == mmar.HF_REVISION
    assert source["official_scorer_revision"] == mmar.OFFICIAL_SCORER_REVISION
    (message,) = source["messages"]
    assert message["audio"] == {"path": audio_path, "duration": 30.0}
    assert message["content"].endswith("Options:\nA. 10\nB. 2\nC. 20\nD. 12\n\n" + mmar.ANSWER_INSTRUCTION)

    unparsable = mmar.format_source_row(_meta_row("clip", timestamp="n/a"), "/a")
    assert unparsable["messages"][0]["audio"] == {"path": "/a/clip.wav"}
    reversed_clip = mmar.format_source_row(_meta_row("clip", timestamp="00:00:30,00:00:10"), "/a")
    assert "duration" not in reversed_clip["messages"][0]["audio"]


def test_audio_filename_rejects_an_empty_path() -> None:
    with pytest.raises(ValueError, match="Invalid MMAR audio_path"):
        mmar._audio_filename({"audio_path": ".."})


def test_extract_audio_streams_only_wanted_clips_and_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    archive = _write_archive(tmp_path / "audio.tar.gz", ["a.wav", "b.wav", "c.wav"])
    filesystem = _FakeHfFileSystem(archive)
    monkeypatch.setattr(mmar, "HfFileSystem", lambda: filesystem)

    audio_dir = tmp_path / "audio"
    rows = [_meta_row("a"), _meta_row("c")]
    mmar.extract_audio(rows, audio_dir)
    assert sorted(path.name for path in audio_dir.iterdir()) == ["a.wav", "c.wav"]
    assert (audio_dir / "a.wav").read_bytes() == b"audio:a.wav"

    filesystem.opened.clear()
    mmar.extract_audio(rows, audio_dir)
    assert filesystem.opened == [], "existing clips must not be fetched again"

    with pytest.raises(FileNotFoundError, match="missing 1 clips"):
        mmar.extract_audio([_meta_row("z")], audio_dir)


@pytest.fixture
def offline_prepare(tmp_path: Path, monkeypatch, small_census) -> Path:
    """Run prepare() against a synthetic metadata file and archive, writing under tmp_path."""
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps(_small_meta()))
    archive = _write_archive(tmp_path / "audio.tar.gz", ["a.wav", "m.wav", "b.wav", "x.wav", "c.wav"])
    monkeypatch.setattr(mmar, "hf_hub_download", lambda *args, **kwargs: str(meta))
    monkeypatch.setattr(mmar, "HfFileSystem", lambda: _FakeHfFileSystem(archive))
    data_dir = tmp_path / "data"
    monkeypatch.setattr(mmar, "DATA_DIR", data_dir)
    monkeypatch.setattr(mmar, "OUTPUT_FPATH", data_dir / "mmar_benchmark.jsonl")
    return data_dir


def test_prepare_defaults_to_every_modality_with_existing_audio(offline_prepare: Path) -> None:
    output = mmar.prepare()
    assert output == offline_prepare / "mmar_benchmark.jsonl"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == [f"mmar:{i}" for i in range(1, 6)]
    assert [row["modality"] for row in rows] == ["speech", "music", "speech", "mix-sound-speech", "speech"]
    assert {row["benchmark_profile"] for row in rows} == {"mmar"}
    assert sorted(path.name for path in (offline_prepare / "audio").iterdir()) == [f"{n}.wav" for n in "abcmx"]

    first = rows[0]
    request = first["responses_create_params"]
    audio_path = Path(request["metadata"]["audio_path"])
    assert audio_path == (offline_prepare / "audio" / "a.wav").resolve() and audio_path.is_file()
    system, user = request["input"]
    assert system["role"] == "system" and system["content"].startswith("Listen to the provided audio")
    assert user["content"].startswith("Listen to the audio and answer the multiple-choice question.")
    assert first["ng_source_row"]["messages"][0]["audio"]["path"] == str(audio_path)
    assert first["ng_source_row_sha256"] == audio_manifest.source_hash(first["ng_source_row"])
    assert first["subset_for_metrics"] == "Semantic Layer" and first["expected_answer"] == "20"

    assert mmar.is_prepared_data_current(output)
    assert not mmar.is_prepared_data_current(offline_prepare / "other.jsonl")


def test_prepare_modalities_narrow_rows_and_downloads(offline_prepare: Path) -> None:
    for modalities in ("speech", ["speech"], mmar.SPEECH_MODALITIES):
        output = mmar.prepare(modalities=modalities)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert [row["id"] for row in rows] == ["a", "b", "c"] and {row["modality"] for row in rows} == {"speech"}
        assert not mmar.is_prepared_data_current(output), "narrowed data must be prepared again by default"
    assert sorted(path.name for path in (offline_prepare / "audio").iterdir()) == ["a.wav", "b.wav", "c.wav"]
    with pytest.raises(ValueError, match="Empty or unknown"):
        mmar.prepare(modalities="voice")


def test_prepare_with_every_modality_listed_is_still_the_default(offline_prepare: Path) -> None:
    output = mmar.prepare(modalities=["mix-sound-speech", "music", "speech"])
    assert mmar.is_prepared_data_current(output)


def test_prepare_audio_root_and_row_cap_override(offline_prepare: Path) -> None:
    output = mmar.prepare(modalities="speech", audio_root="/data/mmar", max_samples_per_profile=2)
    assert_capped_and_remapped_rows(mmar, output, rows=2, audio_path="/data/mmar/audio/a.wav")
    assert (offline_prepare / "audio" / "a.wav").is_file(), "audio is written locally whatever the row prefix"
    assert not (offline_prepare / "audio" / "c.wav").exists(), "capped runs fetch only the kept rows"
