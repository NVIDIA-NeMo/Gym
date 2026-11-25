# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the MUCS 2021 benchmark: declaration, discovery, metadata census, row building, prepare()."""

import hashlib
import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks import audio_manifest
from benchmarks.mucs2021 import prepare as mucs
from nemo_gym.environment.benchmarks import BenchmarkConfig
from nemo_gym.environment.manifest import load_manifest
from resources_servers.audio_eval import mucs_assets
from resources_servers.audio_eval.score import evaluator_settings, resolve
from tests.unit_tests.benchmarks.audio_benchmark_helpers import pcm_wav_bytes


USES_AUDIO_LIBRARIES = True
ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "mucs2021"


def test_declaration_specifics() -> None:
    dataset = BenchmarkConfig.from_config_path(BENCHMARK / "config.yaml").dataset
    manifest = load_manifest(BENCHMARK / "manifest.yaml")
    assert "soundfile" in dataset.prepare_dependencies
    assert manifest.canonical_split == "test" and manifest.prompt_source == "prepared" and manifest.experimental
    assert manifest.licensing == "CC-BY-SA-4.0" and manifest.domain == "asr"


def test_profile_contracts_and_census_constants() -> None:
    profiles = audio_manifest.load_profiles(mucs.PROFILES)["profiles"]
    assert list(profiles) == ["mucs2021.bengali-english", "mucs2021.hindi-english"]
    for contract in profiles.values():
        assert contract["request"] == "greedy" and contract["prompt_override"]["system_prompt"] == ""
        assert contract["skills"]["METRICS_TYPE"] == "nemo_skills.dataset.mucs2021.scorer::MUCS2021Metrics"
    assert {language.rows for language in mucs.LANGUAGES.values()} == {4034, 3130}
    assert mucs.SOURCE_SHA256 == mucs_assets.ARCHIVE_SHA256 and mucs.SOURCE_URL == mucs_assets.ARCHIVE_URL


def test_profile_contracts_resolve_through_the_server() -> None:
    for contract in audio_manifest.load_profiles(mucs.PROFILES)["profiles"].values():
        evaluator, settings = evaluator_settings(contract)
        assert resolve("evaluators", evaluator).__name__ == "MUCS2021Evaluator"
        assert settings == {"cleanup_generation_loops": True}
        assert resolve("metrics", contract["skills"]["METRICS_TYPE"]).__name__ == "MUCS2021Metrics"


def test_select_languages_defaults_to_all_and_accepts_lists_and_strings() -> None:
    assert mucs.select_languages(None) == ("hindi-english", "bengali-english")
    assert mucs.select_languages("bengali-english") == ("bengali-english",)
    assert mucs.select_languages(["bengali-english", "hindi-english"]) == ("hindi-english", "bengali-english")
    for bad in ("xx", "hindi-english,xx", "", []):
        with pytest.raises(ValueError, match="Empty or unknown"):
            mucs.select_languages(bad)


# Two tiny languages; the second recording of "hindi" is never referenced by a kept segment.
METADATA = {
    "hindi": {
        "text": "h_1 par parsing\nh_2 \nh_3 ek do\n",
        "segments": "h_1 recA 0.0 0.5\nh_2 recA 0.5 1.0\nh_3 recB 0.25 1.25\n",
        "wav.scp": "recA recA.wav\nrecB recB.wav\n",
        "utt2spk": "h_1 s1\nh_2 s1\nh_3 s2\n",
    },
    "bengali": {
        "text": "b_1 ami bhalo\n",
        "segments": "b_1 recC 1.0 2.0\n",
        "wav.scp": "recC recC.wav\n",
        "utt2spk": "b_1 s3\n",
    },
}
ASSETS = {"punct.mapping.txt": "a b\n", "hin2engwords.txt": "ek one\n", "ben2engwords.txt": "ami i\n"}


@pytest.fixture
def offline_archive(tmp_path: Path, monkeypatch) -> Path:
    """A tiny archive with the release layout, and census constants adapted to it."""
    archive_path = tmp_path / "remote" / "blindtest.tar.gz"
    archive_path.parent.mkdir()
    recordings = {"hindi/recA.wav": 1.0, "hindi/recB.wav": 1.5, "bengali/recC.wav": 2.0}

    def add(archive: tarfile.TarFile, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(f"{mucs.ARCHIVE_ROOT}/{name}")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))

    with tarfile.open(archive_path, "w:gz") as archive:
        for name, seconds in recordings.items():
            add(archive, name, pcm_wav_bytes(int(seconds * 16000)))
        for directory, files in METADATA.items():
            for name, text in files.items():
                add(archive, f"{directory}/files/{name}", text.encode())
        for name, text in ASSETS.items():
            add(archive, f"inference/{name}", text.encode())

    def digest(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    languages = {
        "hindi-english": replace(
            mucs.LANGUAGES["hindi-english"],
            rows=3,
            recordings=2,
            empty_references=1,
            metadata_sha256={name: digest(text) for name, text in METADATA["hindi"].items()},
        ),
        "bengali-english": replace(
            mucs.LANGUAGES["bengali-english"],
            rows=1,
            recordings=1,
            empty_references=0,
            metadata_sha256={name: digest(text) for name, text in METADATA["bengali"].items()},
        ),
    }
    monkeypatch.setattr(mucs, "LANGUAGES", languages)
    monkeypatch.setattr(mucs, "SOURCE_URL", archive_path.as_uri())
    monkeypatch.setattr(mucs, "SOURCE_SHA256", audio_manifest.sha256_file(archive_path))
    monkeypatch.setattr(mucs_assets, "ASSET_SHA256", {name: digest(text) for name, text in ASSETS.items()})
    monkeypatch.setattr(mucs_assets, "assets_dir", lambda: tmp_path / "assets")
    data_dir = tmp_path / "data"
    monkeypatch.setattr(mucs, "DATA_DIR", data_dir)
    monkeypatch.setattr(mucs, "OUTPUT_FPATH", data_dir / "mucs2021_benchmark.jsonl")
    return archive_path


def test_metadata_and_assets_are_read_and_checked_against_their_pins(offline_archive: Path, tmp_path: Path) -> None:
    metadata, inference = mucs.read_metadata(offline_archive)
    assert set(metadata) == {"hindi", "bengali"} and set(inference) == set(ASSETS)
    segments = mucs.parse_language(mucs.LANGUAGES["hindi-english"], metadata["hindi"])
    assert [segment["id"] for segment in segments] == ["h_1", "h_2", "h_3"]
    assert segments[1]["text"] == "" and segments[2]["speaker"] == "s2" and segments[2]["recording"] == "recB"

    mucs.install_assets(inference)
    assert (tmp_path / "assets" / "hin2engwords.txt").read_text() == "ek one\n"
    with pytest.raises(ValueError, match="does not match"):
        mucs.install_assets({**inference, "punct.mapping.txt": b"tampered"})

    tampered = {**metadata["hindi"], "text": b"h_1 changed\nh_2 \nh_3 ek do\n"}
    with pytest.raises(ValueError, match="differs from the pinned release"):
        mucs.parse_language(mucs.LANGUAGES["hindi-english"], tampered)
    census = replace(mucs.LANGUAGES["hindi-english"], empty_references=0)
    with pytest.raises(ValueError, match="census differs"):
        mucs.parse_language(census, metadata["hindi"])


def test_format_source_row_keeps_the_validated_schema() -> None:
    language = mucs.LANGUAGES["hindi-english"]
    segment = {"id": "u_0", "recording": "recA", "start": "1.0", "end": "2.5", "text": "par parsing", "speaker": "s1"}
    row = mucs.format_source_row(language, segment, "/mnt/hindi-english/audio")
    assert row["id"] == "u_0" and row["expected_answer"] == "par parsing" and row["duration"] == 1.5
    assert row["audio_path"] == "/mnt/hindi-english/audio/u_0.wav"
    assert row["messages"] == [
        {
            "audio": {"duration": 1.5, "path": "/mnt/hindi-english/audio/u_0.wav"},
            "content": "Transcribe the following audio.",
            "role": "user",
        }
    ]
    assert row["task_type"] == "Code-Switched-ASR" and row["subset_for_metrics"] == "hindi-english"
    extra = row["extra_fields"]
    assert extra["official_word_mapping"] == "hin2engwords.txt" and extra["speaker_id"] == "s1"
    assert extra["source_archive_member"] == "subtask2_blindtest_wReadme/hindi/recA.wav"
    assert extra["segment_start"] == "1.0" and extra["segment_end"] == "2.5"
    with pytest.raises(ValueError, match="not sample-aligned"):
        mucs.format_source_row(language, {**segment, "end": "2.0000001"}, "/mnt")


def test_prepare_defaults_to_both_languages(offline_archive: Path, tmp_path: Path) -> None:
    output = mucs.prepare()
    assert output == mucs.DATA_DIR / "mucs2021_benchmark.jsonl"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["id"] for row in rows] == ["h_1", "h_2", "h_3", "b_1"]
    assert [row["sample_id"] for row in rows] == [
        "mucs2021.hindi-english:1",
        "mucs2021.hindi-english:2",
        "mucs2021.hindi-english:3",
        "mucs2021.bengali-english:1",
    ]
    for row in rows:
        audio_path = Path(row["responses_create_params"]["metadata"]["audio_path"])
        assert audio_path.is_file() and audio_path.stem == row["id"]
        assert row["ng_source_row_sha256"] == audio_manifest.source_hash(row["ng_source_row"])
    first = rows[0]
    (message,) = first["responses_create_params"]["input"]
    assert message["role"] == "user" and message["content"].startswith("Transcribe the audio word for word")
    assert first["ng_source_row"]["messages"][0]["content"] == "Transcribe the following audio."

    import soundfile

    assert soundfile.info(first["responses_create_params"]["metadata"]["audio_path"]).frames == 8000
    assert soundfile.info(rows[2]["responses_create_params"]["metadata"]["audio_path"]).frames == 16000
    assert (tmp_path / "assets" / "ben2engwords.txt").is_file()

    assert mucs.is_prepared_data_current(output)
    assert not mucs.is_prepared_data_current(mucs.DATA_DIR / "other.jsonl")


def test_prepare_narrowing_and_row_cap(offline_archive: Path) -> None:
    output = mucs.prepare(languages="bengali-english", audio_root="/data/mucs2021")
    (row,) = [json.loads(line) for line in output.read_text().splitlines()]
    assert row["id"] == "b_1"
    assert row["responses_create_params"]["metadata"]["audio_path"] == "/data/mucs2021/bengali-english/audio/b_1.wav"
    assert (mucs.DATA_DIR / "bengali-english" / "audio" / "b_1.wav").is_file()
    assert not (mucs.DATA_DIR / "hindi-english" / "audio").exists(), "languages outside the selection are not cut"
    assert not mucs.is_prepared_data_current(output), "narrowed or remapped data must be prepared again by default"

    output = mucs.prepare(max_samples_per_profile=1)
    assert [json.loads(line)["id"] for line in output.read_text().splitlines()] == ["h_1", "b_1"]
    assert not (mucs.DATA_DIR / "hindi-english" / "audio" / "h_3.wav").exists(), "only the capped segments are cut"
    assert not mucs.is_prepared_data_current(output)

    with pytest.raises(ValueError, match="Empty or unknown"):
        mucs.prepare(languages="xx")
    for bad_cap in (0, -1):
        with pytest.raises(ValueError, match="positive integer"):
            mucs.prepare(max_samples_per_profile=bad_cap)
    assert mucs.prepare(languages=["hindi-english", "bengali-english"]).is_file()
    assert mucs.is_prepared_data_current(mucs.OUTPUT_FPATH)
