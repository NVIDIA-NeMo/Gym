# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the VoxLingua107 benchmark: declaration, discovery, archive census, row building, prepare()."""

import json
import zipfile
from pathlib import Path

import pytest

from benchmarks import audio_manifest
from benchmarks.voxlingua107 import prepare as vox
from nemo_gym.environment.benchmarks import BenchmarkConfig
from nemo_gym.environment.manifest import load_manifest
from tests.unit_tests.benchmarks.audio_benchmark_helpers import assert_capped_and_remapped_rows, pcm_wav_bytes


USES_AUDIO_LIBRARIES = True  # prepare() reads WAV durations: use the stdlib stand-ins of the audio libraries
ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "voxlingua107"


def test_declaration_specifics() -> None:
    dataset = BenchmarkConfig.from_config_path(BENCHMARK / "config.yaml").dataset
    manifest = load_manifest(BENCHMARK / "manifest.yaml")
    assert "soundfile" in dataset.prepare_dependencies
    assert manifest.canonical_split == "dev" and manifest.prompt_source == "prepared" and manifest.experimental
    assert manifest.licensing == "CC-BY-4.0"


def test_profile_contract_resolves_and_matches_the_source_prompt() -> None:
    contract = audio_manifest.load_profiles(vox.PROFILES)["profiles"]["voxlingua107"]
    assert contract["skills"]["METRICS_TYPE"] == "language_id"
    # The label list is generated from the selected languages, so the profile only drops the system prompt.
    assert contract["prompt_override"] == {"system_prompt": ""}
    assert vox.EXPECTED_ROWS == sum(vox.ARCHIVE_LABEL_COUNTS.values()) == 1609 and len(vox.ARCHIVE_LABEL_COUNTS) == 33
    assert set(vox.LANGUAGES_24) <= set(vox.ARCHIVE_LABEL_COUNTS) and len(vox.LANGUAGES_24) == 24
    assert sum(vox.ARCHIVE_LABEL_COUNTS[label] for label in vox.LANGUAGES_24) == 1259


def test_select_languages_defaults_to_all_and_accepts_lists_and_strings() -> None:
    assert (
        vox.select_languages(None) == tuple(sorted(vox.ARCHIVE_LABEL_COUNTS)) and len(vox.select_languages(None)) == 33
    )
    assert vox.select_languages("fr, de,fr") == ("de", "fr")
    assert vox.select_languages(["zh", "ar"]) == ("ar", "zh")
    assert vox.select_languages(vox.LANGUAGES_24) == vox.LANGUAGES_24
    for bad in ("xx", "de,xx", "", []):
        with pytest.raises(ValueError, match="Empty or unknown"):
            vox.select_languages(bad)


def test_24_language_subset_prompt_lists_exactly_those_codes() -> None:
    assert vox.language_prompt(vox.LANGUAGES_24) == (
        "Identify the language spoken in this audio. Respond with exactly one of the following language labels "
        "and nothing else: ar, da, de, el, en, es, et, fi, fr, hr, hu, it, ja, lt, lv, nl, pl, pt, ru, sl, sv, tr, "
        "uk, zh."
    )
    assert vox.language_prompt(("de", "fr")).endswith("nothing else: de, fr.")


@pytest.fixture
def offline_archive(tmp_path: Path, monkeypatch) -> Path:
    """A tiny dev.zip: three languages, adapted census constants."""
    members = {
        "fr/b-clip.wav": 1.0,
        "fr/a-clip.wav": 2.0,
        "de/c-clip.wav": 1.5,
        "az/z-clip.wav": 1.0,
    }
    archive_path = tmp_path / "remote" / "dev.zip"
    archive_path.parent.mkdir()
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, seconds in members.items():
            archive.writestr(name, pcm_wav_bytes(int(seconds * 16000)))
    monkeypatch.setattr(vox, "SOURCE_URL", archive_path.as_uri())
    monkeypatch.setattr(vox, "SOURCE_SHA256", audio_manifest.sha256_file(archive_path))
    monkeypatch.setattr(vox, "ARCHIVE_LABEL_COUNTS", {"fr": 2, "de": 1, "az": 1})
    monkeypatch.setattr(vox, "EXPECTED_ROWS", 4)
    data_dir = tmp_path / "data"
    monkeypatch.setattr(vox, "DATA_DIR", data_dir)
    monkeypatch.setattr(vox, "OUTPUT_FPATH", data_dir / "voxlingua107_benchmark.jsonl")
    return archive_path


def test_list_clips_checks_the_census_and_orders_by_label_and_name(offline_archive: Path, monkeypatch) -> None:
    with zipfile.ZipFile(offline_archive) as archive:
        clips = vox.list_clips(archive, ("az", "de", "fr"))
        assert [(label, name) for _, label, name in clips] == [
            ("az", "z-clip.wav"),
            ("de", "c-clip.wav"),
            ("fr", "a-clip.wav"),
            ("fr", "b-clip.wav"),
        ]
        assert [(label, name) for _, label, name in vox.list_clips(archive, ("fr",))] == [
            ("fr", "a-clip.wav"),
            ("fr", "b-clip.wav"),
        ]
        monkeypatch.setattr(vox, "ARCHIVE_LABEL_COUNTS", {"fr": 2, "de": 2, "az": 1})
        with pytest.raises(ValueError, match="census drift"):
            vox.list_clips(archive, ("fr",))
        monkeypatch.setattr(vox, "ARCHIVE_LABEL_COUNTS", {"fr": 2, "de": 1, "az": 1})
        monkeypatch.setattr(vox, "EXPECTED_ROWS", 5)
        with pytest.raises(ValueError, match="row-count drift"):
            vox.list_clips(archive, ("fr",))


def test_list_clips_rejects_unexpected_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../evil.wav", b"x")
    with zipfile.ZipFile(archive_path) as archive, pytest.raises(ValueError, match="Unexpected"):
        vox.list_clips(archive, ("fr",))


def test_format_source_row_keeps_the_validated_schema() -> None:
    row = vox.format_source_row("fr", "abc.wav", 4.73, "/mnt/audio", "Pick one: de, fr.")
    assert row["id"] == "fr-abc" and row["expected_answer"] == "fr" and row["subset_for_metrics"] == "fr"
    assert row["audio_path"] == "/mnt/audio/fr/abc.wav" and row["duration"] == 4.73
    assert row["messages"] == [
        {"role": "user", "content": "Pick one: de, fr.", "audio": {"path": "/mnt/audio/fr/abc.wav", "duration": 4.73}}
    ]
    assert row["task_type"] == "VoxLingua107-LID"
    assert row["extra_fields"]["source_split"] == "dev" and row["extra_fields"]["language_label"] == "fr"


def test_prepare_defaults_to_all_languages_and_lists_them_in_the_prompt(offline_archive: Path) -> None:
    output = vox.prepare()
    assert output == vox.DATA_DIR / "voxlingua107_benchmark.jsonl"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["id"] for row in rows] == ["az-z-clip", "de-c-clip", "fr-a-clip", "fr-b-clip"]
    assert [row["sample_id"] for row in rows] == [f"voxlingua107:{i}" for i in range(1, 5)]
    assert {row["benchmark_profile"] for row in rows} == {"voxlingua107"}

    first = rows[0]
    audio_path = Path(first["responses_create_params"]["metadata"]["audio_path"])
    assert audio_path == (vox.DATA_DIR / "audio" / "az" / "z-clip.wav").resolve() and audio_path.is_file()
    assert first["duration"] == 1.0 and first["expected_answer"] == "az"
    (message,) = first["responses_create_params"]["input"]
    assert message["role"] == "user" and message["content"].endswith("nothing else: az, de, fr.")
    assert first["ng_source_row"]["messages"][0]["content"] == message["content"]
    assert first["ng_source_row"]["messages"][0]["audio"]["path"] == str(audio_path)
    assert first["ng_source_row_sha256"] == audio_manifest.source_hash(first["ng_source_row"])

    assert vox.is_prepared_data_current(output)
    assert not vox.is_prepared_data_current(vox.DATA_DIR / "other.jsonl")


def test_prepare_languages_narrow_rows_prompt_and_extraction(offline_archive: Path) -> None:
    for languages in ("fr,de", ["fr", "de"]):
        output = vox.prepare(languages=languages)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert [row["id"] for row in rows] == ["de-c-clip", "fr-a-clip", "fr-b-clip"]
        assert all(row["responses_create_params"]["input"][0]["content"].endswith(": de, fr.") for row in rows)
        assert not vox.is_prepared_data_current(output), "narrowed data must be prepared again by default"
    assert not (vox.DATA_DIR / "audio" / "az").exists(), "languages outside the selection are not extracted"
    with pytest.raises(ValueError, match="Empty or unknown"):
        vox.prepare(languages="xx")


def test_prepare_with_every_language_listed_is_still_the_default(offline_archive: Path) -> None:
    output = vox.prepare(languages=["fr", "de", "az"])
    assert vox.is_prepared_data_current(output)


def test_prepare_audio_root_and_row_cap_override(offline_archive: Path) -> None:
    output = vox.prepare(languages="de,fr", audio_root="/data/voxlingua107", max_samples_per_profile=2)
    assert_capped_and_remapped_rows(vox, output, rows=2, audio_path="/data/voxlingua107/audio/fr/a-clip.wav", index=1)
    assert (vox.DATA_DIR / "audio" / "fr" / "a-clip.wav").is_file()
    assert not (vox.DATA_DIR / "audio" / "fr" / "b-clip.wav").exists(), "only the capped clips are extracted"
