# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for Preference-ASR: declaration, discovery, census, row building, audio, prepare()."""

import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from benchmarks import audio_manifest
from benchmarks.preference_asr import prepare as pref
from nemo_gym.environment.benchmarks import BenchmarkConfig
from nemo_gym.environment.manifest import load_manifest


USES_AUDIO_LIBRARIES = True
ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "preference_asr"

CV_ID = "ab12"  # a Common Voice clip id of the Open ASR leaderboard files


def test_declaration_specifics() -> None:
    dataset = BenchmarkConfig.from_config_path(BENCHMARK / "config.yaml").dataset
    manifest = load_manifest(BENCHMARK / "manifest.yaml")
    assert "soundfile" in dataset.prepare_dependencies
    assert manifest.canonical_split == "test" and manifest.prompt_source == "prepared" and manifest.experimental
    assert manifest.licensing == "CC-BY-NC-4.0"


def test_profiles_keep_the_scorer_contract() -> None:
    profiles = audio_manifest.load_profiles(pref.PROFILES)["profiles"]
    assert list(profiles) == [pref.PROFILE_PREFIX + variant for variant in pref.VARIANTS]
    for contract in profiles.values():
        assert contract["prompt_override"] == {"system_prompt": ""}
        assert contract["skills"]["METRICS_TYPE"].endswith("PreferenceASRMetrics")
        assert contract["skills"]["EVAL_ARGS"].split()[0].endswith("PreferenceASREvaluator")
        assert audio_manifest.request_controls(contract)["max_output_tokens"] == 256
    assert sum(pref.EXPECTED_CORPUS_ROWS.values()) == sum(pref.EXPECTED_CATEGORY_ROWS.values()) == pref.EXPECTED_ROWS
    assert sum(pref.EXPECTED_DIRECTION_ROWS.values()) == pref.EXPECTED_ROWS == 2822


def test_narrowing_arguments_default_to_all() -> None:
    assert audio_manifest.select(None, pref.CORPORA, "corpora") == pref.CORPORA
    assert audio_manifest.select("voxpopuli, ami", pref.CORPORA, "corpora") == ("ami", "voxpopuli")
    assert audio_manifest.select(["entities"], pref.CATEGORIES, "categories") == ("entities",)
    for bad in ("nope", "ami,nope", "", []):
        with pytest.raises(ValueError, match="Empty or unknown"):
            audio_manifest.select(bad, pref.CORPORA, "corpora")


def _manifest_row(corpus: str, name: str, category: str = "standard", **overrides) -> dict:
    row = {
        "audio_filepath": f"data/{corpus}/{name}",
        "duration": 0.5,
        "text": "Three o'clock.",
        "preference_text": "3 o'clock.",
        "instruction": "Transcribe the speech in the input audio. Use digits.",
        "preference_type": category,
        "sub_category": ["numbers"],
        "dataset": corpus,
    }
    return row | overrides


def test_instruction_rewrite_changes_only_the_two_openings() -> None:
    assert pref.rewrite_instruction("Transcribe the speech in the input English audio") == (
        "Transcribe this English recording."
    )
    assert pref.rewrite_instruction("Transcribe the speech in the input audio. Use digits.") == (
        "Transcribe this recording. Use digits."
    )
    assert (
        pref.rewrite_instruction("Perform speech recognition. Use digits.")
        == "Perform speech recognition. Use digits."
    )


def test_source_rows_follow_the_validated_layout() -> None:
    row = _manifest_row("ami", "a.flac", "case", sub_category=["lower_case"], direction="tn")
    default = pref.format_source_row(row, 7, "default", "/r/audio")
    instructed = pref.format_source_row(row, 7, "instructed", "/r/audio")
    assert default["id"] == "preference-asr-public-default-0007"
    assert default["messages"][0]["content"] == pref.DEFAULT_PROMPT
    assert instructed["messages"][0]["content"] == "Transcribe this recording. Use digits."
    assert default["messages"][0]["audio"] == {"path": "/r/audio/ami/a.wav", "duration": 0.5}
    assert default["expected_answer"] == "3 o'clock." and default["standard_reference"] == "Three o'clock."
    assert default["target_case"] == "lower_case" and default["direction"] == "tn"
    assert default["subset_for_metrics"] == "case" and default["preference_prompt_variant"] == "default"
    assert instructed["source_instruction"] == row["instruction"] and instructed["prompt_policy"] == pref.PROMPT_POLICY
    assert default["extra_fields"]["source_audio_filepath"] == "data/ami/a.flac"
    with pytest.raises(ValueError, match="ambiguous target"):
        pref.format_source_row(_manifest_row("ami", "a.flac", "case"), 1, "default", "/r")


def test_audio_parts_reject_unsafe_paths() -> None:
    assert pref.audio_parts(_manifest_row("ami", "a_b-1.flac")) == ("ami", "a_b-1.flac")
    for path in ("/abs/ami/a.flac", "data/ami/sub/a.flac", "data/voxpopuli/a.flac", "data/ami/a b.flac"):
        with pytest.raises(ValueError, match="Unsafe|mismatch"):
            pref.audio_parts(_manifest_row("ami", "a.flac", audio_filepath=path))


def test_esb_clip_names_match_the_manifest_spelling() -> None:
    assert pref.esb_clip_name("ab12") == "ab12.flac"
    assert pref.esb_clip_name("dir/ab12.wav") == "dir_ab12.flac"


def _tone(rate: int, seconds: float = 0.5) -> np.ndarray:
    return (0.25 * np.sin(2 * np.pi * 440 * np.arange(int(rate * seconds)) / rate)).astype(np.float32)


def _encoded(rate: int = 16000, fmt: str = "WAV", seconds: float = 0.5) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, _tone(rate, seconds), rate, format=fmt)
    return buffer.getvalue()


def test_canonical_samples_resample_and_validate() -> None:
    assert pref.canonical_samples(_encoded(8000)).shape == (8000,)
    assert pref.canonical_samples(_encoded(16000, "FLAC")).shape == (8000,)
    with pytest.raises(ValueError, match="too short"):
        pref.canonical_samples(_encoded(16000, seconds=0.001))


def test_check_wav_enforces_format_and_duration(tmp_path: Path) -> None:
    path = tmp_path / "x.wav"
    pref.write_wav(path, _tone(16000))
    pref.check_wav(path, 0.5)
    with pytest.raises(ValueError, match="duration drift"):
        pref.check_wav(path, 2.0)
    with pytest.raises(ValueError, match="missing or unreadable"):
        pref.check_wav(tmp_path / "none.wav", 0.5)
    (tmp_path / "wrong.wav").write_bytes(_encoded(8000))
    with pytest.raises(ValueError, match="contract failed"):
        pref.check_wav(tmp_path / "wrong.wav", 0.5)


MANIFEST = [
    _manifest_row("ami", "a1.flac", "standard"),
    _manifest_row("ami", "a1.flac", "case", sub_category=["lower_case"]),
    _manifest_row("librispeech", "l1.flac", "entities", sub_category=["person"]),
    _manifest_row("voxpopuli", "v1.wav", "normalization", direction="itn"),
    _manifest_row("common_voice", f"{CV_ID}.flac", "disfluencies", preference_text="", direction="remove"),
]


class _FakeFileSystem:
    """Serves one local parquet shard under the Hugging Face layout the Common Voice reader globs."""

    def __init__(self, shard: Path) -> None:
        self.shard = shard
        self.opened = 0

    def glob(self, pattern: str) -> list[str]:
        assert f"{pref.ESB_REPO}@{pref.ESB_REVISION}/common_voice/test-*.parquet" in pattern
        return [str(self.shard)]

    def open(self, path: str, mode: str):
        self.opened += 1
        return open(path, mode)


@pytest.fixture
def offline_prepare(tmp_path: Path, monkeypatch) -> Path:
    """Run prepare() on a five-row manifest with synthetic clips and a synthetic Common Voice shard."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in MANIFEST))
    clips = {}
    for name in ("data/ami/a1.flac", "data/librispeech/l1.flac", "data/voxpopuli/v1.wav"):
        clips[name] = tmp_path / Path(name).name
        clips[name].write_bytes(_encoded(16000, "FLAC" if name.endswith("flac") else "WAV"))
    shard = tmp_path / "cv.parquet"
    ids = ["zz", CV_ID, CV_ID]
    audio = [{"bytes": _encoded(16000 if i == CV_ID else 8000), "path": None} for i in ids]
    pq.write_table(pa.table({"id": ids, "audio": audio}), shard, row_group_size=2)

    def fake_download(repo, filename, **kwargs):
        assert repo == pref.HF_RELEASE_REPO and kwargs["revision"] == pref.HF_RELEASE_REVISION
        return str(manifest if filename == pref.MANIFEST_FILE else clips[filename])

    monkeypatch.setattr(pref, "hf_hub_download", fake_download)
    monkeypatch.setattr(pref, "HfFileSystem", lambda: _FakeFileSystem(shard))
    monkeypatch.setattr(pref, "MANIFEST_SHA256", audio_manifest.sha256_file(manifest))
    monkeypatch.setattr(pref, "EXPECTED_ROWS", 5)
    monkeypatch.setattr(pref, "EXPECTED_CLIPS", 4)
    monkeypatch.setattr(pref, "EXPECTED_DURATION_SECONDS", 2.5)
    monkeypatch.setattr(pref, "EXPECTED_EMPTY_PREFERENCE_TEXT", 1)
    monkeypatch.setattr(pref, "EXPECTED_CORPUS_ROWS", dict(Counter(row["dataset"] for row in MANIFEST)))
    monkeypatch.setattr(pref, "EXPECTED_CATEGORY_ROWS", dict(Counter(row["preference_type"] for row in MANIFEST)))
    monkeypatch.setattr(pref, "EXPECTED_DIRECTION_ROWS", dict(Counter(row.get("direction") for row in MANIFEST)))
    data_dir = tmp_path / "data"
    monkeypatch.setattr(pref, "DATA_DIR", data_dir)
    monkeypatch.setattr(pref, "OUTPUT_FPATH", data_dir / "preference_asr_benchmark.jsonl")
    return data_dir


def test_prepare_writes_every_row_in_both_conditions(offline_prepare: Path) -> None:
    output = pref.prepare()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 10
    assert [row["benchmark_profile"] for row in rows[::5]] == [
        "preference-asr-public.default",
        "preference-asr-public.instructed",
    ]
    assert [row["sample_id"] for row in rows[:5]] == [f"preference-asr-public.default:{i}" for i in range(1, 6)]
    assert [row["id"] for row in rows[5:7]] == [
        "preference-asr-public-instructed-0001",
        "preference-asr-public-instructed-0002",
    ]
    for row in rows:
        assert Path(row["responses_create_params"]["metadata"]["audio_path"]).is_file()
        assert row["ng_source_row_sha256"] == audio_manifest.source_hash(row["ng_source_row"])
    audio = {p.relative_to(offline_prepare / "audio").as_posix() for p in (offline_prepare / "audio").rglob("*.wav")}
    assert audio == {"ami/a1.wav", "librispeech/l1.wav", "voxpopuli/v1.wav", f"common_voice/{CV_ID}.wav"}
    assert rows[0]["responses_create_params"]["input"][0]["content"] == "Transcribe the English audio into text"
    assert rows[5]["responses_create_params"]["input"][0]["content"] == "Transcribe this recording. Use digits."
    assert rows[4]["expected_answer"] == "" and rows[4]["preference_prompt_variant"] == "default"
    assert pref.is_prepared_data_current(output) and not pref.is_prepared_data_current(offline_prepare / "other.jsonl")


def test_prepare_narrowing_audio_root_and_row_cap(offline_prepare: Path) -> None:
    output = pref.prepare(
        corpora="ami,voxpopuli", variants="instructed", audio_root="/data/pa", max_samples_per_profile=2
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["preference_source_dataset"] for row in rows] == ["ami", "ami"]
    assert {row["benchmark_profile"] for row in rows} == {"preference-asr-public.instructed"}
    assert rows[0]["responses_create_params"]["metadata"]["audio_path"] == "/data/pa/audio/ami/a1.wav"
    assert not (offline_prepare / "audio" / "common_voice").exists(), "unselected corpora are not fetched"
    assert not pref.is_prepared_data_current(output)

    output = pref.prepare(corpora="voxpopuli", categories="normalization")
    row = json.loads(output.read_text().splitlines()[0])
    assert row["id"] == "preference-asr-public-default-0004", "ids keep the position in the full manifest"
    with pytest.raises(ValueError, match="No Preference-ASR rows"):
        pref.prepare(corpora="voxpopuli", categories="case")


def test_missing_common_voice_clip_is_reported(offline_prepare: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing from"):
        pref.extract_common_voice({"nope.flac": offline_prepare / "nope.wav"})


@pytest.mark.parametrize("cap", [0, -1, True, 1.5, "2"])
def test_prepare_rejects_an_invalid_row_cap_before_any_download(cap, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("prepare() must validate its arguments before downloading")

    monkeypatch.setattr(pref, "hf_hub_download", forbidden)
    with pytest.raises(ValueError, match="positive integer"):
        pref.prepare(max_samples_per_profile=cap)


def test_prepare_rejects_a_drifted_manifest(offline_prepare: Path, monkeypatch) -> None:
    monkeypatch.setattr(pref, "EXPECTED_ROWS", 6)
    with pytest.raises(ValueError, match="row-count drift"):
        pref.load_manifest()
    monkeypatch.setattr(pref, "EXPECTED_ROWS", 5)
    monkeypatch.setattr(pref, "EXPECTED_EMPTY_PREFERENCE_TEXT", 0)
    with pytest.raises(ValueError, match="census drift"):
        pref.load_manifest()
    monkeypatch.setattr(pref, "MANIFEST_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="hash drift"):
        pref.load_manifest()
