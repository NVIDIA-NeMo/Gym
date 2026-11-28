# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the CoVoST 2 benchmark: declaration, language and pair selection, rows, prepare()."""

import json
from functools import partial
from pathlib import Path

import pytest

from benchmarks import audio_manifest
from benchmarks.covost2 import prepare as cv
from nemo_gym.environment.benchmarks import BenchmarkConfig
from nemo_gym.environment.manifest import load_manifest
from tests.unit_tests.benchmarks.audio_benchmark_helpers import pcm_wav_bytes


USES_AUDIO_LIBRARIES = True  # prepare() decodes audio: use the stdlib stand-ins for soundfile and soxr

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "covost2"


def test_declaration_specifics() -> None:
    dataset = BenchmarkConfig.from_config_path(BENCHMARK / "config.yaml").dataset
    manifest = load_manifest(BENCHMARK / "manifest.yaml")
    assert {"soundfile", "soxr"} <= set(dataset.prepare_dependencies)
    assert manifest.canonical_split == "test" and manifest.prompt_source == "prepared" and manifest.experimental
    assert manifest.licensing == "CC0-1.0"


def test_profile_contracts() -> None:
    profiles = audio_manifest.load_profiles(cv.PROFILES)["profiles"]
    assert list(profiles) == ["covost2.asr", "covost2.st"]
    for contract in profiles.values():
        assert contract["skills"]["METRICS_TYPE"] == "audio"
        assert audio_manifest.request_controls(contract)["max_output_tokens"] == 256
    assert profiles["covost2.asr"]["prompt_override"]["user_prompt"].startswith(
        "Audio transcription. Language: as spoken."
    )
    assert profiles["covost2.asr"]["prompt_override"]["system_prompt"] == ""
    assert profiles["covost2.st"]["prompt_override"] == {"system_prompt": ""}  # the source prompt names the language


def test_language_and_pair_selection() -> None:
    assert cv.select_languages(None) == cv.ALL_LANGUAGES and len(cv.ALL_LANGUAGES) == 22
    assert cv.select_languages("card") == cv.CARD_LANGUAGES and len(cv.CARD_LANGUAGES) == 17
    assert cv.select_languages("sv-SE, de,de") == ("de", "sv-SE")
    assert cv.select_languages(["zh-CN", "ar"]) == ("ar", "zh-CN")
    for bad in ("xx", "de,xx", "", []):
        with pytest.raises(ValueError, match="Empty or unknown"):
            cv.select_languages(bad)

    assert len(cv.translation_pairs(cv.ALL_LANGUAGES)) == 36
    card_pairs = cv.translation_pairs(cv.CARD_LANGUAGES)
    assert len(card_pairs) == 26 and "en_fa" not in card_pairs and "ca_en" not in card_pairs
    assert cv.translation_pairs(["de", "ja"]) == []  # both ends must be selected, and every pair involves English
    assert cv.translation_pairs(["en", "sv-SE"]) == ["en_sv-SE", "sv-SE_en"]
    assert [cv.pair_languages(pair) for pair in ("de_en", "en_de", "sv-SE_en", "en_zh-CN")] == [
        ("de", "en"),
        ("en", "de"),
        ("sv-SE", "en"),
        ("en", "zh-CN"),
    ]
    assert cv.audio_pair("en") == "en_de" and cv.audio_pair("fr") == "fr_en"


def test_card_subset_matches_the_validated_row_counts() -> None:
    rows = cv.PAIR_TEST_ROWS
    assert sum(rows[cv.audio_pair(language)] for language in cv.CARD_LANGUAGES) == 92901
    assert sum(rows[pair] for pair in cv.translation_pairs(cv.CARD_LANGUAGES)) == 232680
    assert set(rows) == {f"{lang}_en" for lang in cv.ALL_LANGUAGES if lang != "en"} | {
        f"en_{lang}" for lang in "ar ca cy de et fa id ja lv mn sl sv-SE ta tr zh-CN".split()
    }


_wav_bytes = partial(pcm_wav_bytes, rate=48000, sample=b"\x10\x00")


def test_decode_clip_resamples_to_16_khz_mono() -> None:
    assert len(cv.decode_clip(_wav_bytes(48000))) == 16000
    assert len(cv.decode_clip(_wav_bytes(16000, rate=16000))) == 16000
    with pytest.raises(ValueError, match="mono"):
        cv.decode_clip(_wav_bytes(1000, channels=2))


def test_format_rows_keep_the_validated_schema() -> None:
    clip = {"id": "common_voice_de_1", "sentence": "Hallo Welt.", "translation": "Hello world."}
    asr = cv.format_asr_row("zh-CN", clip, 2.5, "/mnt/audio")
    assert asr["expected_answer"] == "Hallo Welt." and asr["subset_for_metrics"] == "zh-CN"
    assert (
        asr["audio_path"] == "/mnt/audio/zh-CN/test/common_voice_de_1.wav" and asr["task_type"] == "Multilingual-ASR"
    )
    assert asr["messages"] == [
        {"role": "system", "content": "You are a helpful assistant. /no_think"},
        {
            "role": "user",
            "content": "Transcribe the following audio.",
            "audio": {"path": asr["audio_path"], "duration": 2.5},
        },
    ]
    assert asr["extra_fields"] == {
        "src_text": "Hallo Welt.",
        "src_lang_name": "Chinese",
        "src_lang": "zh-CN",
        "use_cer": True,
    }

    st = cv.format_st_row("de", "sv-SE", clip, 1.0, "/mnt/audio")
    assert st["expected_answer"] == "Hello world." and "source" not in st and "reference" not in st
    assert st["subset_for_metrics"] == "de->sv-SE" and st["task_type"] == "Multilingual-AST"
    assert st["messages"][1]["content"] == "Please translate the given speech to Swedish."
    assert st["extra_fields"] == {
        "src_text": "Hallo Welt.",
        "tgt_text": "Hello world.",
        "src_lang_name": "German",
        "tgt_lang_name": "Swedish",
        "src_lang": "de",
        "tgt_lang": "sv-SE",
    }


def test_only_chinese_and_japanese_are_scored_by_character_error_rate() -> None:
    clip = {"id": "x", "sentence": "s", "translation": "t"}
    use_cer = {
        language: cv.format_asr_row(language, clip, 1.0, "/a")["extra_fields"]["use_cer"]
        for language in cv.ALL_LANGUAGES
    }
    assert {language for language, flag in use_cer.items() if flag} == {"ja", "zh-CN"}


def test_the_source_is_pinned_to_one_revision() -> None:
    assert len(cv.HF_REVISION) == 40 and set(cv.HF_REVISION) <= set("0123456789abcdef")


def test_read_pair_rejects_duplicate_clip_ids(offline_covost: Path, monkeypatch) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    duplicated = offline_covost / "dup.parquet"
    table = pa.table({"id": ["a", "a"], "sentence": ["x", "y"], "translation": ["x", "y"]})
    pq.write_table(table, duplicated)
    monkeypatch.setattr(cv, "list_shards", lambda pair: [str(duplicated)])
    monkeypatch.setitem(cv.PAIR_TEST_ROWS, "de_en", 2)
    with pytest.raises(ValueError, match="unique clips"):
        cv.read_pair("de_en")


def test_extract_language_reports_clips_missing_from_the_source(offline_covost: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="clips missing from the source: \\['cv_de_404'\\]"):
        cv.extract_language("de", {"cv_de_1", "cv_de_404"}, tmp_path / "audio")


def test_capped_keeps_the_first_rows_of_every_subset() -> None:
    items = [("a", 1), ("a", 2), ("a", 3), ("b", 4), ("b", 5)]
    assert cv._capped(items, None, None) == items
    assert cv._capped(items, 2, None) == [("a", 1), ("a", 2), ("b", 4), ("b", 5)]
    assert cv._capped(items, 1, 1) == [("a", 1)]
    assert cv._capped(items, None, 4) == items[:4]


@pytest.fixture
def offline_covost(tmp_path: Path, monkeypatch) -> Path:
    """A three-language mirror: de->en (3 clips), en->de (2), ja->en (2), en->ja (2); two shards per pair."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    remote = tmp_path / "remote"
    remote.mkdir()
    clips = {  # pair -> (id, sentence, translation, seconds)
        "de_en": [("cv_de_2", "zwei", "two", 1), ("cv_de_1", "eins", "one", 2), ("cv_de_3", "drei", "three", 3)],
        "en_de": [("cv_en_2", "two", "zwei", 1), ("cv_en_1", "one", "eins", 2)],
        "ja_en": [("cv_ja_1", "ichi", "one", 1), ("cv_ja_2", "ni", "two", 2)],
        "en_ja": [("cv_en_2", "two", "ni", 1), ("cv_en_1", "one", "ichi", 2)],
    }
    shards: dict[str, list[str]] = {}
    for pair, items in clips.items():
        shards[pair] = []
        for number, chunk in enumerate([items[:1], items[1:]]):
            table = pa.table(
                {
                    "id": [item[0] for item in chunk],
                    "sentence": [item[1] for item in chunk],
                    "translation": [item[2] for item in chunk],
                    "audio": [{"bytes": _wav_bytes(48000 * item[3]), "path": item[0] + ".mp3"} for item in chunk],
                }
            )
            path = remote / f"{pair}-test-{number}.parquet"
            pq.write_table(table, path, row_group_size=1)
            shards[pair].append(str(path))
    monkeypatch.setattr(cv, "PAIR_TEST_ROWS", {pair: len(items) for pair, items in clips.items()})
    monkeypatch.setattr(cv, "ALL_LANGUAGES", ("de", "en", "ja"))
    monkeypatch.setattr(cv, "list_shards", lambda pair: shards[pair])
    monkeypatch.setattr(cv, "open_shard", lambda path: open(path, "rb"))
    monkeypatch.setattr(cv, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(cv, "OUTPUT_FPATH", tmp_path / "data" / "covost2_benchmark.jsonl")
    return remote


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_read_pair_checks_the_census(offline_covost: Path, monkeypatch) -> None:
    assert [row["id"] for row in cv.read_pair("de_en")] == ["cv_de_1", "cv_de_2", "cv_de_3"]
    monkeypatch.setitem(cv.PAIR_TEST_ROWS, "de_en", 4)
    with pytest.raises(ValueError, match="expected 4 unique clips"):
        cv.read_pair("de_en")


def test_prepare_defaults_to_every_language_and_pair(offline_covost: Path) -> None:
    output = cv.prepare()
    rows = _read(output)
    asr = [row for row in rows if row["benchmark_profile"] == "covost2.asr"]
    st = [row for row in rows if row["benchmark_profile"] == "covost2.st"]
    assert [(row["subset_for_metrics"], row["audio_path"].rsplit("/", 1)[1]) for row in asr] == [
        ("de", "cv_de_1.wav"),
        ("de", "cv_de_2.wav"),
        ("de", "cv_de_3.wav"),
        ("en", "cv_en_1.wav"),
        ("en", "cv_en_2.wav"),
        ("ja", "cv_ja_1.wav"),
        ("ja", "cv_ja_2.wav"),
    ], "ASR rows are ordered by language and clip file name"
    assert [(row["subset_for_metrics"], row["audio_path"].rsplit("/", 1)[1]) for row in st] == [
        ("de->en", "cv_de_1.wav"),
        ("de->en", "cv_de_2.wav"),
        ("de->en", "cv_de_3.wav"),
        ("en->de", "cv_en_1.wav"),
        ("en->de", "cv_en_2.wav"),
        ("en->ja", "cv_en_1.wav"),
        ("en->ja", "cv_en_2.wav"),
        ("ja->en", "cv_ja_1.wav"),
        ("ja->en", "cv_ja_2.wav"),
    ], "translation rows are ordered by pair and clip file name, whatever the order of the source shards"
    assert [row["sample_id"] for row in asr][:2] == ["covost2.asr:1", "covost2.asr:2"]
    assert [row["sample_id"] for row in st][-1] == "covost2.st:9"

    first = asr[0]
    audio_path = Path(first["responses_create_params"]["metadata"]["audio_path"])
    assert audio_path == (cv.DATA_DIR / "audio" / "de" / "test" / "cv_de_1.wav").resolve() and audio_path.is_file()
    assert first["duration"] == 2.0 and first["expected_answer"] == "eins"
    (message,) = first["responses_create_params"]["input"]
    assert message["content"].startswith("Audio transcription. Language: as spoken.")
    assert first["ng_source_row"]["messages"][1]["content"] == "Transcribe the following audio."
    assert first["ng_source_row_sha256"] == audio_manifest.source_hash(first["ng_source_row"])
    assert st[0]["responses_create_params"]["input"][0]["content"] == "Please translate the given speech to English."
    assert st[0]["expected_answer"] == "one" and st[0]["extra_fields"]["src_text"] == "eins"
    assert (cv.DATA_DIR / "audio" / "en" / "test" / "cv_en_1.wav").is_file()
    assert cv.is_prepared_data_current(output)
    assert not cv.is_prepared_data_current(cv.DATA_DIR / "other.jsonl")


def test_prepare_narrows_languages(offline_covost: Path) -> None:
    output = cv.prepare(languages="de,en")
    rows = _read(output)
    assert {(row["benchmark_profile"], row["subset_for_metrics"]) for row in rows} == {
        ("covost2.asr", "de"),
        ("covost2.asr", "en"),
        ("covost2.st", "de->en"),
        ("covost2.st", "en->de"),
    }
    assert not (cv.DATA_DIR / "audio" / "ja").exists(), "languages outside the selection are not extracted"
    assert not cv.is_prepared_data_current(output), "narrowed data must be prepared again by default"
    only_japanese = _read(cv.prepare(languages=["ja"]))
    assert {row["benchmark_profile"] for row in only_japanese} == {"covost2.asr"}, "no pair without English selected"
    with pytest.raises(ValueError, match="Empty or unknown"):
        cv.prepare(languages="xx")


@pytest.mark.parametrize("name", ["max_samples_per_profile", "max_samples_per_subset"])
@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "5"])
def test_prepare_rejects_invalid_row_caps_before_downloading(name: str, bad, monkeypatch) -> None:
    monkeypatch.setattr(cv, "read_pair", lambda pair: pytest.fail("a bad cap must fail before any download"))
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        cv.prepare(**{name: bad})


def test_the_server_accepts_the_profiles(tmp_path: Path) -> None:
    """The audio_eval server resolves every profile's evaluator, metrics and settings when it starts."""
    pytest.importorskip("sacrebleu", reason="the scoring libraries are installed in the audio_eval server environment")
    from unittest.mock import MagicMock

    from nemo_gym.client import ServerClient
    from resources_servers.audio_eval.app import AudioEvalConfig, AudioEvalResourcesServer

    server = AudioEvalResourcesServer(
        config=AudioEvalConfig(
            name="audio_eval", host="127.0.0.1", port=1, entrypoint="app.py", profile_contracts=str(cv.PROFILES)
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    assert sorted(server._profiles) == ["covost2.asr", "covost2.st"]


def test_prepare_caps_and_audio_root(offline_covost: Path) -> None:
    rows = _read(cv.prepare(audio_root="/data/covost2", max_samples_per_subset=1))
    assert [(row["benchmark_profile"], row["subset_for_metrics"]) for row in rows] == [
        ("covost2.asr", "de"),
        ("covost2.asr", "en"),
        ("covost2.asr", "ja"),
        ("covost2.st", "de->en"),
        ("covost2.st", "en->de"),
        ("covost2.st", "en->ja"),
        ("covost2.st", "ja->en"),
    ]
    assert rows[0]["responses_create_params"]["metadata"]["audio_path"] == "/data/covost2/audio/de/test/cv_de_1.wav"
    assert not (cv.DATA_DIR / "audio" / "de" / "test" / "cv_de_3.wav").exists(), "only the kept clips are written"
    assert not cv.is_prepared_data_current(cv.OUTPUT_FPATH)
    rows = _read(cv.prepare(max_samples_per_profile=2))
    assert [row["benchmark_profile"] for row in rows] == ["covost2.asr"] * 2 + ["covost2.st"] * 2
