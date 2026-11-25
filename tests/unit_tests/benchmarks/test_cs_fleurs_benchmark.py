# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the CS-FLEURS benchmark: declaration, discovery, row building, download, prepare()."""

import fnmatch
import json
from pathlib import Path

import pytest

from benchmarks import audio_manifest
from benchmarks.cs_fleurs import prepare as cs_fleurs
from nemo_gym.environment.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "cs_fleurs"
PROFILES = ("cs-fleurs.read", "cs-fleurs.mms", "cs-fleurs.xtts-test1", "cs-fleurs.xtts-test2")


def _metadata_row(index: int, language: str = "ara-eng", file_name: str | None = None, **extra) -> dict:
    return {
        "id": f"{language}_{index}",
        "file_name": file_name or f"audio/{language.split('-')[0]}/{language}_{index}.wav",
        "text": f"text {index}",
        "duration": 1.5 + index,
        "language": language,
        "speaker": "spk0",
    } | extra


def test_declaration_specifics() -> None:
    manifest = load_manifest(BENCHMARK / "manifest.yaml")
    assert manifest.canonical_split == "test" and manifest.prompt_source == "prepared" and manifest.experimental
    assert manifest.licensing == "CC-BY-NC-4.0"


def test_profile_contracts_cover_the_four_test_sets() -> None:
    profiles = audio_manifest.load_profiles(cs_fleurs.PROFILES)["profiles"]
    assert set(profiles) == set(PROFILES) == set(cs_fleurs.SUBSETS)
    for contract in profiles.values():
        assert audio_manifest.request_controls(contract)["max_output_tokens"] == 256
        assert contract["skills"]["METRICS_TYPE"] == "audio"
        args = contract["skills"]["EVAL_ARGS"].split()
        assert "++eval_type=audio" in args and "++eval_config.normalization_mode=multilingual" in args
        assert "++eval_config.cleanup_generation_loops=true" in args


def test_language_tables_are_consistent() -> None:
    assert cs_fleurs.GRAPHEME_LANGUAGES <= set(cs_fleurs.LANGUAGES)
    assert cs_fleurs.split_pair("ara-eng") == ("ara", "eng")
    assert cs_fleurs.split_pair("cmn_eng") == ("cmn", "eng")
    assert cs_fleurs.split_pair("hin") == ("hin", "")


def test_format_source_row_derives_the_scoring_fields() -> None:
    row = _metadata_row(0, "ara-eng", file_name="audio/ara/ara_1_BT.wav", fluency="1")
    source = cs_fleurs.format_source_row(row, "cs-fleurs.read", "/mnt/raw")
    assert source["audio_path"] == "/mnt/raw/read/test/audio/ara/ara_1_BT.wav"
    assert source["expected_answer"] == "text 0" and source["subset_for_metrics"] == "ara-eng"
    assert source["duration"] == 1.5 and source["task_type"] == "Multilingual-ASR"
    assert source["extra_fields"] == {
        "src_text": "text 0",
        "src_lang": "ar",
        "matrix_lang": "ara",
        "matrix_lang_name": "Arabic",
        "embedded_lang": "eng",
        "embedded_lang_name": "English",
        "lang_pair": "ara-eng",
        "use_mer": True,
        "mer_grapheme": False,
        "speaker": "spk0",
        "fluency": "1",
    }
    assert source["source_provenance"]["subset"] == "read"
    assert source["source_provenance"]["metadata_sha256"] == cs_fleurs.SUBSETS["cs-fleurs.read"][3]
    assert [m["role"] for m in source["messages"]] == ["user"]
    assert source["messages"][0]["audio"] == {"path": source["audio_path"], "duration": 1.5}

    # Combining-mark scripts, as matrix or embedded language, are segmented by grapheme cluster.
    chinese = cs_fleurs.format_source_row(_metadata_row(1, "cmn-eng"), "cs-fleurs.mms", "/mnt/raw")
    assert chinese["extra_fields"]["use_mer"] and not chinese["extra_fields"]["mer_grapheme"]
    thai = cs_fleurs.format_source_row(_metadata_row(2, "tha-eng"), "cs-fleurs.mms", "/mnt/raw")
    assert thai["extra_fields"]["mer_grapheme"]
    embedded_khmer = cs_fleurs.format_source_row(_metadata_row(3, "hin-khm"), "cs-fleurs.xtts-test2", "/mnt/raw")
    assert embedded_khmer["extra_fields"]["mer_grapheme"]
    assert (
        cs_fleurs.format_source_row(_metadata_row(4, "yue-eng"), "cs-fleurs.mms", "/mnt")["extra_fields"]["src_lang"]
        is None
    )
    assert "fluency" not in chinese["extra_fields"]
    with pytest.raises(KeyError):
        cs_fleurs.format_source_row(_metadata_row(5, "xxx-eng"), "cs-fleurs.mms", "/mnt")


def test_clip_path_collapses_a_doubled_slash() -> None:
    source = cs_fleurs.format_source_row(
        _metadata_row(0, file_name="audio/cs_ara_fra_x//1_spk0.wav", language="ara-fra"), "cs-fleurs.xtts-test2", "/r"
    )
    assert source["audio_path"] == "/r/xtts/test2/audio/cs_ara_fra_x/1_spk0.wav"


def test_request_row_carries_the_prompt_and_keeps_the_source() -> None:
    contract = audio_manifest.load_profiles(cs_fleurs.PROFILES)["profiles"]["cs-fleurs.mms"]
    source = cs_fleurs.format_source_row(_metadata_row(0), "cs-fleurs.mms", "/r")
    row = audio_manifest.build_row(source, profile="cs-fleurs.mms", contract=contract, index=7)
    request = row["responses_create_params"]
    assert request["input"] == [{"role": "user", "content": "Produce a verbatim transcript of the audio."}]
    assert request["metadata"]["audio_path"] == source["audio_path"]
    assert row["sample_id"] == "cs-fleurs.mms:7" and row["benchmark_profile"] == "cs-fleurs.mms"
    assert row["ng_source_row"] == source and row["ng_source_row_sha256"] == audio_manifest.source_hash(source)


def _publish_metadata(directory: Path, profile: str, rows: list[dict], monkeypatch) -> Path:
    path = directory / f"{profile}.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    hf_dir, split, _, _ = cs_fleurs.SUBSETS[profile]
    monkeypatch.setitem(cs_fleurs.SUBSETS, profile, (hf_dir, split, len(rows), audio_manifest.sha256_file(path)))
    return path


def test_load_metadata_authenticates_the_release(tmp_path: Path, monkeypatch) -> None:
    rows = [_metadata_row(0), _metadata_row(1)]
    path = _publish_metadata(tmp_path, "cs-fleurs.read", rows, monkeypatch)
    monkeypatch.setattr(cs_fleurs, "hf_hub_download", lambda *args, **kwargs: str(path))
    assert cs_fleurs.load_metadata("cs-fleurs.read") == rows

    hf_dir, split, count, digest = cs_fleurs.SUBSETS["cs-fleurs.read"]
    monkeypatch.setitem(cs_fleurs.SUBSETS, "cs-fleurs.read", (hf_dir, split, count, "0" * 64))
    with pytest.raises(ValueError, match="hash drift"):
        cs_fleurs.load_metadata("cs-fleurs.read")
    monkeypatch.setitem(cs_fleurs.SUBSETS, "cs-fleurs.read", (hf_dir, split, count + 1, digest))
    with pytest.raises(ValueError, match="row-count drift"):
        cs_fleurs.load_metadata("cs-fleurs.read")


@pytest.mark.parametrize(
    "duplicate_field, message", [("id", "duplicate row ids"), ("file_name", "duplicate audio paths")]
)
def test_load_metadata_rejects_duplicates(tmp_path: Path, monkeypatch, duplicate_field: str, message: str) -> None:
    second = _metadata_row(1) | {duplicate_field: _metadata_row(0)[duplicate_field]}
    path = _publish_metadata(tmp_path, "cs-fleurs.read", [_metadata_row(0), second], monkeypatch)
    monkeypatch.setattr(cs_fleurs, "hf_hub_download", lambda *args, **kwargs: str(path))
    with pytest.raises(ValueError, match=message):
        cs_fleurs.load_metadata("cs-fleurs.read")


class _FakeHub:
    """Stands in for the dataset repository: snapshot_download copies the files that match the patterns."""

    def __init__(self, repo: Path):
        self.repo = repo
        self.calls: list[list[str]] = []

    def snapshot_download(self, *, local_dir: str, allow_patterns: list[str], **kwargs) -> str:
        self.calls.append(allow_patterns)
        for source in self.repo.rglob("*.wav"):
            relative = source.relative_to(self.repo).as_posix()
            if any(fnmatch.fnmatch(relative, pattern) for pattern in allow_patterns):
                target = Path(local_dir) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
        return local_dir


def _build_repo(tmp_path: Path, monkeypatch) -> _FakeHub:
    """Create a tiny repository with two clips per test set and point the module at it."""
    repo = tmp_path / "repo"
    for number, profile in enumerate(cs_fleurs.SUBSETS):
        hf_dir, split, _, _ = cs_fleurs.SUBSETS[profile]
        rows = [_metadata_row(i, "cmn-eng" if i else "ara-eng") for i in range(2)]
        for row in rows:
            clip = repo / hf_dir / split / row["file_name"]
            clip.parent.mkdir(parents=True, exist_ok=True)
            clip.write_bytes(f"{profile}:{row['id']}".encode())
        # Reuse the same helper that pins the released hash and row count.
        _publish_metadata(tmp_path, profile, rows, monkeypatch)
    monkeypatch.setattr(
        cs_fleurs, "hf_hub_download", lambda repo_id, filename, **kwargs: _metadata_path(tmp_path, filename)
    )
    hub = _FakeHub(repo)
    monkeypatch.setattr(cs_fleurs, "snapshot_download", hub.snapshot_download)
    data_dir = tmp_path / "data"
    monkeypatch.setattr(cs_fleurs, "DATA_DIR", data_dir)
    monkeypatch.setattr(cs_fleurs, "OUTPUT_FPATH", data_dir / "cs_fleurs_benchmark.jsonl")
    return hub


def _metadata_path(directory: Path, filename: str) -> str:
    hf_dir, _, split_and_name = filename.partition("/")
    split = split_and_name.partition("/")[0]
    profile = next(p for p, (d, s, _, _) in cs_fleurs.SUBSETS.items() if (d, s) == (hf_dir, split))
    return str(directory / f"{profile}.jsonl")


def test_download_audio_fetches_the_listed_clips_or_the_whole_set(tmp_path: Path, monkeypatch) -> None:
    hub = _build_repo(tmp_path, monkeypatch)
    rows = cs_fleurs.load_metadata("cs-fleurs.xtts-test2")

    cs_fleurs.download_audio("cs-fleurs.xtts-test2", rows[:1], capped=True)
    assert hub.calls[-1] == ["xtts/test2/audio/ara/ara-eng_0.wav"]
    assert [p.name for p in (tmp_path / "data" / "raw" / "xtts" / "test2").rglob("*.wav")] == ["ara-eng_0.wav"]

    cs_fleurs.download_audio("cs-fleurs.xtts-test2", rows, capped=False)
    assert hub.calls[-1] == ["xtts/test2/audio/**"]
    assert len(list((tmp_path / "data" / "raw" / "xtts" / "test2").rglob("*.wav"))) == 2

    with pytest.raises(FileNotFoundError, match="clips missing after download"):
        cs_fleurs.download_audio("cs-fleurs.xtts-test2", [_metadata_row(9)], capped=True)


def test_prepare_writes_one_combined_jsonl_with_all_profiles(tmp_path: Path, monkeypatch) -> None:
    _build_repo(tmp_path, monkeypatch)
    output = cs_fleurs.prepare()
    assert output == tmp_path / "data" / "cs_fleurs_benchmark.jsonl"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 8
    expected_order = list(audio_manifest.load_profiles(cs_fleurs.PROFILES)["profiles"])
    assert [row["benchmark_profile"] for row in rows] == [name for name in expected_order for _ in range(2)]
    assert [row["sample_id"] for row in rows if row["benchmark_profile"] == "cs-fleurs.mms"] == [
        "cs-fleurs.mms:1",
        "cs-fleurs.mms:2",
    ]
    for row in rows:
        audio_path = Path(row["responses_create_params"]["metadata"]["audio_path"])
        assert audio_path.is_absolute() and audio_path.is_file()
        assert row["ng_source_row_sha256"] == audio_manifest.source_hash(row["ng_source_row"])
    assert cs_fleurs.is_prepared_data_current(output)


def test_prepare_audio_root_and_row_cap_override(tmp_path: Path, monkeypatch) -> None:
    hub = _build_repo(tmp_path, monkeypatch)
    output = cs_fleurs.prepare(audio_root="/data/cs-fleurs", max_samples_per_profile=1)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 4
    by_profile = {row["benchmark_profile"]: row for row in rows}
    assert by_profile["cs-fleurs.xtts-test1"]["responses_create_params"]["metadata"]["audio_path"] == (
        "/data/cs-fleurs/raw/xtts/test1/audio/ara/ara-eng_0.wav"
    )
    assert all(len(call) == 1 for call in hub.calls), "a capped run must download only the listed clips"
    assert len(list((tmp_path / "data" / "raw").rglob("*.wav"))) == 4, "audio is written locally whatever the prefix"
    assert not cs_fleurs.is_prepared_data_current(output), "capped or remapped data must be prepared again by default"


def test_prepare_rejects_profiles_that_disagree_with_the_test_sets(tmp_path: Path, monkeypatch) -> None:
    _build_repo(tmp_path, monkeypatch)
    monkeypatch.delitem(cs_fleurs.SUBSETS, "cs-fleurs.read")
    with pytest.raises(ValueError, match="disagree"):
        cs_fleurs.prepare()


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "5"])
def test_prepare_rejects_a_bad_row_cap_before_any_download(monkeypatch, bad) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("prepare() must validate max_samples_per_profile before touching the hub")

    monkeypatch.setattr(cs_fleurs, "hf_hub_download", forbidden)
    monkeypatch.setattr(cs_fleurs, "snapshot_download", forbidden)
    with pytest.raises(ValueError, match="max_samples_per_profile must be a positive integer"):
        cs_fleurs.prepare(max_samples_per_profile=bad)


def test_the_pinned_revision_and_hashes_are_complete() -> None:
    assert len(cs_fleurs.HF_REVISION) == 40 and set(cs_fleurs.HF_REVISION) <= set("0123456789abcdef")
    assert all(len(digest) == 64 for *_, digest in cs_fleurs.SUBSETS.values())


def test_the_server_accepts_the_profiles() -> None:
    """The audio_eval server resolves every profile's evaluator, metrics and settings when it starts."""
    pytest.importorskip("jiwer", reason="the scoring libraries are installed in the audio_eval server environment")
    from unittest.mock import MagicMock

    from nemo_gym.client import ServerClient
    from resources_servers.audio_eval.app import AudioEvalConfig, AudioEvalResourcesServer

    server = AudioEvalResourcesServer(
        config=AudioEvalConfig(
            name="audio_eval", host="127.0.0.1", port=1, entrypoint="app.py", profile_contracts=str(cs_fleurs.PROFILES)
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    assert sorted(server._profiles) == sorted(PROFILES)


def test_prepared_rows_are_scored_as_mixed_error_rate() -> None:
    """A prepared row goes through the generic evaluator: graphemes for Thai, words for English, one utterance."""
    pytest.importorskip("jiwer", reason="the scoring libraries are installed in the audio_eval server environment")
    from resources_servers.audio_eval._scoring.evaluation.evaluator.audio import AudioEvaluatorConfig, evaluate_sample

    row = _metadata_row(0, "tha-eng") | {"text": "สวัสดี hello world"}
    source = cs_fleurs.format_source_row(row, "cs-fleurs.mms", "/r")
    config = AudioEvaluatorConfig(normalization_mode="multilingual")

    # Four Thai grapheme clusters plus two English words: six units; case and punctuation do not count.
    exact = evaluate_sample(source | {"generation": "สวัสดี Hello, world."}, config)
    assert exact["wer"] == 0.0 and exact["cer"] == 0.0 and exact["wer_ref_words"] == 6

    wrong_word = evaluate_sample(source | {"generation": "สวัสดี hello there"}, config)
    assert (wrong_word["wer_errors"], wrong_word["wer_ref_words"]) == (1, 6)
    assert wrong_word["cer"] > 0

    # One wrong Thai grapheme cluster (a missing vowel sign) is one error, not one per code point.
    wrong_cluster = evaluate_sample(source | {"generation": "สวัดดี hello world"}, config)
    assert (wrong_cluster["wer_errors"], wrong_cluster["wer_ref_words"]) == (1, 6)

    # Without grapheme segmentation the same reference has eight units (one per code point).
    code_points = evaluate_sample(
        source | {"extra_fields": source["extra_fields"] | {"mer_grapheme": False}, "generation": "สวัสดี hello world"},
        config,
    )
    assert code_points["wer_ref_words"] == 8


def test_the_prompt_of_the_source_rows_is_the_prompt_of_the_profiles() -> None:
    for contract in audio_manifest.load_profiles(cs_fleurs.PROFILES)["profiles"].values():
        assert contract["prompt_override"] == {"system_prompt": "", "user_prompt": cs_fleurs.USER_PROMPT}
