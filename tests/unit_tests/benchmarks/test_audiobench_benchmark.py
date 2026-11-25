# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the AudioBench benchmark: declaration, discovery, profiles, row building, prepare()."""

import json
import re
from pathlib import Path

import pytest
import yaml

from benchmarks import audio_manifest
from benchmarks.audiobench import prepare as ab
from nemo_gym.environment.benchmarks import BenchmarkConfig
from nemo_gym.environment.manifest import load_manifest
from tests.unit_tests.benchmarks.audio_benchmark_helpers import pcm_wav_bytes


USES_AUDIO_LIBRARIES = True  # prepare() decodes audio: use the stdlib stand-ins for soundfile

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "audiobench"


def test_declaration_specifics() -> None:
    dataset = BenchmarkConfig.from_config_path(BENCHMARK / "config.yaml").dataset
    manifest = load_manifest(BENCHMARK / "manifest.yaml")
    config = yaml.safe_load((BENCHMARK / "config.yaml").read_text())
    assert {"soundfile", "pyarrow"} <= set(dataset.prepare_dependencies)
    assert manifest.canonical_split == "test" and manifest.prompt_source == "prepared" and manifest.experimental
    # The expression is the union of the per-test-set terms in the README (LicenseRef names must validate).
    assert set(manifest.licensing.split(" AND ")) == {
        "CC0-1.0",
        "CC-BY-4.0",
        "CC-BY-SA-4.0",
        "Apache-2.0",
        "CC-BY-NC-4.0",
        "CC-BY-NC-ND-3.0",
        "LicenseRef-SODL-1.0",
        "LicenseRef-AudioBench-Dataset-Terms",
    }
    server = config["audiobench_resources_server"]["resources_servers"]["audio_eval"]
    assert server["judge_model_server"] == {"type": "responses_api_models", "name": "judge_model"}
    assert "judge_model" in config


def test_census_and_pins() -> None:
    assert len(ab.SUBSETS) == 63 and len(ab.JUDGE_SUBSETS) == 32 and len(ab.AUTOMATIC_SUBSETS) == 31
    assert sum(subset.rows for subset in ab.SUBSETS.values()) == 240701
    assert {subset.repo for subset in ab.SUBSETS.values()} == set(ab.REVISIONS)
    assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in ab.REVISIONS.values())
    assert set(ab.SPOKEN_MQA) <= set(ab.SUBSETS)
    assert sum(ab.SUBSETS[name].rows for name in ab.SPOKEN_MQA) == 100 + 173 + 594 + 1402
    assert len(ab.WER_SUBSETS) + len(ab.SPOKEN_MQA) + len(ab.TRANSLATION_TARGETS) == len(ab.AUTOMATIC_SUBSETS)


def test_profiles_cover_every_subset_with_the_right_scoring() -> None:
    profiles = audio_manifest.load_profiles(ab.PROFILES)["profiles"]
    assert set(profiles) == {ab.profile_name(name) for name in ab.SUBSETS}
    assert ab.profile_name("spoken-mqa_long_digit") == "audiobench.spoken-mqa.long_digit"
    assert ab.profile_name("librispeech_test_clean") == "audiobench.librispeech_test_clean"
    for name in ab.SUBSETS:
        contract = profiles[ab.profile_name(name)]
        assert {"prompt_override", "request", "skills"} <= set(contract) <= {"judge_prompt_template", *contract}
        assert (
            set(contract["skills"]) == {"EVAL_ARGS", "METRICS_TYPE"}
            and contract["skills"]["METRICS_TYPE"] == "audiobench"
            and contract["skills"]["EVAL_ARGS"].startswith("++eval_type=audiobench ")
        )
        assert ("judge_prompt_template" in contract) == (name in ab.JUDGE_SUBSETS)
        audio_manifest.request_controls(contract)  # resolves
    judged = profiles["audiobench.alpaca_audio_test"]["judge_prompt_template"]["user"]
    assert all(f"{{{field}}}" in judged for field in ("expected_answer", "generation", "question"))
    assert judged.rstrip().endswith("Rating: <a single integer from 0 to 5>")

    mqa = profiles["audiobench.spoken-mqa.long_digit"]
    assert mqa["request"] == "reasoning" and mqa["prompt_override"]["user_prompt"] is None
    assert mqa["prompt_override"]["system_prompt"].startswith("You are a voice-controlled assistant.")
    assert mqa["skills"]["EVAL_ARGS"] == "++eval_type=audiobench ++eval_config.normalization_mode=none"
    earnings = profiles["audiobench.earnings21_test"]
    assert earnings["skills"]["EVAL_ARGS"].endswith("normalization_mode=audiobench_earnings")
    assert audio_manifest.request_controls(earnings)["max_output_tokens"] == 16384


def test_select_subsets_defaults_to_all_and_narrows() -> None:
    assert ab.select_subsets(None, None) == tuple(ab.SUBSETS)
    assert ab.select_subsets(tuple(ab.SPOKEN_MQA), None) == tuple(n for n in ab.SUBSETS if n in ab.SPOKEN_MQA)
    assert ab.select_subsets("librispeech_test_clean, mmau_mini", None) == ("mmau_mini", "librispeech_test_clean")
    assert ab.select_subsets("librispeech_test_clean,mmau_mini", "judge") == ("mmau_mini",)
    assert ab.select_subsets("alpaca_audio", None) == ("alpaca_audio_test",), "the _test suffix is optional"
    assert ab.select_subsets(None, ["judge", "nonjudge"]) == tuple(ab.SUBSETS)
    assert set(ab.select_subsets(None, "judge")) == set(ab.JUDGE_SUBSETS)
    for bad in ("nonsense", "alpaca_audio_test,nonsense", ""):
        with pytest.raises(ValueError, match="Empty or unknown AudioBench subsets"):
            ab.select_subsets(bad, None)
    with pytest.raises(ValueError, match="categories"):
        ab.select_subsets(None, "other")
    with pytest.raises(ValueError, match="do not intersect"):
        ab.select_subsets("librispeech_test_clean", "judge")


def test_spoken_mqa_source_row_keeps_the_validated_schema() -> None:
    sample = {
        "instruction": {"text": "Perform the required arithmetic operation and provide the result.", "audio": {}},
        "answer": {"text": ["731"], "audio": {}},
        "context_transcript": "ignored",
    }
    row = ab.format_source_row("spoken-mqa_short_digit", sample, 0, "/data/a.wav", 4.2)
    audio = {"path": "/data/a.wav", "duration": 4.2}
    assert row == {
        "expected_answer": ["731"],
        "audio_path": ["/data/a.wav"],
        "category": "closed",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. /no_think"},
            {
                "role": "user",
                "content": "Perform the required arithmetic operation and provide the result.",
                "audio": audio,
                "audios": [audio],
            },
        ],
        "dataset": "spoken-mqa_short_digit",
        "subset_for_metrics": "spoken-mqa_short_digit",
        "sample_id": 0,
        "task_type": "MathQA",
        "question": "Perform the required arithmetic operation and provide the result.",
        "spoken_mqa_protocol": "arithmetic",
        "source_provenance": {
            "benchmark": "AudioBench",
            "benchmark_revision": "85dcdb88c41e6ebb2bcb3ce19b15d061dfef9dba",
            "dataset": "amao0o0/spoken-mqa",
            "dataset_revision": "a1f9ec6143c6ce12966d7c6b3283905d679645c9",
            "split": "short_digit",
        },
    }
    with pytest.raises(ValueError, match="answer.text must be a list"):
        ab.format_source_row("spoken-mqa_short_digit", {**sample, "answer": "731"}, 0, "/data/a.wav", 1.0)


def test_source_rows_of_the_other_scoring_kinds() -> None:
    sample = {"instruction": "Please transcribe.", "answer": "HELLO WORLD"}
    asr = ab.format_source_row("librispeech_test_clean", sample, 3, "/a.wav", 1.0)
    assert asr["task_type"] == "ASR" and asr["category"] == "closed" and asr["expected_answer"] == "HELLO WORLD"
    assert asr["question"] == "Please transcribe." and "extra_fields" not in asr

    translation = ab.format_source_row("covost2_en_zh_test", sample, 0, "/a.wav", 1.0)
    assert translation["task_type"] == "Translation" and translation["extra_fields"] == {"tgt_lang": "zh"}

    judged = ab.format_source_row("alpaca_audio_test", sample, 0, "/a.wav", 1.0)
    assert judged["category"] == "open" and judged["task_type"] == "unknown"

    # Earnings draws its instruction from the benchmark's instruction list with a fixed seed; the dataset's own is ignored.
    earnings = ab.format_source_row("earnings21_test", sample, 1, "/a.wav", 1.0)
    assert earnings["question"] == "Write down what is said in this recording."
    assert [ab.earnings_instruction(i) for i in range(3)] == [
        "Transcribe this recording into text.",
        "Write down what is said in this recording.",
        "Provide a written transcription of everything that is spoken.",
    ]
    assert earnings["source_provenance"]["normalization"] == "audiobench_earnings" and earnings["task_type"] == "ASR"


def test_answer_options_are_appended_in_the_official_layout() -> None:
    sample = {"instruction": "What does the woman want?", "choices": "(A) a map\n(B) a pen", "answer": "(A) a map"}
    row = ab.format_source_row("cn_college_listen_mcq_test", sample, 0, "/a.wav", 1.0)
    assert row["question"] == "Question:\nWhat does the woman want?\n Choices:\n(A) a map\n(B) a pen"
    assert row["messages"][1]["content"] == row["question"] and row["choices"] == sample["choices"]
    mmau = ab.format_source_row("mmau_mini", {**sample, "choices": ["(A) Man", "(B) Woman"]}, 0, "/a.wav", 1.0)
    assert mmau["question"] == "Question:\nWhat does the woman want?\nChoices:\n(A) Man (B) Woman"
    plain = ab.format_source_row("alpaca_audio_test", {**sample, "choices": "(A) x"}, 0, "/a.wav", 1.0)
    assert plain["question"] == "What does the woman want?", "options are only appended for multiple-choice sets"


def test_decode_audio_returns_mono_float32() -> None:
    array, rate = ab.decode_audio({"bytes": pcm_wav_bytes(8000, channels=2)})
    assert rate == 16000 and array.shape == (8000,) and array.dtype.name == "float32"
    with pytest.raises(ValueError, match="without embedded audio"):
        ab.decode_audio({"bytes": None, "path": "/somewhere/else.wav"})


@pytest.fixture
def offline_data(tmp_path: Path, monkeypatch):
    """Synthetic samples for four real test sets, served instead of the Hugging Face parquet files."""
    samples = {
        "spoken-mqa_short_digit": [
            {
                "context": {"bytes": pcm_wav_bytes(16000)},
                "instruction": {"text": "Compute.", "audio": None},
                "answer": {"text": ["7"], "audio": None},
            },
            {
                "context": {"bytes": pcm_wav_bytes(32000)},
                "instruction": {"text": "Compute.", "audio": None},
                "answer": {"text": ["8"], "audio": None},
            },
        ],
        "librispeech_test_clean": [
            {"context": {"bytes": pcm_wav_bytes(24000)}, "instruction": "Transcribe.", "answer": "ONE"},
            {"context": {"bytes": pcm_wav_bytes(0)}, "instruction": "Transcribe.", "answer": "EMPTY"},
            {"context": {"bytes": pcm_wav_bytes(8000)}, "instruction": "Transcribe.", "answer": "THREE"},
        ],
        "alpaca_audio_test": [
            {"context": {"bytes": pcm_wav_bytes(16000)}, "instruction": "Follow it.", "answer": "Sure."}
        ],
    }
    monkeypatch.setattr(ab, "iter_samples", lambda subset, limit=None: iter(samples[_name_of(subset)][:limit]))
    monkeypatch.setattr(ab, "SUBSETS", {name: spec for name, spec in ab.SUBSETS.items() if name in samples})
    monkeypatch.setattr(ab, "JUDGE_SUBSETS", {"alpaca_audio_test": ab.SUBSETS["alpaca_audio_test"]})
    monkeypatch.setattr(ab, "AUTOMATIC_SUBSETS", {n: ab.SUBSETS[n] for n in samples if n != "alpaca_audio_test"})
    data_dir = tmp_path / "data"
    monkeypatch.setattr(ab, "DATA_DIR", data_dir)
    monkeypatch.setattr(ab, "OUTPUT_FPATH", data_dir / "audiobench_benchmark.jsonl")
    return samples


def _name_of(subset) -> str:
    return next(name for name, spec in ab.SUBSETS.items() if spec == subset)


def test_prepare_defaults_to_every_subset(offline_data) -> None:
    output = ab.prepare()
    assert output == ab.DATA_DIR / "audiobench_benchmark.jsonl"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == [
        "audiobench.alpaca_audio_test:1",
        "audiobench.librispeech_test_clean:1",
        "audiobench.librispeech_test_clean:2",
        "audiobench.spoken-mqa.short_digit:1",
        "audiobench.spoken-mqa.short_digit:2",
    ]
    # The empty clip (index 1) is skipped without leaving a gap in the profile's ids or the file numbering.
    assert [row["ng_source_row"]["sample_id"] for row in rows[1:3]] == [0, 2]
    for row in rows:
        audio_path = Path(row["responses_create_params"]["metadata"]["audio_path"])
        assert audio_path.is_file() and audio_path.parent.parent == (ab.DATA_DIR / "audio").resolve()
        assert row["ng_source_row_sha256"] == audio_manifest.source_hash(row["ng_source_row"])
    mqa = rows[3]
    assert mqa["responses_create_params"]["input"][0]["content"].startswith("You are a voice-controlled assistant.")
    assert mqa["responses_create_params"]["input"][1] == {"role": "user", "content": "Compute."}
    assert mqa["responses_create_params"]["temperature"] == 0.6
    asr = rows[1]
    assert asr["responses_create_params"]["input"] == [{"role": "user", "content": "Transcribe."}]
    assert (
        asr["responses_create_params"]["max_output_tokens"] == 1024
        and asr["responses_create_params"]["temperature"] == 0
    )
    assert ab.is_prepared_data_current(output)
    assert not ab.is_prepared_data_current(ab.DATA_DIR / "other.jsonl")


def test_prepare_narrowing_audio_root_and_row_cap(offline_data) -> None:
    output = ab.prepare(subsets="spoken-mqa_short_digit", audio_root="/data/audiobench")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[1]["responses_create_params"]["metadata"]["audio_path"] == (
        "/data/audiobench/audio/spoken-mqa_short_digit/spoken-mqa_short_digit_000001.wav"
    )
    assert not (ab.DATA_DIR / "audio" / "librispeech_test_clean").exists(), "unselected subsets are not written"
    assert not ab.is_prepared_data_current(output), "narrowed or remapped data must be prepared again by default"

    output = ab.prepare(categories="nonjudge", max_samples_per_profile=1)
    ids = [json.loads(line)["sample_id"] for line in output.read_text().splitlines()]
    assert ids == ["audiobench.librispeech_test_clean:1", "audiobench.spoken-mqa.short_digit:1"]
    assert not ab.is_prepared_data_current(output)

    assert ab.is_prepared_data_current(ab.prepare(subsets=list(ab.SUBSETS), categories=None))
    with pytest.raises(ValueError, match="Empty or unknown"):
        ab.prepare(subsets="nonsense")


def test_iter_samples_reads_pinned_parquet_files_and_checks_the_row_count(tmp_path: Path, monkeypatch) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    import huggingface_hub

    for number, rows in enumerate(([1, 2, 3], [4, 5])):
        pq.write_table(pa.table({"answer": [str(v) for v in rows]}), tmp_path / f"test-0000{number}-of-00002.parquet")
    pq.write_table(pa.table({"answer": ["x"]}), tmp_path / "train-00000-of-00001.parquet")
    seen = {}

    class FakeFileSystem:
        def ls(self, directory, detail=False):
            seen["directory"] = directory
            return sorted(str(path) for path in tmp_path.iterdir())

        def open(self, name, mode):
            return open(name, mode)

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", FakeFileSystem)
    subset = ab.Subset("AudioLLMs/alpaca_audio_test", "data", "test", 5)
    assert [s["answer"] for s in ab.iter_samples(subset)] == ["1", "2", "3", "4", "5"]
    assert (
        seen["directory"] == f"datasets/AudioLLMs/alpaca_audio_test@{ab.REVISIONS['AudioLLMs/alpaca_audio_test']}/data"
    )
    assert [s["answer"] for s in ab.iter_samples(subset, limit=2)] == ["1", "2"]
    with pytest.raises(ValueError, match="expected 6 rows, found 5"):
        list(ab.iter_samples(ab.Subset(subset.repo, "data", "test", 6)))


@pytest.mark.parametrize("cap", [0, -1, 1.5, "3", True])
def test_prepare_rejects_an_invalid_row_cap_before_reading_anything(cap, monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("prepare() must validate the cap before it reads a sample")

    monkeypatch.setattr(ab, "iter_samples", fail)
    with pytest.raises(ValueError, match="max_samples_per_profile must be a positive integer"):
        ab.prepare(subsets="librispeech_test_clean", max_samples_per_profile=cap)


def test_the_cap_reaches_iter_samples(monkeypatch) -> None:
    limits = []

    def source(subset, limit=None):
        limits.append(limit)
        return iter(())

    monkeypatch.setattr(ab, "iter_samples", source)
    monkeypatch.setattr(ab.audio_manifest, "write_rows", lambda path, rows, **kwargs: list(rows))
    ab.prepare(subsets=["librispeech_test_clean", "covost2_zh_en_test"], max_samples_per_profile=7)
    assert limits == [7, 7]
    limits.clear()
    ab.prepare(subsets="librispeech_test_clean")
    assert limits == [None]


def test_readme_license_table_lists_every_test_set_once() -> None:
    readme = (BENCHMARK / "README.md").read_text()
    rows = re.findall(r"^\| `([^`]+)` \| [^|]+ \| [^|]+ \| ([^|]+) \|$", readme, flags=re.MULTILINE)
    assert sorted(name for name, _ in rows) == sorted(ab.SUBSETS)
    classes = [re.sub(r"\*", "", cls).strip() for _, cls in rows]
    assert {cls: classes.count(cls) for cls in set(classes)} == {
        "attribution required": 29,
        "non-commercial": 12,
        "no clear license": 11,
        "permissive": 7,
        "access restricted": 4,
    }


def test_the_server_resolves_every_profile_to_the_audiobench_scorers(monkeypatch) -> None:
    """The audio_eval server resolves each profile's evaluator, metrics and settings, and the judge, when it starts."""
    pytest.importorskip("sacrebleu", reason="the scoring libraries are installed in the audio_eval server environment")
    pytest.importorskip("jiwer")
    pytest.importorskip("whisper_normalizer")
    from unittest.mock import MagicMock

    from nemo_gym.client import ServerClient
    from resources_servers.audio_eval.app import AudioEvalConfig, AudioEvalResourcesServer
    from resources_servers.audio_eval.score import resolve

    monkeypatch.setattr("resources_servers.audio_eval.app.get_global_config_dict", lambda: {})
    server = AudioEvalResourcesServer(
        config=AudioEvalConfig(
            name="audio_eval",
            host="127.0.0.1",
            port=1,
            entrypoint="app.py",
            profile_contracts=str(ab.PROFILES),
            judge_model_server={"type": "responses_api_models", "name": "judge_model"},
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    assert sorted(server._profiles) == sorted(ab.profile_name(name) for name in ab.SUBSETS)
    judged = {name for name, profile in server._profiles.items() if profile.judge is not None}
    assert judged == {ab.profile_name(name) for name in ab.JUDGE_SUBSETS}
    for contract in audio_manifest.load_profiles(ab.PROFILES)["profiles"].values():
        assert resolve("metrics", contract["skills"]["METRICS_TYPE"]).__name__ == "AudioBenchMetrics"
