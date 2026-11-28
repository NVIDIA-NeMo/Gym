# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for Contextual Earnings-22: declaration, discovery, row building, audio, prepare()."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from benchmarks import audio_manifest
from benchmarks.contextual_earnings22 import prepare as ce22
from nemo_gym.environment.benchmarks import BenchmarkConfig
from nemo_gym.environment.manifest import load_manifest
from resources_servers.audio_eval.score import evaluator_settings, resolve
from tests.unit_tests.benchmarks.audio_benchmark_helpers import sine_wav_bytes


USES_AUDIO_LIBRARIES = True
ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "contextual_earnings22"


def _sample(**overrides) -> dict:
    sample = {
        "file_id": "111_chunk1",
        "audio_filename": "111_chunk1__000015.00__0.wav",
        "audio_start": 15.0,
        "audio_end": 30.0,
        "duration": 15.0,
        "source_duration": 15.0,
        "text": "Hello, Anna.",
        "norm_text": "hello anna",
        "local_keywords": ["Anna"],
        "global_keywords": ["Anna", "Boris"],
        "source_row_id": "111_chunk1:15.00:0",
    }
    return sample | overrides


def test_declaration_specifics() -> None:
    dataset = BenchmarkConfig.from_config_path(BENCHMARK / "config.yaml").dataset
    manifest = load_manifest(BENCHMARK / "manifest.yaml")
    assert "soundfile" in dataset.prepare_dependencies
    assert manifest.canonical_split == "test" and manifest.prompt_source == "prepared" and manifest.experimental
    assert manifest.licensing == "CC-BY-SA-4.0" and manifest.domain == "asr"


def test_profiles_cover_every_mode_and_keep_the_scorer_contract() -> None:
    profiles = audio_manifest.load_profiles(ce22.PROFILES)["profiles"]
    assert list(profiles) == [ce22.PROFILE_PREFIX + mode for mode in ce22.MODES]
    for contract in profiles.values():
        assert contract["skills"]["METRICS_TYPE"] == "contextual_earnings22"
        assert evaluator_settings(contract) == (
            "contextual_earnings22",
            {"normalization_mode": "open_asr", "cleanup_generation_loops": True},
        )
        assert resolve("evaluators", "contextual_earnings22").__name__ == "ContextualEarnings22Evaluator"
        assert resolve("metrics", "contextual_earnings22").__name__ == "ContextualEarnings22Metrics"
        assert audio_manifest.request_controls(contract)["max_output_tokens"] == 256


def test_select_modes_defaults_to_all_and_keeps_canonical_order() -> None:
    assert ce22.select_modes(None) == ce22.MODES
    assert ce22.select_modes("global, contextless") == ("contextless", "global")
    assert ce22.select_modes(["local"]) == ("local",)
    for bad in ("nope", "local,nope", "", []):
        with pytest.raises(ValueError, match="Empty or unknown"):
            ce22.select_modes(bad)


def test_source_rows_carry_the_context_of_their_mode() -> None:
    contextless = ce22.format_source_row(_sample(), "contextless", "/a")
    local = ce22.format_source_row(_sample(), "local", "/a")
    context = ce22.format_source_row(_sample(), "global", "/a")
    assert contextless["messages"][0]["content"] == ce22.PROMPT_CONTEXTLESS
    assert local["messages"][0]["content"].startswith("This audio may contain the following words or phrases: Anna. ")
    assert "(some may not be present in the clip): Anna, Boris." in context["messages"][0]["content"]
    # Keyword scoring always uses the clip's own dictionary, whatever the prompt lists.
    assert contextless["keyword_list"] == local["keyword_list"] == context["keyword_list"] == ["Anna"]
    assert local["id"] == "contextual-earnings22-local-111_chunk1:15.00:0"
    assert local["messages"][0]["audio"] == {"path": "/a/111_chunk1__000015.00__0.wav", "duration": 15.0}
    assert local["expected_answer"] == "hello anna" and local["expected_answer_cased"] == "Hello, Anna."
    assert local["subset_for_metrics"] == "111_chunk1"
    with pytest.raises(ValueError, match="Unknown mode"):
        ce22.format_source_row(_sample(), "other", "/a")


def test_audio_name_decode_and_wav_checks(tmp_path: Path) -> None:
    assert ce22.audio_filename("7_chunk3", 1634.08, 571) == "7_chunk3__001634.08__571.wav"
    samples = ce22.decode_audio(sine_wav_bytes())
    assert samples.dtype == np.float32 and samples.shape == (8000,)
    with pytest.raises(ValueError, match="must be mono"):
        ce22.decode_audio(sine_wav_bytes(channels=2))
    with pytest.raises(ValueError, match="sample-rate drift"):
        ce22.decode_audio(sine_wav_bytes(rate=8000))
    with pytest.raises(ValueError, match="short"):
        ce22.decode_audio(sine_wav_bytes(seconds=0.001))

    ce22.write_wav(tmp_path / "x.wav", np.array([0.0, 1.5, -1.5] * 200, dtype=np.float32))
    assert ce22.wav_frames(tmp_path / "x.wav") == 600 and not (tmp_path / "x.wav.tmp").exists()
    with wave.open(str(tmp_path / "x.wav")) as stream:
        assert np.frombuffer(stream.readframes(3), dtype=np.int16).tolist() == [0, 32767, -32767]
    with pytest.raises(ValueError, match="missing or not a WAV"):
        ce22.wav_frames(tmp_path / "missing.wav")
    (tmp_path / "bad.wav").write_bytes(sine_wav_bytes(rate=8000))
    with pytest.raises(ValueError, match="contract failed"):
        ce22.wav_frames(tmp_path / "bad.wav")


@pytest.fixture
def offline_prepare(tmp_path: Path, monkeypatch) -> Path:
    """Run prepare() against a synthetic three-row parquet file, writing under tmp_path."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "audio": [{"bytes": sine_wav_bytes(), "path": None} for _ in range(3)],
            "file_id": ["10_a", "10_a", "11_b"],
            "transcript": [["hello"], ["bye"], ["yes"]],
            "text": ["Hello.", "Bye.", "Yes."],
            "norm_text": ["hello", "bye", "yes"],
            "dictionary": [["Anna"], [], ["Boris", "Eve"]],
            "keywords": [["Anna", "Eve"], ["Anna", "Eve"], ["Boris", "Eve"]],
            "audio_start": pa.array([0.0, 15.0, 0.0], pa.float32()),
            "audio_end": pa.array([15.0, 30.0, 15.0], pa.float32()),
            "duration": pa.array([15.0, 15.0, 15.0], pa.float32()),
            "is_flagged": [False, False, True],
        }
    )
    parquet = tmp_path / "test.parquet"
    pq.write_table(table, parquet)
    monkeypatch.setattr(ce22, "hf_hub_download", lambda *args, **kwargs: str(parquet))
    monkeypatch.setattr(ce22, "PARQUET_SHA256", audio_manifest.sha256_file(parquet))
    monkeypatch.setattr(ce22, "EXPECTED_ROWS", 3)
    data_dir = tmp_path / "data"
    monkeypatch.setattr(ce22, "DATA_DIR", data_dir)
    monkeypatch.setattr(ce22, "OUTPUT_FPATH", data_dir / "contextual_earnings22_benchmark.jsonl")
    return data_dir


def test_prepare_writes_every_mode_with_existing_audio(offline_prepare: Path) -> None:
    output = ce22.prepare()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 9
    assert [row["benchmark_profile"] for row in rows[::3]] == [ce22.PROFILE_PREFIX + mode for mode in ce22.MODES]
    assert [row["sample_id"] for row in rows[:3]] == ["contextual-earnings22.contextless:1"] + [
        f"contextual-earnings22.contextless:{i}" for i in (2, 3)
    ]
    first = rows[0]
    audio_path = Path(first["responses_create_params"]["metadata"]["audio_path"])
    assert audio_path == (offline_prepare / "audio" / "10_a__000000.00__0.wav").resolve() and audio_path.is_file()
    assert first["responses_create_params"]["input"] == [{"role": "user", "content": ce22.PROMPT_CONTEXTLESS}]
    assert first["duration"] == 0.5 and first["ng_source_row_sha256"] == audio_manifest.source_hash(
        first["ng_source_row"]
    )
    assert rows[8]["keyword_list"] == ["Boris", "Eve"]
    assert "Boris, Eve" in rows[8]["responses_create_params"]["input"][0]["content"]

    assert ce22.is_prepared_data_current(output)
    assert not ce22.is_prepared_data_current(offline_prepare / "other.jsonl")


def test_prepare_narrowing_audio_root_and_row_cap(offline_prepare: Path) -> None:
    output = ce22.prepare(modes="local", audio_root="/data/ce22", max_samples_per_profile=2)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["benchmark_profile"] for row in rows] == ["contextual-earnings22.local"] * 2
    assert rows[0]["responses_create_params"]["metadata"]["audio_path"] == "/data/ce22/audio/10_a__000000.00__0.wav"
    assert (offline_prepare / "audio" / "10_a__000000.00__0.wav").is_file()
    assert not (offline_prepare / "audio" / "11_b__000000.00__2.wav").exists(), "rows beyond the cap are not written"
    assert not ce22.is_prepared_data_current(output), "narrowed data must be prepared again by default"


def test_prepare_rejects_a_drifted_source(offline_prepare: Path, monkeypatch) -> None:
    monkeypatch.setattr(ce22, "EXPECTED_ROWS", 4)
    with pytest.raises(ValueError, match="row-count drift"):
        ce22.prepare()
    monkeypatch.setattr(ce22, "PARQUET_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="hash drift"):
        ce22.prepare()


def test_prepare_rejects_a_non_positive_row_cap(offline_prepare: Path) -> None:
    for bad in (0, -1):
        with pytest.raises(ValueError, match="positive integer"):
            ce22.prepare(max_samples_per_profile=bad)
    assert not (offline_prepare / "audio").exists(), "the cap is checked before any download or audio write"
