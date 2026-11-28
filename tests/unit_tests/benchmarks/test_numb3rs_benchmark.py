# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the Numb3rs benchmark: declaration, discovery, manifest census and row building."""

import json
import re
from pathlib import Path

import pytest

from benchmarks.numb3rs import prepare as numb3rs
from nemo_gym.environment.benchmarks import BenchmarkConfig
from nemo_gym.environment.manifest import load_manifest
from resources_servers.asr_with_pc.task_data import TaskData


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "numb3rs"


def _manifest_rows() -> list[dict]:
    return [
        {
            "audio_filepath": "./TIME/TIME_1_two_pm.wav",
            "duration": 1.5,
            "info": "TIME",
            "original_text": "2 pm",
            "text": "two p m",
        },
        {
            "audio_filepath": "./ADDRESS/ADDRESS_2_C6.wav",
            "duration": 1.0,
            "info": "ADDRESS",
            "original_text": "C6",
            "text": "c six",
        },
        {
            "audio_filepath": "./ADDRESS/ADDRESS_3_A6.wav",
            "duration": 1.2,
            "info": "ADDRESS",
            "original_text": "A6",
            "text": "a six",
        },
    ]


@pytest.fixture
def offline(tmp_path: Path, monkeypatch) -> Path:
    """Serve the synthetic manifest and one-byte audio files instead of downloading the dataset."""
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in _manifest_rows()))
    monkeypatch.setattr(numb3rs, "hf_hub_download", lambda *args, **kwargs: str(manifest))
    monkeypatch.setattr(numb3rs, "EXPECTED_ROWS", 3)

    def snapshot_download(*, allow_patterns, local_dir, **kwargs):
        for pattern in allow_patterns:
            path = Path(local_dir) / pattern
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"RIFF")

    monkeypatch.setattr(numb3rs, "snapshot_download", snapshot_download)
    monkeypatch.setattr(numb3rs, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(numb3rs, "OUTPUT_FPATH", tmp_path / "data" / "numb3rs_benchmark.jsonl")
    return tmp_path


def test_declaration_specifics() -> None:
    dataset = BenchmarkConfig.from_config_path(BENCHMARK / "config.yaml").dataset
    manifest = load_manifest(BENCHMARK / "manifest.yaml")
    assert dataset.prompt_config == Path("benchmarks/numb3rs/prompts/default.yaml")
    assert manifest.licensing == "CC-BY-NC-SA-4.0"
    assert manifest.standard_prompt_config == "benchmarks/numb3rs/prompts/default.yaml"
    assert manifest.domain == "asr" and manifest.resources_server == "asr_with_pc"
    assert "cleanup_generation_loops: true" in (BENCHMARK / "config.yaml").read_text()


def test_manifest_census_and_category_order(offline: Path) -> None:
    rows = numb3rs.load_manifest()
    # Category by category, each in manifest order.
    assert [row["audio_filepath"] for row in rows] == [
        "./ADDRESS/ADDRESS_2_C6.wav",
        "./ADDRESS/ADDRESS_3_A6.wav",
        "./TIME/TIME_1_two_pm.wav",
    ]
    assert len(numb3rs.load_manifest(2)) == 2
    numb3rs.EXPECTED_ROWS = 4
    with pytest.raises(ValueError, match="row-count drift"):
        numb3rs.load_manifest()


def test_manifest_rejects_a_path_outside_its_category(offline: Path) -> None:
    rows = _manifest_rows()
    rows[0]["audio_filepath"] = "./DATE/TIME_1_two_pm.wav"
    (offline / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="unexpected audio path"):
        numb3rs.load_manifest()


def test_prepare_writes_the_three_profiles_over_the_same_clips(offline: Path) -> None:
    path = numb3rs.prepare()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 9
    by_profile = {name: [row for row in rows if row["benchmark_profile"] == name] for name in numb3rs.PROFILES}
    assert {name: len(group) for name, group in by_profile.items()} == {
        "numb3rs": 3,
        "numb3rs.tn": 3,
        "numb3rs.itn": 3,
    }

    neutral, tn, itn = (by_profile[name][0] for name in ("numb3rs", "numb3rs.tn", "numb3rs.itn"))
    assert neutral["expected_answer"] == itn["expected_answer"] == "c six" and tn["expected_answer"] == "C6"
    assert neutral["text_tn"] == "C6" and neutral["text_itn"] == "c six"
    assert neutral["prompt"] == "Transcribe the audio file into English text."
    assert tn["prompt"] == (
        "Transcribe the audio into written form with numbers as digits (e.g., '$100', '3.14', 'Jan 1')."
    )
    assert itn["prompt"] == (
        "Transcribe the audio into spoken form with numbers spelled out "
        "(e.g., 'one hundred dollars', 'three point one four', 'january first')."
    )

    for row in rows:
        TaskData.model_validate(row)  # the asr_with_pc server's own row schema
        assert row["task_type"] == "ASR_LEADERBOARD" and row["normalization"] == "numb3rs"
        assert row["category"] in {"ADDRESS", "TIME"}
        assert row["reference_fields"] == ["text_tn", "text_itn"]
        assert set(row["responses_create_params"]) == {"metadata"}  # prompt and sampling come from the config
        assert Path(row["responses_create_params"]["metadata"]["audio_path"]).is_file()
    # Categories never pool across profiles.
    assert neutral["subset_for_metrics"] == "numb3rs_ADDRESS" and tn["subset_for_metrics"] == "numb3rs.tn_ADDRESS"
    assert neutral["sample_id"] == "numb3rs:ADDRESS_2_C6" and neutral["audio_duration"] == 1.0
    assert numb3rs.is_prepared_data_current(path)


def test_the_prompt_config_renders_each_profile_prompt() -> None:
    from nemo_gym.data.prompt import apply_prompt_to_row, load_prompt_config

    prompt = load_prompt_config("benchmarks/numb3rs/prompts/default.yaml")
    for spec in numb3rs.PROFILES.values():
        row = {"responses_create_params": {"metadata": {}}, "prompt": spec["prompt"]}
        messages = apply_prompt_to_row(row, prompt)["responses_create_params"]["input"]
        assert messages == [
            {"role": "system", "content": "You are a helpful assistant. /no_think"},
            {"role": "user", "content": spec["prompt"]},
        ]
    structured = load_prompt_config("benchmarks/numb3rs/prompts/structured.yaml")
    assert structured.system is None and structured.user.startswith("Audio transcription. Language: as spoken.")


def test_prepare_audio_root_profile_and_row_cap_override(offline: Path) -> None:
    path = numb3rs.prepare(audio_root="/data/numb3rs", profiles=["numb3rs.tn"], max_samples_per_profile=1)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["benchmark_profile"] == "numb3rs.tn"
    assert (
        rows[0]["responses_create_params"]["metadata"]["audio_path"] == "/data/numb3rs/source/ADDRESS/ADDRESS_2_C6.wav"
    )
    assert not numb3rs.is_prepared_data_current(path)
    with pytest.raises(ValueError, match="unknown Numb3rs profiles"):
        numb3rs.prepare(profiles=["numb3rs.other"])


def test_prepare_takes_one_profile_as_a_string_or_a_comma_separated_string(offline: Path) -> None:
    # `+prepare_script_args.profiles=numb3rs` is a str (not the list `[numb3rs]`); it must not be split into characters.
    for profiles in ("numb3rs.tn", "numb3rs.tn,numb3rs.itn"):
        names = {
            json.loads(line)["benchmark_profile"]
            for line in numb3rs.prepare(profiles=profiles).read_text().splitlines()
        }
        assert names == set(profiles.split(","))


def test_prepare_fails_when_a_clip_was_not_downloaded(offline: Path, monkeypatch) -> None:
    monkeypatch.setattr(numb3rs, "snapshot_download", lambda **kwargs: None)
    with pytest.raises(FileNotFoundError, match="not downloaded"):
        numb3rs.prepare()


@pytest.mark.parametrize("cap", [0, -1, 1.5, True, "3"])
def test_prepare_rejects_an_invalid_row_cap_before_downloading(offline: Path, monkeypatch, cap) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("nothing may be fetched for an invalid cap")

    monkeypatch.setattr(numb3rs, "hf_hub_download", fail)
    monkeypatch.setattr(numb3rs, "snapshot_download", fail)
    with pytest.raises(ValueError, match="positive integer"):
        numb3rs.prepare(max_samples_per_profile=cap)


def test_the_dataset_revision_and_census_are_pinned() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", numb3rs.HF_REVISION) and numb3rs.EXPECTED_ROWS == 10131


def test_a_lowercase_manifest_category_is_uppercased_everywhere() -> None:
    row = {
        "audio_filepath": "number/a.wav",
        "duration": 1.0,
        "info": "number",
        "original_text": "42",
        "text": "forty two",
    }
    (built,) = numb3rs.iter_rows([row], "/audio", ["numb3rs"])
    assert built["category"] == "NUMBER" and built["subset_for_metrics"] == "numb3rs_NUMBER"
