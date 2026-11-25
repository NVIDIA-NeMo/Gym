# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the Open ASR leaderboard (English) benchmark: declaration, prepare(), chunk handling.

These tests live with the ``asr_with_pc`` server because ``prepare()`` applies that server's English text normalizer
(``whisper-normalizer``) to filter references, a dependency the core test environment does not install.
"""

import asyncio
import dataclasses
import json
import re
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from benchmarks import hf_parquet
from benchmarks.asr_leaderboard import prepare as leaderboard
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse
from nemo_gym.environment import BenchmarkConfig, load_manifest
from resources_servers.asr_with_pc.app import ASRWithPCConfig, ASRWithPCResourcesServer, ASRWithPCVerifyRequest
from tests.unit_tests.benchmarks.audio_benchmark_helpers import sine_wav_bytes


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "asr_leaderboard"


_wav_bytes = partial(sine_wav_bytes, seconds=0.25)


@pytest.fixture
def offline(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    """Synthetic sources for every subset, each request recorded; urgent2024 is 8 kHz and has 3 rows."""
    requests: list[tuple] = []
    chunk_audio = tmp_path / "chunk.wav"
    chunk_audio.write_bytes(_wav_bytes())

    def entries(name: str, rate: int = 16000) -> list[dict]:
        audio = {"bytes": _wav_bytes(rate), "path": None}
        texts = ["Hello, World!", "ignore time segment in scoring", "  ", "Third one."]
        return [{"id": f"{name}/{i}", "text": text, "audio": audio} for i, text in enumerate(texts)]

    def fake_config_files(repo, revision, config, split):
        requests.append((repo, config, {"revision": revision, "split": split}))
        return [f"{repo}|{config}"]

    def fake_iter_rows(repo, revision, files, *, limit=None):
        assert revision == leaderboard.LEADERBOARD_REVISION
        config = files[0].split("|")[1]
        rate = 8000 if config == "urgent2024" else 16000
        return iter(entries(config, rate)[:limit])

    def fake_iter_json_rows(repo, revision, files):
        if repo == leaderboard.EARNINGS22_PARENT_REPO:
            assert revision == leaderboard.EARNINGS22_PARENT_REVISION
            return iter(
                [{"id": "call-1", "transcript": "Parent call text."}, {"id": "call-2", "transcript": "Other."}]
            )
        assert repo == leaderboard.EARNINGS22_REPO and revision == leaderboard.EARNINGS22_REVISION
        return iter(
            [
                {"id": "c0", "parent_id": "call-1", "chunk_index": 0, "file_name": "c0.flac", "duration": 1.0},
                {"id": "c1", "parent_id": "call-1", "chunk_index": 1, "file_name": "c1.flac", "duration": 2.0},
                {"id": "c2", "parent_id": "call-2", "chunk_index": 0, "file_name": "c2.flac", "duration": 3.0},
            ]
        )

    downloaded: list[str] = []

    def fake_hf_hub_download(repo, filename, **kwargs):
        assert repo == leaderboard.EARNINGS22_REPO and kwargs["revision"] == leaderboard.EARNINGS22_REVISION
        downloaded.append(filename)
        return str(chunk_audio)

    monkeypatch.setattr(hf_parquet, "config_files", fake_config_files)
    monkeypatch.setattr(hf_parquet, "iter_rows", fake_iter_rows)
    monkeypatch.setattr(hf_parquet, "iter_json_rows", fake_iter_json_rows)
    monkeypatch.setattr(leaderboard, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(leaderboard, "DATASETS", {**leaderboard.DATASETS, "urgent2024": _urgent(expected_rows=4)})
    data_dir = tmp_path / "data"
    monkeypatch.setattr(leaderboard, "DATA_DIR", data_dir)
    monkeypatch.setattr(leaderboard, "OUTPUT_FPATH", data_dir / "asr_leaderboard_benchmark.jsonl")
    return SimpleNamespace(requests=requests, downloaded=downloaded)


def locked_version(package: str) -> str:
    """The version of ``package`` that the server's lock file pins."""
    lock = (ROOT / "resources_servers" / "asr_with_pc" / "uv.lock").read_text()
    return re.search(rf'name = "{package}"\nversion = "([^"]+)"', lock).group(1)


def _urgent(expected_rows: int):
    return dataclasses.replace(leaderboard.DATASETS["urgent2024"], expected_rows=expected_rows)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_declaration_specifics() -> None:
    dataset = BenchmarkConfig.from_config_path(BENCHMARK / "config.yaml", strict=False).dataset
    manifest = load_manifest(BENCHMARK / "manifest.yaml")
    config = yaml.safe_load((BENCHMARK / "config.yaml").read_text())
    assert "soundfile" in dataset.prepare_dependencies
    # prepare.py imports the Open ASR normalizer, which needs this package in the process that runs it, at the
    # version the server locks (its scoring depends on the version)
    assert f"whisper-normalizer=={locked_version('whisper-normalizer')}" in dataset.prepare_dependencies
    for license_id in ("CC-BY-4.0", "CC0-1.0", "CC-BY-SA-4.0", "Apache-2.0", "LicenseRef-SPGISpeech-Terms"):
        assert license_id.lower() in manifest.licensing.lower()
    assert manifest.domain == "asr"
    assert manifest.prompt_source == "prepared" and manifest.standard_prompt_config is None
    server = config["asr_leaderboard_asr_with_pc_resources_server"]["resources_servers"]["asr_with_pc"]
    assert server["task_type"] == "OpenASR" and server["cleanup_generation_loops"] is True


def test_prepared_file_is_the_declared_dataset_file() -> None:
    dataset = BenchmarkConfig.from_config_path(BENCHMARK / "config.yaml", strict=False).dataset
    assert Path(dataset.jsonl_fpath) == Path("benchmarks") / "asr_leaderboard" / "data" / leaderboard.OUTPUT_FPATH.name
    assert (ROOT / dataset.jsonl_fpath).resolve() == leaderboard.OUTPUT_FPATH.resolve()


def test_revisions_are_pinned() -> None:
    for revision in (
        leaderboard.LEADERBOARD_REVISION,
        leaderboard.EARNINGS22_REVISION,
        leaderboard.EARNINGS22_PARENT_REVISION,
    ):
        assert len(revision) == 40 and set(revision) <= set("0123456789abcdef")
    assert all(spec.revision for spec in leaderboard.DATASETS.values())
    assert leaderboard.DATASETS["earnings22_aa"].chunked and not leaderboard.DATASETS["ami"].chunked


def test_reference_filter_matches_the_leaderboard() -> None:
    assert leaderboard.is_target_text_in_range("Hello there")
    assert not leaderboard.is_target_text_in_range("ignore time segment in scoring")
    assert not leaderboard.is_target_text_in_range("   ")


def test_prepare_defaults_to_every_subset_and_writes_rows(offline: SimpleNamespace, tmp_path: Path) -> None:
    output = leaderboard.prepare()
    assert [config for repo, config, _ in offline.requests if repo == leaderboard.LEADERBOARD_REPO] == [
        "librispeech",
        "librispeech",
        "voxpopuli_cleaned_aa",
        "gigaspeech_cleaned",
        "spgispeech",
        "ami_cleaned",
        "urgent2024",
    ]
    assert all(kwargs["revision"] and kwargs["split"] for _, _, kwargs in offline.requests)

    rows = _rows(output)
    by_subset: dict[str, list[dict]] = {}
    for row in rows:
        by_subset.setdefault(row["subset_for_metrics"], []).append(row)
    assert list(by_subset) == list(leaderboard.DATASETS)
    # The reference filter drops the "ignore" and the empty rows: two of four remain per regular subset.
    first = by_subset["librispeech_clean"][0]
    assert [row["sample_id"] for row in by_subset["librispeech_clean"]] == ["librispeech_0", "librispeech_3"]
    assert first["expected_answer"] == "Hello, World!" and first["benchmark_profile"] == "asr_leaderboard"
    assert first["responses_create_params"]["input"] == [{"role": "user", "content": leaderboard.VERBATIM_PROMPT}]
    assert "task_type" not in first, "the server config sets the task type"
    spgi = by_subset["spgispeech"][0]["responses_create_params"]["input"][0]["content"]
    assert spgi == leaderboard.NO_FILLERS_PROMPT != leaderboard.VERBATIM_PROMPT

    audio_path = Path(first["responses_create_params"]["metadata"]["audio_path"])
    assert audio_path == (tmp_path / "data" / "audio" / "librispeech_clean" / "librispeech_0.flac").resolve()
    assert audio_path.is_file() and first["audio_duration"] == pytest.approx(0.25)
    assert leaderboard.is_prepared_data_current(output)


def test_urgent2024_is_resampled_and_other_rates_are_rejected(
    offline: SimpleNamespace, tmp_path: Path, monkeypatch
) -> None:
    import soundfile as sf

    row = _rows(leaderboard.prepare(datasets=["urgent2024"]))[0]
    path = Path(row["responses_create_params"]["metadata"]["audio_path"])
    samples, rate = sf.read(path)
    assert rate == 16000 and len(samples) == 4000 and row["audio_duration"] == pytest.approx(0.25)

    monkeypatch.setitem(leaderboard.DATASETS, "urgent2024", _no_resample())
    with pytest.raises(ValueError, match="Expected 16000 Hz"):
        leaderboard.prepare(datasets=["urgent2024"])


def _no_resample():
    return dataclasses.replace(_urgent(expected_rows=4), resample_audio=False)


def test_source_row_count_is_checked_for_urgent2024(offline: SimpleNamespace, monkeypatch) -> None:
    monkeypatch.setitem(leaderboard.DATASETS, "urgent2024", _urgent(expected_rows=1000))
    with pytest.raises(ValueError, match="expected 1000 source rows, found 4"):
        leaderboard.prepare(datasets=["urgent2024"])
    # A capped smoke run reads fewer rows on purpose and skips the check.
    assert len(_rows(leaderboard.prepare(datasets=["urgent2024"], max_samples_per_subset=2))) == 1


def test_earnings22_chunks_carry_parent_references(offline: SimpleNamespace) -> None:
    rows = _rows(leaderboard.prepare(datasets=["earnings22_aa"]))
    assert [(row["parent_id"], row["chunk_index"]) for row in rows] == [("call-1", 0), ("call-1", 1), ("call-2", 0)]
    # Every chunk's reference is its parent's full transcript.
    assert [row["expected_answer"] for row in rows] == ["Parent call text.", "Parent call text.", "Other."]
    assert [row["sample_id"] for row in rows] == ["c0", "c1", "c2"]
    assert rows[0]["audio_duration"] == pytest.approx(0.25), "duration comes from the decoded audio file"
    assert offline.downloaded == ["audio/c0.flac", "audio/c1.flac", "audio/c2.flac"]


def test_earnings22_chunk_without_a_parent_transcript_is_an_error(offline: SimpleNamespace, monkeypatch) -> None:
    def broken_parents(repo, revision, files):
        if repo == leaderboard.EARNINGS22_PARENT_REPO:
            return iter([{"id": "call-2", "transcript": "Other."}])
        return original(repo, revision, files)

    original = hf_parquet.iter_json_rows
    monkeypatch.setattr(hf_parquet, "iter_json_rows", broken_parents)
    with pytest.raises(ValueError, match="No transcript"):
        leaderboard.prepare(datasets=["earnings22_aa"])


@pytest.mark.parametrize("cap", [0, -1, 1.5, True, "5"])
def test_row_cap_is_validated_before_any_download(offline: SimpleNamespace, cap) -> None:
    with pytest.raises(ValueError, match="max_samples_per_subset must be a positive integer"):
        leaderboard.prepare(max_samples_per_subset=cap)
    assert offline.requests == [] and offline.downloaded == []


def test_prepare_narrowing_arguments(offline: SimpleNamespace, tmp_path: Path) -> None:
    rows = _rows(leaderboard.prepare(datasets=["ami", "librispeech_other"], max_samples_per_subset=1))
    assert [row["subset_for_metrics"] for row in rows] == ["ami", "librispeech_other"]
    assert not leaderboard.is_prepared_data_current(leaderboard.OUTPUT_FPATH), "narrowed data is never current"

    rows = _rows(leaderboard.prepare(datasets=["ami"], audio_root="/data/asr_leaderboard/"))
    assert (
        rows[0]["responses_create_params"]["metadata"]["audio_path"]
        == "/data/asr_leaderboard/audio/ami/ami_cleaned_0.flac"
    )
    assert (tmp_path / "data" / "audio" / "ami" / "ami_cleaned_0.flac").is_file()

    with pytest.raises(ValueError, match="Unknown datasets"):
        leaderboard.prepare(datasets=["tedlium"])


def test_chunked_rows_are_scored_per_parent_call(offline: SimpleNamespace) -> None:
    """Rows feed the OpenASR server unchanged: chunk hypotheses are joined by ``chunk_index`` and scored once."""
    rows = _rows(leaderboard.prepare(datasets=["earnings22_aa"]))
    config = ASRWithPCConfig(host="0.0.0.0", port=8080, entrypoint="", name="", task_type="OpenASR")
    server = ASRWithPCResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    hypotheses = ["parent call", "text.", "Other."]

    def rollout(row: dict, hypothesis: str) -> dict:
        response = NeMoGymResponse(
            id="r",
            created_at=0.0,
            model="m",
            object="response",
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
            output=[
                {
                    "id": "m",
                    "role": "assistant",
                    "type": "message",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": hypothesis, "annotations": []}],
                }
            ],
        )
        request = ASRWithPCVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "x"}], "parallel_tool_calls": True},
            response=response,
            **{key: value for key, value in row.items() if key != "responses_create_params"},
        )
        return asyncio.run(server.verify(request)).model_dump()

    metrics = server.compute_metrics([[rollout(row, hypothesis)] for row, hypothesis in zip(rows, hypotheses)])
    assert metrics["corpus_wer@k=1"] == pytest.approx(0.0), "split hypotheses match their parent transcripts"
