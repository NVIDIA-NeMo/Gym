# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from unittest.mock import MagicMock

import datasets
import pytest
import yaml
from huggingface_hub.errors import GatedRepoError

from benchmarks.audiobench import _prepare_lib
from benchmarks.audiobench.DATASETS import DATASETS, HF_REVISIONS
from nemo_gym.environment.manifest import load_manifest


BUCKETS = {"judge": 32, "asr": 21, "bleu": 6, "exact_match": 4}


class FakeDataset:
    """Just enough of ``datasets.Dataset`` for ``_iter_dataset_rows``."""

    column_names = ["context", "instruction", "answer"]

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def cast_column(self, column: str, feature: object) -> "FakeDataset":
        return self

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


@pytest.fixture
def local_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(_prepare_lib, "DATA_DIR", tmp_path)
    monkeypatch.setattr(_prepare_lib, "AUDIO_DIR", tmp_path / "audio")
    return tmp_path


def test_registry_matches_skills_sub_dataset_counts() -> None:
    counts = {bucket: sum(spec["bucket"] == bucket for spec in DATASETS.values()) for bucket in BUCKETS}
    assert counts == BUCKETS
    assert len(DATASETS) == 63


def test_every_source_repo_is_pinned() -> None:
    assert {spec["hf_repo"] for spec in DATASETS.values()} == set(HF_REVISIONS)
    assert all(len(sha) == 40 for sha in HF_REVISIONS.values())


@pytest.mark.parametrize("bucket", BUCKETS)
def test_manifest_mirrors_config(bucket: str) -> None:
    bucket_dir = Path(_prepare_lib.BENCHMARK_DIR) / bucket
    manifest = load_manifest(bucket_dir / "manifest.yaml")
    config = yaml.safe_load((bucket_dir / "config.yaml").read_text())
    (agent,) = [v for k, v in config.items() if k.endswith("_simple_agent")]
    (config_dataset,) = agent["responses_api_agents"]["simple_agent"]["datasets"]
    (manifest_dataset,) = manifest.datasets

    assert manifest.name == f"audiobench/{bucket}"
    assert manifest.resources_server == ("audiobench_judge" if bucket == "judge" else "asr_with_pc")
    for field in ("name", "jsonl_fpath", "prepare_script", "prompt_config", "num_repeats"):
        assert getattr(manifest_dataset, field) == config_dataset[field]


def test_prepare_writes_pinned_rows_without_agent_ref(local_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_load_dataset(repo: str, **kwargs: object) -> FakeDataset:
        calls.append((repo, kwargs))
        answer = {"text": ["42"], "audio": None}  # Spoken-MQA's answer shape
        return FakeDataset([{"context": {"bytes": b"RIFF"}, "instruction": "Answer it.", "answer": answer}] * 3)

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    out_path = _prepare_lib.prepare_audiobench_bucket(
        "exact_match",
        datasets=["spoken_mqa_long_digit"],
        max_samples_per_dataset=2,
        audio_prefix="/mnt/audio/",
    )

    assert out_path == local_data_dir / "audiobench_exact_match.jsonl"
    assert calls == [("amao0o0/spoken-mqa", {"split": "long_digit", "revision": HF_REVISIONS["amao0o0/spoken-mqa"]})]
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0] == {
        "responses_create_params": {"metadata": {"audio_path": "/mnt/audio/spoken_mqa_long_digit_000000.wav"}},
        "instruction": DATASETS["spoken_mqa_long_digit"]["instruction"],
        "question": DATASETS["spoken_mqa_long_digit"]["instruction"],
        "expected_answer": "42",
        "subset_for_metrics": "spoken_mqa_long_digit",
        "sample_id": "spoken_mqa_long_digit_000000",
        "split": "long_digit",
    }
    assert (local_data_dir / "audio" / "spoken_mqa_long_digit_000001.wav").read_bytes() == b"RIFF"


def test_prepare_skips_only_license_gated_failures(local_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def gated(repo: str, **kwargs: object) -> FakeDataset:
        raise GatedRepoError("accept the license first", response=MagicMock())

    monkeypatch.setattr(datasets, "load_dataset", gated)
    out_path = _prepare_lib.prepare_audiobench_bucket("judge", datasets=["imda_ar_sentence"])
    assert out_path.read_text() == ""

    with pytest.raises(GatedRepoError):
        _prepare_lib.prepare_audiobench_bucket("judge", datasets=["alpaca_audio_test"])


def test_prepare_rejects_sub_dataset_from_another_bucket(local_data_dir: Path) -> None:
    with pytest.raises(ValueError, match="bucket='exact_match', not 'judge'"):
        _prepare_lib.prepare_audiobench_bucket("judge", datasets=["spoken_mqa_long_digit"])
