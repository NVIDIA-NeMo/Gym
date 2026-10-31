# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Gym collation must keep every generated wrapper and sidecar outside Git."""

import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.agentic_if_realworld import prepare
from nemo_gym.train_data_utils import TrainDataProcessor, TrainDataProcessorConfig
from resources_servers.agentic_if_realworld.benchmark.core import Dataset
from resources_servers.agentic_if_realworld.benchmark.synthetic import create
from tests.unit_tests.test_train_data_utils import _make_agent_instance_config


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    script = checkout / "benchmarks/agentic_if_realworld/prepare.py"
    script.parent.mkdir(parents=True)
    create(tmp_path / "release")
    monkeypatch.setattr(prepare, "__file__", str(script))
    monkeypatch.setenv("AGENTIC_IF_SYNTHETIC", "1")
    monkeypatch.setenv("AGENTIC_IF_DATA_DIR", str(tmp_path / "release"))
    return script.parent / "data", tmp_path / "prepared"


def test_prepared_directory_symlink_is_ignored_by_git(preparation, monkeypatch):
    link, private = preparation
    monkeypatch.setenv("AGENTIC_IF_PREPARED_PATH", str(private / "prepared.jsonl"))
    prepare.prepare()
    source = Path(__file__).parents[3] / "benchmarks/agentic_if_realworld/.gitignore"
    (link.parent / ".gitignore").write_bytes(source.read_bytes())
    checkout = link.parents[2]
    subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
    ignored = subprocess.run(["git", "check-ignore", "--quiet", "--", str(link)], cwd=checkout)
    assert ignored.returncode == 0


@pytest.mark.parametrize("basename", ["prepared.jsonl", "custom.jsonl"])
def test_prepare_and_real_collation_keep_adjacent_files_private(preparation, monkeypatch, basename, tmp_path):
    link, private = preparation
    target = private / basename
    monkeypatch.setenv("AGENTIC_IF_PREPARED_PATH", str(target))
    configured = prepare.prepare()
    dataset = Dataset(tmp_path / "release", synthetic=True)
    assert configured == link / "prepared.jsonl"
    assert link.is_symlink() and link.resolve() == private
    assert target.exists() and configured.resolve() == target
    assert [json.loads(line) for line in target.open()] == dataset.prepared()
    assert prepare.prepare() == configured

    configs = [
        _make_agent_instance_config(
            "synthetic", [{"name": "synthetic", "type": "example", "jsonl_fpath": str(configured)}]
        )
    ]
    processor = TrainDataProcessor()
    metrics = processor.validate_samples_and_aggregate_metrics(configs, overwrite_metrics_conflicts=False)
    processor.collate_samples(
        TrainDataProcessorConfig(output_dirpath=str(tmp_path / "collated"), mode="example_validation"),
        configs,
        metrics,
    )
    for name in ["prepared_prepare.jsonl", "prepared_metrics.json"]:
        generated = link / name
        assert generated.is_file() and generated.resolve() == private / name
    rows = [json.loads(line) for line in (tmp_path / "collated/example.jsonl").open()]
    assert [{key: row[key] for key in dataset.prepared()[0]} for row in rows] == dataset.prepared()
    assert all(path.is_symlink() or path.is_dir() for path in link.parent.iterdir())


@pytest.mark.parametrize("conflict", ["legacy_directory", "different_directory", "private_file", "private_link"])
def test_prepare_conflicts_preserve_existing_artifacts(preparation, monkeypatch, conflict, tmp_path):
    link, private = preparation
    private.mkdir()
    target = private / "custom.jsonl"
    target.write_text("preserve target")
    monkeypatch.setenv("AGENTIC_IF_PREPARED_PATH", str(target))
    if conflict == "legacy_directory":
        link.mkdir()
        (link / "prepared.jsonl").symlink_to(target)
    elif conflict == "different_directory":
        other = tmp_path / "other"
        other.mkdir()
        link.symlink_to(other, target_is_directory=True)
    elif conflict == "private_file":
        (private / "prepared.jsonl").write_text("preserve conflicting input")
    else:
        (private / "prepared.jsonl").symlink_to(tmp_path / "elsewhere.jsonl")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file() and not path.is_symlink()}
    with pytest.raises(ValueError, match="reconcile"):
        prepare.prepare()
    assert all(path.read_bytes() == original for path, original in before.items())
    assert target.read_text() == "preserve target"


@pytest.mark.parametrize("basename", ["prepared_prepare.jsonl", "prepared_metrics.json"])
def test_prepare_rejects_names_reserved_for_collation(preparation, monkeypatch, basename):
    _, private = preparation
    private.mkdir()
    target = private / basename
    target.write_text("preserve target")
    monkeypatch.setenv("AGENTIC_IF_PREPARED_PATH", str(target))
    with pytest.raises(ValueError, match="sidecar"):
        prepare.prepare()
    assert target.read_text() == "preserve target"
