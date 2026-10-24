# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private-data trust boundaries use independent synthetic bytes only."""

import hashlib
from pathlib import Path

import pytest

from resources_servers.agentic_if_realworld.benchmark import registry


TRACKING_URI = "https://registry.example.invalid/api/v4/projects/17/ml/mlflow"


@pytest.fixture
def source(monkeypatch):
    monkeypatch.setattr(registry, "APPROVED_TRACKING_URI_SHA256", hashlib.sha256(TRACKING_URI.encode()).hexdigest())
    values = {"input.jsonl": b'{"synthetic":true}\n', "criteria.jsonl": b'{"literal":"OK."}\n'}
    monkeypatch.setattr(
        registry,
        "release_artifacts",
        lambda: {name: hashlib.sha256(value).hexdigest() for name, value in values.items()},
    )
    monkeypatch.setattr(
        registry,
        "get_global_config_dict",
        lambda: {
            "mlflow_tracking_uri": TRACKING_URI,
            "mlflow_tracking_token": "synthetic-registry-token",
        },
    )
    return values


def test_verified_download_is_private_and_reused(source, tmp_path, monkeypatch):
    calls = []

    def download(config):
        calls.append(config.artifact_fpath)
        Path(config.output_fpath).write_bytes(source[config.artifact_fpath])

    monkeypatch.setattr(registry, "download_jsonl_dataset", download)
    directory = registry.fetch_release(tmp_path / "cache")
    assert set(calls) == set(source)
    for name, value in source.items():
        assert (directory / name).read_bytes() == value
        assert (directory / name).stat().st_mode & 0o077 == 0
    assert registry.fetch_release(directory) == directory
    assert len(calls) == len(source)


def test_wrong_download_never_replaces_existing_bytes(source, tmp_path, monkeypatch):
    directory = tmp_path / "cache"
    directory.mkdir()
    (directory / "input.jsonl").write_bytes(b"previous bytes")
    monkeypatch.setattr(
        registry, "download_jsonl_dataset", lambda config: Path(config.output_fpath).write_bytes(b"error page")
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        registry.fetch_release(directory)
    assert (directory / "input.jsonl").read_bytes() == b"previous bytes"
    assert not list(directory.glob("*.download"))


@pytest.mark.parametrize(
    "uri",
    [
        "https://huggingface.co/datasets/private",
        "https://registry.example.invalid.attacker.example/api/v4/projects/17/ml/mlflow",
        "http://registry.example.invalid/api/v4/projects/17/ml/mlflow",
        "https://registry.example.invalid/api/v4/projects/18/ml/mlflow",
        TRACKING_URI + "?token=synthetic",
    ],
)
def test_external_registry_is_rejected_before_network(source, tmp_path, monkeypatch, uri):
    monkeypatch.setattr(registry, "get_global_config_dict", lambda: {"mlflow_tracking_uri": uri})
    monkeypatch.setattr(registry, "download_jsonl_dataset", lambda config: pytest.fail("External network call"))
    with pytest.raises(ValueError, match="approved registry"):
        registry.fetch_release(tmp_path / "cache")


@pytest.mark.parametrize("symlink", [False, True])
def test_git_checkout_and_symlink_destinations_are_rejected(tmp_path, symlink):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    target = checkout / "private"
    if symlink:
        linked = tmp_path / "linked"
        linked.symlink_to(target)
        target = linked
    with pytest.raises(ValueError, match="outside Git"):
        registry.private_path(target)
