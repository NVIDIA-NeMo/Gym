# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nemo_gym.config_types import ConfigError
from nemo_gym.environment import artifacts


DIGEST = "sha256:" + "a" * 64


def git(directory, *args):
    return subprocess.run(
        ["git", "-C", str(directory), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def hub(tmp_path, monkeypatch):
    for name, value in {
        "GIT_AUTHOR_NAME": "Test Publisher",
        "GIT_AUTHOR_EMAIL": "publisher@example.test",
        "GIT_COMMITTER_NAME": "Test Publisher",
        "GIT_COMMITTER_EMAIL": "publisher@example.test",
    }.items():
        monkeypatch.setenv(name, value)
    remote = tmp_path / "hub.git"
    remote.mkdir()
    git(remote, "init", "--bare")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    git(checkout, "init", "-b", "main")
    (checkout / "catalog.json").write_text('{"environments": []}\n')
    git(checkout, "add", ".")
    git(checkout, "commit", "-m", "Initial catalog")
    git(checkout, "remote", "add", "origin", str(remote))
    git(checkout, "push", "origin", "main")
    git(checkout, "checkout", "-b", "drafts")
    git(checkout, "push", "origin", "drafts")
    monkeypatch.setattr(artifacts, "HUB_REPOSITORY", str(remote))
    package = tmp_path / "package"
    package.mkdir()
    metadata = {
        "manifest_path": "manifest.yaml",
        "config_path": "config.yaml",
        "name": "alpha",
        "version": "1.0.0",
        "kind": "environment",
    }
    (package / "gym-package.json").write_text(json.dumps(metadata))
    (package / "config.yaml").write_text("{}\n")
    manifest = SimpleNamespace(
        name="alpha",
        version="1.0.0",
        kind=SimpleNamespace(value="environment"),
        domain=SimpleNamespace(value="agent"),
        description="Packaged description",
        sandbox=None,
        datasets=[SimpleNamespace(type=SimpleNamespace(value="example"), name="example")],
        model_server="policy",
    )
    monkeypatch.setattr(artifacts, "load_manifest", Mock(return_value=manifest))
    monkeypatch.setattr(artifacts, "_verify_package", Mock(return_value=metadata))
    monkeypatch.setattr(artifacts, "pull_environment_package", Mock(return_value=package))
    monkeypatch.setattr(artifacts, "_oras", Mock(return_value=subprocess.CompletedProcess([], 0, DIGEST + "\n", "")))
    return SimpleNamespace(remote=remote, checkout=checkout, package=package)


def test_submit_stages_release_without_changing_main(hub):
    main = git(hub.remote, "rev-parse", "main")
    reference = artifacts.HUB_REGISTRY + "/alice/alpha:1.0.0"
    url = artifacts.submit_environment_package(reference)
    record = json.loads(git(hub.remote, "show", "drafts:submissions/alice--alpha.json"))
    assert record["id"] == "alice/alpha"
    assert record["version"] == "1.0.0"
    assert record["status"] == "submitted"
    assert record["tag"] == reference
    assert record["digest_reference"] == reference.rsplit(":", 1)[0] + "@" + DIGEST
    assert record["description"] == "Packaged description"
    assert git(hub.remote, "rev-parse", "main") == main
    assert url.startswith(artifacts.HUB_URL)
    artifacts.pull_environment_package.assert_called_once()
    assert artifacts.pull_environment_package.call_args.args[0] == record["digest_reference"]


def test_submit_preserves_review_edits_and_retry_is_idempotent(hub):
    record = {
        "id": "alice/alpha",
        "title": "Reviewer title",
        "description": "Reviewer description",
        "tags": ["curated"],
        "run": {"model_type": "vllm_model"},
        "status": "published",
        "version": "0.9.0",
        "digest_reference": artifacts.HUB_REGISTRY + "/alice/alpha@sha256:" + "b" * 64,
        "comments": [{"body": "Please keep this discussion"}],
        "runs": [{"id": "previous-run", "version": "0.9.0"}],
    }
    submissions = hub.checkout / "submissions"
    submissions.mkdir()
    (submissions / "alice--alpha.json").write_text(json.dumps(record))
    git(hub.checkout, "add", ".")
    git(hub.checkout, "commit", "-m", "Reviewed metadata")
    git(hub.checkout, "push", "origin", "drafts")
    reference = artifacts.HUB_REGISTRY + "/alice/alpha:1.0.0"
    artifacts.submit_environment_package(reference)
    updated = json.loads(git(hub.remote, "show", "drafts:submissions/alice--alpha.json"))
    for key in ("title", "description", "tags", "run", "comments", "runs"):
        assert updated[key] == record[key]
    assert updated["status"] == "submitted"
    assert updated["version"] == "1.0.0"
    commit = git(hub.remote, "rev-parse", "drafts")
    artifacts.submit_environment_package(reference)
    assert git(hub.remote, "rev-parse", "drafts") == commit


@pytest.mark.parametrize("reference", ["registry.example/alice/alpha:1.0.0", "extra/alice/alpha:1.0.0"])
def test_submit_rejects_wrong_registry_and_nested_identity(hub, reference):
    if reference.startswith("extra/"):
        reference = artifacts.HUB_REGISTRY + "/" + reference
    with pytest.raises(ConfigError):
        artifacts.submit_environment_package(reference)
    artifacts.pull_environment_package.assert_not_called()


def test_submit_rejects_colliding_draft_filename(hub):
    submissions = hub.checkout / "submissions"
    submissions.mkdir()
    (submissions / "alice--team--alpha.json").write_text(json.dumps({"id": "alice/team--alpha"}))
    git(hub.checkout, "add", ".")
    git(hub.checkout, "commit", "-m", "Existing namespace")
    git(hub.checkout, "push", "origin", "drafts")
    before = git(hub.remote, "rev-parse", "drafts")
    with pytest.raises(ConfigError):
        artifacts.submit_environment_package(artifacts.HUB_REGISTRY + "/alice--team/alpha:1.0.0")
    assert git(hub.remote, "rev-parse", "drafts") == before


def test_submit_retries_after_concurrent_draft_creation(hub, monkeypatch):
    original_run = subprocess.run
    concurrent_push = False

    def run(command, *args, **kwargs):
        nonlocal concurrent_push
        if command[0] == "git" and "push" in command and not concurrent_push:
            concurrent_push = True
            submissions = hub.checkout / "submissions"
            submissions.mkdir()
            (submissions / "alice--alpha.json").write_text(
                json.dumps(
                    {
                        "id": "alice/alpha",
                        "title": "Concurrent reviewer title",
                        "description": "Concurrent reviewer description",
                        "comments": [{"body": "Keep me"}],
                        "runs": [],
                        "tags": [],
                        "run": {"model_type": "vllm_model"},
                        "status": "draft",
                    }
                )
            )
            git(hub.checkout, "add", ".")
            git(hub.checkout, "commit", "-m", "Concurrent reviewer edit")
            git(hub.checkout, "push", "origin", "drafts")
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    artifacts.submit_environment_package(artifacts.HUB_REGISTRY + "/alice/alpha:1.0.0")
    updated = json.loads(git(hub.remote, "show", "drafts:submissions/alice--alpha.json"))
    assert concurrent_push
    assert updated["title"] == "Concurrent reviewer title"
    assert updated["comments"] == [{"body": "Keep me"}]
    assert updated["status"] == "submitted"
    assert updated["version"] == "1.0.0"


def test_submit_rejects_symlinked_submission_directory(hub, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (hub.checkout / "submissions").symlink_to(outside, target_is_directory=True)
    git(hub.checkout, "add", ".")
    git(hub.checkout, "commit", "-m", "Symlinked submission directory")
    git(hub.checkout, "push", "origin", "drafts")
    before = git(hub.remote, "rev-parse", "drafts")
    with pytest.raises(ConfigError):
        artifacts.submit_environment_package(artifacts.HUB_REGISTRY + "/alice/alpha:1.0.0")
    assert list(outside.iterdir()) == []
    assert git(hub.remote, "rev-parse", "drafts") == before
