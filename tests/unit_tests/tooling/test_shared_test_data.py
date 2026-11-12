# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate shared test resources and their CI environment configuration."""

import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from scripts.ci.check_test_data import (
    IFBENCH_COMMIT,
    REQUIRED_FILES,
    SACREBLEU_MODEL,
    check_test_data,
    prepare_nltk_cache,
)


@pytest.fixture
def shared_data(tmp_path, monkeypatch):
    """Supply a complete bundle with a pinned synthetic model for inventory tests."""
    for relative in REQUIRED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    (tmp_path / REQUIRED_FILES[0]).write_text(IFBENCH_COMMIT + "\n")
    monkeypatch.setattr("scripts.ci.check_test_data.SACREBLEU_MODEL_SHA256", hashlib.sha256(b"fixture\n").hexdigest())
    return tmp_path


def test_complete_inventory_passes(shared_data):
    """Accept a complete bundle containing the expected tokenizer bytes."""
    check_test_data(shared_data)


def test_inventory_rejects_corrupted_tokenizer(shared_data):
    """A nonempty replacement model must fail the pinned checksum check."""
    (shared_data / SACREBLEU_MODEL).write_bytes(b"different tokenizer")
    with pytest.raises(ValueError, match="FLORES-200 tokenizer checksum mismatch"):
        check_test_data(shared_data)


def test_inventory_exports_local_tokenizer_and_private_nltk(shared_data, tmp_path, monkeypatch):
    """The GitHub preflight exports paths consumed by subsequent test steps."""
    from scripts.ci.check_test_data import main

    output = tmp_path / "github-env"
    monkeypatch.setenv("GITHUB_ENV", str(output))
    monkeypatch.setattr(sys, "argv", ["check_test_data.py", str(shared_data), "--nltk-cache", str(tmp_path / "cache")])
    main()
    environment = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert environment["SACREBLEU"] == str(shared_data / "sacrebleu")
    assert (Path(environment["NLTK_DATA"]) / "corpora/stopwords/english").read_bytes() == b"fixture\n"


def test_worker_environment_uses_supplied_tokenizer_root(tmp_path):
    """Native CI workers point SacreBLEU at the provider's read-only data mount."""
    script = Path(__file__).resolve().parents[3] / "scripts/ci/test_worker_env.sh"
    root = tmp_path / "mounted data"
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s" "$SACREBLEU"', "workers", str(script)],
        env=dict(os.environ, NEMO_GYM_TEST_DATA_ROOT=f"{root}/", SACREBLEU="/unused"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == str(root / "sacrebleu")


@pytest.mark.parametrize("relative", REQUIRED_FILES)
@pytest.mark.parametrize("state", ["missing", "empty", "directory"])
def test_inventory_rejects_each_unusable_file(shared_data, relative, state):
    path = shared_data / relative
    path.unlink()
    if state == "empty":
        path.touch()
    elif state == "directory":
        path.mkdir()
    with pytest.raises(FileNotFoundError, match="Missing or empty Gym test data"):
        check_test_data(shared_data)


def test_inventory_rejects_wrong_revision(shared_data):
    (shared_data / REQUIRED_FILES[0]).write_text("wrong")
    with pytest.raises(ValueError, match="revision mismatch"):
        check_test_data(shared_data)


def test_nltk_cache_copies_shared_data_without_weakening_ownership_checks(shared_data, tmp_path):
    source = shared_data / "nltk_data"
    original = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    for path in (source, *source.rglob("*")):
        path.chmod(0o555 if path.is_dir() else 0o444)
    cache = prepare_nltk_cache(shared_data, tmp_path / "private-cache")
    assert {path.relative_to(cache): path.read_bytes() for path in cache.rglob("*") if path.is_file()} == original
    for path in (cache, *cache.rglob("*")):
        assert path.stat().st_uid == os.getuid()
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)
    assert {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()} == original


@pytest.mark.parametrize(
    ("workflow", "job", "workload"),
    [
        ("unit-tests.yml", "test", "Core unit tests"),
        ("unit-tests.yml", "sandbox-tests", "Sandbox unit tests"),
        ("unit-tests.yml", "changed-server-tests", "Changed server tests"),
        ("unit-tests.yml", "server-suite", "Server tests (shard"),
        ("full-test-suite.yml", "server-suite", "Server tests (shard"),
    ],
)
def test_ci_checks_existing_arc_mount_before_native_tests(workflow, job, workload):
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load((root / ".github/workflows" / workflow).read_text())["jobs"][job]
    assert config["runs-on"] == "nemo-ci-aws-cpu-4-16"
    data = "/mnt/s3-data-ro/nemo-fw/TestData/nemo_gym/v1"
    assert config["env"]["NEMO_GYM_TEST_DATA_ROOT"] == data
    assert config["env"]["NLTK_DATA"] == f"{data}/nltk_data"
    steps = config["steps"]
    inventory = [
        index
        for index, step in enumerate(steps)
        if step.get("run", "").startswith("python3 scripts/ci/check_test_data.py --nltk-cache ")
    ]
    native = next(index for index, step in enumerate(steps) if step.get("name", "").startswith(workload))
    assert len(inventory) == 1 and inventory[0] < native
