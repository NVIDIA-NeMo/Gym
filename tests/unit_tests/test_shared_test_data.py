# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml
from scripts.ci.check_test_data import IFBENCH_COMMIT, REQUIRED_FILES, check_test_data


@pytest.fixture
def shared_data(tmp_path):
    for relative in REQUIRED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    (tmp_path / REQUIRED_FILES[0]).write_text(IFBENCH_COMMIT + "\n")
    return tmp_path


def test_complete_inventory_passes(shared_data):
    check_test_data(shared_data)


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


@pytest.mark.parametrize(
    ("workflow", "job", "workload"),
    [
        ("unit-tests.yml", "test", "Core + sandbox unit tests (coverage gate)"),
        ("unit-tests.yml", "server-suite", "Server tests (shard"),
        ("full-test-suite.yml", "server-suite", "Server tests (shard"),
    ],
)
def test_ci_checks_existing_arc_mount_before_native_tests(workflow, job, workload):
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / ".github/workflows" / workflow).read_text())["jobs"][job]
    assert config["runs-on"] == "nemo-ci-aws-cpu-4-16"
    data = "/mnt/s3-data-ro/nemo-fw/TestData/nemo_gym/v1"
    assert config["env"]["NEMO_GYM_TEST_DATA_ROOT"] == data
    assert config["env"]["NLTK_DATA"] == f"{data}/nltk_data"
    steps = config["steps"]
    inventory = [
        index for index, step in enumerate(steps) if step.get("run") == "python3 scripts/ci/check_test_data.py"
    ]
    native = next(index for index, step in enumerate(steps) if step.get("name", "").startswith(workload))
    assert len(inventory) == 1 and inventory[0] < native
