# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
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
