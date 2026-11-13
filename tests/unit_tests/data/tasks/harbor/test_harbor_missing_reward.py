# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A dataset's ``missing_reward.py`` beside ``dataset.toml``: loading it, and what it and the reward keys add to the
task digest. What each verdict does to an episode is the Harbor server's (environment_servers/harbor/tests)."""

import hashlib
import json
from pathlib import Path

import pytest

from nemo_gym.data.tasks.harbor.dataset_config import GymTaskOverride
from nemo_gym.data.tasks.harbor.missing_reward import MissingRewardClassifierError, load_missing_reward_classifier
from nemo_gym.data.tasks.harbor.task import load_task


TASK_TOML = """schema_version = "1.4"

[environment]
docker_image = "ubuntu:24.04"
workdir = "/app"

[agent]
timeout_sec = 60

[verifier]
timeout_sec = 30
"""


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("raise RuntimeError('boom')", "failed to import"),
        ("x = 1", "does not export"),
        ("classify_missing_reward = 1", "is not callable"),
    ],
    ids=["import-fails", "no-export", "not-callable"],
)
def test_a_classifier_that_does_not_load_is_refused(tmp_path, source, fragment):
    (tmp_path / "missing_reward.py").write_text(source)
    with pytest.raises(MissingRewardClassifierError, match=fragment):
        load_missing_reward_classifier(tmp_path)


def test_a_dataset_without_a_classifier_loads_none(tmp_path):
    assert load_missing_reward_classifier(tmp_path) is None


def test_a_linked_classifier_is_refused(tmp_path):
    (tmp_path / "elsewhere.py").write_text("def classify_missing_reward(evidence):\n    return None\n")
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "missing_reward.py").symlink_to(tmp_path / "elsewhere.py")
    with pytest.raises(MissingRewardClassifierError, match="escapes the dataset|symbolic link"):
        load_missing_reward_classifier(tmp_path / "ds")


def _dataset(tmp_path: Path, defaults: str, *, classifier: str | None = None) -> Path:
    """One task in a dataset whose dataset.toml holds ``defaults`` in [gym.defaults]."""
    task = tmp_path / "ds" / "copy"
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text(TASK_TOML)
    (task / "instruction.md").write_text("Copy input.txt to output.txt.\n")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    (task.parent / "dataset.toml").write_text(f"[gym.defaults]\n{defaults}\n")
    if classifier is not None:
        (task.parent / "missing_reward.py").write_text(classifier)
    return task


def test_a_shaping_dataset_that_sets_no_reward_key_keeps_its_digest_from_before_the_keys(tmp_path):
    task = load_task(_dataset(tmp_path, "timeout_multiplier = 2.0"))
    # The digest formula of the release before the reward keys existed, written out.
    slice_before = {
        "defaults": {"timeout_multiplier": 2.0, "resource_multiplier": 1.0},
        "task": GymTaskOverride().model_dump(mode="json"),
    }
    expected = hashlib.sha256(f"{task.folder_digest}\n{json.dumps(slice_before, sort_keys=True)}".encode()).hexdigest()
    assert task.digest == expected


def test_a_reward_key_or_a_classifier_changes_the_task_digest(tmp_path):
    plain = load_task(_dataset(tmp_path / "a", "timeout_multiplier = 2.0")).digest
    txt = load_task(_dataset(tmp_path / "b", 'timeout_multiplier = 2.0\nreward_file_precedence = "txt"')).digest
    first = load_task(_dataset(tmp_path / "c", "timeout_multiplier = 2.0", classifier="x = 1\n")).digest
    second = load_task(_dataset(tmp_path / "d", "timeout_multiplier = 2.0", classifier="x = 2\n")).digest
    assert len({plain, txt, first, second}) == 4
    # A classifier alone, in a dataset whose table shapes nothing, still enters the digest.
    bare = load_task(_dataset(tmp_path / "e", "num_repeats = 1", classifier="x = 1\n"))
    assert bare.digest != bare.folder_digest
