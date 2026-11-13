# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A dataset's optional classifier for a verifier run that wrote no reward.

By default a missing reward is the task's measured 0. A dataset whose graders can also fail for reasons that are not
the agent's (a judge service that refused the call, a grader that crashed) ships ``missing_reward.py`` beside its
``dataset.toml``, exporting ``classify_missing_reward(evidence)``. The function reads the verifier's logs on the
host and returns a :class:`MissingRewardVerdict`, or ``None`` to keep the measured 0.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from nemo_gym.data.tasks.harbor.errors import HarborTaskError
from nemo_gym.data.tasks.harbor.task_data import import_dataset_module


MISSING_REWARD_MODULE = "missing_reward.py"
MISSING_REWARD_EXPORT = "classify_missing_reward"
MISSING_REWARD_OUTCOMES = ("task", "infrastructure", "judge_failed")


@dataclass(frozen=True)
class MissingRewardEvidence:
    """What the classifier may read: host copies only, never the sandbox."""

    task_id: str
    # The episode's snapshot of the task folder (task.toml, tests/); its parent is a temporary folder.
    task_path: Path
    # The task's `[metadata]` table, where a dataset can record what its classifier needs to know.
    metadata: Mapping[str, Any]
    # The host copy of /logs/verifier: test-stdout.txt, the exit record and whatever else test.sh wrote there.
    logs_dir: Path
    # The host copy of /logs/artifacts as test.sh left it (links skipped), or None when the folder does not exist.
    artifacts_dir: Path | None
    return_code: int | None


@dataclass(frozen=True)
class MissingRewardVerdict:
    """``task``: a measured 0 under ``kind``. ``infrastructure``: a masked sample under ``kind``. ``judge_failed``:
    Gym's judge-failure class, a masked row routed to the failures file and run again on resume."""

    outcome: Literal["task", "infrastructure", "judge_failed"]
    kind: str
    reason: str


MissingRewardClassifier = Callable[[MissingRewardEvidence], MissingRewardVerdict | None]


class MissingRewardClassifierError(HarborTaskError):
    """The dataset's ``missing_reward.py`` does not load or does not satisfy the protocol."""


def _checked_dataset_file(dataset_root: Path, relative: str) -> Path:
    """Return a dataset file without following symlinks or allowing an escape from its root."""
    root = dataset_root.resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise MissingRewardClassifierError(f"{path}: path escapes the dataset")
    if path.is_symlink():
        raise MissingRewardClassifierError(f"{path}: dataset files must not be symbolic links")
    return path


def load_missing_reward_classifier(dataset_root: Path) -> MissingRewardClassifier | None:
    """Import ``<dataset_root>/missing_reward.py``; ``None`` when the dataset ships none."""
    path = _checked_dataset_file(Path(dataset_root), MISSING_REWARD_MODULE)
    if not path.is_file():
        return None
    try:
        module = import_dataset_module(path, prefix="nemo_gym_harbor_missing_reward")
    except Exception as exc:
        raise MissingRewardClassifierError(f"{path}: failed to import: {type(exc).__name__}: {exc}") from exc
    classifier = getattr(module, MISSING_REWARD_EXPORT, None)
    if classifier is None:
        raise MissingRewardClassifierError(f"{path} does not export `{MISSING_REWARD_EXPORT}`")
    if not callable(classifier):
        raise MissingRewardClassifierError(f"`{MISSING_REWARD_EXPORT}` in {path} is not callable")
    return classifier
