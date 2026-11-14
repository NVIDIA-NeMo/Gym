# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The git commit of the nemo-gym that is running."""

import json
import logging
import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


logger = logging.getLogger(__name__)

# The nemo_gym package directory of the code that is running, however it got
# onto sys.path: an install, an editable checkout, or a plain source tree.
_PACKAGE_DIR = Path(__file__).resolve().parents[1]


def installed_gym_commit() -> str | None:
    """Return the git commit of the nemo-gym that is running, or None when it cannot be determined.

    A git install (``pip install git+...``) records its commit in the distribution
    metadata. Otherwise the running package's own source tree is asked: a git
    checkout reports its HEAD, suffixed ``-dirty`` when it has uncommitted changes.
    """
    commit = _git_install_commit()
    if commit is not None:
        return commit
    return _source_checkout_commit(_PACKAGE_DIR)


def _git_install_commit() -> str | None:
    """The commit recorded in the distribution's PEP 610 ``direct_url.json``, or None."""
    try:
        raw = distribution("nemo-gym").read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if raw is None:
        return None
    commit = (json.loads(raw).get("vcs_info") or {}).get("commit_id")
    return str(commit) if commit else None


def _source_checkout_commit(package_dir: Path) -> str | None:
    """HEAD of the git checkout ``package_dir`` is tracked in, ``-dirty`` when it has uncommitted changes."""
    try:
        tracked = _git(package_dir, "ls-files", "--error-unmatch", "__init__.py")
    except FileNotFoundError:
        return _unknown_commit("git is not available to inspect %s.", package_dir)
    if tracked.returncode != 0:
        return _unknown_commit("nemo-gym is not a git install and %s is not tracked in a git checkout.", package_dir)
    head = _git(package_dir, "rev-parse", "HEAD")
    status = _git(package_dir, "status", "--porcelain")
    if head.returncode != 0 or status.returncode != 0:
        return _unknown_commit("git failed in %s: %s", package_dir, (head.stderr or status.stderr).strip())
    commit = head.stdout.strip()
    return f"{commit}-dirty" if status.stdout.strip() else commit


def _git(directory: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True)


def _unknown_commit(reason: str, *args: object) -> None:
    logger.warning("Cannot determine the installed Gym commit: " + reason, *args)
    return None
