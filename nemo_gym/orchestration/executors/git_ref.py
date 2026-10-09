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
import logging
import os
import re
import subprocess
import tempfile

from nemo_gym.orchestration.api import SubmitConfig


logger = logging.getLogger(__name__)

# Set to "true" to skip the check, e.g. when the submitting machine cannot reach the repo.
NEMO_GYM_SUBMIT_NO_REF_CHECK_ENV_VAR_NAME = "NEMO_GYM_SUBMIT_NO_REF_CHECK"
_GIT_TIMEOUT_SECONDS = 10
_HEX = re.compile(r"[0-9a-f]{4,40}")
_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "true"}


def _git(*args: str, cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, timeout=_GIT_TIMEOUT_SECONDS, env=_GIT_ENV, cwd=cwd
    )


def validate_gym_install_ref(config: SubmitConfig) -> None:
    """Fail the submit early when `driver.gym_install.ref` cannot be found in `repo`.

    A branch or tag is looked up with one `git ls-remote`, which transfers only ref names. A full
    commit hash that is not a branch/tag tip is probed with a depth-1, blob-less fetch. Any git
    failure (unreachable or nonexistent repo, auth, timeout, git missing) also fails the submit;
    set NEMO_GYM_SUBMIT_NO_REF_CHECK=true to skip the check.
    """
    install = config.driver.gym_install
    if install is None or os.environ.get(NEMO_GYM_SUBMIT_NO_REF_CHECK_ENV_VAR_NAME, "").lower() == "true":
        return
    repo, ref = install.repo, install.ref
    try:
        found = _ref_exists(repo, ref)
    except (OSError, subprocess.SubprocessError) as e:
        raise ValueError(f"Could not verify driver.gym_install.ref {ref!r} in {repo}: {e}") from e
    if not found:
        raise ValueError(
            f"driver.gym_install.ref {ref!r} was not found in {repo}. "
            "Give an existing branch, tag, or full commit hash."
        )


def _ref_exists(repo: str, ref: str) -> bool:
    listing = _git("ls-remote", "--heads", "--tags", repo)
    if listing.returncode != 0:
        raise ValueError(f"Could not list {repo} to verify driver.gym_install.ref {ref!r}: {listing.stderr.strip()}")

    wanted = {f"refs/heads/{ref}", f"refs/tags/{ref}"}
    is_hex = _HEX.fullmatch(ref.lower()) is not None
    for line in listing.stdout.splitlines():
        sha, _, name = line.partition("\t")
        if name.removesuffix("^{}") in wanted or (is_hex and sha.startswith(ref.lower())):
            return True

    if not _FULL_SHA.fullmatch(ref.lower()):
        return False

    with tempfile.TemporaryDirectory(prefix="gym-ref-check-") as tmp:
        _git("init", "-q", cwd=tmp)
        fetched = _git("fetch", "-q", "--depth=1", "--filter=blob:none", repo, ref.lower(), cwd=tmp)
    if fetched.returncode != 0:
        logger.debug("Fetching %s from %s failed: %s", ref, repo, fetched.stderr.strip())
    return fetched.returncode == 0
