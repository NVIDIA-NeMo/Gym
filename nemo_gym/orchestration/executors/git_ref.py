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
import time
from pathlib import Path

from nemo_gym.orchestration.api import SubmitConfig


logger = logging.getLogger(__name__)

# Set to "true" to skip the check, e.g. when the submitting machine cannot reach the repo.
NEMO_GYM_SUBMIT_NO_REF_CHECK_ENV_VAR_NAME = "NEMO_GYM_SUBMIT_NO_REF_CHECK"
_GIT_TIMEOUT_SECONDS = 10
_HEX = re.compile(r"[0-9a-f]{4,40}")
_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "true"}
# Waits before each retry of the network check: a timeout or an unreachable host is often a short network hiccup.
_RETRY_WAITS_SECONDS = (10, 30, 60)
# The checkout this Gym runs from, when it is a git clone: a pinned commit its remote-tracking refs hold needs no network.
_GYM_CHECKOUT = Path(__file__).resolve().parents[3]


class _NetworkCheckError(Exception):
    """The repo could not be listed: a network, access or remote failure, not an answer about the ref."""


def _git(*args: str, cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, timeout=_GIT_TIMEOUT_SECONDS, env=_GIT_ENV, cwd=cwd
    )


def validate_gym_install_ref(config: SubmitConfig) -> None:
    """Fail the submit early when `driver.gym_install.ref` cannot be found in `repo`.

    A branch or tag is looked up with one `git ls-remote`, which transfers only ref names, and a
    full commit hash that is not a branch/tag tip is probed with a depth-1, blob-less fetch. A
    failure to reach the repo (timeout, unreachable or nonexistent repo, auth) is retried after each
    of `_RETRY_WAITS_SECONDS`; when every attempt fails, the submit fails with an error that names a
    network check, not a missing ref. A missing git binary is not retried.

    The repo always answers when it can be reached; a ref it does not have fails the submit. Only when
    every attempt failed to reach it is a full commit hash checked in this Gym's own clone of `repo`,
    and only after a `git fetch --prune` of that remote succeeds, so a remote branch deleted since the
    clone's last fetch cannot vouch for it; a hash such a fresh remote-tracking ref contains is accepted
    with a warning that the local fallback was used. Set NEMO_GYM_SUBMIT_NO_REF_CHECK=true to skip the
    check.
    """
    install = config.driver.gym_install
    if install is None or os.environ.get(NEMO_GYM_SUBMIT_NO_REF_CHECK_ENV_VAR_NAME, "").lower() == "true":
        return
    repo, ref = install.repo, install.ref
    attempts = len(_RETRY_WAITS_SECONDS) + 1
    for attempt in range(1, attempts + 1):
        try:
            found = _ref_exists(repo, ref)
            break
        except (OSError, subprocess.SubprocessError, _NetworkCheckError) as e:
            last = attempt == attempts or isinstance(e, FileNotFoundError)
            if last and _FULL_SHA.fullmatch(ref.lower()) and _pushed_in_local_clone(repo, ref.lower()):
                logger.warning(
                    "Local fallback used: could not verify driver.gym_install.ref %r in %s over the network after "
                    "%d attempt(s) (%s); a remote-tracking ref of this Gym's own clone of that repo contains it "
                    "after a fetch --prune of that remote.",
                    ref,
                    repo,
                    attempt,
                    e,
                )
                return
            if last:
                raise ValueError(
                    f"Could not verify driver.gym_install.ref {ref!r} in {repo}: the network check (git ls-remote) "
                    f"failed {attempt} time(s); last error: {e}. This is a network or access failure, not evidence "
                    f"that the ref is missing: submit again, or set {NEMO_GYM_SUBMIT_NO_REF_CHECK_ENV_VAR_NAME}=true "
                    "to skip the check."
                ) from e
            wait = _RETRY_WAITS_SECONDS[attempt - 1]
            logger.warning(
                "Verifying driver.gym_install.ref %r in %s failed (%s); retrying in %d s (attempt %d of %d).",
                ref,
                repo,
                e,
                wait,
                attempt + 1,
                attempts,
            )
            time.sleep(wait)
    if not found:
        raise ValueError(
            f"driver.gym_install.ref {ref!r} was not found in {repo}. "
            "Give an existing branch, tag, or full commit hash."
        )


def _ref_exists(repo: str, ref: str) -> bool:
    listing = _git("ls-remote", "--heads", "--tags", repo)
    if listing.returncode != 0:
        raise _NetworkCheckError(f"Could not list {repo}: {listing.stderr.strip()}")

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


def _repo_key(url: str) -> str:
    """`url` without scheme, user, `.git` and case, so an https and an ssh URL of one repo compare equal."""
    key = url.strip().rstrip("/").removesuffix(".git").lower()
    key = re.sub(r"^[a-z][a-z0-9+.-]*://([^@/]*@)?", "", key)
    return re.sub(r"^[^@/]+@([^:/]+):", r"\1/", key)


def _pushed_in_local_clone(repo: str, sha: str) -> bool:
    """Whether a remote-tracking ref of `repo` in this Gym's own clone contains commit `sha` right after a
    successful `git fetch --prune` of that remote: then `repo` serves it.

    Only remote-tracking refs count, so a local commit that was never pushed does not. A remote whose fetch fails is
    not consulted, so a stale ref (a branch deleted since the last fetch) never vouches for a commit. Any failure here
    (no clone, no such remote, git missing) answers False.
    """
    try:
        remotes = _git("config", "--get-regexp", r"^remote\..*\.url$", cwd=str(_GYM_CHECKOUT))
        if remotes.returncode != 0:
            return False
        names = [
            key.removeprefix("remote.").removesuffix(".url")
            for key, _, url in (line.partition(" ") for line in remotes.stdout.splitlines())
            if _repo_key(url) == _repo_key(repo)
        ]
        for name in names:
            if _git("fetch", "-q", "--prune", name, cwd=str(_GYM_CHECKOUT)).returncode != 0:
                continue
            refs = _git(
                "for-each-ref",
                "--contains",
                sha,
                "--format=%(refname)",
                f"refs/remotes/{name}",
                cwd=str(_GYM_CHECKOUT),
            )
            if refs.returncode == 0 and refs.stdout.strip():
                return True
    except (OSError, subprocess.SubprocessError):
        return False
    return False
