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
import shutil
import subprocess
from types import SimpleNamespace

import pytest

import nemo_gym.orchestration.executors.git_ref as git_ref
from nemo_gym.orchestration.executors.git_ref import validate_gym_install_ref


pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


@pytest.fixture(autouse=True)
def _no_waits_and_no_own_clone(monkeypatch, tmp_path):
    # Retries do not wait in tests, and the local check looks at no real clone unless a test gives it one.
    sleeps = []
    monkeypatch.setattr(git_ref, "time", SimpleNamespace(sleep=sleeps.append))
    monkeypatch.setattr(git_ref, "_GYM_CHECKOUT", tmp_path / "no-clone")
    return sleeps


def _config(repo: str, ref: str):
    return SimpleNamespace(driver=SimpleNamespace(gym_install=SimpleNamespace(repo=repo, ref=ref)))


@pytest.fixture
def repo(tmp_path):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True).stdout

    git("init", "-q", "-b", "main")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "one")
    git("tag", "v1")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "two")
    git("config", "uploadpack.allowAnySHA1InWant", "true")
    first = git("rev-parse", "v1").strip()
    git("checkout", "-q", "--detach", "HEAD")  # leaves `first` reachable only through the v1 tag
    git("tag", "-d", "v1")
    return SimpleNamespace(url=str(tmp_path), first=first, head=git("rev-parse", "HEAD").strip())


def test_branch_exists(repo):
    validate_gym_install_ref(_config(repo.url, "main"))


def test_head_hash_and_prefix_exist_via_tips(repo):
    validate_gym_install_ref(_config(repo.url, repo.head))


def test_non_tip_full_hash_is_found_by_fetch(repo):
    validate_gym_install_ref(_config(repo.url, repo.first))


def test_missing_branch_fails(repo):
    with pytest.raises(ValueError, match="not found"):
        validate_gym_install_ref(_config(repo.url, "nope"))


def test_missing_full_hash_fails(repo):
    with pytest.raises(ValueError, match="not found"):
        validate_gym_install_ref(_config(repo.url, "0" * 40))


def test_unreachable_repo_fails(tmp_path):
    with pytest.raises(ValueError, match="Could not list"):
        validate_gym_install_ref(_config(str(tmp_path / "missing"), "main"))


def test_a_network_timeout_is_retried_with_backoff_and_then_succeeds(repo, monkeypatch, _no_waits_and_no_own_clone):
    real_git, calls = git_ref._git, []

    def flaky(*args, **kwargs):
        if args[0] == "ls-remote":
            calls.append(args)
            if len(calls) <= 2:
                raise subprocess.TimeoutExpired(["git", *args], 10)
        return real_git(*args, **kwargs)

    monkeypatch.setattr(git_ref, "_git", flaky)

    validate_gym_install_ref(_config(repo.url, "main"))

    assert len(calls) == 3 and _no_waits_and_no_own_clone == [10, 30]


def test_a_network_check_that_keeps_timing_out_says_so_and_not_that_the_ref_is_missing(
    repo, monkeypatch, _no_waits_and_no_own_clone
):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(["git", *args], 10)

    monkeypatch.setattr(git_ref, "_git", timeout)

    with pytest.raises(ValueError) as raised:
        validate_gym_install_ref(_config(repo.url, repo.head))

    message = str(raised.value)
    assert "network check" in message and "not evidence that the ref is missing" in message
    assert "failed 4 time(s)" in message and "not found" not in message
    assert _no_waits_and_no_own_clone == [10, 30, 60]


def test_a_missing_git_binary_is_not_retried(repo, monkeypatch, _no_waits_and_no_own_clone):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(git_ref, "_git", missing)

    with pytest.raises(ValueError, match="failed 1 time"):
        validate_gym_install_ref(_config(repo.url, "main"))
    assert _no_waits_and_no_own_clone == []


def _own_clone(tmp_path, origin_url):
    clone = tmp_path / "own-clone"
    subprocess.run(["git", "clone", "-q", origin_url, str(clone)], check=True, capture_output=True)
    return clone


def _unreachable(*args, **kwargs):
    raise git_ref._NetworkCheckError("Could not list the repo: timed out")


def test_after_every_attempt_failed_a_commit_the_freshly_pruned_own_clone_has_is_accepted(
    repo, tmp_path, monkeypatch, caplog, _no_waits_and_no_own_clone
):
    monkeypatch.setattr(git_ref, "_GYM_CHECKOUT", _own_clone(tmp_path, repo.url))
    monkeypatch.setattr(git_ref, "_ref_exists", _unreachable)

    validate_gym_install_ref(_config(repo.url + "/", repo.head))

    assert _no_waits_and_no_own_clone == [10, 30, 60]
    assert "Local fallback used" in caplog.text


def test_a_commit_only_in_a_stale_remote_tracking_ref_is_rejected_after_the_prune(
    repo, tmp_path, monkeypatch, _no_waits_and_no_own_clone
):
    def origin(*args):
        subprocess.run(["git", "-C", repo.url, *args], check=True, capture_output=True)

    origin("checkout", "-q", "-b", "gone")
    origin("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "gone")
    gone = subprocess.run(
        ["git", "-C", repo.url, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    clone = _own_clone(tmp_path, repo.url)
    origin("checkout", "-q", "--detach", repo.head)
    origin("branch", "-D", "gone")
    monkeypatch.setattr(git_ref, "_GYM_CHECKOUT", clone)
    monkeypatch.setattr(git_ref, "_ref_exists", _unreachable)
    stale = subprocess.run(
        ["git", "-C", str(clone), "for-each-ref", "--contains", gone, "refs/remotes"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "refs/remotes/origin/gone" in stale  # the clone's ref is stale before the check runs

    with pytest.raises(ValueError, match="network check"):
        validate_gym_install_ref(_config(repo.url, gone))


def test_no_local_fallback_when_the_prune_fetch_fails(repo, tmp_path, monkeypatch):
    clone = _own_clone(tmp_path, repo.url)
    subprocess.run(["git", "-C", str(clone), "remote", "set-url", "origin", str(tmp_path / "moved")], check=True)
    subprocess.run(["git", "-C", str(clone), "config", "remote.origin.pushurl", repo.url], check=True)
    monkeypatch.setattr(git_ref, "_GYM_CHECKOUT", clone)

    # The fetch URL no longer matches the repo, and the repo is unreachable: nothing vouches for the commit.
    monkeypatch.setattr(git_ref, "_ref_exists", _unreachable)
    with pytest.raises(ValueError, match="network check"):
        validate_gym_install_ref(_config(repo.url, repo.head))


def test_a_reachable_repo_decides_even_when_the_own_clone_holds_a_stale_remote_branch(repo, tmp_path, monkeypatch):
    # The clone still has origin/gone after the branch, and its only commit, left the repo: the repo's answer counts.
    def origin(*args):
        subprocess.run(["git", "-C", repo.url, *args], check=True, capture_output=True)

    origin("checkout", "-q", "-b", "gone")
    origin("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "gone")
    gone = subprocess.run(
        ["git", "-C", repo.url, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    clone = _own_clone(tmp_path, repo.url)
    origin("checkout", "-q", "--detach", repo.head)
    origin("branch", "-D", "gone")
    origin("reflog", "expire", "--expire=now", "--all")
    origin("gc", "-q", "--prune=now")
    monkeypatch.setattr(git_ref, "_GYM_CHECKOUT", clone)
    stale = subprocess.run(
        ["git", "-C", str(clone), "for-each-ref", "--contains", gone, "refs/remotes"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "refs/remotes/origin/gone" in stale  # the clone still holds the deleted branch

    with pytest.raises(ValueError, match="not found"):
        validate_gym_install_ref(_config(repo.url, gone))


def test_an_unpushed_local_commit_is_still_checked_over_the_network(repo, tmp_path, monkeypatch):
    clone = _own_clone(tmp_path, repo.url)
    subprocess.run(
        [
            "git",
            "-C",
            str(clone),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "x",
        ],
        check=True,
    )
    local = subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    monkeypatch.setattr(git_ref, "_GYM_CHECKOUT", clone)

    with pytest.raises(ValueError, match="not found"):
        validate_gym_install_ref(_config(repo.url, local))


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("https://github.com/Org/Repo.git", "git@github.com:org/repo"),
        ("https://github.com/Org/Repo/", "ssh://git@github.com/org/repo.git"),
    ],
)
def test_one_repo_by_https_and_ssh_urls_is_the_same_repo(a, b):
    assert git_ref._repo_key(a) == git_ref._repo_key(b)


def test_no_gym_install_is_a_noop():
    validate_gym_install_ref(SimpleNamespace(driver=SimpleNamespace(gym_install=None)))


def test_env_var_skips_the_check(repo, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_SUBMIT_NO_REF_CHECK", "true")
    validate_gym_install_ref(_config(repo.url, "nope"))


def test_env_var_other_than_true_does_not_skip(repo, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_SUBMIT_NO_REF_CHECK", "0")
    with pytest.raises(ValueError, match="not found"):
        validate_gym_install_ref(_config(repo.url, "nope"))


def test_abbreviated_hash_down_to_four_chars_matches_a_tip(repo):
    validate_gym_install_ref(_config(repo.url, repo.head[:4]))


def test_abbreviated_hash_that_is_not_a_tip_fails(repo):
    with pytest.raises(ValueError, match="not found"):
        validate_gym_install_ref(_config(repo.url, repo.first[:4]))


def test_three_char_hex_is_a_name_and_must_exist(repo):
    with pytest.raises(ValueError, match="not found"):
        validate_gym_install_ref(_config(repo.url, "abc"))
