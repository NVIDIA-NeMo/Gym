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

from nemo_gym.orchestration.executors.git_ref import validate_gym_install_ref


pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


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
