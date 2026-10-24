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
"""Reject profile changes before starting a benchmark trial."""

import pytest
from pydantic import ValidationError

from environments.skill_trigger.agent import SkillOptions, check_private_mounts
from environments.skill_trigger.scoring import EvidenceError


@pytest.mark.parametrize("source", [".", "release", "release/manifest.json", "private/other-trial/evidence.json"])
def test_mount_cannot_expose_private_directories_or_individual_files(tmp_path, source):
    with pytest.raises(EvidenceError, match="private evidence"):
        check_private_mounts(sources=[tmp_path / source], hidden=[tmp_path / "release", tmp_path / "private"])


def test_mount_alias_cannot_expose_private_file(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    (release / "manifest.json").write_text("hidden labels")
    alias = tmp_path / "alias"
    alias.symlink_to(release / "manifest.json")
    with pytest.raises(EvidenceError, match="private evidence"):
        check_private_mounts(sources=[alias], hidden=[release])


def test_trial_logs_and_similarly_named_sibling_paths_remain_allowed(tmp_path):
    check_private_mounts(
        sources=[tmp_path / "trial/agent", tmp_path / "release-copy", tmp_path / "private-log"],
        hidden=[tmp_path / "release", tmp_path / "private"],
    )


@pytest.mark.parametrize("change", [{"provider": None}, {"api": None}, {"gateway_host": "0.0.0.0"}])
def test_pi_requires_explicit_api_identity_and_loopback(change):
    options = dict(
        release="release",
        manifest_sha256="hash",
        evidence_root="private",
        upstream_url="http://provider",
        api_key_env="TEST_KEY",
        gateway_host="127.0.0.1",
        harness="pi",
        provider="nvidia",
        api="openai-completions",
    )
    assert SkillOptions(**options).api == "openai-completions"
    with pytest.raises(ValidationError):
        SkillOptions(**(options | change))


def test_hermes_cannot_inherit_a_pi_api_override():
    with pytest.raises(ValidationError, match="frozen Responses API profiles"):
        SkillOptions(
            release="release",
            manifest_sha256="hash",
            evidence_root="private",
            upstream_url="http://provider",
            api_key_env="TEST_KEY",
            gateway_host="172.17.0.1",
            harness="hermes",
            api="openai-completions",
        )
