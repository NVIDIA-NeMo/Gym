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
"""Files that Gym finds by path must still be there after a module moves."""

import tomllib
from pathlib import Path

import pytest

import nemo_gym
from nemo_gym.h2_ping_sidecar import launcher


REPO_ROOT = Path(nemo_gym.__file__).resolve().parents[1]
PACKAGE_DATA = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]


@pytest.mark.parametrize("pattern", PACKAGE_DATA["nemo_gym"])
def test_every_nemo_gym_package_data_pattern_matches_a_file(pattern: str) -> None:
    # With git metadata, setuptools-scm ships every tracked file under nemo_gym/ whatever the patterns say.
    # The patterns matter only for a build without it, such as a wheel built from an unpacked sdist.
    # setuptools ignores a pattern that matches no file, so this catches a pattern left behind by a move.
    assert list(Path(nemo_gym.__file__).parent.glob(pattern)), f"package-data pattern {pattern!r} matches no file"


def test_sidecar_source_dir_holds_the_go_module() -> None:
    assert (launcher.SIDECAR_SOURCE_DIR / "go.mod").is_file()
    assert (launcher.SIDECAR_SOURCE_DIR / "main.go").is_file()
