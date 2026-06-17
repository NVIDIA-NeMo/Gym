# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import tomllib

import pytest
from pytest import MonkeyPatch

import nemo_gym.cli.legacy as legacy
from nemo_gym import PARENT_DIR


def _legacy_scripts() -> list[tuple[str, str]]:
    """All console scripts whose name is a legacy ng_*/nemo_gym_* alias."""
    with (PARENT_DIR / "pyproject.toml").open("rb") as f:
        scripts = tomllib.load(f)["project"]["scripts"]
    return [(name, target) for name, target in scripts.items() if name.startswith(("ng_", "nemo_gym_"))]


LEGACY_SCRIPTS = _legacy_scripts()


class TestLegacyDeprecation:
    """Remove these tests once the legacy commands are removed."""

    def test_legacy_scripts_were_discovered(self) -> None:
        # Guard so the parametrized test below can't pass vacuously if discovery breaks.
        assert len(LEGACY_SCRIPTS) > 1

    @pytest.mark.parametrize("name, target", LEGACY_SCRIPTS)
    def test_legacy_command_shows_deprecation(self, monkeypatch: MonkeyPatch, capsys, name: str, target: str) -> None:
        # Every legacy alias must route through the shim, which prints a deprecation notice and keeps working.
        assert target == "nemo_gym.cli.legacy:main", f"{name} should route through the legacy shim"

        # Stub the actual execution paths so nothing real runs.
        monkeypatch.setattr(legacy, "gym_main", lambda: None)
        monkeypatch.setattr(legacy, "dispatch", lambda *a, **k: None)
        monkeypatch.setattr(sys, "argv", [name])

        legacy.main()

        assert "deprecated" in capsys.readouterr().err
