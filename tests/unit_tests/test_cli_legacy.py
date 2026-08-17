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

import sys
import tomllib

import pytest
from pytest import MonkeyPatch

import nemo_gym.cli.legacy as legacy
import nemo_gym.cli.main as cli_main
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

    def test_legacy_mapping_is_populated(self) -> None:
        # Guard so the parametrized test below can't pass vacuously if LEGACY is emptied.
        assert len(legacy.LEGACY) > 1

    @pytest.mark.parametrize("key, tokens", list(legacy.LEGACY.items()))
    def test_legacy_tokens_resolve_to_real_gym_command(
        self, monkeypatch: MonkeyPatch, capsys, key: str, tokens: list[str]
    ) -> None:
        # The tests above confirm each alias routes through the shim, but not that the mapped tokens
        # are a real `gym` command. Feed every mapping through the actual parser so a typo in the
        # mapping (e.g. ["env", "ruun"]) fails here instead of only at user runtime.
        monkeypatch.setattr(cli_main, "dispatch", lambda target, overrides: None)
        monkeypatch.setattr(sys, "argv", ["gym", *tokens])
        try:
            cli_main.main()
        except SystemExit as exc:
            # `gym --help` is the only mapping that legitimately exits 0 (argparse prints help).
            if tokens == ["--help"]:
                assert exc.code == 0
                return
            # Many commands exit because they bare-resolve without their required flags; that still
            # proves the command path resolved. Only an unresolved command (a typo in the mapping) is a
            # failure, which argparse reports as an "invalid choice".
            assert "invalid choice" not in capsys.readouterr().err, (
                f"`gym {' '.join(tokens)}` (legacy `{key}`) is not a valid command"
            )

    def test_unknown_alias_exits_nonzero(self, monkeypatch: MonkeyPatch, capsys) -> None:
        # An alias with no LEGACY mapping (stale script or user typo) must fail loudly, not KeyError.
        monkeypatch.setattr(legacy, "gym_main", lambda: None)
        monkeypatch.setattr(legacy, "dispatch", lambda *a, **k: None)
        monkeypatch.setattr(sys, "argv", ["ng_does_not_exist"])

        with pytest.raises(SystemExit) as exc_info:
            legacy.main()

        assert exc_info.value.code == 1
        assert "no known `gym` equivalent" in capsys.readouterr().err
