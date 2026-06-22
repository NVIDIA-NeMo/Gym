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
from pathlib import Path
from typing import List, Optional

from pytest import mark, raises

from nemo_gym.registry import ENVIRONMENTS_DIR, EnvironmentNotFoundError
from nemo_gym.suite_manager import (
    EmptySuiteError,
    Suite,
    SuiteNotFoundError,
    discover_suites,
    resolve_suite_config_paths,
)


def _make_env(environments_dir: Path, name: str) -> Path:
    env_dir = environments_dir / name
    env_dir.mkdir(parents=True)
    config_path = env_dir / "config.yaml"
    config_path.write_text(f"{name}:\n  resources_servers:\n    {name}:\n      domain: other\n")
    return config_path


def _write_suite(suites_dir: Path, name: str, environments: List[str], description: Optional[str] = None) -> Path:
    suites_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    if description is not None:
        lines.append(f"description: {description}")
    if environments:
        lines.append("environments:")
        lines.extend(f"  - {env}" for env in environments)
    else:
        lines.append("environments: []")
    path = suites_dir / f"{name}.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


class TestDiscoverSuites:
    def test_discovers_builtin_and_user_suites(self, tmp_path: Path) -> None:
        builtin, user = tmp_path / "builtin", tmp_path / "user"
        _write_suite(builtin, "coding", ["a"], description="Coding suite")
        _write_suite(user, "mine", ["b"])

        suites = discover_suites(builtin_dir=builtin, user_dir=user)

        assert set(suites) == {"coding", "mine"}
        assert suites["coding"].is_builtin is True
        assert suites["coding"].description == "Coding suite"
        assert suites["coding"].environments == ("a",)
        assert suites["mine"].is_builtin is False

    def test_user_suite_shadows_builtin_of_same_name(self, tmp_path: Path) -> None:
        builtin, user = tmp_path / "builtin", tmp_path / "user"
        _write_suite(builtin, "dup", ["builtin_env"])
        _write_suite(user, "dup", ["user_env"])

        suites = discover_suites(builtin_dir=builtin, user_dir=user)

        assert suites["dup"].is_builtin is False
        assert suites["dup"].environments == ("user_env",)

    def test_missing_directories_yield_no_suites(self, tmp_path: Path) -> None:
        assert discover_suites(builtin_dir=tmp_path / "nope", user_dir=tmp_path / "also_nope") == {}

    def test_non_suite_and_malformed_files_are_skipped(self, tmp_path: Path) -> None:
        suites_dir = tmp_path / "builtin"
        suites_dir.mkdir()
        # Mapping without an `environments` list -> not a suite.
        (suites_dir / "not_a_suite.yaml").write_text("description: just a note\n")
        # `environments` present but not a list -> not a suite.
        (suites_dir / "wrong_type.yaml").write_text("environments: nope\n")
        # Top-level list -> not a mapping -> not a suite.
        (suites_dir / "a_list.yaml").write_text("- a\n- b\n")
        # Unparseable YAML -> skipped, not raised.
        (suites_dir / "broken.yaml").write_text("environments: [unclosed\n")
        # A valid one to prove discovery still works around the bad files.
        _write_suite(suites_dir, "good", ["a"])

        suites = discover_suites(builtin_dir=suites_dir, user_dir=tmp_path / "user")

        assert set(suites) == {"good"}

    def test_non_string_environment_entries_are_dropped(self, tmp_path: Path) -> None:
        suites_dir = tmp_path / "builtin"
        suites_dir.mkdir()
        (suites_dir / "mixed.yaml").write_text("environments:\n  - good\n  - 123\n  - true\n")

        suites = discover_suites(builtin_dir=suites_dir, user_dir=tmp_path / "user")

        assert suites["mixed"].environments == ("good",)


class TestResolveSuiteConfigPaths:
    def test_expands_to_ordered_config_paths(self, tmp_path: Path) -> None:
        environments_dir = tmp_path / "environments"
        path_a = _make_env(environments_dir, "alpha")
        path_b = _make_env(environments_dir, "beta")
        builtin = tmp_path / "builtin"
        _write_suite(builtin, "both", ["alpha", "beta"])

        result = resolve_suite_config_paths(
            "both", builtin_dir=builtin, user_dir=tmp_path / "user", environments_dir=environments_dir
        )

        assert result == [str(path_a), str(path_b)]

    def test_duplicate_environments_are_deduped_preserving_order(self, tmp_path: Path) -> None:
        environments_dir = tmp_path / "environments"
        path_a = _make_env(environments_dir, "alpha")
        path_b = _make_env(environments_dir, "beta")
        builtin = tmp_path / "builtin"
        _write_suite(builtin, "dupes", ["alpha", "beta", "alpha"])

        result = resolve_suite_config_paths(
            "dupes", builtin_dir=builtin, user_dir=tmp_path / "user", environments_dir=environments_dir
        )

        assert result == [str(path_a), str(path_b)]

    def test_unknown_suite_raises_with_suggestion(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        _write_suite(builtin, "ultra_v3", ["alpha"])

        with raises(SuiteNotFoundError, match="Did you mean"):
            resolve_suite_config_paths(
                "ultra_v", builtin_dir=builtin, user_dir=tmp_path / "user", environments_dir=tmp_path / "environments"
            )

    def test_unknown_suite_without_close_match_lists_available(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        _write_suite(builtin, "coding", ["alpha"])

        with raises(SuiteNotFoundError, match="Available suites"):
            resolve_suite_config_paths(
                "zzzzz", builtin_dir=builtin, user_dir=tmp_path / "user", environments_dir=tmp_path / "environments"
            )

    def test_empty_suite_raises(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        _write_suite(builtin, "empty", [])

        with raises(EmptySuiteError, match="lists no environments"):
            resolve_suite_config_paths(
                "empty", builtin_dir=builtin, user_dir=tmp_path / "user", environments_dir=tmp_path / "environments"
            )

    def test_unknown_environment_in_suite_raises_with_suite_context(self, tmp_path: Path) -> None:
        environments_dir = tmp_path / "environments"
        _make_env(environments_dir, "alpha")
        builtin = tmp_path / "builtin"
        _write_suite(builtin, "broken", ["alpha", "ghost"])

        with raises(EnvironmentNotFoundError, match="Suite 'broken' references an unknown environment"):
            resolve_suite_config_paths(
                "broken", builtin_dir=builtin, user_dir=tmp_path / "user", environments_dir=environments_dir
            )


class TestRealEnvironments:
    @mark.skipif(
        not (ENVIRONMENTS_DIR / "workplace_assistant" / "config.yaml").is_file(),
        reason="workplace_assistant environment is not present",
    )
    def test_user_suite_resolves_real_environment(self, tmp_path: Path) -> None:
        user = tmp_path / "user"
        _write_suite(user, "mine", ["workplace_assistant"])

        result = resolve_suite_config_paths("mine", builtin_dir=tmp_path / "none", user_dir=user)

        assert len(result) == 1
        assert result[0].endswith("environments/workplace_assistant/config.yaml")

    def test_suite_dataclass_is_hashable(self) -> None:
        # frozen dataclass -> usable as a dict key / in a set, like registry's EnvironmentEntry.
        suite = Suite(name="s", environments=("a",), description=None, path=Path("s.yaml"), is_builtin=True)
        assert {suite: 1}[suite] == 1
