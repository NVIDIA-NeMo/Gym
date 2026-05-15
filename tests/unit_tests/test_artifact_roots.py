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
"""Tests for the Phase 1 artifact-root resolver and benchmark discovery across roots."""

from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf
from yaml import safe_load

import nemo_gym
import nemo_gym.benchmarks
from nemo_gym import (
    ArtifactCollisionError,
    _parse_bool_env_var,
    _parse_extra_roots,
    get_artifact_roots,
    resolve_artifact,
)
from nemo_gym.benchmarks import _discover_benchmark_config_paths, list_benchmarks


def _set_roots(
    monkeypatch: pytest.MonkeyPatch,
    *,
    root_dir: Path,
    extra_roots: list[Path],
    allow_override: bool = False,
) -> None:
    """Pin the resolver to the given (NEMO_GYM_ROOT_DIR, EXTRA_ROOTS, ALLOW_ROOT_OVERRIDE) for one test."""
    monkeypatch.setattr(nemo_gym, "NEMO_GYM_ROOT_DIR", root_dir)
    monkeypatch.setattr(nemo_gym, "EXTRA_ROOTS", list(extra_roots))
    monkeypatch.setattr(nemo_gym, "ALLOW_ROOT_OVERRIDE", allow_override)


class TestParseBoolEnvVar:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "YES", "on", "  on  "])
    def test_truthy(self, value: str) -> None:
        assert _parse_bool_env_var(value) is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "anything else"])
    def test_falsy(self, value: str) -> None:
        assert _parse_bool_env_var(value) is False


class TestParseExtraRoots:
    def test_empty_string(self) -> None:
        assert _parse_extra_roots("") == []

    def test_single_root(self) -> None:
        assert _parse_extra_roots("/a/b") == [Path("/a/b")]

    def test_multiple_roots(self) -> None:
        assert _parse_extra_roots("/a:/b/c:/d") == [Path("/a"), Path("/b/c"), Path("/d")]

    def test_skips_empty_entries(self) -> None:
        assert _parse_extra_roots("/a::/b:") == [Path("/a"), Path("/b")]

    def test_expands_user(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", "/home/test_user")
        assert _parse_extra_roots("~/plugins") == [Path("/home/test_user/plugins")]


class TestGetArtifactRoots:
    def test_default_order_is_cwd_then_root_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        root = tmp_path / "root"
        root.mkdir()

        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[])

        assert get_artifact_roots() == [cwd, root]

    def test_extra_roots_inserted_between_cwd_and_root_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        root = tmp_path / "root"
        root.mkdir()
        plugin_a = tmp_path / "plugin_a"
        plugin_b = tmp_path / "plugin_b"

        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[plugin_a, plugin_b])

        assert get_artifact_roots() == [cwd, plugin_a, plugin_b, root]


class TestResolveArtifact:
    def test_absolute_path_passthrough(self, tmp_path: Path) -> None:
        absolute = tmp_path / "data.jsonl"
        assert resolve_artifact(absolute) == absolute

    def test_string_input_accepted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        target = cwd / "rel.txt"
        target.write_text("hi")
        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=tmp_path / "root", extra_roots=[])
        assert resolve_artifact("rel.txt") == target

    def test_cwd_wins_over_extra_root_with_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cwd = tmp_path / "cwd"
        plugin = tmp_path / "plugin"
        root = tmp_path / "root"
        for d in (cwd, plugin, root):
            d.mkdir()
        rel = "resources_servers/foo"
        for parent in (cwd, plugin, root):
            (parent / rel).mkdir(parents=True)

        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[plugin], allow_override=True)

        assert resolve_artifact(rel) == cwd / rel

    def test_extra_root_wins_over_root_dir_with_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cwd = tmp_path / "cwd"  # cwd does NOT have the artifact
        plugin = tmp_path / "plugin"
        root = tmp_path / "root"
        for d in (cwd, plugin, root):
            d.mkdir()
        rel = "resources_servers/foo"
        (plugin / rel).mkdir(parents=True)
        (root / rel).mkdir(parents=True)

        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[plugin], allow_override=True)

        assert resolve_artifact(rel) == plugin / rel

    def test_extra_roots_searched_in_listed_order_with_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cwd = tmp_path / "cwd"
        plugin_a = tmp_path / "plugin_a"
        plugin_b = tmp_path / "plugin_b"
        root = tmp_path / "root"
        for d in (cwd, plugin_a, plugin_b, root):
            d.mkdir()
        rel = "resources_servers/foo"
        # both plugins have it; A comes first
        (plugin_a / rel).mkdir(parents=True)
        (plugin_b / rel).mkdir(parents=True)

        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[plugin_a, plugin_b], allow_override=True)

        assert resolve_artifact(rel) == plugin_a / rel

    def test_falls_back_to_root_dir_when_nothing_else_matches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cwd = tmp_path / "cwd"
        plugin = tmp_path / "plugin"
        root = tmp_path / "root"
        for d in (cwd, plugin, root):
            d.mkdir()
        rel = "configs/only_in_root.yaml"
        (root / "configs").mkdir()
        (root / rel).write_text("k: v\n")

        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[plugin])

        assert resolve_artifact(rel) == root / rel

    def test_returns_root_dir_path_when_nothing_exists(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Match pre-existing fallback behavior: caller produces "not found" against the canonical default."""
        cwd = tmp_path / "cwd"
        root = tmp_path / "root"
        for d in (cwd, root):
            d.mkdir()
        rel = "missing/file.yaml"
        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[])
        assert resolve_artifact(rel) == root / rel

    def test_raises_on_collision_by_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cwd = tmp_path / "cwd"
        plugin = tmp_path / "plugin"
        root = tmp_path / "root"
        for d in (cwd, plugin, root):
            d.mkdir()
        rel = "resources_servers/foo"
        for parent in (cwd, plugin, root):
            (parent / rel).mkdir(parents=True)

        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[plugin])

        with pytest.raises(ArtifactCollisionError) as excinfo:
            resolve_artifact(rel)
        msg = str(excinfo.value)
        assert "resources_servers/foo" in msg
        assert str(cwd / rel) in msg
        assert str(plugin / rel) in msg
        assert str(root / rel) in msg
        assert "NEMO_GYM_ALLOW_ROOT_OVERRIDE" in msg

    def test_no_collision_when_only_one_root_matches(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cwd = tmp_path / "cwd"
        plugin = tmp_path / "plugin"
        root = tmp_path / "root"
        for d in (cwd, plugin, root):
            d.mkdir()
        rel = "resources_servers/foo"
        (root / rel).mkdir(parents=True)

        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[plugin])

        assert resolve_artifact(rel) == root / rel

    def test_no_collision_when_roots_share_realpath(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Running ng_run from inside the Gym tree must not flag cwd vs root_dir as a collision."""
        root = tmp_path / "root"
        root.mkdir()
        rel = "resources_servers/foo"
        (root / rel).mkdir(parents=True)

        monkeypatch.chdir(root)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[])

        assert resolve_artifact(rel) == root / rel

    def test_collision_message_dedupes_aliased_roots(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A symlinked extra root pointing at the same dir as root_dir is not a collision."""
        root = tmp_path / "root"
        root.mkdir()
        symlink = tmp_path / "symlink_to_root"
        symlink.symlink_to(root)
        rel = "resources_servers/foo"
        (root / rel).mkdir(parents=True)

        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[symlink])

        # Both extra (symlink) and root resolve to the same realpath; no collision raised.
        result = resolve_artifact(rel)
        assert result.resolve() == (root / rel).resolve()

    def test_collision_ignored_when_validator_filters_one_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The validator runs before collision detection: invalid candidates don't count."""
        cwd = tmp_path / "cwd"
        plugin = tmp_path / "plugin"
        root = tmp_path / "root"
        for d in (cwd, plugin, root):
            d.mkdir()
        rel = "resources_servers/foo"
        # cwd has dir but no marker (filtered out); plugin and root both have the marker.
        for parent in (cwd, plugin, root):
            (parent / rel).mkdir(parents=True)
        for parent in (plugin, root):
            (parent / rel / "requirements.txt").write_text("nemo-gym\n")

        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[plugin])

        # plugin and root both validate; should still raise (two valid candidates).
        with pytest.raises(ArtifactCollisionError):
            resolve_artifact(rel, validator=lambda d: (d / "requirements.txt").exists())

    def test_validator_overrides_default_exists_check(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cwd = tmp_path / "cwd"
        root = tmp_path / "root"
        for d in (cwd, root):
            d.mkdir()
        # cwd has the dir but no marker; root has the dir AND a marker.
        (cwd / "resources_servers" / "foo").mkdir(parents=True)
        (root / "resources_servers" / "foo").mkdir(parents=True)
        (root / "resources_servers" / "foo" / "requirements.txt").write_text("nemo-gym\n")

        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[])

        resolved = resolve_artifact(
            "resources_servers/foo",
            validator=lambda d: (d / "requirements.txt").exists(),
        )
        assert resolved == root / "resources_servers" / "foo"


class TestAugmentSysPath:
    """``_augment_sys_path`` is what makes plugin benchmarks' ``prepare.py`` importable."""

    def test_appends_parent_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import sys

        from nemo_gym import _augment_sys_path

        monkeypatch.setattr(sys, "path", list(sys.path))
        parent = tmp_path / "parent"
        _augment_sys_path([], parent, parent)
        assert str(parent) in sys.path

    def test_extras_precede_parent_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Extras must come before parent_dir so an extras-root prepare.py wins under override."""
        import sys

        from nemo_gym import _augment_sys_path

        monkeypatch.setattr(sys, "path", [])
        parent = tmp_path / "parent"
        plugin_a = tmp_path / "plugin_a"
        plugin_b = tmp_path / "plugin_b"
        _augment_sys_path([plugin_a, plugin_b], parent, parent)
        assert sys.path == [str(plugin_a), str(plugin_b), str(parent)]

    def test_root_dir_between_extras_and_parent_when_distinct(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When NEMO_GYM_ROOT differs from PARENT_DIR: extras > root_dir > parent_dir."""
        import sys

        from nemo_gym import _augment_sys_path

        monkeypatch.setattr(sys, "path", [])
        parent = tmp_path / "parent"
        root = tmp_path / "root"
        plugin = tmp_path / "plugin"
        parent.mkdir()
        root.mkdir()
        _augment_sys_path([plugin], root, parent)
        assert sys.path == [str(plugin), str(root), str(parent)]

    def test_skips_root_dir_when_same_as_parent(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import sys

        from nemo_gym import _augment_sys_path

        monkeypatch.setattr(sys, "path", [])
        parent = tmp_path / "parent"
        parent.mkdir()
        _augment_sys_path([], parent, parent)
        # Only one entry — parent_dir, not duplicated.
        assert sys.path == [str(parent)]

    def test_does_not_duplicate_existing_entries(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import sys

        from nemo_gym import _augment_sys_path

        plugin = tmp_path / "plugin"
        monkeypatch.setattr(sys, "path", [str(plugin)])
        parent = tmp_path / "parent"
        _augment_sys_path([plugin], parent, parent)
        assert sys.path.count(str(plugin)) == 1


class TestEnvVarParsingOnImport:
    """The two env vars are read at module import time. Verify the parser logic directly."""

    def test_extra_roots_env_var_name(self) -> None:
        assert nemo_gym.NEMO_GYM_EXTRA_ROOTS_ENV_VAR == "NEMO_GYM_EXTRA_ROOTS"

    def test_root_env_var_name(self) -> None:
        assert nemo_gym.NEMO_GYM_ROOT_ENV_VAR == "NEMO_GYM_ROOT"

    def test_default_root_dir_is_parent_dir(self) -> None:
        # Without NEMO_GYM_ROOT set in this test process's environment, the import-time
        # default should be PARENT_DIR.
        assert nemo_gym.NEMO_GYM_ROOT_DIR == nemo_gym.PARENT_DIR


class TestBenchmarksDiscoveryAcrossRoots:
    """``_discover_benchmark_config_paths`` honors the artifact-root search order."""

    def _make_benchmark_yaml(self, dir_path: Path, name: str, marker: str) -> Path:
        bench_dir = dir_path / "benchmarks" / name
        bench_dir.mkdir(parents=True)
        config_path = bench_dir / "config.yaml"
        # Minimal YAML body — only the file needs to exist for discovery; loading is tested elsewhere.
        config_path.write_text(f"# {marker}\n")
        return config_path

    def test_discovers_only_root_dir_by_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cwd = tmp_path / "cwd"
        root = tmp_path / "root"
        for d in (cwd, root):
            d.mkdir()
        self._make_benchmark_yaml(root, "aime24", "from_root")

        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[])

        paths = _discover_benchmark_config_paths()
        assert paths == [root / "benchmarks" / "aime24" / "config.yaml"]

    def test_discovers_across_cwd_extra_roots_and_root_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cwd = tmp_path / "cwd"
        plugin = tmp_path / "plugin"
        root = tmp_path / "root"
        for d in (cwd, plugin, root):
            d.mkdir()
        cwd_yaml = self._make_benchmark_yaml(cwd, "from_cwd", "cwd")
        plugin_yaml = self._make_benchmark_yaml(plugin, "from_plugin", "plugin")
        root_yaml = self._make_benchmark_yaml(root, "from_root", "root")

        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[plugin])

        paths = set(_discover_benchmark_config_paths())
        assert paths == {cwd_yaml, plugin_yaml, root_yaml}

    def test_dedupes_by_name_with_first_root_winning_under_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cwd = tmp_path / "cwd"
        plugin = tmp_path / "plugin"
        root = tmp_path / "root"
        for d in (cwd, plugin, root):
            d.mkdir()
        cwd_yaml = self._make_benchmark_yaml(cwd, "shared_name", "cwd")
        self._make_benchmark_yaml(plugin, "shared_name", "plugin")
        self._make_benchmark_yaml(root, "shared_name", "root")
        # Plus one unique benchmark in root that should still surface.
        root_unique = self._make_benchmark_yaml(root, "root_only", "root")

        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[plugin], allow_override=True)

        paths = _discover_benchmark_config_paths()
        # Cwd wins for shared_name; root_only comes from root.
        assert set(paths) == {cwd_yaml, root_unique}

    def test_raises_on_duplicate_benchmark_names_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cwd = tmp_path / "cwd"
        plugin = tmp_path / "plugin"
        root = tmp_path / "root"
        for d in (cwd, plugin, root):
            d.mkdir()
        self._make_benchmark_yaml(cwd, "shared_name", "cwd")
        self._make_benchmark_yaml(plugin, "shared_name", "plugin")
        self._make_benchmark_yaml(root, "shared_name", "root")

        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[plugin])

        with pytest.raises(ArtifactCollisionError) as excinfo:
            _discover_benchmark_config_paths()
        msg = str(excinfo.value)
        assert "'shared_name'" in msg
        assert str(cwd / "benchmarks" / "shared_name" / "config.yaml") in msg
        assert str(plugin / "benchmarks" / "shared_name" / "config.yaml") in msg
        assert "NEMO_GYM_ALLOW_ROOT_OVERRIDE" in msg

    def test_skips_duplicate_root_paths(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """If cwd and NEMO_GYM_ROOT_DIR resolve to the same directory, glob runs once."""
        root = tmp_path / "root"
        root.mkdir()
        bench_yaml = self._make_benchmark_yaml(root, "only_one", "root")

        monkeypatch.chdir(root)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[])

        paths = _discover_benchmark_config_paths()
        assert paths == [bench_yaml]

    def test_returns_empty_when_no_roots_have_benchmarks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cwd = tmp_path / "cwd"
        root = tmp_path / "root"
        for d in (cwd, root):
            d.mkdir()
        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[])
        assert _discover_benchmark_config_paths() == []


class TestListBenchmarksAcrossRoots:
    """End-to-end check that the CLI command surfaces plugin benchmarks."""

    def _write_benchmark(self, dir_path: Path, name: str, jsonl_fpath: Path) -> Path:
        bench_dir = dir_path / "benchmarks" / name
        bench_dir.mkdir(parents=True)
        config_path = bench_dir / "config.yaml"
        config_path.write_text(
            f"""dummy_agent:
  responses_api_agents:
    simple_agent:
      datasets:
      - name: {name}
        type: benchmark
        jsonl_fpath: {jsonl_fpath}
        prepare_script: {bench_dir / "prepare.py"}
        prompt_config: null
        num_repeats: 1
"""
        )
        (bench_dir / "prepare.py").write_text("")
        return config_path

    def test_list_benchmarks_includes_plugin_benchmarks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        cwd = tmp_path / "cwd"
        plugin = tmp_path / "plugin"
        root = tmp_path / "root"
        for d in (cwd, plugin, root):
            d.mkdir()
        self._write_benchmark(plugin, "plugin_benchmark", tmp_path / "out.jsonl")

        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[plugin])

        with patch(
            "nemo_gym.benchmarks.get_global_config_dict",
            return_value=OmegaConf.create({}),
        ):
            list_benchmarks()
        assert "plugin_benchmark" in capsys.readouterr().out

    def test_list_benchmarks_reports_searched_roots_when_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        cwd = tmp_path / "cwd"
        root = tmp_path / "root"
        for d in (cwd, root):
            d.mkdir()
        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[])

        with patch(
            "nemo_gym.benchmarks.get_global_config_dict",
            return_value=OmegaConf.create({}),
        ):
            list_benchmarks()
        out = capsys.readouterr().out
        assert "No benchmarks found" in out
        assert str(root) in out


class TestResolveArtifactCallSites:
    """Spot-check that the four refactored call sites use the new resolver."""

    def test_load_extra_config_paths_uses_extra_roots(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from nemo_gym.global_config import GlobalConfigDictParser

        cwd = tmp_path / "cwd"
        plugin = tmp_path / "plugin"
        root = tmp_path / "root"
        for d in (cwd, plugin, root):
            d.mkdir()
        (plugin / "my_config.yaml").write_text("my_key: from_plugin\n")

        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[plugin])

        parser = GlobalConfigDictParser()
        _, extra_configs = parser.load_extra_config_paths(["my_config.yaml"])
        assert extra_configs[0]["my_key"] == "from_plugin"

    def test_env_yaml_resolved_via_extra_roots(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import sys

        import nemo_gym.global_config
        from nemo_gym.global_config import (
            NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
            GlobalConfigDictParser,
            GlobalConfigDictParserConfig,
        )

        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)
        monkeypatch.setattr(nemo_gym.global_config, "openai_version", "x")
        monkeypatch.setattr(nemo_gym.global_config, "ray_version", "x")
        monkeypatch.setattr(nemo_gym.global_config, "python_version", lambda: "x")
        monkeypatch.setattr(sys, "argv", [sys.argv[0]])

        cwd = tmp_path / "cwd"
        plugin = tmp_path / "plugin"
        root = tmp_path / "root"
        for d in (cwd, plugin, root):
            d.mkdir()
        (plugin / "env.yaml").write_text("custom_env_key: from_plugin\n")

        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[plugin])

        parser = GlobalConfigDictParser()
        global_config_dict = parser.parse(GlobalConfigDictParserConfig(skip_load_from_cli=True))
        assert global_config_dict["custom_env_key"] == "from_plugin"

    def test_run_helper_finds_server_in_extra_root(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        cwd = tmp_path / "cwd"
        plugin = tmp_path / "plugin"
        root = tmp_path / "root"
        for d in (cwd, plugin, root):
            d.mkdir()
        server_dir = plugin / "resources_servers" / "my_server"
        server_dir.mkdir(parents=True)
        (server_dir / "requirements.txt").write_text("nemo-gym\n")

        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[plugin])

        # Re-create the cli.py:resolve_artifact() call exactly to verify the contract.
        resolved = resolve_artifact(
            Path("resources_servers", "my_server"),
            validator=lambda d: (d / "requirements.txt").exists() or (d / "pyproject.toml").exists(),
        )
        assert resolved == server_dir


class TestPromptResolution:
    def test_prompt_resolves_via_extra_roots(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from nemo_gym.prompt import _resolve_path

        cwd = tmp_path / "cwd"
        plugin = tmp_path / "plugin"
        root = tmp_path / "root"
        for d in (cwd, plugin, root):
            d.mkdir()
        prompt_path = plugin / "benchmarks" / "foo" / "prompts" / "default.yaml"
        prompt_path.parent.mkdir(parents=True)
        prompt_path.write_text("user: hi\n")

        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[plugin])

        assert _resolve_path("benchmarks/foo/prompts/default.yaml") == prompt_path


class TestRolloutCollectionInputResolution:
    def test_rollout_input_resolves_via_extra_roots(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cwd = tmp_path / "cwd"
        plugin = tmp_path / "plugin"
        root = tmp_path / "root"
        for d in (cwd, plugin, root):
            d.mkdir()
        rel = "data/inputs.jsonl"
        (plugin / "data").mkdir()
        (plugin / rel).write_text('{"x": 1}\n')

        monkeypatch.chdir(cwd)
        _set_roots(monkeypatch, root_dir=root, extra_roots=[plugin])

        # The rollout call site does: resolve_artifact(config.input_jsonl_fpath)
        assert resolve_artifact(rel) == plugin / rel


class TestPrepareBenchmarkUsesResolvedDirs:
    """Regression: the existing prepare_benchmark flow must keep working when given absolute paths."""

    def test_calls_prepare_with_absolute_paths(self, tmp_path: Path) -> None:
        # Mirror tests/unit_tests/test_benchmarks.py::TestPrepareBenchmark::test_calls_prepare,
        # but assert that the absolute jsonl_fpath flows through unchanged via resolve_artifact.
        from unittest.mock import MagicMock, patch

        from nemo_gym.benchmarks import prepare_benchmark

        bench_dir = tmp_path / "benchmarks" / "fake_bench"
        bench_dir.mkdir(parents=True)
        prepare_path = bench_dir / "prepare.py"
        prepare_path.write_text("")
        config_path = bench_dir / "config.yaml"
        out_jsonl = tmp_path / "output.jsonl"
        config_path.write_text(
            f"""dummy_agent:
  responses_api_agents:
    simple_agent:
      datasets:
      - name: dummy
        type: benchmark
        jsonl_fpath: {out_jsonl}
        prepare_script: {prepare_path}
        prompt_config: null
        num_repeats: 1
"""
        )

        mock_module = MagicMock()
        mock_module.prepare.return_value = out_jsonl

        with (
            patch(
                "nemo_gym.benchmarks.get_global_config_dict",
                return_value=OmegaConf.create(
                    {"config_paths": [str(config_path)], **safe_load(config_path.read_text())}
                ),
            ),
            patch("nemo_gym.benchmarks.importlib.import_module", return_value=mock_module),
        ):
            prepare_benchmark()
            mock_module.prepare.assert_called_once()
