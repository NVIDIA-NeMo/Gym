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
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym import (
    NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME,
    PARENT_DIR,
    _augment_sys_path,
    component_search_roots,
)
from nemo_gym.discovery import (
    _UNSET_VALUE_PLACEHOLDER,
    _parse_no_environment_tolerating_unset_values,
    merge_by_name,
    read_config_metadata,
)


class TestComponentSearchRoots:
    def test_default_includes_cwd_and_install_root(self) -> None:
        resolved = {root.resolve() for root in component_search_roots()}
        assert Path.cwd().resolve() in resolved
        assert PARENT_DIR.resolve() in resolved

    def test_env_var_roots_added_before_builtins(self, tmp_path: Path, monkeypatch) -> None:
        # NEMO_GYM_EXTRA_ROOTS is the sole source of extra roots (--search-dir is folded into it). Its roots
        # are searched ahead of cwd/built-ins, in listed order, so they can shadow them.
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        monkeypatch.setenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, os.pathsep.join([str(a), str(b)]))

        roots = component_search_roots()

        assert roots[0] == a and roots[1] == b  # os.pathsep-separated, order preserved
        assert PARENT_DIR.resolve() in {root.resolve() for root in roots}  # built-ins still scanned

    def test_dedupes_roots_by_resolved_path(self, monkeypatch) -> None:
        # An extra root that resolves to the install root must not be scanned twice.
        monkeypatch.setenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, str(PARENT_DIR))
        roots = component_search_roots()
        resolved = [root.resolve() for root in roots]

        assert resolved.count(PARENT_DIR.resolve()) == 1
        assert roots[0].resolve() == PARENT_DIR.resolve()  # the extra root still takes precedence

    def test_empty_or_unset_env_var_adds_nothing(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, "")
        assert component_search_roots()[0].resolve() == Path.cwd().resolve()

    def test_sys_path_woven_after_extra_roots_and_before_cwd(self, tmp_path: Path, monkeypatch) -> None:
        # The import-precedence use (sys_path given): existing entries slot in AFTER the explicit extra roots
        # but BEFORE cwd/built-ins, so extra roots shadow site-packages while cwd does not.
        a = tmp_path / "a"
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        monkeypatch.setenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, str(a))

        roots = [str(r) for r in component_search_roots(sys_path=[Path("/some/site-packages")])]

        assert roots == [str(a), "/some/site-packages", str(Path.cwd()), str(PARENT_DIR)]

    def test_sys_path_builtins_entry_moved_last_even_if_already_present(self, tmp_path: Path, monkeypatch) -> None:
        # Wheel install: PARENT_DIR is site-packages, already on sys.path. It must be dropped from the woven
        # entries and appended last, so cwd/built-ins stay last and import order matches file lookup.
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        monkeypatch.delenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, raising=False)

        resolved = [r.resolve() for r in component_search_roots(sys_path=[Path("/usr/lib/python3.12"), PARENT_DIR])]

        assert resolved.count(PARENT_DIR.resolve()) == 1  # not duplicated
        assert resolved[-2:] == [Path.cwd().resolve(), PARENT_DIR.resolve()]  # cwd then built-ins, always last


class TestAugmentSysPath:
    def test_rebuilds_sys_path_from_component_search_roots(self, tmp_path: Path, monkeypatch) -> None:
        # Import order follows component_search_roots: extra roots (explicit --search-dir) first so they
        # shadow everything, then the existing sys.path entries, then cwd and PARENT_DIR (built-ins) last so a
        # plugin module wins over a same-named Gym module.
        a = tmp_path / "a"
        b = tmp_path / "b"
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        monkeypatch.setenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, os.pathsep.join([str(a), str(b)]))
        # Start with PARENT_DIR ahead of where it should land, to prove it gets moved to the end.
        monkeypatch.setattr(sys, "path", [str(PARENT_DIR), "/some/site-packages"])

        _augment_sys_path()

        cwd_on_path = str(Path.cwd())  # component_search_roots uses Path.cwd(), which may canonicalize symlinks
        assert sys.path == [str(a), str(b), "/some/site-packages", cwd_on_path, str(PARENT_DIR)]

    def test_idempotent(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, raising=False)
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        monkeypatch.setattr(sys, "path", ["/some/site-packages"])
        cwd_on_path = str(Path.cwd())

        _augment_sys_path()
        first = list(sys.path)
        _augment_sys_path()  # re-run (e.g. after --search-dir folds roots into the env) must not pile up entries

        # existing entries preserved, then cwd, then built-ins last.
        assert sys.path == first == ["/some/site-packages", cwd_on_path, str(PARENT_DIR)]


class TestMergeByName:
    def test_merges_disjoint_roots(self) -> None:
        assert merge_by_name([{"a": 1}, {"b": 2}]) == {"a": 1, "b": 2}

    def test_earlier_root_shadows_later_on_name_collision(self) -> None:
        # A component found in an earlier root (e.g. the user's cwd) wins over a same-named one in a
        # later root (e.g. a built-in) — the collision policy every `gym list` command relies on.
        merged = merge_by_name([{"dup": "from_first"}, {"dup": "from_second", "other": "kept"}])
        assert merged == {"dup": "from_first", "other": "kept"}

    def test_preserves_order_within_and_across_roots(self) -> None:
        assert list(merge_by_name([{"a": 1, "b": 2}, {"c": 3}])) == ["a", "b", "c"]

    def test_empty_input(self) -> None:
        assert merge_by_name([]) == {}


class TestTolerantInterpolationParse:
    # Unset `???` values and unresolved `${...}` interpolations reference runtime-only values that aren't
    # needed to identify a component; listing fills them with a placeholder so the config still resolves.
    def _resolve(self, d: dict):
        return _parse_no_environment_tolerating_unset_values(OmegaConf.create(d))

    def test_single_interpolation(self) -> None:
        resolved = self._resolve({"foo": "${bar}"})
        assert resolved["foo"] == _UNSET_VALUE_PLACEHOLDER

    def test_single_missing_value(self) -> None:
        resolved = self._resolve({"foo": "???"})
        assert resolved["foo"] == _UNSET_VALUE_PLACEHOLDER

    def test_mix(self) -> None:
        # A mix across nested dicts: resolvable literals (incl. nested) pass through untouched, while an
        # undefined `${...}` interpolation and unset `???` values (incl. nested) are filled with the
        # placeholder.
        resolved = self._resolve(
            {
                "name": "my_bench",
                "num_repeats": 3,
                "api_key": "${some_api_key}",
                "server": {
                    "endpoint": "https://example.com",
                    "nested": {
                        "enabled": True,
                        "token": "???",
                    },
                },
            }
        )
        # Correct key-value pairs are unmodified.
        assert resolved["name"] == "my_bench"
        assert resolved["num_repeats"] == 3
        assert resolved["server"]["endpoint"] == "https://example.com"
        assert resolved["server"]["nested"]["enabled"] is True
        # Undefined `${...}` and unset `???` values are filled.
        assert resolved["api_key"] == _UNSET_VALUE_PLACEHOLDER
        assert resolved["server"]["nested"]["token"] == _UNSET_VALUE_PLACEHOLDER

    def test_does_not_mutate_input(self) -> None:
        cfg = OmegaConf.create({"foo": "???", "bar": "${baz}"})
        before = OmegaConf.to_container(cfg, resolve=False, throw_on_missing=False)
        _parse_no_environment_tolerating_unset_values(cfg)
        after = OmegaConf.to_container(cfg, resolve=False, throw_on_missing=False)
        assert after == before == {"foo": "???", "bar": "${baz}"}


class TestReadConfigMetadata:
    def test_reads_domain_and_description_from_real_benchmark_config(self) -> None:
        # aime24 inherits both fields from its resources server via `config_paths`/`_inherit_from`, so
        # this only resolves via the tolerant fallback parse.
        from nemo_gym.benchmarks import BENCHMARKS_DIR

        domain, description = read_config_metadata(BENCHMARKS_DIR / "aime24" / "config.yaml")

        assert domain == "math"
        assert description

    def test_reads_domain_defined_on_agent(self, tmp_path: Path) -> None:
        # `domain` can be declared on the agent (responses_api_agents.<agent>.domain) rather than on a
        # resources server, as the tau2 config does.
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """tau2_agent:
  responses_api_agents:
    tau2:
      entrypoint: app.py
      domain: agent
"""
        )

        assert read_config_metadata(config_path)[0] == "agent"

    def test_reads_inline_metadata_without_resolving_external_server_refs(self, tmp_path: Path) -> None:
        # Environment configs reference model/agent servers defined elsewhere; resolving one in isolation
        # would raise. Inline domain/description must still be read from the raw config, no resolution.
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "env:\n"
            "  resources_servers:\n"
            "    env:\n"
            "      entrypoint: app.py\n"
            "      domain: rlhf\n"
            "      description: inline desc\n"
            "      judge_model_server:\n"
            "        type: responses_api_models\n"
            "        name: some_absent_model\n"
        )

        assert read_config_metadata(config_path) == ("rlhf", "inline desc")
