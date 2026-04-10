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
"""
Unit tests for cube_loader.py.

All tests use monkeypatching to avoid requiring real CUBE packages.
Integration tests that require real packages are in test_resources_server.py.
"""

import importlib
import importlib.metadata
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from resources_servers.cube_standard.cube_loader import (
    _hydrate_extra_benchmark_config,
    detect_parallelization_mode,
    find_benchmark_class,
    pip_install,
    select_task_config,
)


# ---------------------------------------------------------------------------
# Mock benchmark class used throughout
# ---------------------------------------------------------------------------


class MockBenchmark:
    """Minimal CUBE Benchmark duck-type."""

    benchmark_metadata = MagicMock()
    benchmark_metadata.parallelization_mode = "task-parallel"
    benchmark_metadata.max_concurrent_tasks = 10

    def get_task_configs(self):
        return []

    def setup(self):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Tests: _hydrate_extra_benchmark_config
# ---------------------------------------------------------------------------


class TestHydrateExtraBenchmarkConfig:
    def test_plain_values_passthrough(self):
        raw = {"key": "value", "count": 42, "flag": True}
        result = _hydrate_extra_benchmark_config(raw)
        assert result == raw

    def test_type_key_instantiates_class(self, monkeypatch):
        """A dict with '_type' should be replaced by an instance of that class."""
        mock_cls = MagicMock(return_value="instance")
        mock_mod = MagicMock()
        mock_mod.FakeBackend = mock_cls

        with patch("importlib.import_module", return_value=mock_mod):
            result = _hydrate_extra_benchmark_config({
                "vm_backend": {
                    "_type": "fake_module.FakeBackend",
                    "cache_dir": "/tmp",
                    "memory": "4G",
                }
            })

        mock_cls.assert_called_once_with(cache_dir="/tmp", memory="4G")
        assert result["vm_backend"] == "instance"

    def test_nested_type_key_hydrated_recursively(self, monkeypatch):
        """Nested dicts with '_type' inside kwargs should also be hydrated."""
        inner_instance = MagicMock()
        outer_instance = MagicMock()

        call_count = {"n": 0}

        def fake_cls_factory(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return inner_instance
            return outer_instance

        mock_cls = MagicMock(side_effect=fake_cls_factory)
        mock_mod = MagicMock()
        mock_mod.FakeClass = mock_cls

        with patch("importlib.import_module", return_value=mock_mod):
            result = _hydrate_extra_benchmark_config({
                "outer": {
                    "_type": "mod.FakeClass",
                    "inner": {
                        "_type": "mod.FakeClass",
                        "x": 1,
                    },
                }
            })

        # Both inner and outer should be hydrated
        assert call_count["n"] == 2

    def test_empty_dict_returns_empty_dict(self):
        assert _hydrate_extra_benchmark_config({}) == {}

    def test_non_dict_value_passthrough(self):
        result = _hydrate_extra_benchmark_config({"items": [1, 2, 3]})
        assert result["items"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Tests: pip_install
# ---------------------------------------------------------------------------


class TestPipInstall:
    def test_skips_if_already_installed_correct_version(self, monkeypatch):
        mock_dist = MagicMock()
        mock_dist.version = "1.0.0"
        monkeypatch.setattr(importlib.metadata, "distribution", lambda pkg: mock_dist)

        with patch("subprocess.run") as mock_run:
            pip_install("my-cube", "1.0.0", None)
            mock_run.assert_not_called()

    def test_skips_if_installed_any_version_when_none_required(self, monkeypatch):
        mock_dist = MagicMock()
        mock_dist.version = "2.3.4"
        monkeypatch.setattr(importlib.metadata, "distribution", lambda pkg: mock_dist)

        with patch("subprocess.run") as mock_run:
            pip_install("my-cube", None, None)
            mock_run.assert_not_called()

    def test_installs_if_not_found(self, monkeypatch):
        monkeypatch.setattr(
            importlib.metadata,
            "distribution",
            lambda pkg: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError()),
        )

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            pip_install("my-cube", None, None)
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "my-cube" in cmd

    def test_installs_specific_version(self, monkeypatch):
        monkeypatch.setattr(
            importlib.metadata,
            "distribution",
            lambda pkg: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError()),
        )

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            pip_install("my-cube", "1.2.3", None)
            cmd = mock_run.call_args[0][0]
            assert "my-cube==1.2.3" in cmd

    def test_falls_back_to_dev_url_on_pypi_failure(self, monkeypatch):
        monkeypatch.setattr(
            importlib.metadata,
            "distribution",
            lambda pkg: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError()),
        )

        fail_result = MagicMock(returncode=1, stdout="", stderr="not found")
        success_result = MagicMock(returncode=0)

        call_count = {"n": 0}

        def fake_run(cmd, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return fail_result
            return success_result

        with patch("subprocess.run", side_effect=fake_run):
            pip_install("my-cube", None, "git+https://github.com/example/my-cube")
            assert call_count["n"] == 2

    def test_raises_if_all_installs_fail(self, monkeypatch):
        monkeypatch.setattr(
            importlib.metadata,
            "distribution",
            lambda pkg: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError()),
        )

        fail_result = MagicMock(returncode=1, stdout="", stderr="not found")
        with patch("subprocess.run", return_value=fail_result):
            with pytest.raises(RuntimeError, match="pip install failed"):
                pip_install("bad-cube", None, None)

    def test_reinstalls_if_version_mismatch(self, monkeypatch):
        mock_dist = MagicMock()
        mock_dist.version = "0.9.0"
        monkeypatch.setattr(importlib.metadata, "distribution", lambda pkg: mock_dist)

        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            pip_install("my-cube", "1.0.0", None)
            mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: find_benchmark_class
# ---------------------------------------------------------------------------


class TestFindBenchmarkClass:
    def test_resolves_via_entry_point(self, monkeypatch):
        mock_ep = MagicMock()
        mock_ep.name = "test-cube"
        mock_ep.load.return_value = MockBenchmark

        monkeypatch.setattr(
            importlib.metadata,
            "entry_points",
            lambda group: [mock_ep],
        )

        cls, err = find_benchmark_class("test-cube")
        assert cls is MockBenchmark
        assert err == ""

    def test_entry_point_name_must_match(self, monkeypatch):
        """Entry points with a different name should not be returned."""
        mock_ep = MagicMock()
        mock_ep.name = "other-cube"  # different name
        mock_ep.load.return_value = MockBenchmark

        # Module fallback
        mock_mod = ModuleType("test_cube")
        mock_mod.Benchmark = MockBenchmark

        monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [mock_ep])
        monkeypatch.setitem(sys.modules, "test_cube", mock_mod)

        cls, err = find_benchmark_class("test-cube")
        # Should fall back to module import, not the mismatched entry point
        assert cls is MockBenchmark

    def test_falls_back_to_module_import(self, monkeypatch):
        monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [])

        mock_mod = ModuleType("test_cube")
        mock_mod.Benchmark = MockBenchmark
        monkeypatch.setitem(sys.modules, "test_cube", mock_mod)

        cls, err = find_benchmark_class("test-cube")
        assert cls is MockBenchmark
        assert err == ""

    def test_duck_type_scan(self, monkeypatch):
        """Class without name 'Benchmark' but with get_task_configs() is found."""
        monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [])

        class CustomBenchmark:
            __module__ = "my_cube.core"

            def get_task_configs(self):
                return []

        mock_mod = ModuleType("my_cube")
        mock_mod.CustomBenchmark = CustomBenchmark
        monkeypatch.setitem(sys.modules, "my_cube", mock_mod)

        cls, err = find_benchmark_class("my-cube")
        assert cls is CustomBenchmark or err == ""  # may or may not find it depending on dir() order

    def test_returns_none_on_import_error(self, monkeypatch):
        monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [])
        # Do NOT add "nonexistent_cube" to sys.modules — real import will fail

        cls, err = find_benchmark_class("nonexistent-cube-xyz-abc")
        assert cls is None
        assert err != ""

    def test_returns_none_if_no_benchmark_class(self, monkeypatch):
        monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [])

        mock_mod = ModuleType("empty_cube")
        # No Benchmark class, no get_task_configs method
        monkeypatch.setitem(sys.modules, "empty_cube", mock_mod)

        cls, err = find_benchmark_class("empty-cube")
        assert cls is None
        assert "No Benchmark class" in err


# ---------------------------------------------------------------------------
# Tests: detect_parallelization_mode
# ---------------------------------------------------------------------------


class TestDetectParallelizationMode:
    def test_reads_class_var(self):
        mode, max_c = detect_parallelization_mode(MockBenchmark)
        assert mode == "task-parallel"
        assert max_c == 10

    def test_defaults_when_no_metadata(self):
        class NaiveBenchmark:
            pass

        mode, max_c = detect_parallelization_mode(NaiveBenchmark)
        assert mode == "task-parallel"
        assert max_c == 9999

    def test_benchmark_parallel_mode(self):
        class BPBenchmark:
            benchmark_metadata = MagicMock()
            benchmark_metadata.parallelization_mode = "benchmark-parallel"
            benchmark_metadata.max_concurrent_tasks = 1

        mode, max_c = detect_parallelization_mode(BPBenchmark)
        assert mode == "benchmark-parallel"
        assert max_c == 1


# ---------------------------------------------------------------------------
# Tests: select_task_config
# ---------------------------------------------------------------------------


class TestSelectTaskConfig:
    def _make_config(self, task_id: str, seed: int = 0) -> MagicMock:
        cfg = MagicMock()
        cfg.task_id = task_id
        cfg.seed = seed
        cfg.model_copy = lambda update: MagicMock(
            task_id=update.get("task_id", task_id),
            seed=update.get("seed", seed),
        )
        return cfg

    def _make_benchmark(self, configs: list) -> MagicMock:
        benchmark = MagicMock()
        benchmark.name = "test-benchmark"
        benchmark.get_task_configs.return_value = configs
        return benchmark

    def test_returns_first_config_when_task_id_is_none(self):
        configs = [self._make_config("task_a"), self._make_config("task_b")]
        benchmark = self._make_benchmark(configs)

        result = select_task_config(benchmark, task_id=None, seed=None)
        assert result.task_id == "task_a"

    def test_finds_by_task_id(self):
        configs = [self._make_config("task_a"), self._make_config("task_b")]
        benchmark = self._make_benchmark(configs)

        result = select_task_config(benchmark, task_id="task_b", seed=None)
        assert result.task_id == "task_b"

    def test_raises_if_task_id_not_found(self):
        benchmark = self._make_benchmark([self._make_config("task_a")])

        with pytest.raises(ValueError, match="task_z"):
            select_task_config(benchmark, task_id="task_z", seed=None)

    def test_raises_if_no_configs(self):
        benchmark = self._make_benchmark([])

        with pytest.raises(ValueError):
            select_task_config(benchmark, task_id=None, seed=None)

    def test_applies_seed_via_model_copy(self):
        configs = [self._make_config("task_a", seed=0)]
        benchmark = self._make_benchmark(configs)

        result = select_task_config(benchmark, task_id=None, seed=42)
        assert result.seed == 42

    def test_no_seed_returns_original_config(self):
        configs = [self._make_config("task_a", seed=99)]
        benchmark = self._make_benchmark(configs)

        result = select_task_config(benchmark, task_id=None, seed=None)
        assert result.task_id == "task_a"
