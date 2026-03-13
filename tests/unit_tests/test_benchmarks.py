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
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from nemo_gym.benchmarks import (
    BenchmarkConfig,
    _find_benchmark_dirs_from_config_paths,
    discover_benchmarks,
    get_benchmark,
    list_benchmarks,
    prepare_benchmark,
)


class TestBenchmarkConfig:
    def test_properties(self) -> None:
        config = BenchmarkConfig(
            name="test",
            path=Path("/fake"),
            config_dict={"agent_name": "my_agent", "num_repeats": 16},
        )
        assert config.agent_name == "my_agent"
        assert config.num_repeats == 16

    def test_properties_missing(self) -> None:
        config = BenchmarkConfig(name="test", path=Path("/fake"), config_dict={})
        assert config.agent_name is None
        assert config.num_repeats is None


class TestDiscoverBenchmarks:
    def test_discovers_aime24(self) -> None:
        benchmarks = discover_benchmarks()
        assert "aime24" in benchmarks
        bench = benchmarks["aime24"]
        assert bench.name == "aime24"
        assert bench.agent_name == "math_with_judge_simple_agent"
        assert bench.num_repeats == 32

    def test_empty_when_dir_missing(self, tmp_path: Path) -> None:
        with patch("nemo_gym.benchmarks.BENCHMARKS_DIR", tmp_path / "nonexistent"):
            assert discover_benchmarks() == {}

    def test_skips_dirs_without_config(self, tmp_path: Path) -> None:
        (tmp_path / "no_config_bench").mkdir()
        with patch("nemo_gym.benchmarks.BENCHMARKS_DIR", tmp_path):
            assert discover_benchmarks() == {}

    def test_discovers_from_custom_dir(self, tmp_path: Path) -> None:
        bench_dir = tmp_path / "my_bench"
        bench_dir.mkdir()
        config = {"agent_name": "test_agent", "num_repeats": 4}
        (bench_dir / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(config)))

        with patch("nemo_gym.benchmarks.BENCHMARKS_DIR", tmp_path):
            benchmarks = discover_benchmarks()
            assert "my_bench" in benchmarks
            assert benchmarks["my_bench"].agent_name == "test_agent"
            assert benchmarks["my_bench"].num_repeats == 4


class TestGetBenchmark:
    def test_found(self) -> None:
        bench = get_benchmark("aime24")
        assert bench.name == "aime24"

    def test_not_found(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            get_benchmark("nonexistent_benchmark")


def _mock_global_config(config: dict = None):
    """Return an OmegaConf config without CLI/file parsing."""
    return OmegaConf.create(config or {})


class TestListBenchmarks:
    def test_lists_found_benchmarks(self, capsys) -> None:
        with patch("nemo_gym.benchmarks.get_global_config_dict", return_value=_mock_global_config()):
            list_benchmarks()
        assert "aime24" in capsys.readouterr().out

    def test_no_benchmarks(self, capsys) -> None:
        with (
            patch("nemo_gym.benchmarks.get_global_config_dict", return_value=_mock_global_config()),
            patch("nemo_gym.benchmarks.discover_benchmarks", return_value={}),
        ):
            list_benchmarks()
        assert "No benchmarks found" in capsys.readouterr().out


class TestFindBenchmarkDirs:
    def test_finds_benchmark_from_config_paths(self, tmp_path: Path) -> None:
        benchmarks_dir = tmp_path / "benchmarks"
        bench_dir = benchmarks_dir / "my_bench"
        bench_dir.mkdir(parents=True)
        config_path = str(bench_dir / "config.yaml")

        with patch("nemo_gym.benchmarks.BENCHMARKS_DIR", benchmarks_dir):
            dirs = _find_benchmark_dirs_from_config_paths([config_path])
        assert dirs == [bench_dir]

    def test_skips_non_benchmark_paths(self) -> None:
        dirs = _find_benchmark_dirs_from_config_paths(["resources_servers/foo/configs/foo.yaml"])
        assert dirs == []


class TestPrepareBenchmark:
    def _make_bench_dir(self, tmp_path: Path, name: str = "fake_bench") -> Path:
        benchmarks_dir = tmp_path / "benchmarks"
        bench_dir = benchmarks_dir / name
        bench_dir.mkdir(parents=True)
        (bench_dir / "config.yaml").write_text("agent_name: test\n")
        return bench_dir

    def test_calls_prepare(self, tmp_path: Path) -> None:
        bench_dir = self._make_bench_dir(tmp_path)
        (bench_dir / "prepare.py").write_text("")
        config_path = str(bench_dir / "config.yaml")

        mock_module = MagicMock()
        mock_module.prepare.return_value = tmp_path / "output.jsonl"

        with (
            patch(
                "nemo_gym.benchmarks.get_global_config_dict",
                return_value=_mock_global_config({"config_paths": [config_path]}),
            ),
            patch("nemo_gym.benchmarks.BENCHMARKS_DIR", bench_dir.parent),
            patch("nemo_gym.benchmarks.importlib.import_module", return_value=mock_module),
        ):
            prepare_benchmark()
            mock_module.prepare.assert_called_once()

    def test_missing_prepare_py(self, tmp_path: Path) -> None:
        bench_dir = self._make_bench_dir(tmp_path)
        config_path = str(bench_dir / "config.yaml")

        with (
            patch(
                "nemo_gym.benchmarks.get_global_config_dict",
                return_value=_mock_global_config({"config_paths": [config_path]}),
            ),
            patch("nemo_gym.benchmarks.BENCHMARKS_DIR", bench_dir.parent),
        ):
            with pytest.raises(ValueError, match="No prepare.py found"):
                prepare_benchmark()

    def test_missing_prepare_function(self, tmp_path: Path) -> None:
        bench_dir = self._make_bench_dir(tmp_path)
        (bench_dir / "prepare.py").write_text("")
        config_path = str(bench_dir / "config.yaml")

        mock_module = MagicMock(spec=[])  # empty spec = no attributes

        with (
            patch(
                "nemo_gym.benchmarks.get_global_config_dict",
                return_value=_mock_global_config({"config_paths": [config_path]}),
            ),
            patch("nemo_gym.benchmarks.BENCHMARKS_DIR", bench_dir.parent),
            patch("nemo_gym.benchmarks.importlib.import_module", return_value=mock_module),
        ):
            with pytest.raises(ValueError, match="must define a `prepare\\(\\)` function"):
                prepare_benchmark()

    def test_no_benchmark_in_config_paths(self) -> None:
        with patch(
            "nemo_gym.benchmarks.get_global_config_dict",
            return_value=_mock_global_config({"config_paths": ["resources_servers/foo/configs/foo.yaml"]}),
        ):
            with pytest.raises(ValueError, match="No benchmark config found"):
                prepare_benchmark()
