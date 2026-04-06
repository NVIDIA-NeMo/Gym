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
from yaml import safe_load

from nemo_gym.benchmarks import list_benchmarks, prepare_benchmark


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
            patch("nemo_gym.benchmarks._load_benchmarks_from_config_paths", return_value={}),
        ):
            list_benchmarks()
        assert "No benchmarks found" in capsys.readouterr().out


class TestPrepareBenchmark:
    def _make_bench_dir(self, tmp_path: Path, name: str = "fake_bench") -> tuple[Path, Path]:
        benchmarks_dir = tmp_path / "benchmarks"
        bench_dir = benchmarks_dir / name
        bench_dir.mkdir(parents=True)

        prepare_scripts_path = bench_dir / "prepare.py"
        prepare_scripts_path.write_text("")

        config_path = bench_dir / "config.yaml"
        config_path.write_text(f"""dummy_agent:
  responses_api_agents:
    simple_agent:
      datasets:
      - name: dummy_benchmark_name
        type: benchmark
        jsonl_fpath: {tmp_path / "output.jsonl"}
        prompt_config: benchmarks/dummy/prompts/default.yaml
        prepare_script: {prepare_scripts_path}
        num_repeats: 32""")

        return bench_dir, config_path

    def test_calls_prepare(self, tmp_path: Path) -> None:
        bench_dir, config_path = self._make_bench_dir(tmp_path)

        mock_module = MagicMock()
        mock_module.prepare.return_value = tmp_path / "output.jsonl"

        with (
            patch(
                "nemo_gym.benchmarks.get_global_config_dict",
                return_value=_mock_global_config(
                    {"config_paths": [str(config_path)], **safe_load(config_path.read_text())}
                ),
            ),
            patch("nemo_gym.benchmarks.BENCHMARKS_DIR", bench_dir.parent),
            patch("nemo_gym.benchmarks.importlib.import_module", return_value=mock_module),
        ):
            prepare_benchmark()
            mock_module.prepare.assert_called_once()

    def test_missing_prepare_py(self, tmp_path: Path) -> None:
        bench_dir, config_path = self._make_bench_dir(tmp_path)
        (bench_dir / "prepare.py").unlink()

        with (
            patch(
                "nemo_gym.benchmarks.get_global_config_dict",
                return_value=_mock_global_config(
                    {"config_paths": [str(config_path)], **safe_load(config_path.read_text())}
                ),
            ),
            patch("nemo_gym.benchmarks.BENCHMARKS_DIR", bench_dir.parent),
        ):
            with pytest.raises(RuntimeError, match="The following benchmarks are missing a valid prepare script"):
                prepare_benchmark()

    def test_missing_prepare_function(self, tmp_path: Path) -> None:
        bench_dir, config_path = self._make_bench_dir(tmp_path)

        mock_module = MagicMock()

        with (
            patch(
                "nemo_gym.benchmarks.get_global_config_dict",
                return_value=_mock_global_config(
                    {"config_paths": [str(config_path)], **safe_load(config_path.read_text())}
                ),
            ),
            patch("nemo_gym.benchmarks.BENCHMARKS_DIR", bench_dir.parent),
            patch("nemo_gym.benchmarks.importlib.import_module", return_value=mock_module),
        ):
            with pytest.raises(
                AssertionError,
                match="Expected the actual prepared dataset output fpath to match the jsonl_fpath set in the config",
            ):
                prepare_benchmark()

    def test_no_benchmark_in_config_paths(self) -> None:
        with (
            patch(
                "nemo_gym.benchmarks.get_global_config_dict",
                return_value=_mock_global_config({"config_paths": ["nemo_gym/resources_servers/foo/configs/foo.yaml"]}),
            ),
            patch("nemo_gym.benchmarks._load_benchmarks_from_config_paths", return_value={}),
        ):
            with pytest.raises(AssertionError, match="No benchmark config found in config_paths"):
                prepare_benchmark()

    def test_caching_sanity(self, tmp_path: Path) -> None:
        bench_dir, config_path = self._make_bench_dir(tmp_path)
        (tmp_path / "output.jsonl").write_text("blah blah text for file")

        mock_module = MagicMock()
        mock_module.prepare.return_value = tmp_path / "output.jsonl"

        with (
            patch(
                "nemo_gym.benchmarks.get_global_config_dict",
                return_value=_mock_global_config(
                    {
                        "use_cached_prepared_benchmarks": True,
                        "config_paths": [str(config_path)],
                        **safe_load(config_path.read_text()),
                    }
                ),
            ),
            patch("nemo_gym.benchmarks.BENCHMARKS_DIR", bench_dir.parent),
            patch("nemo_gym.benchmarks.importlib.import_module", return_value=mock_module),
        ):
            prepare_benchmark()

        assert mock_module.prepare.call_count == 0
