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

import textwrap
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

import nemo_gym.global_config
from benchmarks import BENCHMARKS_DIR, discover_benchmarks, get_benchmark_config_path
from nemo_gym import PARENT_DIR
from nemo_gym.global_config import (
    BENCHMARK_KEY_NAME,
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_RESERVED_TOP_LEVEL_KEYS,
    GlobalConfigDictParser,
)


class TestBenchmarkDiscovery:
    def test_discover_benchmarks_finds_all(self):
        benchmarks = discover_benchmarks()
        assert "livecodebench" in benchmarks
        assert "aime25" in benchmarks
        assert "gpqa" in benchmarks

    def test_discover_benchmarks_returns_sorted(self):
        benchmarks = discover_benchmarks()
        assert benchmarks == sorted(benchmarks)

    def test_get_benchmark_config_path_exists(self):
        path = get_benchmark_config_path("livecodebench")
        assert path.exists()
        assert path.name == "config.yaml"

    def test_get_benchmark_config_path_not_found(self):
        with pytest.raises(FileNotFoundError, match="not_a_benchmark"):
            get_benchmark_config_path("not_a_benchmark")

    def test_benchmarks_dir_is_correct(self):
        assert BENCHMARKS_DIR.name == "benchmarks"
        assert BENCHMARKS_DIR.is_dir()


class TestBenchmarkReservedKey:
    def test_benchmark_in_reserved_keys(self):
        assert BENCHMARK_KEY_NAME in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS

    def test_benchmark_key_name_value(self):
        assert BENCHMARK_KEY_NAME == "benchmark"


class TestBenchmarkConfigResolution:
    def test_benchmark_config_inserts_config_path(self):
        """Verify that +benchmark=X inserts benchmarks/X/config.yaml into config_paths."""
        # The benchmark config path should be resolved relative to PARENT_DIR
        for benchmark_name in discover_benchmarks():
            benchmark_config_path = f"benchmarks/{benchmark_name}/config.yaml"
            assert (PARENT_DIR / benchmark_config_path).exists()

    def test_benchmark_config_chains_to_resource_server(self):
        """Verify benchmark configs chain to their resource server configs via config_paths."""
        for benchmark_name in discover_benchmarks():
            config_path = get_benchmark_config_path(benchmark_name)
            config = OmegaConf.load(config_path)
            config_paths = config.get("config_paths", [])
            assert len(config_paths) > 0, f"Benchmark {benchmark_name} has no config_paths"
            for chained_path in config_paths:
                full_path = PARENT_DIR / chained_path
                assert full_path.exists(), f"Chained config {chained_path} not found for {benchmark_name}"

    def test_benchmark_configs_have_required_keys(self):
        """Verify all benchmark configs set the required rollout collection keys."""
        required_keys = ["agent_name", "input_jsonl_fpath", "num_repeats", "responses_create_params"]
        for benchmark_name in discover_benchmarks():
            config_path = get_benchmark_config_path(benchmark_name)
            config = OmegaConf.load(config_path)
            for key in required_keys:
                assert key in config, f"Benchmark {benchmark_name} missing required key: {key}"

    def test_benchmark_configs_reference_valid_agents(self):
        """Verify benchmark agent_name matches an agent defined in the chained config."""
        for benchmark_name in discover_benchmarks():
            config_path = get_benchmark_config_path(benchmark_name)
            config = OmegaConf.load(config_path)
            agent_name = config["agent_name"]

            # Load the chained resource server config to check agent exists
            for chained_path in config.get("config_paths", []):
                chained_config = OmegaConf.load(PARENT_DIR / chained_path)
                if agent_name in chained_config:
                    break
            else:
                pytest.fail(f"Benchmark {benchmark_name} references agent '{agent_name}' not found in chained configs")


class TestBenchmarkPrepareModules:
    def test_prepare_modules_are_importable(self):
        """Verify each benchmark has an importable prepare module with a prepare() function."""
        import importlib

        for benchmark_name in discover_benchmarks():
            module = importlib.import_module(f"benchmarks.{benchmark_name}.prepare")
            assert hasattr(module, "prepare"), f"Benchmark {benchmark_name} prepare module missing prepare()"
            assert callable(module.prepare)

    def test_prepare_modules_have_output_path(self):
        """Verify each benchmark prepare module defines an OUTPUT_PATH."""
        import importlib

        for benchmark_name in discover_benchmarks():
            module = importlib.import_module(f"benchmarks.{benchmark_name}.prepare")
            assert hasattr(module, "OUTPUT_PATH"), f"Benchmark {benchmark_name} prepare module missing OUTPUT_PATH"


class TestBenchmarkConfigIntegration:
    """Test that benchmark configs integrate correctly with the global config parser."""

    def test_parse_with_benchmark_resolves_config_paths(self, monkeypatch, tmp_path):
        """Verify that parse() with benchmark key inserts benchmark config path."""
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)

        # Mock versions
        monkeypatch.setattr(nemo_gym.global_config, "openai_version", "test")
        monkeypatch.setattr(nemo_gym.global_config, "ray_version", "test")
        monkeypatch.setattr(nemo_gym.global_config, "python_version", MagicMock(return_value="test"))

        parser = GlobalConfigDictParser()

        # Create a minimal benchmark config in tmp_path
        benchmark_dir = tmp_path / "benchmarks" / "test_bench"
        benchmark_dir.mkdir(parents=True)
        (benchmark_dir / "config.yaml").write_text(
            textwrap.dedent("""\
                agent_name: test_agent
                num_repeats: 5
            """)
        )

        # Test the load_extra_config_paths with the benchmark config
        config_paths = [str(benchmark_dir / "config.yaml")]
        result_paths, extra_configs = parser.load_extra_config_paths(config_paths)

        assert len(extra_configs) == 1
        assert extra_configs[0].get("agent_name") == "test_agent"
        assert extra_configs[0].get("num_repeats") == 5
