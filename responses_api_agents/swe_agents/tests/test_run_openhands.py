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
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from responses_api_agents.swe_agents.run_openhands import (
    NS_TO_OPENAI_PARAM,
    NS_TO_OPENHANDS_PARAM,
    SUPPORTED_DATASETS,
    RunOpenHandsAgent,
    SupportedAgentFrameworks,
    SweBenchGenerationConfig,
    SweBenchInferenceConfig,
)


class TestSupportedAgentFrameworks:
    """Tests for the SupportedAgentFrameworks enum."""

    def test_enum_values(self) -> None:
        """Test that the enum has the expected values."""
        assert SupportedAgentFrameworks.swe_agent == "swe_agent"
        assert SupportedAgentFrameworks.openhands == "openhands"

    def test_enum_membership(self) -> None:
        """Test enum membership checks."""
        assert "swe_agent" in [e.value for e in SupportedAgentFrameworks]
        assert "openhands" in [e.value for e in SupportedAgentFrameworks]

    def test_enum_from_string(self) -> None:
        """Test creating enum from string value."""
        assert SupportedAgentFrameworks("swe_agent") == SupportedAgentFrameworks.swe_agent
        assert SupportedAgentFrameworks("openhands") == SupportedAgentFrameworks.openhands


class TestSupportedDatasets:
    """Tests for the SUPPORTED_DATASETS constant."""

    def test_supported_datasets(self) -> None:
        """Test that SUPPORTED_DATASETS contains expected values."""
        assert "SWE-Gym/SWE-Gym" in SUPPORTED_DATASETS
        assert "R2E-Gym/R2E-Gym-Subset" in SUPPORTED_DATASETS
        assert "princeton-nlp/SWE-bench_Verified" in SUPPORTED_DATASETS
        assert "nv-internal-1" in SUPPORTED_DATASETS
        assert len(SUPPORTED_DATASETS) == 4


class TestSweBenchInferenceConfig:
    """Tests for the SweBenchInferenceConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SweBenchInferenceConfig()
        assert config.temperature == 1.0
        assert config.top_k is None
        assert config.top_p == 1.0
        assert config.min_p is None
        assert config.random_seed is None
        assert config.tokens_to_generate is None
        assert config.repetition_penalty is None
        assert config.top_logprobs is None

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = SweBenchInferenceConfig(
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            min_p=0.1,
            random_seed=42,
            tokens_to_generate=1024,
            repetition_penalty=1.1,
            top_logprobs=5,
        )
        assert config.temperature == 0.7
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.min_p == 0.1
        assert config.random_seed == 42
        assert config.tokens_to_generate == 1024
        assert config.repetition_penalty == 1.1
        assert config.top_logprobs == 5


class TestSweBenchGenerationConfig:
    """Tests for the SweBenchGenerationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SweBenchGenerationConfig(
            output_file=Path("/tmp/output.jsonl"),
            agent_framework=SupportedAgentFrameworks.openhands,
        )
        assert config.output_file == Path("/tmp/output.jsonl")
        assert config.agent_framework == SupportedAgentFrameworks.openhands
        assert config.agent_framework_repo is None
        assert config.agent_framework_commit == "HEAD"
        assert config.agent_config is None
        assert config.agent_max_turns == 100
        assert config.swebench_tests_timeout == 60 * 30
        assert isinstance(config.inference, SweBenchInferenceConfig)
        assert config.server == {}

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        inference_config = SweBenchInferenceConfig(temperature=0.5)
        config = SweBenchGenerationConfig(
            output_file=Path("/tmp/custom_output.jsonl"),
            agent_framework=SupportedAgentFrameworks.swe_agent,
            agent_framework_repo="https://github.com/custom/repo.git",
            agent_framework_commit="v1.0.0",
            agent_config="custom/config",
            agent_max_turns=50,
            swebench_tests_timeout=1800,
            inference=inference_config,
            server={"model": "gpt-4", "base_url": "http://localhost:8000"},
        )
        assert config.output_file == Path("/tmp/custom_output.jsonl")
        assert config.agent_framework == SupportedAgentFrameworks.swe_agent
        assert config.agent_framework_repo == "https://github.com/custom/repo.git"
        assert config.agent_framework_commit == "v1.0.0"
        assert config.agent_config == "custom/config"
        assert config.agent_max_turns == 50
        assert config.swebench_tests_timeout == 1800
        assert config.inference.temperature == 0.5
        assert config.server == {"model": "gpt-4", "base_url": "http://localhost:8000"}


class TestParameterMappings:
    """Tests for parameter name mappings."""

    def test_ns_to_openai_param(self) -> None:
        """Test NS_TO_OPENAI_PARAM mapping."""
        assert NS_TO_OPENAI_PARAM["tokens_to_generate"] == "max_tokens"
        assert NS_TO_OPENAI_PARAM["top_logprobs"] == "top_logprobs"
        assert NS_TO_OPENAI_PARAM["random_seed"] == "seed"
        assert NS_TO_OPENAI_PARAM["top_k"] == "top_k"
        assert NS_TO_OPENAI_PARAM["min_p"] == "min_p"
        assert NS_TO_OPENAI_PARAM["repetition_penalty"] == "repetition_penalty"

    def test_ns_to_openhands_param(self) -> None:
        """Test NS_TO_OPENHANDS_PARAM mapping."""
        assert NS_TO_OPENHANDS_PARAM["tokens_to_generate"] == "max_output_tokens"
        assert NS_TO_OPENHANDS_PARAM["top_k"] == "top_k"
        assert NS_TO_OPENHANDS_PARAM["random_seed"] == "seed"
        # These params are not supported by OpenHands
        assert NS_TO_OPENHANDS_PARAM["min_p"] is None
        assert NS_TO_OPENHANDS_PARAM["repetition_penalty"] is None
        assert NS_TO_OPENHANDS_PARAM["top_logprobs"] is None


class TestFindContainer:
    """Tests for the _find_container method."""

    def _create_agent(self, output_dir: Path) -> RunOpenHandsAgent:
        """Helper to create a RunOpenHandsAgent instance."""
        cfg = SweBenchGenerationConfig(
            output_file=output_dir / "output.jsonl",
            agent_framework=SupportedAgentFrameworks.openhands,
        )
        return RunOpenHandsAgent(cfg=cfg)

    def test_find_container_exact_match(self) -> None:
        """Test finding container with exact instance_id match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            # Create a mock container file
            container_dir = output_dir / "containers"
            container_dir.mkdir()
            container_file = container_dir / "django__django-12345.sif"
            container_file.touch()

            data_point = {
                "instance_id": "django__django-12345",
                "dataset_name": "SWE-bench",
                "container_formatter": str(container_dir / "{instance_id}.sif"),
            }

            result = agent._find_container(data_point)
            assert result == str(container_file)

    def test_find_container_with_1776_replacement(self) -> None:
        """Test finding container with __->_1776_ replacement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            container_dir = output_dir / "containers"
            container_dir.mkdir()
            # Container uses _1776_ instead of __
            container_file = container_dir / "django_1776_django-12345.sif"
            container_file.touch()

            data_point = {
                "instance_id": "django__django-12345",
                "dataset_name": "SWE-bench",
                "container_formatter": str(container_dir / "{instance_id}.sif"),
            }

            result = agent._find_container(data_point)
            assert result == str(container_file)

    def test_find_container_with_s_replacement(self) -> None:
        """Test finding container with __->_s_ replacement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            container_dir = output_dir / "containers"
            container_dir.mkdir()
            # Container uses _s_ instead of __
            container_file = container_dir / "django_s_django-12345.sif"
            container_file.touch()

            data_point = {
                "instance_id": "django__django-12345",
                "dataset_name": "SWE-bench",
                "container_formatter": str(container_dir / "{instance_id}.sif"),
            }

            result = agent._find_container(data_point)
            assert result == str(container_file)

    def test_find_container_lowercase_match(self) -> None:
        """Test finding container with lowercase match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            container_dir = output_dir / "containers"
            container_dir.mkdir()
            # Container uses lowercase
            container_file = container_dir / "django_1776_django-12345.sif"
            container_file.touch()

            data_point = {
                "instance_id": "Django__Django-12345",  # Mixed case
                "dataset_name": "SWE-bench",
                "container_formatter": str(container_dir / "{instance_id}.sif"),
            }

            result = agent._find_container(data_point)
            assert result == str(container_file)

    def test_find_container_fuzzy_search(self) -> None:
        """Test finding container with fuzzy search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            container_dir = output_dir / "containers"
            container_dir.mkdir()
            # Container has extra characters
            container_file = container_dir / "prefix_django__django-12345_suffix.sif"
            container_file.touch()

            data_point = {
                "instance_id": "django__django-12345",
                "dataset_name": "SWE-bench",
                "container_formatter": str(container_dir / "{instance_id}.sif"),
            }

            result = agent._find_container(data_point)
            assert result == str(container_file)

    def test_find_container_r2e_gym_format(self) -> None:
        """Test finding container for R2E-Gym dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            container_dir = output_dir / "containers"
            container_dir.mkdir()
            # R2E-Gym format: django_final_bugfix.sif
            container_file = container_dir / "django_final_4.1-bugfix.sif"
            container_file.touch()

            data_point = {
                "instance_id": "r2e__django-4.1-bugfix",
                "dataset_name": "R2E-Gym/R2E-Gym-Subset",
                "container_formatter": str(container_dir / "{instance_id}.sif"),
            }

            result = agent._find_container(data_point)
            assert result == str(container_file)

    def test_find_container_list_of_formatters(self) -> None:
        """Test finding container with list of formatters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            # Create two container directories
            container_dir1 = output_dir / "containers1"
            container_dir1.mkdir()
            container_dir2 = output_dir / "containers2"
            container_dir2.mkdir()

            # Container exists in second directory
            container_file = container_dir2 / "django__django-12345.sif"
            container_file.touch()

            data_point = {
                "instance_id": "django__django-12345",
                "dataset_name": "SWE-bench",
                "container_formatter": [
                    str(container_dir1 / "{instance_id}.sif"),
                    str(container_dir2 / "{instance_id}.sif"),
                ],
            }

            result = agent._find_container(data_point)
            assert result == str(container_file)

    def test_find_container_not_found(self) -> None:
        """Test that FileNotFoundError is raised when container not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            container_dir = output_dir / "containers"
            container_dir.mkdir()

            data_point = {
                "instance_id": "nonexistent__instance-123",
                "dataset_name": "SWE-bench",
                "container_formatter": str(container_dir / "{instance_id}.sif"),
            }

            with pytest.raises(FileNotFoundError) as exc_info:
                agent._find_container(data_point)

            assert "No container file found" in str(exc_info.value)
            assert "nonexistent__instance-123" in str(exc_info.value)

    def test_find_container_nonexistent_directory(self) -> None:
        """Test behavior when container directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            # Directory doesn't exist
            nonexistent_dir = output_dir / "nonexistent_containers"

            data_point = {
                "instance_id": "django__django-12345",
                "dataset_name": "SWE-bench",
                "container_formatter": str(nonexistent_dir / "{instance_id}.sif"),
            }

            with pytest.raises(FileNotFoundError):
                agent._find_container(data_point)

    def test_find_container_string_formatter_converted_to_list(self) -> None:
        """Test that string formatter is converted to list internally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            container_dir = output_dir / "containers"
            container_dir.mkdir()
            container_file = container_dir / "django__django-12345.sif"
            container_file.touch()

            # Use string formatter
            data_point = {
                "instance_id": "django__django-12345",
                "dataset_name": "SWE-bench",
                "container_formatter": str(container_dir / "{instance_id}.sif"),  # String, not list
            }

            result = agent._find_container(data_point)
            assert result == str(container_file)


class TestWriteInstanceDataset:
    """Tests for the _write_instance_dataset method."""

    def _create_agent(self, output_dir: Path) -> RunOpenHandsAgent:
        """Helper to create a RunOpenHandsAgent instance."""
        cfg = SweBenchGenerationConfig(
            output_file=output_dir / "output.jsonl",
            agent_framework=SupportedAgentFrameworks.openhands,
        )
        agent = RunOpenHandsAgent(cfg=cfg)
        agent.output_dir = output_dir
        return agent

    def test_write_instance_dataset(self) -> None:
        """Test writing instance dataset file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            instance_dict = {
                "instance_id": "django__django-12345",
                "repo": "django/django",
                "base_commit": "abc123",
            }

            data_point = {
                "instance_id": "django__django-12345",
                "instance_dict": json.dumps(instance_dict),
            }

            result_path = agent._write_instance_dataset(data_point, "test_run_123")

            assert result_path.exists()
            with open(result_path, "r") as f:
                written_data = json.loads(f.read().strip())

            assert written_data["instance_id"] == "django__django-12345"
            assert written_data["repo"] == "django/django"
            # repo_name should be added from repo
            assert written_data["repo_name"] == "django/django"

    def test_write_instance_dataset_with_repo_name(self) -> None:
        """Test that repo_name is not overwritten if it already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            instance_dict = {
                "instance_id": "django__django-12345",
                "repo": "django/django",
                "repo_name": "custom/repo_name",  # Already has repo_name
            }

            data_point = {
                "instance_id": "django__django-12345",
                "instance_dict": json.dumps(instance_dict),
            }

            result_path = agent._write_instance_dataset(data_point, "test_run_456")

            with open(result_path, "r") as f:
                written_data = json.loads(f.read().strip())

            # repo_name should remain unchanged
            assert written_data["repo_name"] == "custom/repo_name"


class TestCleanupInstanceDataset:
    """Tests for the _cleanup_instance_dataset method."""

    def _create_agent(self, output_dir: Path) -> RunOpenHandsAgent:
        """Helper to create a RunOpenHandsAgent instance."""
        cfg = SweBenchGenerationConfig(
            output_file=output_dir / "output.jsonl",
            agent_framework=SupportedAgentFrameworks.openhands,
        )
        return RunOpenHandsAgent(cfg=cfg)

    def test_cleanup_instance_dataset(self) -> None:
        """Test cleaning up instance dataset file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            # Create a dataset file
            dataset_dir = output_dir / "instance_datasets"
            dataset_dir.mkdir()
            dataset_file = dataset_dir / "test_dataset.jsonl"
            dataset_file.write_text('{"test": "data"}')

            assert dataset_file.exists()
            agent._cleanup_instance_dataset(str(dataset_file))
            assert not dataset_file.exists()
            # Parent directory should be removed if empty
            assert not dataset_dir.exists()

    def test_cleanup_instance_dataset_none(self) -> None:
        """Test cleanup with None path does nothing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            # Should not raise any exception
            agent._cleanup_instance_dataset(None)

    def test_cleanup_instance_dataset_nonexistent(self) -> None:
        """Test cleanup with non-existent path does nothing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            # Should not raise any exception
            agent._cleanup_instance_dataset("/nonexistent/path/file.jsonl")

    def test_cleanup_instance_dataset_nonempty_parent(self) -> None:
        """Test cleanup doesn't remove parent directory if not empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            # Create a dataset file
            dataset_dir = output_dir / "instance_datasets"
            dataset_dir.mkdir()
            dataset_file = dataset_dir / "test_dataset.jsonl"
            dataset_file.write_text('{"test": "data"}')
            # Add another file to make directory non-empty
            other_file = dataset_dir / "other_file.txt"
            other_file.write_text("other content")

            agent._cleanup_instance_dataset(str(dataset_file))

            assert not dataset_file.exists()
            # Parent directory should still exist because it's not empty
            assert dataset_dir.exists()
            assert other_file.exists()


class TestCheckTestsPassed:
    """Tests for the check_tests_passed method."""

    def _create_agent(self, output_dir: Path) -> RunOpenHandsAgent:
        """Helper to create a RunOpenHandsAgent instance."""
        cfg = SweBenchGenerationConfig(
            output_file=output_dir / "output.jsonl",
            agent_framework=SupportedAgentFrameworks.openhands,
        )
        return RunOpenHandsAgent(cfg=cfg)

    def test_all_tests_passed(self) -> None:
        """Test when all required tests pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            test_results = {
                "tests": [
                    {"name": "test_1", "status": "PASSED"},
                    {"name": "test_2", "status": "PASSED"},
                    {"name": "test_3", "status": "PASSED"},
                ]
            }
            f2p = {"test_1", "test_2"}
            p2p = {"test_3"}

            result = agent.check_tests_passed(test_results, f2p, p2p)
            assert result is True

    def test_some_tests_failed(self) -> None:
        """Test when some required tests fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            test_results = {
                "tests": [
                    {"name": "test_1", "status": "PASSED"},
                    {"name": "test_2", "status": "FAILED"},  # Failed
                    {"name": "test_3", "status": "PASSED"},
                ]
            }
            f2p = {"test_1", "test_2"}
            p2p = {"test_3"}

            result = agent.check_tests_passed(test_results, f2p, p2p)
            assert result is False

    def test_empty_test_results(self) -> None:
        """Test with empty test results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            test_results = {}
            f2p = {"test_1"}
            p2p = {"test_2"}

            result = agent.check_tests_passed(test_results, f2p, p2p)
            assert result is False

    def test_none_test_results(self) -> None:
        """Test with None test results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            result = agent.check_tests_passed(None, {"test_1"}, {"test_2"})
            assert result is False

    def test_no_required_tests(self) -> None:
        """Test with no required tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            test_results = {
                "tests": [
                    {"name": "test_1", "status": "PASSED"},
                ]
            }
            f2p = set()
            p2p = set()

            result = agent.check_tests_passed(test_results, f2p, p2p)
            assert result is False  # Empty required tests means failure

    def test_no_passed_tests(self) -> None:
        """Test with no passed tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            test_results = {
                "tests": [
                    {"name": "test_1", "status": "FAILED"},
                    {"name": "test_2", "status": "ERROR"},
                ]
            }
            f2p = {"test_1"}
            p2p = {"test_2"}

            result = agent.check_tests_passed(test_results, f2p, p2p)
            assert result is False

    def test_extra_passed_tests(self) -> None:
        """Test when more tests pass than required."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            test_results = {
                "tests": [
                    {"name": "test_1", "status": "PASSED"},
                    {"name": "test_2", "status": "PASSED"},
                    {"name": "test_3", "status": "PASSED"},
                    {"name": "test_4", "status": "PASSED"},  # Extra
                ]
            }
            f2p = {"test_1"}
            p2p = {"test_2"}

            result = agent.check_tests_passed(test_results, f2p, p2p)
            assert result is True

    def test_empty_tests_list(self) -> None:
        """Test with empty tests list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            test_results = {"tests": []}
            f2p = {"test_1"}
            p2p = set()

            result = agent.check_tests_passed(test_results, f2p, p2p)
            assert result is False


class TestPrepareNvInternalEval:
    """Tests for the prepare_nv_internal_eval method."""

    def _create_agent(self, output_dir: Path) -> RunOpenHandsAgent:
        """Helper to create a RunOpenHandsAgent instance."""
        cfg = SweBenchGenerationConfig(
            output_file=output_dir / "output.jsonl",
            agent_framework=SupportedAgentFrameworks.openhands,
            swebench_tests_timeout=1800,
        )
        agent = RunOpenHandsAgent(cfg=cfg)
        agent.output_dir = output_dir
        return agent

    @pytest.mark.asyncio
    async def test_prepare_nv_internal_eval_basic(self) -> None:
        """Test basic nv-internal eval command generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            instance_dict = {
                "base_commit": "abc123",
                "base_dockerfile": "FROM python:3.12\nENV MY_VAR=value1",
                "instance_dockerfile": "ENV ANOTHER_VAR value2",
                "before_repo_set_cmd": "pip install -e .",
                "selected_test_files_to_run": '["test_file1.py", "test_file2.py"]',
                "run_script.sh": "#!/bin/bash\npytest $@",
                "parsing_script.py": "import json\nprint('done')",
            }

            data_point = {
                "instance_id": "test-instance-123",
                "instance_dict": json.dumps(instance_dict),
            }

            model_patch = "diff --git a/file.py\n+new line"

            cmd = await agent.prepare_nv_internal_eval(data_point, model_patch)

            # Verify command structure
            assert "git reset --hard abc123" in cmd
            assert "git checkout abc123" in cmd
            assert "git apply" in cmd
            assert "/root/patch.diff" in cmd
            assert "pip install -e ." in cmd
            assert "test_file1.py,test_file2.py" in cmd
            assert "/trajectories_mount/eval_results" in cmd

            # Verify files were written
            assert (output_dir / "run_script.sh").exists()
            assert (output_dir / "parsing_script.py").exists()
            assert (output_dir / "patch.diff").exists()

            # Verify patch has newline at end
            patch_content = (output_dir / "patch.diff").read_text()
            assert patch_content.endswith("\n")

    @pytest.mark.asyncio
    async def test_prepare_nv_internal_eval_env_parsing(self) -> None:
        """Test that ENV variables are correctly parsed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            instance_dict = {
                "base_dockerfile": "ENV KEY1=val1\nENV KEY2 val2\nENV KEY3=val3",
                "instance_dockerfile": "",
                "before_repo_set_cmd": "",
                "selected_test_files_to_run": "[]",
                "run_script.sh": "echo test",
                "parsing_script.py": "pass",
            }

            data_point = {
                "instance_id": "test-123",
                "instance_dict": json.dumps(instance_dict),
            }

            cmd = await agent.prepare_nv_internal_eval(data_point, "patch")

            # Check ENV parsing
            assert "export KEY1=val1" in cmd
            assert 'export KEY2="val2"' in cmd
            assert "export KEY3=val3" in cmd

    @pytest.mark.asyncio
    async def test_prepare_nv_internal_eval_test_files_as_list(self) -> None:
        """Test with test_files as list instead of string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            instance_dict = {
                "base_dockerfile": "",
                "instance_dockerfile": "",
                "before_repo_set_cmd": "",
                "selected_test_files_to_run": ["test1.py", "test2.py"],  # List, not string
                "run_script.sh": "echo test",
                "parsing_script.py": "pass",
            }

            data_point = {
                "instance_id": "test-123",
                "instance_dict": json.dumps(instance_dict),
            }

            cmd = await agent.prepare_nv_internal_eval(data_point, "patch")

            assert "test1.py,test2.py" in cmd


class TestRunEvaluations:
    """Tests for evaluation methods."""

    def _create_agent(
        self, output_dir: Path, framework: SupportedAgentFrameworks = SupportedAgentFrameworks.openhands
    ) -> RunOpenHandsAgent:
        """Helper to create a RunOpenHandsAgent instance."""
        cfg = SweBenchGenerationConfig(
            output_file=output_dir / "output.jsonl",
            agent_framework=framework,
            swebench_tests_timeout=1800,
            server={"model": "test-model"},
        )
        agent = RunOpenHandsAgent(
            cfg=cfg,
            dataset_path="/mock/dataset.jsonl",
            swebench_setup_dir=Path("/mock/swebench"),
            r2e_gym_setup_dir=Path("/mock/r2e_gym"),
        )
        agent.output_dir = output_dir
        return agent

    @pytest.mark.asyncio
    async def test_run_swebench_eval(self) -> None:
        """Test SWE-bench evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            # Create expected output
            eval_dir = output_dir / "run-123" / "test" / "test__instance-123"
            eval_dir.mkdir(parents=True)
            report_file = eval_dir / "report.json"
            report_file.write_text('{"test__instance-123": {"resolved": true}}')

            data_point = {
                "instance_id": "test__instance-123",
                "dataset_name": "SWE-bench",
                "split": "test",
            }

            with patch.object(agent, "_execute_container_command", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value = str(report_file)

                result = await agent._run_swebench_eval(
                    "/trajectories_mount/pred.jsonl",
                    data_point,
                    "run-123",
                    "/mock/instance_dataset.jsonl",
                )

                assert result == str(report_file)

    @pytest.mark.asyncio
    async def test_run_r2e_gym_eval(self) -> None:
        """Test R2E-Gym evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            # Create expected output
            eval_dir = output_dir / "eval-outputs" / "run-123"
            eval_dir.mkdir(parents=True)
            report_file = eval_dir / "report.json"
            report_file.write_text('{"test__instance-123": {"resolved": true}}')

            data_point = {
                "instance_id": "test__instance-123",
                "dataset_name": "R2E-Gym/R2E-Gym-Subset",
            }

            with patch.object(agent, "_execute_container_command", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value = str(report_file)

                result = await agent._run_r2e_gym_eval(
                    "/trajectories_mount/pred.jsonl",
                    data_point,
                    "run-123",
                    "/mock/instance_dataset.jsonl",
                )

                assert result == str(report_file)

    @pytest.mark.asyncio
    async def test_run_nv_internal_eval(self) -> None:
        """Test NV internal evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir)

            # Create expected output
            eval_dir = output_dir / "eval_results"
            eval_dir.mkdir(parents=True)
            report_file = eval_dir / "output.json"
            report_file.write_text('{"tests": [{"name": "test_1", "status": "PASSED"}]}')

            instance_dict = {
                "base_dockerfile": "",
                "instance_dockerfile": "",
                "before_repo_set_cmd": "",
                "selected_test_files_to_run": "[]",
                "run_script.sh": "echo test",
                "parsing_script.py": "pass",
                "fail_to_pass": '["test_1"]',
                "pass_to_pass": "[]",
            }

            data_point = {
                "instance_id": "test__instance-123",
                "dataset_name": "nv-internal-1",
                "instance_dict": json.dumps(instance_dict),
            }

            with patch.object(agent, "_execute_container_command", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value = str(report_file)

                result = await agent._run_nv_internal_eval(
                    data_point,
                    "diff --git a/test.py",
                    "/mock/instance_dataset.jsonl",
                )

                # Should return the report file path
                assert result == str(report_file)

                # Should have modified the report file
                with open(report_file, "r") as f:
                    report = json.load(f)
                assert "test__instance-123" in report


class TestProcessSingleDatapoint:
    """Tests for the process_single_datapoint method."""

    def _create_agent(
        self, output_dir: Path, framework: SupportedAgentFrameworks = SupportedAgentFrameworks.openhands
    ) -> RunOpenHandsAgent:
        """Helper to create a RunOpenHandsAgent instance."""
        cfg = SweBenchGenerationConfig(
            output_file=output_dir / "output.jsonl",
            agent_framework=framework,
            server={"model": "test-model", "base_url": "http://localhost:8000/v1"},
        )
        agent = RunOpenHandsAgent(
            cfg=cfg,
            dataset_path="/mock/dataset.jsonl",
            swebench_setup_dir=Path("/mock/swebench"),
            r2e_gym_setup_dir=Path("/mock/r2e_gym"),
            openhands_setup_dir=Path("/mock/openhands"),
        )
        return agent

    @pytest.mark.asyncio
    async def test_process_single_datapoint_openhands_resolved(self) -> None:
        """Test processing a single datapoint with OpenHands that resolves."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir, SupportedAgentFrameworks.openhands)

            data_point = {
                "instance_id": "test__instance-123",
                "base_commit": "abc123",
                "problem_statement": "Fix bug",
                "dataset_name": "SWE-bench",
                "split": "test",
                "instance_dict": '{"repo": "test/repo"}',
                "container_formatter": "/mock/{instance_id}.sif",
            }

            # Create real files for the test
            pred_file_content = (
                '{"model_patch": "diff --git", "model_name_or_path": "test", "instance_id": "test__instance-123"}'
            )
            report_content = {"resolved": True, "patch_exists": True, "patch_successfully_applied": True}

            mock_pred_file = tmpdir + "/pred.jsonl"
            with open(mock_pred_file, "w") as f:
                f.write(pred_file_content)

            mock_report_file = tmpdir + "/report.json"
            with open(mock_report_file, "w") as f:
                json.dump({"test__instance-123": report_content}, f)

            with (
                patch.object(agent, "_write_instance_dataset", return_value=Path("/mock/dataset.jsonl")),
                patch.object(agent, "_cleanup_instance_dataset"),
                patch.object(agent, "_run_openhands", new_callable=AsyncMock) as mock_openhands,
                patch.object(agent, "_run_swebench_eval", new_callable=AsyncMock) as mock_eval,
            ):
                mock_openhands.return_value = mock_pred_file
                mock_eval.return_value = mock_report_file

                result = await agent.process_single_datapoint(data_point)

                assert result["swe-bench-metrics"]["resolved"] is True
                assert "generation_time" in result

    @pytest.mark.asyncio
    async def test_process_single_datapoint_no_patch(self) -> None:
        """Test processing when no patch is generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir, SupportedAgentFrameworks.openhands)

            data_point = {
                "instance_id": "test__instance-123",
                "base_commit": "abc123",
                "problem_statement": "Fix bug",
                "dataset_name": "SWE-bench",
                "split": "test",
                "instance_dict": '{"repo": "test/repo"}',
                "container_formatter": "/mock/{instance_id}.sif",
            }

            # Mock with no patch
            pred_file_content = (
                '{"model_patch": null, "model_name_or_path": "test", "instance_id": "test__instance-123"}'
            )

            with (
                patch.object(agent, "_write_instance_dataset", return_value=Path("/mock/dataset.jsonl")),
                patch.object(agent, "_cleanup_instance_dataset"),
                patch.object(agent, "_run_openhands", new_callable=AsyncMock) as mock_openhands,
            ):
                mock_pred_file = tmpdir + "/pred.jsonl"
                with open(mock_pred_file, "w") as f:
                    f.write(pred_file_content)
                mock_openhands.return_value = mock_pred_file

                result = await agent.process_single_datapoint(data_point)

                # Should return not resolved with no patch
                assert result["swe-bench-metrics"]["resolved"] is False
                assert result["swe-bench-metrics"]["patch_exists"] is False

    @pytest.mark.asyncio
    async def test_process_single_datapoint_swe_agent(self) -> None:
        """Test processing with SWE-agent framework."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir, SupportedAgentFrameworks.swe_agent)

            data_point = {
                "instance_id": "test__instance-123",
                "base_commit": "abc123",
                "problem_statement": "Fix bug",
                "dataset_name": "SWE-bench",
                "split": "test",
                "instance_dict": '{"repo": "test/repo"}',
                "container_formatter": "/mock/{instance_id}.sif",
            }

            pred_file_content = (
                '{"model_patch": "diff", "model_name_or_path": "test", "instance_id": "test__instance-123"}'
            )
            report_content = {"resolved": True, "patch_exists": True, "patch_successfully_applied": True}

            with (
                patch.object(agent, "_write_instance_dataset", return_value=Path("/mock/dataset.jsonl")),
                patch.object(agent, "_cleanup_instance_dataset"),
                patch.object(agent, "_run_swe_agent", new_callable=AsyncMock) as mock_swe_agent,
                patch.object(agent, "_run_swebench_eval", new_callable=AsyncMock) as mock_eval,
            ):
                mock_pred_file = tmpdir + "/pred.jsonl"
                with open(mock_pred_file, "w") as f:
                    f.write(pred_file_content)
                mock_swe_agent.return_value = mock_pred_file

                mock_report_file = tmpdir + "/report.json"
                with open(mock_report_file, "w") as f:
                    json.dump({"test__instance-123": report_content}, f)
                mock_eval.return_value = mock_report_file

                result = await agent.process_single_datapoint(data_point)

                mock_swe_agent.assert_called_once()
                assert result["swe-bench-metrics"]["resolved"] is True

    @pytest.mark.asyncio
    async def test_process_single_datapoint_r2e_gym(self) -> None:
        """Test processing with R2E-Gym dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir, SupportedAgentFrameworks.openhands)

            data_point = {
                "instance_id": "test__instance-123",
                "base_commit": "abc123",
                "problem_statement": "Fix bug",
                "dataset_name": "R2E-Gym/R2E-Gym-Subset",
                "split": "test",
                "instance_dict": '{"repo": "test/repo"}',
                "container_formatter": "/mock/{instance_id}.sif",
            }

            pred_file_content = (
                '{"model_patch": "diff", "model_name_or_path": "test", "instance_id": "test__instance-123"}'
            )
            report_content = {"resolved": True, "patch_exists": True, "patch_successfully_applied": True}

            with (
                patch.object(agent, "_write_instance_dataset", return_value=Path("/mock/dataset.jsonl")),
                patch.object(agent, "_cleanup_instance_dataset"),
                patch.object(agent, "_run_openhands", new_callable=AsyncMock) as mock_openhands,
                patch.object(agent, "_run_r2e_gym_eval", new_callable=AsyncMock) as mock_eval,
            ):
                mock_pred_file = tmpdir + "/pred.jsonl"
                with open(mock_pred_file, "w") as f:
                    f.write(pred_file_content)
                mock_openhands.return_value = mock_pred_file

                mock_report_file = tmpdir + "/report.json"
                with open(mock_report_file, "w") as f:
                    json.dump({"test__instance-123": report_content}, f)
                mock_eval.return_value = mock_report_file

                result = await agent.process_single_datapoint(data_point)

                mock_eval.assert_called_once()
                assert result["swe-bench-metrics"]["resolved"] is True

    @pytest.mark.asyncio
    async def test_process_single_datapoint_eval_failure(self) -> None:
        """Test processing when evaluation fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            agent = self._create_agent(output_dir, SupportedAgentFrameworks.openhands)

            data_point = {
                "instance_id": "test__instance-123",
                "base_commit": "abc123",
                "problem_statement": "Fix bug",
                "dataset_name": "SWE-bench",
                "split": "test",
                "instance_dict": '{"repo": "test/repo"}',
                "container_formatter": "/mock/{instance_id}.sif",
            }

            pred_file_content = (
                '{"model_patch": "diff", "model_name_or_path": "test", "instance_id": "test__instance-123"}'
            )

            with (
                patch.object(agent, "_write_instance_dataset", return_value=Path("/mock/dataset.jsonl")),
                patch.object(agent, "_cleanup_instance_dataset"),
                patch.object(agent, "_run_openhands", new_callable=AsyncMock) as mock_openhands,
                patch.object(agent, "_run_swebench_eval", new_callable=AsyncMock) as mock_eval,
            ):
                mock_pred_file = tmpdir + "/pred.jsonl"
                with open(mock_pred_file, "w") as f:
                    f.write(pred_file_content)
                mock_openhands.return_value = mock_pred_file

                # Evaluation raises an error
                mock_eval.side_effect = ValueError("Evaluation failed")

                result = await agent.process_single_datapoint(data_point)

                # Should return failure metrics
                assert result["swe-bench-metrics"]["resolved"] is False
                assert result["swe-bench-metrics"]["patch_exists"] is True
                assert result["swe-bench-metrics"]["patch_successfully_applied"] is False

    @pytest.mark.asyncio
    async def test_process_single_datapoint_unsupported_framework(self) -> None:
        """Test processing with unsupported framework raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            cfg = SweBenchGenerationConfig(
                output_file=output_dir / "output.jsonl",
                agent_framework=SupportedAgentFrameworks.openhands,
                server={"model": "test-model", "base_url": "http://localhost:8000/v1"},
            )
            agent = RunOpenHandsAgent(cfg=cfg, dataset_path="/mock/dataset.jsonl")

            # Force an invalid framework
            agent.cfg.agent_framework = "invalid_framework"

            data_point = {
                "instance_id": "test__instance-123",
                "base_commit": "abc123",
                "problem_statement": "Fix bug",
                "dataset_name": "SWE-bench",
                "split": "test",
                "instance_dict": '{"repo": "test/repo"}',
                "container_formatter": "/mock/{instance_id}.sif",
            }

            with (
                patch.object(agent, "_write_instance_dataset", return_value=Path("/mock/dataset.jsonl")),
                patch.object(agent, "_cleanup_instance_dataset"),
            ):
                with pytest.raises(ValueError, match="Unsupported agent framework"):
                    await agent.process_single_datapoint(data_point)
