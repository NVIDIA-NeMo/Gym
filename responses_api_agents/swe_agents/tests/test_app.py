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
import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import responses_api_agents.swe_agents.app as swe_app
from nemo_gym.config_types import ModelServerRef, OmegaConf
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.swe_agents.app import (
    ActiveContainerCommand,
    AgentPromptOverride,
    BaseDatasetHarnessProcessor,
    ExecuteContainerCommandArgs,
    NVInternalDatasetProcessor,
    OpenCodeHarnessProcessor,
    OpenHandsHarnessProcessor,
    R2EGymDatasetProcessor,
    RunOpenHandsAgent,
    SweBenchDatasetProcessor,
    SWEBenchMetrics,
    SweBenchMultilingualDatasetProcessor,
    SWEBenchVerifyResponse,
    SWEBenchWrapper,
    SWEBenchWrapperConfig,
    SWEBenchWrapperInstanceConfig,
    SWEBenchWrapperServerConfig,
    SWERebenchDatasetProcessor,
    _extract_instance_dict,
    _render_opencode_user_message,
    _resolve_opencode_workspace_path,
    file_lock,
    runner_ray_remote,
    update_and_read_metrics,
)


SWE_AGENTS_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _cleanup_swebench_results():
    """Remove swebench_results_* dirs that model_post_init creates in the source tree."""
    yield
    for d in SWE_AGENTS_DIR.glob("swebench_results_*"):
        shutil.rmtree(d, ignore_errors=True)


########################################
# Helpers
########################################


def _minimal_server_config() -> SWEBenchWrapperConfig:
    return SWEBenchWrapperConfig(
        host="localhost",
        port=9003,
        name="test_swe_agent",
        entrypoint="responses_api_agents/swe_agents",
        container_formatter=["docker://custom/{instance_id}"],
        swebench_tests_timeout=900,
        model_server=ModelServerRef(type="responses_api_models", name="test_model"),
        concurrency=1,
    )


def _create_wrapper(monkeypatch) -> SWEBenchWrapper:
    """Create a SWEBenchWrapper with all setup calls mocked."""
    monkeypatch.setattr(swe_app, "get_global_config_dict", MagicMock(return_value=OmegaConf.create({})))
    monkeypatch.setattr(BaseDatasetHarnessProcessor, "_run_setup_command", MagicMock(return_value=None))

    config = _minimal_server_config()
    wrapper = SWEBenchWrapper(config=config, server_client=MagicMock(spec=ServerClient))
    return wrapper


def _make_instance_config(tmpdir: str, **overrides) -> SWEBenchWrapperInstanceConfig:
    """Build a minimal SWEBenchWrapperInstanceConfig for testing."""
    persistent_dir = Path(tmpdir) / "persistent"
    persistent_dir.mkdir(parents=True, exist_ok=True)
    base_mounted_dir = Path("/trajectories_mount")

    defaults = dict(
        host="localhost",
        port=9003,
        name="test_swe_agent",
        entrypoint="responses_api_agents/swe_agents",
        agent_framework="openhands",
        container_formatter=["docker://custom/{instance_id}"],
        swebench_tests_timeout=900,
        model_server=ModelServerRef(type="responses_api_models", name="test_model"),
        concurrency=1,
        ng_global_config_dict_str="'{}'",
        model_server_name="test_model",
        openhands_setup_dir=Path(tmpdir) / "openhands",
        swebench_setup_dir=Path(tmpdir) / "swebench",
        swebench_multilingual_setup_dir=Path(tmpdir) / "swebench_multilingual",
        r2e_gym_setup_dir=Path(tmpdir) / "r2e",
        swe_rebench_setup_dir=Path(tmpdir) / "rebench",
        run_session_id="test_session",
        base_results_dir=Path(tmpdir) / "results",
        metrics_fpath=persistent_dir / "metrics.json",
        problem_info={
            "problem_statement": "Fix bug",
            "instance_id": "django__django-12345",
            "base_commit": "abc123",
            "dataset_name": "SWE-bench",
            "split": "test",
            "instance_dict": "{}",
            "container_formatter": ["docker://custom/{instance_id}"],
        },
        body=NeMoGymResponseCreateParamsNonStreaming(
            model="test-model",
            input=[],
            metadata={
                "problem_statement": "Fix bug",
                "instance_id": "django__django-12345",
                "base_commit": "abc123",
                "dataset_name": "SWE-bench",
                "split": "test",
                "instance_dict": "{}",
            },
        ),
        persistent_dir=persistent_dir,
        ray_queue_timestamp=time.time(),
        inference_params={"temperature": 1.0, "top_p": 1.0},
        agent_run_id="test_run_123",
        instance_dataset_path=persistent_dir / "data.jsonl",
        agent_instance_dataset_path=persistent_dir / "data_agent.jsonl",
        trajectories_root=persistent_dir / "trajectories" / "django__django-12345",
        prediction_path=persistent_dir / "output.jsonl",
        output_for_eval_mounted_path=base_mounted_dir / "output_for_eval.jsonl",
        output_for_eval_path=persistent_dir / "output_for_eval.jsonl",
        model_patch_path=persistent_dir / "patch.diff",
        container="/path/to/container.sif",
        eval_dir_in_openhands="evaluation/oh/test_run_123",
        openhands_config_file_path="/tmp/config_test.toml",
        agent_script_path=persistent_dir / "agent_script.sh",
        final_eval_apptainer_spinup_timestamp_fpath=persistent_dir / "final_eval_ts",
        final_eval_apptainer_spinup_timestamp_mounted_fpath=base_mounted_dir / "final_eval_ts",
        generation_apptainer_spinup_timestamp_fpath=persistent_dir / "gen_ts",
        generation_apptainer_spinup_timestamp_mounted_fpath=base_mounted_dir / "gen_ts",
        base_mounted_dir=base_mounted_dir,
        profiling_dir=persistent_dir / "profiling",
        profiling_mounted_dir=base_mounted_dir / "profiling",
    )
    defaults.update(overrides)
    return SWEBenchWrapperInstanceConfig(**defaults)


########################################
# Config model tests
########################################


class TestAgentPromptOverride:
    def test_defaults(self) -> None:
        override = AgentPromptOverride()
        assert override.user_prompt_template is None
        assert override.system_prompt_template is None
        assert override.agent_cls == "CodeActAgent"
        assert override.diversify_tool_names is False
        assert override.camel_case_tool_names is False

    def test_custom_values(self) -> None:
        override = AgentPromptOverride(
            user_prompt_template="/path/user.j2",
            system_prompt_template="/path/system.j2",
            agent_cls="CodexAgent",
            diversify_tool_names=True,
            camel_case_tool_names=True,
        )
        assert override.agent_cls == "CodexAgent"
        assert override.diversify_tool_names is True
        assert override.camel_case_tool_names is True

    def test_all_agent_cls_values(self) -> None:
        for cls in ["CodeActAgent", "OpenCodeAgent", "CodexAgent", "Terminus2Agent"]:
            override = AgentPromptOverride(agent_cls=cls)
            assert override.agent_cls == cls


class TestSWEBenchWrapperConfig:
    def test_default_values(self) -> None:
        config = SWEBenchWrapperConfig(
            host="localhost",
            port=9003,
            name="test_agent",
            entrypoint="responses_api_agents/swe_agents",
            model_server=ModelServerRef(type="responses_api_models", name="test"),
        )
        assert config.agent_config is None
        assert config.agent_tools_file is None
        assert config.agent_max_turns == 100
        assert config.swebench_tests_timeout == 30 * 60
        assert config.swebench_agent_timeout == 45 * 60
        assert config.apptainer_memory_limit_mb == 64 * 1024
        assert config.command_exec_timeout == 5 * 60
        assert config.concurrency == 256
        assert config.dataset_path is None
        assert config.agent_prompt_overrides is None
        assert config.agent_prompt_override_random is False
        assert config.openhands_should_log is False
        assert config.debug is False
        assert config.agent_framework_repo is None
        assert config.agent_framework_commit == "HEAD"

    def test_custom_values(self) -> None:
        config = SWEBenchWrapperConfig(
            host="localhost",
            port=9003,
            name="test_agent",
            entrypoint="responses_api_agents/swe_agents",
            agent_config="custom/config",
            agent_tools_file="tools.json",
            agent_max_turns=50,
            container_formatter=["docker://custom/{instance_id}"],
            swebench_tests_timeout=900,
            model_server=ModelServerRef(type="responses_api_models", name="test_model"),
        )
        assert config.agent_config == "custom/config"
        assert config.agent_tools_file == "tools.json"
        assert config.agent_max_turns == 50
        assert config.container_formatter == ["docker://custom/{instance_id}"]
        assert config.swebench_tests_timeout == 900

    def test_multiple_container_formatters(self) -> None:
        config = SWEBenchWrapperConfig(
            host="localhost",
            port=9003,
            name="test_agent",
            entrypoint="responses_api_agents/swe_agents",
            container_formatter=[
                "docker://first/{instance_id}",
                "docker://second/{instance_id}",
            ],
            model_server=ModelServerRef(type="responses_api_models", name="test"),
        )
        assert len(config.container_formatter) == 2

    def test_string_container_formatter(self) -> None:
        config = SWEBenchWrapperConfig(
            host="localhost",
            port=9003,
            name="test_agent",
            entrypoint="responses_api_agents/swe_agents",
            container_formatter="docker://single/{instance_id}",
            model_server=ModelServerRef(type="responses_api_models", name="test"),
        )
        assert config.container_formatter == "docker://single/{instance_id}"

    def test_with_agent_prompt_overrides(self) -> None:
        config = SWEBenchWrapperConfig(
            host="localhost",
            port=9003,
            name="test_agent",
            entrypoint="responses_api_agents/swe_agents",
            model_server=ModelServerRef(type="responses_api_models", name="test"),
            agent_prompt_overrides=[
                AgentPromptOverride(agent_cls="CodeActAgent"),
                AgentPromptOverride(agent_cls="CodexAgent"),
            ],
        )
        assert len(config.agent_prompt_overrides) == 2


class TestSWEBenchWrapperServerConfig:
    def test_creation(self) -> None:
        config = SWEBenchWrapperServerConfig(
            ng_global_config_dict_str="'{}'",
            model_server_name="test_model",
            openhands_setup_dir=Path("/tmp/openhands"),
            swebench_setup_dir=Path("/tmp/swebench"),
            r2e_gym_setup_dir=Path("/tmp/r2e"),
            swe_rebench_setup_dir=Path("/tmp/rebench"),
            swebench_multilingual_setup_dir=Path("/tmp/swebench_ml"),
            run_session_id="test123",
            base_results_dir=Path("/tmp/results"),
        )
        assert config.model_server_name == "test_model"
        assert config.run_session_id == "test123"


class TestExecuteContainerCommandArgs:
    def test_creation(self) -> None:
        args = ExecuteContainerCommandArgs(
            command="echo hello",
            expected_file_pattern="/tmp/output.json",
            mode="agent",
            timeout=300,
        )
        assert args.command == "echo hello"
        assert args.mode == "agent"
        assert args.timeout == 300

    def test_eval_mode(self) -> None:
        args = ExecuteContainerCommandArgs(
            command="run_eval",
            expected_file_pattern="/tmp/report.json",
            mode="eval",
            timeout=600,
        )
        assert args.mode == "eval"


class TestSWEBenchWrapperInstanceConfig:
    def test_instance_id_property(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_instance_config(tmpdir)
            assert config.instance_id == "django__django-12345"

    def test_resolved_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_instance_config(tmpdir)
            assert config.resolved_user_prompt_template is None
            assert config.resolved_system_prompt_template is None
            assert config.resolved_agent_cls == "CodeActAgent"
            assert config.resolved_diversify_tool_names is False
            assert config.resolved_camel_case_tool_names is False


class TestSWEBenchMetrics:
    def test_defaults(self) -> None:
        metrics = SWEBenchMetrics()
        assert metrics.resolved is None
        assert metrics.patch_exists is None
        assert metrics.ray_queue_time is None
        assert metrics.openhands_run_time is None
        assert metrics.final_eval_time is None

    def test_with_values(self) -> None:
        metrics = SWEBenchMetrics(resolved=True, patch_exists=True, ray_queue_time=1.5)
        assert metrics.resolved is True
        assert metrics.ray_queue_time == 1.5


class TestSWEBenchVerifyResponse:
    def test_fields_exist(self) -> None:
        fields = SWEBenchVerifyResponse.model_fields
        assert "resolved" in fields
        assert "patch_exists" in fields
        assert "instance_config" in fields
        assert "subagent_trajectories" in fields


########################################
# update_metrics tests
########################################


class TestUpdateMetrics:
    def test_basic_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "metrics.json"
            fpath.write_text(json.dumps({"a": 1, "b": 2}))

            result = update_and_read_metrics(fpath, {"b": 3, "c": 4})
            assert result == {"a": 1, "b": 3, "c": 4}

    def test_none_values_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "metrics.json"
            fpath.write_text(json.dumps({"a": 1, "b": None}))

            result = update_and_read_metrics(fpath, {"c": None, "d": 5})

            assert result == {"a": 1, "d": 5}
            assert "b" not in result
            assert "c" not in result


########################################
# BaseDatasetHarnessProcessor tests
########################################


class TestBaseDatasetHarnessProcessor:
    def test_parent_dir(self) -> None:
        config = _minimal_server_config()
        processor = BaseDatasetHarnessProcessor(config=config)
        assert processor.parent_dir == Path(swe_app.__file__).parent

    def test_setup_returns_none(self) -> None:
        config = _minimal_server_config()
        processor = BaseDatasetHarnessProcessor(config=config)
        assert processor.setup() is None

    def test_get_run_command_returns_none(self) -> None:
        config = _minimal_server_config()
        processor = BaseDatasetHarnessProcessor(config=config)
        assert processor.get_run_command() is None

    def test_postprocess_after_run_returns_none(self) -> None:
        config = _minimal_server_config()
        processor = BaseDatasetHarnessProcessor(config=config)
        assert processor.postprocess_after_run(Path("/tmp/report.json")) is None

    def test_get_command_sleep_until_predictions_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_instance_config(tmpdir)
            processor = BaseDatasetHarnessProcessor(config=config)
            cmd = processor._get_command_sleep_until_predictions_file()
            assert "until" in cmd
            assert "sleep 5" in cmd
            assert str(config.output_for_eval_mounted_path) in cmd

    def test_run_setup_command_success(self) -> None:
        config = _minimal_server_config()
        processor = BaseDatasetHarnessProcessor(config=config)
        processor._run_setup_command("true")

    def test_run_setup_command_failure(self) -> None:
        config = _minimal_server_config()
        processor = BaseDatasetHarnessProcessor(config=config)
        with pytest.raises(AssertionError, match="Command failed"):
            processor._run_setup_command("false")

    def test_file_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_dir = Path(tmpdir) / "target"
            setup_dir.mkdir()
            lock_path = setup_dir.parent / f".{setup_dir.name}.lockdir"
            with file_lock(setup_dir, "test"):
                assert lock_path.exists()
            assert not lock_path.exists()

    def test_file_lock_stale_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_dir = Path(tmpdir) / "target"
            setup_dir.mkdir()
            lock_path = setup_dir.parent / f".{setup_dir.name}.lockdir"
            lock_path.mkdir()
            # Make it appear stale by backdating mtime
            import os

            old_time = time.time() - 7200  # 2 hours ago
            os.utime(lock_path, (old_time, old_time))

            with file_lock(setup_dir, "test"):
                pass  # should break the stale lock


########################################
# NVInternalDatasetProcessor tests
########################################


class TestNVInternalDatasetProcessor:
    def _make_processor(self, tmpdir, instance_dict_override=None) -> NVInternalDatasetProcessor:
        instance_dict = {
            "base_dockerfile": "ENV FOO=bar",
            "instance_dockerfile": "ENV BAZ=qux",
            "before_repo_set_cmd": "cd /app\npip install .",
            "selected_test_files_to_run": '["test_a.py", "test_b.py"]',
            "run_script.sh": "#!/bin/bash\npytest $1",
            "parsing_script.py": "import sys\nprint('done')",
            "base_commit": "abc123",
        }
        if instance_dict_override:
            instance_dict.update(instance_dict_override)

        config = _make_instance_config(
            tmpdir,
            problem_info={
                "problem_statement": "Fix bug",
                "instance_id": "nv__test-123",
                "base_commit": "abc123",
                "dataset_name": "nv-internal-1",
                "split": "test",
                "instance_dict": json.dumps(instance_dict),
                "container_formatter": ["docker://custom/{instance_id}"],
            },
        )
        return NVInternalDatasetProcessor(config=config)

    def test_get_run_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._make_processor(tmpdir)
            result = processor.get_run_command()
            assert isinstance(result, ExecuteContainerCommandArgs)
            assert result.mode == "eval"
            assert "git reset --hard abc123" in result.command
            assert "git apply" in result.command
            assert "run_script.sh" in result.command
            assert "parsing_script.py" in result.command

    def test_get_run_command_env_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._make_processor(
                tmpdir,
                {
                    "base_dockerfile": "ENV KEY=VALUE\nENV SPACE_KEY some_value",
                    "instance_dockerfile": "",
                },
            )
            result = processor.get_run_command()
            assert "export KEY=VALUE" in result.command
            assert 'export SPACE_KEY="some_value"' in result.command

    def test_get_run_command_list_test_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._make_processor(
                tmpdir,
                {
                    "selected_test_files_to_run": ["test_x.py", "test_y.py"],
                },
            )
            result = processor.get_run_command()
            assert "test_x.py,test_y.py" in result.command

    def test_get_run_command_no_repo_cmd(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._make_processor(
                tmpdir,
                {
                    "before_repo_set_cmd": "",
                },
            )
            result = processor.get_run_command()
            assert isinstance(result, ExecuteContainerCommandArgs)

    def test_check_tests_passed_all_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._make_processor(tmpdir)
            result = processor.check_tests_passed(
                {"tests": [{"name": "test_a", "status": "PASSED"}, {"name": "test_b", "status": "PASSED"}]},
                {"test_a"},
                {"test_b"},
            )
            assert result is True

    def test_check_tests_passed_some_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._make_processor(tmpdir)
            result = processor.check_tests_passed(
                {"tests": [{"name": "test_a", "status": "PASSED"}, {"name": "test_b", "status": "FAILED"}]},
                {"test_a"},
                {"test_b"},
            )
            assert result is False

    def test_check_tests_passed_empty_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._make_processor(tmpdir)
            assert processor.check_tests_passed({}, set(), set()) is False
            assert processor.check_tests_passed(None, set(), set()) is False

    def test_check_tests_passed_no_passed_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._make_processor(tmpdir)
            result = processor.check_tests_passed(
                {"tests": [{"name": "test_a", "status": "FAILED"}]},
                {"test_a"},
                set(),
            )
            assert result is False

    def test_check_tests_passed_empty_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._make_processor(tmpdir)
            result = processor.check_tests_passed(
                {"tests": [{"name": "test_a", "status": "PASSED"}]},
                set(),
                set(),
            )
            assert result is False

    def test_postprocess_after_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._make_processor(
                tmpdir,
                {
                    "fail_to_pass": '["test_a"]',
                    "pass_to_pass": '["test_b"]',
                },
            )
            report_file = Path(tmpdir) / "report.json"
            report_file.write_text(
                json.dumps(
                    {
                        "tests": [
                            {"name": "test_a", "status": "PASSED"},
                            {"name": "test_b", "status": "PASSED"},
                        ]
                    }
                )
            )
            processor.postprocess_after_run(report_file)
            result = json.loads(report_file.read_text())
            assert processor.config.instance_id in result
            assert result[processor.config.instance_id]["resolved"] is True

    def test_postprocess_after_run_list_f2p(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._make_processor(
                tmpdir,
                {
                    "fail_to_pass_select": ["test_a"],
                    "pass_to_pass_select": ["test_b"],
                },
            )
            report_file = Path(tmpdir) / "report.json"
            report_file.write_text(
                json.dumps(
                    {
                        "tests": [
                            {"name": "test_a", "status": "PASSED"},
                            {"name": "test_b", "status": "FAILED"},
                        ]
                    }
                )
            )
            processor.postprocess_after_run(report_file)
            result = json.loads(report_file.read_text())
            assert result[processor.config.instance_id]["resolved"] is False


########################################
# SWERebenchDatasetProcessor tests
########################################


class TestSWERebenchDatasetProcessor:
    def test_normalize_test_name_timing_bracket(self) -> None:
        assert SWERebenchDatasetProcessor._normalize_test_name("test_foo [1.5ms]") == "test_foo"

    def test_normalize_test_name_timing_in(self) -> None:
        assert SWERebenchDatasetProcessor._normalize_test_name("test_foo in 200 msec") == "test_foo"

    def test_normalize_test_name_timing_paren(self) -> None:
        assert SWERebenchDatasetProcessor._normalize_test_name("test_foo (1.5s)") == "test_foo"

    def test_normalize_test_name_no_change(self) -> None:
        assert SWERebenchDatasetProcessor._normalize_test_name("test_bar") == "test_bar"

    def test_normalize_test_name_multiple_patterns(self) -> None:
        # Only the matching pattern should be removed
        assert SWERebenchDatasetProcessor._normalize_test_name("test_foo [2s]") == "test_foo"
        assert SWERebenchDatasetProcessor._normalize_test_name("test_foo [200ms]") == "test_foo"

    def test_get_run_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            instance_dict = {
                "install_config": {
                    "test_cmd": ["pytest tests/"],
                    "install": ["pip install -e ."],
                    "log_parser": "pytest_parser",
                },
                "repo": "owner/repo_name",
                "test_patch": "diff --git a/test.py b/test.py\n",
                "FAIL_TO_PASS": '["test_a"]',
                "PASS_TO_PASS": '["test_b"]',
            }
            config = _make_instance_config(
                tmpdir,
                problem_info={
                    "problem_statement": "Fix",
                    "instance_id": "owner__repo-123",
                    "base_commit": "abc",
                    "dataset_name": "SWE-rebench",
                    "split": "test",
                    "instance_dict": json.dumps(instance_dict),
                    "container_formatter": ["/containers/{instance_id}.sif"],
                },
            )
            config.eval_private_dir.mkdir(parents=True, exist_ok=True)
            processor = SWERebenchDatasetProcessor(config=config)
            result = processor.get_run_command()
            assert isinstance(result, ExecuteContainerCommandArgs)
            assert "pytest tests/" in result.command
            assert "pip install -e ." in result.command
            assert "git apply" in result.command
            assert result.mode == "eval"

            # Check that eval metadata files were written
            eval_meta_dir = config.eval_private_dir / "eval_meta"
            assert (eval_meta_dir / "expected_passed.json").exists()
            assert (eval_meta_dir / "fail_to_pass.json").exists()
            assert (eval_meta_dir / "pass_to_pass.json").exists()

    def test_get_run_command_string_test_cmd(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            instance_dict = {
                "install_config": {"test_cmd": "pytest tests/", "install": "pip install ."},
                "repo": "owner/repo",
                "test_patch": "",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
            }
            config = _make_instance_config(
                tmpdir,
                problem_info={
                    "problem_statement": "Fix",
                    "instance_id": "owner__repo-1",
                    "base_commit": "abc",
                    "dataset_name": "SWE-rebench",
                    "split": "test",
                    "instance_dict": json.dumps(instance_dict),
                    "container_formatter": ["/containers/{instance_id}.sif"],
                },
            )
            config.eval_private_dir.mkdir(parents=True, exist_ok=True)
            processor = SWERebenchDatasetProcessor(config=config)
            result = processor.get_run_command()
            assert "pytest tests/" in result.command

    def test_postprocess_no_test_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            instance_dict = {"install_config": {"log_parser": "pytest_parser"}}
            config = _make_instance_config(
                tmpdir,
                problem_info={
                    "problem_statement": "Fix",
                    "instance_id": "owner__repo-1",
                    "base_commit": "abc",
                    "dataset_name": "SWE-rebench",
                    "split": "test",
                    "instance_dict": json.dumps(instance_dict),
                    "container_formatter": ["/containers/{instance_id}.sif"],
                },
            )
            processor = SWERebenchDatasetProcessor(config=config)
            report_file = Path(tmpdir) / "report.json"
            report_file.write_text("{}")
            # test_output.log does not exist
            processor.postprocess_after_run(report_file)
            result = json.loads(report_file.read_text())
            assert result["owner__repo-1"]["resolved"] is False
            assert "No test output" in result["owner__repo-1"]["error"]


########################################
# SweBenchDatasetProcessor tests
########################################


class TestSweBenchDatasetProcessor:
    def test_setup_already_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _minimal_server_config()

            with patch.object(
                BaseDatasetHarnessProcessor,
                "parent_dir",
                new_callable=lambda: property(lambda self: Path(tmpdir)),
            ):
                setup_dir = Path(tmpdir) / "swe_swebench_setup"
                setup_dir.mkdir()
                swebench_dir = setup_dir / "SWE-bench"
                swebench_dir.mkdir()
                (setup_dir / "uv").mkdir()
                (setup_dir / "python").mkdir()

                processor = SweBenchDatasetProcessor(config=config)
                result = processor.setup()
                assert result == setup_dir

    def test_get_run_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_instance_config(tmpdir)
            processor = SweBenchDatasetProcessor(config=config)
            result = processor.get_run_command()
            assert isinstance(result, ExecuteContainerCommandArgs)
            assert "run_local_evaluation" in result.command
            assert "django__django-12345" in result.command
            assert result.mode == "eval"
            assert result.timeout == config.swebench_tests_timeout + 120


########################################
# SweBenchMultilingualDatasetProcessor tests
########################################


class TestSweBenchMultilingualDatasetProcessor:
    def test_get_run_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_instance_config(
                tmpdir,
                swebench_multilingual_setup_dir=Path(tmpdir) / "swebench_ml",
            )
            processor = SweBenchMultilingualDatasetProcessor(config=config)
            result = processor.get_run_command()
            assert isinstance(result, ExecuteContainerCommandArgs)
            assert "SWE-bench_Multilingual" in result.command
            assert result.mode == "eval"


########################################
# R2EGymDatasetProcessor tests
########################################


class TestR2EGymDatasetProcessor:
    def test_get_run_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_instance_config(tmpdir)
            processor = R2EGymDatasetProcessor(config=config)
            result = processor.get_run_command()
            assert isinstance(result, ExecuteContainerCommandArgs)
            assert "run_local_evaluation.py" in result.command
            assert result.mode == "eval"


########################################
# OpenHandsHarnessProcessor tests
########################################


class TestOpenHandsHarnessProcessor:
    def test_get_run_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_instance_config(tmpdir)
            config.persistent_dir.mkdir(parents=True, exist_ok=True)
            processor = OpenHandsHarnessProcessor(config=config)
            result = processor.get_run_command()
            assert isinstance(result, ExecuteContainerCommandArgs)
            assert result.mode == "agent"
            assert "timeout" in result.command
            assert "run_infer.sh" in self._read_agent_script(config)

    def _read_agent_script(self, config) -> str:
        # The script is written at persistent_dir / agent_script_{agent_run_id}.sh
        script_path = config.persistent_dir / f"agent_script_{config.agent_run_id}.sh"
        return script_path.read_text()

    def test_get_run_command_with_debug(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_instance_config(tmpdir, debug=True)
            config.persistent_dir.mkdir(parents=True, exist_ok=True)
            processor = OpenHandsHarnessProcessor(config=config)
            processor.get_run_command()
            assert "NG_PROFILING_DIR" in self._read_agent_script(config)

    def test_get_run_command_with_logging(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_instance_config(tmpdir, openhands_should_log=True)
            config.persistent_dir.mkdir(parents=True, exist_ok=True)
            processor = OpenHandsHarnessProcessor(config=config)
            processor.get_run_command()
            assert "LOG_LEVEL=DEBUG" in self._read_agent_script(config)

    def test_get_run_command_nv_internal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_instance_config(
                tmpdir,
                problem_info={
                    "problem_statement": "Fix",
                    "instance_id": "nv__test-1",
                    "base_commit": "abc",
                    "dataset_name": "nv-internal-1",
                    "split": "test",
                    "instance_dict": "{}",
                    "container_formatter": ["/containers/{instance_id}.sif"],
                },
            )
            config.persistent_dir.mkdir(parents=True, exist_ok=True)
            processor = OpenHandsHarnessProcessor(config=config)
            processor.get_run_command()
            assert "cryptography" in self._read_agent_script(config)

    def test_get_run_command_swe_rebench(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_instance_config(
                tmpdir,
                problem_info={
                    "problem_statement": "Fix",
                    "instance_id": "owner__repo-1",
                    "base_commit": "abc",
                    "dataset_name": "SWE-rebench",
                    "split": "test",
                    "instance_dict": "{}",
                    "container_formatter": ["/containers/{instance_id}.sif"],
                },
            )
            config.persistent_dir.mkdir(parents=True, exist_ok=True)
            processor = OpenHandsHarnessProcessor(config=config)
            processor.get_run_command()
            script = self._read_agent_script(config)
            # Should skip workspace check for SWE-rebench
            assert "Exiting because /workspace" not in script

    def test_get_run_command_with_prompt_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_instance_config(
                tmpdir,
                resolved_user_prompt_template="/path/to/user_prompt.j2",
                resolved_system_prompt_template="/path/to/system_prompt.j2",
            )
            config.persistent_dir.mkdir(parents=True, exist_ok=True)
            processor = OpenHandsHarnessProcessor(config=config)
            processor.get_run_command()
            script = self._read_agent_script(config)
            assert "user_prompt.j2" in script
            assert "system_prompt.j2" in script

    def test_get_run_command_diversify_tool_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_instance_config(tmpdir, resolved_diversify_tool_names=True)
            config.persistent_dir.mkdir(parents=True, exist_ok=True)
            processor = OpenHandsHarnessProcessor(config=config)
            processor.get_run_command()
            assert "DIVERSIFY_TOOL_NAMES=true" in self._read_agent_script(config)

    def test_get_run_command_camel_case_tool_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_instance_config(tmpdir, resolved_camel_case_tool_names=True)
            config.persistent_dir.mkdir(parents=True, exist_ok=True)
            processor = OpenHandsHarnessProcessor(config=config)
            processor.get_run_command()
            assert "CAMEL_CASE_TOOL_NAMES=true" in self._read_agent_script(config)


########################################
# Workspace path + user-message resolver tests
########################################


class TestResolveOpencodeWorkspacePath:
    def _info(self, dataset_name, instance_dict=None):
        return {
            "dataset_name": dataset_name,
            "instance_dict": json.dumps(instance_dict or {}),
        }

    def test_default_swebench(self):
        assert (
            _resolve_opencode_workspace_path(
                self._info("princeton-nlp/SWE-bench_Verified", {"repo": "django/django", "version": "4.2"})
            )
            == "/testbed"
        )

    def test_swebench_live_uses_instance_id(self):
        assert (
            _resolve_opencode_workspace_path(self._info("SWE-bench-Live", {"instance_id": "django__django-12345"}))
            == "/testbed"
        )

    def test_nv_internal_1(self):
        assert _resolve_opencode_workspace_path(self._info("nv-internal-1")) == "/app"

    def test_swe_bench_ext(self):
        assert _resolve_opencode_workspace_path(self._info("swe-bench-ext")) == "/workspace/repo"

    def test_swe_rebench_v2_uses_repo_name(self):
        assert _resolve_opencode_workspace_path(self._info("SWE-rebench-V2", {"repo": "owner/myrepo"})) == "/myrepo"

    def test_swe_rebench_v2_no_slash_in_repo(self):
        assert _resolve_opencode_workspace_path(self._info("SWE-rebench-V2", {"repo": "myrepo"})) == "/myrepo"

    def test_r2e_gym(self):
        assert _resolve_opencode_workspace_path(self._info("R2E-Gym")) == "/testbed"

    def test_swe_gym(self):
        assert _resolve_opencode_workspace_path(self._info("SWE-Gym")) == "/testbed"

    def test_invalid_instance_dict_falls_back_safely(self):
        # Bad JSON shouldn't crash; the resolver just sees an empty instance dict.
        info = {"dataset_name": "princeton-nlp/SWE-bench_Verified", "instance_dict": "not json"}
        assert _resolve_opencode_workspace_path(info) == "/testbed"


class TestRenderOpencodeUserMessage:
    def test_default_includes_problem_and_workspace(self):
        msg = _render_opencode_user_message({"problem_statement": "MY PROBLEM"}, "/workspace/foo__1.0")
        assert "MY PROBLEM" in msg
        assert "/workspace/foo__1.0" in msg
        assert "<issue_description>" in msg

    def test_override_template_used(self, tmp_path):
        tpl = tmp_path / "user.txt"
        tpl.write_text("HEAD\nworkspace={workspace_path}\nissue={problem_statement}\nTAIL")
        msg = _render_opencode_user_message({"problem_statement": "P"}, "/testbed", template_override_path=str(tpl))
        assert msg == "HEAD\nworkspace=/testbed\nissue=P\nTAIL"

    def test_override_missing_falls_back(self, tmp_path):
        msg = _render_opencode_user_message(
            {"problem_statement": "P"}, "/testbed", template_override_path=str(tmp_path / "nope")
        )
        # Should not raise; should produce default output.
        assert "<issue_description>" in msg
        assert "/testbed" in msg

    def test_override_with_unknown_brace_falls_back(self, tmp_path):
        # Template author included an unknown slot; we tolerate it by falling back.
        tpl = tmp_path / "user.txt"
        tpl.write_text("{unknown_slot}")
        msg = _render_opencode_user_message({"problem_statement": "P"}, "/testbed", template_override_path=str(tpl))
        assert "<issue_description>" in msg


########################################
# OpenCodeHarnessProcessor tests
########################################


class TestOpenCodeHarnessProcessor:
    """Cover the new opencode harness path. Uses the same _make_instance_config
    helper as the openhands tests; only the agent_framework + opencode_setup_dir
    fields differ."""

    def _opencode_config(self, tmpdir, **overrides) -> SWEBenchWrapperInstanceConfig:
        opencode_setup_dir = Path(tmpdir) / "opencode_setup"
        opencode_setup_dir.mkdir(parents=True, exist_ok=True)
        return _make_instance_config(
            tmpdir,
            agent_framework="opencode",
            opencode_setup_dir=opencode_setup_dir,
            agent_framework_repo="https://example.invalid/opencode.git",
            agent_framework_commit="deadbeef",
            **overrides,
        )

    def _read_agent_script(self, config) -> str:
        return (config.persistent_dir / f"agent_script_{config.agent_run_id}.sh").read_text()

    @pytest.fixture
    def _stub_model_server_lookup(self, monkeypatch):
        """Stub get_first_server_config_dict so we don't need a live gym head server."""

        def _fake(_global, name):
            assert name == "test_model"
            return type("Cfg", (), {"host": "test-host", "port": 12345})()

        monkeypatch.setattr(swe_app, "get_first_server_config_dict", _fake)
        monkeypatch.setattr(swe_app, "get_global_config_dict", MagicMock(return_value=OmegaConf.create({})))
        return _fake

    def test_get_run_command_writes_script(self, _stub_model_server_lookup) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._opencode_config(tmpdir)
            config.persistent_dir.mkdir(parents=True, exist_ok=True)
            processor = OpenCodeHarnessProcessor(config=config)
            result = processor.get_run_command()

            assert isinstance(result, ExecuteContainerCommandArgs)
            assert result.mode == "agent"
            assert "timeout" in result.command  # wrapped with `timeout --signal=TERM`

            script = self._read_agent_script(config)
            # Bun runtime + opencode dir mount points must be on PATH.
            assert "/opencode_setup/bun/bin" in script
            assert "/opencode_setup/opencode" in script
            # Hits the in-tree run_infer.sh.
            assert "evaluation/benchmarks/swe_bench/scripts/run_infer.sh" in script
            # Resolved gym envs.
            assert "NEMO_GYM_MODEL_SERVER_NAME=test_model" in script
            assert "NEMO_GYM_MODEL_SERVER_BASE_URL=" in script
            assert "http://test-host:12345" in script
            # Per-instance args propagated.
            assert config.problem_info["instance_id"] in script
            assert "--max-turns" not in script  # max_turns is positional
            assert str(config.agent_max_turns) in script

    def test_get_run_command_subagents_disabled_by_default(self, _stub_model_server_lookup) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._opencode_config(tmpdir)
            config.persistent_dir.mkdir(parents=True, exist_ok=True)
            OpenCodeHarnessProcessor(config=config).get_run_command()
            script = self._read_agent_script(config)
            assert "ENABLE_SUBAGENTS=0" in script

    def test_get_run_command_subagents_enabled(self, _stub_model_server_lookup) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._opencode_config(tmpdir, opencode_subagents_enabled=True)
            config.persistent_dir.mkdir(parents=True, exist_ok=True)
            OpenCodeHarnessProcessor(config=config).get_run_command()
            script = self._read_agent_script(config)
            assert "ENABLE_SUBAGENTS=1" in script

    def test_get_run_command_does_not_export_removed_legacy_envs(self, _stub_model_server_lookup) -> None:
        """Opencode path no longer wires DIVERSIFY/CAMEL_CASE/NG_OPENCODE_LOG_LEVEL —
        bench/cli.ts and language-model.ts don't honor them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._opencode_config(
                tmpdir, resolved_diversify_tool_names=True, resolved_camel_case_tool_names=True
            )
            config.persistent_dir.mkdir(parents=True, exist_ok=True)
            OpenCodeHarnessProcessor(config=config).get_run_command()
            script = self._read_agent_script(config)
            assert "DIVERSIFY_TOOL_NAMES" not in script
            assert "CAMEL_CASE_TOOL_NAMES" not in script
            assert "NG_OPENCODE_LOG_LEVEL" not in script

    def test_get_run_command_propagates_prompt_overrides(self, _stub_model_server_lookup, tmp_path) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            user_template_path = tmp_path / "user_template.txt"
            user_template_path.write_text(
                "OVERRIDE-USER-TEMPLATE\nworkspace={workspace_path}\nproblem={problem_statement}\n"
            )
            config = self._opencode_config(
                tmpdir,
                resolved_user_prompt_template=str(user_template_path),
                resolved_system_prompt_template="/host/system.txt",
            )
            config.persistent_dir.mkdir(parents=True, exist_ok=True)
            processor = OpenCodeHarnessProcessor(config=config)
            processor.get_run_command()
            script = self._read_agent_script(config)
            # The pre-rendered user message file is always mounted; the system
            # override lands at a fixed path in the SIF when configured.
            assert "/opencode_setup/opencode/user_message.txt" in script
            assert "/opencode_setup/opencode/system_prompt.txt" in script

            # The user-message file should contain the rendered override (not the default).
            user_msg_file = config.persistent_dir / f"user_message_{config.agent_run_id}.txt"
            rendered = user_msg_file.read_text()
            assert "OVERRIDE-USER-TEMPLATE" in rendered
            # workspace_path comes from the dataset resolver (default SWE-bench → /testbed)
            assert "workspace=/testbed" in rendered
            assert "problem=Fix bug" in rendered

    def test_get_run_command_default_user_message_is_workspace_aware(self, _stub_model_server_lookup) -> None:
        """Default user message includes the resolved workspace path + problem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._opencode_config(tmpdir)
            config.persistent_dir.mkdir(parents=True, exist_ok=True)
            processor = OpenCodeHarnessProcessor(config=config)
            processor.get_run_command()
            user_msg_file = config.persistent_dir / f"user_message_{config.agent_run_id}.txt"
            rendered = user_msg_file.read_text()
            assert "<issue_description>" in rendered
            assert "Fix bug" in rendered
            # default SWE-bench dataset → /testbed
            assert "/testbed" in rendered

    def test_get_run_command_search_path_targets_opencode_dir(self, _stub_model_server_lookup) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._opencode_config(tmpdir)
            config.persistent_dir.mkdir(parents=True, exist_ok=True)
            processor = OpenCodeHarnessProcessor(config=config)
            result = processor.get_run_command()
            # gym globs `expected_file_pattern` post-run; must point under the
            # opencode setup dir, not the openhands one.
            assert str(config.opencode_setup_dir) in result.expected_file_pattern
            assert "output.jsonl" in result.expected_file_pattern

    def test_get_run_command_requires_opencode_setup_dir(self, _stub_model_server_lookup) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._opencode_config(tmpdir)
            config.opencode_setup_dir = None  # type: ignore[assignment]
            config.persistent_dir.mkdir(parents=True, exist_ok=True)
            with pytest.raises(AssertionError, match="opencode setup directory"):
                OpenCodeHarnessProcessor(config=config).get_run_command()


########################################
# _extract_instance_dict tests
########################################


class TestExtractInstanceDict:
    def test_returns_dict_passthrough(self):
        d = {"instance_id": "x", "base_commit": "abc"}
        assert _extract_instance_dict({"instance_dict": d}) == d

    def test_parses_json_string(self):
        d = {"instance_id": "x", "base_commit": "abc"}
        assert _extract_instance_dict({"instance_dict": json.dumps(d)}) == d

    def test_returns_empty_on_invalid_json(self):
        assert _extract_instance_dict({"instance_dict": "{not json"}) == {}

    def test_returns_empty_when_missing(self):
        assert _extract_instance_dict({}) == {}

    def test_returns_empty_for_non_string_non_dict(self):
        assert _extract_instance_dict({"instance_dict": 123}) == {}
        assert _extract_instance_dict({"instance_dict": None}) == {}
        assert _extract_instance_dict({"instance_dict": ["x"]}) == {}


########################################
# Per-session host-copy + trajectory extractor tests
########################################


def _write_completion(path: Path, *, session_id, parent_session_id, turn, content_text="hello"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "Fix bug"}],
                "response": {"choices": [{"message": {"role": "assistant", "content": content_text}}]},
                "provider_specific_fields": {"prompt_token_ids": [1, 2, 3]},
                "kwargs": {"tools": [{"name": "edit"}]},
                "session_id": session_id,
                "parent_session_id": parent_session_id,
                "turn": turn,
                "timestamp": 1.0,
            }
        )
    )


class TestOpencodeMultiSessionCopy:
    """`_openhands_dir_copy_from_host` must keep latest-per-session, not just one
    global latest, when the opencode bench writes session-tagged JSONs."""

    def _agent(self, tmpdir) -> RunOpenHandsAgent:
        opencode_setup_dir = Path(tmpdir) / "opencode_setup"
        opencode_setup_dir.mkdir(parents=True, exist_ok=True)
        cfg = _make_instance_config(
            tmpdir,
            agent_framework="opencode",
            opencode_setup_dir=opencode_setup_dir,
            agent_framework_repo="https://example.invalid/opencode.git",
            agent_framework_commit="deadbeef",
        )
        return RunOpenHandsAgent(config=cfg)

    def _eval_dir(self, agent: RunOpenHandsAgent) -> Path:
        eval_dir = Path(agent.config.opencode_setup_dir) / "opencode" / agent.config.eval_dir_in_openhands
        eval_dir.mkdir(parents=True, exist_ok=True)
        return eval_dir

    def test_copies_latest_per_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._agent(tmpdir)
            eval_dir = self._eval_dir(agent)
            traj_root = agent.config.trajectories_root
            traj_root.mkdir(parents=True, exist_ok=True)

            inst = agent.config.problem_info["instance_id"]
            comp_root = eval_dir / inst / "bench_run" / "llm_completions" / inst
            # main session: two turns (turn 0 older, turn 1 newer)
            _write_completion(comp_root / "m-0.json", session_id="ses_main", parent_session_id=None, turn=0)
            _write_completion(comp_root / "m-1.json", session_id="ses_main", parent_session_id=None, turn=1)
            # subagent A: one turn
            _write_completion(comp_root / "a-0.json", session_id="ses_a", parent_session_id="ses_main", turn=0)
            # subagent B: two turns
            _write_completion(comp_root / "b-0.json", session_id="ses_b", parent_session_id="ses_main", turn=0)
            _write_completion(comp_root / "b-1.json", session_id="ses_b", parent_session_id="ses_main", turn=1)

            # Bump mtimes: latest = m-1, a-0, b-1.
            now = time.time()
            for name, off in [
                ("m-0.json", 100),
                ("m-1.json", 10),
                ("a-0.json", 50),
                ("b-0.json", 80),
                ("b-1.json", 5),
            ]:
                p = comp_root / name
                os.utime(p, (now - off, now - off))

            # Create an output.jsonl so the copy path returns normally.
            (eval_dir / "output.jsonl").write_text("{}\n")
            agent.config.prediction_path.parent.mkdir(parents=True, exist_ok=True)
            agent._openhands_dir_copy_from_host(output_file_path=str(eval_dir / "output.jsonl"))

            copied = sorted((traj_root / "llm_completions" / inst).glob("*.json"))
            # 3 unique sessions → 3 files copied (latest from each).
            names = {p.name for p in copied}
            assert names == {"m-1.json", "a-0.json", "b-1.json"}

    def test_falls_back_to_single_latest_when_files_untagged(self) -> None:
        """Openhands-style files (no session_id) bucket under "main" → one copy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._agent(tmpdir)
            eval_dir = self._eval_dir(agent)
            traj_root = agent.config.trajectories_root
            traj_root.mkdir(parents=True, exist_ok=True)
            inst = agent.config.problem_info["instance_id"]

            comp_root = eval_dir / inst / "a" / "b" / "llm_completions" / inst
            comp_root.mkdir(parents=True)
            (comp_root / "old.json").write_text(
                json.dumps({"messages": [], "response": {"choices": [{"message": {}}]}})
            )
            (comp_root / "new.json").write_text(
                json.dumps({"messages": [], "response": {"choices": [{"message": {}}]}})
            )
            now = time.time()
            os.utime(comp_root / "old.json", (now - 100, now - 100))
            os.utime(comp_root / "new.json", (now - 1, now - 1))

            (eval_dir / "output.jsonl").write_text("{}\n")
            agent.config.prediction_path.parent.mkdir(parents=True, exist_ok=True)
            agent._openhands_dir_copy_from_host(output_file_path=str(eval_dir / "output.jsonl"))

            copied = sorted((traj_root / "llm_completions" / inst).glob("*.json"))
            assert [p.name for p in copied] == ["new.json"]


class TestGetOpenhandsTrajectoryFromCompletions:
    def _wrapper(self, tmpdir) -> SWEBenchWrapper:
        # Use the same fixture pattern as TestSWEBenchWrapper if present; we
        # only need the method, not a fully booted server, so use a stub
        # subclass that skips post-init.
        class _Stub(SWEBenchWrapper):
            def model_post_init(self, *_args, **_kwargs):
                return None

        return _Stub.model_construct()

    def test_prefers_main_session_when_tagged(self, tmp_path) -> None:
        inst = "demo-1"
        comp_dir = tmp_path / inst / "llm_completions" / inst
        _write_completion(
            comp_dir / "sub.json", session_id="ses_sub", parent_session_id="ses_main", turn=5, content_text="SUBAGENT"
        )
        _write_completion(
            comp_dir / "main.json", session_id="ses_main", parent_session_id=None, turn=2, content_text="MAIN"
        )

        w = self._wrapper(tmp_path)
        messages, tools, _ = w.get_openhands_trajectory_from_completions(tmp_path, inst)
        # The final assistant message of the main session ("MAIN") should ride
        # along; the subagent's content should NOT be in the returned messages.
        joined = json.dumps(messages)
        assert "MAIN" in joined
        assert "SUBAGENT" not in joined
        assert tools == [{"name": "edit"}]

    def test_falls_back_to_last_file_for_untagged(self, tmp_path) -> None:
        inst = "demo-2"
        comp_dir = tmp_path / inst / "llm_completions" / inst
        comp_dir.mkdir(parents=True)
        # Two openhands-style files, no session_id field.
        for name, content in [("00.json", "OLD"), ("01.json", "LAST")]:
            (comp_dir / name).write_text(
                json.dumps(
                    {
                        "messages": [{"role": "user", "content": "x"}],
                        "response": {"choices": [{"message": {"content": content}}]},
                        "kwargs": {"tools": []},
                    }
                )
            )
        w = self._wrapper(tmp_path)
        messages, _, _ = w.get_openhands_trajectory_from_completions(tmp_path, inst)
        assert any("LAST" in json.dumps(m) for m in messages)

    def test_returns_empty_when_dir_missing(self, tmp_path) -> None:
        w = self._wrapper(tmp_path)
        msgs, tools, _ = w.get_openhands_trajectory_from_completions(tmp_path, "no-such-instance")
        assert msgs == [] and tools == []


class TestGetAllSessionTrajectories:
    def _wrapper(self, _tmp) -> SWEBenchWrapper:
        class _Stub(SWEBenchWrapper):
            def model_post_init(self, *_args, **_kwargs):
                return None

        return _Stub.model_construct()

    def test_returns_one_entry_per_session(self, tmp_path) -> None:
        inst = "multi"
        comp_dir = tmp_path / inst / "llm_completions" / inst
        _write_completion(comp_dir / "m.json", session_id="ses_main", parent_session_id=None, turn=3)
        _write_completion(comp_dir / "a.json", session_id="ses_a", parent_session_id="ses_main", turn=1)
        _write_completion(comp_dir / "b.json", session_id="ses_b", parent_session_id="ses_main", turn=7)

        w = self._wrapper(tmp_path)
        out = w.get_all_session_trajectories_from_completions(tmp_path, inst)
        ids = {entry["session_id"] for entry in out}
        assert ids == {"ses_main", "ses_a", "ses_b"}
        # Each entry has messages + tools + parent linkage.
        for entry in out:
            assert "messages" in entry and "tools" in entry
            if entry["session_id"] == "ses_main":
                assert entry["parent_session_id"] is None
            else:
                assert entry["parent_session_id"] == "ses_main"

    def test_returns_empty_when_files_have_no_session_id(self, tmp_path) -> None:
        inst = "untagged"
        comp_dir = tmp_path / inst / "llm_completions" / inst
        comp_dir.mkdir(parents=True)
        (comp_dir / "00.json").write_text(
            json.dumps({"messages": [], "response": {"choices": [{"message": {}}]}, "kwargs": {"tools": []}})
        )
        w = self._wrapper(tmp_path)
        assert w.get_all_session_trajectories_from_completions(tmp_path, inst) == []

    def test_returns_empty_when_dir_missing(self, tmp_path) -> None:
        w = self._wrapper(tmp_path)
        assert w.get_all_session_trajectories_from_completions(tmp_path, "nope") == []


########################################
# runner_ray_remote tests
########################################


class TestRunnerRayRemote:
    def test_is_ray_remote(self) -> None:
        assert hasattr(runner_ray_remote, "remote")


########################################
# ActiveContainerCommand tests
########################################


class TestActiveContainerCommand:
    @pytest.mark.asyncio
    async def test_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            log_file = open(log_path, "w")
            try:
                process = await asyncio.create_subprocess_shell("true", stdout=log_file, stderr=log_file)
                cmd = ActiveContainerCommand(
                    process=process,
                    log_file=log_file,
                    log_file_path=log_path,
                )
                assert cmd.log_file_path == log_path
                await process.wait()
            finally:
                log_file.close()


########################################
# RunOpenHandsAgent tests
########################################


class TestRunOpenHandsAgent:
    def _make_agent(self, tmpdir, **overrides) -> RunOpenHandsAgent:
        config = _make_instance_config(tmpdir, **overrides)
        return RunOpenHandsAgent(config=config)

    def test_openhands_dir_copy_from_host_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(tmpdir)
            # Create required dirs
            eval_dir = Path(agent.config.openhands_setup_dir) / "OpenHands" / agent.config.eval_dir_in_openhands
            eval_dir.mkdir(parents=True, exist_ok=True)
            traj_root = agent.config.trajectories_root
            traj_root.mkdir(parents=True, exist_ok=True)

            result = agent._openhands_dir_copy_from_host(output_file_path=None)
            assert result is None

    def test_openhands_dir_copy_from_host_with_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(tmpdir)
            eval_dir = Path(agent.config.openhands_setup_dir) / "OpenHands" / agent.config.eval_dir_in_openhands
            eval_dir.mkdir(parents=True, exist_ok=True)
            traj_root = agent.config.trajectories_root
            traj_root.mkdir(parents=True, exist_ok=True)

            # Create an output file in the eval dir
            output_file = eval_dir / "output.jsonl"
            output_file.write_text('{"test": true}\n')

            agent.config.prediction_path.parent.mkdir(parents=True, exist_ok=True)
            result = agent._openhands_dir_copy_from_host(output_file_path=str(output_file))
            assert result == str(agent.config.prediction_path)
            assert agent.config.prediction_path.exists()

    def test_openhands_dir_copy_from_host_relative_output_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(tmpdir)
            eval_dir = Path(agent.config.openhands_setup_dir) / "OpenHands" / agent.config.eval_dir_in_openhands
            eval_dir.mkdir(parents=True, exist_ok=True)
            traj_root = agent.config.trajectories_root
            traj_root.mkdir(parents=True, exist_ok=True)

            # Create output.jsonl in a subdirectory matching the glob pattern
            sub_dir = eval_dir / "a" / "b" / "c"
            sub_dir.mkdir(parents=True)
            (sub_dir / "output.jsonl").write_text('{"data": 1}\n')

            agent.config.prediction_path.parent.mkdir(parents=True, exist_ok=True)
            result = agent._openhands_dir_copy_from_host(output_file_path="nonexistent.jsonl")
            assert result is not None

    def test_openhands_dir_copy_no_output_file_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(tmpdir)
            eval_dir = Path(agent.config.openhands_setup_dir) / "OpenHands" / agent.config.eval_dir_in_openhands
            eval_dir.mkdir(parents=True, exist_ok=True)
            traj_root = agent.config.trajectories_root
            traj_root.mkdir(parents=True, exist_ok=True)

            agent.config.prediction_path.parent.mkdir(parents=True, exist_ok=True)
            with pytest.raises(FileNotFoundError, match="No output.jsonl found"):
                agent._openhands_dir_copy_from_host(output_file_path="nonexistent.jsonl")

    @pytest.mark.asyncio
    async def test_start_container_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(tmpdir)
            agent.config.persistent_dir.mkdir(parents=True, exist_ok=True)

            cmd = ExecuteContainerCommandArgs(
                command="echo hello",
                expected_file_pattern="/tmp/*.json",
                mode="agent",
                timeout=10,
            )

            active = await agent._start_container_command(cmd, "echo done")
            await active.process.wait()
            active.log_file.close()
            assert active.log_file_path.exists()

    @pytest.mark.asyncio
    async def test_finish_container_command_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(tmpdir)
            agent.config.persistent_dir.mkdir(parents=True, exist_ok=True)

            expected_file = Path(tmpdir) / "output.json"
            expected_file.write_text("{}")

            cmd = ExecuteContainerCommandArgs(
                command="echo hello",
                expected_file_pattern=str(expected_file),
                mode="eval",
                timeout=10,
            )
            active = await agent._start_container_command(cmd, "echo done")
            result = await agent._finish_container_command(active, cmd)
            assert result == str(expected_file)

    @pytest.mark.asyncio
    async def test_finish_container_command_no_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(tmpdir)
            agent.config.persistent_dir.mkdir(parents=True, exist_ok=True)

            cmd = ExecuteContainerCommandArgs(
                command="echo hello",
                expected_file_pattern=str(Path(tmpdir) / "nonexistent*.json"),
                mode="eval",
                timeout=10,
            )
            active = await agent._start_container_command(cmd, "echo done")
            with pytest.raises(ValueError, match="Expected exactly one file"):
                await agent._finish_container_command(active, cmd)

    @pytest.mark.asyncio
    async def test_finish_container_command_multiple_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(tmpdir)
            agent.config.persistent_dir.mkdir(parents=True, exist_ok=True)

            (Path(tmpdir) / "output1.json").write_text("{}")
            import time as _time

            _time.sleep(0.05)
            (Path(tmpdir) / "output2.json").write_text("{}")

            cmd = ExecuteContainerCommandArgs(
                command="echo hello",
                expected_file_pattern=str(Path(tmpdir) / "output*.json"),
                mode="eval",
                timeout=10,
            )
            active = await agent._start_container_command(cmd, "echo done")
            result = await agent._finish_container_command(active, cmd)
            assert "output2.json" in result  # should pick latest

    @pytest.mark.asyncio
    async def test_finish_container_command_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(tmpdir)
            agent.config.persistent_dir.mkdir(parents=True, exist_ok=True)

            cmd = ExecuteContainerCommandArgs(
                command="sleep 100",
                expected_file_pattern=str(Path(tmpdir) / "*.json"),
                mode="agent",
                timeout=1,
            )
            active = await agent._start_container_command(cmd, "sleep 100")
            with pytest.raises(ValueError, match="timed out"):
                await agent._finish_container_command(active, cmd)

    @pytest.mark.asyncio
    async def test_finish_container_command_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(tmpdir)
            agent.config.persistent_dir.mkdir(parents=True, exist_ok=True)

            cmd = ExecuteContainerCommandArgs(
                command="exit 1",
                expected_file_pattern=str(Path(tmpdir) / "*.json"),
                mode="eval",
                timeout=10,
            )
            active = await agent._start_container_command(cmd, "bash -c 'exit 1'")
            with pytest.raises(RuntimeError, match="Command failed with return code"):
                await agent._finish_container_command(active, cmd)

    @pytest.mark.asyncio
    async def test_kill_active_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(tmpdir)
            agent.config.persistent_dir.mkdir(parents=True, exist_ok=True)

            cmd = ExecuteContainerCommandArgs(
                command="sleep 100",
                expected_file_pattern="/tmp/*.json",
                mode="agent",
                timeout=60,
            )
            active = await agent._start_container_command(cmd, "sleep 100")
            await agent._kill_active_command(active)
            assert active.process.returncode is not None

    @pytest.mark.asyncio
    async def test_kill_active_command_already_finished(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(tmpdir)
            agent.config.persistent_dir.mkdir(parents=True, exist_ok=True)

            cmd = ExecuteContainerCommandArgs(
                command="true",
                expected_file_pattern="/tmp/*.json",
                mode="agent",
                timeout=10,
            )
            active = await agent._start_container_command(cmd, "true")
            await active.process.wait()
            # Should not raise even if already finished
            await agent._kill_active_command(active)


########################################
# SWEBenchWrapper tests
########################################


class TestSWEBenchWrapper:
    def test_model_post_init(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        assert wrapper._sem is not None
        assert wrapper._vllm_converter is not None
        assert wrapper._swe_bench_wrapper_server_config is not None
        assert wrapper._swe_bench_wrapper_server_config.run_session_id is not None

    def test_resolve_absolute_path_none(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        assert wrapper._resolve_absolute_path(None) is None
        assert wrapper._resolve_absolute_path("") is None

    def test_resolve_absolute_path_absolute(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        result = wrapper._resolve_absolute_path("/absolute/path/file.txt")
        assert result == "/absolute/path/file.txt"

    def test_resolve_absolute_path_relative(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        result = wrapper._resolve_absolute_path("relative/path/file.txt")
        assert result.endswith("relative/path/file.txt")
        assert Path(result).is_absolute()


class TestSWEBenchWrapperFindContainer:
    def _create_wrapper_for_find(self, monkeypatch) -> SWEBenchWrapper:
        return _create_wrapper(monkeypatch)

    def test_exact_match(self, monkeypatch) -> None:
        wrapper = self._create_wrapper_for_find(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            container_file = Path(tmpdir) / "django__django-12345.sif"
            container_file.touch()

            data_point = {
                "instance_id": "django__django-12345",
                "dataset_name": "SWE-bench",
                "container_formatter": [str(Path(tmpdir) / "{instance_id}.sif")],
            }
            result = wrapper._find_container(data_point)
            assert result == str(container_file)

    def test_string_container_formatter(self, monkeypatch) -> None:
        wrapper = self._create_wrapper_for_find(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            container_file = Path(tmpdir) / "django__django-12345.sif"
            container_file.touch()

            data_point = {
                "instance_id": "django__django-12345",
                "dataset_name": "SWE-bench",
                "container_formatter": str(Path(tmpdir) / "{instance_id}.sif"),
            }
            result = wrapper._find_container(data_point)
            assert result == str(container_file)

    def test_1776_replacement(self, monkeypatch) -> None:
        wrapper = self._create_wrapper_for_find(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            container_file = Path(tmpdir) / "django_1776_django-12345.sif"
            container_file.touch()

            data_point = {
                "instance_id": "django__django-12345",
                "dataset_name": "SWE-bench",
                "container_formatter": [str(Path(tmpdir) / "{instance_id}.sif")],
            }
            result = wrapper._find_container(data_point)
            assert result == str(container_file)

    def test_s_replacement(self, monkeypatch) -> None:
        wrapper = self._create_wrapper_for_find(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            container_file = Path(tmpdir) / "django_s_django-12345.sif"
            container_file.touch()

            data_point = {
                "instance_id": "django__django-12345",
                "dataset_name": "SWE-bench",
                "container_formatter": [str(Path(tmpdir) / "{instance_id}.sif")],
            }
            result = wrapper._find_container(data_point)
            assert result == str(container_file)

    def test_lowercase_match(self, monkeypatch) -> None:
        wrapper = self._create_wrapper_for_find(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            container_file = Path(tmpdir) / "django_1776_django-12345.sif"
            container_file.touch()

            data_point = {
                "instance_id": "Django__Django-12345",
                "dataset_name": "SWE-bench",
                "container_formatter": [str(Path(tmpdir) / "{instance_id}.sif")],
            }
            result = wrapper._find_container(data_point)
            assert "django" in result.lower()

    def test_fuzzy_search(self, monkeypatch) -> None:
        wrapper = self._create_wrapper_for_find(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            container_file = Path(tmpdir) / "prefix_django__django-12345_suffix.sif"
            container_file.touch()

            data_point = {
                "instance_id": "django__django-12345",
                "dataset_name": "SWE-bench",
                "container_formatter": [str(Path(tmpdir) / "{instance_id}.sif")],
            }
            result = wrapper._find_container(data_point)
            assert result == str(container_file)

    def test_not_found(self, monkeypatch) -> None:
        wrapper = self._create_wrapper_for_find(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            data_point = {
                "instance_id": "nonexistent__repo-123",
                "dataset_name": "SWE-bench",
                "container_formatter": [str(Path(tmpdir) / "{instance_id}.sif")],
            }
            with pytest.raises(FileNotFoundError, match="No container file found"):
                wrapper._find_container(data_point)

    def test_r2e_gym_dataset(self, monkeypatch) -> None:
        wrapper = self._create_wrapper_for_find(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            # R2E-Gym modifies instance_id: org__RepoName- -> reponame_final_
            container_file = Path(tmpdir) / "reponame_final_123.sif"
            container_file.touch()

            data_point = {
                "instance_id": "org__RepoName-123",
                "dataset_name": "R2E-Gym/R2E-Gym-Subset",
                "container_formatter": [str(Path(tmpdir) / "{instance_id}.sif")],
            }
            result = wrapper._find_container(data_point)
            assert result == str(container_file)

    def test_swe_rebench_dataset(self, monkeypatch) -> None:
        wrapper = self._create_wrapper_for_find(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            # SWE-rebench fuzzy match: glob {instance_id}*.sif against the directory
            container_file = Path(tmpdir) / "owner__repo-123-abc.sif"
            container_file.touch()

            data_point = {
                "instance_id": "owner__repo-123",
                "dataset_name": "SWE-rebench",
                "container_formatter": [str(Path(tmpdir) / "{instance_id}.sif")],
            }
            result = wrapper._find_container(data_point)
            assert result == str(container_file)

    def test_swe_rebench_exact_match(self, monkeypatch) -> None:
        wrapper = self._create_wrapper_for_find(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            container_file = Path(tmpdir) / "owner__repo-123.sif"
            container_file.touch()

            data_point = {
                "instance_id": "owner__repo-123",
                "dataset_name": "SWE-rebench",
                "container_formatter": [str(Path(tmpdir) / "{instance_id}.sif")],
            }
            result = wrapper._find_container(data_point)
            assert result == str(container_file)

    def test_swe_rebench_not_found(self, monkeypatch) -> None:
        wrapper = self._create_wrapper_for_find(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            data_point = {
                "instance_id": "owner__repo-123",
                "dataset_name": "SWE-rebench",
                "container_formatter": [str(Path(tmpdir) / "{instance_id}.sif")],
            }
            with pytest.raises(FileNotFoundError, match="No SIF found"):
                wrapper._find_container(data_point)

    def test_multiple_container_formatters(self, monkeypatch) -> None:
        wrapper = self._create_wrapper_for_find(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            dir2 = Path(tmpdir) / "dir2"
            dir2.mkdir()
            container_file = dir2 / "django__django-12345.sif"
            container_file.touch()

            data_point = {
                "instance_id": "django__django-12345",
                "dataset_name": "SWE-bench",
                "container_formatter": [
                    str(Path(tmpdir) / "dir1" / "{instance_id}.sif"),
                    str(dir2 / "{instance_id}.sif"),
                ],
            }
            result = wrapper._find_container(data_point)
            assert result == str(container_file)


class TestSWEBenchWrapperBuildApptainerCommand:
    def test_basic_command(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            params = _make_instance_config(tmpdir)
            params.persistent_dir.mkdir(parents=True, exist_ok=True)
            (params.persistent_dir / "container_scripts").mkdir(parents=True, exist_ok=True)

            # Create openhands dirs needed for mount
            oh_dir = Path(params.openhands_setup_dir) / "OpenHands"
            for subdir in [".eval_sessions", "logs", "evaluation/oh"]:
                (oh_dir / subdir).mkdir(parents=True, exist_ok=True)
            miniforge = Path(params.openhands_setup_dir) / "miniforge3"
            miniforge.mkdir(parents=True, exist_ok=True)

            cmd_args = ExecuteContainerCommandArgs(
                command="echo hello",
                expected_file_pattern="/tmp/*.json",
                mode="agent",
                timeout=300,
            )
            result = wrapper._build_apptainer_command(params, cmd_args)
            assert "apptainer exec" in result
            assert "--writable-tmpfs" in result
            assert params.container in result

    def test_eval_mode_swebench_mounts(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            params = _make_instance_config(tmpdir)
            params.persistent_dir.mkdir(parents=True, exist_ok=True)

            oh_dir = Path(params.openhands_setup_dir) / "OpenHands"
            for subdir in [".eval_sessions", "logs", "evaluation/oh"]:
                (oh_dir / subdir).mkdir(parents=True, exist_ok=True)
            (Path(params.openhands_setup_dir) / "miniforge3").mkdir(parents=True, exist_ok=True)

            cmd_args = ExecuteContainerCommandArgs(
                command="run_eval",
                expected_file_pattern="/tmp/*.json",
                mode="eval",
                timeout=300,
            )
            result = wrapper._build_apptainer_command(params, cmd_args)
            assert "/swebench_setup" in result

    def test_memory_limit(self, monkeypatch) -> None:
        # No cgroups in the enroot sandbox, so the memory limit is enforced by
        # the gym-side RSS watchdog (_memory_watchdog), not a static ulimit
        # baked into the apptainer command.
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            params = _make_instance_config(tmpdir, apptainer_memory_limit_mb=16384)
            params.persistent_dir.mkdir(parents=True, exist_ok=True)

            oh_dir = Path(params.openhands_setup_dir) / "OpenHands"
            for subdir in [".eval_sessions", "logs", "evaluation/oh"]:
                (oh_dir / subdir).mkdir(parents=True, exist_ok=True)
            (Path(params.openhands_setup_dir) / "miniforge3").mkdir(parents=True, exist_ok=True)

            cmd_args = ExecuteContainerCommandArgs(
                command="echo hello",
                expected_file_pattern="/tmp/*.json",
                mode="agent",
                timeout=300,
            )
            result = wrapper._build_apptainer_command(params, cmd_args)
            assert "ulimit -v" not in result

    def test_nv_internal_eval_mounts(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            params = _make_instance_config(
                tmpdir,
                problem_info={
                    "problem_statement": "Fix",
                    "instance_id": "nv__test-1",
                    "base_commit": "abc",
                    "dataset_name": "nv-internal-1",
                    "split": "test",
                    "instance_dict": "{}",
                    "container_formatter": ["/containers/{instance_id}.sif"],
                },
            )
            params.persistent_dir.mkdir(parents=True, exist_ok=True)
            (params.persistent_dir / "run_script.sh").write_text("#!/bin/bash")
            (params.persistent_dir / "parsing_script.py").write_text("print('ok')")

            oh_dir = Path(params.openhands_setup_dir) / "OpenHands"
            for subdir in [".eval_sessions", "logs", "evaluation/oh"]:
                (oh_dir / subdir).mkdir(parents=True, exist_ok=True)
            (Path(params.openhands_setup_dir) / "miniforge3").mkdir(parents=True, exist_ok=True)

            cmd_args = ExecuteContainerCommandArgs(
                command="run_eval",
                expected_file_pattern="/tmp/*.json",
                mode="eval",
                timeout=300,
            )
            result = wrapper._build_apptainer_command(params, cmd_args)
            assert "/root/run_script.sh" in result
            assert "/root/parsing_script.py" in result
            assert "/root/patch.diff" in result

    def test_r2e_gym_agent_removes_tests(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            params = _make_instance_config(
                tmpdir,
                problem_info={
                    "problem_statement": "Fix",
                    "instance_id": "org__Repo-1",
                    "base_commit": "abc",
                    "dataset_name": "R2E-Gym/R2E-Gym-Subset",
                    "split": "test",
                    "instance_dict": "{}",
                    "container_formatter": ["/containers/{instance_id}.sif"],
                },
            )
            params.persistent_dir.mkdir(parents=True, exist_ok=True)

            oh_dir = Path(params.openhands_setup_dir) / "OpenHands"
            for subdir in [".eval_sessions", "logs", "evaluation/oh"]:
                (oh_dir / subdir).mkdir(parents=True, exist_ok=True)
            (Path(params.openhands_setup_dir) / "miniforge3").mkdir(parents=True, exist_ok=True)

            cmd_args = ExecuteContainerCommandArgs(
                command="run_agent",
                expected_file_pattern="/tmp/*.json",
                mode="agent",
                timeout=300,
            )
            wrapper._build_apptainer_command(params, cmd_args)
            # The rm -rf commands are in the container script, not the apptainer command
            script_path = params.persistent_dir / "container_scripts" / "agent_script.sh"
            script_content = script_path.read_text()
            assert "rm -rf" in script_content
            assert "r2e_tests" in script_content

    def test_swe_rebench_eval_env_args(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            params = _make_instance_config(
                tmpdir,
                problem_info={
                    "problem_statement": "Fix",
                    "instance_id": "owner__repo-1",
                    "base_commit": "abc",
                    "dataset_name": "SWE-rebench",
                    "split": "test",
                    "instance_dict": "{}",
                    "container_formatter": ["/containers/{instance_id}.sif"],
                },
            )
            params.persistent_dir.mkdir(parents=True, exist_ok=True)

            # Create eval meta files
            eval_meta_dir = params.persistent_dir / "eval_meta"
            eval_meta_dir.mkdir(parents=True, exist_ok=True)
            (eval_meta_dir / "expected_passed.json").write_text("[]")
            (eval_meta_dir / "fail_to_pass.json").write_text("[]")
            (eval_meta_dir / "pass_to_pass.json").write_text("[]")

            oh_dir = Path(params.openhands_setup_dir) / "OpenHands"
            for subdir in [".eval_sessions", "logs", "evaluation/oh"]:
                (oh_dir / subdir).mkdir(parents=True, exist_ok=True)
            (Path(params.openhands_setup_dir) / "miniforge3").mkdir(parents=True, exist_ok=True)

            cmd_args = ExecuteContainerCommandArgs(
                command="run_eval",
                expected_file_pattern="/tmp/*.json",
                mode="eval",
                timeout=300,
            )
            result = wrapper._build_apptainer_command(params, cmd_args)
            assert "_JAVA_OPTIONS" in result
            assert "/swe_rebench_setup" in result

    def test_prompt_template_mounts(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            user_prompt = Path(tmpdir) / "user_prompt.j2"
            system_prompt = Path(tmpdir) / "system_prompt.j2"
            user_prompt.write_text("user prompt")
            system_prompt.write_text("system prompt")

            params = _make_instance_config(
                tmpdir,
                resolved_user_prompt_template=str(user_prompt),
                resolved_system_prompt_template=str(system_prompt),
            )
            params.persistent_dir.mkdir(parents=True, exist_ok=True)

            oh_dir = Path(params.openhands_setup_dir) / "OpenHands"
            for subdir in [".eval_sessions", "logs", "evaluation/oh"]:
                (oh_dir / subdir).mkdir(parents=True, exist_ok=True)
            (Path(params.openhands_setup_dir) / "miniforge3").mkdir(parents=True, exist_ok=True)

            cmd_args = ExecuteContainerCommandArgs(
                command="echo hello",
                expected_file_pattern="/tmp/*.json",
                mode="agent",
                timeout=300,
            )
            result = wrapper._build_apptainer_command(params, cmd_args)
            assert "user_prompt.j2" in result
            assert "system_prompt.j2" in result


class TestSWEBenchWrapperGetOpenhandsTrajectory:
    def test_with_completions(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            instance_id = "django__django-12345"
            completions_dir = Path(tmpdir) / instance_id / "llm_completions" / instance_id
            completions_dir.mkdir(parents=True)

            completion_data = {
                "messages": [
                    {"content": [{"type": "text", "text": "system prompt"}], "role": "system"},
                    {"content": [{"type": "text", "text": "Fix the bug"}], "role": "user"},
                ],
                "provider_specific_fields": {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3, 4],
                    "routed_experts": [[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]]],
                },
                "response": {
                    "choices": [
                        {
                            "message": {
                                "content": "I'll fix it",
                                "role": "assistant",
                            }
                        }
                    ]
                },
                "kwargs": {"tools": [{"type": "function", "function": {"name": "execute_bash"}}]},
            }
            (completions_dir / "001_completion.json").write_text(json.dumps(completion_data))

            messages, tools, _ = wrapper.get_openhands_trajectory_from_completions(Path(tmpdir), instance_id)
            assert len(messages) == 3  # system, user, assistant
            assert messages[2]["role"] == "assistant"
            assert messages[2]["prompt_token_ids"] == [1, 2]
            assert messages[2]["routed_experts"] == [
                [[0, 1]],
                [[2, 3]],
                [[4, 5]],
                [[6, 7]],
            ]
            assert len(tools) == 1

    def test_no_completions_dir(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            messages, tools, _ = wrapper.get_openhands_trajectory_from_completions(Path(tmpdir), "nonexistent")
            assert messages == []
            assert tools == []

    def test_no_completion_files(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            instance_id = "test-instance"
            completions_dir = Path(tmpdir) / instance_id / "llm_completions" / instance_id
            completions_dir.mkdir(parents=True)

            messages, tools, _ = wrapper.get_openhands_trajectory_from_completions(Path(tmpdir), instance_id)
            assert messages == []
            assert tools == []

    def test_assistant_with_no_content_or_tool_calls(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            instance_id = "test-instance"
            completions_dir = Path(tmpdir) / instance_id / "llm_completions" / instance_id
            completions_dir.mkdir(parents=True)

            completion_data = {
                "messages": [{"role": "user", "content": "hello"}],
                "response": {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "role": "assistant",
                            }
                        }
                    ]
                },
                "kwargs": {},
            }
            (completions_dir / "001_completion.json").write_text(json.dumps(completion_data))

            messages, tools, _ = wrapper.get_openhands_trajectory_from_completions(Path(tmpdir), instance_id)
            assert len(messages) == 1  # only user, assistant not appended


class TestSWEBenchWrapperSetupParams:
    def _setup_oh_dirs(self, wrapper):
        oh_dir = wrapper._swe_bench_wrapper_server_config.openhands_setup_dir / "OpenHands"
        for subdir in [".eval_sessions", "logs", "evaluation/oh"]:
            (oh_dir / subdir).mkdir(parents=True, exist_ok=True)
        miniforge = wrapper._swe_bench_wrapper_server_config.openhands_setup_dir / "miniforge3"
        miniforge.mkdir(parents=True, exist_ok=True)

    def test_basic_setup_params(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            container_file = Path(tmpdir) / "django__django-12345.sif"
            container_file.touch()

            wrapper.config.container_formatter = [str(Path(tmpdir) / "{instance_id}.sif")]
            self._setup_oh_dirs(wrapper)

            body = NeMoGymResponseCreateParamsNonStreaming(
                model="test-model",
                input=[],
                temperature=1.0,
                top_p=1.0,
                metadata={
                    "problem_statement": "Fix bug",
                    "instance_id": "django__django-12345",
                    "base_commit": "abc123",
                    "dataset_name": "SWE-bench",
                    "split": "test",
                    "instance_dict": json.dumps({"repo": "django/django"}),
                },
            )

            params, processor = wrapper._setup_params(body)
            assert isinstance(params, SWEBenchWrapperInstanceConfig)
            assert isinstance(processor, SweBenchDatasetProcessor)
            assert params.instance_id == "django__django-12345"
            assert params.eval_command is not None
            assert params.agent_command is not None
            assert params.metrics_fpath.exists()

    def test_setup_params_nv_internal(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            container_file = Path(tmpdir) / "nv__test-1.sif"
            container_file.touch()

            wrapper.config.container_formatter = [str(Path(tmpdir) / "{instance_id}.sif")]
            self._setup_oh_dirs(wrapper)

            body = NeMoGymResponseCreateParamsNonStreaming(
                model="test-model",
                input=[],
                temperature=1.0,
                top_p=1.0,
                metadata={
                    "problem_statement": "Fix",
                    "instance_id": "nv__test-1",
                    "base_commit": "abc",
                    "dataset_name": "nv-internal-1",
                    "split": "test",
                    "instance_dict": json.dumps(
                        {
                            "base_dockerfile": "",
                            "instance_dockerfile": "",
                            "before_repo_set_cmd": "",
                            "selected_test_files_to_run": "[]",
                            "run_script.sh": "#!/bin/bash",
                            "parsing_script.py": "print('ok')",
                            "base_commit": "abc",
                        }
                    ),
                },
            )

            params, processor = wrapper._setup_params(body)
            assert isinstance(processor, NVInternalDatasetProcessor)

    def test_setup_params_r2e_gym(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            container_file = Path(tmpdir) / "repo_final_1.sif"
            container_file.touch()

            wrapper.config.container_formatter = [str(Path(tmpdir) / "{instance_id}.sif")]
            self._setup_oh_dirs(wrapper)

            body = NeMoGymResponseCreateParamsNonStreaming(
                model="test-model",
                input=[],
                temperature=1.0,
                top_p=1.0,
                metadata={
                    "problem_statement": "Fix",
                    "instance_id": "org__Repo-1",
                    "base_commit": "abc",
                    "dataset_name": "R2E-Gym/R2E-Gym-Subset",
                    "split": "test",
                    "instance_dict": json.dumps({"repo": "org/Repo"}),
                },
            )

            params, processor = wrapper._setup_params(body)
            assert isinstance(processor, R2EGymDatasetProcessor)

    def test_setup_params_with_prompt_overrides(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        wrapper.config.agent_prompt_overrides = [
            AgentPromptOverride(agent_cls="CodexAgent"),
            AgentPromptOverride(agent_cls="OpenCodeAgent"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            container_file = Path(tmpdir) / "django__django-12345.sif"
            container_file.touch()

            wrapper.config.container_formatter = [str(Path(tmpdir) / "{instance_id}.sif")]
            self._setup_oh_dirs(wrapper)

            body = NeMoGymResponseCreateParamsNonStreaming(
                model="test-model",
                input=[],
                temperature=1.0,
                top_p=1.0,
                metadata={
                    "problem_statement": "Fix",
                    "instance_id": "django__django-12345",
                    "base_commit": "abc",
                    "dataset_name": "SWE-bench",
                    "split": "test",
                    "instance_dict": json.dumps({"repo": "django/django"}),
                },
            )

            params, _ = wrapper._setup_params(body)
            # deterministic selection based on instance_id
            assert params.resolved_agent_cls in ["CodexAgent", "OpenCodeAgent"]


class TestSWEBenchWrapperResponses:
    def _setup_oh_dirs(self, wrapper):
        oh_dir = wrapper._swe_bench_wrapper_server_config.openhands_setup_dir / "OpenHands"
        for subdir in [".eval_sessions", "logs", "evaluation/oh"]:
            (oh_dir / subdir).mkdir(parents=True, exist_ok=True)
        miniforge = wrapper._swe_bench_wrapper_server_config.openhands_setup_dir / "miniforge3"
        miniforge.mkdir(parents=True, exist_ok=True)

    @pytest.mark.asyncio
    async def test_responses_success(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            container_file = Path(tmpdir) / "django__django-12345.sif"
            container_file.touch()

            wrapper.config.container_formatter = [str(Path(tmpdir) / "{instance_id}.sif")]
            self._setup_oh_dirs(wrapper)

            body = NeMoGymResponseCreateParamsNonStreaming(
                model="test-model",
                input=[],
                temperature=1.0,
                top_p=1.0,
                metadata={
                    "problem_statement": "Fix bug",
                    "instance_id": "django__django-12345",
                    "base_commit": "abc123",
                    "dataset_name": "SWE-bench",
                    "split": "test",
                    "instance_dict": json.dumps({"repo": "django/django"}),
                },
            )

            mock_response = NeMoGymResponse(
                id="swebench-django__django-12345",
                created_at=123,
                model="test-model",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
                metadata={
                    "input": "[]",
                    "metrics": json.dumps({"resolved": True}),
                    "instance_config": "{}",
                },
            )

            with patch.object(wrapper, "_inner_responses", new_callable=AsyncMock, return_value=mock_response):
                result = await wrapper.responses(body)
                assert result.id == "swebench-django__django-12345"

    @pytest.mark.asyncio
    async def test_responses_exception_writes_traceback(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            container_file = Path(tmpdir) / "django__django-12345.sif"
            container_file.touch()

            wrapper.config.container_formatter = [str(Path(tmpdir) / "{instance_id}.sif")]
            self._setup_oh_dirs(wrapper)

            body = NeMoGymResponseCreateParamsNonStreaming(
                model="test-model",
                input=[],
                temperature=1.0,
                top_p=1.0,
                metadata={
                    "problem_statement": "Fix bug",
                    "instance_id": "django__django-12345",
                    "base_commit": "abc",
                    "dataset_name": "SWE-bench",
                    "split": "test",
                    "instance_dict": json.dumps({"repo": "django/django"}),
                },
            )

            with patch.object(
                wrapper,
                "_inner_responses",
                new_callable=AsyncMock,
                side_effect=RuntimeError("test error"),
            ):
                with pytest.raises(RuntimeError, match="test error"):
                    await wrapper.responses(body)


class TestSWEBenchWrapperRun:
    @pytest.mark.asyncio
    async def test_run_resolved(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)

        mock_response = NeMoGymResponse(
            id="swebench-test",
            created_at=123,
            model="test-model",
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
            metadata={
                "input": "[]",
                "metrics": json.dumps({"resolved": True, "patch_exists": True}),
                "instance_config": _make_instance_config(tempfile.mkdtemp()).model_dump_json(),
            },
        )

        with patch.object(SWEBenchWrapper, "responses", new_callable=AsyncMock, return_value=mock_response):
            from nemo_gym.base_resources_server import BaseRunRequest

            body = BaseRunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                    model="test-model",
                    input=[],
                    metadata={
                        "problem_statement": "Fix",
                        "instance_id": "test-1",
                        "base_commit": "abc",
                        "dataset_name": "SWE-bench",
                        "split": "test",
                        "instance_dict": "{}",
                    },
                )
            )

            result = await wrapper.run(body)
            assert isinstance(result, SWEBenchVerifyResponse)
            assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_run_not_resolved(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)

        mock_response = NeMoGymResponse(
            id="swebench-test",
            created_at=123,
            model="test-model",
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
            metadata={
                "input": "[]",
                "metrics": json.dumps({"resolved": False, "patch_exists": True}),
                "instance_config": _make_instance_config(tempfile.mkdtemp()).model_dump_json(),
            },
        )

        with patch.object(SWEBenchWrapper, "responses", new_callable=AsyncMock, return_value=mock_response):
            from nemo_gym.base_resources_server import BaseRunRequest

            body = BaseRunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                    model="test-model",
                    input=[],
                    metadata={
                        "problem_statement": "Fix",
                        "instance_id": "test-1",
                        "base_commit": "abc",
                        "dataset_name": "SWE-bench",
                        "split": "test",
                        "instance_dict": "{}",
                    },
                )
            )

            result = await wrapper.run(body)
            assert isinstance(result, SWEBenchVerifyResponse)
            assert result.reward == 0.0


########################################
# _load_rebench_log_parsers tests
########################################


class TestLoadRebenchLogParsers:
    def test_loads_from_agent_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rebench_dir = Path(tmpdir)
            agent_dir = rebench_dir / "agent"
            agent_dir.mkdir()
            (agent_dir / "log_parsers.py").write_text("NAME_TO_PARSER = {'test': lambda x: {}}\n")

            from responses_api_agents.swe_agents.app import _load_rebench_log_parsers

            mod = _load_rebench_log_parsers(rebench_dir)
            assert hasattr(mod, "NAME_TO_PARSER")
            assert "test" in mod.NAME_TO_PARSER

    def test_loads_from_lib_agent_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rebench_dir = Path(tmpdir)
            lib_agent_dir = rebench_dir / "lib" / "agent"
            lib_agent_dir.mkdir(parents=True)
            (lib_agent_dir / "log_parsers.py").write_text("NAME_TO_PARSER = {'lib_test': lambda x: {}}\n")

            from responses_api_agents.swe_agents.app import _load_rebench_log_parsers

            mod = _load_rebench_log_parsers(rebench_dir)
            assert "lib_test" in mod.NAME_TO_PARSER
