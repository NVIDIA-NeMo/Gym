# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for anyterminal_agent.

Modeled on anyswe_agent's tests: these exercise pure logic (no Apptainer/Docker) —
runner-script generation, container discovery, deps-key derivation, setup-script presence,
and example-data shape. Heavy side effects in model_post_init (deps + harness install) are
bypassed by calling staticmethods/properties directly rather than constructing the agent.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from nemo_gym import PARENT_DIR
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.anyterminal_agent.app import (
    _RUNNER_TEMPLATE,
    ActiveContainerProcess,
    AnyTerminalAgent,
    AnyTerminalAgentConfig,
    AnyTerminalInstanceConfig,
    GymAgentHarnessProcessor,
    RunTerminalAgent,
    _instruction_from_input,
    _read_task_meta,
    _safe_config_json,
    update_metrics,
)


def _config(**overrides) -> AnyTerminalAgentConfig:
    base = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="anyterminal_agent",
        model_server={"type": "responses_api_models", "name": "policy_model"},
        agent_server_module="responses_api_agents.hermes_agent.app",
        agent_server_class="HermesAgent",
        agent_config_class="HermesAgentConfig",
    )
    base.update(overrides)
    return AnyTerminalAgentConfig(**base)


class TestRunnerTemplate:
    def _render(self) -> str:
        return _RUNNER_TEMPLATE.format(
            agent_module="responses_api_agents.hermes_agent.app",
            agent_class="HermesAgent",
            agent_cfg_class="HermesAgentConfig",
            agent_class_lower="hermesagent",
        )

    def test_renders_valid_python(self) -> None:
        rendered = self._render()
        # Must be syntactically valid Python and reference the agent class.
        compile(rendered, "<runner>", "exec")
        assert "HermesAgent(config=config" in rendered

    def test_response_is_written_back(self) -> None:
        # The runner's agent-agnostic contract is to persist the response where the host reads it.
        assert "/trajectories_mount/response.json" in _RUNNER_TEMPLATE
        assert "response.model_dump_json()" in _RUNNER_TEMPLATE

    def test_sampling_is_forwarded(self) -> None:
        rendered = self._render()
        compile(rendered, "<runner>", "exec")
        # Read from env, forwarded onto the body, and filtered to the agent config's fields.
        assert "NGTB_SAMPLING" in rendered
        assert "**SAMPLING," in rendered
        assert "HermesAgentConfig.model_fields" in rendered


class TestAgentKey:
    def test_key_from_module(self) -> None:
        proc = GymAgentHarnessProcessor(config=_config())
        assert proc._agent_key == "hermes_agent"

    def test_key_for_claude(self) -> None:
        proc = GymAgentHarnessProcessor(
            config=_config(
                agent_server_module="responses_api_agents.claude_code_agent.app",
                agent_server_class="ClaudeCodeAgent",
                agent_config_class="ClaudeCodeAgentConfig",
            )
        )
        assert proc._agent_key == "claude_code_agent"


class TestFindContainer:
    def _stub(self, **cfg_overrides) -> SimpleNamespace:
        # _find_container only touches self.config.tb_sif_dir; avoid building the whole agent.
        return SimpleNamespace(config=_config(**cfg_overrides))

    def test_prebuilt_sif_exact_match(self, tmp_path: Path) -> None:
        sif = tmp_path / "fix-git.sif"
        sif.write_text("")
        found = AnyTerminalAgent._find_container(self._stub(tb_sif_dir=str(tmp_path)), "fix-git", "ubuntu:22.04")
        assert found == str(sif.resolve())

    def test_sif_name_variant_underscore(self, tmp_path: Path) -> None:
        # Task names with dashes also match an underscored SIF filename.
        sif = tmp_path / "fix_git.sif"
        sif.write_text("")
        found = AnyTerminalAgent._find_container(self._stub(tb_sif_dir=str(tmp_path)), "fix-git", "ubuntu:22.04")
        assert found == str(sif.resolve())

    def test_falls_back_to_docker_uri_when_no_sif(self, tmp_path: Path) -> None:
        found = AnyTerminalAgent._find_container(self._stub(tb_sif_dir=str(tmp_path)), "nope", "ubuntu:22.04")
        assert found == "docker://ubuntu:22.04"

    def test_falls_back_to_docker_uri_when_no_sif_dir(self) -> None:
        found = AnyTerminalAgent._find_container(self._stub(), "nope", "ubuntu:22.04")
        assert found == "docker://ubuntu:22.04"

    def test_existing_docker_uri_passed_through(self) -> None:
        found = AnyTerminalAgent._find_container(self._stub(), "nope", "docker://myrepo/img:tag")
        assert found == "docker://myrepo/img:tag"


class TestSetupScriptsExist:
    def test_supported_agents_have_deps_scripts(self) -> None:
        scripts = Path(__file__).parent.parent / "setup_scripts"
        assert (scripts / "hermes_agent_deps.sh").exists()
        assert (scripts / "_portable_python.sh").exists()


class TestExampleData:
    def _example(self) -> Path:
        # Data files are gitignored; test whichever materialized example is present, else skip.
        data_dir = Path(__file__).parent.parent / "data"
        for name in ("terminal_bench_smoke.jsonl", "terminal_bench_example.jsonl"):
            p = data_dir / name
            if p.exists():
                return p
        candidates = sorted(data_dir.glob("*.jsonl"))
        return candidates[0] if candidates else data_dir / "missing.jsonl"

    def test_example_jsonl_parses(self) -> None:
        example = self._example()
        if not example.exists():
            pytest.skip("no example .jsonl present (data/ is gitignored)")
        rows = [json.loads(line) for line in example.read_text().splitlines() if line.strip()]
        assert rows
        for row in rows:
            rcp = row["responses_create_params"]
            assert "input" in rcp
            assert "metadata" in rcp
            md = rcp["metadata"]
            assert "task_name" in md or "instance_id" in md


# ── helpers ───────────────────────────────────────────────────────────────────────


def _make_body(content: str = "solve this") -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": content}], model="test-model")


def _make_instance_config(tmp_path: Path, **overrides) -> AnyTerminalInstanceConfig:
    persistent_dir = tmp_path / "persistent"
    persistent_dir.mkdir(parents=True, exist_ok=True)
    verifier_dir = tmp_path / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    metrics_fpath = tmp_path / "metrics.json"
    metrics_fpath.write_text("{}")
    defaults: dict = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="anyterminal_agent",
        model_server=None,
        agent_server_module="responses_api_agents.hermes_agent.app",
        agent_server_class="HermesAgent",
        agent_config_class="HermesAgentConfig",
        run_session_id="test_session",
        base_results_dir=tmp_path / "results",
        model_server_url="",
        model_name="test-model",
        nemo_gym_root=PARENT_DIR,
        agent_deps_dir=tmp_path / "deps",
        problem_info={"task_name": "fix-git", "task_dir": str(tmp_path), "docker_image": "ubuntu:22.04"},
        body=_make_body(),
        persistent_dir=persistent_dir,
        verifier_dir=verifier_dir,
        agent_run_id="fix-git_12345",
        metrics_fpath=metrics_fpath,
        container="docker://ubuntu:22.04",
        ray_queue_timestamp=0.0,
    )
    defaults.update(overrides)
    return AnyTerminalInstanceConfig(**defaults)


# ── _read_task_meta ───────────────────────────────────────────────────────────────


class TestReadTaskMeta:
    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        assert _read_task_meta(tmp_path / "nonexistent") == {}

    def test_reads_timeouts_from_toml(self, tmp_path: Path) -> None:
        (tmp_path / "task.toml").write_bytes(b"[agent]\ntimeout_sec = 1800\n[verifier]\ntimeout_sec = 300\n")
        result = _read_task_meta(tmp_path)
        assert result["agent_timeout_sec"] == 1800
        assert result["verifier_timeout_sec"] == 300

    def test_reads_workdir_from_dockerfile(self, tmp_path: Path) -> None:
        (tmp_path / "environment").mkdir()
        (tmp_path / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\nWORKDIR /app\n")
        assert _read_task_meta(tmp_path).get("workdir") == "/app"

    def test_last_workdir_wins(self, tmp_path: Path) -> None:
        (tmp_path / "environment").mkdir()
        (tmp_path / "environment" / "Dockerfile").write_text("FROM ubuntu\nWORKDIR /first\nWORKDIR /workspace\n")
        assert _read_task_meta(tmp_path)["workdir"] == "/workspace"

    def test_toml_without_sections_returns_nones(self, tmp_path: Path) -> None:
        (tmp_path / "task.toml").write_bytes(b"[environment]\ndocker_image = 'ubuntu:22.04'\n")
        result = _read_task_meta(tmp_path)
        assert result.get("agent_timeout_sec") is None
        assert result.get("verifier_timeout_sec") is None


# ── _instruction_from_input ───────────────────────────────────────────────────────


class TestInstructionFromInput:
    def _body(self, input_val):
        return SimpleNamespace(input=input_val)

    def test_str_input_passthrough(self) -> None:
        assert _instruction_from_input(self._body("hello world")) == "hello world"

    def test_dict_messages_joined(self) -> None:
        msgs = [{"role": "user", "content": "part1"}, {"role": "user", "content": "part2"}]
        assert _instruction_from_input(self._body(msgs)) == "part1\npart2"

    def test_object_messages(self) -> None:
        msgs = [SimpleNamespace(content="foo"), SimpleNamespace(content="bar")]
        assert _instruction_from_input(self._body(msgs)) == "foo\nbar"

    def test_content_part_list(self) -> None:
        msg = {"content": [{"type": "text", "text": "hello"}, {"type": "text", "text": " world"}]}
        assert _instruction_from_input(self._body([msg])) == "hello world"

    def test_none_content_skipped(self) -> None:
        msgs = [{"role": "user", "content": None}, {"role": "user", "content": "hi"}]
        assert _instruction_from_input(self._body(msgs)) == "hi"

    def test_empty_list(self) -> None:
        assert _instruction_from_input(self._body([])) == ""


# ── update_metrics ────────────────────────────────────────────────────────────────


class TestUpdateMetrics:
    def test_merges_with_existing(self, tmp_path: Path) -> None:
        fpath = tmp_path / "m.json"
        fpath.write_text('{"resolved": true}')
        update_metrics(fpath, {"total_run_time": 5.0})
        d = json.loads(fpath.read_text())
        assert d["resolved"] is True
        assert d["total_run_time"] == 5.0

    def test_none_values_excluded(self, tmp_path: Path) -> None:
        fpath = tmp_path / "m.json"
        fpath.write_text("{}")
        update_metrics(fpath, {"resolved": None, "total_run_time": 3.0})
        d = json.loads(fpath.read_text())
        assert "resolved" not in d
        assert d["total_run_time"] == 3.0

    def test_overwrites_existing_key(self, tmp_path: Path) -> None:
        fpath = tmp_path / "m.json"
        fpath.write_text('{"resolved": false}')
        update_metrics(fpath, {"resolved": True})
        assert json.loads(fpath.read_text())["resolved"] is True


# ── _safe_config_json ─────────────────────────────────────────────────────────────


class TestSafeConfigJson:
    def test_api_key_redacted(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, agent_kwargs={"api_key": "sk-secret", "model": "gpt-4"})
        result = json.loads(_safe_config_json(cfg))
        assert result["agent_kwargs"]["api_key"] == "***"
        assert result["agent_kwargs"]["model"] == "gpt-4"

    def test_secret_token_password_redacted(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, agent_kwargs={"my_secret": "x", "auth_token": "y", "password": "z"})
        result = json.loads(_safe_config_json(cfg))
        for key in ("my_secret", "auth_token", "password"):
            assert result["agent_kwargs"][key] == "***"

    def test_agent_command_str_excluded(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, agent_command_str="apptainer exec ...")
        assert "agent_command_str" not in json.loads(_safe_config_json(cfg))

    def test_indent_produces_multiline(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        assert "\n" in _safe_config_json(cfg, indent=2)


# ── AnyTerminalInstanceConfig properties ──────────────────────────────────────────


class TestInstanceConfigProperties:
    def test_task_name_from_problem_info(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, problem_info={"task_name": "my-task", "task_dir": str(tmp_path)})
        assert cfg.task_name == "my-task"

    def test_task_name_falls_back_to_instance_id(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, problem_info={"instance_id": "tb::my-task", "task_dir": str(tmp_path)})
        assert cfg.task_name == "tb::my-task"

    def test_task_name_unknown_when_neither_present(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, problem_info={"task_dir": str(tmp_path)})
        assert cfg.task_name == "unknown"

    def test_instance_id_from_problem_info(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(
            tmp_path,
            problem_info={"task_name": "t", "instance_id": "tb::t", "task_dir": str(tmp_path)},
        )
        assert cfg.instance_id == "tb::t"

    def test_instance_id_falls_back_to_task_name(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, problem_info={"task_name": "my-task", "task_dir": str(tmp_path)})
        assert cfg.instance_id == "my-task"


# ── _apptainer_exec ───────────────────────────────────────────────────────────────


class TestApptainerExec:
    def _params(self, **kw) -> AnyTerminalInstanceConfig:
        defaults = dict(container="docker://ubuntu:22.04", apptainer_memory_limit_mb=32768)
        defaults.update(kw)
        return AnyTerminalInstanceConfig.model_construct(**defaults)

    def test_basic_structure(self) -> None:
        cmd = AnyTerminalAgent._apptainer_exec(self._params(), [], "bash /run.sh")
        assert "apptainer exec" in cmd
        assert "docker://ubuntu:22.04" in cmd
        assert "bash /run.sh" in cmd

    def test_memory_limit_applied(self) -> None:
        cmd = AnyTerminalAgent._apptainer_exec(self._params(apptainer_memory_limit_mb=4096), [], "bash /run.sh")
        assert f"ulimit -v {4096 * 1024}" in cmd

    def test_no_ulimit_when_limit_zero(self) -> None:
        cmd = AnyTerminalAgent._apptainer_exec(self._params(apptainer_memory_limit_mb=0), [], "bash /run.sh")
        assert "ulimit" not in cmd

    def test_workdir_flag(self) -> None:
        cmd = AnyTerminalAgent._apptainer_exec(self._params(), [], "bash /run.sh", workdir="/workspace")
        assert "--pwd /workspace" in cmd

    def test_no_workdir_by_default(self) -> None:
        cmd = AnyTerminalAgent._apptainer_exec(self._params(), [], "bash /run.sh")
        assert "--pwd" not in cmd

    def test_mounts_included(self) -> None:
        mounts = ["--mount type=bind,src=/src,dst=/dst"]
        cmd = AnyTerminalAgent._apptainer_exec(self._params(), mounts, "bash /run.sh")
        assert "--mount type=bind,src=/src,dst=/dst" in cmd

    def test_env_included(self) -> None:
        cmd = AnyTerminalAgent._apptainer_exec(self._params(), [], "bash /run.sh", env="--env FOO=bar ")
        assert "--env FOO=bar" in cmd


# ── _ray_resource_opts ────────────────────────────────────────────────────────────


class TestRayResourceOpts:
    def _params(self, problem_info: dict, agent_overhead_mb: int = 2048) -> AnyTerminalInstanceConfig:
        return AnyTerminalInstanceConfig.model_construct(
            problem_info=problem_info,
            agent_overhead_mb=agent_overhead_mb,
        )

    def test_defaults_to_one_cpu(self) -> None:
        assert AnyTerminalAgent._ray_resource_opts(self._params({}))["num_cpus"] == 1

    def test_custom_cpus(self) -> None:
        assert AnyTerminalAgent._ray_resource_opts(self._params({"cpus": "2.5"}))["num_cpus"] == 2.5

    def test_memory_includes_overhead(self) -> None:
        opts = AnyTerminalAgent._ray_resource_opts(self._params({"memory_mb": "4096"}))
        assert opts["memory"] == (4096 + 2048) * 1024 * 1024

    def test_zero_memory_excluded(self) -> None:
        assert "memory" not in AnyTerminalAgent._ray_resource_opts(self._params({"memory_mb": "0"}))

    def test_gpus_set(self) -> None:
        assert AnyTerminalAgent._ray_resource_opts(self._params({"gpus": "1"}))["num_gpus"] == 1.0

    def test_zero_gpus_excluded(self) -> None:
        assert "num_gpus" not in AnyTerminalAgent._ray_resource_opts(self._params({"gpus": "0"}))

    def test_invalid_cpus_defaults_to_one(self) -> None:
        opts = AnyTerminalAgent._ray_resource_opts(self._params({"cpus": "not-a-number"}))
        assert opts["num_cpus"] == 1

    def test_none_cpus_defaults_to_one(self) -> None:
        assert AnyTerminalAgent._ray_resource_opts(self._params({"cpus": None}))["num_cpus"] == 1


# ── GymAgentHarnessProcessor.setup ───────────────────────────────────────────────


class TestHarnessProcessorSetup:
    def _proc_no_script(self) -> GymAgentHarnessProcessor:
        return GymAgentHarnessProcessor(
            config=_config(
                agent_server_module="responses_api_agents.no_such_agent.app",
                agent_server_class="NoSuchAgent",
                agent_config_class="NoSuchAgentConfig",
            )
        )

    def test_no_script_creates_empty_deps(self, tmp_path: Path) -> None:
        proc = self._proc_no_script()
        with patch.object(type(proc), "_parent", new_callable=PropertyMock, return_value=tmp_path):
            result = proc.setup()
        expected = tmp_path / "deps" / "anyterminal_no_such_agent_deps"
        assert result == expected
        assert expected.exists()
        assert (expected / ".installed").exists()

    def test_sentinel_match_skips_reinstall(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        proc = self._proc_no_script()
        with patch.object(type(proc), "_parent", new_callable=PropertyMock, return_value=tmp_path):
            proc.setup()
            proc.setup()
        assert "already at" in capsys.readouterr().out


# ── GymAgentHarnessProcessor.get_run_command ─────────────────────────────────────


class TestGetRunCommand:
    def test_writes_instruction_and_runner(self, tmp_path: Path) -> None:
        persistent_dir = tmp_path / "persistent"
        persistent_dir.mkdir()
        cfg = AnyTerminalInstanceConfig.model_construct(
            agent_server_module="responses_api_agents.hermes_agent.app",
            agent_server_class="HermesAgent",
            agent_config_class="HermesAgentConfig",
            body=SimpleNamespace(input=[SimpleNamespace(content="the task")]),
            persistent_dir=persistent_dir,
            tb_agent_timeout=1800,
        )
        cmd = GymAgentHarnessProcessor(config=cfg).get_run_command()
        assert (persistent_dir / "instruction.txt").read_text() == "the task"
        runner = (persistent_dir / "agent_runner.py").read_text()
        assert "HermesAgent" in runner
        assert "HermesAgentConfig" in runner
        assert "timeout 1800" in cmd
        assert "/agent_deps_mount/bin/python" in cmd


# ── _build_agent_cmd ──────────────────────────────────────────────────────────────


class TestBuildAgentCmd:
    def _stub(self):
        return SimpleNamespace(_apptainer_exec=AnyTerminalAgent._apptainer_exec)

    def test_produces_apptainer_command(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        cmd = AnyTerminalAgent._build_agent_cmd(self._stub(), cfg)
        assert "apptainer exec" in cmd
        assert "run_script.sh" in cmd

    def test_script_written_to_disk(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        AnyTerminalAgent._build_agent_cmd(self._stub(), cfg)
        script = (cfg.persistent_dir / "container_scripts" / "run_script.sh").read_text()
        assert "agent_spinup_timestamp" in script
        assert "agent_done" in script

    def test_model_url_env_when_set(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, model_server_url="http://model:8000")
        cmd = AnyTerminalAgent._build_agent_cmd(self._stub(), cfg)
        assert "NGTB_MODEL_URL" in cmd

    def test_no_model_url_env_when_empty(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, model_server_url="")
        cmd = AnyTerminalAgent._build_agent_cmd(self._stub(), cfg)
        assert "NGTB_MODEL_URL" not in cmd

    def test_workdir_passed_when_in_problem_info(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(
            tmp_path,
            problem_info={
                "task_name": "t",
                "task_dir": str(tmp_path),
                "docker_image": "ubuntu:22.04",
                "workdir": "/app",
            },
        )
        cmd = AnyTerminalAgent._build_agent_cmd(self._stub(), cfg)
        assert "--pwd /app" in cmd


# ── RunTerminalAgent.process_single_datapoint ────────────────────────────────────


class TestProcessSingleDatapoint:
    def _mock_container(self, tmp_path: Path, returncode: int = 0):
        proc = MagicMock()
        proc.returncode = returncode
        proc.pid = 99999
        proc.communicate = AsyncMock(return_value=(b"", b""))
        proc.wait = AsyncMock()
        log_file = open(tmp_path / "log.txt", "w")
        return ActiveContainerProcess.model_construct(
            process=proc, log_file=log_file, log_file_path=tmp_path / "log.txt"
        )

    async def test_resolved_when_reward_positive(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        (cfg.verifier_dir / "reward.txt").write_text("1.0")
        mock_ctr = self._mock_container(tmp_path)

        runner = RunTerminalAgent(config=cfg)
        with patch.object(RunTerminalAgent, "_start", new=AsyncMock(return_value=mock_ctr)):
            with patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock(return_value=None)):
                result = await runner.process_single_datapoint()

        assert result is True
        assert json.loads(cfg.metrics_fpath.read_text())["resolved"] is True

    async def test_unresolved_when_no_reward_file(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        mock_ctr = self._mock_container(tmp_path)

        runner = RunTerminalAgent(config=cfg)
        with patch.object(RunTerminalAgent, "_start", new=AsyncMock(return_value=mock_ctr)):
            with patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock(return_value=None)):
                result = await runner.process_single_datapoint()

        assert result is False

    async def test_container_timeout_sets_flag(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        proc = MagicMock()
        proc.returncode = None
        proc.pid = 99999
        proc.wait = AsyncMock()
        log_file = open(tmp_path / "log.txt", "w")
        mock_ctr = ActiveContainerProcess.model_construct(
            process=proc, log_file=log_file, log_file_path=tmp_path / "log.txt"
        )

        runner = RunTerminalAgent(config=cfg)
        with patch.object(RunTerminalAgent, "_start", new=AsyncMock(return_value=mock_ctr)):
            with patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock(return_value=None)):
                with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
                    with patch("os.killpg"), patch("os.getpgid", return_value=99999):
                        result = await runner.process_single_datapoint()

        assert result is False
        assert json.loads(cfg.metrics_fpath.read_text())["container_timed_out"] is True

    async def test_timing_files_parsed(self, tmp_path: Path) -> None:
        import time as _time

        cfg = _make_instance_config(tmp_path)
        spinup = _time.time()
        (cfg.persistent_dir / "agent_spinup_timestamp").write_text(str(spinup))
        (cfg.persistent_dir / "eval_start_timestamp").write_text(str(spinup + 30.0))
        mock_ctr = self._mock_container(tmp_path)

        runner = RunTerminalAgent(config=cfg)
        with patch.object(RunTerminalAgent, "_start", new=AsyncMock(return_value=mock_ctr)):
            with patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock(return_value=None)):
                await runner.process_single_datapoint()

        metrics = json.loads(cfg.metrics_fpath.read_text())
        assert metrics["agent_run_time"] == pytest.approx(30.0, abs=1.0)

    async def test_nonzero_exit_logs_output(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        cfg = _make_instance_config(tmp_path)
        mock_ctr = self._mock_container(tmp_path, returncode=1)
        (tmp_path / "log.txt").write_text("something went wrong")

        runner = RunTerminalAgent(config=cfg)
        with patch.object(RunTerminalAgent, "_start", new=AsyncMock(return_value=mock_ctr)):
            with patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock(return_value=None)):
                await runner.process_single_datapoint()

        assert "container exit 1" in capsys.readouterr().out


# ── RunTerminalAgent._start ───────────────────────────────────────────────────────


class TestStart:
    async def test_creates_log_file_and_returns_container(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        runner = RunTerminalAgent(config=cfg)
        ctr = await runner._start("echo hello")
        await ctr.process.wait()
        ctr.log_file.close()
        assert ctr.log_file_path.exists()
        assert (cfg.persistent_dir / "apptainer_logs").is_dir()


# ── RunTerminalAgent._stage_tests ────────────────────────────────────────────────


class TestStageTests:
    async def test_copies_tests_and_signals_ready(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        (cfg.persistent_dir / "staging").mkdir(parents=True, exist_ok=True)
        tests_src = Path(cfg.problem_info["task_dir"]) / "tests"
        tests_src.mkdir()
        (tests_src / "test.sh").write_text("#!/bin/bash")
        (cfg.persistent_dir / "agent_done").touch()

        await RunTerminalAgent(config=cfg)._stage_tests(cfg)

        assert (cfg.persistent_dir / "tests_ready").exists()
        assert (cfg.persistent_dir / "staging" / "tests" / "test.sh").exists()

    async def test_removes_stale_staging_before_copy(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        staging_tests = cfg.persistent_dir / "staging" / "tests"
        staging_tests.mkdir(parents=True)
        (staging_tests / "stale.sh").write_text("old")
        tests_src = Path(cfg.problem_info["task_dir"]) / "tests"
        tests_src.mkdir()
        (tests_src / "fresh.sh").write_text("new")
        (cfg.persistent_dir / "agent_done").touch()

        await RunTerminalAgent(config=cfg)._stage_tests(cfg)

        assert not (staging_tests / "stale.sh").exists()
        assert (staging_tests / "fresh.sh").exists()
