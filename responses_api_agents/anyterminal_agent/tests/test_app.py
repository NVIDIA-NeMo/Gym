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

Modeled on anyswe_agent's tests: these exercise pure logic (no real Apptainer/Docker) —
runner-script generation, provider selection, container discovery, deps-key derivation,
setup-script presence, and example-data shape. Heavy side effects in model_post_init (deps +
harness install) are bypassed by calling staticmethods/properties directly rather than
constructing the agent.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest

from nemo_gym import PARENT_DIR
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.providers.apptainer import ApptainerProvider
from nemo_gym.sandbox.providers.apptainer import provider as apptainer_provider
from nemo_gym.sandbox.providers.docker import DockerProvider
from responses_api_agents.anyterminal_agent.app import (
    _RUNNER_TEMPLATE,
    AnyTerminalAgent,
    AnyTerminalAgentConfig,
    AnyTerminalInstanceConfig,
    GymAgentHarnessProcessor,
    RunTerminalAgent,
    _build_provider,
    _format_container,
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
        assert 'object.__setattr__(agent, "resolve_model_base_url"' in rendered

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


class TestFormatContainer:
    def test_default_template(self) -> None:
        assert _format_container(None, "t", "ubuntu:22.04") == "docker://ubuntu:22.04"

    def test_list_formatter_uses_first(self) -> None:
        assert _format_container(["docker://{docker_image}", "unused"], "t", "ubuntu:22.04") == "docker://ubuntu:22.04"

    def test_strips_existing_docker_scheme_from_image(self) -> None:
        assert _format_container("docker://{docker_image}", "t", "docker://ubuntu:22.04") == "docker://ubuntu:22.04"

    def test_task_name_placeholder(self) -> None:
        assert _format_container("docker://reg/{task_name}", "my-task", "ubuntu:22.04") == "docker://reg/my-task"

    def test_dotted_path_treated_as_local(self) -> None:
        assert _format_container("./images/{task_name}.sif", "t", "ubuntu:22.04") == "./images/t.sif"


class TestSetupScriptsExist:
    def test_supported_agents_have_deps_scripts(self) -> None:
        assert (PARENT_DIR / "responses_api_agents" / "hermes_agent" / "scripts" / "hermes_agent_deps.sh").exists()
        assert (Path(__file__).parent.parent / "setup_scripts" / "_portable_python.sh").exists()


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

    def test_nested_provider_api_key_redacted(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(
            tmp_path,
            sandbox_provider={"opensandbox": {"connection": {"api_key": "secret"}}},  # pragma: allowlist secret
        )
        result = json.loads(_safe_config_json(cfg))
        assert result["sandbox_provider"]["opensandbox"]["connection"]["api_key"] == "***"

    def test_agent_command_str_excluded(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, agent_command_str="/agent_deps_mount/bin/python ...")
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


# ── _build_provider ────────────────────────────────────────────────────────────────


class TestBuildProvider:
    @pytest.fixture(autouse=True)
    def _fake_apptainer_binary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Constructing ApptainerProvider hard-errors if the real binary isn't on PATH.
        monkeypatch.setattr(apptainer_provider, "_require_apptainer", lambda: "/usr/bin/apptainer")

    def test_default_is_docker(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        assert isinstance(_build_provider(cfg), DockerProvider)

    def test_apptainer_binds_are_wired(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, sandbox_provider={"apptainer": {}})
        provider = _build_provider(cfg)
        assert isinstance(provider, ApptainerProvider)
        binds = provider._exec_config.default_binds
        assert any(str(cfg.persistent_dir) in b and "/trajectories_mount" in b for b in binds)
        assert any(str(cfg.nemo_gym_root) in b and "/nemo_gym_mount" in b for b in binds)
        assert any(str(cfg.agent_deps_dir) in b and "/agent_deps_mount" in b for b in binds)
        assert any(str(cfg.verifier_dir) in b and "/logs/verifier" in b for b in binds)

    def test_apptainer_writable_and_no_home(self, tmp_path: Path) -> None:
        provider = _build_provider(_make_instance_config(tmp_path, sandbox_provider={"apptainer": {}}))
        start_args = provider._create_config.extra_start_args
        assert "--writable-tmpfs" in start_args
        assert "--no-mount" in start_args and "home" in start_args
        assert "tmp,bind-paths" not in provider._exec_config.extra_exec_args

    def test_docker_provider_selected(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, sandbox_provider={"docker": {}})
        provider = _build_provider(cfg)
        assert isinstance(provider, DockerProvider)

    def test_docker_run_args_include_mounts(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, sandbox_provider={"docker": {}})
        provider = _build_provider(cfg)
        assert any(str(cfg.persistent_dir) in a for a in provider._create_config.extra_run_args)
        assert any(str(cfg.verifier_dir) in a for a in provider._create_config.extra_run_args)

    def test_unknown_provider_returned_as_is(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, sandbox_provider={"opensandbox": {"foo": "bar"}})
        assert _build_provider(cfg) == {"opensandbox": {"foo": "bar"}}


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
        )
        cmd = GymAgentHarnessProcessor(config=cfg).get_run_command()
        assert (persistent_dir / "instruction.txt").read_text() == "the task"
        runner = (persistent_dir / "agent_runner.py").read_text()
        assert "HermesAgent" in runner
        assert "HermesAgentConfig" in runner
        assert cmd == "/agent_deps_mount/bin/python /trajectories_mount/agent_runner.py"


# ── RunTerminalAgent ──────────────────────────────────────────────────────────────


def _sandbox_result(return_code: int = 0, error_type: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(stdout="", stderr="", return_code=return_code, error_type=error_type)


class TestRunAgentEnv:
    def test_model_url_included_when_set(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, model_server_url="http://model:8000")
        env = RunTerminalAgent(config=cfg)._agent_env(cfg)
        assert env["NGTB_MODEL_URL"] == "http://model:8000"

    def test_no_model_url_when_empty(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, model_server_url="")
        env = RunTerminalAgent(config=cfg)._agent_env(cfg)
        assert "NGTB_MODEL_URL" not in env

    def test_sampling_and_kwargs_forwarded(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(
            tmp_path,
            agent_kwargs={"model": "my-model"},
            body=_make_body(),
        )
        env = RunTerminalAgent(config=cfg)._agent_env(cfg)
        assert env["NGTB_MODEL_NAME"] == "my-model"
        assert json.loads(env["NGTB_AGENT_KWARGS"]) == {"model": "my-model"}


class TestProcessSingleDatapoint:
    @pytest.fixture(autouse=True)
    def _no_real_provider(self):
        # AsyncSandbox is mocked below, so the provider it's built with is never used —
        # skip building a real ApptainerProvider (which hard-errors without the binary).
        with patch("responses_api_agents.anyterminal_agent.app._build_provider", return_value=None):
            yield

    async def test_resolved_when_reward_positive(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        (cfg.verifier_dir / "reward.txt").write_text("1.0")

        sandbox = SimpleNamespace(
            start=AsyncMock(),
            exec=AsyncMock(return_value=_sandbox_result()),
            stop=AsyncMock(),
        )
        with patch("responses_api_agents.anyterminal_agent.app.AsyncSandbox", return_value=sandbox):
            with patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock(return_value=None)):
                result = await RunTerminalAgent(config=cfg).process_single_datapoint()

        assert result is True
        assert json.loads(cfg.metrics_fpath.read_text())["resolved"] is True
        sandbox.stop.assert_awaited_once()

    async def test_unresolved_when_no_reward_file(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        sandbox = SimpleNamespace(
            start=AsyncMock(),
            exec=AsyncMock(return_value=_sandbox_result()),
            stop=AsyncMock(),
        )
        with patch("responses_api_agents.anyterminal_agent.app.AsyncSandbox", return_value=sandbox):
            with patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock(return_value=None)):
                result = await RunTerminalAgent(config=cfg).process_single_datapoint()

        assert result is False

    async def test_agent_timeout_sets_flag_and_masks(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        sandbox = SimpleNamespace(
            start=AsyncMock(),
            exec=AsyncMock(return_value=_sandbox_result(return_code=124, error_type="timeout")),
            stop=AsyncMock(),
        )
        with patch("responses_api_agents.anyterminal_agent.app.AsyncSandbox", return_value=sandbox):
            with patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock(return_value=None)):
                await RunTerminalAgent(config=cfg).process_single_datapoint()

        metrics = json.loads(cfg.metrics_fpath.read_text())
        assert metrics["agent_timed_out"] is True
        assert metrics["mask_sample"] is True

    async def test_sandbox_start_failure_is_isolated(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        sandbox = SimpleNamespace(
            start=AsyncMock(side_effect=RuntimeError("boom")),
            exec=AsyncMock(),
            stop=AsyncMock(),
        )
        with patch("responses_api_agents.anyterminal_agent.app.AsyncSandbox", return_value=sandbox):
            result = await RunTerminalAgent(config=cfg).process_single_datapoint()

        assert result is False
        metrics = json.loads(cfg.metrics_fpath.read_text())
        assert metrics["sandbox_failed"] is True
        assert metrics["mask_sample"] is True
        sandbox.stop.assert_awaited_once()

    async def test_remote_provider_stages_and_collects(self, tmp_path: Path) -> None:
        archive = tmp_path / "deps.tar.gz"
        archive.write_bytes(b"deps")
        cfg = _make_instance_config(
            tmp_path,
            sandbox_provider={"opensandbox": {}},
            agent_deps_archive=archive,
        )
        sandbox = SimpleNamespace(
            start=AsyncMock(),
            exec=AsyncMock(return_value=_sandbox_result()),
            upload=AsyncMock(),
            download=AsyncMock(),
            stop=AsyncMock(),
        )
        with patch("responses_api_agents.anyterminal_agent.app.AsyncSandbox", return_value=sandbox):
            with (
                patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock()),
                patch.object(RunTerminalAgent, "_stage_remote_tests", new=AsyncMock()) as stage_tests,
                patch.object(RunTerminalAgent, "_collect_remote_outputs", new=AsyncMock()) as collect,
            ):
                await RunTerminalAgent(config=cfg).process_single_datapoint()

        uploaded = {call.args[1] for call in sandbox.upload.await_args_list}
        assert "/trajectories_mount/instruction.txt" in uploaded
        assert "/trajectories_mount/agent_runner.py" in uploaded
        assert "/tmp/anyterminal-agent-deps.tar.gz" in uploaded
        stage_tests.assert_awaited_once()
        collect.assert_awaited_once()

    async def test_stops_sandbox_even_on_eval_timeout(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        # First exec (agent) succeeds, second exec (eval) times out.
        sandbox = SimpleNamespace(
            start=AsyncMock(),
            exec=AsyncMock(side_effect=[_sandbox_result(), _sandbox_result(return_code=124, error_type="timeout")]),
            stop=AsyncMock(),
        )
        with patch("responses_api_agents.anyterminal_agent.app.AsyncSandbox", return_value=sandbox):
            with patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock(return_value=None)):
                await RunTerminalAgent(config=cfg).process_single_datapoint()

        metrics = json.loads(cfg.metrics_fpath.read_text())
        assert metrics["container_timed_out"] is True
        sandbox.stop.assert_awaited_once()


# ── RunTerminalAgent._stage_tests ────────────────────────────────────────────────


class TestStageTests:
    async def test_copies_tests(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        (cfg.persistent_dir / "staging").mkdir(parents=True, exist_ok=True)
        tests_src = Path(cfg.problem_info["task_dir"]) / "tests"
        tests_src.mkdir()
        (tests_src / "test.sh").write_text("#!/bin/bash")

        await RunTerminalAgent(config=cfg)._stage_tests(cfg)

        assert (cfg.persistent_dir / "staging" / "tests" / "test.sh").exists()

    async def test_removes_stale_staging_before_copy(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        staging_tests = cfg.persistent_dir / "staging" / "tests"
        staging_tests.mkdir(parents=True)
        (staging_tests / "stale.sh").write_text("old")
        tests_src = Path(cfg.problem_info["task_dir"]) / "tests"
        tests_src.mkdir()
        (tests_src / "fresh.sh").write_text("new")

        await RunTerminalAgent(config=cfg)._stage_tests(cfg)

        assert not (staging_tests / "stale.sh").exists()
        assert (staging_tests / "fresh.sh").exists()
