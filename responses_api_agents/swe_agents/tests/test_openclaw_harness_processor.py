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

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from responses_api_agents.swe_agents.app import (
    OpenClawHarnessProcessor,
)


@pytest.fixture
def patched_model_config(monkeypatch):
    """OpenClaw's get_run_command resolves the model-server URL from the global config; patch
    it to a deterministic vllm_model proxy endpoint (host:port = the shim's upstream)."""
    from responses_api_agents.swe_agents import app as swe_app

    server_cfg = {"host": "127.0.0.1", "port": 9000, "api_key": "dummy", "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct"}
    monkeypatch.setattr(swe_app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(swe_app, "get_first_server_config_dict", lambda _gc, _name: server_cfg)
    return server_cfg


def _make_instance(
    tmp_path: Path,
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
    inference_params=None,
    repo_language=None,
):
    """A config-bound SWEBenchWrapperInstanceConfig (model_construct = no validation) — the same
    object the wrapper passes to ``OpenClawHarnessProcessor(config=...).get_run_command()``."""
    from responses_api_agents.swe_agents.app import SWEBenchWrapperInstanceConfig

    # Unique per dataset_name so a single test can build two instances under one tmp_path
    # without the second clobbering the first's chmod-0o444 config files.
    slug = "".join(c if c.isalnum() else "_" for c in dataset_name)
    persistent = tmp_path / f"persistent_{slug}"
    if not persistent.exists():
        persistent.mkdir()
        (persistent / "openclaw_home" / "agents" / "main" / "agent").mkdir(parents=True)
    problem_info = {
        "dataset_name": dataset_name,
        "instance_id": "astropy__astropy-12345",
        "instance_dict": json.dumps(
            {
                "problem_statement": "fix the bug",
                "instance_id": "astropy__astropy-12345",
                "base_commit": "deadbeef123",
            }
        ),
    }
    if repo_language is not None:
        problem_info["repo_language"] = repo_language
    return SWEBenchWrapperInstanceConfig.model_construct(
        harness="openclaw",
        model_server=SimpleNamespace(name="vllm_model"),
        agent_max_turns=100,
        swebench_agent_timeout=2700,
        agent_run_id="00000000-0000-0000-0000-000000000000",
        persistent_dir=str(persistent),
        inference_params=inference_params or {},
        body=MagicMock(model="placeholder"),
        problem_info=problem_info,
    )


@pytest.mark.parametrize(
    "dataset_name, repo_language, expected_dataset_type, expected_repo_language, expected_workspace",
    [
        # SWE-bench default: workspace=/testbed, no repo_language => empty default.
        ("princeton-nlp/SWE-bench_Verified", None, "SWE-bench", "", "/testbed"),
        # Multilingual carries a raw (un-normalized) repo_language through to the env.
        ("SWE-bench/SWE-bench_Multilingual", "C++", "SWE-bench_Multilingual", "C++", "/testbed"),
        # nv-internal datasets resolve to the /app workspace.
        ("nv-internal-1", None, "nv-internal-1", "", "/app"),
    ],
)
def test_env_dataset_type_repo_language_and_workspace(
    tmp_path,
    patched_model_config,
    dataset_name,
    repo_language,
    expected_dataset_type,
    expected_repo_language,
    expected_workspace,
):
    """DATASET_TYPE / REPO_LANGUAGE / OPENCLAW_WORKSPACE are all resolved from the
    dataset name (+ per-instance repo_language). repo_language is raw, not normalized —
    normalization happens in-script."""
    inst = _make_instance(tmp_path, dataset_name=dataset_name, repo_language=repo_language)
    env = OpenClawHarnessProcessor(config=inst).get_run_command().env
    assert env["DATASET_TYPE"] == expected_dataset_type
    assert env["REPO_LANGUAGE"] == expected_repo_language
    assert env["OPENCLAW_WORKSPACE"] == expected_workspace


def test_java_options_set_for_rebench_only(tmp_path, patched_model_config):
    rebench = _make_instance(tmp_path, dataset_name="nebius/SWE-rebench")
    env = OpenClawHarnessProcessor(config=rebench).get_run_command().env
    assert env["_JAVA_OPTIONS"] == "-Djava.net.preferIPv6Addresses=false"

    non_rebench = _make_instance(tmp_path, dataset_name="princeton-nlp/SWE-bench_Verified")
    env2 = OpenClawHarnessProcessor(config=non_rebench).get_run_command().env
    assert "_JAVA_OPTIONS" not in env2


def test_get_run_command_uses_dataset_workspace_path(tmp_path, patched_model_config):
    cmd = OpenClawHarnessProcessor(config=_make_instance(tmp_path)).get_run_command()
    env = cmd.env
    assert env["OPENCLAW_WORKSPACE"] == "/testbed"
    assert env["OPENCLAW_STATE_DIR"] == "/.openclaw"
    assert env["OPENCLAW_TRAJECTORY_DIR"] == "/trajectories_mount/openclaw_trajectory"
    assert env["OPENCLAW_SETUP_DIR"] == "/openclaw_setup"
    assert env["AGENT_MAX_TURNS"] == "100"
    assert env["AGENT_RUN_ID"] == "00000000-0000-0000-0000-000000000000"
    assert env["VLLM_MODEL_BASE_URL"] == "http://127.0.0.1:9000/v1"
    # base_commit is passed so the patch is diffed against it (parity with OpenHands).
    assert env["SWEBENCH_BASE_COMMIT"] == "deadbeef123"


def test_get_run_command_threads_sampling_params(tmp_path, patched_model_config):
    """inference_params reach inference: temperature + max_output_tokens ride openclaw.json
    agents.defaults.params (maxTokens), while top_p is injected by the stream shim via the
    TOP_P env (the openai-responses transport can't carry top_p). temperature=0 must survive."""
    inst = _make_instance(tmp_path, inference_params={"temperature": 0.0, "tokens_to_generate": 4096, "top_p": 0.95})
    cmd = OpenClawHarnessProcessor(config=inst).get_run_command()
    cfg = json.load(open(Path(inst.persistent_dir) / "openclaw_home" / "openclaw.json"))
    assert cfg["agents"]["defaults"]["params"] == {"temperature": 0.0, "maxTokens": 4096}
    assert cmd.env["TOP_P"] == "0.95"


def test_get_run_command_top_p_unset_is_empty_env(tmp_path, patched_model_config):
    """No top_p in inference_params => TOP_P env is empty (shim leaves the request untouched),
    and no params block is emitted when no sampling params are configured."""
    inst = _make_instance(tmp_path, inference_params={})
    cmd = OpenClawHarnessProcessor(config=inst).get_run_command()
    assert cmd.env["TOP_P"] == ""
    cfg = json.load(open(Path(inst.persistent_dir) / "openclaw_home" / "openclaw.json"))
    assert "params" not in cfg["agents"]["defaults"]


@pytest.mark.parametrize(
    "dataset_name, expected_path_prepend",
    [
        # The agent's `python` must resolve to the repo interpreter. openclaw rebuilds the
        # exec PATH from a sanitized base, so the testbed env bin must ride on exec.pathPrepend
        # (after the wrapper dir, which stays first). SWE-bench => conda testbed env appended.
        ("princeton-nlp/SWE-bench_Verified", ["/openclaw_setup/bin", "/opt/miniconda3/envs/testbed/bin"]),
        # nv-internal uses the image's default interpreter => wrapper dir only, no env bin.
        ("nv-internal-1", ["/openclaw_setup/bin"]),
    ],
)
def test_get_run_command_writes_testbed_bin_into_exec_pathprepend(
    tmp_path, patched_model_config, dataset_name, expected_path_prepend
):
    inst = _make_instance(tmp_path, dataset_name=dataset_name)
    OpenClawHarnessProcessor(config=inst).get_run_command()
    cfg = json.load(open(Path(inst.persistent_dir) / "openclaw_home" / "openclaw.json"))
    assert cfg["tools"]["exec"]["pathPrepend"] == expected_path_prepend


def test_get_run_command_chmods_configs_readonly(tmp_path, patched_model_config):
    inst = _make_instance(tmp_path)
    OpenClawHarnessProcessor(config=inst).get_run_command()
    for p in [
        Path(inst.persistent_dir) / "openclaw_home" / "openclaw.json",
        Path(inst.persistent_dir) / "openclaw_home" / "exec-approvals.json",
        Path(inst.persistent_dir) / "openclaw_home" / "agents" / "main" / "agent" / "settings.json",
    ]:
        mode = p.stat().st_mode & 0o777
        assert mode == 0o444, f"{p} should be 0o444, got 0o{mode:o}"


def test_get_run_command_uses_per_run_uuid_for_gateway_token(tmp_path, patched_model_config):
    import re

    inst = _make_instance(tmp_path)
    OpenClawHarnessProcessor(config=inst).get_run_command()
    cfg = json.load(open(Path(inst.persistent_dir) / "openclaw_home" / "openclaw.json"))
    token = cfg["gateway"]["auth"]["token"]
    assert token != "x"
    assert re.match(r"^[0-9a-f-]{36}$", token)


def _make_wrapper(harness: str = "openclaw"):
    from responses_api_agents.swe_agents.app import SWEBenchWrapper, SWEBenchWrapperConfig

    cfg = SWEBenchWrapperConfig(
        host="localhost",
        port=9003,
        name="test_swe_agent",
        entrypoint="responses_api_agents/swe_agents",
        harness=harness,
        container_formatter="/sif/{instance_id}.sif",
        model_server={"type": "responses_api_models", "name": "vllm_model"},
        datasets=[],
    )
    wrapper = MagicMock(spec=SWEBenchWrapper)
    wrapper.config = cfg
    # Bind real methods so they use self=wrapper and dispatch correctly.
    wrapper._build_apptainer_command = SWEBenchWrapper._build_apptainer_command.__get__(wrapper)
    wrapper._build_openclaw_apptainer_command = SWEBenchWrapper._build_openclaw_apptainer_command.__get__(wrapper)
    return wrapper


def test_apptainer_command_has_openclaw_mounts(tmp_path):
    from responses_api_agents.swe_agents.app import ExecuteContainerCommandArgs

    # Build a minimal params namespace — only the fields the OpenClaw branch reads.
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    dataset_file = tmp_path / "data.jsonl"
    dataset_file.write_text("")
    params = SimpleNamespace(
        persistent_dir=persistent,
        instance_dataset_path=dataset_file,
        container="/sif/test.sif",
        apptainer_memory_limit_mb=None,
    )
    command = ExecuteContainerCommandArgs(
        command="/openclaw_setup/run_openclaw.sh",
        expected_file_pattern="foo",
        mode="agent",
        timeout=600,
        env={"OPENCLAW_WORKSPACE": "/testbed", "AGENT_RUN_ID": "test-uuid-1234"},
    )

    wrapper = _make_wrapper(harness="openclaw")
    cmd = wrapper._build_apptainer_command(params, command)

    setup_dir = str(Path(__file__).resolve().parents[1] / "swe_openclaw_setup")
    assert f"--mount type=bind,src={setup_dir},dst=/openclaw_setup,ro" in cmd
    assert f"--mount type=bind,src={setup_dir},dst={setup_dir},ro" in cmd
    assert f"--mount type=bind,src={persistent},dst=/trajectories_mount" in cmd
    assert f"--mount type=bind,src={persistent / 'openclaw_home'},dst=/.openclaw" in cmd
    assert f"--mount type=bind,src={dataset_file},dst=/root/dataset/data.jsonl,ro" in cmd
    assert "--env OPENCLAW_WORKSPACE=/testbed" in cmd
    assert "--env AGENT_RUN_ID=test-uuid-1234" in cmd


def test_apptainer_command_eval_mode_does_not_use_openclaw_branch(tmp_path):
    """Eval mode with harness='openclaw' must NOT hit the OpenClaw branch (falls through to OpenHands)."""
    from responses_api_agents.swe_agents.app import ExecuteContainerCommandArgs

    # The OpenClaw branch only triggers for mode="agent". For mode="eval", the function
    # falls through to the OpenHands path, which reads openhands-specific fields.
    persistent = tmp_path / "persistent2"
    persistent.mkdir()
    dataset_file = tmp_path / "data2.jsonl"
    dataset_file.write_text("")
    params = SimpleNamespace(
        persistent_dir=persistent,
        instance_dataset_path=dataset_file,
        container="/sif/test.sif",
        apptainer_memory_limit_mb=None,
    )
    command = ExecuteContainerCommandArgs(
        command="echo eval",
        expected_file_pattern="foo",
        mode="eval",
        timeout=600,
        env=None,
    )

    wrapper = _make_wrapper(harness="openclaw")
    # The OpenHands fall-through reads params.problem_info, which is absent on this bare config → AttributeError, proving the OpenClaw agent branch was NOT taken for eval mode.
    with pytest.raises((AttributeError, KeyError)):
        wrapper._build_apptainer_command(params, command)


def test_run_openclaw_agent_classifies_max_iteration(tmp_path):
    """A proxy log ending in max_iteration is classified correctly.

    The mask_sample decision lives downstream in _inner_responses (computed
    from the persisted metrics), not in the finalize artifacts."""
    from responses_api_agents.swe_agents.app import RunOpenClawAgent

    persistent = tmp_path / "p"
    persistent.mkdir()

    # Pre-seed an "as-if" rollout's artifacts.
    (persistent / "openclaw_proxy.jsonl").write_text(
        json.dumps(
            {
                "turn": 0,
                "endpoint": "/v1/responses",
                "request": {"input": [{"role": "user", "content": "go"}], "tools": []},
                "response": {
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "wip"}],
                            "prompt_token_ids": [1],
                            "generation_token_ids": [2],
                        }
                    ]
                },
                "upstream_status": 200,
                "error": None,
            }
        )
        + "\n"
        + json.dumps(
            {
                "turn": 1,
                "endpoint": "/v1/responses",
                "request": None,
                "response": None,
                "upstream_status": None,
                "error": "max_iteration",
            }
        )
        + "\n"
    )
    (persistent / "openclaw_trajectory").mkdir()
    (persistent / "openclaw_trajectory" / "rs1.jsonl").write_text("")

    runner = RunOpenClawAgent()
    result = runner._finalize_from_artifacts(
        persistent_dir=str(persistent),
        agent_timed_out=False,
        subprocess_exit_code=1,
        agent_run_id="rs1",
    )

    assert result.agent_error_kind == "max_iteration"
    # Token IDs stripped after reconstruction.
    log_lines = (persistent / "openclaw_proxy.jsonl").read_text().splitlines()
    e0 = json.loads(log_lines[0])
    assert "prompt_token_ids" not in e0["response"]["output"][0]


# ---------------------------------------------------------------------------
# process_single_datapoint happy-path test
# ---------------------------------------------------------------------------


def _make_instance_config(tmp_path: Path):
    """Build a SWEBenchWrapperInstanceConfig via model_construct (no validation)."""
    from responses_api_agents.swe_agents.app import (
        ExecuteContainerCommandArgs,
        SWEBenchWrapperInstanceConfig,
    )

    persistent = tmp_path / "persistent_proc"
    persistent.mkdir(parents=True, exist_ok=True)
    (persistent / "openclaw_trajectory").mkdir()

    # Seed artifacts that process_single_datapoint reads from disk.
    (persistent / "openclaw_proxy.jsonl").write_text(
        json.dumps(
            {
                "turn": 0,
                "endpoint": "/v1/responses",
                "request": {"input": [{"role": "user", "content": "solve it"}], "tools": []},
                "response": {
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "done"}],
                        }
                    ]
                },
                "upstream_status": 200,
                "error": None,
            }
        )
        + "\n"
    )
    (persistent / "openclaw_exit_code.txt").write_text("0\n")
    patch_text = "diff --git a/x.py b/x.py\n+pass\n"
    # run_openclaw.sh writes the agent patch straight to patch.diff (= model_patch_path);
    # there is no separate model.patch handoff file anymore.
    (persistent / "patch.diff").write_text(patch_text)
    (persistent / "openclaw_trajectory" / "run-001.jsonl").write_text("")

    # metrics.json is read by update_metrics — needs to be pre-seeded.
    metrics_fpath = persistent / "metrics.json"
    metrics_fpath.write_text("{}")

    # final_eval_apptainer_spinup_timestamp_fpath is read when eval succeeds.
    ts_fpath = persistent / "final_eval_ts.txt"
    ts_fpath.write_text("0.0")

    eval_report = persistent / "report.json"
    eval_report.write_text(json.dumps({"django__django-99999": {"resolved": True}}))

    agent_command = ExecuteContainerCommandArgs(
        command="/openclaw_setup/run_openclaw.sh",
        expected_file_pattern="*",
        mode="agent",
        timeout=2700,
        env={},
    )
    eval_command = ExecuteContainerCommandArgs(
        command="python run_evaluation.py",
        expected_file_pattern=str(eval_report),
        mode="eval",
        timeout=1800,
        env=None,
    )

    return SWEBenchWrapperInstanceConfig.model_construct(
        # SWEBenchWrapperServerConfig fields
        ng_global_config_dict_str="{}",
        model_server_name="vllm_model",
        openhands_setup_dir=tmp_path / "oh_setup",
        swebench_setup_dir=tmp_path / "swe_setup",
        r2e_gym_setup_dir=tmp_path / "r2e_setup",
        swe_rebench_setup_dir=tmp_path / "rebench_setup",
        swebench_multilingual_setup_dir=tmp_path / "multilingual_setup",
        run_session_id="rs1",
        base_results_dir=tmp_path / "results",
        # SWEBenchWrapperConfig fields
        model_server={"type": "responses_api_models", "name": "vllm_model"},
        harness="openclaw",
        swebench_agent_timeout=2700,
        swebench_tests_timeout=1800,
        skip_eval=False,
        openclaw_keep_token_ids_in_proxy_log=False,
        # SWEBenchWrapperInstanceConfig fields
        metrics_fpath=metrics_fpath,
        problem_info={"instance_id": "django__django-99999", "dataset_name": "swe-bench-ext"},
        instance_id="django__django-99999",
        body=MagicMock(model="test-model"),
        persistent_dir=persistent,
        ray_queue_timestamp=0.0,
        inference_params={},
        agent_run_id="run-001",
        instance_dataset_path=tmp_path / "data.jsonl",
        trajectories_root=persistent / "trajectories",
        prediction_path=persistent / "prediction.jsonl",
        output_for_eval_mounted_path=persistent / "output_for_eval.jsonl",
        output_for_eval_path=persistent / "output_for_eval.jsonl",
        model_patch_path=persistent / "patch.diff",
        container="/sif/django.sif",
        eval_dir_in_openhands="eval_dir",
        openhands_config_file_path=str(persistent / "oh_config.toml"),
        agent_script_path=persistent / "agent_script.sh",
        final_eval_apptainer_spinup_timestamp_fpath=ts_fpath,
        final_eval_apptainer_spinup_timestamp_mounted_fpath=ts_fpath,
        generation_apptainer_spinup_timestamp_fpath=ts_fpath,
        generation_apptainer_spinup_timestamp_mounted_fpath=ts_fpath,
        base_mounted_dir=persistent,
        profiling_dir=persistent / "profiling",
        profiling_mounted_dir=persistent / "profiling",
        # Pre-built apptainer commands (built by _setup_params before runner instantiation).
        agent_apptainer_command_str="apptainer exec /sif/django.sif /openclaw_setup/run_openclaw.sh",
        eval_apptainer_command_str="apptainer exec /sif/django.sif python run_evaluation.py",
        agent_command=agent_command,
        eval_command=eval_command,
        agent_script="#!/bin/bash\n",
        mask_sample=False,
        apptainer_memory_limit_mb=32768,
    )


async def test_process_single_datapoint_happy_path(tmp_path, monkeypatch):
    """Happy path: agent exits 0, patch present, eval returns report path. The
    reconstructed artifacts are returned (Ray object store), NOT written to a
    openclaw_finalize.json sidecar file."""
    from responses_api_agents.swe_agents.app import (
        ActiveContainerCommand,
        RunOpenClawAgent,
    )

    config = _make_instance_config(tmp_path)
    runner = RunOpenClawAgent(config=config)

    # Build a fake process that exits immediately with code 0.
    fake_proc = MagicMock()
    fake_proc.returncode = 0

    async def _fake_wait():
        return 0

    fake_proc.wait = _fake_wait

    eval_report = config.persistent_dir / "report.json"

    # Use io.StringIO so a double-close raises ValueError, catching regressions.
    agent_log_file = io.StringIO()
    eval_log_file = io.StringIO()

    # Use model_construct to bypass Process-type validation for the fake MagicMock proc.
    agent_ac = ActiveContainerCommand.model_construct(
        process=fake_proc,
        log_file=agent_log_file,
        log_file_path=config.persistent_dir / "apptainer_logs" / "agent.log",
    )
    eval_ac = ActiveContainerCommand.model_construct(
        process=fake_proc,
        log_file=eval_log_file,
        log_file_path=config.persistent_dir / "apptainer_logs" / "eval.log",
    )

    start_calls = []

    async def fake_start_container_command(command, apptainer_cmd):
        start_calls.append(command.mode)
        return agent_ac if command.mode == "agent" else eval_ac

    async def fake_finish_container_command(active_command, command):
        return str(eval_report)

    monkeypatch.setattr(runner, "_start_container_command", fake_start_container_command)
    monkeypatch.setattr(runner, "_finish_container_command", fake_finish_container_command)

    result = await runner.process_single_datapoint()

    # Returns (report_file, final_artifacts) — items ride the Ray return value.
    report_file, final = result
    assert report_file == eval_report
    assert final is not None
    assert final.agent_error_kind is None  # clean agent exit (rc=0, no error in proxy log) => no error kind

    # Both agent and eval containers were spawned.
    assert "agent" in start_calls
    assert "eval" in start_calls

    # model_patch_path written with the patch content.
    assert config.model_patch_path.exists()

    # No sidecar file is written — artifacts flow through the return value, not a file.
    assert not (config.persistent_dir / "openclaw_finalize.json").exists()

    # metrics.json was updated (still valid JSON).
    metrics_text = config.metrics_fpath.read_text()
    metrics_data = json.loads(metrics_text)
    assert "patch_exists" in metrics_data or "openhands_run_time" in metrics_data


def test_get_run_command_writes_instance_payload(tmp_path, patched_model_config):
    """The host writes the instance dict to openclaw_instance.json. Rendering
    happens container-side via /openclaw_setup/user_prompt.j2 in run_openclaw.sh."""
    inst = _make_instance(tmp_path)
    OpenClawHarnessProcessor(config=inst).get_run_command()
    payload_path = Path(inst.persistent_dir) / "openclaw_instance.json"
    payload = json.loads(payload_path.read_text())
    assert payload["workspace_path"] == "/testbed"
    assert payload["instance"]["problem_statement"] == "fix the bug"
    # repo_language is carried in the payload (empty when absent from problem_info).
    assert payload["repo_language"] == ""


async def test_process_single_datapoint_no_patch_returns_artifacts(tmp_path, monkeypatch):
    """When the agent produces no patch, the runner still returns the reconstructed
    artifacts (None report, non-None final) — no sidecar file is written."""
    from responses_api_agents.swe_agents.app import (
        ActiveContainerCommand,
        RunOpenClawAgent,
    )

    config = _make_instance_config(tmp_path)
    # Empty the patch (= model_patch_path) so the no-patch branch is taken.
    config.model_patch_path.write_text("")

    runner = RunOpenClawAgent(config=config)

    fake_proc = MagicMock()
    fake_proc.returncode = 1

    async def _fake_wait():
        return 1

    fake_proc.wait = _fake_wait

    agent_log_file = io.StringIO()
    eval_log_file = io.StringIO()

    agent_ac = ActiveContainerCommand.model_construct(
        process=fake_proc,
        log_file=agent_log_file,
        log_file_path=config.persistent_dir / "apptainer_logs" / "agent.log",
    )
    eval_ac = ActiveContainerCommand.model_construct(
        process=fake_proc,
        log_file=eval_log_file,
        log_file_path=config.persistent_dir / "apptainer_logs" / "eval.log",
    )

    async def fake_start_container_command(command, apptainer_cmd):
        return agent_ac if command.mode == "agent" else eval_ac

    async def fake_finish_container_command(active_command, command):
        return str(config.persistent_dir / "report.json")

    monkeypatch.setattr(runner, "_start_container_command", fake_start_container_command)
    monkeypatch.setattr(runner, "_finish_container_command", fake_finish_container_command)

    result = await runner.process_single_datapoint()

    # No report (no patch), but the reconstructed artifacts still come back.
    report_file, final = result
    assert report_file is None
    assert final is not None, "Artifacts must be returned even when no patch is produced"
    assert hasattr(final, "agent_error_kind")

    # No sidecar file is written.
    assert not (config.persistent_dir / "openclaw_finalize.json").exists()


async def test_process_single_datapoint_writes_prediction_file_for_eval(tmp_path, monkeypatch):
    """The eval container blocks on `until [ -f output_for_eval.jsonl ]`. When a patch
    is produced, the OpenClaw runner must write that SWE-bench prediction file (as the
    OpenHands runner does) or eval silently hangs until timeout."""
    from responses_api_agents.swe_agents.app import ActiveContainerCommand, RunOpenClawAgent

    config = _make_instance_config(tmp_path)  # seeds patch.diff (= model_patch_path) with a real diff
    runner = RunOpenClawAgent(config=config)

    fake_proc = MagicMock()
    fake_proc.returncode = 0

    async def _fake_wait():
        return 0

    fake_proc.wait = _fake_wait

    eval_report = config.persistent_dir / "report.json"
    agent_ac = ActiveContainerCommand.model_construct(
        process=fake_proc, log_file=io.StringIO(), log_file_path=config.persistent_dir / "agent.log"
    )
    eval_ac = ActiveContainerCommand.model_construct(
        process=fake_proc, log_file=io.StringIO(), log_file_path=config.persistent_dir / "eval.log"
    )

    async def fake_start(command, apptainer_cmd):
        return agent_ac if command.mode == "agent" else eval_ac

    async def fake_finish(active_command, command):
        return str(eval_report)

    monkeypatch.setattr(runner, "_start_container_command", fake_start)
    monkeypatch.setattr(runner, "_finish_container_command", fake_finish)

    await runner.process_single_datapoint()

    pred = config.output_for_eval_path
    assert pred.exists(), "output_for_eval.jsonl must be written so the eval container does not hang"
    data = json.loads(pred.read_text())
    assert data["instance_id"] == "django__django-99999"
    assert data["model_patch"].startswith("diff --git")
    assert data["model_name_or_path"] == "test-model"


async def test_inner_responses_openclaw_builds_response_from_ray_return(tmp_path, monkeypatch):
    """The OpenClaw runner returns (report_file, _OpenClawFinalArtifacts) via the Ray
    object store — no openclaw_finalize.json file. The artifacts carry items as plain
    dicts; _inner_responses must validate them into pydantic items so the shared
    response builder (output=..., metadata["input"]=json.dumps([i.model_dump() ...]))
    does not crash with AttributeError: 'dict' object has no attribute 'model_dump'."""
    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from responses_api_agents.swe_agents import app as swe_app
    from responses_api_agents.swe_agents.app import SWEBenchWrapper, _OpenClawFinalArtifacts

    config = _make_instance_config(tmp_path)
    config.body = NeMoGymResponseCreateParamsNonStreaming(model="Qwen/Qwen3-Coder-30B-A3B-Instruct", input=[])

    final = _OpenClawFinalArtifacts(
        input_items=[{"role": "user", "content": "Fix the bug"}],
        output_items=[
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Investigating.", "annotations": []}],
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "exec",
                "arguments": '{"command": "ls"}',
                "status": "completed",
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "file.py"},
        ],
        tools=[
            {
                "type": "function",
                "name": "exec",
                "description": "run",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            }
        ],
        agent_error_kind="other",
    )

    # No eval report -> resolved=False, skips report read & postprocess_after_run.
    # The runner returns (report_file, final) — items ride the return value, not a file.
    class _FakeRemote:
        @staticmethod
        def remote(_payload):
            async def _coro():
                return (None, final)

            return _coro()

    monkeypatch.setattr(swe_app, "openclaw_runner_ray_remote", _FakeRemote)
    # Guard: ensure no sidecar file is involved.
    assert not (config.persistent_dir / "openclaw_finalize.json").exists()

    wrapper = _make_wrapper(harness="openclaw")
    wrapper._inner_responses = SWEBenchWrapper._inner_responses.__get__(wrapper)

    resp = await wrapper._inner_responses(config, MagicMock())

    assert resp.model == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert len(resp.output) == 3  # message + function_call + function_call_output
    # The line that previously crashed: input items must be model_dump-able.
    decoded_input = json.loads(resp.metadata["input"])
    assert decoded_input[0]["role"] == "user"


async def test_process_single_datapoint_skip_eval_returns_no_artifacts(tmp_path, monkeypatch):
    """skip_eval=True with a patch present: the eval container is never started and the runner
    returns (None, None) — used to collect agent trajectories without paying the eval cost."""
    from responses_api_agents.swe_agents.app import ActiveContainerCommand, RunOpenClawAgent

    config = _make_instance_config(tmp_path)  # seeds a real patch.diff (= model_patch_path)
    config.skip_eval = True
    runner = RunOpenClawAgent(config=config)

    fake_proc = MagicMock()
    fake_proc.returncode = 0

    async def _fake_wait():
        return 0

    fake_proc.wait = _fake_wait

    agent_ac = ActiveContainerCommand.model_construct(
        process=fake_proc, log_file=io.StringIO(), log_file_path=config.persistent_dir / "agent.log"
    )
    started = []

    async def fake_start(command, apptainer_cmd):
        started.append(command.mode)
        return agent_ac

    async def fake_finish(active_command, command):
        raise AssertionError("eval container must not run when skip_eval=True")

    monkeypatch.setattr(runner, "_start_container_command", fake_start)
    monkeypatch.setattr(runner, "_finish_container_command", fake_finish)

    report_file, final = await runner.process_single_datapoint()
    assert report_file is None and final is None
    assert "eval" not in started  # eval container never spawned


async def test_process_single_datapoint_eval_failure_returns_artifacts(tmp_path, monkeypatch):
    """If the eval container fails (_finish_container_command raises), the runner swallows it,
    persists metrics, and still returns the reconstructed artifacts (None report, non-None
    final) instead of crashing the whole rollout collection."""
    from responses_api_agents.swe_agents.app import ActiveContainerCommand, RunOpenClawAgent

    config = _make_instance_config(tmp_path)  # seeds a real patch.diff
    runner = RunOpenClawAgent(config=config)

    fake_proc = MagicMock()
    fake_proc.returncode = 0

    async def _fake_wait():
        return 0

    fake_proc.wait = _fake_wait

    agent_ac = ActiveContainerCommand.model_construct(
        process=fake_proc, log_file=io.StringIO(), log_file_path=config.persistent_dir / "agent.log"
    )
    eval_ac = ActiveContainerCommand.model_construct(
        process=fake_proc, log_file=io.StringIO(), log_file_path=config.persistent_dir / "eval.log"
    )

    async def fake_start(command, apptainer_cmd):
        return agent_ac if command.mode == "agent" else eval_ac

    async def fake_finish(active_command, command):
        raise RuntimeError("eval container exploded")

    monkeypatch.setattr(runner, "_start_container_command", fake_start)
    monkeypatch.setattr(runner, "_finish_container_command", fake_finish)

    report_file, final = await runner.process_single_datapoint()
    assert report_file is None
    assert final is not None  # artifacts still returned despite the eval failure
    metrics = json.loads(config.metrics_fpath.read_text())  # metrics persisted, no crash
    assert "openhands_run_time" in metrics
