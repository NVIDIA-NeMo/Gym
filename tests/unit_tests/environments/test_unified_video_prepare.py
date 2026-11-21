# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import errno
import hashlib
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from environment_servers.harbor.verifier import parse_reward_file
from environments.unified_video import prepare as module
from environments.unified_video.prepare import CLAW, client_environment, stage_claw
from responses_api_agents.claweval_agent.core import inventory
from responses_api_agents.claweval_agent.tests.test_core import task_package


@pytest.mark.parametrize("hardlinks_supported", [True, False])
def test_claw_native_task_preserves_private_grader_and_grants_media(tmp_path, monkeypatch, hardlinks_supported):
    if not hardlinks_supported:

        def unsupported_link(*args, **kwargs):
            raise OSError(errno.EXDEV, "Cross-device link")

        monkeypatch.setattr(module.os, "link", unsupported_link)
    root, source = task_package(tmp_path)
    task = inventory({"original_100": root})["original_100::G_video_qa_deadbeef"]
    metadata = {key: task[key] for key in ("task_id", "task_variant", "family", "task_package_sha256")}
    metadata.update(source_id="original_100__fixture", prompt_sha256=hashlib.sha256(b"fresh prompt").hexdigest())
    row = {
        "responses_create_params": {"input": [{"role": "user", "content": "fresh prompt"}]},
        "verifier_metadata": metadata,
    }
    target = tmp_path / "staged"
    stage_claw(row, task, source=source, destination=target, environment_image=None)
    config = tomllib.loads((target / "task.toml").read_text())
    assert config["environment"]["mcp_servers"] == [module.CLAW_MEDIA_SERVER]
    assert "workdir" not in config["environment"]
    assert config["environment"]["env"]["CLAW_MEDIA_MAX_OUTPUT_BYTES"] == str(1024**3)
    assert config["steps"][0]["name"] == "solve"
    setup = (target / "steps/solve/workdir/setup.sh").read_text()
    assert "CLAW_MEDIA_MAX_OUTPUT_BYTES=1073741824" in setup
    assert "start_media_http.py" in setup
    assert config["steps"][0]["verifier"]["env"]["CLAW_EVAL_JUDGE_API_KEY"] == "${CLAW_EVAL_JUDGE_API_KEY}"
    assert (target / "steps/solve/tests/grader.py").read_bytes() == (root / task["task_id"] / "grader.py").read_bytes()
    assert not (target / "steps/solve/workdir/grader.py").exists()
    assert (target / "steps/solve/workdir/.claw_tools/mcp_server.py").is_file()
    assert (target / "steps/solve/instruction.md").read_text() == "fresh prompt"
    assert json.loads((target / "task_data.json").read_text()) == {"video_source": CLAW, "verifier_metadata": metadata}
    assert "project_reward" not in (target / "steps/solve/tests/test.sh").read_text()
    grader = target / "steps/solve/tests/grader.py"
    upstream_grader = root / task["task_id"] / "grader.py"
    original_bytes = upstream_grader.read_bytes()
    assert not grader.samefile(upstream_grader)
    grader.write_text("changed staged grader\n")
    assert upstream_grader.read_bytes() == original_bytes


def test_native_reward_reader_preserves_partial_reward_and_complete_diagnostics(tmp_path):
    record = {"reward": 0.375, "completion": 0.5, "safety": 1.0, "task_id": "private-task"}
    raw = json.dumps(record, indent=2).encode()
    (tmp_path / "reward.json").write_bytes(raw)
    rewards, problem = parse_reward_file(tmp_path)
    assert (tmp_path / "reward.json").read_bytes() == raw
    assert problem is None and rewards == {"reward": 0.375, "completion": 0.5, "safety": 1.0}


@pytest.mark.parametrize("reward", [None, "0.5", float("nan"), float("inf")])
def test_native_reward_reader_does_not_turn_unavailable_into_zero(tmp_path, reward):
    raw = json.dumps({"reward": reward}).encode()
    (tmp_path / "reward.json").write_bytes(raw)
    rewards, problem = parse_reward_file(tmp_path)
    assert rewards is None and problem
    assert (tmp_path / "reward.json").read_bytes() == raw


def test_prepare_client_selection_restores_host_environment(monkeypatch):
    import os

    monkeypatch.setenv("DOCKER_HOST", "tcp://original")
    monkeypatch.setenv("DOCKER_CONTEXT", "unrelated-context")
    monkeypatch.delenv("DOCKER_CERT_PATH", raising=False)
    with pytest.raises(RuntimeError), client_environment({"DOCKER_HOST": "tcp://claw", "DOCKER_CERT_PATH": "/claw"}):
        assert os.environ["DOCKER_HOST"] == "tcp://claw"
        assert "DOCKER_CONTEXT" not in os.environ
        raise RuntimeError("build failure")
    assert os.environ["DOCKER_HOST"] == "tcp://original"
    assert os.environ["DOCKER_CONTEXT"] == "unrelated-context"
    assert "DOCKER_CERT_PATH" not in os.environ


def test_stage_once_and_typed_harness_rows_preserve_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Load the real Harbor task/config models only for this preparation integration case.
    from environment_servers.harbor.tests.conftest import make_client
    from environment_servers.unified_video_harbor.app import (
        UnifiedVideoEpisodeRequest,
        UnifiedVideoHarborConfig,
        UnifiedVideoHarborServer,
    )
    from nemo_gym.config.config_types import DatasetConfig
    from nemo_gym.data.tasks.harbor import prepare as native_prepare
    from nemo_gym.data.tasks.harbor.materialize import materialize_task
    from nemo_gym.data.tasks.harbor.task import discover_tasks, load_task
    from nemo_gym.rollouts import rollout_collection as collector
    from tests.unit_tests.data.tasks.harbor.test_harbor_tasks import write_task

    claw_root, claw_source = task_package(tmp_path)
    claw_task = inventory({"original_100": claw_root})["original_100::G_video_qa_deadbeef"]
    claw_metadata = {key: claw_task[key] for key in ("task_id", "task_variant", "family", "task_package_sha256")}
    claw_metadata.update(source_id="original_100__fixture", prompt_sha256=hashlib.sha256(b"fresh prompt").hexdigest())
    stock_task = load_task(write_task(tmp_path / "stock-avb" / "assembly"))
    stock_row = materialize_task(stock_task, "avb-opencode").model_dump(mode="json", exclude_unset=True)
    stock_row["task_source"] = "agentic_vbench_train"
    rows = [
        stock_row,
        {
            "task_source": CLAW,
            "responses_create_params": {
                "input": [{"role": "user", "content": "fresh prompt"}],
                "temperature": 0.75,
                "top_p": 0.9,
                "max_output_tokens": 128,
            },
            "verifier_metadata": claw_metadata,
        },
        {
            "task_source": "videomme_benchmark_simple_agent",
            "responses_create_params": {"input": "video question"},
            "answer": "B",
        },
    ]
    training = tmp_path / "input.jsonl"
    training.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def no_backend(*args: object, **kwargs: object) -> None:
        raise AssertionError("stage-only must never access a Docker backend")

    monkeypatch.setattr(module, "backend_environment", no_backend)
    options = dict(
        training_data=training,
        output=tmp_path / "prepared",
        claw_roots={"original_100": str(claw_root)},
        claw_source=claw_source,
        environment_image="sha256:" + "a" * 64,
    )
    staged = module.prepare(**options, stage_only=True)
    assert staged.name == "staging.json"
    assert not (staged.parent / "train.jsonl").exists()
    assert not (staged.parent / "agentic_vbench_agent").exists()
    assert module.prepare(**options, stage_only=True, reuse_staged=True) == staged
    runtime = staged.parent / CLAW / claw_task["task_id"] / "steps/solve/workdir/.claw_tools/mcp_server.py"
    original_runtime = runtime.read_bytes()
    runtime.write_bytes(original_runtime + b"\n# stale cached runtime\n")
    with pytest.raises(ValueError, match="Staged Claw media runtime or byte allowance changed"):
        module.prepare(**options, stage_only=True, reuse_staged=True)
    runtime.write_bytes(original_runtime)
    task_config = runtime.parents[4] / "task.toml"
    original_config = task_config.read_text()
    task_config.write_text(original_config.replace(str(1024**3), str(256 * 1024**2)))
    with pytest.raises(ValueError, match="Staged Claw media runtime or byte allowance changed"):
        module.prepare(**options, stage_only=True, reuse_staged=True)
    task_config.write_text(original_config)
    task_config.write_text(original_config.replace("streamable-http", "sse"))
    with pytest.raises(ValueError, match="Staged Claw media launch changed"):
        module.prepare(**options, stage_only=True, reuse_staged=True)
    task_config.write_text(original_config)
    monkeypatch.setattr(module, "backend_environment", lambda *args: {"DOCKER_HOST": "tcp://fixture"})

    prepared_calls = []

    def prepared_tasks(target: str, **kwargs: object) -> SimpleNamespace:
        folder = module.Path(target)
        prepared_calls.append(folder)
        return SimpleNamespace(folder=folder, tasks=discover_tasks(folder), skipped={})

    monkeypatch.setattr(native_prepare, "prepare_target", prepared_tasks)
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\nexit 99\n")
    docker.chmod(0o755)
    output = module.prepare(
        **options,
        reuse_staged=True,
        claw_backend=tmp_path / "claw-backend",
        docker_binary=str(docker),
    )
    converted = [json.loads(line) for line in output.read_text().splitlines()]
    assert converted[0] == rows[0]
    assert converted[-1] == rows[2]
    assert len(converted) == 5
    assert prepared_calls == [staged.parent / CLAW]
    variants, original = converted[1:4], rows[1]
    assert [row["task_id"]["taskset"] for row in variants] == list(module.CLAW_HARNESS_TASKSETS.values())
    for derived, harness in zip(variants, module.CLAW_HARNESS_TASKSETS, strict=True):
        assert derived["responses_create_params"] == original["responses_create_params"]
        assert derived["verifier_metadata"] == original["verifier_metadata"]
        assert derived["task_source"] == original["task_source"]
        assert "_unified_video_task" not in derived
        assert derived["task_input"]["harness"] == harness
        assert derived["task_id"]["task_id"] == claw_task["task_id"]
        assert "harness" not in derived["task_input"]["task_data"]
        assert "responses_create_params" not in derived["task_input"]
        assert len(derived["task_input"]["turns"]) == 1
        assert {
            key: derived["task_input"]["turns"][0]["responses_create_params"][key]
            for key in ("temperature", "top_p", "max_output_tokens")
        } == {
            "temperature": 0.75,
            "top_p": 0.9,
            "max_output_tokens": 128,
        }
    assert all(row["task_input"]["task_data"] == variants[0]["task_input"]["task_data"] for row in variants)
    config = yaml.safe_load((output.parent / "gym_config.yaml").read_text())
    assert config["environment_server_routes"] == {
        taskset: "claweval_environment_server" for taskset in module.CLAW_HARNESS_TASKSETS.values()
    }
    assert "agentic_vbench_agent" not in config and "agentic_vbench_environment_server" not in config
    registration = config[CLAW]["responses_api_agents"]["opencode_agent"]["datasets"][0]
    DatasetConfig.model_validate(registration)
    assert registration["license"] == "NVIDIA Internal Use Only, Do Not Distribute"
    assert registration["type"] == "train" and registration["jsonl_fpath"] == str(output)
    assert module.Path(config["cache_dir"]).is_relative_to(output.parent)
    assert module.Path(config["results_dir"]).is_relative_to(output.parent)
    values = config["claweval_environment_server"]["environment_servers"]["unified_video_harbor"]
    parsed = UnifiedVideoHarborConfig.model_validate(values | {"name": "claw", "host": "127.0.0.1", "port": 8000})
    assert set(parsed.harness_agents) == {"opencode", "codex", "hermes"}
    assert parsed.task_mcp_grants == "required"
    assert set(parsed.tasksets) == set(module.CLAW_HARNESS_TASKSETS.values())
    assert all(mapping.folder == staged.parent / CLAW for mapping in parsed.tasksets.values())
    server = UnifiedVideoHarborServer(config=parsed, server_client=make_client())
    routing = SimpleNamespace(
        environment_server_routes=config["environment_server_routes"], environment_routing_mode="agent"
    )
    try:
        for index, (row, harness) in enumerate(zip(variants, module.CLAW_HARNESS_TASKSETS, strict=True)):
            numbered = row | {collector.TASK_INDEX_KEY_NAME: index, collector.ROLLOUT_INDEX_KEY_NAME: 0}
            assert collector._environment_server_for_config_row(numbered, routing) == "claweval_environment_server"
            body = collector._episode_request_body(numbered)
            assert set(body["task"]) == {"task_id", "task_input"}
            request = UnifiedVideoEpisodeRequest.model_validate(body)
            restored = UnifiedVideoEpisodeRequest.model_validate(
                {"episode_id": body["episode_id"], "task": request.task.model_dump(mode="json")}
            )
            assert restored.harness == harness
            assert server.agent_server_for_episode(restored) == parsed.harness_agents[harness]
            route = server._route(restored)
            resolved = asyncio.run(route._resolve_task(restored.task.task_id, restored.task.task_input.task_data))
            route._check_task_data(restored.task.task_id.taskset, resolved, restored.task.task_input.task_data)
            assert resolved.task_id == claw_task["task_id"]
            task_data = row["task_input"]["task_data"]
            assert resolved.task_data == {key: value for key, value in task_data.items() if key != "_ng_digest"}
            assert resolved.digest == task_data["_ng_digest"]
    finally:
        asyncio.run(server.shutdown())
    assert config[CLAW]["responses_api_agents"]["opencode_agent"]["opencode_config"]["compaction"] == {
        "auto": False,
        "prune": False,
    }
    training.write_text(training.read_text() + json.dumps(rows[2]) + "\n")
    with pytest.raises(ValueError, match="changed"):
        module.prepare(**options, stage_only=True, reuse_staged=True)


def test_generated_routes_use_actual_native_agent_config_models(tmp_path, monkeypatch):
    from responses_api_agents.codex_agent.app import CodexAgentConfig
    from responses_api_agents.opencode_agent.app import OpenCodeAgentConfig

    monkeypatch.setattr(module, "backend_environment", lambda _: {})
    config = module.component_config(
        module.CLAW,
        prepared=SimpleNamespace(tasks=[], folder=tmp_path),
        rows=tmp_path / "rows.jsonl",
        context_tokens=262144,
        output_tokens=8192,
        max_steps=32,
    )
    for name, implementation, kind in (
        (module.CLAW, "opencode_agent", OpenCodeAgentConfig),
        (module.CLAW + "_codex", "codex_agent", CodexAgentConfig),
    ):
        values = config[name]["responses_api_agents"][implementation]
        result = kind.model_validate(values | {"name": name, "host": "127.0.0.1", "port": 1})
        assert result.model_server.name == "policy_model"


def test_task_metadata_preserves_existing_fields_and_refuses_conflicts(tmp_path):
    path = tmp_path / "task_data.json"
    path.write_text('{"native_field": {"unchanged": true}}')
    module._task_metadata(tmp_path, {"task_id": "x"}, source=module.CLAW)
    result = json.loads(path.read_text())
    assert result["native_field"] == {"unchanged": True}
    before = path.read_bytes()
    with pytest.raises(ValueError, match="conflicts"):
        module._task_metadata(tmp_path, {"task_id": "different"}, source=module.CLAW)
    assert path.read_bytes() == before


def test_flat_avb_is_refused_before_creating_a_custom_task_tree(tmp_path: Path) -> None:
    training = tmp_path / "input.jsonl"
    training.write_text(json.dumps({"task_source": "agentic_vbench_agent"}) + "\n")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="stock Harbor preparation"):
        module.prepare(training_data=training, output=output, claw_roots={}, claw_source=tmp_path)
    assert not output.exists()


def test_backend_settings_keep_tls_local_and_wait_only_for_the_selected_file(tmp_path: Path) -> None:
    with pytest.raises(TimeoutError, match="No Docker backend"):
        module.backend_environment(tmp_path, wait_seconds=0)
    (tmp_path / "client.env").write_text(
        '# Fixture settings; no connection is opened.\nexport DOCKER_HOST="tcp://fixture:2376"\n'
        'DOCKER_CERT_PATH="/backend-host-only"\nDOCKER_BUILDKIT=1\n'
    )
    assert module.backend_environment(tmp_path, wait_seconds=0) == {
        "DOCKER_HOST": "tcp://fixture:2376",
        "DOCKER_TLS_VERIFY": "1",
        "DOCKER_CERT_PATH": str(tmp_path),
        "DOCKER_BUILDKIT": "1",
    }
