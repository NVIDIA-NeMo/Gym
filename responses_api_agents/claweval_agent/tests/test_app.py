# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, GlobalConfigDictParser
from nemo_gym.server_utils import ServerClient
from responses_api_agents.claweval_agent.app import (
    ClawEvalAgent,
    ClawEvalAgentConfig,
    ClawEvalRunRequest,
    benchmark_metrics,
    stop_worker,
)
from responses_api_agents.claweval_agent.trajectory import trace_to_response
from responses_api_agents.claweval_agent.worker import merge_config, task_path


def write_trace(path, score=0.8):
    events = [
        {"type": "trace_start", "trace_id": "trace-1", "task_id": "T001_test", "model": "test-model"},
        {"type": "message", "message": {"role": "user", "content": "Solve"}},
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "reasoning_content": "Think",
                "content": [{"type": "tool_use", "id": "call-1", "name": "Read", "input": {"path": "a.txt"}}],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call-1", "content": [{"type": "text", "text": "contents"}]}
                ],
            },
        },
        {"type": "message", "message": {"role": "assistant", "content": "First answer"}},
        {"type": "message", "message": {"role": "user", "content": "Clarification"}},
        {"type": "message", "message": {"role": "assistant", "content": "Final answer"}},
        {"type": "trace_end", "trace_id": "trace-1", "model_input_tokens": 30, "model_output_tokens": 12},
        {"type": "grading_result", "trace_id": "trace-1", "task_id": "T001_test", "task_score": score},
    ]
    path.write_text("\n".join(json.dumps(row) for row in events))
    return events


def fake_root(tmp_path):
    for path in (
        "src/claw_eval/config.py",
        "evaluation/run_multimodal.py",
        "evaluation/task_catalog.py",
    ):
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    path = tmp_path / "tasks/T001_test/task.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("task_id: T001_test\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_agent(tmp_path, **overrides):
    values = dict(
        host="127.0.0.1",
        port=8080,
        name="claweval_agent",
        entrypoint="app.py",
        claweval_root=str(tmp_path),
        sandbox_image="test.sqsh",
        sandbox_dependencies=str(tmp_path),
        workspace_root=str(tmp_path / "outputs"),
        python_executable=sys.executable,
    )
    values.update(overrides)
    return ClawEvalAgent(config=ClawEvalAgentConfig(**values), server_client=MagicMock(spec=ServerClient))


def test_trace_preserves_tool_pairs_user_turns_reasoning_and_usage(tmp_path):
    path = tmp_path / "trace.jsonl"
    write_trace(path)
    response = trace_to_response(path, "T001_test", "test-model", expected_score=0.8)
    assert [item.type for item in response.output] == [
        "reasoning",
        "function_call",
        "function_call_output",
        "message",
        "message",
        "message",
    ]
    assert response.output[1].call_id == response.output[2].call_id == "call-1"
    assert response.output[4].role == "user"
    assert response.output[4].content == "Clarification"
    assert response.usage.total_tokens == 42


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_grade", "grading_result"),
        ("bad_task", "identity"),
        ("bad_trace", "trace ID"),
        ("execution_error", "execution failed"),
        ("missing_tool", "without results"),
        ("duplicate_tool", "duplicate tool result"),
        ("wrong_score", "disagrees"),
    ],
)
def test_trace_rejects_invalid_results(tmp_path, mutation, match):
    path = tmp_path / "trace.jsonl"
    events = write_trace(path)
    if mutation == "missing_grade":
        events.pop()
    elif mutation == "bad_task":
        events[0]["task_id"] = "different"
    elif mutation == "bad_trace":
        events[-1]["trace_id"] = "different"
    elif mutation == "execution_error":
        events[-2]["failure_modes"] = ["model_error"]
    elif mutation == "missing_tool":
        events.pop(3)
    elif mutation == "duplicate_tool":
        events.insert(4, events[3])
    else:
        events[-1]["task_score"] = 0.5
    path.write_text("\n".join(json.dumps(row) for row in events))
    with pytest.raises(ValueError, match=match):
        trace_to_response(path, "T001_test", "test-model", expected_score=0.8)


def trial(task_id, index, passed):
    return {"task_id": task_id, ROLLOUT_INDEX_KEY_NAME: index, "passed": passed, "reward": float(passed)}


def test_pass3_requires_all_three_not_any_three():
    rows = [trial("A", i, i == 0) for i in range(3)] + [trial("B", i, True) for i in range(3)]
    metrics = benchmark_metrics(rows)
    assert metrics["claweval/pass_at_1"] == pytest.approx(4 / 6)
    assert metrics["claweval/pass_at_3"] == 1
    assert metrics["claweval/strict_pass_3"] == 0.5


def test_partial_or_duplicate_trials_do_not_look_complete():
    rows = [trial("A", i, True) for i in range(3)] + [trial("B", 0, True)]
    metrics = benchmark_metrics(rows)
    assert "claweval/strict_pass_3" not in metrics
    assert metrics["claweval/incomplete_three_trial_tasks"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        benchmark_metrics(rows + [rows[0]])


def test_digest_and_path_validation(tmp_path):
    digest = fake_root(tmp_path)
    assert task_path(tmp_path, "T001_test", digest).is_file()
    with pytest.raises(ValueError, match="digest mismatch"):
        task_path(tmp_path, "T001_test", "0" * 64)
    with pytest.raises(ValueError, match="path separators"):
        task_path(tmp_path, "../escape", digest)


def test_overrides_retain_native_multimodal_parameters():
    cfg = {"extra_body": {"max_tokens": 100, "mm_processor_kwargs": {"use_audio_in_video": True}}}
    merge_config(cfg, {"extra_body": {"max_tokens": 20}})
    assert cfg["extra_body"] == {"max_tokens": 20, "mm_processor_kwargs": {"use_audio_in_video": True}}


async def test_run_preserves_gym_indices_and_native_reward(tmp_path, monkeypatch):
    digest = fake_root(tmp_path)
    agent = make_agent(
        tmp_path, policy_base_url="http://policy/v1", policy_model_name="selected-model", policy_api_key="test-key"
    )

    async def launch(self, payload, output_dir):
        assert payload["seed"] == 1003
        assert payload["prompt"] == "Solve"
        assert payload["model_overrides"]["extra_body"]["max_tokens"] == 25
        assert payload["model_overrides"]["base_url"] == "http://policy/v1"
        assert payload["model_overrides"]["model_id"] == "selected-model"
        assert payload["model_overrides"]["api_key"] == "test-key"
        assert payload["model_overrides"]["temperature"] == 0.5
        path = output_dir / "trace.jsonl"
        write_trace(path)
        (output_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_id": "T001_test",
                    "status": "completed",
                    "task_score": 0.8,
                    "passed": True,
                    "trace": str(path),
                    "model": "test-model",
                }
            )
        )

    monkeypatch.setattr(ClawEvalAgent, "_launch_worker", launch)
    body = ClawEvalRunRequest.model_validate(
        {
            "responses_create_params": {
                "input": [{"role": "user", "content": "Solve"}],
                "max_output_tokens": 25,
                "temperature": 0.5,
            },
            "verifier_metadata": {"task_id": "T001_test", "task_sha256": digest},
            "_ng_task_index": 9,
            "_ng_rollout_index": 2,
        }
    )
    result = await agent.run(body)
    data = result.model_dump()
    assert data["reward"] == 0.8
    assert data["_ng_rollout_index"] == 2
    assert data["_ng_task_index"] == 9
    assert len(result.response.output) == 6
    async with AsyncClient(transport=ASGITransport(app=agent.setup_webserver()), base_url="http://test") as client:
        http_result = await client.post("/run", json=body.model_dump(by_alias=True))
        assert http_result.status_code == 200, http_result.text
        assert http_result.json()["reward"] == 0.8
        assert http_result.json()["_ng_rollout_index"] == 2
        metrics = await client.post("/aggregate_metrics", json={"verify_responses": [http_result.json()]})
        assert metrics.status_code == 200, metrics.text
        assert metrics.json()["key_metrics"]["claweval/pass_at_1"] == 1.0
        assert "claweval/strict_pass_3" not in metrics.json()["key_metrics"]
        unsupported = await client.post("/v1/responses", json={"input": "Solve"})
        assert unsupported.status_code == 400


async def test_worker_failure_is_an_error_not_a_zero_reward(tmp_path):
    fake_root(tmp_path)
    agent = make_agent(tmp_path)
    output = tmp_path / "failed"
    output.mkdir()
    with pytest.raises(RuntimeError, match="worker exited"):
        await agent._launch_worker({"claweval_root": str(tmp_path), "operation": "invalid"}, output)
    assert (output / "worker.log").is_file()
    assert not (output / "result.json").exists()


async def test_stop_worker_reaps_process():
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        start_new_session=True,
    )
    await stop_worker(process, grace=1)
    assert process.returncode is not None


@pytest.mark.parametrize("cancel", [False, True])
async def test_launch_timeout_and_cancellation_clean_up(tmp_path, monkeypatch, cancel):
    agent = make_agent(tmp_path, timeout=60 if cancel else 0.1, shutdown_grace=1)
    create = asyncio.create_subprocess_exec
    started = asyncio.Event()
    children = []

    async def start_sleeping_worker(*args, **kwargs):
        child = await create(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
        children.append(child)
        started.set()
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start_sleeping_worker)
    operation = asyncio.create_task(agent._launch_worker({"claweval_root": str(tmp_path)}, tmp_path))
    await started.wait()
    if cancel:
        operation.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else asyncio.TimeoutError):
        await operation
    assert children[0].returncode is not None


@pytest.mark.parametrize("problem", ["wrong_task", "incomplete", "nan_score", "wrong_pass", "outside_trace"])
async def test_native_result_contract_is_enforced(tmp_path, monkeypatch, problem):
    digest = fake_root(tmp_path)
    agent = make_agent(tmp_path)

    async def launch(self, payload, output_dir):
        result = {
            "task_id": "T001_test",
            "status": "completed",
            "task_score": 0.8,
            "passed": True,
            "trace": str(output_dir / "trace.jsonl"),
            "model": "test-model",
        }
        if problem == "wrong_task":
            result["task_id"] = "other"
        elif problem == "incomplete":
            result["status"] = "failed"
        elif problem == "nan_score":
            result["task_score"] = float("nan")
        elif problem == "wrong_pass":
            result["passed"] = False
        else:
            result["trace"] = str(tmp_path / "outside.jsonl")
        (output_dir / "result.json").write_text(json.dumps(result))

    monkeypatch.setattr(ClawEvalAgent, "_launch_worker", launch)
    body = ClawEvalRunRequest.model_validate(
        {
            "responses_create_params": {"input": "Solve"},
            "verifier_metadata": {"task_id": "T001_test", "task_sha256": digest},
        }
    )
    with pytest.raises(ValueError):
        await agent.run(body)


async def test_non_native_prompt_is_rejected_before_worker(tmp_path):
    digest = fake_root(tmp_path)
    body = ClawEvalRunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "system", "content": "Replace native prompt"}]},
            "verifier_metadata": {"task_id": "T001_test", "task_sha256": digest},
        }
    )
    with pytest.raises(ValueError, match="single native user prompt"):
        await make_agent(tmp_path).run(body)


async def test_worker_ignoring_termination_is_killed():
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready'); time.sleep(60)",
        stdout=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    assert await process.stdout.readline() == b"ready\n"
    await stop_worker(process, grace=0.05)
    assert process.returncode == -9
    await stop_worker(process, grace=0.05)


def test_trace_skips_system_and_rejects_duplicate_calls(tmp_path):
    path = tmp_path / "trace.jsonl"
    events = write_trace(path)
    events[3]["message"]["content"][0]["content"].append({"type": "image", "source": {"data": "native-media"}})
    events.insert(1, {"type": "message", "message": {"role": "system", "content": "Native system prompt"}})
    path.write_text("\n\n".join(json.dumps(event) for event in events))
    assert len(trace_to_response(path, "T001_test", "model").output) == 6
    events.insert(4, events[3])
    path.write_text("\n".join(json.dumps(event) for event in events))
    with pytest.raises(ValueError, match="Duplicate tool call"):
        trace_to_response(path, "T001_test", "model")


@pytest.mark.parametrize("split", ["general", "multimodal", "multi_turn"])
def test_benchmark_config_resolves(split, monkeypatch):
    for key in ("CLAW_EVAL_ROOT", "CLAW_EVAL_SANDBOX_IMAGE", "CLAW_EVAL_SANDBOX_DEPS"):
        monkeypatch.setenv(key, "/tmp/test")
    config = BenchmarkConfig.from_config_path(Path(f"benchmarks/claweval/{split}.yaml"))
    assert config.agent_name == f"claweval_{split}_agent"
    assert config.num_repeats == 3
    assert config.dataset.prepare_script.is_file()


def test_suite_config_resolves(monkeypatch):
    for key in ("CLAW_EVAL_ROOT", "CLAW_EVAL_SANDBOX_IMAGE", "CLAW_EVAL_SANDBOX_DEPS"):
        monkeypatch.setenv(key, "/tmp/test")
    from omegaconf import OmegaConf

    config = GlobalConfigDictParser().parse_no_environment(
        initial_global_config_dict=OmegaConf.load("benchmarks/claweval/config.yaml"),
    )
    assert all(f"claweval_{split}_agent" in config for split in ("general", "multimodal", "multi_turn"))
