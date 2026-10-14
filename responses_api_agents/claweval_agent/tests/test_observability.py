# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise native request forwarding, collector joins, and the onboarding evidence gate."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_model import merge_model_call_capture_into_record
from nemo_gym.harness_capabilities.checker import inspect_record
from nemo_gym.rollout_collection import RolloutCollectionHelper, _attach_trajectory_record
from nemo_gym.rollout_health import run_health_checks
from nemo_gym.server_utils import ServerClient
from responses_api_agents.claweval_agent.app import ClawEvalAgent, ClawEvalRunRequest, benchmark_metrics
from responses_api_agents.claweval_agent.tests.test_app import fake_root, make_agent, write_trace
from responses_api_models.openai_model.app import NeMoGymAsyncOpenAI, SimpleModelServer, SimpleModelServerConfig


@pytest.mark.parametrize("explicit_id", [None, "unique-capture"])
async def test_streaming_native_media_is_captured_and_passes_p0(tmp_path, monkeypatch, explicit_id):
    capture_id = explicit_id or "9-2"
    digest = fake_root(tmp_path)
    captures = tmp_path / "captures"
    model = SimpleModelServer(
        config=SimpleModelServerConfig(
            host="127.0.0.1",
            port=18001,
            name="policy_model",
            entrypoint="app.py",
            openai_base_url="http://upstream/v1",
            openai_api_key="unused",
            openai_model="nano",
            forward_streaming_chat_provider_extensions=True,
            upstream_max_num_tries=1,
        ),
        server_client=MagicMock(
            spec=ServerClient,
            global_config_dict={
                "observability_enabled": True,
                "model_call_capture_dir": str(captures),
            },
        ),
    )
    model._client = MagicMock(spec=NeMoGymAsyncOpenAI)
    replies = [
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Think",
            "tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": "Read", "arguments": '{"path":"a.txt"}'}},
            ],
        },
        {"role": "assistant", "content": "Final answer", "reasoning_content": "Done"},
    ]
    model._client.create_chat_completion = AsyncMock(
        side_effect=[
            {
                "id": f"chat-{i}",
                "object": "chat.completion",
                "created": 100,
                "model": "nano",
                "choices": [{"index": 0, "message": reply, "finish_reason": "tool_calls" if i == 0 else "stop"}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 6, "total_tokens": 21},
            }
            for i, reply in enumerate(replies)
        ]
    )
    client = TestClient(model.setup_webserver())
    agent = make_agent(tmp_path, model_server={"type": "responses_api_models", "name": "policy_model"})
    agent.server_client.global_config_dict = {"observability_enabled": True}
    requested = []

    def resolve(self, name, rollout_id):
        assert name == "policy_model" and rollout_id == capture_id
        return f"http://gym/ng-rollout/{rollout_id}/v1"

    async def launch(self, payload, output_dir):
        assert payload["model_overrides"]["base_url"] == f"http://gym/ng-rollout/{capture_id}/v1"
        media = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}},
            {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,Yg=="}},
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,Yw=="}},
        ]
        messages = [{"role": "user", "content": [{"type": "text", "text": "Solve"}, *media]}]
        for i in range(2):
            request = {
                "model": "nano",
                "messages": list(messages),
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": 0.6,
                "top_p": 0.95,
                "seed": 1003,
                "max_tokens": 20480,
                "thinking_token_budget": 17408,
                "chat_template_kwargs": {"enable_thinking": True, "reasoning_budget": 16384},
                "mm_processor_kwargs": {"use_audio_in_video": True},
                "tools": [{"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}}],
            }
            requested.append(request)
            result = client.post(
                f"/ng-rollout/{capture_id}/v1/chat/completions",
                json=request,
                headers={"x-session-id": payload["invocation_id"]},
            )
            assert result.status_code == 200, result.text
            assert '"total_tokens": 21' in result.text
            messages.extend([replies[i], {"role": "tool", "tool_call_id": "call-1", "content": "contents"}])
        path = output_dir / "trace.jsonl"
        events = write_trace(path)
        events.insert(
            3,
            {
                "type": "tool_dispatch",
                "tool_use_id": "call-1",
                "tool_name": "Read",
                "response_status": 200,
                "latency_ms": 12,
                "timestamp": "2026-10-13T10:00:00+00:00",
            },
        )
        path.write_text("\n".join(json.dumps(event) for event in events))
        (output_dir / "sandboxes.json").write_text(
            json.dumps(
                [
                    {
                        "kind": "sandbox",
                        "role": "environment",
                        "provider": "slurm_pyxis",
                        "sandbox_id": "actual-instance",
                        "outcome": "completed",
                        "wall_time_s": 2,
                    }
                ]
            )
        )
        (output_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_id": "T001_test",
                    "status": "completed",
                    "task_score": 0.8,
                    "passed": True,
                    "trace": str(path),
                    "model": "nano",
                }
            )
        )

    monkeypatch.setattr(ClawEvalAgent, "resolve_model_base_url", resolve)
    monkeypatch.setattr(ClawEvalAgent, "_launch_worker", launch)
    row = {
        "responses_create_params": {"input": "Solve"},
        "verifier_metadata": {"task_id": "T001_test", "task_sha256": digest},
        "_ng_task_index": 9,
        "_ng_rollout_index": 2,
    }
    row["_ng_rollout_id"] = explicit_id
    result = (await agent.run(ClawEvalRunRequest.model_validate(row))).model_dump(by_alias=True)
    result["_ng_rollout_id"] = explicit_id
    for forwarded, original in zip(model._client.create_chat_completion.call_args_list, requested, strict=True):
        sent = forwarded.kwargs
        for key in (
            "messages",
            "seed",
            "max_tokens",
            "thinking_token_budget",
            "mm_processor_kwargs",
            "chat_template_kwargs",
        ):
            assert sent[key] == original[key]
    merge_model_call_capture_into_record(result, [captures], include_payloads=True)
    _attach_trajectory_record(row, result)
    evidence = inspect_record(result)
    failures = [check for check in evidence["checks"] if check["tier"] == "P0" and check["status"] == "fail"]
    assert not failures, failures
    assert len(result["ng_trajectory"]["turns"]) == 2
    assert result["ng_trajectory"]["tool_calls"][0]["output"] == "contents"
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(json.dumps(result) + "\n")
    run_health_checks(rollouts, output_dir=tmp_path, workers=1)
    summary = json.loads((tmp_path / "quality_summary.json").read_text())
    assert summary["run"]["verdicts"] == {"healthy": 1, "unhealthy": 0, "unobserved": 0}


async def test_timeout_is_masked_and_excluded_from_scores(tmp_path, monkeypatch):
    digest = fake_root(tmp_path)
    monkeypatch.setattr(ClawEvalAgent, "_launch_worker", AsyncMock(side_effect=TimeoutError))
    result = await make_agent(tmp_path).run(
        ClawEvalRunRequest.model_validate(
            {
                "responses_create_params": {"input": "Solve"},
                "verifier_metadata": {"task_id": "T001_test", "task_sha256": digest},
            }
        )
    )
    assert result.mask_sample and result.failure_kind == "agent_timeout"
    metrics = benchmark_metrics([result.model_dump()])
    assert metrics["claweval/masked_trials"] == 1
    assert "claweval/mean_task_score" not in metrics


@pytest.mark.parametrize("repeats", [1, 3])
def test_repetition_produces_exact_task_seed_pairs(repeats, monkeypatch):
    from nemo_gym.benchmarks import BenchmarkConfig
    from nemo_gym.train_data_utils import TrainDataProcessor

    for key in ("CLAW_EVAL_ROOT", "CLAW_EVAL_SANDBOX_IMAGE", "CLAW_EVAL_SANDBOX_DEPS"):
        monkeypatch.setenv(key, "/tmp/test")
    cfg = BenchmarkConfig.from_config_path(Path("benchmarks/claweval/multimodal.yaml"))
    cfg.dataset.jsonl_fpath = Path("responses_api_agents/claweval_agent/data/example.jsonl")
    rows = [json.loads(line) for line in TrainDataProcessor()._iter_dataset_lines(cfg.dataset)]
    expanded = RolloutCollectionHelper().preprocess_examples(rows, num_repeats=repeats)
    assert len(expanded) == len(rows) * repeats
    assert {row["_ng_rollout_index"] for row in expanded} == set(range(repeats))
    pairs = [(row["verifier_metadata"]["task_id"], 1001 + row["_ng_rollout_index"]) for row in expanded]
    assert len(set(pairs)) == len(pairs)
