# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise adaptive dispatch, production verification and Gym's capture reader together."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from benchmarks.agentic_if_realworld import rollout_driver
from nemo_gym import rollout_collection, server_utils
from nemo_gym.base_responses_api_model import CaptureStore
from nemo_gym.harness_capabilities.checker import EvidenceScope, inspect_record
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.path_utils import materialized_path_for
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.rollout_health import run_health_checks
from nemo_gym.secret_utils import recursively_hide_secrets
from resources_servers.agentic_if_realworld.app import Verifier, VerifyRequest
from resources_servers.agentic_if_realworld.benchmark import core
from resources_servers.agentic_if_realworld.benchmark.synthetic import create, review
from resources_servers.agentic_if_realworld.tests.test_transport import Response
from responses_api_agents.simple_agent.tests.test_app import _make_agent, _mock_response


@pytest.mark.parametrize(
    "complete_stage,fail_last_dispatch,interrupted,fresh_resume",
    [
        (1, False, False, False),
        (3, False, False, False),
        (None, False, False, False),
        (None, True, False, False),
        (1, False, True, False),
        (1, False, False, True),
    ],
)
def test_adaptive_capture_selected_parameters_and_resume(
    tmp_path, monkeypatch, complete_stage, fail_last_dispatch, interrupted, fresh_resume
):
    create(tmp_path / "data", examples=interrupted)
    dataset = core.Dataset(tmp_path / "data", synthetic=True)
    prepared = tmp_path / "prepared.jsonl"
    prepared.write_text("".join(json.dumps(row) + "\n" for row in dataset.prepared()))
    store = CaptureStore(tmp_path / "captures")
    calls, verification, judge_calls = [], [], []
    model_ref = {"type": "responses_api_models", "name": "policy_model"}

    verifier = Verifier(
        dataset=dataset,
        receipt_dir=tmp_path / "receipts",
        endpoint="http://synthetic.invalid",
        model="synthetic-judge",
        key_env="SYNTHETIC",
    )

    async def judge_request(*args, **kwargs):
        payload = json.loads(kwargs["json"]["messages"][0]["content"].split("UNTRUSTED_EVALUATION_PAYLOAD:\n", 1)[1])
        judge_calls.append(1)
        return Response(
            200,
            {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(review(payload))}}]},
        )

    async def post(**kwargs):
        body = kwargs["json"]
        verification.append(copy.deepcopy(body))
        result = await verifier.verify(VerifyRequest.model_validate(body))
        return SimpleNamespace(value=result.model_dump(), release=lambda: None)

    async def response_json(response):
        return response.value

    async def status(response):
        return None

    def preprocess(self, rows, **kwargs):
        return [
            {
                **copy.deepcopy(row),
                "_ng_task_index": index,
                "_ng_rollout_index": 0,
                "agent_ref": {"name": "agentic_if_realworld_agent"},
            }
            for index, row in enumerate(rows)
        ]

    def run_examples(self, rows, **kwargs):
        async def completed():
            row = copy.deepcopy(rows[0])
            if interrupted and row["task_id"] != "synthetic":
                raise asyncio.CancelledError
            params = row["responses_create_params"]
            metadata = params["metadata"]
            receipt_id = metadata["receipt_id"]
            stage = len(calls)
            capture_key = maybe_rollout_id_from_run_body(row)
            calls.append((int(metadata["seed"]), params["max_output_tokens"], capture_key))
            complete = stage == complete_stage
            text, finish = ("OK.", "stop") if complete else ("partial", "length")
            upstream = {
                "model": "synthetic-policy",
                **dataset.request("synthetic"),
                "seed": int(metadata["seed"]),
                "max_tokens": params["max_output_tokens"],
            }
            raw_response = {"choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": text}}]}
            raw_path = verifier.receipt_dir / "model" / (receipt_id + ".http")
            core.write(raw_path, raw_response)
            core.write(
                raw_path.with_suffix(".json"),
                {
                    "status": "generated",
                    "request_body": upstream,
                    "request_sha256": core.judge.sha(upstream),
                    "content_type": "application/json",
                    "response": raw_response,
                    "raw_attempts": [{"path": str(raw_path), "sha256": core.digest(raw_path)}],
                },
            )
            response = {
                "id": receipt_id,
                "created_at": 0,
                "object": "response",
                "model": "synthetic-policy",
                "status": "completed" if complete else "incomplete",
                "metadata": {"receipt_id": receipt_id},
                "tools": [],
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
                "output": [
                    {
                        "id": "m" + receipt_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed" if complete else "incomplete",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
            }
            store.record(
                capture_key,
                {
                    "model_call_id": receipt_id,
                    "model_ref": model_ref,
                    "dialect": "responses",
                    "status_code": 200,
                    "client_session_id": "root",
                    "started_at": stage,
                    "completed_at": stage + 0.5,
                    "request": params,
                    "response": response,
                },
            )
            agent, client = _make_agent(True)
            agent.config.model_server.name = "policy_model"
            client.post = AsyncMock(return_value=_mock_response(response))
            projected, trajectory, _, _ = await agent._create_episode(
                NeMoGymResponseCreateParamsNonStreaming.model_validate(params),
                model_url_path="/v1/responses",
                rollout_id=capture_key,
                task_id="synthetic",
                collect_trajectory=True,
            )
            result = {
                **row,
                "response": projected.model_dump(mode="json"),
                "ng_trajectory": trajectory.model_dump(mode="json"),
            }
            if fail_last_dispatch and stage == len(core.PHASES) - 1:
                return row, {"_ng_failure_class": "agent_run_error", "_ng_failure_http_status": 500}
            return row, result

        class Completions:
            def __iter__(self):
                return iter([completed()])

            async def aclose(self):
                return None

        return Completions()

    monkeypatch.setattr(rollout_collection.RolloutCollectionHelper, "preprocess_examples", preprocess)
    monkeypatch.setattr(rollout_collection.RolloutCollectionHelper, "run_examples", run_examples)
    monkeypatch.setattr(
        rollout_collection.RolloutCollectionHelper, "setup_server_client", lambda self: SimpleNamespace(post=post)
    )
    monkeypatch.setattr(server_utils, "raise_for_status", status)
    monkeypatch.setattr(server_utils, "get_response_json", response_json)
    monkeypatch.setattr(server_utils, "request", judge_request)
    monkeypatch.setenv("SYNTHETIC", "synthetic")
    config = SimpleNamespace(
        num_repeats=1,
        fan_out=None,
        responses_create_params=None,
        input_jsonl_fpath=str(prepared),
        output_jsonl_fpath=str(tmp_path / "out.jsonl"),
        limit=2 if interrupted else None,
        resume_from_cache=fresh_resume,
    )
    settings = {
        "policy_api_key": "synthetic-secret-never-save",
        "observability_enabled": True,
        "model_call_capture_dir": str(store.root),
        "agentic_if_realworld_resources": {
            "resources_servers": {
                "agentic_if_realworld": {
                    "dataset_dir": str(dataset.directory),
                    "synthetic": True,
                    "receipt_dir": str(verifier.receipt_dir),
                    "judge_endpoint": "http://synthetic.invalid",
                    "judge_model": "synthetic-judge",
                    "judge_api_key_env": "SYNTHETIC",
                    "judge_reasoning": "medium",
                }
            }
        },
        "policy_model": {
            "responses_api_models": {
                "agentic_if_text": {
                    "endpoint": "http://synthetic.invalid",
                    "model": "synthetic-policy",
                    "api_key_env": "SYNTHETIC",
                }
            }
        },
    }
    if fresh_resume:
        prewritten = OmegaConf.create(settings)
        recursively_hide_secrets(prewritten)
        core.write(tmp_path / "resolved-config.yaml", OmegaConf.to_container(prewritten))
    if interrupted:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(rollout_driver.run_rollout_collection(config, settings))
    else:
        asyncio.run(rollout_driver.run_rollout_collection(config, settings))
    inventory_path = materialized_path_for(tmp_path / "out.jsonl")
    inventory_bytes = inventory_path.read_bytes()
    inventory = [json.loads(line) for line in inventory_bytes.splitlines()]
    assert len(inventory) == (2 if interrupted else 1)
    assert inventory_path.stat().st_mode & 0o077 == 0
    assert all("_ng_attempt_index" not in row for row in inventory)
    assert [row["_ng_task_index"] for row in inventory] == list(range(len(inventory)))
    health = run_health_checks(tmp_path / "out.jsonl", workers=1).summary["run"]["artifacts"]
    assert health["expected"] == len(inventory) and health["missing"] == int(interrupted)
    assert health["healthy"] is (not interrupted)
    if interrupted:
        assert core.read(tmp_path / "out.summary.json")["expected_items"] == 5
        saved = core.read(tmp_path / "out.jsonl")["response"]
        config.resume_from_cache = True
        config.limit = 1
        with pytest.raises(ValueError, match="inventory"):
            asyncio.run(rollout_driver.run_rollout_collection(config, settings))
        config.limit = 2
        with pytest.raises(ValueError, match="Ambiguous"):
            asyncio.run(rollout_driver.run_rollout_collection(config, settings))
        assert inventory_path.read_bytes() == inventory_bytes
        assert len(calls) == 2 and len(judge_calls) == 1
        assert core.read(tmp_path / "out.jsonl")["response"] == saved
        return
    snapshot_path = tmp_path / "resolved-config.yaml"
    snapshot = OmegaConf.load(snapshot_path)
    assert snapshot.policy_api_key == "****"
    assert settings["policy_api_key"] == "synthetic-secret-never-save"
    assert "synthetic-secret-never-save" not in snapshot_path.read_text()
    assert snapshot.policy_model.responses_api_models.agentic_if_text.model == "synthetic-policy"
    assert snapshot.observability_enabled is True and snapshot_path.stat().st_mode & 0o077 == 0
    snapshot_bytes = snapshot_path.read_bytes()
    last_stage = complete_stage if complete_stage is not None else len(core.PHASES) - 1
    assert [(seed, cap) for seed, cap, _ in calls] == list(core.PHASES[: last_stage + 1])
    assert len({key for _, _, key in calls}) == len(calls)
    record = core.read(tmp_path / "out.jsonl")
    selected = core.PHASES[last_stage]
    assert record["responses_create_params"]["max_output_tokens"] == selected[1]
    assert record["responses_create_params"]["metadata"]["seed"] == str(selected[0])
    if complete_stage is not None:
        assert verification[0]["responses_create_params"]["metadata"] == record["responses_create_params"]["metadata"]
        assert verification[0]["responses_create_params"]["max_output_tokens"] == selected[1]
        assert record["item_verdict"] == "pass" and record["mask_sample"] is False
    else:
        assert not verification and record["mask_sample"] is True and record["scoring_status"] == "not_scored"
    assert record["_ng_attempt_index"] == last_stage
    assert record["ng_model_call_capture"]["rollout_id"] == calls[-1][2]
    assert len(record["ng_model_call_capture"]["calls"]) == 1
    findings = inspect_record(record, scope=EvidenceScope(tools=False))["findings"]
    if fail_last_dispatch:
        assert not record["ng_trajectory"]["turns"]
        assert {"turns_unavailable", "conversation_unavailable"} <= {
            gap["code"] for gap in record["ng_trajectory"]["gaps"]
        }
        assert record["_ng_failure_class"] == "agent_run_error" and record["mask_sample"] is True
    elif complete_stage is not None:
        assert len(record["ng_trajectory"]["turns"]) == 1 and not findings
    else:
        assert not [finding for finding in findings if finding["assertion"].startswith(("steps.", "ownership."))]
    before = {path: core.digest(path) for path in [*store.root.glob("*"), *verifier.receipt_dir.rglob("*.json")]}
    config.resume_from_cache = True
    asyncio.run(rollout_driver.run_rollout_collection(config, settings))
    assert inventory_path.read_bytes() == inventory_bytes
    assert len(calls) == last_stage + 1 and len(judge_calls) == int(complete_stage is not None)
    assert len(verification) == 2 * int(complete_stage is not None)
    assert all(core.digest(path) == digest for path, digest in before.items())
    assert core.read(tmp_path / "out.jsonl").get("response") == record.get("response")
    preserved_inventory = inventory_path.with_suffix(".saved")
    inventory_path.rename(preserved_inventory)
    try:
        with pytest.raises(ValueError, match="Missing materialized rollout inventory"):
            asyncio.run(rollout_driver.run_rollout_collection(config, settings))
        assert not inventory_path.exists()
    finally:
        preserved_inventory.rename(inventory_path)
    assert inventory_path.read_bytes() == inventory_bytes
    settings["agentic_if_realworld_resources"]["resources_servers"]["agentic_if_realworld"]["response_source"] = (
        "gym_response"
    )
    with pytest.raises(ValueError, match="fingerprint"):
        asyncio.run(rollout_driver.run_rollout_collection(config, settings))
    assert snapshot_path.read_bytes() == snapshot_bytes
    assert len(calls) == last_stage + 1 and len(judge_calls) == int(complete_stage is not None)
    assert len(verification) == 2 * int(complete_stage is not None)
    assert all(core.digest(path) == digest for path, digest in before.items())
    if complete_stage is not None:
        settings["agentic_if_realworld_resources"]["resources_servers"]["agentic_if_realworld"].pop("response_source")
        selected = core.read(next((verifier.receipt_dir / "cells").glob("*.json")))["selected"]
        receipt = core.read(Path(selected["receipt_path"]))
        paths = [
            inventory_path,
            Path(receipt["raw_attempts"][-1]["path"]),
            next(verifier.receipt_dir.glob("judges/*/*.http")),
            next(verifier.receipt_dir.glob("judges/*/result.json")),
        ]
        for saved in paths:
            original = saved.read_bytes()
            try:
                saved.write_text("changed")
                with pytest.raises(ValueError):
                    asyncio.run(rollout_driver.run_rollout_collection(config, settings))
                assert saved.read_text() == "changed"
            finally:
                saved.write_bytes(original)
        assert len(calls) == last_stage + 1 and len(judge_calls) == 1


def test_runtime_fingerprint_covers_active_agent_and_core(monkeypatch):
    from nemo_gym.orchestration import jobs

    monkeypatch.setattr(jobs, "installed_gym_commit", lambda: "runtime-revision")
    before = core.runtime_identity()
    assert before["gym_commit"] == "runtime-revision"
    assert before["source_gym_commit"] == core.read(core.ROOT / "source_manifest.json")["gym_commit"]
    paths = ["gym/responses_api_agents/simple_agent/app.py", "gym/nemo_gym/rollout_collection.py"]
    assert all(path in before["sources"] for path in paths)
    digest = core.digest
    monkeypatch.setattr(
        core,
        "digest",
        lambda path: "modified-agent" if path.as_posix().endswith("simple_agent/app.py") else digest(path),
    )
    assert core.runtime_identity() != before
