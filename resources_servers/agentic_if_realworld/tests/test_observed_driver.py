# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise resume through the HTTP verifier and the text adapter's capture middleware."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiohttp import (
    ClientConnectorError,
    ClientPayloadError,
    ClientResponseError,
    ServerDisconnectedError,
    SocketTimeoutError,
)
from fastapi.testclient import TestClient

from benchmarks.agentic_if_realworld import rollout_driver
from nemo_gym.client import ServerClient, http_client
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming, maybe_rollout_id_from_run_body
from nemo_gym.rollouts import RolloutCollectionHelper, await_model_call_capture_quiescence, run_health_checks
from nemo_gym.rollouts.harness_capabilities import EvidenceScope, inspect_record
from nemo_gym.utils import materialized_path_for
from resources_servers.agentic_if_realworld.app import ResourcesConfig, ResourcesServer
from resources_servers.agentic_if_realworld.benchmark import core
from resources_servers.agentic_if_realworld.benchmark.synthetic import create, review
from resources_servers.agentic_if_realworld.benchmark.transport import RetryableTransportError
from resources_servers.agentic_if_realworld.tests.test_transport import Response
from responses_api_agents.simple_agent.tests.test_app import _make_agent, _mock_response
from responses_api_models.agentic_if_text.app import TextModel, TextModelConfig
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


@pytest.mark.parametrize(
    "observed,connection_failures,incomplete_first,text,stream_failure,standard",
    [
        (True, 0, False, "OK.", None, False),
        (False, 0, False, "OK.", None, False),
        (True, 1, False, "OK.", None, False),
        (True, 2, False, "OK..", None, False),
        (True, 1, True, "OK.", None, False),
        (True, 0, True, "OK.", SocketTimeoutError, False),
        (True, 0, True, "OK.", ServerDisconnectedError, False),
        (True, 0, True, "OK.", ClientPayloadError, False),
        (True, 0, False, "OK.", None, True),
        (False, 0, False, "OK.", None, True),
        (True, 0, True, "OK.", None, True),
        (True, 0, False, "OK..", None, True),
        (True, 0, False, "context-overflow", None, True),
    ],
)
def test_http_resume_and_connection_recovery(
    tmp_path, monkeypatch, observed, connection_failures, incomplete_first, text, stream_failure, standard
):
    # Other CI workers change source files while installing and testing components.
    # Keep this synthetic run stable; exercise a deliberate runtime change below.
    runtime = core.runtime_identity()
    monkeypatch.setattr(rollout_driver, "runtime_identity", lambda: copy.deepcopy(runtime))
    create(tmp_path / "data")
    dataset = core.Dataset(tmp_path / "data", synthetic=True)
    prepared = tmp_path / "prepared.jsonl"
    prepared.write_text("".join(json.dumps(row) + "\n" for row in dataset.prepared()))
    receipts, captures = tmp_path / "receipts", tmp_path / "captures"
    resource = dict(
        dataset_dir=str(dataset.directory),
        synthetic=True,
        receipt_dir=str(receipts),
        judge_endpoint="http://synthetic.invalid/judge",
        judge_model="synthetic-judge",
        judge_api_key_env="SYNTHETIC",
        judge_reasoning="medium",
    )
    model = dict(
        endpoint="http://synthetic.invalid/policy",
        model="synthetic-policy",
        api_key_env="SYNTHETIC",
        receipt_dir=str(receipts),
    )
    settings = {
        "observability_enabled": observed,
        "model_call_capture_dir": str(captures),
        "agentic_if_realworld_resources": {"resources_servers": {"agentic_if_realworld": resource}},
        "policy_model": {"responses_api_models": {"agentic_if_text": model}},
    }
    if standard:
        resource["response_source"] = "gym_response"
        model = dict(
            base_url="http://synthetic.invalid/v1",
            api_key="synthetic",
            model="synthetic-policy",
            return_token_id_information=True,
            uses_reasoning_parser=True,
            chat_template_kwargs={"reasoning_effort": "max"},
        )
        settings["policy_model"]["responses_api_models"] = {"vllm_model": model}
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = settings
    server_client.assistant_message_header.return_value = None
    resources = ResourcesServer(
        config=ResourcesConfig(
            host="", port=0, name="agentic_if_realworld_resources", entrypoint="app.py", **resource
        ),
        server_client=server_client,
    )
    if not standard:
        adapter = TextModel(
            config=TextModelConfig(host="", port=0, name="policy_model", entrypoint="app.py", **model),
            server_client=server_client,
        )
    if standard:
        adapter = VLLMModel(
            config=VLLMModelConfig(host="", port=0, name="policy_model", entrypoint="app.py", **model),
            server_client=server_client,
        )
    monkeypatch.setenv("SYNTHETIC", "synthetic")
    policy_calls, judge_calls, dispatched, verifications = [], [], [], []

    async def upstream(method, endpoint, **kwargs):
        body = kwargs["json"]
        if endpoint.endswith("/policy"):
            policy_calls.append((body["seed"], body["max_tokens"]))
            if len(policy_calls) <= connection_failures:
                raise ClientConnectorError(None, ConnectionRefusedError())
            partial = incomplete_first and len(policy_calls) == connection_failures + 1
            if partial and stream_failure:

                class InterruptedStream(Response):
                    async def iter_chunked(self, size):
                        yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                        raise stream_failure("Synthetic interrupted HTTP 200 stream")

                response = InterruptedStream(200, None)
                response.headers = {"content-type": "text/event-stream"}
                return response
            return Response(
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "length" if partial else "stop",
                            "message": {"role": "assistant", "content": "partial" if partial else text},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                },
            )
        assert endpoint.endswith("/judge")
        judge_calls.append(1)
        payload = json.loads(body["messages"][0]["content"].split("UNTRUSTED_EVALUATION_PAYLOAD:\n", 1)[1])
        return Response(
            200, {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(review(payload))}}]}
        )

    monkeypatch.setattr(http_client, "request", upstream)
    if standard:

        async def engine(client, **body):
            assert len(body["messages"]) == 1 and body["messages"][0]["role"] == "user"
            prompt = body["messages"][0]["content"]
            if isinstance(prompt, list):
                prompt = "".join(part["text"] for part in prompt)
            assert prompt == dataset.request(next(iter(dataset.inputs)))["messages"][0]["content"]
            assert body["temperature"] == 0.7
            assert body["chat_template_kwargs"] == {"reasoning_effort": "max"}
            if text == "context-overflow":
                policy_calls.append((body["seed"], body["max_tokens"]))
                error = ClientResponseError(None, (), status=400, message="context length")
                error.response_content = b'{"error":{"message":"maximum context length exceeded"}}'
                raise error
            value = (await upstream("POST", "http://synthetic.invalid/policy", json=body)).value
            value.update(id="synthetic-completion", object="chat.completion", created=0, model="synthetic-policy")
            choice = value["choices"][0]
            choice["index"] = 0
            choice["message"].update(
                reasoning="Think first",
                prompt_token_ids=[1, 2],
                generation_token_ids=[3, 4],
                generation_log_probs=[-0.1, -0.2],
            )
            return value

        monkeypatch.setattr(type(adapter._clients[0]), "create_chat_completion", engine)

    def preprocess(self, rows, **kwargs):
        return [
            dict(
                copy.deepcopy(row),
                _ng_task_index=index,
                _ng_rollout_index=0,
                agent_ref={"name": "agentic_if_realworld_agent"},
            )
            for index, row in enumerate(rows)
        ]

    monkeypatch.setattr(RolloutCollectionHelper, "preprocess_examples", preprocess)
    config = SimpleNamespace(
        num_repeats=1,
        fan_out=None,
        responses_create_params=None,
        input_jsonl_fpath=str(prepared),
        output_jsonl_fpath=str(tmp_path / "out.jsonl"),
        limit=None,
        resume_from_cache=False,
        num_samples_in_parallel=1,
    )
    with TestClient(resources.setup_webserver()) as resource_http, TestClient(adapter.setup_webserver()) as model_http:

        async def post(*, server_name, url_path, json, **kwargs):
            body = json.model_dump(mode="json") if hasattr(json, "model_dump") else json
            if url_path == "/verify":
                identity = maybe_rollout_id_from_run_body(body)
                response = resource_http.post(f"/ng-rollout/{identity}/verify", json=body)
                verifications.append(response.json())
            else:
                assert url_path == "/aggregate_metrics"
                response = resource_http.post(url_path, json=body)
            response.raise_for_status()
            return _mock_response(response.json())

        def run_examples(self, rows, **kwargs):
            async def completed():
                row = copy.deepcopy(rows[0])
                identity = maybe_rollout_id_from_run_body(row)
                dispatched.append(row)
                agent, client = _make_agent(observed)
                agent.config.model_server.name = "policy_model"

                async def model_post(**kwargs):
                    body = kwargs["json"]
                    body = body.model_dump(mode="json") if hasattr(body, "model_dump") else body
                    response = model_http.post(f"/ng-rollout/{identity}/v1/responses", json=body)
                    response.raise_for_status()
                    return _mock_response(response.json())

                client.post = model_post
                response, trajectory, _, _ = await agent._create_episode(
                    NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"]),
                    model_url_path="/v1/responses",
                    rollout_id=identity,
                    task_id=row["task_id"],
                    collect_trajectory=observed,
                )
                await await_model_call_capture_quiescence(identity, [captures])
                result = {"response": response.model_dump(mode="json")}
                if trajectory is not None:
                    result["ng_trajectory"] = trajectory.model_dump(mode="json")
                return row, result

            class Completions:
                def __iter__(self):
                    return iter([completed()])

                async def aclose(self):
                    pass

            return Completions()

        monkeypatch.setattr(RolloutCollectionHelper, "run_examples", run_examples)
        monkeypatch.setattr(RolloutCollectionHelper, "setup_server_client", lambda self: SimpleNamespace(post=post))
        preserved = {}
        for _ in range(connection_failures):
            with pytest.raises(RetryableTransportError, match="explicitly resume"):
                asyncio.run(rollout_driver.run_rollout_collection(config, settings))
            assert all(path.read_bytes() == value for path, value in preserved.items())
            preserved = {
                path: path.read_bytes()
                for parent in [receipts / "model", captures]
                for path in parent.rglob("*")
                if path.is_file()
            }
            config.resume_from_cache = True
        asyncio.run(rollout_driver.run_rollout_collection(config, settings))
        assert all(path.read_bytes() == value for path, value in preserved.items())
        assert policy_calls == [core.PHASES[0]] * (connection_failures + 1) + (
            [core.PHASES[1]] if incomplete_first else []
        )
        if text == "context-overflow":
            assert not judge_calls
            journal = core.read(next((receipts / "cells").glob("*.json")))
            assert len(journal["attempts"]) == 1 and journal["attempts"][0]["terminal"]
            assert journal["result"]["mask_sample"] is True
            config.resume_from_cache = True
            asyncio.run(rollout_driver.run_rollout_collection(config, settings))
            assert len(policy_calls) == 1 and not judge_calls
            return
        assert len(judge_calls) == 1
        identities = [maybe_rollout_id_from_run_body(row) for row in dispatched]
        assert len(set(identities)) == len(dispatched)
        receipt_ids = [row["responses_create_params"]["metadata"]["receipt_id"] for row in dispatched]
        assert len(set(receipt_ids)) == len(dispatched)
        record = core.read(tmp_path / "out.jsonl")
        assert record["item_verdict"] == ("pass" if text == "OK." else "fail") and record["mask_sample"] is False
        if standard:
            output = record["response"]["output"]
            assert output[0]["type"] == "reasoning"
            assert output[0]["summary"][0]["text"] == "Think first"
            assert output[-1]["type"] == "message"
            assert output[-1]["prompt_token_ids"] == [1, 2]
            assert output[-1]["generation_token_ids"] == [3, 4]
            assert output[-1]["generation_log_probs"] == [-0.1, -0.2]
            assert len(list((receipts / "model").glob("*.response.json"))) == len(dispatched)
        inventory = materialized_path_for(tmp_path / "out.jsonl")
        assert len(inventory.read_text().splitlines()) == 1
        health = run_health_checks(tmp_path / "out.jsonl", workers=1).summary["run"]["artifacts"]
        assert health["expected"] == 1 and health["missing"] == 0 and health["healthy"]
        if observed:
            assert record["ng_model_call_capture"]["rollout_id"] == identities[-1]
            assert len(record["ng_model_call_capture"]["calls"]) == 1
            assert not inspect_record(record, scope=EvidenceScope(tools=False))["findings"]
        saved = {
            path: path.read_bytes() for parent in [receipts, captures] for path in parent.rglob("*") if path.is_file()
        }
        config.resume_from_cache = True
        if standard:
            model["port"] = 12345
        asyncio.run(rollout_driver.run_rollout_collection(config, settings))
        assert all(path.read_bytes() == value for path, value in saved.items())
        assert core.read(tmp_path / "out.jsonl") == record
        assert len(policy_calls) == len(dispatched) == connection_failures + 1 + int(incomplete_first)
        assert len(judge_calls) == 1 and len(verifications) == 2
        original_commit = runtime["gym_commit"]
        runtime["gym_commit"] = "changed-runtime"
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            asyncio.run(rollout_driver.run_rollout_collection(config, settings))
        assert all(path.read_bytes() == value for path, value in saved.items())
        assert core.read(tmp_path / "out.jsonl") == record
        assert len(policy_calls) == len(dispatched) == connection_failures + 1 + int(incomplete_first)
        assert len(judge_calls) == 1 and len(verifications) == 2
        runtime["gym_commit"] = original_commit
        if observed:
            assert verifications[0]["ng_verifier_invocations"] != verifications[1]["ng_verifier_invocations"]
            assert record["ng_trajectory"]["verifier_invocations"] == verifications[0]["ng_verifier_invocations"]
        if stream_failure:
            failed = core.read(receipts / "model" / (receipt_ids[0] + ".json"))
            assert failed["reason"] == stream_failure.__name__
            assert failed["dispatch_status"] == "response_received"
            assert failed["response_body_complete"] is False
            assert b'"partial"' in Path(failed["raw_attempts"][0]["path"]).read_bytes()
        if observed and connection_failures == 0 and not incomplete_first:
            journal_path = next((receipts / "cells").glob("*.json"))
            journal = core.read(journal_path)
            assert journal["result"]["ng_verifier_invocations"] == verifications[0]["ng_verifier_invocations"]
            for field, value in [
                ("reward", 1.0 - record["reward"]),
                ("mask_sample", True),
                ("judge_model", "changed"),
                ("judge_request_sha256", "changed"),
                ("response", {}),
                ("responses_create_params", {}),
                ("verifier_metadata", {}),
            ]:
                damaged = copy.deepcopy(journal)
                damaged["result"][field] = value
                damaged["result_sha256"] = core.judge.sha(damaged["result"])
                core.write(journal_path, damaged)
                with pytest.raises(ValueError, match="Saved result differs"):
                    asyncio.run(rollout_driver.run_rollout_collection(config, settings))
                journal_path.write_bytes(saved[journal_path])
                assert core.read(tmp_path / "out.jsonl") == record
                assert len(policy_calls) == len(judge_calls) == 1

        if standard:
            response_path = next((receipts / "model").glob("*.response.json"))
            original = response_path.read_bytes()
            damaged = core.read(response_path)
            damaged["output"][-1]["generation_token_ids"] = [99, 100]
            core.write(response_path, damaged)
            with pytest.raises(ValueError, match="Raw transport binding changed"):
                asyncio.run(rollout_driver.run_rollout_collection(config, settings))
            response_path.write_bytes(original)
            assert core.read(tmp_path / "out.jsonl") == record
