# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-worker gate integration over two independently constructed apps."""

from typing import Any
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter
from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture
from nemo_gym.token_id_capture.staging.records import (
    CaptureAdmission,
    StagedCallRecord,
    StageResult,
)
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


CONTROL_TOKEN = "test-control-token"  # pragma: allowlist secret
CONTROL_HEADERS = {"Authorization": f"Bearer {CONTROL_TOKEN}"}
CONTROL_PREFIX = "/training-token-capture/control"


class _StagingSink:
    def __init__(self) -> None:
        self.records: list[StagedCallRecord] = []

    def stage(self, record: StagedCallRecord) -> StageResult:
        self.records.append(record)
        return StageResult(ok=True, staging_key=f"stage/{record.model_call_id}")


class _FakeWorker:
    def __init__(self, name: str) -> None:
        self.name = name
        self.requests: list[dict[str, Any]] = []
        self.sink = _StagingSink()
        self.capture = RolloutTokenCapture(
            sink=self.sink,
            weight_version_fn=lambda: 4,
            adapter=VLLMCaptureAdapter(),
        )

    async def create_chat_completion(self, **body: Any) -> dict[str, Any]:
        self.requests.append(body)
        admission = CaptureAdmission.model_validate(body["ng_capture"])
        prompt_token_ids = list(admission.required_prefix_token_ids)
        prompt_token_ids.extend([10, 11] if admission.mode == "text" else [12])
        generated_token_ids = [20 + len(self.requests)]
        generated_logprobs = [-0.25]
        content = f"{self.name}-turn-{len(self.requests)}"
        payload = {
            "id": f"chatcmpl-{self.name}-{len(self.requests)}",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "prompt_token_ids": prompt_token_ids,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                    "logprobs": {
                        "content": [
                            {
                                "token": f"token_id:{generated_token_ids[0]}",
                                "logprob": generated_logprobs[0],
                            }
                        ]
                    },
                }
            ],
        }
        call = self.capture.begin_call(admission)
        coords = self.capture.complete_call_from_response(call, payload)
        payload["ng_commit_coords"] = coords.model_dump(mode="json")
        return payload


def _build_server(global_config: dict[str, Any], worker: _FakeWorker) -> VLLMModel:
    config = VLLMModelConfig(
        host="0.0.0.0",
        port=8081,
        base_url="http://worker.invalid/v1",
        api_key="worker-key",  # pragma: allowlist secret
        model="dummy-model",
        entrypoint="",
        name="policy_model",
        num_workers=2,
        return_token_id_information=False,
        uses_reasoning_parser=False,
    )
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = global_config
    server = VLLMModel(config=config, server_client=server_client)
    server._clients = [worker]
    return server


def test_register_call_and_seal_can_land_on_different_workers(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("TEST_GYM_GATE_CONTROL_TOKEN", CONTROL_TOKEN)
    state_path = tmp_path / "shared-gate.json"
    lineage_root = tmp_path / "lineage"
    global_config = {
        "token_id_capture": {
            "enabled": True,
            "all_agents": True,
            "rebuild_response": False,
            "lineage_store": "nemo_gym.token_id_capture.lineage:FileLineageStore",
            "lineage_store_kwargs": {"root": str(lineage_root)},
            "gate": {
                "enabled": True,
                "state_store_path": str(state_path),
                "control_auth_token_env": "TEST_GYM_GATE_CONTROL_TOKEN",
            },
        }
    }
    worker_a = _FakeWorker("a")
    worker_b = _FakeWorker("b")
    server_a = _build_server(global_config, worker_a)
    server_b = _build_server(global_config, worker_b)

    with (
        TestClient(server_a.setup_webserver()) as client_a,
        TestClient(server_b.setup_webserver()) as client_b,
    ):
        registration_response = client_a.put(
            f"{CONTROL_PREFIX}/rollouts/rollout-1",
            headers=CONTROL_HEADERS,
            json={"owner_id": "controller-1", "operation_id": "register-1"},
        )
        assert registration_response.status_code == 200
        capability = registration_response.json()["data_capability"]
        data_headers = {"x-nemo-gym-capture-capability": capability}

        first_response = client_b.post(
            "/ng-rollout/rollout-1/training-token-capture/v1/chat/completions",
            headers=data_headers,
            json={"messages": [{"role": "user", "content": "start"}]},
        )
        assert first_response.status_code == 200, first_response.text
        first_payload = first_response.json()
        first_message = first_payload["choices"][0]["message"]
        assert "ng_commit_coords" not in first_payload
        assert "prompt_token_ids" not in first_payload
        assert "generation_token_ids" not in first_message
        assert first_payload["choices"][0].get("logprobs") is None

        second_response = client_a.post(
            "/ng-rollout/rollout-1/training-token-capture/v1/chat/completions",
            headers=data_headers,
            json={
                "messages": [
                    {"role": "user", "content": "start"},
                    {"role": "assistant", "content": first_message["content"]},
                    {"role": "user", "content": "finish"},
                ]
            },
        )
        assert second_response.status_code == 200, second_response.text
        admission = worker_a.requests[0]["ng_capture"]
        assert admission["mode"] == "token_in"
        assert worker_a.requests[0]["required_prefix_token_ids"] == [10, 11, 21]

        seal_response = client_b.post(
            f"{CONTROL_PREFIX}/rollouts/rollout-1/seal",
            headers=CONTROL_HEADERS,
            json={
                "owner_id": "controller-1",
                "operation_id": "seal-1",
                "reward": 1.0,
                "terminal_logical_request_id": "chatcmpl-a-1",
            },
        )
        assert seal_response.status_code == 200, seal_response.text
        receipt = seal_response.json()
        assert receipt["terminal_model_call_id"] == admission["model_call_id"]
        assert len(receipt["manifest"]) == 2

        metrics = client_a.get(
            f"{CONTROL_PREFIX}/metrics",
            headers=CONTROL_HEADERS,
        ).json()
        assert metrics["registered"] == 1
        assert metrics["admitted"] == 2
        assert metrics["text"] == 1
        assert metrics["token_in"] == 1
        assert metrics["staged"] == 2
        assert metrics["sealed"] == 1

    assert state_path.stat().st_mode & 0o777 == 0o600
