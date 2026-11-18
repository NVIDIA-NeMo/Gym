# Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model aliases share a capture ledger without losing their served identity.

Worker acknowledgements are synthetic; this is not tensor or optimizer acceptance.
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from nemo_gym.client.server_client import ServerClient
from nemo_gym.models.base_responses_api_model import BaseResponsesAPIModelConfig
from tests.unit_tests.models.test_rollout_budget_integration import CapturingModel


def test_original_model_control_reads_alias_capture_and_preserves_model_ref(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_TOKEN_CAPTURE_CONTROL_TOKEN", "test-only")
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {
        "token_id_capture": {
            "enabled": True,
            "external_staging": True,
            "parent_resolution": "source_history",
            "dir": str(tmp_path / "capture"),
            "rebuild_response": False,
        }
    }
    models = {
        name: CapturingModel(
            config=BaseResponsesAPIModelConfig(name=name, host="", port=0, entrypoint="", openai_api_key="test"),
            server_client=client,
        )
        for name in ("policy_model", "deepcad_policy_model")
    }
    apps = {name: model.setup_webserver() for name, model in models.items()}
    for name, app in apps.items():
        with TestClient(app, base_url="http://fixture") as http:
            result = http.post(
                f"/ng-rollout/{name}-rollout/training-token-capture/v1/responses",
                json={"model": "fixture", "input": "go"},
            )
            assert result.status_code == 200, result.text
    # The RL control client targets the original instance, not the CAD alias.
    with TestClient(apps["policy_model"], base_url="http://fixture") as control:
        for name, model in models.items():
            response = control.get(
                f"/training-token-capture/control/rollouts/{name}-rollout/manifest",
                headers={"Authorization": "Bearer test-only"},
            )
            assert response.status_code == 200, response.text
            manifest = response.json()
            assert manifest["capture_key"] == f"{name}-rollout"
            assert manifest["pending_call_ids"] == manifest["failures"] == []
            assert manifest["attempted_call_ids"] == model._calls
            assert len(manifest["records"]) == 1
            record = manifest["records"][0]
            assert record["model_call_id"] == model._calls[0]
            assert record["model_ref"] == {"type": "responses_api_models", "name": name}
            assert record["response_id"] == f"response-{model._calls[0]}"
