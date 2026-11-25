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
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import yaml
from fastapi.testclient import TestClient

from nemo_gym.global_config import OBSERVABILITY_ENABLED_KEY_NAME
from nemo_gym.server_utils import ServerClient
from responses_api_agents.prime_general_agent.app import PrimeGeneralAgent
from responses_api_agents.verifiers_agent.app import VerifiersAgentConfig


ENV_DIR = Path(__file__).resolve().parents[3] / "environments" / "prime_general_agent"
STOP = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
}


def _agent() -> PrimeGeneralAgent:
    shipped = yaml.safe_load((ENV_DIR / "config.yaml").read_text())["prime_general_agent"]["responses_api_agents"]
    config = VerifiersAgentConfig(**{**shipped["prime_general_agent"], "host": "0.0.0.0", "port": 8080, "name": ""})
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {OBSERVABILITY_ENABLED_KEY_NAME: False}
    return PrimeGeneralAgent(config=config, server_client=server_client)


def test_prepare_writes_runnable_rows(tmp_path: Path, monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("prime_general_agent_prepare", ENV_DIR / "prepare.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    monkeypatch.setattr(sys, "argv", ["prepare.py", "--limit", "2", "--output", str(tmp_path / "rows.jsonl")])
    prepare.main()

    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert [m["role"] for m in row["responses_create_params"]["input"]] == ["user"]
        assert isinstance(row["responses_create_params"]["input"][0]["content"], str)
        assert row["info"]["task_dir"] == row["info"]["task_name"]


def test_example_rows_run_against_the_installed_tasks() -> None:
    agent = _agent()
    agent._get_env(agent.config.vf_env_id)  # load on the main thread, as uvicorn does
    rows = [json.loads(line) for line in (ENV_DIR / "data" / "example.jsonl").read_text().splitlines()]
    assert rows
    with patch.object(type(agent), "resolve_model_base_url", return_value="http://policy.test/v1"):
        agent._get_client().client._client._transport = httpx.MockTransport(lambda r: httpx.Response(200, json=STOP))
        client = TestClient(agent.setup_webserver())
        for row in rows:
            response = client.post("/run", json=row)
            assert response.status_code == 200, response.text
            assert response.json()["reward"] == 0.0
