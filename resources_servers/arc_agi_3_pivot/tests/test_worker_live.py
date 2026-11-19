# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""The example pivots through the server with the real harness (Linux; skipped unless the harness is installed):

    ARC3_PIVOT_HARNESS_ROOT=<harness install root> ARC3_PIVOT_WORKER_PYTHON=<python 3.12 with the harness runtime> \\
        pytest resources_servers/arc_agi_3_pivot/tests/test_worker_live.py

Each window restores its pivot (the harness's own pivot request must be the recorded one, byte for byte, and the oracle
distance the recorded one). The policy's turns are then played with the recording's own next replies (and their token
counts, which advance the run clock): the harness's next requests must be the recorded ones too, so the live path
continues the recorded game exactly.
"""

import json
import os
from http.cookiejar import CookieJar
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from resources_servers.arc_agi_3_pivot import recording
from resources_servers.arc_agi_3_pivot.app import ArcAgi3PivotConfig, ArcAgi3PivotResourcesServer


HERE = Path(__file__).resolve().parents[1]
POOL = HERE / "data" / "example_pool"
ROOT = os.environ.get("ARC3_PIVOT_HARNESS_ROOT")
PYTHON = os.environ.get("ARC3_PIVOT_WORKER_PYTHON")
pytestmark = pytest.mark.skipif(
    not (ROOT and PYTHON), reason="needs ARC3_PIVOT_HARNESS_ROOT and ARC3_PIVOT_WORKER_PYTHON"
)

ROWS = [json.loads(line) for line in (HERE / "data" / "example.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def client():
    config = ArcAgi3PivotConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        pool_dir=str(POOL),
        harness_root=ROOT,
        worker_python=PYTHON,
        max_calls=24,
        max_actions=100,
    )
    server = ArcAgi3PivotResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    class StatelessCookieJar(CookieJar):
        def extract_cookies(self, response: object, request: object) -> None:
            pass

    return TestClient(server.setup_webserver(), cookies=StatelessCookieJar())


def verify_body(meta: dict) -> dict:
    response = {
        "id": "r",
        "created_at": 0,
        "model": "m",
        "object": "response",
        "output": [],
        "tool_choice": "auto",
        "tools": [],
        "parallel_tool_calls": False,
    }
    return {
        "responses_create_params": {"input": [{"role": "user", "content": "x"}]},
        "verifier_metadata": meta,
        "response": response,
    }


@pytest.mark.parametrize("row", ROWS, ids=[r["verifier_metadata"]["pivot_id"] for r in ROWS])
def test_example_pivot_restores_and_continues_the_recording(client, row):
    meta = row["verifier_metadata"]
    pivot = json.loads(meta["pivot"]["pivot_json"])
    calls = recording.parse((POOL / pivot["snapshot"]["snap_file"]).read_bytes())["calls"]
    upto = next(i for i, c in enumerate(calls) if c["n"] == pivot["provenance"]["call"])

    seeded = client.post("/seed_session", json={"verifier_metadata": meta})
    prompt, cookies = seeded.json(), seeded.cookies
    assert prompt["restore_error"] is None, prompt["restore_error"]
    assert prompt["messages"] == calls[upto]["payload"]["messages"]
    assert prompt["tools"] == calls[upto]["payload"]["tools"]

    k, step = upto, None
    while recording.reply_of(calls, k) is not None:  # the recording's own continuation, as the policy's turns
        message = recording.reply_of(calls, k)
        tokens = (calls[k].get("usage") or {}).get("completion_tokens", 100)  # the run clock advances by them
        step = client.post("/step", json={"message": message, "gen_tokens": tokens}, cookies=cookies).json()
        if step["done"]:
            break
        assert step["request_messages"] == calls[k + 1]["payload"]["messages"], (
            f"request {k + 1}: {recording.first_difference(step['request_messages'], calls[k + 1]['payload']['messages'])}"
        )
        k += 1
    if step is None:  # the recording ends at the pivot: one new reply, and the harness's request extends the chain
        sent = prompt["messages"]
        message = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Look at the board size first.",
            "tool_calls": [
                {
                    "id": "call_live_1",
                    "type": "function",
                    "function": {"name": "python", "arguments": json.dumps({"code": "print(1 + 1)"})},
                }
            ],
        }
        step = client.post("/step", json={"message": message, "gen_tokens": 20}, cookies=cookies).json()
        assert step["done"] is False and step["request_messages"][: len(sent)] == sent
        tool = step["messages"][0]
        assert (
            tool["role"] == "tool" and tool["tool_call_id"] == "call_live_1" and '"stdout": "2\\n"' in tool["content"]
        )
    result = client.post("/verify", json=verify_body(meta), cookies=cookies).json()
    assert result["d0"] == pivot["po_d0"]
    if step["done"]:
        assert result["mask_sample"] is False, result.get("failure_reason")
        assert result["reward"] == result["d0"] - result["d_end"]


def test_record_then_restore(tmp_path):
    """record.py records an attempt with the scripted stub model and adds a pivot that restores exactly."""
    import shutil
    import threading

    import pyarrow.parquet as pq

    from resources_servers.arc_agi_3_pivot import record
    from responses_api_agents.arc_agi_3_agent.tests import stub_model

    pool = tmp_path / "pool"
    shutil.copytree(POOL / "env", pool / "env")
    shutil.copy(POOL / "games.parquet", pool / "games.parquet")
    threading.Thread(target=stub_model.serve, args=(18765, str(tmp_path / "stub.jsonl")), daemon=True).start()
    common = ["--harness-env", "env/kit-harness_env.toml", "--model", "stub", "--harness-root", ROOT]
    record.main(
        [str(pool), "--game", "cn04-2fe56bfb", "--model-url", "http://127.0.0.1:18765/v1", "--calls", "4"]
        + ["--pivots", "2", "--attempt", "a1", "--worker-python", PYTHON, *common]
    )
    (row,) = pq.read_table(pool / "example.parquet").to_pylist()
    assert row["pivot_id"] == "a1-c2" and row["restore_ok"] is True and row["po_d0"] is not None
    calls = recording.parse((pool / "snap-raw" / "a1.snap.jsonl").read_bytes())["calls"]
    assert [c["n"] for c in calls] == [0, 1, 2, 3, 4] and json.loads(row["pivot_json"])["call_index"] == 2
