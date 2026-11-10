# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-runtime verification using the actual pinned author parser and dataset."""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from benchmarks.vs_bench.prepare import convert_row
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse
from resources_servers.vs_bench_reference.app import (
    VSBenchReasoningRequest,
    VSBenchReferenceConfig,
    VSBenchReferenceServer,
)


AUTHOR = Path(os.environ.get("VS_BENCH_TEST_UPSTREAM", "/nonexistent"))
DATA = Path(os.environ.get("VS_BENCH_TEST_DATASET", "/nonexistent"))
PYTHON = os.environ.get("VS_BENCH_TEST_AUTHOR_PYTHON", "")
pytestmark = pytest.mark.skipif(
    not AUTHOR.is_dir() or not DATA.is_dir() or not Path(PYTHON).is_file(),
    reason="Pinned author checkout, dataset and 3.10 runtime required",
)


def test_actual_gym_verifier_uses_author_parser_and_rejects_retry_case():
    raw = next(pq.ParquetFile(DATA / "reasoning/kuhn_poker.parquet").iter_batches(batch_size=1)).to_pylist()[0]
    prepared = convert_row("reasoning", "kuhn_poker", 0, raw)
    legal = json.loads(raw["next_legal_actions"])
    answer = json.dumps({"action": legal[str(raw["next_action"][0])]})
    config = VSBenchReferenceConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="",
        name="vs_bench_reference",
        author_root=str(AUTHOR),
        author_python=PYTHON,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    server = VSBenchReferenceServer(config=config, server_client=client)

    def request(text):
        response = NeMoGymResponse.model_validate(
            {
                "id": "resp",
                "created_at": 0,
                "model": "test",
                "object": "response",
                "output": [
                    {
                        "id": "message",
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        return VSBenchReasoningRequest(**prepared, response=response)

    result = asyncio.run(server.verify(request(answer)))
    assert result.reward == 1.0
    assert result.author_source_variant == server._author_source_variant
    assert result.action in raw["next_action"]
    with pytest.raises(ValueError, match="retry loop"):
        asyncio.run(server.verify(request("I cannot decide.")))
    illegal = asyncio.run(server.verify(request('{"action": "definitely-illegal"}')))
    assert illegal.reward == 0.0
    assert illegal.action == -99999999
    assert illegal.parse_outcome == "illegal_action"

    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as http:
        valid = http.post("/verify", json=request(answer).model_dump(mode="json"))
        assert valid.status_code == 200, valid.text
        assert valid.json()["reward"] == 1.0
        invalid = http.post("/verify", json=request("I cannot decide.").model_dump(mode="json"))
        assert invalid.status_code >= 400

    metadata = dict(prepared["verifier_metadata"], track="perception")
    with pytest.raises(ValueError, match="Only reference reasoning"):
        asyncio.run(server.score_attempt(metadata, answer))
