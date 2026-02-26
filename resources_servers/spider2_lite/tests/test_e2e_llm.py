# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end test with a real LLM via vLLM.

Starts a local vLLM server, runs the 5 example tasks through the full pipeline
(LLM inference -> SQL extraction -> execution-based verification), and reports
the pass rate.

Usage:
    pytest tests/test_e2e_llm.py -m e2e_llm -v -s

Environment variables:
    SPIDER2_LLM_MODEL   HuggingFace model ID (default: openai/gpt-oss-20b)
    SPIDER2_LLM_PORT    vLLM listen port (default: 18765)
    SPIDER2_LLM_GPU     CUDA_VISIBLE_DEVICES value (default: 0)
    SPIDER2_LLM_URL     Skip vLLM startup and use this base URL directly
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.spider2_lite.app import Spider2LiteResourcesServer, Spider2LiteResourcesServerConfig
from resources_servers.spider2_lite.setup_spider2 import _DEFAULT_DIR

pytestmark = pytest.mark.e2e_llm

_EXAMPLE_JSONL = Path(__file__).parent.parent / "data" / "example.jsonl"


@pytest.fixture(scope="module")
def spider2_client():
    """TestClient backed by a real Spider2-Lite resource server with real DBs."""
    cfg = Spider2LiteResourcesServerConfig(
        host="127.0.0.1",
        port=20099,
        entrypoint="",
        spider2_lite_dir=str(_DEFAULT_DIR),
        max_concurrency=4,
        sql_execution_timeout_s=60.0,
    )
    srv = Spider2LiteResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))
    with TestClient(srv.setup_webserver()) as c:
        yield c


def _load_example_tasks() -> list[dict]:
    return [json.loads(line) for line in _EXAMPLE_JSONL.read_text().splitlines() if line.strip()]


EXAMPLE_TASKS = _load_example_tasks()


def _chat(base_url: str, model: str, messages: list, timeout: int = 120) -> str:
    """Call vLLM chat completions and return the assistant message text."""
    r = requests.post(
        f"{base_url}/chat/completions",
        json={"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 2048},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _verify(client: TestClient, task: dict, model_output: str) -> dict:
    body = {
        "responses_create_params": {"input": []},
        "response": {
            "id": "r", "created_at": 0, "model": "m", "object": "response",
            "output": [{"id": "msg", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": model_output, "annotations": []}]}],
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        },
        "instance_id": task.get("instance_id"),
        "db_id": task["db_id"],
        "question": task["question"],
        "gold_sql": task.get("gold_sql"),
        "ignore_order": task.get("ignore_order", True),
        "condition_cols": task.get("condition_cols"),
    }
    resp = client.post("/verify", json=body)
    assert resp.status_code == 200, f"Verify endpoint returned {resp.status_code}: {resp.text}"
    return resp.json()


# ---------------------------------------------------------------------------
# Per-task pipeline smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", EXAMPLE_TASKS, ids=lambda t: t["instance_id"])
def test_llm_pipeline_no_errors(vllm_url, llm_model, spider2_client, task):
    """The full pipeline must produce a valid reward for every example task.

    Does not assert reward=1.0 (LLM quality varies), but asserts:
    - Model returns a non-empty response
    - SQL is extracted from the response
    - Verifier returns 0.0 or 1.0 without unknown_error
    """
    messages = task["responses_create_params"]["input"]
    model_output = _chat(vllm_url, llm_model, messages)

    assert model_output, "LLM returned an empty response"

    result = _verify(spider2_client, task, model_output)

    assert result["failure_reason"] != "unknown_error", (
        f"{task['instance_id']}: unexpected server error — {result}"
    )
    assert result["failure_reason"] != "no_sql_extracted", (
        f"{task['instance_id']}: model did not produce a ```sql``` block.\n"
        f"Model output:\n{model_output[:500]}"
    )
    assert result["reward"] in (0.0, 1.0)

    # Print for visibility when running with -s
    print(
        f"\n{task['instance_id']}: reward={result['reward']}  "
        f"failure={result['failure_reason']}  "
        f"sql={result.get('extracted_sql', '')[:80]!r}"
    )


