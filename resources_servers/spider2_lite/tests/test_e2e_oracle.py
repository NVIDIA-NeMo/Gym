# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
E2E oracle validation for Spider 2.0-Lite resource server.

Runs the gold SQL through the real server against real downloaded databases.
Every oracle call must return reward=1.0. Not part of CI; run locally.

Usage:
    pytest tests/test_e2e_oracle.py -m e2e -v

Requires:
    - Downloaded databases (auto-triggered by conftest.py)
    - Spider2 reference repo at ~/code/spider2-lite-ref (or set SPIDER2_LITE_REF_DIR)
"""

import csv
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.spider2_lite.app import Spider2LiteResourcesServer, Spider2LiteResourcesServerConfig
from resources_servers.spider2_lite.setup_spider2 import _DEFAULT_DIR

pytestmark = pytest.mark.e2e

_SPIDER2_REF = Path(os.environ.get("SPIDER2_LITE_REF_DIR", "~/code/spider2-lite-ref")).expanduser()
_GOLD_SQL_DIR = _SPIDER2_REF / "spider2-lite/evaluation_suite/gold/sql"
_GOLD_CSV_DIR = _SPIDER2_REF / "spider2-lite/evaluation_suite/gold/exec_result"
_EVAL_JSONL = _SPIDER2_REF / "spider2-lite/evaluation_suite/gold/spider2lite_eval.jsonl"
_MAIN_JSONL = _SPIDER2_REF / "spider2-lite/spider2-lite.jsonl"


@pytest.fixture(scope="module")
def client():
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


def _load_oracle_tasks():
    main = {json.loads(l)["instance_id"]: json.loads(l) for l in open(_MAIN_JSONL)}
    eval_meta = {json.loads(l)["instance_id"]: json.loads(l) for l in open(_EVAL_JSONL)}
    tasks = []
    for sql_file in sorted(_GOLD_SQL_DIR.glob("local*.sql")):
        iid = sql_file.stem
        t, e = main[iid], eval_meta[iid]
        tasks.append({
            "instance_id": iid,
            "db_id": t["db"],
            "question": t["question"],
            "gold_sql": sql_file.read_text().strip(),
            "ignore_order": e["ignore_order"],
            "condition_cols": e["condition_cols"],
        })
    return tasks


ORACLE_TASKS = _load_oracle_tasks()


@pytest.fixture(scope="module")
def oracle_tasks():
    return ORACLE_TASKS


def _verify(client, task, model_output, **overrides):
    body = {
        "responses_create_params": {"input": []},
        "response": {
            "id": "r", "created_at": 0, "model": "m", "object": "response",
            "output": [{"id": "msg", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": model_output, "annotations": []}]}],
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        },
        "instance_id": task["instance_id"],
        "db_id": task["db_id"],
        "question": task["question"],
        "gold_sql": task["gold_sql"],
        "ignore_order": task["ignore_order"],
        "condition_cols": task["condition_cols"],
        **overrides,
    }
    resp = client.post("/verify", json=body)
    assert resp.status_code == 200
    return resp.json()


@pytest.mark.parametrize("task", ORACLE_TASKS, ids=lambda t: t["instance_id"])
def test_oracle_gold_sql_reward_1(client, task):
    """Gold SQL as model output must always yield reward=1.0."""
    result = _verify(client, task, f"```sql\n{task['gold_sql']}\n```")
    assert result["reward"] == 1.0, (
        f"{task['instance_id']}: reward={result['reward']}, failure={result.get('failure_reason')}"
    )
    assert result["execution_match"] is True


@pytest.mark.parametrize("instance_id", ["local022", "local038", "local039"])
def test_wrong_sql_reward_0(client, oracle_tasks, instance_id):
    """Trivially wrong SQL must yield reward=0.0."""
    task = next(t for t in oracle_tasks if t["instance_id"] == instance_id)
    result = _verify(client, task, "```sql\nSELECT 1;\n```")
    assert result["reward"] == 0.0


def _load_gold_result(instance_id: str) -> list[list]:
    result_sets = []
    for csv_file in sorted(_GOLD_CSV_DIR.glob(f"{instance_id}_*.csv")):
        with open(csv_file) as f:
            rows = list(csv.reader(f))
        result_sets.append(rows[1:])  # skip header
    if not result_sets:
        plain = _GOLD_CSV_DIR / f"{instance_id}.csv"
        if plain.exists():
            with open(plain) as f:
                rows = list(csv.reader(f))
            result_sets.append(rows[1:])
    return result_sets


@pytest.mark.parametrize("instance_id", ["local022", "local039"])
def test_gold_result_mode_oracle(client, oracle_tasks, instance_id):
    """gold_result mode: oracle SQL against pre-computed CSV rows must yield reward=1.0."""
    task = next(t for t in oracle_tasks if t["instance_id"] == instance_id)
    gold_result = _load_gold_result(instance_id)
    assert gold_result, f"No gold CSV found for {instance_id}"
    result = _verify(
        client, task,
        f"```sql\n{task['gold_sql']}\n```",
        gold_sql=None,
        gold_result=gold_result,
    )
    assert result["reward"] == 1.0, (
        f"{instance_id}: reward={result['reward']}, failure={result.get('failure_reason')}"
    )


def test_oracle_pass_rate(client, oracle_tasks):
    """All 24 tasks with gold SQL must achieve 100% pass rate with oracle input."""
    failures = []
    for task in oracle_tasks:
        result = _verify(client, task, f"```sql\n{task['gold_sql']}\n```")
        if result["reward"] != 1.0:
            failures.append(f"{task['instance_id']}: reward={result['reward']}, reason={result.get('failure_reason')}")
    assert not failures, f"{len(failures)}/24 oracle tasks failed:\n" + "\n".join(failures)
