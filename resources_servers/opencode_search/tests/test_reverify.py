# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grading a saved answer again, as gym eval reverify does and as recovery does for a row whose judge never answered:
the /verify body both build from a saved row, and the grade this server gives a row that ended on its budget."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME, ModelServerRef
from nemo_gym.data import materialize_task
from nemo_gym.eval.rollout_reverification import InputRolloutPair, _build_verify_payload
from resources_servers.opencode_search.app import SearchResourcesServer, SearchResourcesServerConfig


CORRECT = "extracted_final_answer: Paris\nreasoning: same city\ncorrect: yes"
ROW = {
    "responses_create_params": {"input": [{"role": "user", "content": "Which city?"}]},
    "question": "Which city?",
    "ground_truth": "Paris",
}
IDENTITY = {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}
SENTINEL = "[nemo-gym-plugin] loaded sentinel version=1 plugins=\n"
# The line nemo_gym_turn_budget.js reports for each budget ending, which the plugin log of a run that ended so holds.
ENDING_LINES = {
    "max_iteration": "[nemo-gym-plugin] terminal=max_iteration session=s calls=401 cap=400\n",
    "context_window": "[nemo-gym-plugin] context_overflow session=s\n",
}
# The materialized input a saved row is paired with: the dataset row itself, as both BrowseComp presets collect it, or
# the task it becomes for an Environment Server that routes by taskset.
MATERIALIZED_INPUTS = {
    "dataset_row": ROW | IDENTITY,
    "task": materialize_task(ROW, taskset="browsecomp", task_index=0) | IDENTITY,
}


def _response(text: str) -> dict[str, Any]:
    return {
        "id": "resp",
        "created_at": 0.0,
        "model": "model",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": "msg",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def _saved_row(saved_as: str, reason: str, kind: str) -> dict[str, Any]:
    """A saved row of a run that ended on its budget, with the receipts verify took from its plugin log: graded and
    masked with the ending's kind, or sent to the failures sidecar because its judge never answered."""
    receipts = {
        "opencode_terminal_reason": reason,
        "opencode_plugins_loaded": True,
        "opencode_plugin_log": SENTINEL + ENDING_LINES[reason],
    }
    row = ROW | IDENTITY | receipts | {"response": _response("Exact Answer: Paris"), "mask_sample": True}
    if saved_as == "graded":
        return row | {"reward": 1.0, "failure_kind": kind, "evaluation_completed": True}
    # judge_failsafe's row, which stays in the failures sidecar until recovery grades it again.
    return row | {
        "reward": 0.0,
        "failure_kind": "judge_failed",
        "evaluation_completed": False,
        "_ng_failure_class": "judge_failed",
    }


def _server() -> SearchResourcesServer:
    """This server as the session preset sets it up, reading the plugin log, with a judge that answers yes.

    The relay preset sets no plugin_log_path, so its verify reads only the body anyway.
    """
    config = SearchResourcesServerConfig(
        name="opencode_search",
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        sandbox_provider="sandbox",
        sandbox_config={},
        sandbox_image="python:3.12",
        sandbox_workdir="/workspace",
        plugin_log_path="/tmp/nemo-gym-plugin-receipts.log",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
    )
    server = SearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    reply = MagicMock()
    reply.read = AsyncMock(return_value=json.dumps(_response(CORRECT)).encode())
    server.server_client.post = AsyncMock(return_value=reply)
    return server


@pytest.mark.parametrize(
    ("reason", "kind"),
    [("max_iteration", "agent_limit"), ("context_window", "model_context_limit")],
    ids=["max_iteration", "context_window"],
)
@pytest.mark.parametrize("saved_as", ["graded", "judge_failed"])
@pytest.mark.parametrize("materialized", ["dataset_row", "task"])
def test_a_row_that_ended_on_its_budget_is_graded_again_without_its_mask(materialized, saved_as, reason, kind) -> None:
    """The body carries the saved answer but none of the receipts, and the plugin log that gave the ending went with the
    sandbox, so the row comes back graded and unmasked."""
    pair = InputRolloutPair(input=MATERIALIZED_INPUTS[materialized], rollout=_saved_row(saved_as, reason, kind))

    payload = _build_verify_payload(pair)

    assert [key for key in payload if key.startswith("opencode_")] == []
    server = _server()
    verified = TestClient(server.setup_webserver()).post("/verify", json=payload)
    assert verified.status_code == 200, verified.text
    row = verified.json()
    graded = (row["reward"], row["mask_sample"], row["failure_kind"], row["evaluation_completed"])
    assert graded == (1.0, False, None, True)
    # The saved answer was judged again.
    assert server.server_client.post.await_count == 1
