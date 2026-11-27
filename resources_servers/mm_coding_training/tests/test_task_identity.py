# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise training rows without the eval collector's positional task index."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
from environment_servers.single_agent_turn_legacy.tests.test_app import _environment_server
from nemo_gym.data import materialize_task
from nemo_gym.server_runtime import SingleAgentTurnTaskInput
from resources_servers.mm_coding_training.prepare_task_ids import prepare_task_ids, with_task_id
from resources_servers.mm_coding_training.task_data import TaskData


ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = [
    json.loads(line)
    for line in (ROOT / "resources_servers/mm_coding_training/data/example.jsonl").read_text().splitlines()
]


@pytest.mark.parametrize("row", EXAMPLES, ids=lambda row: row["task"]["benchmark"])
def test_single_training_row_runs_without_eval_task_index(row):
    environment, client = _environment_server()
    adapter = SingleAgentTurnLegacyEnvironmentServer(config=environment.config, server_client=client)
    # Bind the checked-in row to the test deployment. These are rollout bookkeeping,
    # not the eval collector's absent _ng_task_index or a replacement task identity.
    payload = row | {"agent_ref": {"name": "agent"}, "_ng_rollout_index": 0, "_ng_rollout_id": "training-0"}
    response = TestClient(adapter.setup_webserver()).post("/run", json=payload)

    assert response.status_code == 200, response.text
    assert response.json()["reward"] == 1.0
    seed = client.calls[0][2]["json"]
    assert seed["task_id"] == {
        "taskset": "resources",
        "task_id": f"{row['task']['benchmark']}:{row['task']['task_id']}",
    }
    assert TaskData.model_validate(seed["task_data"]).task == TaskData(task=row["task"]).task
    assert seed["task_data"]["task"] == row["task"]
    assert seed["responses_create_params"] == row["responses_create_params"]
    assert client.calls[-1][1] == "/close_session"


def test_missing_task_identity_reproduces_training_422_before_agent_execution():
    environment, client = _environment_server()
    adapter = SingleAgentTurnLegacyEnvironmentServer(config=environment.config, server_client=client)
    row = {key: value for key, value in EXAMPLES[2].items() if key != "problem_id"}
    response = TestClient(adapter.setup_webserver()).post(
        "/run", json=row | {"agent_ref": {"name": "agent"}, "_ng_rollout_id": "training-0"}
    )
    assert response.status_code == 422
    assert "requires a non-negative task_index" in response.json()["detail"]
    assert not client.calls


def test_conversion_preserves_inputs_and_ids_across_order_and_repeated_rollouts(tmp_path):
    rows = deepcopy(EXAMPLES)
    for row in rows:
        row.pop("problem_id")
        row["task"]["task_id"] = "same-source-id"
    source, output = tmp_path / "source.jsonl", tmp_path / "prepared.jsonl"
    original = "".join(json.dumps(row) + "\n" for row in rows).encode()
    source.write_bytes(original)
    assert prepare_task_ids(source, output) == 5
    prepared = [json.loads(line) for line in output.read_text().splitlines()]
    assert source.read_bytes() == original
    assert [{key: value for key, value in row.items() if key != "problem_id"} for row in prepared] == rows
    assert [with_task_id(row) for row in reversed(prepared)] == list(reversed(prepared))
    identities = []
    for row in prepared:
        first = materialize_task(row | {"_ng_rollout_index": 0}, taskset="coding")
        repeated = materialize_task(row | {"_ng_rollout_index": 9}, taskset="coding")
        assert first == repeated
        TaskData.model_validate(SingleAgentTurnTaskInput.model_validate(first["task_input"]).task_data)
        identities.append(first["task_id"]["task_id"])
    assert len(set(identities)) == 5


@pytest.mark.parametrize("change", [{"problem_id": "wrong:task"}, {"task_id": "other"}, {"instance_id": "other"}])
def test_conversion_refuses_ambiguous_ids_without_partial_publication(tmp_path, change):
    source, output = tmp_path / "source.jsonl", tmp_path / "prepared.jsonl"
    source.write_text(json.dumps(EXAMPLES[0]) + "\n" + json.dumps(EXAMPLES[1] | change) + "\n")
    with pytest.raises(ValueError):
        prepare_task_ids(source, output)
    assert not output.exists()
    assert set(tmp_path.iterdir()) == {source}


def test_conversion_refuses_duplicates_and_overwriting_either_file(tmp_path):
    source, output = tmp_path / "source.jsonl", tmp_path / "prepared.jsonl"
    original = json.dumps(EXAMPLES[0]) + "\n"
    source.write_text(original * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_task_ids(source, output)
    assert not output.exists()
    source.write_text(original)
    output.write_text("keep existing output")
    with pytest.raises(FileExistsError):
        prepare_task_ids(source, output)
    with pytest.raises(FileExistsError):
        prepare_task_ids(source, source)
    assert output.read_text() == "keep existing output"
    assert source.read_text() == original


def test_seed_schema_rejects_stale_identity_but_accepts_task_without_transport_id():
    row = EXAMPLES[0]
    TaskData(task=row["task"])
    with pytest.raises(ValidationError, match="must match"):
        TaskData(task=row["task"], problem_id="wrong:task")
