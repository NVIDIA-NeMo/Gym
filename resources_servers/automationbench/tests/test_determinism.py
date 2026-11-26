# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`deterministic: true`: a replay from the same record reproduces ids, timestamps and tool text."""

import base64
import copy
import json
import random
import time
import uuid
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from resources_servers.automationbench import determinism
from resources_servers.automationbench.app import DETERMINISM_KEY, new_world_state, run_tool
from resources_servers.automationbench.tests.test_app import TASK, _post_seed, _seed, _server


RAW = base64.urlsafe_b64encode(b"To: a@example.com\r\nSubject: hi\r\n\r\nhello").decode()
SEND = {
    "method": "POST",
    "url": "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
    "params": {},
    "body": {"raw": RAW},
}
CREATE_LABEL = {
    "method": "POST",
    "url": "https://gmail.googleapis.com/gmail/v1/users/me/labels",
    "params": {},
    "body": {"name": "Refunds"},
}
GMAIL_TASK = {"initial_state": {"gmail": {"messages": []}}, "assertions": [{"type": "gmail_message_sent"}]}
BASE = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)


def _replay(record: dict) -> tuple[list[str], str]:
    """A fresh episode under a copy of `record`, two writes; their tool text and the final world."""
    state = new_world_state(GMAIL_TASK, copy.deepcopy(record))
    outputs = [run_tool("api_fetch", call, state) for call in (SEND, CREATE_LABEL)]
    return outputs, state["world"].model_dump_json(exclude={"meta"})


def test_the_same_record_replays_byte_for_byte() -> None:
    record = determinism.new_record("task-7", BASE)
    first, second = _replay(record), _replay(record)
    assert first == second
    # Both writes mint ids (message, label); they must match across replays.
    assert "id" in json.loads(first[0][0]) and "id" in json.loads(first[0][1])


def test_a_different_seed_mints_different_ids() -> None:
    a, _ = _replay(determinism.new_record("task-7", BASE))
    b, _ = _replay(determinism.new_record("task-8", BASE))
    assert json.loads(a[0])["id"] != json.loads(b[0])["id"]


def test_the_clock_is_frozen_per_call_and_advances_by_one_second() -> None:
    record, holder, seen = determinism.new_record("t", BASE), {}, []
    for _ in range(2):
        with determinism.deterministic_call(record, holder):
            seen.append(time.time())
    assert seen == [1793534400.0, 1793534401.0]
    assert record["calls"] == 2


def test_the_global_rng_and_uuid4_are_untouched_outside_a_call() -> None:
    random.seed(123)
    expected_next = random.Random(123).random()
    with determinism.deterministic_call(determinism.new_record("t"), {}):
        random.random()
    assert random.random() == expected_next
    assert uuid.uuid4() != uuid.uuid4()


def test_building_the_world_is_call_zero_and_keeps_embedded_invitees() -> None:
    # Building a world pops invitees embedded in Calendly events out of its input; the episode's own copy keeps them.
    task = {
        "initial_state": {
            "calendly": {
                "scheduled_events": [
                    {
                        "id": "evt_1",
                        "name": "Intro",
                        "invitees": [{"id": "inv_1", "name": "Ada", "email": "a@example.com"}],
                    }
                ]
            }
        },
        "assertions": [{"type": "calendly_event_exists"}],
    }
    state = new_world_state(task, determinism.new_record("t", BASE))
    assert state[DETERMINISM_KEY]["calls"] == 1
    assert [i.email for i in state["world"].calendly.get_invitees_for_event("evt_1")] == ["a@example.com"]


def test_a_server_without_the_option_keeps_no_record() -> None:
    server = _server()
    with TestClient(server.setup_webserver()) as client:
        assert _post_seed(client, _seed()).status_code == 200
        assert DETERMINISM_KEY not in server._episodes["session"].state


def test_a_rows_record_starts_the_episode_and_each_tool_call_advances_it() -> None:
    record = determinism.new_record("captured", BASE)
    server = _server(deterministic=True)
    with TestClient(server.setup_webserver()) as client:
        assert _post_seed(client, _seed(task=TASK | {"determinism": record})).status_code == 200
        live = server._episodes["session"].state[DETERMINISM_KEY]
        assert (live["seed"], live["clock_base"], live["calls"]) == (record["seed"], record["clock_base"], 1)
        assert client.post("/api_search", json={"query": "salesforce opportunity", "top_k": 1}).status_code == 200
        assert live["calls"] == 2


def test_rows_without_a_record_are_seeded_from_their_task_id() -> None:
    seeds = []
    for session in ("a", "b"):
        server = _server(deterministic=True)
        with TestClient(server.setup_webserver()) as client:
            assert _post_seed(client, _seed(session)).status_code == 200
            seeds.append(server._episodes[session].state[DETERMINISM_KEY]["seed"])
    assert seeds[0] == seeds[1] == determinism.new_record(TASK["task_id"])["seed"]
