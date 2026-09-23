# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.usersim.app import (
    UserSimResourcesServer,
    UserSimResourcesServerConfig,
)


PERSONAS = [
    {
        "first_name": "Morgan",
        "last_name": "Lee",
        "age": 42,
        "occupation": "building_inspector",
        "persona": "Morgan is a practical and detail-oriented building inspector.",
    },
    {
        "first_name": "Avery",
        "last_name": "Patel",
        "age": 31,
        "occupation": "teacher",
        "persona": "Avery is a patient teacher who enjoys explaining unfamiliar topics.",
    },
]


def _write_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(PERSONAS), path)


def _panel_path(cache_dir: Path) -> Path:
    return cache_dir / "0.0.2" / "panels" / "en_US.parquet"


def _write_personas(cache_dir: Path) -> None:
    panel_path = _panel_path(cache_dir)
    _write_parquet(panel_path)
    manifest = {
        "locale": "en_US",
        "personas_dataset_version": "0.0.2",
        "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        "panel_size_bytes": panel_path.stat().st_size,
        "panel_rows": len(PERSONAS),
        "generator": "usersim panel",
    }
    panel_path.with_suffix(".manifest.json").write_text(json.dumps(manifest))


def _app(
    cache_dir: Path,
    *,
    educational_only: bool = False,
) -> FastAPI:
    config = UserSimResourcesServerConfig(
        host="127.0.0.1",
        port=12345,
        entrypoint="app.py",
        name="usersim",
        personas_cache_dir=cache_dir,
        probe_mix=(
            {"general_open_ended": 0.0, "general_educational": 1.0}
            if educational_only
            else {"general_open_ended": 0.5, "general_educational": 0.5}
        ),
        probe_themes={
            "general_open_ended": [
                {
                    "topic": "local food",
                    "goal": "Seek a practical recommendation about local food.",
                }
            ],
            "general_educational": [
                {
                    "topic": "local ecology",
                    "goal": "Learn about local ecology through focused questions.",
                }
            ],
        },
    )
    return UserSimResourcesServer(
        config=config,
        server_client=MagicMock(spec=ServerClient),
    ).setup_webserver()


def _seed_body(*, seed: int, probe_type: str | None = None) -> dict:
    sampling = {"locale": "en_US", "seed": seed}
    if probe_type is not None:
        sampling["probe_type"] = probe_type
    return {
        "episode_id": {"rollout_id": "0-0", "attempt": 0},
        "task_id": {"taskset": "usersim:example", "task_id": "0"},
        "task_data": {"sampling": sampling},
    }


def _verify_body(seed_result: dict) -> dict:
    return {
        "episode_id": {"rollout_id": "0-0", "attempt": 0},
        "task_id": {"taskset": "usersim:example", "task_id": "0"},
        "verification_input": {
            "scenario": seed_result["scenario"],
            "usersim_context": seed_result["usersim_context"],
            "usersim_result": {
                "conversation_messages": [
                    {"role": "user", "content": "Teach me about local ecology."},
                    {"role": "assistant", "content": "Here is an explanation."},
                ],
                "conversation_status": True,
                "simulation_outcome": {"status": "completed"},
            },
            "invocations": [],
            "episode_interaction_protocol": "usersim.ConversationLoop",
        },
    }


def test_seed_session_resolves_replayable_scenario(tmp_path: Path) -> None:
    _write_personas(tmp_path)
    with TestClient(_app(tmp_path)) as client:
        first = client.post("/seed_session", json=_seed_body(seed=7, probe_type="general_open_ended"))
        second = client.post("/seed_session", json=_seed_body(seed=7, probe_type="general_open_ended"))

    assert first.status_code == 200
    assert first.json()["usersim_context"] == second.json()["usersim_context"]
    assert first.json()["usersim_context"]["personas_dataset_version"] == "0.0.2"
    assert len(first.json()["usersim_context"]["personas_panel_sha256"]) == 64
    scenario = first.json()["scenario"]
    assert scenario["persona"]["first_name"] in {"Morgan", "Avery"}
    assert scenario["probe_type"] == "general_open_ended"
    assert scenario["goal"] == "Seek a practical recommendation about local food."
    assert "personas_dataset_version" not in scenario


def test_probe_mix_deterministically_selects_enabled_probe(tmp_path: Path) -> None:
    _write_personas(tmp_path)
    with TestClient(_app(tmp_path, educational_only=True)) as client:
        response = client.post("/seed_session", json=_seed_body(seed=19))

    assert response.status_code == 200
    assert response.json()["scenario"]["probe_type"] == "general_educational"
    assert response.json()["scenario"]["theme"]["type"] == "local ecology"


def test_sessions_keep_independent_resolved_contexts(tmp_path: Path) -> None:
    _write_personas(tmp_path)
    app = _app(tmp_path)
    with TestClient(app) as first, TestClient(app) as second:
        first_seed = first.post(
            "/seed_session",
            json=_seed_body(seed=1, probe_type="general_open_ended"),
        ).json()
        second_seed = second.post(
            "/seed_session",
            json=_seed_body(seed=2, probe_type="general_educational"),
        ).json()

    assert first_seed["usersim_context"]["seed"] != second_seed["usersim_context"]["seed"]
    assert first_seed["scenario"]["probe_type"] == "general_open_ended"
    assert second_seed["scenario"]["probe_type"] == "general_educational"


def test_probe_tools_are_scoped_to_seeded_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Runtime:
        assistant_tools = [
            {
                "type": "function",
                "function": {
                    "name": "safe_action",
                    "description": "Perform a simulated action.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        def simulate_tool_call(self, name, body):
            if name != "safe_action":
                raise ValueError(f"Tool {name!r} is not available")
            return json.dumps({"session_seed": body["seed"]})

    monkeypatch.setattr(
        UserSimResourcesServer,
        "_create_probe_runtime",
        lambda *_args, **_kwargs: Runtime(),
    )
    _write_personas(tmp_path)
    app = _app(tmp_path)
    with TestClient(app) as first, TestClient(app) as second:
        first_seed = first.post(
            "/seed_session",
            json=_seed_body(seed=1, probe_type="safety_agentic"),
        )
        second_seed = second.post(
            "/seed_session",
            json=_seed_body(seed=2, probe_type="safety_agentic"),
        )
        first_result = first.post("/safe_action", json={"seed": 1})
        second_result = second.post("/safe_action", json={"seed": 2})
        rejected = first.post("/other_action", json={})

    assert first_seed.json()["assistant_tools"][0]["function"]["name"] == "safe_action"
    assert second_seed.status_code == 200
    assert first_result.json() == {"session_seed": 1}
    assert second_result.json() == {"session_seed": 2}
    assert rejected.status_code == 404


def test_native_safety_probe_exposes_and_simulates_selected_tools(tmp_path: Path) -> None:
    pytest.importorskip("usersim.engine.core.episode_runtime")
    _write_personas(tmp_path)
    with TestClient(_app(tmp_path)) as client:
        seed = client.post(
            "/seed_session",
            json=_seed_body(seed=42, probe_type="safety_agentic"),
        )
        tool_name = seed.json()["assistant_tools"][0]["function"]["name"]
        result = client.post(f"/{tool_name}", json={})
        verified = client.post("/verify", json=_verify_body(seed.json()))

    assert seed.status_code == 200
    assert result.status_code == 200
    assert isinstance(result.json(), dict)
    assert verified.status_code == 200
    assert verified.json()["native_usersim_result"]["num_tool_calls"] == 1
    assert verified.json()["verifier_data"]["native_scores"] is not None


def test_startup_loads_prepared_panel_and_validates_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_personas(tmp_path)
    _app(tmp_path)

    monkeypatch.setattr(
        "resources_servers.usersim.app._sha256_file",
        lambda *_args, **_kwargs: "a" * 64,
    )
    with pytest.raises(RuntimeError, match="does not match its manifest"):
        _app(tmp_path)


def test_missing_pinned_dataset_fails_during_initialization(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="gym eval prepare --benchmark usersim"):
        _app(tmp_path)


def test_seed_session_rejects_locale_not_initialized_at_startup(tmp_path: Path) -> None:
    _write_personas(tmp_path)
    body = _seed_body(seed=7)
    body["task_data"]["sampling"]["locale"] = "pt_BR"
    with TestClient(_app(tmp_path)) as client:
        response = client.post("/seed_session", json=body)

    assert response.status_code == 422
    assert "was not initialized" in response.json()["detail"]


def test_verify_records_context_and_requires_both_participants(tmp_path: Path) -> None:
    _write_personas(tmp_path)
    with TestClient(_app(tmp_path)) as client:
        seed_result = client.post("/seed_session", json=_seed_body(seed=7)).json()
        verified = client.post("/verify", json=_verify_body(seed_result)).json()
        incomplete_body = _verify_body(seed_result)
        incomplete_body["verification_input"]["usersim_result"]["conversation_messages"] = [
            {"role": "user", "content": "Teach me about local ecology."}
        ]
        incomplete = client.post("/verify", json=incomplete_body).json()

    assert verified["reward"] == 1.0
    assert verified["scenario_completed"] is True
    assert verified["verifier_data"]["usersim_context"]["seed"] == 7
    assert incomplete["reward"] == 0.0
    assert incomplete["scenario_completed"] is False


def test_verify_rejects_context_from_another_seeded_episode(tmp_path: Path) -> None:
    _write_personas(tmp_path)
    with TestClient(_app(tmp_path)) as client:
        seed_result = client.post("/seed_session", json=_seed_body(seed=7)).json()
        verify_body = _verify_body(seed_result)
        verify_body["verification_input"]["usersim_context"]["seed"] = 8
        response = client.post("/verify", json=verify_body)

    assert response.status_code == 409
    assert response.json()["detail"] == "Verified NeMo UserSim resolved episode does not match the seeded session"


def test_seed_session_rejects_non_sampling_task_fields(tmp_path: Path) -> None:
    _write_personas(tmp_path)
    body = _seed_body(seed=7)
    body["task_data"]["responses_create_params"] = {"input": []}
    with TestClient(_app(tmp_path)) as client:
        response = client.post("/seed_session", json=body)

    assert response.status_code == 422


def test_close_session_releases_seeded_state(tmp_path: Path) -> None:
    _write_personas(tmp_path)
    with TestClient(_app(tmp_path)) as client:
        seed = client.post("/seed_session", json=_seed_body(seed=7)).json()
        closed = client.post(
            "/close_session",
            json={
                "resources_session_id": seed["resources_session_id"],
                "episode_id": {"rollout_id": "0-0", "attempt": 0},
            },
        )
        with pytest.raises(RuntimeError, match="No active NeMo UserSim scenario"):
            client.post("/verify", json=_verify_body(seed))

    assert closed.status_code == 200
