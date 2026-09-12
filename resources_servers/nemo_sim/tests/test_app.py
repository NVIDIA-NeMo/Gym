# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.nemo_sim.app import (
    NeMoSimResourcesServer,
    NeMoSimResourcesServerConfig,
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


def _source_path(cache_dir: Path) -> Path:
    return cache_dir / "0.0.2" / "source" / "en_US.parquet"


def _write_personas(cache_dir: Path) -> None:
    _write_parquet(_source_path(cache_dir))


def _app(
    cache_dir: Path,
    *,
    educational_only: bool = False,
) -> FastAPI:
    config = NeMoSimResourcesServerConfig(
        host="127.0.0.1",
        port=12345,
        entrypoint="app.py",
        name="nemo_sim",
        personas_cache_dir=cache_dir,
        personas_panel_size=2,
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
    return NeMoSimResourcesServer(
        config=config,
        server_client=MagicMock(spec=ServerClient),
    ).setup_webserver()


def _seed_body(*, seed: int, probe_type: str | None = None) -> dict:
    sampling = {"locale": "en_US", "seed": seed}
    if probe_type is not None:
        sampling["probe_type"] = probe_type
    return {
        "responses_create_params": {"input": []},
        "nemo_sim_sampling": sampling,
    }


def _response(text: str) -> dict:
    return {
        "id": "response",
        "created_at": 1,
        "model": "model",
        "object": "response",
        "output": [
            {
                "id": "message",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _verify_body() -> dict:
    assistant_response = _response("Here is an explanation.")
    return {
        **_seed_body(seed=7),
        "scenario": {
            "persona": PERSONAS[0],
            "probe_type": "general_educational",
            "theme": {"type": "local ecology", "description": "Learn about local ecology."},
            "locale": "en_US",
        },
        "response": assistant_response,
        "nemo_sim_result": {
            "conversation_messages": [
                {"role": "user", "content": "Teach me about local ecology."},
                {"role": "assistant", "content": "Here is an explanation."},
            ]
        },
        "invocations": [],
        "episode_interaction_protocol": "nemo_sim.ConversationLoop",
    }


def test_seed_session_resolves_replayable_scenario(tmp_path: Path) -> None:
    _write_personas(tmp_path)
    with TestClient(_app(tmp_path)) as client:
        first = client.post("/seed_session", json=_seed_body(seed=7, probe_type="general_open_ended"))
        second = client.post("/seed_session", json=_seed_body(seed=7, probe_type="general_open_ended"))

    assert first.status_code == 200
    assert first.json()["nemo_sim_context"] == second.json()["nemo_sim_context"]
    assert first.json()["nemo_sim_context"]["personas_dataset_version"] == "0.0.2"
    assert len(first.json()["nemo_sim_context"]["personas_source_sha256"]) == 64
    scenario = first.json()["scenario"]
    assert scenario["persona"]["first_name"] in {"Morgan", "Avery"}
    assert scenario["probe_type"] == "general_open_ended"
    assert scenario["goal"] == "Seek a practical recommendation about local food."
    assert scenario["personas_dataset_version"] == "0.0.2"
    assert len(scenario["personas_source_sha256"]) == 64


def test_probe_mix_deterministically_selects_enabled_probe(tmp_path: Path) -> None:
    _write_personas(tmp_path)
    with TestClient(_app(tmp_path, educational_only=True)) as client:
        response = client.post("/seed_session", json=_seed_body(seed=19))

    assert response.status_code == 200
    assert response.json()["nemo_sim_context"]["probe_type"] == "general_educational"
    assert response.json()["nemo_sim_context"]["theme"]["topic"] == "local ecology"


def test_sessions_keep_independent_resolved_contexts(tmp_path: Path) -> None:
    _write_personas(tmp_path)
    app = _app(tmp_path)
    with TestClient(app) as first, TestClient(app) as second:
        first_context = first.post(
            "/seed_session",
            json=_seed_body(seed=1, probe_type="general_open_ended"),
        ).json()["nemo_sim_context"]
        second_context = second.post(
            "/seed_session",
            json=_seed_body(seed=2, probe_type="general_educational"),
        ).json()["nemo_sim_context"]
        first_status = first.post("/episode_status", json={}).json()
        second_status = second.post("/episode_status", json={}).json()

    assert first_context["probe_type"] != second_context["probe_type"]
    assert first_status["state"]["probe_type"] == "general_open_ended"
    assert second_status["state"]["probe_type"] == "general_educational"


def test_startup_prepares_panel_then_reuses_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_personas(tmp_path)
    monkeypatch.setattr(
        "resources_servers.nemo_sim.app._sha256_file",
        lambda *_args, **_kwargs: "a" * 64,
    )
    _app(tmp_path)

    assert _source_path(tmp_path).with_suffix(".manifest.json").is_file()
    assert (tmp_path / "0.0.2" / "panels" / "en_US-n2-seed42.parquet").is_file()

    monkeypatch.setattr(
        "resources_servers.nemo_sim.app._sha256_file",
        lambda *_args, **_kwargs: pytest.fail("matching prepared manifests must be reused"),
    )
    _app(tmp_path)


def test_missing_pinned_dataset_fails_during_initialization(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="gym eval prepare --benchmark nemo_sim"):
        _app(tmp_path)


def test_seed_session_rejects_locale_not_initialized_at_startup(tmp_path: Path) -> None:
    _write_personas(tmp_path)
    body = _seed_body(seed=7)
    body["nemo_sim_sampling"]["locale"] = "pt_BR"
    with TestClient(_app(tmp_path)) as client:
        response = client.post("/seed_session", json=body)

    assert response.status_code == 422
    assert "was not initialized" in response.json()["detail"]


def test_verify_records_context_and_requires_both_participants(tmp_path: Path) -> None:
    _write_personas(tmp_path)
    with TestClient(_app(tmp_path)) as client:
        client.post("/seed_session", json=_seed_body(seed=7))
        verified = client.post("/verify", json=_verify_body()).json()
        incomplete_body = _verify_body()
        incomplete_body["nemo_sim_result"]["conversation_messages"] = [
            {"role": "user", "content": "Teach me about local ecology."}
        ]
        incomplete = client.post("/verify", json=incomplete_body).json()

    assert verified["reward"] == 1.0
    assert verified["scenario_completed"] is True
    assert verified["nemo_sim_context"]["seed"] == 7
    assert incomplete["reward"] == 0.0
    assert incomplete["scenario_completed"] is False
