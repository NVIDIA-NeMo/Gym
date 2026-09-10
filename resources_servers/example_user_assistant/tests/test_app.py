# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.example_user_assistant.app import (
    ExampleUserAssistantConfig,
    ExampleUserAssistantServer,
)


def _app() -> FastAPI:
    server = ExampleUserAssistantServer(
        config=ExampleUserAssistantConfig(
            host="127.0.0.1",
            port=12345,
            entrypoint="app.py",
            name="example_user_assistant",
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    return server.setup_webserver()


def _client() -> TestClient:
    return TestClient(_app())


def _response(text: str = "Here is a matching meal.") -> dict:
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
    return {
        "responses_create_params": {"input": "Help the customer."},
        "user_responses_create_params": {"input": "Find a vegetarian meal."},
        "response": _response(),
        "assistant_trajectory": [
            {
                "turn_index": 0,
                "participant": "assistant",
                "request": {"input": "Help the customer."},
                "response": _response(),
            }
        ],
        "user_trajectory": [],
        "episode_trajectory": [
            {
                "sequence": 0,
                "turn_index": 0,
                "kind": "response_item",
                "participant": "assistant",
                "data": _response()["output"][0],
            }
        ],
        "termination_reason": "recommendation_accepted",
        "turns_completed": 1,
    }


def test_user_state_is_visible_to_assistant_and_terminates_on_acceptance() -> None:
    with _client() as client:
        assert client.post("/seed_session", json={}).status_code == 200
        saved = client.post("/save_preference", json={"diet": "Vegetarian", "max_price": 20}).json()
        assert saved["preferences"] == {"diet": "vegetarian", "max_price": 20.0}

        assert client.post("/read_preferences", json={}).json()["preferences"] == saved["preferences"]
        status = client.post("/episode_status", json={}).json()
        assert status["terminated"] is False
        assert status["state"]["preferences"]["diet"] == "vegetarian"

        client.post("/recommend_meal", json={"name": "Vegetable curry", "diet": "vegetarian", "price": 16})
        accepted = client.post("/accept_recommendation", json={}).json()
        assert accepted["accepted"] is True
        status = client.post("/episode_status", json={}).json()
        assert status["terminated"] is True
        assert status["reason"] == "recommendation_accepted"

        verified = client.post("/verify", json=_verify_body()).json()
        assert verified["reward"] == 1.0
        assert verified["preference_satisfied"] is True


def test_verifier_rejects_a_recommendation_that_violates_persisted_state() -> None:
    with _client() as client:
        client.post("/seed_session", json={})
        client.post("/save_preference", json={"diet": "vegetarian", "max_price": 20})
        client.post("/recommend_meal", json={"name": "Steak", "diet": "omnivore", "price": 30})
        client.post("/accept_recommendation", json={})

        verified = client.post("/verify", json=_verify_body()).json()
        assert verified["reward"] == 0.0
        assert verified["preference_satisfied"] is False


def test_sessions_are_isolated() -> None:
    app = _app()
    with TestClient(app) as first, TestClient(app) as second:
        first.post("/seed_session", json={})
        second.post("/seed_session", json={})
        first.post("/save_preference", json={"diet": "vegetarian", "max_price": 20})

        assert first.post("/read_preferences", json={}).json()["preferences"]["diet"] == "vegetarian"
        assert second.post("/read_preferences", json={}).json()["preferences"] == {}
