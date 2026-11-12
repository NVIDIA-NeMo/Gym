# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.server_runtime.base_resources_server import (
    BaseSeedSessionResponse,
    BaseVerifyResponse,
    ResourcesSeedSessionResponse,
)
from nemo_gym.server_runtime.single_agent_turn_types import SingleAgentTurnResult


def test_empty_seed_response_preserves_legacy_wire_shape() -> None:
    assert BaseSeedSessionResponse().model_dump() == {}


def test_base_verify_response_preserves_legacy_extra_field_behavior() -> None:
    response = BaseVerifyResponse.model_validate(
        {
            "responses_create_params": {"input": "hi"},
            "response": NeMoGymResponse.model_construct(id="response", output=[]),
            "reward": 0.5,
            "request_only_field": "ignored",
        }
    )
    assert "request_only_field" not in response.model_dump()


def test_single_agent_turn_result_preserves_benchmark_fields() -> None:
    response = SingleAgentTurnResult.model_validate(
        {
            "responses_create_params": {"input": "hi"},
            "response": NeMoGymResponse.model_construct(id="response", output=[]),
            "reward": 0.5,
            "benchmark_diagnostic": "kept",
        }
    )
    assert response.model_dump()["benchmark_diagnostic"] == "kept"


def test_resources_seed_response_omits_unset_observation_without_changing_other_defaults() -> None:
    response = ResourcesSeedSessionResponse(resources_session_id="resources-session")
    expected = {"resources_session_id": "resources-session", "resources_tools": None, "sandbox_access": None}

    assert response.observation is None
    assert response.model_dump() == expected
    assert response.model_dump(mode="json") == expected
    assert json.loads(response.model_dump_json()) == expected
    restored = ResourcesSeedSessionResponse.model_validate_json(response.model_dump_json())
    assert "observation" not in restored.model_fields_set
    assert restored.model_dump() == expected


@pytest.mark.parametrize("observation", [None, "", " \nReset prompt\t", "Reset prompt"])
def test_resources_seed_response_accepts_an_optional_observation(observation: str | None) -> None:
    payload = {"resources_session_id": "resources-session", "observation": observation}
    response = ResourcesSeedSessionResponse.model_validate(payload)
    expected = {"resources_tools": None, "sandbox_access": None, **payload}

    assert response.model_dump() == expected
    assert response.model_dump(mode="json") == expected
    assert json.loads(response.model_dump_json()) == expected
    assert response.model_dump(exclude_unset=True) == payload
    assert json.loads(response.model_dump_json(exclude_unset=True)) == payload
    restored = ResourcesSeedSessionResponse.model_validate_json(response.model_dump_json())
    assert "observation" in restored.model_fields_set
    assert restored.observation == observation
    assert restored.model_dump() == expected


@pytest.mark.parametrize("fields", [{}, {"observation": None}, {"observation": ""}, {"observation": "Reset prompt"}])
def test_resources_seed_observation_serialization_respects_field_selection(fields: dict) -> None:
    response = ResourcesSeedSessionResponse(resources_session_id="resources-session", **fields)
    expected = {"resources_session_id": "resources-session", "resources_tools": None, "sandbox_access": None}

    assert response.model_dump(exclude={"observation"}) == expected
    assert json.loads(response.model_dump_json(exclude={"observation"})) == expected
    assert response.model_dump(include={"observation"}) == fields
    assert json.loads(response.model_dump_json(include={"observation"})) == fields
    non_null = {"resources_session_id": "resources-session"}
    if fields.get("observation") is not None:
        non_null.update(fields)
    assert response.model_dump(exclude_none=True) == non_null


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_resources_seed_response_keeps_its_object_schema(mode: str) -> None:
    schema = ResourcesSeedSessionResponse.model_json_schema(mode=mode)

    assert set(schema["properties"]) == {"resources_session_id", "resources_tools", "sandbox_access", "observation"}
    assert schema["required"] == ["resources_session_id"]
    assert schema["additionalProperties"] is False


def test_resources_seed_response_keeps_its_http_response_schema() -> None:
    app = FastAPI()

    @app.post("/seed_session")
    def seed_session() -> ResourcesSeedSessionResponse:
        return ResourcesSeedSessionResponse(resources_session_id="resources-session")

    with TestClient(app) as client:
        document = client.get("/openapi.json").json()
    response_schema = document["paths"]["/seed_session"]["post"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ]
    schema = document["components"]["schemas"][response_schema["$ref"].rsplit("/", 1)[1]]

    assert set(schema["properties"]) == {"resources_session_id", "resources_tools", "sandbox_access", "observation"}
    assert schema["required"] == ["resources_session_id"]
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    "fields",
    [{"observation": 42}, {"observation": True}, {"observation": []}, {"observation": {}}, {"unknown": "value"}],
)
def test_resources_seed_response_keeps_observation_types_and_extra_fields_strict(fields: dict) -> None:
    with pytest.raises(ValidationError):
        ResourcesSeedSessionResponse.model_validate({"resources_session_id": "resources-session", **fields})
