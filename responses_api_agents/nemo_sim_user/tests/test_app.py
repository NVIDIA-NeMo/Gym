# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nemo_sim_user import app
from responses_api_agents.nemo_sim_user.app import NeMoSimUserAgent, NeMoSimUserAgentConfig


PERSONA = {
    "first_name": "Morgan",
    "last_name": "Lee",
    "age": 42,
    "occupation": "building_inspector",
}


def _agent() -> NeMoSimUserAgent:
    return NeMoSimUserAgent(
        config=NeMoSimUserAgentConfig(
            host="",
            port=0,
            entrypoint="app.py",
            name="nemo_sim_user",
            resources_server=ResourcesServerRef(type="resources_servers", name="environment"),
            model_server=ModelServerRef(type="responses_api_models", name="model"),
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def _params(raw_context: str | None = None) -> NeMoGymResponseCreateParamsNonStreaming:
    context = raw_context or json.dumps(
        {
            "locale": "en_US",
            "goal": "Find an affordable vegetarian dinner.",
            "persona": PERSONA,
        }
    )
    return NeMoGymResponseCreateParamsNonStreaming(
        input="Wait for the assistant to speak.",
        metadata={"nemo_sim": context, "trace": "keep"},
    )


def test_persona_enriches_user_input_and_private_metadata_is_removed(monkeypatch) -> None:
    profile = {"patience": 0.5}
    api = {
        "compute_behavioral_profile": MagicMock(return_value=profile),
        "compute_disclosure_style": MagicMock(return_value="incremental"),
        "compute_user_interaction_style": MagicMock(return_value="impatient"),
        "format_behavioral_profile_for_prompt": MagicMock(return_value="<BEHAVIOR>impatient</BEHAVIOR>"),
        "format_disclosure_instructions": MagicMock(return_value="<DISCLOSURE>incremental</DISCLOSURE>"),
        "format_interaction_style_instructions": MagicMock(return_value="<STYLE>impatient</STYLE>"),
        "format_persona_for_prompt": MagicMock(return_value="Name: Morgan Lee"),
        "get_conversation_language": MagicMock(return_value="English"),
    }
    monkeypatch.setattr(app, "_nemo_sim_api", lambda: api)
    original = _params()

    prepared = _agent().prepare_response_params(original)

    assert original.input == "Wait for the assistant to speak."
    assert "nemo_sim" in original.metadata
    assert prepared.input[0].role == "developer"
    assert "Find an affordable vegetarian dinner." in prepared.input[0].content
    assert "Name: Morgan Lee" in prepared.input[0].content
    assert "<BEHAVIOR>impatient</BEHAVIOR>" in prepared.input[0].content
    assert prepared.input[-1].content == "Wait for the assistant to speak."
    assert prepared.metadata == {"trace": "keep"}
    api["compute_behavioral_profile"].assert_called_once_with(PERSONA, locale="en_US")


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, "JSON-encoded"),
        ({"nemo_sim": "not-json"}, "valid JSON"),
        ({"nemo_sim": "[]"}, "decode to an object"),
        ({"nemo_sim": "{}"}, "nemo_sim.persona"),
    ],
)
def test_invalid_persona_metadata_is_rejected(metadata: dict[str, str], message: str) -> None:
    params = NeMoGymResponseCreateParamsNonStreaming(input="Wait.", metadata=metadata)

    with pytest.raises(ValueError, match=message):
        _agent().prepare_response_params(params)


def test_agent_exposes_only_the_responses_api() -> None:
    paths = {route.path for route in _agent().setup_webserver().routes}
    assert "/v1/responses" in paths
    assert "/run" not in paths
