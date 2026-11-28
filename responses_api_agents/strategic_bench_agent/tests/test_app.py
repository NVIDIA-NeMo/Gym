"""Tests for the Strategic Bench agent server.

Focuses on:
- Config validation (required fields, defaults).
- get_agent_prompt: system prompt construction for each agent role and
  a representative sample of personality modes.
- _generate_user_response: correct text extraction from both the nested
  OpenAI Responses API format and the plain-string fallback format; None
  returned when the model produces no usable output.

All model/resources server HTTP calls are mocked via ServerClient.
"""

import json
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.strategic_bench_agent.app import (
    StrategicBenchAgent,
    StrategicBenchAgentConfig,
    StrategicBenchAgentRunRequest,
    get_agent_prompt,
)


# ---------------------------------------------------------------------------
# Helpers: config and request factories
# ---------------------------------------------------------------------------

_SAMPLE_SCENARIO = {
    "agent1_role": "Seller",
    "agent2_role": "Buyer",
    "generic": "A sale negotiation.",
    "agent1_specific": "Sell high.",
    "agent2_specific": "Buy low.",
    "agent1_rewards": ["1.0"],
    "agent2_rewards": ["0.5"],
}


def _make_config(**overrides) -> StrategicBenchAgentConfig:
    defaults = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="test_agent",
        resources_server={"type": "resources_servers", "name": "strategic_bench"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        user_model_server={"type": "responses_api_models", "name": "user_model"},
        max_turns=5,
        user_model_system_prompt="You are the opposing negotiator.",
    )
    return StrategicBenchAgentConfig(**(defaults | overrides))


def _make_run_request(**overrides) -> StrategicBenchAgentRunRequest:
    defaults = dict(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "Let us start."}],
        )
    )
    return StrategicBenchAgentRunRequest(**(defaults | overrides))


def _text_response(text: str) -> dict:
    """Simulate an OpenAI Responses API message with nested content blocks."""
    return {
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ]
    }


def _plain_string_response(text: str) -> dict:
    """Simulate a message where content is a plain string (user-model format)."""
    return {
        "output": [
            {
                "id": "msg_2",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": text,
            }
        ]
    }


def _empty_response() -> dict:
    return {"output": []}


# ---------------------------------------------------------------------------
# get_agent_prompt tests
# ---------------------------------------------------------------------------


class TestGetAgentPrompt:
    def test_includes_both_roles(self) -> None:
        prompt = get_agent_prompt(_SAMPLE_SCENARIO, "agent1", "")
        assert "Seller" in prompt
        assert "Buyer" in prompt

    def test_includes_agent_specific_info(self) -> None:
        prompt = get_agent_prompt(_SAMPLE_SCENARIO, "agent1", "")
        assert "Sell high." in prompt

    def test_agent2_specific_info(self) -> None:
        prompt = get_agent_prompt(_SAMPLE_SCENARIO, "agent2", "")
        assert "Buy low." in prompt

    def test_includes_turn_limit(self) -> None:
        prompt = get_agent_prompt(_SAMPLE_SCENARIO, "agent1", "", convo_length_limit=10)
        assert "20" in prompt  # 10 * 2

    def test_personality_competitive(self) -> None:
        prompt = get_agent_prompt(_SAMPLE_SCENARIO, "agent1", "competitive")
        assert "competitive" in prompt.lower()

    def test_personality_syncophantic(self) -> None:
        prompt = get_agent_prompt(_SAMPLE_SCENARIO, "agent1", "syncophantic")
        assert "eager to please" in prompt.lower()

    def test_personality_balanced(self) -> None:
        prompt = get_agent_prompt(_SAMPLE_SCENARIO, "agent1", "balanced")
        assert "balance" in prompt.lower()

    def test_personality_risk_averse(self) -> None:
        prompt = get_agent_prompt(_SAMPLE_SCENARIO, "agent1", "risk_averse")
        assert "downside risk" in prompt.lower()

    def test_empty_personality_no_extra_text(self) -> None:
        prompt_empty = get_agent_prompt(_SAMPLE_SCENARIO, "agent1", "")
        prompt_neutral = get_agent_prompt(_SAMPLE_SCENARIO, "agent1", "nonexistent_mode")
        # Both should produce the same base prompt (no personality suffix)
        assert prompt_empty == prompt_neutral

    def test_empty_scenario_data_fallback(self) -> None:
        prompt = get_agent_prompt({}, "agent1", "")
        assert "negotiator" in prompt.lower()


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_sanity(self) -> None:
        config = _make_config()
        StrategicBenchAgent(config=config, server_client=MagicMock(spec=ServerClient))

    def test_max_steps_per_turn_default_none(self) -> None:
        config = _make_config()
        assert config.max_steps_per_turn is None

    def test_stop_token_default_none(self) -> None:
        config = _make_config()
        assert config.user_model_stop_token is None

    def test_custom_stop_token(self) -> None:
        config = _make_config(user_model_stop_token="[DONE]")
        assert config.user_model_stop_token == "[DONE]"


# ---------------------------------------------------------------------------
# _generate_user_response tests
# ---------------------------------------------------------------------------


class TestGenerateUserResponse:
    """Test the user model interaction method directly.

    The call sequence for server_client.post in _generate_user_response is:
      1. POST to user_model_server /v1/responses  (get opponent's response)

    All HTTP calls are mocked.
    """

    def _make_agent(self, **config_overrides) -> StrategicBenchAgent:
        config = _make_config(**config_overrides)
        return StrategicBenchAgent(config=config, server_client=MagicMock(spec=ServerClient))

    async def _call_generate(
        self, agent: StrategicBenchAgent, body=None, opponent_prompt: str = "You are a buyer."
    ):
        if body is None:
            body = _make_run_request()
        return await agent._generate_user_response(
            body=body,
            original_input=[{"role": "user", "content": "Start."}],
            all_turn_outputs=[],
            opponent_system_prompt=opponent_prompt,
            cookies={},
        )

    def _mock_user_response(self, agent: StrategicBenchAgent, response_data: dict) -> None:
        """Set up server_client.post to return the given JSON."""
        mock_resp = AsyncMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.cookies = {}
        agent.server_client.post.return_value = mock_resp

    async def test_extracts_text_from_nested_content(self) -> None:
        """Text inside content[].output_text is returned correctly."""
        agent = self._make_agent()
        self._mock_user_response(agent, _text_response("Counter-offer: 400k."))
        result = await self._call_generate(agent)
        assert result == "Counter-offer: 400k."

    async def test_extracts_plain_string_content(self) -> None:
        """Plain string content (user-model format) is returned correctly."""
        agent = self._make_agent()
        self._mock_user_response(agent, _plain_string_response("I accept your offer."))
        result = await self._call_generate(agent)
        assert result == "I accept your offer."

    async def test_returns_none_on_empty_output(self) -> None:
        """When the user model produces no output, None is returned."""
        agent = self._make_agent()
        self._mock_user_response(agent, _empty_response())
        result = await self._call_generate(agent)
        assert result is None

    async def test_strips_whitespace(self) -> None:
        """Leading/trailing whitespace in the model's text is stripped."""
        agent = self._make_agent()
        self._mock_user_response(agent, _text_response("   Hello.   "))
        result = await self._call_generate(agent)
        assert result == "Hello."

    async def test_opponent_system_prompt_is_first_message(self) -> None:
        """The opponent system prompt is the first message sent to the user model."""
        agent = self._make_agent()
        self._mock_user_response(agent, _text_response("Done."))
        await self._call_generate(agent, opponent_prompt="You are a tough buyer.")

        call_kwargs = agent.server_client.post.call_args
        sent_input = call_kwargs.kwargs["json"]["input"]
        assert sent_input[0]["role"] == "system"
        assert "tough buyer" in sent_input[0]["content"]

    async def test_policy_system_prompt_stripped(self) -> None:
        """The policy model's system/developer prompt is not included in user model input."""
        agent = self._make_agent()
        self._mock_user_response(agent, _text_response("Ok."))

        body = _make_run_request(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[
                    {"role": "developer", "content": "You are the policy seller."},
                    {"role": "user", "content": "Let's negotiate."},
                ]
            )
        )
        await self._call_generate(agent, body=body)

        call_kwargs = agent.server_client.post.call_args
        sent_input = call_kwargs.kwargs["json"]["input"]
        roles = [m.get("role") if isinstance(m, dict) else getattr(m, "role", None) for m in sent_input]
        assert "developer" not in roles

    async def test_turn_history_appended(self) -> None:
        """Previous turn outputs are appended to the user model's input."""
        agent = self._make_agent()
        self._mock_user_response(agent, _text_response("Ok."))

        prior_output = {"type": "message", "role": "assistant", "content": "My offer is 450k."}
        await agent._generate_user_response(
            body=_make_run_request(),
            original_input=[],
            all_turn_outputs=[prior_output],
            opponent_system_prompt="You are the buyer.",
            cookies={},
        )

        call_kwargs = agent.server_client.post.call_args
        sent_input = call_kwargs.kwargs["json"]["input"]
        # The prior output should appear somewhere in the input
        assert any(m == prior_output for m in sent_input)
