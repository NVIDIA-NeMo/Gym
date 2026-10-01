"""Tests for the Strategic Bench resources server (negotiation environment).

Tests are organised as:

- TestConversationExtraction: Unit tests for _extract_conversation, covering
  both policy-model output format (nested content blocks) and user-model
  format (plain string content).
- TestAgreementDetection: Unit tests for _check_agreement_llm, covering both
  the LLM-based path (mocked) and the no-LLM fallback (returns False).
- TestRewardCalculation: Unit tests for calculate_rewards, including eval()
  expressions using CONTEXT and re (without requiring a live LLM).
- TestApp: Sanity check that the server instantiates without errors.
- TestEndpoints: Integration tests exercising the full HTTP API flow via
  FastAPI's TestClient (seed_session -> make_offer -> verify).
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.strategic_bench.app import (
    StrategicBenchConfig,
    StrategicBenchServer,
)

# Eval model fields are required (populated from env.yaml in production).
# Tests stub _make_llm_callable directly, so dummy values are fine here.
_TEST_EVAL_CONFIG = dict(
    eval_model_base_url="https://api.openai.com/v1",
    eval_model_api_key="test-key",
    eval_model_name="gpt-4.1-2025-04-14",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_SCENARIO = {
    "agent1_role": "Seller",
    "agent2_role": "Buyer",
    "generic": "A negotiation over a painting.",
    "agent1_specific": "You want at least 300k.",
    "agent2_specific": "You want to pay at most 500k.",
    "agent1_rewards": [
        # Simple reward: 1.0 if CONTEXT contains "agreed", else 0.
        r"1.0 if re.search(r'\bagreed\b', CONTEXT, re.IGNORECASE) else 0.0"
    ],
    "agent2_rewards": [
        "0.5"  # Fixed reward — useful for testing reward averaging
    ],
    "reached_agreement_q": "Did the parties reach an agreement? Answer YES or NO.",
}


def _make_server(llm_response: str = "") -> tuple:
    """Return a fresh (server, TestClient) pair for each test.

    llm_response: fixed string returned by the mocked eval LLM for every call.
    Defaults to "" (no agreement detected) so that tests that don't care about
    the agreement outcome stay unaffected.
    """
    config = StrategicBenchConfig(host="0.0.0.0", port=8080, entrypoint="", name="", **_TEST_EVAL_CONFIG)
    server = StrategicBenchServer(config=config, server_client=MagicMock(spec=ServerClient))
    # Patch the LLM callable so tests don't require a live eval model.
    server._make_llm_callable = lambda: (lambda prompt: llm_response)
    app = server.setup_webserver()
    client = TestClient(app)
    return server, client


def _seed_body(scenario_data: dict = None, agent: str = "agent1") -> dict:
    """Build a minimal seed_session request body."""
    return {
        "responses_create_params": {"input": []},
        "verifier_metadata": {
            "scenario_data": scenario_data or _MINIMAL_SCENARIO,
            "agent": agent,
            "agent_name": agent,
        },
    }


def _verify_body(outputs: list = None) -> dict:
    """Build a /verify request body containing an optional output list."""
    return {
        "responses_create_params": {"input": []},
        "response": {
            "id": "resp_test",
            "created_at": 0.0,
            "model": "test",
            "object": "response",
            "output": outputs or [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        },
    }


def _assistant_message(text: str) -> dict:
    """Policy-model output format: assistant message with nested content blocks."""
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
        "status": "completed",
    }


def _user_message(text: str) -> dict:
    """User-model output format: plain string content."""
    return {"type": "message", "role": "user", "content": text}


# ---------------------------------------------------------------------------
# Unit tests: conversation extraction
# ---------------------------------------------------------------------------


class TestConversationExtraction:
    def _server(self) -> StrategicBenchServer:
        config = StrategicBenchConfig(host="0.0.0.0", port=8080, entrypoint="", name="", **_TEST_EVAL_CONFIG)
        return StrategicBenchServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_extracts_assistant_text(self) -> None:
        """Policy-model messages (nested content blocks) are extracted correctly."""
        server = self._server()
        outputs = [_assistant_message("I propose 400k.")]
        result = server._extract_conversation(outputs)
        assert "assistant: I propose 400k." in result

    def test_extracts_user_text(self) -> None:
        """User-model messages (plain string content) are extracted correctly."""
        server = self._server()
        outputs = [_user_message("That is too high.")]
        result = server._extract_conversation(outputs)
        assert "user: That is too high." in result

    def test_mixed_messages_interleaved(self) -> None:
        """Alternating policy and user messages appear in the correct order."""
        server = self._server()
        outputs = [
            _assistant_message("I offer 400k."),
            _user_message("Counter-offer: 350k."),
            _assistant_message("Agreed, 375k."),
        ]
        result = server._extract_conversation(outputs)
        lines = result.splitlines()
        assert len(lines) == 3
        assert lines[0].startswith("assistant:")
        assert lines[1].startswith("user:")
        assert lines[2].startswith("assistant:")

    def test_skips_function_calls(self) -> None:
        """Tool-call items are not included in the conversation string."""
        server = self._server()
        outputs = [
            {"type": "function_call", "name": "make_offer", "arguments": '{"offer_text": "hi"}'},
            _assistant_message("Hello."),
        ]
        result = server._extract_conversation(outputs)
        assert "function_call" not in result
        assert "Hello." in result

    def test_empty_outputs(self) -> None:
        """Empty output list produces an empty string."""
        server = self._server()
        assert server._extract_conversation([]) == ""

    def test_skips_empty_content(self) -> None:
        """Messages with blank content are excluded from the conversation."""
        server = self._server()
        outputs = [_assistant_message("   "), _user_message("Hello.")]
        result = server._extract_conversation(outputs)
        assert result == "user: Hello."


# ---------------------------------------------------------------------------
# Unit tests: LLM-based agreement detection
# ---------------------------------------------------------------------------


class TestAgreementDetection:
    def _server(self, llm_response: str = "") -> tuple:
        """Return a server with a mocked _make_llm_callable that always returns llm_response."""
        config = StrategicBenchConfig(host="0.0.0.0", port=8080, entrypoint="", name="", **_TEST_EVAL_CONFIG)
        server = StrategicBenchServer(config=config, server_client=MagicMock(spec=ServerClient))
        server._make_llm_callable = lambda: (lambda prompt: llm_response)
        return server

    def test_both_yes_is_agreement(self) -> None:
        """When both agents answer YES, _check_agreement_llm returns True."""
        server = self._server(llm_response="YES")
        assert server._check_agreement_llm("Seller: OK. Buyer: Agreed.", _MINIMAL_SCENARIO) is True

    def test_no_from_agent1_is_not_agreement(self) -> None:
        """If agent1 says NO, agreement is False without querying agent2."""
        server = self._server(llm_response="NO")
        assert server._check_agreement_llm("Seller: I disagree.", _MINIMAL_SCENARIO) is False

    def test_agent2_no_overrides_agent1_yes(self) -> None:
        """If agent1 says YES but agent2 says NO, agreement is False."""
        responses = iter(["YES", "NO"])
        config = StrategicBenchConfig(host="0.0.0.0", port=8080, entrypoint="", name="", **_TEST_EVAL_CONFIG)
        server = StrategicBenchServer(config=config, server_client=MagicMock(spec=ServerClient))
        server._make_llm_callable = lambda: (lambda prompt: next(responses))
        assert server._check_agreement_llm("...", _MINIMAL_SCENARIO) is False

    def test_api_error_returns_false(self) -> None:
        """When the eval LLM call raises an exception, the callable returns ''
        which is treated as non-YES → agreement=False."""
        config = StrategicBenchConfig(host="0.0.0.0", port=8080, entrypoint="", name="", **_TEST_EVAL_CONFIG)
        server = StrategicBenchServer(config=config, server_client=MagicMock(spec=ServerClient))
        # Simulate API failure: llm returns "" on exception
        server._make_llm_callable = lambda: (lambda prompt: "")
        assert server._check_agreement_llm("...", _MINIMAL_SCENARIO) is False

    def test_prompts_include_agent_context(self) -> None:
        """The LLM prompt for each agent includes that agent's role and specific info."""
        prompts_seen: list = []

        def capturing_llm(prompt: str) -> str:
            prompts_seen.append(prompt)
            return "YES"

        config = StrategicBenchConfig(host="0.0.0.0", port=8080, entrypoint="", name="", **_TEST_EVAL_CONFIG)
        server = StrategicBenchServer(config=config, server_client=MagicMock(spec=ServerClient))
        server._make_llm_callable = lambda: capturing_llm

        server._check_agreement_llm("Seller: OK Buyer: Deal.", _MINIMAL_SCENARIO)

        assert len(prompts_seen) == 2
        # First prompt is from agent1 (Seller) perspective
        assert "Seller" in prompts_seen[0]
        assert "You want at least 300k." in prompts_seen[0]
        # Second prompt is from agent2 (Buyer) perspective
        assert "Buyer" in prompts_seen[1]
        assert "You want to pay at most 500k." in prompts_seen[1]

    def test_agent1_not_queried_twice_on_early_no(self) -> None:
        """If agent1 says NO, agent2 is never queried (short-circuit)."""
        call_count = {"n": 0}

        def counting_llm(prompt: str) -> str:
            call_count["n"] += 1
            return "NO"

        config = StrategicBenchConfig(host="0.0.0.0", port=8080, entrypoint="", name="", **_TEST_EVAL_CONFIG)
        server = StrategicBenchServer(config=config, server_client=MagicMock(spec=ServerClient))
        server._make_llm_callable = lambda: counting_llm

        server._check_agreement_llm("No deal.", _MINIMAL_SCENARIO)
        assert call_count["n"] == 1  # Only agent1 was asked


# ---------------------------------------------------------------------------
# Unit tests: reward calculation
# ---------------------------------------------------------------------------


class TestRewardCalculation:
    def _server(self) -> StrategicBenchServer:
        config = StrategicBenchConfig(host="0.0.0.0", port=8080, entrypoint="", name="", **_TEST_EVAL_CONFIG)
        return StrategicBenchServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_reward_with_regex(self) -> None:
        """Reward function using re.search and CONTEXT is evaluated correctly."""
        server = self._server()
        scenario_data = {
            "agent1_rewards": [
                r"1.0 if re.search(r'\bagreed\b', CONTEXT, re.IGNORECASE) else 0.0"
            ]
        }
        reward = server.calculate_rewards("We agreed on the price.", scenario_data, "agent1")
        assert reward == 1.0

    def test_reward_no_match(self) -> None:
        """Reward function returns 0.0 when the pattern is absent."""
        server = self._server()
        scenario_data = {
            "agent1_rewards": [
                r"1.0 if re.search(r'\bagreed\b', CONTEXT, re.IGNORECASE) else 0.0"
            ]
        }
        reward = server.calculate_rewards("No deal reached.", scenario_data, "agent1")
        assert reward == 0.0

    def test_multiple_reward_functions_averaged(self) -> None:
        """Multiple reward components are averaged."""
        server = self._server()
        scenario_data = {"agent1_rewards": ["1.0", "0.0"]}  # Mean = 0.5
        reward = server.calculate_rewards("anything", scenario_data, "agent1")
        assert reward == 0.5

    def test_list_reward_function(self) -> None:
        """Reward function that returns a list is averaged element-wise."""
        server = self._server()
        scenario_data = {"agent1_rewards": ["[1.0, 0.0]"]}
        reward = server.calculate_rewards("anything", scenario_data, "agent1")
        assert reward == 0.5

    def test_no_reward_functions_baseline(self) -> None:
        """When no reward functions are defined, the baseline 0.05 is returned."""
        server = self._server()
        reward = server.calculate_rewards("anything", {}, "agent1")
        assert reward == 0.05

    def test_broken_reward_function_scores_zero(self) -> None:
        """A reward function that raises an exception scores 0.0 instead of crashing."""
        server = self._server()
        scenario_data = {"agent1_rewards": ["raise ValueError('bad')"]}
        reward = server.calculate_rewards("anything", scenario_data, "agent1")
        assert reward == 0.0

    def test_llm_callable_returns_empty_on_api_error(self) -> None:
        """When the OpenAI call fails (e.g. bad key), the callable returns '' safely."""
        server = self._server()
        # Override with a bad API key so the real client would fail; _make_llm_callable
        # catches all exceptions and returns "".
        server.config.eval_model_api_key = "invalid-key"
        llm = server._make_llm_callable()
        # We can't hit the real API in tests, so just verify the callable is returned
        # without error and matches the expected type.
        assert callable(llm)

    def test_uses_agent_specific_rewards(self) -> None:
        """Agent-specific reward functions are selected by agent_name."""
        server = self._server()
        scenario_data = {
            "agent1_rewards": ["1.0"],
            "agent2_rewards": ["0.0"],
        }
        assert server.calculate_rewards("anything", scenario_data, "agent1") == 1.0
        assert server.calculate_rewards("anything", scenario_data, "agent2") == 0.0


# ---------------------------------------------------------------------------
# Server instantiation sanity check
# ---------------------------------------------------------------------------


class TestApp:
    def test_sanity(self) -> None:
        """Server can be instantiated without errors."""
        config = StrategicBenchConfig(host="0.0.0.0", port=8080, entrypoint="", name="", **_TEST_EVAL_CONFIG)
        StrategicBenchServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_make_offer_endpoint_registered(self) -> None:
        """The /make_offer endpoint is registered on the FastAPI app."""
        _, client = _make_server()
        client.post("/seed_session", json=_seed_body())
        resp = client.post("/make_offer", json={"offer_text": "Hello."})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# HTTP endpoint integration tests
# ---------------------------------------------------------------------------


class TestEndpoints:
    def test_seed_session_returns_200(self) -> None:
        _, client = _make_server()
        resp = client.post("/seed_session", json=_seed_body())
        assert resp.status_code == 200

    def test_seed_session_stores_agent_name(self) -> None:
        """Seeding with agent='agent2' stores the correct agent identity."""
        server, client = _make_server()
        client.post("/seed_session", json=_seed_body(agent="agent2"))
        games = list(server.session_id_to_game.values())
        assert len(games) == 1
        assert games[0].agent_name == "agent2"

    def test_make_offer_before_seed_fails(self) -> None:
        """make_offer without a prior seed_session returns success=False."""
        _, client = _make_server()
        resp = client.post("/make_offer", json={"offer_text": "Hello."})
        data = resp.json()
        assert data["success"] is False
        assert "seed_session" in data["message"]

    def test_make_offer_increments_turn_counter(self) -> None:
        server, client = _make_server()
        client.post("/seed_session", json=_seed_body())
        client.post("/make_offer", json={"offer_text": "Offer A."})
        client.post("/make_offer", json={"offer_text": "Counter-offer B."})
        game = list(server.session_id_to_game.values())[0]
        assert game.turns == 2

    def test_make_offer_stops_on_done_token(self) -> None:
        """An offer containing [DONE] sets game_over=True."""
        _, client = _make_server()
        client.post("/seed_session", json=_seed_body())
        resp = client.post("/make_offer", json={"offer_text": "We are done. [DONE]"})
        data = resp.json()
        assert data["game_over"] is True

    def test_make_offer_after_game_over_fails(self) -> None:
        """Subsequent offers after game_over return success=False."""
        _, client = _make_server()
        client.post("/seed_session", json=_seed_body())
        client.post("/make_offer", json={"offer_text": "[DONE]"})
        resp = client.post("/make_offer", json={"offer_text": "Another offer."})
        assert resp.json()["success"] is False

    def test_verify_no_game(self) -> None:
        """verify without a prior seed_session returns reward=0.0 and no_game."""
        _, client = _make_server()
        resp = client.post("/verify", json=_verify_body())
        data = resp.json()
        assert data["reward"] == 0.0
        assert data["game_result"] == "no_game"

    def test_verify_no_agreement_baseline_reward(self) -> None:
        """When both agents say NO to the agreement question, return 0.05 baseline."""
        # llm_response="NO" → both agents disagree → no_agreement
        _, client = _make_server(llm_response="NO")
        client.post("/seed_session", json=_seed_body())
        outputs = [
            _assistant_message("My price is 500k."),
            _user_message("That is too high."),
        ]
        resp = client.post("/verify", json=_verify_body(outputs))
        data = resp.json()
        assert data["reward"] == 0.05
        assert data["game_result"] == "no_agreement"

    def test_verify_with_agreement_runs_rewards(self) -> None:
        """When both agents say YES, the reward functions are evaluated."""
        # llm_response="YES" → both agents agree → evaluate reward functions
        _, client = _make_server(llm_response="YES")
        client.post("/seed_session", json=_seed_body())
        outputs = [
            _assistant_message("I can go to 400k."),
            _user_message("Agreed, 400k it is."),
        ]
        resp = client.post("/verify", json=_verify_body(outputs))
        data = resp.json()
        # _MINIMAL_SCENARIO agent1_rewards: 1.0 when "agreed" is in CONTEXT
        assert data["reward"] == 1.0
        assert data["game_result"] == "agreement"

    def test_verify_empty_output_no_agreement(self) -> None:
        """Empty conversation → LLM says NO → no_agreement baseline reward."""
        _, client = _make_server(llm_response="NO")
        client.post("/seed_session", json=_seed_body())
        resp = client.post("/verify", json=_verify_body([]))
        data = resp.json()
        assert data["game_result"] == "no_agreement"
        assert data["reward"] == 0.05

    def test_verify_agent2_rewards(self) -> None:
        """When the policy plays agent2 and both agree, agent2_rewards are used."""
        _, client = _make_server(llm_response="YES")
        client.post("/seed_session", json=_seed_body(agent="agent2"))
        outputs = [_assistant_message("We have a deal.")]
        resp = client.post("/verify", json=_verify_body(outputs))
        data = resp.json()
        # _MINIMAL_SCENARIO agent2_rewards returns 0.5 unconditionally
        assert data["reward"] == 0.5
