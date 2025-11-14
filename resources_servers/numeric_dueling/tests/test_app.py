# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from httpx import Cookies

from nemo_gym.server_utils import ServerClient
from resources_servers.numeric_dueling.app import (
    BustRuleType,
    GameConfig,
    NumericDuelingResourcesServer,
    NumericDuelingResourcesServerConfig,
    WinRuleType,
    resolve_round,
)


class TestResolveRound:
    """Unit tests for resolve_round() function with deterministic R values."""

    def test_standard_bust_highest_player_wins(self):
        """Standard Bust + Highest: Player has higher number and both survive."""
        config = GameConfig(bust_rule=BustRuleType.STANDARD, win_rule=WinRuleType.HIGHEST, max_number=100)
        result = resolve_round(60, 50, 70, config, 1)

        assert result.player_choice == 60
        assert result.opponent_choice == 50
        assert result.random_number == 70
        assert not result.player_busted
        assert not result.opponent_busted
        assert result.player_won == True
        assert result.player_points == 1.0
        assert result.opponent_points == 0.0

    def test_standard_bust_highest_opponent_wins(self):
        """Standard Bust + Highest: Opponent has higher number and both survive."""
        config = GameConfig(bust_rule=BustRuleType.STANDARD, win_rule=WinRuleType.HIGHEST, max_number=100)
        result = resolve_round(40, 60, 70, config, 1)

        assert not result.player_busted
        assert not result.opponent_busted
        assert result.player_won == False
        assert result.player_points == 0.0
        assert result.opponent_points == 1.0

    def test_standard_bust_highest_player_busts(self):
        """Standard Bust + Highest: Player busts, opponent wins."""
        config = GameConfig(bust_rule=BustRuleType.STANDARD, win_rule=WinRuleType.HIGHEST, max_number=100)
        result = resolve_round(80, 50, 60, config, 1)

        assert result.player_busted == True
        assert not result.opponent_busted
        assert result.player_won == False
        assert result.player_points == 0.0
        assert result.opponent_points == 1.0

    def test_standard_bust_highest_both_bust(self):
        """Standard Bust + Highest: Both players bust."""
        config = GameConfig(bust_rule=BustRuleType.STANDARD, win_rule=WinRuleType.HIGHEST, max_number=100)
        result = resolve_round(80, 90, 50, config, 1)

        assert result.player_busted == True
        assert result.opponent_busted == True
        assert result.player_points == 0.0
        assert result.opponent_points == 0.0

    def test_standard_bust_closest_player_closer(self):
        """Standard Bust + Closest: Player is closer to R."""
        config = GameConfig(bust_rule=BustRuleType.STANDARD, win_rule=WinRuleType.CLOSEST, max_number=100)
        result = resolve_round(60, 40, 65, config, 1)

        assert not result.player_busted
        assert not result.opponent_busted
        assert result.player_won == True
        assert result.player_points == 1.0
        assert result.opponent_points == 0.0

    def test_standard_bust_closest_player_over_r(self):
        """Standard Bust + Closest: Player over R is disqualified even if closer."""
        config = GameConfig(bust_rule=BustRuleType.STANDARD, win_rule=WinRuleType.CLOSEST, max_number=100)
        result = resolve_round(65, 40, 60, config, 1)

        assert result.player_busted == True
        assert not result.opponent_busted
        assert result.player_won == False
        assert result.player_points == 0.0
        assert result.opponent_points == 1.0

    def test_standard_bust_closest_both_over_r(self):
        """Standard Bust + Closest: Both over R means no winner."""
        config = GameConfig(bust_rule=BustRuleType.STANDARD, win_rule=WinRuleType.CLOSEST, max_number=100)
        result = resolve_round(70, 80, 50, config, 1)

        assert result.player_busted == True
        assert result.opponent_busted == True
        assert result.player_points == 0.0
        assert result.opponent_points == 0.0

    def test_standard_bust_cumulative_both_survive(self):
        """Standard Bust + Cumulative: Both survive, get proportional points."""
        config = GameConfig(bust_rule=BustRuleType.STANDARD, win_rule=WinRuleType.CUMULATIVE, max_number=100)
        result = resolve_round(30, 70, 80, config, 1)

        assert not result.player_busted
        assert not result.opponent_busted
        assert result.player_points == pytest.approx(0.3)
        assert result.opponent_points == pytest.approx(0.7)
        assert result.player_won == False  # Opponent has more points

    def test_standard_bust_cumulative_one_busts(self):
        """Standard Bust + Cumulative: One busts, other gets proportional points."""
        config = GameConfig(bust_rule=BustRuleType.STANDARD, win_rule=WinRuleType.CUMULATIVE, max_number=100)
        result = resolve_round(40, 90, 50, config, 1)

        assert not result.player_busted
        assert result.opponent_busted == True
        assert result.player_points == pytest.approx(0.4)
        assert result.opponent_points == 0.0
        assert result.player_won == True

    def test_soft_bust_penalty_calculation(self):
        """Soft Bust: Penalty scales with overage."""
        config = GameConfig(bust_rule=BustRuleType.SOFT, win_rule=WinRuleType.HIGHEST, max_number=100)
        result = resolve_round(80, 50, 60, config, 1)

        # Player overage: 80-60 = 20, penalty = 20/100 = 0.2
        assert result.player_busted == True
        assert not result.opponent_busted
        assert result.player_points == pytest.approx(-0.2)
        assert result.opponent_points == 1.0

    def test_soft_bust_penalty_cap(self):
        """Soft Bust: Penalty is capped at 0.5."""
        config = GameConfig(bust_rule=BustRuleType.SOFT, win_rule=WinRuleType.HIGHEST, max_number=100)
        result = resolve_round(100, 50, 20, config, 1)

        # Player overage: 100-20 = 80, penalty = min(0.5, 80/100) = 0.5
        assert result.player_busted == True
        assert result.player_points == pytest.approx(-0.5)
        # Opponent overage: 50-20 = 30, penalty = 30/100 = 0.3
        assert result.opponent_busted == True
        assert result.opponent_points == pytest.approx(-0.3)

    def test_soft_bust_cumulative_penalty(self):
        """Soft Bust + Cumulative: Busted player gets penalty, survivor gets proportional."""
        config = GameConfig(bust_rule=BustRuleType.SOFT, win_rule=WinRuleType.CUMULATIVE, max_number=100)
        result = resolve_round(70, 50, 60, config, 1)

        # Player busts with overage 10, penalty = -0.1
        assert result.player_busted == True
        assert result.player_points == pytest.approx(-0.1)
        # Opponent survives, gets 50/100 = 0.5
        assert not result.opponent_busted
        assert result.opponent_points == pytest.approx(0.5)

    def test_edge_case_choice_equals_r(self):
        """Edge case: Choice exactly equals R (boundary)."""
        config = GameConfig(bust_rule=BustRuleType.STANDARD, win_rule=WinRuleType.HIGHEST, max_number=100)
        result = resolve_round(50, 40, 50, config, 1)

        # Player at R is not busted, should win (higher number)
        assert not result.player_busted
        assert not result.opponent_busted
        assert result.player_won == True
        assert result.player_points == 1.0

    def test_edge_case_choice_equals_r_plus_one(self):
        """Edge case: Choice equals R+1 (minimal overage)."""
        config = GameConfig(bust_rule=BustRuleType.STANDARD, win_rule=WinRuleType.HIGHEST, max_number=100)
        result = resolve_round(51, 40, 50, config, 1)

        # Player busts by 1
        assert result.player_busted == True
        assert not result.opponent_busted
        assert result.player_points == 0.0
        assert result.opponent_points == 1.0

    def test_edge_case_both_choose_same_number(self):
        """Edge case: Both players choose the same number."""
        config = GameConfig(bust_rule=BustRuleType.STANDARD, win_rule=WinRuleType.HIGHEST, max_number=100)
        result = resolve_round(50, 50, 60, config, 1)

        # Tie: no points awarded
        assert not result.player_busted
        assert not result.opponent_busted
        assert result.player_won == False
        assert result.player_points == 0.0
        assert result.opponent_points == 0.0

    def test_edge_case_cumulative_at_boundaries(self):
        """Edge case: Cumulative scoring at min and max numbers."""
        config = GameConfig(
            bust_rule=BustRuleType.STANDARD, win_rule=WinRuleType.CUMULATIVE, min_number=1, max_number=100
        )

        # Min number
        result1 = resolve_round(1, 50, 60, config, 1)
        assert result1.player_points == pytest.approx(0.01)
        assert result1.opponent_points == pytest.approx(0.5)

        # Max number (survives)
        result2 = resolve_round(100, 50, 100, config, 1)
        assert result2.player_points == pytest.approx(1.0)
        assert result2.opponent_points == pytest.approx(0.5)

    def test_different_number_range_scaling(self):
        """Test that formulas scale correctly with different max_number."""
        config = GameConfig(bust_rule=BustRuleType.STANDARD, win_rule=WinRuleType.CUMULATIVE, max_number=200)
        result = resolve_round(100, 150, 180, config, 1)

        # Points should scale to max_number=200
        assert result.player_points == pytest.approx(100 / 200)
        assert result.opponent_points == pytest.approx(150 / 200)

    def test_soft_bust_scales_with_max_number(self):
        """Test that soft bust penalty scales with max_number."""
        config = GameConfig(bust_rule=BustRuleType.SOFT, win_rule=WinRuleType.HIGHEST, max_number=200)
        result = resolve_round(150, 50, 100, config, 1)

        # Overage: 150-100 = 50, penalty = 50/200 = 0.25
        assert result.player_busted == True
        assert result.player_points == pytest.approx(-0.25)


class TestApp:
    def test_sanity(self) -> None:
        """Basic sanity test to ensure server can be instantiated."""
        config = NumericDuelingResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        NumericDuelingResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    @pytest.fixture
    def client(self):
        """Create a test client with stateless cookies for testing."""
        config = NumericDuelingResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        server = NumericDuelingResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        # Use stateless cookies to prevent automatic cookie handling
        class StatelessCookies(Cookies):
            def extract_cookies(self, response):
                pass

        client._cookies = StatelessCookies(client._cookies)
        return client

    def test_standard_bust_highest_wins(self, client):
        """Test Standard Bust + Highest Wins (Classic variant)."""
        game_config = {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": 3,
            "bust_rule": "standard",
            "win_rule": "highest",
            "opponent_type": "fixed",
            "opponent_fixed_value": 50,
        }

        # Seed session
        response = client.post("/seed_session", json={"config": game_config})
        assert response.status_code == 200
        cookies = response.cookies

        # Play round: both under R, player wins (higher number)
        response = client.post("/play_round", json={"player_choice": 60}, cookies=cookies)
        assert response.status_code == 200
        result = response.json()
        assert result["player_choice"] == 60
        assert result["opponent_choice"] == 50
        assert not result["player_busted"]
        assert not result["opponent_busted"]
        if result["random_number"] >= 60:  # Both survive
            assert result["player_won"] == True
            assert result["player_points"] == 1.0
            assert result["opponent_points"] == 0.0

    def test_standard_bust_closest_wins(self, client):
        """Test Standard Bust + Closest to R (Price is Right variant)."""
        game_config = {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": 1,
            "bust_rule": "standard",
            "win_rule": "closest",
            "opponent_type": "fixed",
            "opponent_fixed_value": 50,
        }

        response = client.post("/seed_session", json={"config": game_config})
        cookies = response.cookies

        # Play round with high choice to test "without going over" logic
        response = client.post("/play_round", json={"player_choice": 90}, cookies=cookies)
        result = response.json()

        # Closest to R "without going over" logic:
        # If R < 50: both bust (both over R)
        # If 50 <= R < 90: opponent wins (player over R, disqualified)
        # If R >= 90: player wins (both under R, player closer)
        if result["random_number"] < 50:
            assert result["player_busted"] and result["opponent_busted"]
        elif result["random_number"] < 90:
            # Player is over R (disqualified in "closest without going over")
            assert result["opponent_points"] == 1.0
            assert result["player_points"] == 0.0
        else:  # R >= 90
            # Both under R, player is closer
            assert result["player_won"] == True

    def test_standard_bust_cumulative_scoring(self, client):
        """Test Standard Bust + Cumulative Scoring (High Stakes variant)."""
        game_config = {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": 1,
            "bust_rule": "standard",
            "win_rule": "cumulative",
            "opponent_type": "fixed",
            "opponent_fixed_value": 50,
        }

        response = client.post("/seed_session", json={"config": game_config})
        cookies = response.cookies

        response = client.post("/play_round", json={"player_choice": 30}, cookies=cookies)
        result = response.json()

        # Both survive: get proportional points
        if result["random_number"] >= 50:
            assert result["player_points"] == pytest.approx(0.3)
            assert result["opponent_points"] == pytest.approx(0.5)
        # Opponent busts: only player gets points
        elif result["random_number"] >= 30:
            assert result["player_points"] == pytest.approx(0.3)
            assert result["opponent_points"] == 0.0

    def test_soft_bust_highest_wins(self, client):
        """Test Soft Bust + Highest Wins (Soft Landing variant)."""
        game_config = {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": 1,
            "bust_rule": "soft",
            "win_rule": "highest",
            "opponent_type": "fixed",
            "opponent_fixed_value": 50,
        }

        response = client.post("/seed_session", json={"config": game_config})
        cookies = response.cookies

        # Force a bust scenario
        response = client.post("/play_round", json={"player_choice": 90}, cookies=cookies)
        result = response.json()

        # If player busts (over R), should have negative penalty
        if result["player_busted"]:
            assert result["player_points"] < 0  # Penalty applied
            # Check opponent - might also bust if R is low
            if result["opponent_busted"]:
                assert result["opponent_points"] < 0  # Both get penalties
            else:
                assert result["opponent_points"] == 1.0  # Opponent wins

    def test_soft_bust_closest_wins(self, client):
        """Test Soft Bust + Closest to R."""
        game_config = {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": 1,
            "bust_rule": "soft",
            "win_rule": "closest",
            "opponent_type": "fixed",
            "opponent_fixed_value": 50,
        }

        response = client.post("/seed_session", json={"config": game_config})
        cookies = response.cookies

        response = client.post("/play_round", json={"player_choice": 70}, cookies=cookies)
        result = response.json()

        # If both under R, closest wins
        # If over R, get penalty and can't win
        if result["random_number"] >= 70:
            # Player is closer if R is between 60 and 70
            assert result["player_points"] in [0.0, 1.0]

    def test_soft_bust_cumulative_scoring(self, client):
        """Test Soft Bust + Cumulative Scoring."""
        game_config = {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": 1,
            "bust_rule": "soft",
            "win_rule": "cumulative",
            "opponent_type": "fixed",
            "opponent_fixed_value": 50,
        }

        response = client.post("/seed_session", json={"config": game_config})
        cookies = response.cookies

        response = client.post("/play_round", json={"player_choice": 80}, cookies=cookies)
        result = response.json()

        # If player survives, gets proportional points
        if not result["player_busted"]:
            assert result["player_points"] == pytest.approx(0.8)
        else:
            # If player busts, gets negative penalty
            assert result["player_points"] < 0

    def test_probabilistic_bust_highest_wins(self, client):
        """Test Probabilistic Bust + Highest Wins."""
        game_config = {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": 1,
            "bust_rule": "probabilistic",
            "win_rule": "highest",
            "opponent_type": "fixed",
            "opponent_fixed_value": 50,
        }

        response = client.post("/seed_session", json={"config": game_config})
        cookies = response.cookies

        response = client.post("/play_round", json={"player_choice": 60}, cookies=cookies)
        result = response.json()

        # Probabilistic bust: might bust or survive when over R
        # If survive and both under R, higher wins
        if not result["player_busted"] and not result["opponent_busted"]:
            if result["random_number"] >= 60:
                assert result["player_won"] == True

    def test_probabilistic_bust_closest_wins(self, client):
        """Test Probabilistic Bust + Closest to R."""
        game_config = {
            "min_number": 1,
            "max_number": 200,
            "num_rounds": 1,
            "bust_rule": "probabilistic",
            "win_rule": "closest",
            "opponent_type": "fixed",
            "opponent_fixed_value": 50,
        }

        response = client.post("/seed_session", json={"config": game_config})
        cookies = response.cookies

        response = client.post("/play_round", json={"player_choice": 120}, cookies=cookies)
        result = response.json()

        # If player survives but is over R, can't win in "closest" mode
        if not result["player_busted"] and result["player_choice"] > result["random_number"]:
            assert result["player_points"] == 0.0
            # Opponent wins if under R
            if result["opponent_choice"] <= result["random_number"]:
                assert result["opponent_points"] == 1.0

    def test_probabilistic_bust_cumulative_scoring(self, client):
        """Test Probabilistic Bust + Cumulative Scoring (Risky Business variant)."""
        game_config = {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": 1,
            "bust_rule": "probabilistic",
            "win_rule": "cumulative",
            "opponent_type": "adaptive",
        }

        response = client.post("/seed_session", json={"config": game_config})
        cookies = response.cookies

        response = client.post("/play_round", json={"player_choice": 70}, cookies=cookies)
        result = response.json()

        # If player survives, gets proportional points
        if not result["player_busted"]:
            assert result["player_points"] == pytest.approx(0.7)
        else:
            assert result["player_points"] == 0.0

    def test_both_bust_scenario(self, client):
        """Test that both players busting results in no points."""
        game_config = {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": 1,
            "bust_rule": "standard",
            "win_rule": "highest",
            "opponent_type": "fixed",
            "opponent_fixed_value": 90,
        }

        response = client.post("/seed_session", json={"config": game_config})
        cookies = response.cookies

        response = client.post("/play_round", json={"player_choice": 95}, cookies=cookies)
        result = response.json()

        # If R is low enough, both bust
        if result["player_busted"] and result["opponent_busted"]:
            assert result["player_points"] == 0.0
            assert result["opponent_points"] == 0.0

    def test_multi_round_game_state(self, client):
        """Test that game state persists across multiple rounds."""
        game_config = {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": 3,
            "bust_rule": "standard",
            "win_rule": "cumulative",
            "opponent_type": "fixed",
            "opponent_fixed_value": 50,
        }

        response = client.post("/seed_session", json={"config": game_config})
        cookies = response.cookies

        # Round 1
        response = client.post("/play_round", json={"player_choice": 30}, cookies=cookies)
        result1 = response.json()
        round1_total = result1["player_total_score"]

        # Round 2
        response = client.post("/play_round", json={"player_choice": 40}, cookies=cookies)
        result2 = response.json()
        round2_total = result2["player_total_score"]

        # Total should accumulate
        assert round2_total >= round1_total

        # Round 3 (final)
        response = client.post("/play_round", json={"player_choice": 20}, cookies=cookies)
        result3 = response.json()
        assert result3["game_over"] == True

    def test_different_number_ranges(self, client):
        """Test that custom number ranges work correctly."""
        game_config = {
            "min_number": 1,
            "max_number": 200,
            "num_rounds": 1,
            "bust_rule": "standard",
            "win_rule": "cumulative",
            "opponent_type": "fixed",
            "opponent_fixed_value": 100,
        }

        response = client.post("/seed_session", json={"config": game_config})
        cookies = response.cookies

        response = client.post("/play_round", json={"player_choice": 150}, cookies=cookies)
        result = response.json()

        # Proportional points should be based on max_number=200
        if not result["player_busted"]:
            assert result["player_points"] == pytest.approx(150 / 200)
        if not result["opponent_busted"]:
            assert result["opponent_points"] == pytest.approx(100 / 200)


class TestErrorHandling:
    """Tests for error handling and invalid inputs."""

    @pytest.fixture
    def client(self):
        """Create a test client for error handling tests."""
        config = NumericDuelingResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        server = NumericDuelingResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        return TestClient(app)

    def test_invalid_session_id(self, client):
        """Test that invalid session ID raises ValueError."""
        # Try to play without seeding a session
        # Note: ValueError propagates through TestClient as an exception
        with pytest.raises(ValueError, match="No game state found for session"):
            client.post("/play_round", json={"player_choice": 50}, cookies={"session_id": "nonexistent_session_12345"})

    def test_missing_session_id(self, client):
        """Test that missing session ID raises ValueError."""
        # Try to play without any session (session middleware creates one)
        # But no game state exists for that session
        with pytest.raises(ValueError, match="No game state found for session"):
            client.post("/play_round", json={"player_choice": 50})

    def test_invalid_choice_type(self, client):
        """Test that non-integer choice returns validation error."""
        game_config = {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": 1,
            "bust_rule": "standard",
            "win_rule": "highest",
            "opponent_type": "random",
        }

        response = client.post("/seed_session", json={"config": game_config})
        cookies = response.cookies

        # Try to send a string instead of integer
        response = client.post("/play_round", json={"player_choice": "not_a_number"}, cookies=cookies)

        assert response.status_code == 422  # Unprocessable Entity (Pydantic validation)
        assert "validation" in response.text.lower() or "integer" in response.text.lower()

    def test_missing_player_choice(self, client):
        """Test that missing player_choice field returns validation error."""
        game_config = {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": 1,
            "bust_rule": "standard",
            "win_rule": "highest",
            "opponent_type": "random",
        }

        response = client.post("/seed_session", json={"config": game_config})
        cookies = response.cookies

        # Send empty request body
        response = client.post("/play_round", json={}, cookies=cookies)

        assert response.status_code == 422  # Unprocessable Entity
        assert "player_choice" in response.text.lower() or "required" in response.text.lower()

    def test_invalid_bust_rule(self, client):
        """Test that invalid bust_rule enum value returns validation error."""
        game_config = {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": 1,
            "bust_rule": "invalid_bust_rule",
            "win_rule": "highest",
            "opponent_type": "random",
        }

        response = client.post("/seed_session", json={"config": game_config})

        assert response.status_code == 422  # Pydantic validation error
        assert "bust_rule" in response.text.lower() or "validation" in response.text.lower()

    def test_invalid_win_rule(self, client):
        """Test that invalid win_rule enum value returns validation error."""
        game_config = {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": 1,
            "bust_rule": "standard",
            "win_rule": "invalid_win_rule",
            "opponent_type": "random",
        }

        response = client.post("/seed_session", json={"config": game_config})

        assert response.status_code == 422
        assert "win_rule" in response.text.lower() or "validation" in response.text.lower()

    def test_invalid_opponent_type(self, client):
        """Test that invalid opponent_type enum value returns validation error."""
        game_config = {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": 1,
            "bust_rule": "standard",
            "win_rule": "highest",
            "opponent_type": "invalid_opponent",
        }

        response = client.post("/seed_session", json={"config": game_config})

        assert response.status_code == 422
        assert "opponent_type" in response.text.lower() or "validation" in response.text.lower()

    def test_missing_config_fields(self, client):
        """Test that missing config fields use defaults (Pydantic defaults)."""
        # Missing some fields - Pydantic may have defaults
        incomplete_config = {
            "min_number": 1,
            "max_number": 100,
        }

        response = client.post("/seed_session", json={"config": incomplete_config})

        # GameConfig has defaults for num_rounds, bust_rule, win_rule, opponent_type
        # So this might succeed with defaults
        if response.status_code == 200:
            # Defaults were applied successfully
            assert response.json()["session_id"]
        else:
            # Or validation fails if no defaults exist
            assert response.status_code == 422

    def test_invalid_number_range(self, client):
        """Test that min_number > max_number causes ValueError during gameplay."""
        game_config = {
            "min_number": 100,
            "max_number": 1,  # Invalid: min > max
            "num_rounds": 1,
            "bust_rule": "standard",
            "win_rule": "highest",
            "opponent_type": "random",
        }

        # Seed succeeds (no validation on config values)
        response = client.post("/seed_session", json={"config": game_config})
        assert response.status_code == 200
        cookies = response.cookies

        # But playing crashes when random.randint is called with invalid range
        with pytest.raises(ValueError, match="empty range in randrange"):
            client.post("/play_round", json={"player_choice": 50}, cookies=cookies)

    def test_negative_num_rounds(self, client):
        """Test that negative num_rounds is handled."""
        game_config = {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": -5,  # Invalid: negative rounds
            "bust_rule": "standard",
            "win_rule": "highest",
            "opponent_type": "random",
        }

        response = client.post("/seed_session", json={"config": game_config})

        # Pydantic should validate this or game should handle gracefully
        # Document current behavior
        assert response.status_code in [200, 422]

    def test_malformed_json(self, client):
        """Test that malformed JSON returns appropriate error."""
        response = client.post("/seed_session", data="not valid json{{{", headers={"Content-Type": "application/json"})

        assert response.status_code == 422  # FastAPI validation error
