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

import random
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY


# ============================================================================
# Game Configuration
# ============================================================================


class BustRuleType(str, Enum):
    """Types of bust rules"""

    STANDARD = "standard"  # Bust if n_i > R (lose round)
    SOFT = "soft"  # Lose partial points but stay in game
    PROBABILISTIC = "probabilistic"  # Bust with probability based on overage


class WinRuleType(str, Enum):
    """Types of win rules"""

    HIGHEST = "highest"  # Highest non-busted number wins
    CLOSEST = "closest"  # Closest to R without going over
    CUMULATIVE = "cumulative"  # Points proportional to number


class OpponentType(str, Enum):
    """Types of opponents"""

    RANDOM = "random"  # Random choice from range
    FIXED = "fixed"  # Always same number
    ADAPTIVE = "adaptive"  # Simple adaptive strategy


class GameConfig(BaseModel):
    """Configuration for a numeric dueling game"""

    num_rounds: int = Field(default=5, description="Number of rounds (M)")
    min_number: int = Field(default=1, description="Minimum valid number")
    max_number: int = Field(default=100, description="Maximum valid number")
    bust_rule: BustRuleType = Field(default=BustRuleType.STANDARD, description="How busting works")
    win_rule: WinRuleType = Field(default=WinRuleType.HIGHEST, description="How winners are determined")
    opponent_type: OpponentType = Field(default=OpponentType.RANDOM, description="Opponent strategy")
    opponent_fixed_value: Optional[int] = Field(default=None, description="Fixed value if opponent_type=fixed")


# ============================================================================
# Game State
# ============================================================================


class RoundResult(BaseModel):
    """Result of a single round"""

    round_number: int
    player_choice: int
    opponent_choice: int
    random_number: int
    player_busted: bool
    opponent_busted: bool
    player_won: bool
    player_points: float
    opponent_points: float


class GameState(BaseModel):
    """State of an ongoing game"""

    config: GameConfig
    current_round: int = Field(default=1, description="Current round (1-indexed)")
    player_score: float = Field(default=0.0, description="Player cumulative score")
    opponent_score: float = Field(default=0.0, description="Opponent cumulative score")
    history: List[RoundResult] = Field(default_factory=list, description="History of all rounds")
    game_over: bool = Field(default=False, description="Whether game is complete")

    def is_game_over(self) -> bool:
        """Check if game is complete"""
        return self.current_round > self.config.num_rounds


# ============================================================================
# Opponent Interface
# ============================================================================


class Opponent(ABC):
    """Abstract opponent interface"""

    @abstractmethod
    def make_move(self, game_state: GameState) -> int:
        """Make a move given current game state"""
        pass


class RandomOpponent(Opponent):
    """Opponent that chooses randomly"""

    def make_move(self, game_state: GameState) -> int:
        return random.randint(game_state.config.min_number, game_state.config.max_number)


class FixedOpponent(Opponent):
    """Opponent that always chooses the same number"""

    def __init__(self, value: int):
        self.value = value

    def make_move(self, game_state: GameState) -> int:
        return self.value


class AdaptiveOpponent(Opponent):
    """Simple adaptive opponent - plays slightly below average of random numbers seen"""

    def make_move(self, game_state: GameState) -> int:
        if not game_state.history:
            # First round: play middle
            return (game_state.config.min_number + game_state.config.max_number) // 2

        # Play 5 below the average random number we've seen
        avg_random = sum(r.random_number for r in game_state.history) / len(game_state.history)
        choice = int(avg_random - 5)

        # Clamp to valid range
        return max(game_state.config.min_number, min(game_state.config.max_number, choice))


def create_opponent(config: GameConfig) -> Opponent:
    """Factory function to create opponent based on config"""
    if config.opponent_type == OpponentType.RANDOM:
        return RandomOpponent()
    elif config.opponent_type == OpponentType.FIXED:
        if config.opponent_fixed_value is None:
            raise ValueError("opponent_fixed_value must be set when using fixed opponent")
        return FixedOpponent(config.opponent_fixed_value)
    elif config.opponent_type == OpponentType.ADAPTIVE:
        return AdaptiveOpponent()
    else:
        raise ValueError(f"Unknown opponent type: {config.opponent_type}")


# ============================================================================
# Game Logic
# ============================================================================


def extract_number_from_response(response_text: str, min_val: int, max_val: int) -> Optional[int]:
    """
    Extract a number from LLM response text.

    Parsing priority:
    1. Look for <choice>NUMBER</choice> tags (most reliable)
    2. Fall back to finding any integer in valid range
    """
    # First, try to find number in <choice> tags
    choice_pattern = r"<choice>\s*(\d+)\s*</choice>"
    choice_match = re.search(choice_pattern, response_text, re.IGNORECASE)

    if choice_match:
        try:
            num = int(choice_match.group(1))
            if min_val <= num <= max_val:
                return num
        except ValueError:
            pass

    # Fall back to finding any integer in valid range
    numbers = re.findall(r"\b\d+\b", response_text)

    for num_str in numbers:
        try:
            num = int(num_str)
            if min_val <= num <= max_val:
                return num
        except ValueError:
            continue

    return None


def resolve_round(
    player_choice: int, opponent_choice: int, random_number: int, config: GameConfig, round_number: int
) -> RoundResult:
    """
    Resolve a single round and determine points based on configured rules.
    Supports all bust rule and win rule variants.
    """
    # Apply bust rule
    player_busted = False
    opponent_busted = False
    player_bust_penalty = 0.0
    opponent_bust_penalty = 0.0

    if config.bust_rule == BustRuleType.STANDARD:
        # Standard: if n > R, you bust (lose round)
        player_busted = player_choice > random_number
        opponent_busted = opponent_choice > random_number

    elif config.bust_rule == BustRuleType.SOFT:
        # Soft bust: partial penalty proportional to how much you exceeded R
        if player_choice > random_number:
            player_busted = True
            overage = player_choice - random_number
            # Penalty scales with range: exceeding by full range = 0.5 penalty
            player_bust_penalty = min(0.5, overage / config.max_number)
        if opponent_choice > random_number:
            opponent_busted = True
            overage = opponent_choice - random_number
            opponent_bust_penalty = min(0.5, overage / config.max_number)

    elif config.bust_rule == BustRuleType.PROBABILISTIC:
        # Probabilistic bust: probability based on overage
        if player_choice > random_number:
            overage = player_choice - random_number
            # Probability scales with range: exceeding by half range = 100% bust
            bust_probability = min(1.0, overage / (config.max_number / 2.0))
            player_busted = random.random() < bust_probability
        if opponent_choice > random_number:
            overage = opponent_choice - random_number
            bust_probability = min(1.0, overage / (config.max_number / 2.0))
            opponent_busted = random.random() < bust_probability

    # Determine winner based on win rule
    player_points = 0.0
    opponent_points = 0.0
    player_won = False

    # For cumulative scoring, handle differently: survivors always get proportional points
    if config.win_rule == WinRuleType.CUMULATIVE:
        # In cumulative scoring, survivors get points proportional to their choice
        if player_busted:
            player_points = -player_bust_penalty if player_bust_penalty > 0 else 0.0
        else:
            player_points = player_choice / config.max_number

        if opponent_busted:
            opponent_points = -opponent_bust_penalty if opponent_bust_penalty > 0 else 0.0
        else:
            opponent_points = opponent_choice / config.max_number

        player_won = player_points > opponent_points

    # For other win rules, handle bust/win logic traditionally
    else:
        # First, check if anyone busted
        if player_busted and opponent_busted:
            # Both bust: no points, apply penalties if soft bust
            player_points = -player_bust_penalty if player_bust_penalty > 0 else 0.0
            opponent_points = -opponent_bust_penalty if opponent_bust_penalty > 0 else 0.0
        elif player_busted:
            # Player busts, opponent wins (if not busted)
            opponent_points = 1.0
            player_points = -player_bust_penalty if player_bust_penalty > 0 else 0.0
        elif opponent_busted:
            # Opponent busts, player wins
            player_points = 1.0
            opponent_points = -opponent_bust_penalty if opponent_bust_penalty > 0 else 0.0
            player_won = True
        else:
            # Neither busted - apply win rule
            if config.win_rule == WinRuleType.HIGHEST:
                # Highest number wins
                if player_choice > opponent_choice:
                    player_points = 1.0
                    player_won = True
                elif opponent_choice > player_choice:
                    opponent_points = 1.0
                # If equal, no points

            elif config.win_rule == WinRuleType.CLOSEST:
                # Closest to R without going over wins (Price is Right style)
                # Only consider choices that are <= R
                player_under_r = player_choice <= random_number
                opponent_under_r = opponent_choice <= random_number

                if player_under_r and opponent_under_r:
                    # Both under R: closest wins
                    player_distance = random_number - player_choice
                    opponent_distance = random_number - opponent_choice
                    if player_distance < opponent_distance:
                        player_points = 1.0
                        player_won = True
                    elif opponent_distance < player_distance:
                        opponent_points = 1.0
                    # If equal distance, no points
                elif player_under_r and not opponent_under_r:
                    # Only player under R: player wins
                    player_points = 1.0
                    player_won = True
                elif opponent_under_r and not player_under_r:
                    # Only opponent under R: opponent wins
                    opponent_points = 1.0
                # If both over R, no points (even if they survived probabilistic bust)

    return RoundResult(
        round_number=round_number,
        player_choice=player_choice,
        opponent_choice=opponent_choice,
        random_number=random_number,
        player_busted=player_busted,
        opponent_busted=opponent_busted,
        player_won=player_won,
        player_points=player_points,
        opponent_points=opponent_points,
    )


# ============================================================================
# Prompt Generation
# ============================================================================


def generate_game_prompt(game_state: GameState) -> str:
    """
    Generate a prompt for the LLM based on current game state and configuration.
    Dynamically adapts to all rule variants.
    """
    config = game_state.config

    # Build rules description based on configuration
    bust_rule_desc = {
        BustRuleType.STANDARD: "If your number > R, you BUST and get 0 points for that round",
        BustRuleType.SOFT: "If your number > R, you lose partial points (penalty increases with how much you exceed R)",
        BustRuleType.PROBABILISTIC: "If your number > R, you might bust with probability based on how much you exceed R",
    }[config.bust_rule]

    win_rule_desc = {
        WinRuleType.HIGHEST: "Among non-busted players, the highest number wins 1 point",
        WinRuleType.CLOSEST: "Among non-busted players, the closest to R wins 1 point (Price is Right style)",
        WinRuleType.CUMULATIVE: "Points are proportional to your number (higher numbers = more points, but riskier)",
    }[config.win_rule]

    # Build the prompt
    prompt = f"""You are playing Numeric Dueling. Choose a number from {config.min_number}-{config.max_number}.

Rules:
- A random number R will be drawn from {config.min_number}-{config.max_number}
- {bust_rule_desc}
- {win_rule_desc}
- Game lasts {config.num_rounds} rounds. Highest total score at the end wins.

Put your choice inside <choice></choice> tags. You can think step-by-step first.

Current Status:
- Round: {game_state.current_round}/{config.num_rounds}
- Your score: {game_state.player_score}
- Opponent score: {game_state.opponent_score}
"""

    # Add history if there is any
    if game_state.history:
        prompt += "\nHistory:\n"
        for result in game_state.history:
            outcome = "You won" if result.player_won else "Opponent won"
            if result.player_busted and result.opponent_busted:
                outcome = "Both BUST"
            elif result.player_busted:
                outcome = "You BUST. Opponent won"
            elif result.opponent_busted:
                outcome = "Opponent BUST. You won"

            prompt += f"Round {result.round_number}: You chose {result.player_choice}, Opponent chose {result.opponent_choice}, R was {result.random_number}. {outcome}. "
            prompt += f"(Points: You {result.player_points:.1f}, Opp {result.opponent_points:.1f})\n"

    prompt += "\nWhat number do you choose?"

    return prompt


# ============================================================================
# Request/Response Models
# ============================================================================


class NumericDuelingSeedSessionRequest(BaseSeedSessionRequest):
    """Request to initialize a new game"""

    config: Optional[GameConfig] = Field(
        default=None, description="Game configuration (uses defaults if not provided)"
    )


class NumericDuelingVerifyRequest(BaseVerifyRequest):
    """Request to verify a player's move"""

    pass


class NumericDuelingVerifyResponse(BaseVerifyResponse):
    """Response with game result"""

    round_result: Optional[RoundResult] = Field(default=None, description="Result of this round")
    game_state: Optional[GameState] = Field(default=None, description="Current game state")
    extraction_failed: bool = Field(default=False, description="Whether number extraction failed")
    error_message: Optional[str] = Field(default=None, description="Error message if any")


class GetPromptRequest(BaseModel):
    """Request to get current game prompt"""

    pass


class GetPromptResponse(BaseModel):
    """Response with dynamically generated prompt"""

    prompt: str = Field(description="Generated prompt for current game state")
    game_state: GameState = Field(description="Current game state")


class PlayRoundRequest(BaseModel):
    """Request for human player to play a round"""

    player_choice: int


# ============================================================================
# Server
# ============================================================================


class NumericDuelingResourcesServerConfig(BaseResourcesServerConfig):
    pass


class NumericDuelingResourcesServer(SimpleResourcesServer):
    config: NumericDuelingResourcesServerConfig

    # Store game state per session
    session_to_game_state: Dict[str, GameState] = Field(default_factory=dict)
    session_to_opponent: Dict[str, Opponent] = Field(default_factory=dict)

    # Allow arbitrary types (needed for abstract Opponent class)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Add route for getting dynamically generated prompts
        app.post("/get_prompt")(self.get_prompt)

        # Add simplified route for human players
        app.post("/play_round")(self.play_round)

        return app

    async def get_prompt(self, request: Request, body: GetPromptRequest) -> GetPromptResponse:
        """
        Get dynamically generated prompt for current game state.
        Agents can call this to get the appropriate prompt for the LLM.
        """
        session_id = request.session[SESSION_ID_KEY]

        # Get or create game state
        if session_id not in self.session_to_game_state:
            # Initialize with defaults if not seeded
            await self.seed_session(request, NumericDuelingSeedSessionRequest())

        game_state = self.session_to_game_state[session_id]

        # Generate prompt based on current state
        prompt = generate_game_prompt(game_state)

        return GetPromptResponse(prompt=prompt, game_state=game_state)

    async def seed_session(self, request: Request, body: NumericDuelingSeedSessionRequest) -> Dict[str, str]:
        """Initialize a new game for this session"""
        session_id = request.session[SESSION_ID_KEY]

        # Use provided config or defaults
        config = body.config if body.config else GameConfig()

        # Initialize game state
        game_state = GameState(config=config)
        self.session_to_game_state[session_id] = game_state

        # Create opponent
        opponent = create_opponent(config)
        self.session_to_opponent[session_id] = opponent

        # Return session_id so client can track it
        return {"session_id": session_id}

    async def play_round(self, request: Request, body: PlayRoundRequest) -> Dict[str, Any]:
        """
        Simplified endpoint for human players.
        Expects: {"player_choice": <number>}
        Returns: Round result with all game info
        """
        session_id = request.session[SESSION_ID_KEY]
        player_choice = body.player_choice

        # Get game state and opponent (with error checking)
        if session_id not in self.session_to_game_state:
            raise ValueError(f"No game state found for session {session_id}. Did you call /seed_session first?")
        if session_id not in self.session_to_opponent:
            raise ValueError(f"No opponent found for session {session_id}. Did you call /seed_session first?")

        game_state = self.session_to_game_state[session_id]
        opponent = self.session_to_opponent[session_id]

        # Get opponent's move
        opponent_choice = opponent.make_move(game_state)

        # Draw random number
        R = random.randint(game_state.config.min_number, game_state.config.max_number)

        # Resolve round
        round_result = resolve_round(player_choice, opponent_choice, R, game_state.config, game_state.current_round)

        # Update game state
        game_state.history.append(round_result)
        game_state.player_score += round_result.player_points
        game_state.opponent_score += round_result.opponent_points
        game_state.current_round += 1

        # Check if game is over
        game_over = game_state.current_round > game_state.config.num_rounds

        return {
            **round_result.model_dump(),
            "player_total_score": game_state.player_score,
            "opponent_total_score": game_state.opponent_score,
            "game_over": game_over,
        }

    async def verify(self, request: Request, body: NumericDuelingVerifyRequest) -> NumericDuelingVerifyResponse:
        """
        Verify player's move for this round.

        Flow:
        1. Extract player's number choice from response
        2. Get opponent's move
        3. Draw random number
        4. Resolve round
        5. Update game state
        6. Return reward
        """
        session_id = request.session[SESSION_ID_KEY]

        # Get game state
        if session_id not in self.session_to_game_state:
            # No game initialized - create with defaults
            await self.seed_session(request, NumericDuelingSeedSessionRequest())

        game_state = self.session_to_game_state[session_id]
        opponent = self.session_to_opponent[session_id]

        # Check if game is over
        if game_state.is_game_over():
            return NumericDuelingVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                game_state=game_state,
                error_message="Game is already over",
            )

        # Extract player's choice from LLM response
        response_text = ""
        if body.response.output:
            last_output = body.response.output[-1]
            if hasattr(last_output, "content") and last_output.content:
                response_text = last_output.content[0].text

        player_choice = extract_number_from_response(
            response_text, game_state.config.min_number, game_state.config.max_number
        )

        if player_choice is None:
            # Extraction failed - player automatically loses this round
            return NumericDuelingVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                game_state=game_state,
                extraction_failed=True,
                error_message=f"Could not extract valid number from response: {response_text[:100]}",
            )

        # Get opponent's move
        opponent_choice = opponent.make_move(game_state)

        # Draw random number
        random_number = random.randint(game_state.config.min_number, game_state.config.max_number)

        # Resolve round
        round_result = resolve_round(
            player_choice=player_choice,
            opponent_choice=opponent_choice,
            random_number=random_number,
            config=game_state.config,
            round_number=game_state.current_round,
        )

        # Update game state
        game_state.history.append(round_result)
        game_state.player_score += round_result.player_points
        game_state.opponent_score += round_result.opponent_points
        game_state.current_round += 1
        game_state.game_over = game_state.is_game_over()

        # Store updated state
        self.session_to_game_state[session_id] = game_state

        # Return reward for this round
        return NumericDuelingVerifyResponse(
            **body.model_dump(),
            reward=round_result.player_points,
            round_result=round_result,
            game_state=game_state,
        )


if __name__ == "__main__":
    NumericDuelingResourcesServer.run_webserver()
