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

"""
Interactive client for Numeric Dueling.
Allows a human to play the game against different opponents.

Usage:
    # With default port (5011)
    python client.py

    # With custom port
    PORT=8001 python client.py
    python client.py --port 8001
"""

import os
import sys
from typing import Any, Dict

import requests


class NumericDuelingClient:
    """Client for playing Numeric Dueling interactively."""

    def __init__(self, base_url: str = "http://localhost:5011"):
        self.base_url = base_url
        self.session_id = None
        # Use requests.Session() to maintain cookies between requests
        self.session = requests.Session()

    def start_game(self, game_config: Dict[str, Any]) -> None:
        """Start a new game with the given configuration."""
        response = self.session.post(f"{self.base_url}/seed_session", json={"config": game_config})
        response.raise_for_status()
        result = response.json()
        self.session_id = result["session_id"]
        print(f"\n🎮 Game Started! Session ID: {self.session_id}\n")
        self._print_config(game_config)

    def _print_config(self, config: Dict[str, Any]) -> None:
        """Print game configuration in a readable format."""
        print("=" * 60)
        print("GAME CONFIGURATION")
        print("=" * 60)
        print(f"Number Range: {config['min_number']} - {config['max_number']}")
        print(f"Number of Rounds: {config['num_rounds']}")
        print(f"Bust Rule: {config['bust_rule']}")
        print(f"Win Rule: {config['win_rule']}")
        print(f"Opponent Type: {config['opponent_type']}")
        if config["opponent_type"] == "Fixed":
            print(f"Opponent Choice: {config.get('opponent_fixed_choice', 'N/A')}")
        print("=" * 60)
        print()

    def play_round(self) -> Dict[str, Any]:
        """Get current game state and prompt."""
        response = self.session.post(f"{self.base_url}/get_prompt", json={"session_id": self.session_id})
        response.raise_for_status()
        return response.json()

    def submit_choice(self, choice: int) -> Dict[str, Any]:
        """Submit player's choice and get round result."""
        response = self.session.post(f"{self.base_url}/play_round", json={"player_choice": choice})
        if not response.ok:
            print("\n❌ Error from server:")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
        response.raise_for_status()
        return response.json()

    def print_round_result(self, result: Dict[str, Any]) -> None:
        """Print the result of a round in a readable format."""
        print("\n" + "=" * 60)
        print(f"ROUND {result['round_number']} RESULT")
        print("=" * 60)
        print(f"Your choice: {result['player_choice']}")
        print(f"Opponent choice: {result['opponent_choice']}")
        print(f"Random number (R): {result['random_number']}")
        print()

        # Determine outcome
        if result["player_busted"] and result["opponent_busted"]:
            print("💥 BOTH BUST! No points awarded.")
        elif result["player_busted"]:
            print("💥 YOU BUST! Opponent wins this round.")
        elif result["opponent_busted"]:
            print("🎯 OPPONENT BUSTS! You win this round.")
        elif result["player_won"]:
            print("🏆 YOU WIN this round!")
        else:
            print("😞 Opponent wins this round.")

        print()
        print(f"Points this round: You {result['player_points']:.1f}, Opponent {result['opponent_points']:.1f}")
        print(
            f"Total Score: You {result.get('player_total_score', 0):.1f}, Opponent {result.get('opponent_total_score', 0):.1f}"
        )
        print("=" * 60)


def print_game_menu():
    """Print game configuration menu."""
    print("\n" + "=" * 60)
    print("NUMERIC DUELING - Game Setup")
    print("=" * 60)
    print("Select a game variant:")
    print("1. Classic (Standard Bust, Highest Wins)")
    print("2. Price is Right (Standard Bust, Closest to R)")
    print("3. High Stakes (Standard Bust, Cumulative Scoring)")
    print("4. Soft Landing (Soft Bust, Highest Wins)")
    print("5. Risky Business (Probabilistic Bust, Cumulative Scoring)")
    print("6. Custom")
    print("=" * 60)


def get_game_config() -> Dict[str, Any]:
    """Get game configuration from user."""
    print_game_menu()

    choice = input("\nEnter choice (1-6): ").strip()

    # Preset configurations
    presets = {
        "1": {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": 5,
            "bust_rule": "standard",
            "win_rule": "highest",
            "opponent_type": "random",
        },
        "2": {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": 5,
            "bust_rule": "standard",
            "win_rule": "closest",
            "opponent_type": "random",
        },
        "3": {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": 5,
            "bust_rule": "standard",
            "win_rule": "cumulative",
            "opponent_type": "random",
        },
        "4": {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": 5,
            "bust_rule": "soft",
            "win_rule": "highest",
            "opponent_type": "random",
        },
        "5": {
            "min_number": 1,
            "max_number": 100,
            "num_rounds": 5,
            "bust_rule": "probabilistic",
            "win_rule": "cumulative",
            "opponent_type": "adaptive",
        },
    }

    if choice in presets:
        return presets[choice]
    elif choice == "6":
        # Custom configuration
        config = {}
        config["min_number"] = int(input("Min number (default 1): ") or "1")
        config["max_number"] = int(input("Max number (default 100): ") or "100")
        config["num_rounds"] = int(input("Number of rounds (default 5): ") or "5")

        print("\nBust Rule: standard, soft, probabilistic")
        config["bust_rule"] = (input("Bust rule (default standard): ") or "standard").lower()

        print("\nWin Rule: highest, closest, cumulative")
        config["win_rule"] = (input("Win rule (default highest): ") or "highest").lower()

        print("\nOpponent Type: random, fixed, adaptive")
        config["opponent_type"] = (input("Opponent type (default random): ") or "random").lower()

        if config["opponent_type"] == "fixed":
            config["opponent_fixed_value"] = int(
                input(f"Opponent fixed choice ({config['min_number']}-{config['max_number']}): ")
            )

        return config
    else:
        print("Invalid choice, using Classic variant")
        return presets["1"]


def main():
    """Main game loop."""
    print("\n" + "🎲" * 30)
    print("WELCOME TO NUMERIC DUELING!")
    print("🎲" * 30)

    # Parse port from command line or environment variable
    port = 5011  # Default port
    if "--port" in sys.argv:
        try:
            port_idx = sys.argv.index("--port")
            port = int(sys.argv[port_idx + 1])
        except (IndexError, ValueError):
            print("Error: --port requires a valid integer")
            sys.exit(1)
    elif "PORT" in os.environ:
        try:
            port = int(os.environ["PORT"])
        except ValueError:
            print(f"Error: PORT environment variable must be an integer, got {os.environ['PORT']}")
            sys.exit(1)

    base_url = f"http://localhost:{port}"
    print(f"Connecting to server at {base_url}\n")

    # Get game configuration
    config = get_game_config()

    # Initialize client
    client = NumericDuelingClient(base_url=base_url)

    # Start game
    try:
        client.start_game(config)
    except requests.exceptions.ConnectionError:
        print(f"\n❌ ERROR: Could not connect to game server at {base_url}")
        print("\nTo start the server:")
        print("  1. Activate venv: source .venv/bin/activate")
        print(
            '  2. Run: ng_run "+config_paths=[resources_servers/numeric_dueling/configs/numeric_dueling_server_only.yaml]"'
        )
        print("  3. Check terminal output for the assigned port")
        print("\nThen connect client to the correct port:")
        print("  python client.py --port <port_from_step_3>")
        print("  or: PORT=<port> python client.py")
        return

    # Play rounds
    result = None
    for round_num in range(1, config["num_rounds"] + 1):
        print(f"\n{'🔥' * 30}")
        print(f"ROUND {round_num}/{config['num_rounds']}")
        print("🔥" * 30)

        # Show current score (from previous round result, or 0 for first round)
        if result:
            player_score = result.get("player_total_score", 0)
            opponent_score = result.get("opponent_total_score", 0)
        else:
            player_score = opponent_score = 0
        print(f"\nCurrent Score: You {player_score:.1f}, Opponent {opponent_score:.1f}")

        # Get player input
        while True:
            try:
                choice_str = input(f"\nEnter your choice ({config['min_number']}-{config['max_number']}): ").strip()
                choice = int(choice_str)
                if config["min_number"] <= choice <= config["max_number"]:
                    break
                else:
                    print(f"Please enter a number between {config['min_number']} and {config['max_number']}")
            except ValueError:
                print("Please enter a valid number")

        # Submit choice and get result
        result = client.submit_choice(choice)
        client.print_round_result(result)

        # Check if game is over
        if result.get("game_over", False):
            break

    # Print final results
    print("\n" + "🏁" * 30)
    print("GAME OVER!")
    print("🏁" * 30)

    # Get final scores from last round result
    player_score = result.get("player_total_score", 0)
    opponent_score = result.get("opponent_total_score", 0)

    print(f"\nFinal Score: You {player_score:.1f}, Opponent {opponent_score:.1f}")

    if player_score > opponent_score:
        print("\n🎉🎉🎉 YOU WIN! 🎉🎉🎉")
    elif player_score < opponent_score:
        print("\n😞 You lose. Better luck next time!")
    else:
        print("\n🤝 It's a tie!")

    print("\nThanks for playing Numeric Dueling!")


if __name__ == "__main__":
    main()
