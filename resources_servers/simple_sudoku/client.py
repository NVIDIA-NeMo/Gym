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

from asyncio import run

from nemo_gym.server_utils import ServerClient


async def test_sudoku_server():
    server_client = ServerClient.load_from_global_config()

    # Get initial board
    initial_response = await server_client.post(
        server_name="simple_sudoku",
        url_path="/get_initial_board",
        json={"clues": 8, "scale": 4},
    )

    initial_data = initial_response.json()
    print("Initial Board:")
    print(initial_data["board_text"])  # Use the formatted board_text instead!
    print("\nInstructions:")
    print(initial_data["instructions"])

    # Make a test move (you'd need to analyze the board first)
    game_state = initial_data["game_state"]
    move_response = await server_client.post(
        server_name="simple_sudoku",
        url_path="/make_move",
        json={
            "game_state": game_state,
            "move": "\\boxed{1 1 5}",  # Example move
        },
    )

    move_data = move_response.json()
    print(f"\nMove Result: {move_data['message']}")
    print("Updated Board:")
    print(move_data["board_text"])  # Use the formatted board_text here too!


if __name__ == "__main__":
    run(test_sudoku_server())
