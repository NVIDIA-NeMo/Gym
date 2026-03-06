from asyncio import run

from nemo_gym.server_utils import ServerClient


async def test_wordle_server():
    server_client = ServerClient.load_from_global_config()

    # Get initial board
    initial_response = await server_client.post(
        server_name="simple_wordle",
        url_path="/get_initial_board",
        json={"word_length": 5, "max_turns": 6},
    )

    initial_data = initial_response.json()
    print("Initial Game:")
    print(initial_data["board_text"])
    print("\nInstructions:")
    print(initial_data["instructions"])

    # Make a test move (example guess)
    game_state = initial_data["game_state"]
    move_response = await server_client.post(
        server_name="simple_wordle",
        url_path="/make_move",
        json={
            "game_state": game_state,
            "guess": "CRANE",  # Example guess
        },
    )

    move_data = move_response.json()
    print(f"\nMove Result: {move_data['message']}")
    print("Updated Board:")
    print(move_data["board_text"])


if __name__ == "__main__":
    run(test_wordle_server())
