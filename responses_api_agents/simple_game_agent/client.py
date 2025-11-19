import json
from asyncio import run

from nemo_gym.server_utils import ServerClient


async def test_text_game_agent():
    server_client = ServerClient.load_from_global_config()

    # Run a complete game
    task = server_client.post(
        server_name="simple_game_agent",
        url_path="/run",
        json={
            "responses_create_params": {
                "input": [{"role": "user", "content": "Let's play Sudoku!"}],
                "tools": [],
            },
            "clues": 30,
            "scale": 9,
        },
    )

    result = await task

    print("=== RAW RESPONSE DEBUG ===")
    print(f"Status Code: {result.status_code}")
    print(f"Headers: {dict(result.headers)}")
    print(f"Raw Content (as bytes): {result.content}")
    print(f"Raw Content (as text): {result.text}")
    print(f"Content Length: {len(result.content)}")
    print("=========================")

    print("Game Result:")
    print(json.dumps(result.json(), indent=2))


if __name__ == "__main__":
    run(test_text_game_agent())
