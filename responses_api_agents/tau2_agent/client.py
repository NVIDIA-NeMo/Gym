"""Example client for the Tau2 bench agent.

This script demonstrates how to call the agent's `/run` endpoint using
the shared `ServerClient` helper. It posts a run request with a
minimal `responses_create_params` payload and prints the server result.
"""

import asyncio
import json

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient, get_response_json


async def main():
    server_client = ServerClient.load_from_global_config()

    # Example payload: adjust `task_id` or `task_domain` as needed.
    params = NeMoGymResponseCreateParamsNonStreaming(
        input=[],
        temperature=0.0,
        top_p=1.0,
    )

    payload = {
        "responses_create_params": params.model_dump(exclude_unset=True),
        # Optional fields the agent may accept
        "task_id": "1",
        "task_domain": "airline",
    }

    response = await server_client.post(
        server_name="tau2_agent",
        url_path="/run",
        json=payload,
    )

    resp_json = await get_response_json(response)
    print(json.dumps(resp_json, indent=2))


if __name__ == "__main__":
    print("Tau2 agent client example")
    asyncio.run(main())
