import json

from asyncio import run

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()
task = server_client.post(
    server_name="simple_agent_stateful",
    url_path="/v1/responses",
    json=NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {
                "role": "user",
                "content": "What is the smallest positive integer that leaves a remainder of 4 when divided by 5 and a remainder of 6 when divided by 7?",
            },
        ],
        tools=[
            {
                "type": "function",
                "name": "execute_python",
                "description": "Execute Python code to perform calculations. You have access to numpy, scipy, pandas and basic math operations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute for the calculation.",
                        },
                    },
                    "required": ["code"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ],
    ),
)
result = run(task)
print(json.dumps(result.json()["output"], indent=4))
