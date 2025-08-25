import json

from asyncio import run

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()
task = server_client.post(
    server_name="simple_weather_agent",
    url_path="/v1/responses",
    json=NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {
                "role": "developer",
                "content": "You are a helpful personal assistant that aims to be helpful and reduce any pain points the user has.",
            },
            {"role": "user", "content": "going out in sf tn"},
        ],
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "",
                        },
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ],
    ),
)
result = run(task)
print(json.dumps(result.json()["output"], indent=4))
