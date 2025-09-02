import json

from asyncio import run

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient




server_client = ServerClient.load_from_global_config()
task = server_client.post(
    server_name="simple_agent",
    url_path="/v1/responses",
    json=NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {
                "role": "developer",
                "content": "You are a helpful personal assistant that aims to be helpful and reduce any pain points the user has.",
            },
            {"role": "user", "content": "going out in sf tn. is it safe boo check on google no"},
        ],
        tools=[
            {
                "type": "function",
                "name": "search",
                "description": "Search Google for a query and return up to 10 search results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The term to search for",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ],
    ),
)
result = run(task)
print(json.dumps(result.json(), indent=4))
