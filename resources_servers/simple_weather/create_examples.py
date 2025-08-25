import json

from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymEasyInputMessageParam,
)

queries = [
    "what's it like in sf?",
    "going out in sf tn",
    "humidity in sf",
    "how's the outside?",
    "get the weather for 3 cities in the us",
]
base_response_create_params = NeMoGymResponseCreateParamsNonStreaming(
    input=[
        {
            "role": "developer",
            "content": "You are a helpful personal assistant that aims to be helpful and reduce any pain points the user has.",
        },
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
)

example_strs = []
for query in queries:
    example = base_response_create_params.model_copy(
        update={
            "input": base_response_create_params.input
            + [NeMoGymEasyInputMessageParam(role="user", content=query)]
        }
    )
    example_strs.append(
        json.dumps({"responses_create_params": example.model_dump(exclude_unset=True)})
        + "\n"
    )


with open("resources_servers/simple_weather/data/example.jsonl", "w") as f:
    f.writelines(example_strs)
