import asyncio
import json

from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()
task = server_client.post(
    server_name="library_judge_math",
    url_path="/verify",
    json={
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": "What is 2 + 2?",
                },
            ]
        },
        "response": {
            "id": "response_1",
            "created_at": 1.0,
            "model": "model_1",
            "object": "response",
            "output": [
                {
                    "id": "message_1",
                    "content": [
                        {
                            "annotations": [],
                            "text": "5",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
        },
        "question": "What is 2 + 2?",
        "expected_answer": "4",
    },
)
result = asyncio.run(task)
print(json.dumps(result.json(), indent=4))
