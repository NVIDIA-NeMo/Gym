from asyncio import run

from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()


async def main():
    task_1a = await server_client.post(
        server_name="openai_model",
        url_path="/v1/responses",
        json={"input": [{"role": "user", "content": "hello"}]},
    )
    task_1b = await server_client.post(
        server_name="openai_model",
        url_path="/v1/responses",
        json={
            "input": [
                {"role": "user", "content": "what's it like in sf?"},
            ],
            "tools": [
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
        },
    )
    task_2a = await server_client.post(
        server_name="openai_model",
        url_path="/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    task_2b = await server_client.post(
        server_name="openai_model",
        url_path="/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "what's it like in sf?"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
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
                    },
                }
            ],
        },
    )
    print(task_1a.json())
    print(task_1b.json())
    print(task_2a.json())
    print(task_2b.json())


if __name__ == "__main__":
    run(main())
