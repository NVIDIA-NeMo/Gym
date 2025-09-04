from asyncio import run

from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()


async def main():
    task_1 = await server_client.post(
        server_name="openai_model",
        url_path="/v1/responses",
        json={"input": "hello"},
    )
    task_2 = await server_client.post(
        server_name="openai_model",
        url_path="/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    print(task_1.json())
    print(task_2.json())


if __name__ == "__main__":
    run(main())
