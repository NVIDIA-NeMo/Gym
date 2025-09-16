import json
from asyncio import run

from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()
task = server_client.post(
    server_name="mini_swe_main_agent",
    url_path="/run",
    json={
        "responses_create_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "input": [],
        },
        "instance_id": "pydantic__pydantic-5506",
        "subset": "gym",
        "split": "train",
    },
)
result = run(task)
print(json.dumps(result.json(), indent=4))
