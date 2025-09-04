import json

from asyncio import run

from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()
task = server_client.post(
    server_name="workbench",
    url_path="/email_reply_email",
    json={
        "email_id": "00000057",
        "body": "Thanks for the update - I will get back to you tomorrow.",
    },
)
result = run(task)
print(json.dumps(result.json(), indent=4))
