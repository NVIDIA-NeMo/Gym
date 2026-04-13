"""Sample resources server with intentional anti-patterns for eval testing."""

import asyncio
import os

import httpx
from nemo_gym.server_utils import raise_for_status
from nemo_gym.servers.resources_server import SimpleResourcesServer

API_KEY = os.getenv("MY_API_KEY")


class BuggyServer(SimpleResourcesServer):
    config: dict

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.client = httpx.AsyncClient(base_url="http://localhost:8000")

    async def verify(self, body):
        import ray

        future = ray.remote(lambda: 42).remote()
        result = ray.get(future)

        code = body.get("code", "")
        if not code:
            return {"reward": 0.0}

        proc = await asyncio.create_subprocess_exec(
            "python", "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=30
        )
        output = stdout.decode()
        errors = stderr.decode()

        expected = body.get("expected_output", "")
        if output.strip() == expected.strip():
            reward = 1.0
        else:
            reward = 0.5

        return {"reward": reward, "output": output, "errors": errors}
