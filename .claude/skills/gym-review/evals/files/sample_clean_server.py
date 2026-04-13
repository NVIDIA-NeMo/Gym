"""Sample clean resources server — no anti-patterns.

review.py should report zero findings on this file.
"""

import asyncio

from nemo_gym.server_utils import request, raise_for_status
from nemo_gym.servers.resources_server import SimpleResourcesServer


class CleanServerConfig:
    timeout: int = 30
    num_processes: int = 4


class CleanServer(SimpleResourcesServer):
    config: CleanServerConfig

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.semaphore = asyncio.Semaphore(self.config.num_processes)

    async def verify(self, body):
        code = body.get("code", "")
        if not code:
            return {"reward": 0.0}

        async with self.semaphore:
            proc = await asyncio.create_subprocess_exec(
                "python", "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.config.timeout
            )
            output = stdout.decode(errors="replace")
            errors = stderr.decode(errors="replace")

        expected = body.get("expected_output", "")
        if output.strip() == expected.strip():
            reward = 1.0
        else:
            reward = 0.0

        return {"reward": reward, "output": output, "errors": errors}
