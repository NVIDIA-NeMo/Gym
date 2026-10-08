# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources server for the checkpoint e2e suite whose ``/verify`` blocks while a flag file exists.

``SLOW_VERIFY_MODE`` sets whether its verification may be replayed (``wait`` by default).
Every verification appends a line to the log file, so a test can tell whether a restore re-ran it.
"""

import asyncio
import os
from pathlib import Path

from nemo_gym.base_resources_server import BaseVerifyRequest, BaseVerifyResponse
from resources_servers.example_single_tool_call.app import SimpleWeatherResourcesServer


class SlowVerifyResourcesServer(SimpleWeatherResourcesServer):
    checkpoint_verify = os.environ.get("SLOW_VERIFY_MODE", "wait")

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        with open(os.environ["SLOW_VERIFY_LOG"], "a") as log:
            log.write("verify\n")
        while Path(os.environ["SLOW_VERIFY_FLAG"]).exists():
            await asyncio.sleep(0.05)
        return await super().verify(body)


if __name__ == "__main__":
    SlowVerifyResourcesServer.run_webserver()
