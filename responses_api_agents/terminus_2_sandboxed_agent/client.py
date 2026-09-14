# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from asyncio import run

from nemo_gym.global_config import get_global_config_dict
from nemo_gym.server_utils import ServerClient


async def main():
    with open(get_global_config_dict()["benchmark_jsonl"]) as file:
        first_example = json.loads(next(file))

    server_client = ServerClient.load_from_global_config()
    result = await server_client.post(
        server_name="terminus_2_sandboxed_agent",
        url_path="/run",
        json=first_example,
    )
    print(json.dumps(await result.json(), indent=4))


if __name__ == "__main__":
    run(main())
