# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Smoke-test client: POST a single task to a running benchflow_agent /run endpoint.

The instance_id is "<dataset_alias>::<task_name>"; the task name must exist in the
configured tasks_dir. Replace the task below with one present in your tasks dir.
"""

import json
from asyncio import run

from nemo_gym.server_utils import ServerClient


async def main():
    server_client = ServerClient.load_from_global_config()
    result = await server_client.post(
        server_name="benchflow_agent",
        url_path="/run",
        json={
            "responses_create_params": {
                "input": [],
            },
            "instance_id": "skillsbench::3d-scan-calc",
        },
    )
    data = await result.json()
    print(json.dumps(data, indent=4))


run(main())
