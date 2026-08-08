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
"""Snapshot the tools/list of a running EnterpriseOps MCP gym server to JSON.

EOG task JSONs carry only tool *names* (selected_tools); the schemas live behind the live
MCP servers. NeMo Gym dataset rows must carry full tool definitions, so we snapshot each
domain's tools once and let the converter bake them into the dataset. This also removes the
per-task tools/list discovery the upstream harness performs.

Usage:
    python resources_servers/enterpriseops_gym/snapshot_tools.py \
        --gym-url http://localhost:8001 --gym-name sn-csm-server \
        --output resources_servers/enterpriseops_gym/data/tools/csm.json
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from resources_servers.enterpriseops_gym.mcp_client import MCPGymClient


async def snapshot(gym_url: str, gym_name: str, mcp_endpoint: str, output: Path) -> int:
    client = MCPGymClient(base_url=gym_url, mcp_endpoint=mcp_endpoint)
    tools = await client.list_tools()
    if not tools:
        raise RuntimeError(f"No tools returned by {gym_url}{mcp_endpoint} — is the gym server running?")

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(
            {
                "gym_name": gym_name,
                "gym_url": gym_url,
                "snapshot_at": datetime.now(timezone.utc).isoformat(),
                "tools": tools,
            },
            f,
            indent=2,
        )
    print(f"Wrote {len(tools)} tools for gym '{gym_name}' to {output}")
    return len(tools)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gym-url", required=True, help="Base URL of the running MCP gym server")
    parser.add_argument("--gym-name", required=True, help="EOG gym server name (e.g. sn-csm-server)")
    parser.add_argument("--mcp-endpoint", default="/mcp")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    asyncio.run(snapshot(args.gym_url, args.gym_name, args.mcp_endpoint, args.output))


if __name__ == "__main__":
    main()
