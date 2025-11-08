# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional

import psutil
from pydantic import BaseModel

from nemo_gym.server_utils import ServerStatus


class ServerProcessInfo(BaseModel):
    """Information about a running server process"""

    pid: int
    server_type: str  # "resources_server", "responses_api_model", "responses_api_agent"
    name: str  # e.g. "simple_weather", "policy_model", etc.
    process_name: str  # config path from env var
    host: Optional[str]
    port: Optional[int]
    url: Optional[str]
    uptime_seconds: float
    status: ServerStatus  # "success", "connection_error", etc.
    entrypoint: Optional[str]


def parse_server_info(proc, cmdline: List[str], env: dict) -> Optional[ServerProcessInfo]:
    """TODO: Parse server info"""
    pass


class StatusCommand:
    """Main class to check server status"""

    def discover_servers() -> List[ServerProcessInfo]:
        """Find all running NeMo Gym server processes"""
        servers = []
        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time", "environ"]):
            try:
                env = proc.info.get("environ", {})
                if not env:
                    continue

                cmdline = proc.info["cmdline"]
                server_info = parse_server_info(proc, cmdline, env)

                if server_info:
                    servers.append(server_info)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return servers
