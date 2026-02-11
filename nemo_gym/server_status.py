# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from time import time
from typing import List

import requests
from devtools import pprint

from nemo_gym.server_utils import ServerClient, ServerInstanceDisplayConfig, ServerStatus


class StatusCommand:
    """Main class to check server status"""

    def check_health(self, server_info: ServerInstanceDisplayConfig) -> ServerStatus:
        """Check if server is responding"""
        if not server_info.url:
            return "unknown_error"

        try:
            requests.get(server_info.url, timeout=2)
            return "success"
        except requests.exceptions.ConnectionError:
            return "connection_error"
        except requests.exceptions.Timeout:
            return "timeout"
        except Exception:
            return "unknown_error"

    def discover_servers(self) -> List[ServerInstanceDisplayConfig]:
        """Find all running NeMo Gym server processes"""

        try:
            head_server_config = ServerClient.load_head_server_config()
            head_url = f"http://{head_server_config.host}:{head_server_config.port}"

            response = requests.get(f"{head_url}/server_instances", timeout=5)
            response.raise_for_status()
            instances = response.json()

            servers = []
            current_time = time()

            for inst in instances:
                uptime = current_time - inst.get("start_time", current_time)
                server_info = ServerInstanceDisplayConfig(
                    process_name=inst["process_name"],
                    server_type=inst["server_type"],
                    name=inst["name"],
                    host=inst.get("host"),
                    port=inst.get("port"),
                    url=inst.get("url"),
                    entrypoint=inst.get("entrypoint"),
                    pid=inst.get("pid"),
                    stdout_log=inst.get("stdout_log"),
                    stderr_log=inst.get("stderr_log"),
                    uptime_seconds=uptime,
                    status="unknown_error",
                )
                server_info.status = self.check_health(server_info)
                servers.append(server_info)

            return servers

        except (requests.RequestException, ConnectionError) as e:
            print(f"""
Could not connect to head server: {e}
Is the head server running? Start it with: `ng_run`
            """)
            return []

    def display_status(self, servers: List[ServerInstanceDisplayConfig]) -> None:
        """Show server info in a table"""

        def format_uptime(uptime_seconds: float) -> str:
            """Format uptime in a human readable format"""
            minutes, seconds = divmod(uptime_seconds, 60)
            hours, minutes = divmod(minutes, 60)
            days, hours = divmod(hours, 24)
            return f"{int(days)}d {int(hours)}h {int(minutes)}m {seconds:.1f}s"

        if not servers:
            print("No NeMo Gym servers found running.")
            return

        print("\nNeMo Gym Server Status:\n")

        for i, server in enumerate(servers, 1):
            status_icon = "✓" if server.status == "success" else "✗"
            print(f"[{i}] {status_icon} {server.process_name} ({server.server_type}/{server.name})")
            display_dict = {
                "server_type": server.server_type,
                "name": server.name,
                "port": server.port,
                "pid": server.pid,
                "uptime_seconds": format_uptime(server.uptime_seconds),
                "stdout_log": server.stdout_log,
                "stderr_log": server.stderr_log,
            }
            pprint(display_dict)

        healthy_count = sum(1 for s in servers if s.status == "success")
        print(f"""
{len(servers)} servers found ({healthy_count} healthy, {len(servers) - healthy_count} unhealthy)
""")
