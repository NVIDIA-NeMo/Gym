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
from typing import List, Optional

import psutil
import requests
from omegaconf import OmegaConf

from nemo_gym.global_config import (
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
    get_first_server_config_dict,
)
from nemo_gym.server_utils import ServerInstanceBase, ServerStatus


class ServerProcessInfo(ServerInstanceBase):
    """Information about a running server process"""

    pid: int
    status: ServerStatus
    uptime_seconds: float


def parse_server_info(proc, cmdline: List[str], env: dict) -> Optional[ServerProcessInfo]:
    """Takes process, command line, and env and returns server process information"""
    try:
        process_name = env.get(NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME)
        if not process_name:
            return None

        config_dict_yaml = env.get(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME)
        if not config_dict_yaml:
            return None

        global_config_dict = OmegaConf.create(config_dict_yaml)

        server_config_dict = get_first_server_config_dict(global_config_dict, process_name)

        top_level_config = global_config_dict[process_name]
        server_type = list(top_level_config.keys())[0]
        server_type_config = top_level_config[server_type]
        server_name = list(server_type_config.keys())[0]

        host = server_config_dict.get("host")
        port = server_config_dict.get("port")

        url = f"http://{host}:{port}" if host and port else None

        entrypoint = None
        for i, arg in enumerate(cmdline):
            if arg == "python" and i + 1 < len(cmdline):
                entrypoint = cmdline[i + 1]
                break

        current_time = time()
        create_time = proc.info["create_time"]
        uptime_seconds = current_time - create_time

        return ServerProcessInfo(
            pid=proc.info["pid"],
            server_type=server_type,
            name=server_name,
            process_name=process_name,
            host=host,
            port=port,
            url=url,
            uptime_seconds=uptime_seconds,
            status="unknown_error",
            entrypoint=entrypoint,
        )
    except (KeyError, IndexError, AttributeError, TypeError) as e:
        print(f"Error parsing PID {proc.info['pid']}: {type(e).__name__}: {e}")
        return None


class StatusCommand:
    """Main class to check server status"""

    def check_health(self, server_info: ServerProcessInfo) -> ServerStatus:
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

    def discover_servers(self) -> List[ServerProcessInfo]:
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
                    server_info.status = self.check_health(server_info)
                    servers.append(server_info)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return servers

    def display_status(self, servers: List[ServerProcessInfo]) -> None:
        """Show server info in a table"""

        if not servers:
            print("No NeMo Gym servers found running.")
            return

        print("\nNeMo Gym Server Status:\n")

        def format_uptime(uptime_seconds: float) -> str:
            """Format uptime in a human readable format"""
            minutes, seconds = divmod(uptime_seconds, 60)
            hours, minutes = divmod(minutes, 60)
            days, hours = divmod(hours, 24)
            return f"{int(days)}d {int(hours)}h {int(minutes)}m {seconds:.1f}s"

        healthy_count = 0
        for server in servers:
            status_icon = "✓" if server.status == "success" else "✗"
            if server.status == "success":
                healthy_count += 1

            uptime_str = format_uptime(server.uptime_seconds)

            print(
                f"{status_icon} {server.name} ({server.server_type}) "
                f"Port: {server.port} PID: {server.pid} "
                f"Uptime: {uptime_str}"
            )

        print(f"\n{len(servers)} servers found ({healthy_count} healthy, {len(servers) - healthy_count} unhealthy)\n")
