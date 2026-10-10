# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources server for the checkpoint e2e suite that cannot capture its sessions.

It is the weather server declared ``restart_only``, the default for servers nobody has audited,
so an episode that uses it is a restart: it never holds up a checkpoint,
and after a crash it starts over from its input.
"""

from resources_servers.example_single_tool_call.app import SimpleWeatherResourcesServer


class RestartOnlyWeatherResourcesServer(SimpleWeatherResourcesServer):
    checkpoint_mode = "restart_only"


if __name__ == "__main__":
    RestartOnlyWeatherResourcesServer.run_webserver()
