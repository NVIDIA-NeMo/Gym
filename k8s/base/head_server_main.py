# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone entrypoint for the head server on K8s.

The normal head server is started as a background thread by RunHelper.
On K8s it runs as its own pod, so we need a main that starts it directly.
"""
import uvicorn

from nemo_gym.global_config import get_global_config_dict
from nemo_gym.server_utils import HeadServer

global_config_dict = get_global_config_dict()
head_config_dict = global_config_dict["head_server"]

server = HeadServer(config={"host": head_config_dict["host"], "port": head_config_dict["port"]})
app = server.setup_webserver()

if __name__ == "__main__":
    uvicorn.run(app, host=head_config_dict["host"], port=head_config_dict["port"])
