# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free verifier fixture. Its contracts use code checks only, so no judge is called."""

from unittest.mock import MagicMock

from nemo_gym.client import ServerClient


def create_fixture_server(server_class: type, config_class: type, **config: object):
    """The caller passes its own classes: the fixture loader imports the app module in isolation. `config` sets the
    server's own fields, such as its entrypoint."""
    config = {
        "name": "partner_if",
        "host": "127.0.0.1",
        "port": 0,
        "entrypoint": "app.py",
        "judge_model_server": {"type": "responses_api_models", "name": "partner_if_judge"},
    } | config
    return server_class(config=config_class(**config), server_client=MagicMock(spec=ServerClient))
