# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free verifier fixture. Its contracts use code checks only, so no judge is called."""

from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient


def create_fixture_server(server_class: type, config_class: type):
    """The caller passes its own classes: the fixture loader imports the app module in isolation."""
    config = config_class(
        name="partner_if",
        host="127.0.0.1",
        port=0,
        entrypoint="app.py",
        judge_model_server={"type": "responses_api_models", "name": "partner_if_judge"},
    )
    return server_class(config=config, server_client=MagicMock(spec=ServerClient))
