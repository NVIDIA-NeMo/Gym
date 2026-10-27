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
"""Verifier fixture: each case seeds a campaign, adds its designs with one scoring call, then verifies, all through
the server's own routes, so the cases exercise the same path as a rollout."""

from typing import Any
from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient


def create_server():
    from resources_servers.rna_design.app import RnaDesignResourcesServer, RnaDesignResourcesServerConfig

    config = RnaDesignResourcesServerConfig(host="127.0.0.1", port=0, entrypoint="app.py", name="rna_design")
    return RnaDesignResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def invoke(server, request) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    body = request.model_dump(mode="json")
    with TestClient(server.setup_webserver()) as client:
        client.post("/seed_session", json=body).raise_for_status()
        scored = client.post(
            "/score_fold_match", json={"sequences": body["designs"], "target": body["target"]["name"]}
        )
        scored.raise_for_status()
        return client.post("/verify", json=body).raise_for_status().json()
