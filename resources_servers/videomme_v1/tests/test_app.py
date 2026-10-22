# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi.testclient import TestClient

from resources_servers.video_mcqa.tests.test_app import make_server, request


def test_videomme_v1_resource_routes(tmp_path) -> None:
    server, rows, _ = make_server(tmp_path, benchmark="videomme_v1")
    with TestClient(server.setup_webserver()) as client:
        result = client.post("/verify", json=request(rows[0]).model_dump())
        assert result.status_code == 200
        assert result.json()["outcome"] == "correct"
        assert result.json()["task_id"].startswith("videomme_v1:")
        assert server.checkpoint_verify == "replay"
