# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi.testclient import TestClient

from resources_servers.video_mcqa.tests.test_app import make_server, request


def test_lvbench_resource_routes(tmp_path) -> None:
    server, rows, _ = make_server(tmp_path, benchmark="lvbench")
    with TestClient(server.setup_webserver()) as client:
        result = client.post("/verify", json=request(rows[0]).model_dump())
        assert result.status_code == 200
        assert result.json()["outcome"] == "correct"
        assert result.json()["task_id"].startswith("lvbench:")
        assert server.checkpoint_verify == "replay"
