# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic scorer cases using the real trusted-file lookup and verify method."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from starlette.requests import Request

from resources_servers.officeqa.verifiers.officeqa_pro_task import OfficeQAProTask


if TYPE_CHECKING:
    from resources_servers.officeqa.app import OfficeQAFixtureRequest, OfficeQAResourcesServer, OfficeQAVerifyResponse


def create_fixture_server() -> OfficeQAResourcesServer:
    """Construct an in-process verifier without sandbox or model dependencies."""
    from resources_servers.officeqa.app import OfficeQAResourcesServer, OfficeQAResourcesServerConfig

    config = OfficeQAResourcesServerConfig(
        host="127.0.0.1", port=9000, name="officeqa_fixture", entrypoint="app.py", workspace_root="."
    )
    return OfficeQAResourcesServer.model_construct(config=config)


async def invoke_fixture(server: OfficeQAResourcesServer, body: OfficeQAFixtureRequest) -> OfficeQAVerifyResponse:
    """Keep synthetic trusted answers outside request serialization and the agent view."""
    with TemporaryDirectory() as root:
        task = OfficeQAProTask(
            task_id=body.task_id,
            workspace_id=body.workspace_id,
            uid="synthetic-verifier-fixture",
            question="Synthetic scorer case.",
            expected_answer=body.expected_answer,
            source_revision="0" * 40,
        )
        directory = Path(root) / "tasks"
        directory.mkdir()
        if not body.missing_reference:
            (directory / (body.task_id + ".json")).write_text(task.model_dump_json())
        server.config.workspace_root = root
        return await server.verify(Request({"type": "http"}), body)
