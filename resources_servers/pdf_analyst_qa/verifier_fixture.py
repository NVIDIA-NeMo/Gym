# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process scorer fixtures without a provider or model call."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from pydantic import Field

from nemo_gym.client import SESSION_ID_KEY
from resources_servers.pdf_analyst_qa.task_data import PdfAnalystQaTask


if TYPE_CHECKING:
    from resources_servers.pdf_analyst_qa.app import (
        PdfAnalystQaResourcesServer,
        PdfAnalystQaVerifyRequest,
        PdfAnalystQaVerifyResponse,
    )


def create_fixture_server() -> PdfAnalystQaResourcesServer:
    from resources_servers.pdf_analyst_qa.app import PdfAnalystQaResourcesServer, PdfAnalystQaResourcesServerConfig
    from resources_servers.pdf_analyst_qa.documents import MountedDocumentsConfig

    config = PdfAnalystQaResourcesServerConfig(
        host="127.0.0.1",
        port=9000,
        name="pdf_analyst_qa_fixture",
        entrypoint="app.py",
        dataset_root=".",
        document_image="fixture",
        model_server={"type": "responses_api_models", "name": "policy_model"},
        mounted_documents=MountedDocumentsConfig(prefix="example"),
    )
    return PdfAnalystQaResourcesServer.model_construct(config=config)


async def invoke_fixture(server: PdfAnalystQaResourcesServer, body: Any) -> PdfAnalystQaVerifyResponse:
    from resources_servers.pdf_analyst_qa.app import _SessionState

    session_id = "fixture-session"
    if not body.missing_reference:
        task = PdfAnalystQaTask(
            task_id=body.task_id,
            workspace_id=body.workspace_id,
            data_split="rl",
            question="Fixture question",
            final_answer=body.expected_answer,
            answer_contract=body.answer_contract,
            units=body.units,
        )
        server._sessions[session_id] = _SessionState(task, body.prompt_version)
    return await server.verify(SimpleNamespace(session={SESSION_ID_KEY: session_id}), body)


def fixture_request_model() -> type[PdfAnalystQaVerifyRequest]:
    from resources_servers.pdf_analyst_qa.app import PdfAnalystQaVerifyRequest
    from resources_servers.pdf_analyst_qa.verifiers.answer_formats import AnswerFormat

    class PdfAnalystQaFixtureRequest(PdfAnalystQaVerifyRequest):
        expected_answer: str = Field(default="23", exclude=True)
        answer_contract: AnswerFormat = Field(default="bare_value", exclude=True)
        units: str | None = Field(default=None, exclude=True)
        missing_reference: bool = Field(default=False, exclude=True)

    return PdfAnalystQaFixtureRequest


FIXTURE_CASES_PATH = Path(__file__).parent / "tests/verifier_cases.jsonl"
