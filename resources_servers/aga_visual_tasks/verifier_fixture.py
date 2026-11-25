# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline fixtures for the bridge to the original math, MCQA and string answer graders.

Each case goes through the bridge's own scoring request (`scoring_body_for`) and is scored by the original grader
server in process, over its real `/verify` route. The private data export is not needed, and the math grader runs
without its judge.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
from fastapi import FastAPI

from nemo_gym.client import ServerClient
from nemo_gym.verification import VerifierFixture
from resources_servers.aga_visual_tasks.app import Verify, scoring_body_for
from resources_servers.math_with_judge.app import (
    LibraryJudgeMathResourcesServer,
    LibraryJudgeMathResourcesServerConfig,
)
from resources_servers.mcqa.app import MCQAResourcesServer, MCQAResourcesServerConfig
from resources_servers.string_match.app import StringMatchResourcesServer, StringMatchResourcesServerConfig


class FixtureVerifier:
    def __init__(self) -> None:
        client = MagicMock(spec=ServerClient)

        def common(name: str) -> dict[str, Any]:
            return dict(name=name, host="127.0.0.1", port=1, entrypoint="")

        servers = {
            "math": LibraryJudgeMathResourcesServer(
                config=LibraryJudgeMathResourcesServerConfig(
                    **common("math"),
                    should_use_judge=False,
                    judge_model_server={"type": "responses_api_models", "name": "never-call"},
                    judge_responses_create_params={"input": []},
                ),
                server_client=client,
            ),
            "mcqa": MCQAResourcesServer(config=MCQAResourcesServerConfig(**common("mcqa")), server_client=client),
            "string": StringMatchResourcesServer(
                config=StringMatchResourcesServerConfig(**common("string")), server_client=client
            ),
        }
        self._apps: dict[str, FastAPI] = {}
        for name, server in servers.items():
            app = FastAPI()
            app.post("/verify")(server.verify)
            self._apps[name] = app

    async def verify(self, body: Verify) -> float:
        grader = body.image_tools_base_agent_ref["name"]
        if grader not in self._apps:
            raise ValueError(f"Unknown original grader: {grader}")
        transport = httpx.ASGITransport(app=self._apps[grader])
        async with httpx.AsyncClient(transport=transport, base_url="http://grader") as http:
            reply = await http.post("/verify", json=scoring_body_for(body))
        reply.raise_for_status()
        return float(reply.json()["reward"])


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=FixtureVerifier,
    request_model=Verify,
    cases_path=Path(__file__).parent / "tests/verifier_cases.jsonl",
)
