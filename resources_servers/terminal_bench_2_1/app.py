# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.verifier_fixture import VerifierFixture


class TerminalBench21ResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class TerminalBench21VerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    expected_answer: str


class TerminalBench21Verifier:
    async def verify(self, body: TerminalBench21VerifyRequest) -> BaseVerifyResponse:
        reward = float(body.response.output_text.strip() == body.expected_answer.strip())
        return BaseVerifyResponse(**body.model_dump(), reward=reward)


class TerminalBench21ResourcesServer(TerminalBench21Verifier, SimpleResourcesServer):
    config: TerminalBench21ResourcesServerConfig


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=TerminalBench21Verifier,
    request_model=TerminalBench21VerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    TerminalBench21ResourcesServer.run_webserver()
