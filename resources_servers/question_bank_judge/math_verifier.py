# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reuse Gym's bounded math-verify implementation before semantic judgment."""

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.math_with_judge.app import (
    LibraryJudgeMathResourcesServer,
    LibraryJudgeMathResourcesServerConfig,
)


class MathLibraryVerifier:
    """A library-only view of the existing math_with_judge resources server.

    Construction does not start a webserver or call a model. The actual verifier,
    concurrency bound and terminable subprocess implementation are shared with
    math_with_judge. A False return means semantic judgment is still needed;
    it must never be interpreted as a resolved incorrect answer by the caller.
    """

    def __init__(
        self,
        *,
        server_client: ServerClient,
        judge_model_server: ModelServerRef,
        judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming,
        timeout_seconds: float = 10.0,
        max_concurrency: int = 8,
    ) -> None:
        self._server = LibraryJudgeMathResourcesServer(
            config=LibraryJudgeMathResourcesServerConfig(
                host="127.0.0.1",
                port=1,
                name="stem_math_library",
                entrypoint="",
                should_use_judge=False,
                judge_model_server=judge_model_server,
                judge_responses_create_params=judge_responses_create_params,
                library_verifier_timeout_seconds=timeout_seconds,
                library_verifier_max_concurrency=max_concurrency,
            ),
            server_client=server_client,
        )

    async def verify(self, expected_answer: str, boxed_answer: str) -> bool:
        """Check an already selected final box; return True only on equivalence.

        The caller selects the last box before calling this method. Only that
        content is passed to math-verify, so preceding answers cannot earn credit.
        Existing library timeout/parse failures return False and require the
        caller's judge fallback. Dependency/setup errors propagate to the caller.
        """
        if not expected_answer.strip() or not boxed_answer.strip():
            return False
        score, _ = await self._server._verify_answer_with_library_async(
            expected_answer, "\\boxed{" + boxed_answer + "}"
        )
        return score > 0.5
