# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.question_bank_judge.math_verifier import MathLibraryVerifier


@pytest.fixture
def verifier() -> MathLibraryVerifier:
    return MathLibraryVerifier(
        server_client=MagicMock(spec=ServerClient),
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )


@pytest.mark.parametrize(
    ("expected", "selected", "equivalent"),
    [
        ("0.5", r"\frac{1}{2}", True),
        (r"\frac{2}{3}", r"\frac{4}{6}", True),
        ("2", "-2", False),
        ("2", "3", False),
        ("", "2", False),
        ("2", "", False),
    ],
)
async def test_reuses_actual_math_verify(verifier, expected, selected, equivalent) -> None:
    assert await verifier.verify(expected, selected) is equivalent
    verifier._server.server_client.post.assert_not_called()


async def test_parse_or_timeout_failure_requires_judge_fallback(verifier) -> None:
    verifier._server._verify_answer_with_library_async = AsyncMock(return_value=(0.0, None))
    assert await verifier.verify("2", "3") is False
    verifier._server._verify_answer_with_library_async.assert_awaited_once_with("2", r"\boxed{3}")


async def test_setup_errors_propagate_for_failure_classification(verifier) -> None:
    verifier._server._verify_answer_with_library_async = AsyncMock(side_effect=OSError("cannot create child process"))
    with pytest.raises(OSError, match="cannot create child"):
        await verifier.verify("2", "2")
