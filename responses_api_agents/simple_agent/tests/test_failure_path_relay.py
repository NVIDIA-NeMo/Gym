# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The failure path through the simple agent: hop 2 (its verify response) and hop 4 (the collector's saved row).

Hops 1 and 3 and the helpers live in tests/unit_tests/test_failure_path_chain.py, which owns no agent code.
"""

import pytest

from nemo_gym.rollout_collection import RolloutCollectionHelper
from responses_api_agents.simple_agent.app import SimpleAgentVerifyResponse
from tests.unit_tests.test_failure_path_chain import _ROW, _collector, _judge_row, _Response


async def test_hop2_simple_agent_keeps_the_path_on_its_verify_response() -> None:
    row = await _judge_row(401)
    relayed = SimpleAgentVerifyResponse.model_validate(row).model_dump(mode="json")
    assert relayed["_ng_failure_path"] == row["_ng_failure_path"]
    assert relayed["failure_kind"] == "judge_credentials_rejected"
    assert relayed["_ng_failure_path"][0]["exception_type"] == "nemo_gym.judge.JudgeCredentialsRejected"


async def test_hop4_collector_saves_the_relayed_flat_row_with_its_path(monkeypatch: pytest.MonkeyPatch) -> None:
    row = SimpleAgentVerifyResponse.model_validate(await _judge_row(429)).model_dump(mode="json")
    _collector(monkeypatch, _Response(row))
    _, result = await next(RolloutCollectionHelper().run_examples([dict(_ROW)]))
    assert result["failure_kind"] == "judge_rate_limited" and result["_ng_failure_path"][-1]["http_status"] == 429
    assert result["_ng_failure_class"] == "judge_failed"
