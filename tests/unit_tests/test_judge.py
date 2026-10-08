# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from pydantic import BaseModel, ConfigDict

from nemo_gym.judge import JudgeError, call_judge, judge_failsafe, reraise_judge_errors


class _Req(BaseModel):
    model_config = ConfigDict(extra="allow")

    response: dict = {}


class _Verdict(BaseModel):
    verdict: str


def _client(*, status: int, body: bytes) -> MagicMock:
    """A ServerClient stand-in whose post() returns an aiohttp-shaped response."""
    resp = MagicMock(ok=status < 400, status=status)
    resp.read = AsyncMock(return_value=body)
    resp.content.read = AsyncMock(return_value=body)
    resp.raise_for_status = MagicMock(side_effect=RuntimeError(f"HTTP {status}"))
    return MagicMock(post=AsyncMock(return_value=resp))


async def _call(client) -> _Verdict:
    return await call_judge(client, server_name="judge", url_path="/v1/responses", json={}, response_model=_Verdict)


class TestCallJudge:
    @pytest.mark.asyncio
    async def test_success_parses_response_model(self) -> None:
        assert (await _call(_client(status=200, body=b'{"verdict": "CORRECT"}'))).verdict == "CORRECT"

    @pytest.mark.asyncio
    async def test_http_error_reported_as_http_not_validation_error(self) -> None:
        # A 401 body is not a valid _Verdict. Without the status check the failure
        # would surface as an opaque pydantic ValidationError on the error payload.
        with pytest.raises(JudgeError, match="HTTP 401"):
            await _call(_client(status=401, body=b'{"error": {"code": "401"}}'))

    @pytest.mark.asyncio
    async def test_transport_error_becomes_judge_error(self) -> None:
        with pytest.raises(JudgeError, match="judge unreachable"):
            await _call(MagicMock(post=AsyncMock(side_effect=ConnectionError("judge unreachable"))))


class TestRunJudge:
    @pytest.mark.asyncio
    async def test_success_returns_result(self) -> None:
        async def ok():
            return "verdict"

        assert await reraise_judge_errors(ok()) == "verdict"

    @pytest.mark.asyncio
    async def test_exception_reraised_as_judge_error(self) -> None:
        async def boom():
            raise RuntimeError("judge timeout")

        with pytest.raises(JudgeError, match="RuntimeError: judge timeout"):
            await reraise_judge_errors(boom())


class TestJudgeFailsafe:
    @pytest.mark.asyncio
    async def test_success_passes_through(self) -> None:
        async def verify(body):
            return {"reward": 1.0}

        assert await judge_failsafe(verify)(_Req()) == {"reward": 1.0}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("by_keyword", [False, True])
    async def test_judge_error_routed_to_sidecar(self, by_keyword: bool) -> None:
        async def verify(body):
            raise JudgeError("RuntimeError: judge 401")

        req = _Req(response={"final": "answer"})
        # FastAPI injects by keyword (kwargs["body"]); direct callers pass positionally.
        out = await (judge_failsafe(verify)(body=req) if by_keyword else judge_failsafe(verify)(req))
        data = orjson.loads(out.body)
        assert data["reward"] == 0.0
        assert data["mask_sample"] is True
        assert data["failure_kind"] == "judge_failed"
        assert data["failure_reason"] == "RuntimeError: judge 401"
        assert data["instance_config"]["mask_sample"] is True
        assert data["_ng_failure_class"] == "judge_failed"
        assert data["_ng_failure_judge_error"] == "RuntimeError: judge 401"
        # The model's final output is carried for a later judge-only replay.
        assert data["response"] == {"final": "answer"}
        # Transient: never terminal, so resume re-dispatches it.
        assert "_ng_failure_terminal" not in data

    @pytest.mark.asyncio
    @pytest.mark.parametrize("instance_config", [None, "invalid", {"task": "keep", "mask_sample": False}])
    async def test_mask_contract_overrides_request_defaults_without_mutation(self, instance_config) -> None:
        async def verify(body):
            raise JudgeError("judge offline " + "x" * 3000)

        request = _Req(
            response={"final": "answer"},
            mask_sample=False,
            failure_kind=None,
            instance_config=instance_config,
        )
        original = request.model_dump()
        data = orjson.loads((await judge_failsafe(verify)(request)).body)
        assert data["mask_sample"] is True and data["failure_kind"] == "judge_failed"
        assert data["instance_config"] == {
            **(instance_config if isinstance(instance_config, dict) else {}),
            "mask_sample": True,
        }
        assert data["failure_reason"].startswith("judge offline")
        assert len(data["failure_reason"]) == 2000
        assert data["_ng_failure_judge_error"] == data["failure_reason"]
        assert request.model_dump() == original
