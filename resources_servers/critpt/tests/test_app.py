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
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.critpt.app import (
    CritPtResourcesServer,
    CritPtResourcesServerConfig,
    CritPtVerifyRequest,
    _extract_code,
)


def _make_config(batch_size: int = 70, **kwargs) -> CritPtResourcesServerConfig:
    return CritPtResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        api_key="test-key",
        batch_size=batch_size,
        **kwargs,
    )


def _make_server(config: CritPtResourcesServerConfig | None = None) -> CritPtResourcesServer:
    return CritPtResourcesServer(
        config=config or _make_config(),
        server_client=MagicMock(spec=ServerClient),
    )


def _make_verify_request(output_text: str, problem_id: str = "1") -> CritPtVerifyRequest:
    response = NeMoGymResponse(
        id="test-id",
        created_at=1234.5,
        model="test-model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg-id",
                content=[NeMoGymResponseOutputText(annotations=[], text=output_text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )
    return CritPtVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=response,
        problem_id=problem_id,
    )


def _mock_api(api_result: dict):
    """Patch the module-level request + raise_for_status. Returns the mock_request handle."""
    request_patch = patch("resources_servers.critpt.app.request")
    rfs_patch = patch("resources_servers.critpt.app.raise_for_status", new_callable=AsyncMock)
    mock_request = request_patch.start()
    rfs_patch.start()
    mock_response = AsyncMock()
    mock_response.json = AsyncMock(return_value=api_result)
    mock_request.return_value = mock_response
    return mock_request, [request_patch, rfs_patch]


def _stop_patches(patches):
    for p in patches:
        p.stop()


class TestExtractCode:
    def test_fenced_python_block(self):
        text = "Here is the answer:\n```python\ndef solve():\n    return 42\n```"
        assert _extract_code(text) == "def solve():\n    return 42"

    def test_fenced_block_no_language(self):
        text = "```\ndef solve():\n    return 42\n```"
        assert _extract_code(text) == "def solve():\n    return 42"

    def test_multiple_blocks_returns_last(self):
        text = "```python\ndef first():\n    pass\n```\nThen:\n```python\ndef last():\n    return 1\n```"
        assert _extract_code(text) == "def last():\n    return 1"

    def test_no_fence_returns_stripped_text(self):
        text = "  def solve():\n    return 42  "
        assert _extract_code(text) == "def solve():\n    return 42"

    def test_empty_string_returns_empty(self):
        assert _extract_code("") == ""


class TestApp:
    def test_sanity(self):
        _make_server()

    @pytest.mark.asyncio
    async def test_partial_batch_waits(self):
        """With batch_size=3, a single verify() call should hang (batch not full)."""
        server = _make_server(_make_config(batch_size=3))

        mock_request, patches = _mock_api({"accuracy": 0.5, "timeout_rate": 0.0})
        try:
            task = asyncio.create_task(server.verify(_make_verify_request("```python\nx=1\n```", problem_id="p1")))
            # Give the task a chance to run; it should be blocked awaiting batch fill.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), timeout=0.1)
            mock_request.assert_not_called()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            _stop_patches(patches)

    @pytest.mark.asyncio
    async def test_full_batch_fires_once_and_distributes(self):
        """batch_size=3: three concurrent verify() calls → one API call, all share the aggregate."""
        server = _make_server(_make_config(batch_size=3))

        mock_request, patches = _mock_api({"accuracy": 0.667, "timeout_rate": 0.0})
        try:
            results = await asyncio.gather(
                server.verify(_make_verify_request("```python\na=1\n```", problem_id="p1")),
                server.verify(_make_verify_request("```python\nb=2\n```", problem_id="p2")),
                server.verify(_make_verify_request("```python\nc=3\n```", problem_id="p3")),
            )

            assert mock_request.call_count == 1
            payload = mock_request.call_args.kwargs["json"]
            problem_ids = {s["problem_id"] for s in payload["submissions"]}
            assert problem_ids == {"p1", "p2", "p3"}

            for r in results:
                assert r.reward == 0.667
                assert r.accuracy == 0.667
        finally:
            _stop_patches(patches)

    @pytest.mark.asyncio
    async def test_two_full_batches_fire_twice(self):
        """Six concurrent verify() calls with batch_size=3 → two API calls."""
        server = _make_server(_make_config(batch_size=3))

        mock_request, patches = _mock_api({"accuracy": 0.5, "timeout_rate": 0.0})
        try:
            await asyncio.gather(
                *(server.verify(_make_verify_request(f"```python\nx={i}\n```", problem_id=f"p{i}")) for i in range(6))
            )
            assert mock_request.call_count == 2
        finally:
            _stop_patches(patches)

    @pytest.mark.asyncio
    async def test_status_endpoint_reports_buffer_fill(self):
        """GET /status returns current buffered count and batch_size."""
        server = _make_server(_make_config(batch_size=3))
        client = TestClient(server.setup_webserver())

        # Empty buffer
        resp = client.get("/status")
        assert resp.status_code == 200
        assert resp.json() == {"buffered": 0, "batch_size": 3}

        # After one partial verify, buffer should report 1
        async def add_one():
            mock_request, patches = _mock_api({"accuracy": 0.5, "timeout_rate": 0.0})
            try:
                task = asyncio.create_task(server.verify(_make_verify_request("```python\nx=1\n```", problem_id="p1")))
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                _stop_patches(patches)

        await add_one()
        resp = client.get("/status")
        assert resp.json() == {"buffered": 1, "batch_size": 3}

    @pytest.mark.asyncio
    async def test_empty_code_still_included_in_batch(self):
        """A verify() with no extractable code still contributes to the batch (slot must be filled)."""
        server = _make_server(_make_config(batch_size=2))

        mock_request, patches = _mock_api({"accuracy": 0.5, "timeout_rate": 0.0})
        try:
            await asyncio.gather(
                server.verify(_make_verify_request("", problem_id="p1")),
                server.verify(_make_verify_request("```python\nok=1\n```", problem_id="p2")),
            )
            assert mock_request.call_count == 1
            payload = mock_request.call_args.kwargs["json"]
            submitted = {s["problem_id"]: s["generated_code"] for s in payload["submissions"]}
            assert submitted["p1"] == "```python\n```"  # empty code, still submitted
            assert "ok=1" in submitted["p2"]
        finally:
            _stop_patches(patches)
