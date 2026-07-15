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
import json
from pathlib import Path
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
    _REPO_ROOT,
    CritPtRateLimitExceeded,
    CritPtResourcesServer,
    CritPtResourcesServerConfig,
    CritPtVerifyRequest,
    _extract_code,
    _resolve_cache_dir,
)


def _make_config(batch_size: int = 70, **kwargs) -> CritPtResourcesServerConfig:
    kwargs.setdefault("api_key", "test-key")  # pragma: allowlist secret
    return CritPtResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
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


class TestResolveCacheDir:
    def test_relative_path_anchored_to_repo_root(self):
        resolved = _resolve_cache_dir(Path("results/critpt_cache"))
        assert resolved == _REPO_ROOT / "results/critpt_cache"
        assert resolved.is_absolute()

    def test_absolute_path_unchanged(self, tmp_path):
        assert _resolve_cache_dir(tmp_path) == tmp_path

    def test_server_resolves_relative_cache_dir(self, tmp_path, monkeypatch):
        """A relative cache_dir is anchored to the repo root at construction,
        independent of the process cwd (Gym runs the server from its own dir)."""
        monkeypatch.chdir(tmp_path)
        rel = "results/_pytest_critpt_cache_probe"
        try:
            server = _make_server(_make_config(cache_dir=Path(rel), unique_cache_per_run=False))
            assert server.config.cache_dir == _REPO_ROOT / rel
            assert server.config.cache_dir.is_absolute()
            assert server.config.cache_dir.exists()
        finally:
            probe = _REPO_ROOT / rel
            if probe.exists():
                probe.rmdir()

    def test_unique_cache_per_run_creates_launch_subdir(self, tmp_path):
        """With unique_cache_per_run (the default), each launch writes into its own
        subdirectory of cache_dir so independent runs never share cache files."""
        server = _make_server(_make_config(cache_dir=tmp_path))
        assert server.config.cache_dir.parent == tmp_path
        assert server.config.cache_dir != tmp_path
        assert server.config.cache_dir.is_dir()

    def test_unique_cache_per_run_isolates_independent_launches(self, tmp_path):
        """Two server launches pointed at the same cache_dir get distinct subdirs."""
        a = _make_server(_make_config(cache_dir=tmp_path))
        b = _make_server(_make_config(cache_dir=tmp_path))
        assert a.config.cache_dir != b.config.cache_dir
        assert a.config.cache_dir.parent == b.config.cache_dir.parent == tmp_path

    def test_unique_cache_per_run_disabled_uses_cache_dir_directly(self, tmp_path):
        """Opting out writes straight into cache_dir (e.g. to replay a prior run)."""
        server = _make_server(_make_config(cache_dir=tmp_path, unique_cache_per_run=False))
        assert server.config.cache_dir == tmp_path


def _mock_api(api_result: dict):
    """Patch the module-level request(). Returns the mock_request handle."""
    request_patch = patch("resources_servers.critpt.app.request")
    mock_request = request_patch.start()
    mock_response = AsyncMock()
    mock_response.ok = True
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=api_result)
    mock_request.return_value = mock_response
    return mock_request, [request_patch]


def _build_mock_response(spec: dict) -> AsyncMock:
    """Build a single mock aiohttp-like response from a compact spec.

    spec keys:
        status:  int HTTP status (default 200)
        json:    dict body returned by .json() (for 2xx)
        body:    str body returned by .text() (for non-2xx)
        headers: dict (used for 429 Retry-After / X-Ratelimit-Reset)
    """
    status = spec.get("status", 200)
    m = AsyncMock()
    m.status = status
    m.ok = status < 400
    m.headers = spec.get("headers", {})
    if m.ok:
        m.json = AsyncMock(return_value=spec.get("json", {}))
    m.text = AsyncMock(return_value=spec.get("body", ""))
    return m


def _mock_api_sequence(response_specs: list):
    """Patch module-level request() to return a sequence of responses.

    Each `await request(...)` consumes the next spec. Tests use this to drive
    rotation: e.g. `[{status: 429, ...}, {status: 200, json: {...}}]`.
    """
    request_patch = patch("resources_servers.critpt.app.request")
    mock_request = request_patch.start()
    mock_request.side_effect = [_build_mock_response(s) for s in response_specs]
    return mock_request, [request_patch]


def _stop_patches(patches):
    for p in patches:
        p.stop()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
        assert resp.json() == {"pending_batches": [], "batch_size": 3}

        # After one partial verify, one pending batch with 1 submission
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
        assert resp.json() == {"pending_batches": [1], "batch_size": 3}

    @pytest.mark.asyncio
    async def test_num_repeats_two_routes_duplicates_to_separate_batches(self):
        """num_repeats=2 sim: 2 problems × 2 repeats → 4 verifies → 2 API calls, each with 2 unique problem_ids."""
        server = _make_server(_make_config(batch_size=2))

        mock_request, patches = _mock_api({"accuracy": 0.5, "timeout_rate": 0.0})
        try:
            # Two repeats of p1 and two repeats of p2 (interleaved, mimicking concurrent rollouts).
            await asyncio.gather(
                server.verify(_make_verify_request("```python\na=1\n```", problem_id="p1")),
                server.verify(_make_verify_request("```python\nb=2\n```", problem_id="p2")),
                server.verify(_make_verify_request("```python\na2=1\n```", problem_id="p1")),
                server.verify(_make_verify_request("```python\nb2=2\n```", problem_id="p2")),
            )

            assert mock_request.call_count == 2
            # Each fired batch must contain both unique problem_ids exactly once.
            for call in mock_request.call_args_list:
                payload = call.kwargs["json"]
                problem_ids = [s["problem_id"] for s in payload["submissions"]]
                assert sorted(problem_ids) == ["p1", "p2"]
        finally:
            _stop_patches(patches)

    @pytest.mark.asyncio
    async def test_repeats_open_new_batch_before_first_fills(self):
        """If a repeat of p1 arrives while batch[0] still needs p2, it opens batch[1]; neither fires yet."""
        server = _make_server(_make_config(batch_size=2))

        mock_request, patches = _mock_api({"accuracy": 0.5, "timeout_rate": 0.0})
        try:
            t1 = asyncio.create_task(server.verify(_make_verify_request("```python\na=1\n```", problem_id="p1")))
            t2 = asyncio.create_task(server.verify(_make_verify_request("```python\na2=1\n```", problem_id="p1")))
            # Neither batch is full yet — both verifies should block.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(asyncio.gather(t1, t2)), timeout=0.1)
            mock_request.assert_not_called()

            # /status should report two pending batches each with one submission.
            client = TestClient(server.setup_webserver())
            assert client.get("/status").json() == {"pending_batches": [1, 1], "batch_size": 2}

            for t in (t1, t2):
                t.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await t
        finally:
            _stop_patches(patches)

    @pytest.mark.asyncio
    async def test_verify_times_out_when_batch_never_fills(self):
        """If only some of `batch_size` real submissions arrive (e.g. sibling rollout died),
        waiters time out instead of hanging forever."""
        server = _make_server(_make_config(batch_size=3, verify_timeout_seconds=0.1))

        mock_request, patches = _mock_api({"accuracy": 0.5, "timeout_rate": 0.0})
        try:
            # Only 1 verify arrives; batch needs 3 → never fires.
            with pytest.raises(asyncio.TimeoutError):
                await server.verify(_make_verify_request("```python\nx=1\n```", problem_id="p1"))
            mock_request.assert_not_called()
        finally:
            _stop_patches(patches)

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

    @pytest.mark.asyncio
    async def test_smoke_padding_fires_early_and_pads_to_batch_size(self):
        """fire_after=2 + batch_size=5: fires after 2 real submissions, pads to 5 with empty
        padding submissions drawn from _ALL_PROBLEM_IDS. AA receives 5 (2 real + 3 padded)."""
        # Use the canonical CritPt problem_ids (Challenge_<N>_main) so they collide with the
        # hardcoded _ALL_PROBLEM_IDS list inside app.py.
        server = _make_server(_make_config(batch_size=5, fire_after=2))

        mock_request, patches = _mock_api({"accuracy": 0.0, "timeout_rate": 0.0})
        try:
            results = await asyncio.gather(
                server.verify(_make_verify_request("```python\na=1\n```", problem_id="Challenge_1_main")),
                server.verify(_make_verify_request("```python\nb=2\n```", problem_id="Challenge_2_main")),
            )
            assert mock_request.call_count == 1
            payload = mock_request.call_args.kwargs["json"]
            assert len(payload["submissions"]) == 5
            submitted = {s["problem_id"]: s["generated_code"] for s in payload["submissions"]}
            # The two real submissions are present with real code.
            assert "a=1" in submitted["Challenge_1_main"]
            assert "b=2" in submitted["Challenge_2_main"]
            # Three padded slots are empty padding entries pulled from _ALL_PROBLEM_IDS (in order,
            # skipping the two already-present ones — so Challenge_3, 4, 5).
            for pid in ("Challenge_3_main", "Challenge_4_main", "Challenge_5_main"):
                assert submitted[pid] == "```python\n```"
            # Both real callers get the AA aggregate as their reward.
            for r in results:
                assert r.reward == 0.0
        finally:
            _stop_patches(patches)

    @pytest.mark.asyncio
    async def test_verify_writes_cache_files_on_successful_batch(self, tmp_path):
        """With cache_dir set, a fired batch persists submissions, AA response, and partial metrics."""
        aa_result = {"accuracy": 0.75, "timeout_rate": 0.1, "judge_error_count": 0}
        server = _make_server(_make_config(batch_size=2, cache_dir=tmp_path, unique_cache_per_run=False))

        with patch.object(server, "_call_aa_with_rotation", new_callable=AsyncMock) as mock_aa:
            mock_aa.return_value = aa_result
            results = await asyncio.gather(
                server.verify(_make_verify_request("```python\na=1\n```", problem_id="p1")),
                server.verify(_make_verify_request("```python\nb=2\n```", problem_id="p2")),
            )

        assert mock_aa.await_count == 1
        for r in results:
            assert r.reward == 0.75
            assert r.accuracy == 0.75
            assert r.timeout_rate == 0.1

        submissions = _read_jsonl(tmp_path / "submissions.jsonl")
        assert len(submissions) == 2
        assert [row["submission_id"] for row in submissions] == [0, 1]
        by_pid = {row["submission"]["problem_id"]: row for row in submissions}
        assert set(by_pid) == {"p1", "p2"}
        assert "a=1" in by_pid["p1"]["submission"]["generated_code"]
        assert "b=2" in by_pid["p2"]["submission"]["generated_code"]

        aa_responses = _read_jsonl(tmp_path / "aa_responses.jsonl")
        assert len(aa_responses) == 1
        assert aa_responses[0]["batch_id"] == 0
        assert set(aa_responses[0]["submission_ids"]) == {0, 1}
        assert aa_responses[0]["response"] == aa_result

        partial = json.loads((tmp_path / "partial_metrics.json").read_text())
        assert partial["scored_submissions"] == 2
        assert partial["total_submissions_seen"] == 2
        assert partial["pending_submissions"] == 0
        assert partial["scored_batches"] == 1
        assert partial["mean_accuracy_over_scored"] == 0.75
        assert partial["mean_timeout_rate_over_scored"] == 0.1


class TestKeyRotation:
    """Live-run AA-key rotation on HTTP 429.

    Mirrors the cli replay tool but tested through the server's verify() path:
    a config with multiple keys must transparently rotate on 429, surface only
    the final CritPtRateLimitExceeded once every key has 429'd in one batch,
    and round-robin past the successful key on the next batch.
    """

    @pytest.mark.asyncio
    async def test_single_key_normalized_to_list(self):
        """Pydantic config accepts a bare string and the server stores [key]."""
        server = _make_server(_make_config(batch_size=2, api_key="solo-key"))  # pragma: allowlist secret
        assert server._api_keys == ["solo-key"]
        assert server._key_index == 0

    @pytest.mark.asyncio
    async def test_list_of_keys_preserved_in_order(self):
        """Pydantic config accepts a list and order/identity is preserved."""
        server = _make_server(_make_config(batch_size=2, api_key=["k0", "k1", "k2"]))
        assert server._api_keys == ["k0", "k1", "k2"]
        assert server._key_index == 0

    def test_empty_key_list_rejected(self):
        """An empty list must fail config-construction time, not at first 429."""
        with pytest.raises(ValueError):
            _make_config(batch_size=2, api_key=[])

    def test_empty_string_key_rejected(self):
        """A blank single key must fail config-construction time."""
        with pytest.raises(ValueError):
            _make_config(batch_size=2, api_key="")

    @pytest.mark.asyncio
    async def test_rotates_to_next_key_on_429(self):
        """Two keys, first 429s, server retries with key #2 and succeeds.

        Cursor sticks to the key that worked (k1), so the next batch starts
        on k1 directly — k0 is presumed still rate-limited.
        """
        server = _make_server(_make_config(batch_size=3, api_key=["k0", "k1"]))

        mock_request, patches = _mock_api_sequence(
            [
                {"status": 429, "headers": {"Retry-After": "60", "X-Ratelimit-Reset": "111"}, "body": "rl"},
                {"status": 200, "json": {"accuracy": 0.5, "timeout_rate": 0.0}},
            ]
        )
        try:
            results = await asyncio.gather(
                server.verify(_make_verify_request("```python\na=1\n```", problem_id="p1")),
                server.verify(_make_verify_request("```python\nb=2\n```", problem_id="p2")),
                server.verify(_make_verify_request("```python\nc=3\n```", problem_id="p3")),
            )
            assert mock_request.call_count == 2
            sent_keys = [call.kwargs["headers"]["x-api-key"] for call in mock_request.call_args_list]
            assert sent_keys == ["k0", "k1"]
            for r in results:
                assert r.reward == 0.5
            assert server._key_index == 1
        finally:
            _stop_patches(patches)

    @pytest.mark.asyncio
    async def test_all_keys_429_raises_rate_limited(self):
        """Three keys all 429: server raises CritPtRateLimitExceeded once, no extra attempts."""
        server = _make_server(_make_config(batch_size=2, api_key=["k0", "k1", "k2"]))

        mock_request, patches = _mock_api_sequence(
            [
                {"status": 429, "headers": {"Retry-After": "10"}, "body": "rl"},
                {"status": 429, "headers": {"Retry-After": "20"}, "body": "rl"},
                {"status": 429, "headers": {"Retry-After": "30", "X-Ratelimit-Reset": "999"}, "body": "rl"},
            ]
        )
        try:
            with pytest.raises(CritPtRateLimitExceeded) as exc_info:
                await asyncio.gather(
                    server.verify(_make_verify_request("```python\na=1\n```", problem_id="p1")),
                    server.verify(_make_verify_request("```python\nb=2\n```", problem_id="p2")),
                )
            # The surfaced exception carries the last attempt's signals.
            assert exc_info.value.retry_after_seconds == 30
            assert exc_info.value.reset_unix == 999
            assert mock_request.call_count == 3
        finally:
            _stop_patches(patches)

    @pytest.mark.asyncio
    async def test_single_key_429_no_retry(self):
        """One configured key, 429: raises immediately (one attempt, no rotation)."""
        server = _make_server(_make_config(batch_size=2, api_key=["solo-key"]))  # pragma: allowlist secret

        mock_request, patches = _mock_api_sequence([{"status": 429, "headers": {"Retry-After": "60"}, "body": "rl"}])
        try:
            with pytest.raises(CritPtRateLimitExceeded):
                await asyncio.gather(
                    server.verify(_make_verify_request("```python\na=1\n```", problem_id="p1")),
                    server.verify(_make_verify_request("```python\nb=2\n```", problem_id="p2")),
                )
            assert mock_request.call_count == 1
        finally:
            _stop_patches(patches)

    @pytest.mark.asyncio
    async def test_cursor_sticks_to_working_key(self):
        """Two successful batches in a row both use k0; the cursor never
        moves off a working key absent a 429."""
        server = _make_server(_make_config(batch_size=2, api_key=["k0", "k1"]))

        mock_request, patches = _mock_api_sequence(
            [
                {"status": 200, "json": {"accuracy": 0.1, "timeout_rate": 0.0}},
                {"status": 200, "json": {"accuracy": 0.2, "timeout_rate": 0.0}},
            ]
        )
        try:
            await asyncio.gather(
                server.verify(_make_verify_request("```python\na=1\n```", problem_id="p1")),
                server.verify(_make_verify_request("```python\nb=2\n```", problem_id="p2")),
            )
            await asyncio.gather(
                server.verify(_make_verify_request("```python\na2=1\n```", problem_id="p1")),
                server.verify(_make_verify_request("```python\nb2=2\n```", problem_id="p2")),
            )
            sent_keys = [call.kwargs["headers"]["x-api-key"] for call in mock_request.call_args_list]
            assert sent_keys == ["k0", "k0"]
            assert server._key_index == 0
        finally:
            _stop_patches(patches)
