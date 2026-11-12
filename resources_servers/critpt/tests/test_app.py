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
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NgFailureEvent,
)
from nemo_gym.core.failure_kinds import PROVIDER_CREDENTIALS_REJECTED, PROVIDER_QUOTA_EXHAUSTED
from nemo_gym.verification import is_judge_failure, judge_failsafe
from resources_servers.critpt.app import (
    _REPO_ROOT,
    SMOKE_NOT_GRADED,
    CritPtAccessDenied,
    CritPtRateLimitExceeded,
    CritPtResourcesServer,
    CritPtResourcesServerConfig,
    CritPtVerifyRequest,
    _extract_code,
    _key_fingerprint,
    _resolve_cache_dir,
    graded_accuracy,
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


class _Cause:
    value: BaseException


@contextmanager
def grader_failed(cause_type):
    """Expect verify() to raise a typed judge_failed event caused by ``cause_type``; ``.value`` is that cause."""
    holder = _Cause()
    with pytest.raises(NgFailureEvent) as info:
        yield holder
    assert is_judge_failure(info.value)
    assert isinstance(info.value.__cause__, cause_type)
    holder.value = info.value.__cause__


async def verify_pair(server) -> list:
    """Verify two rollouts that fill a batch of two, through judge_failsafe; the rows are what the collector sees."""
    fence = "```"
    bodies = [
        _make_verify_request(fence + "python" + chr(10) + "a=1" + chr(10) + fence, problem_id="p1"),
        _make_verify_request(fence + "python" + chr(10) + "b=2" + chr(10) + fence, problem_id="p2"),
    ]
    responses = await asyncio.gather(*(judge_failsafe(server.verify)(body=b) for b in bodies))
    return [json.loads(r.body) for r in responses]


def assert_terminal_stop(rows: list, kind: str) -> str:
    """Every row is a judge-class failure row of ``kind``: masked, no score, saved answer kept. Returns the reason."""
    assert len(rows) == 2
    for row in rows:
        assert row["mask_sample"] is True and row["reward"] == 0.0
        assert row["failure_kind"] == kind and row["_ng_failure_class"] == "judge_failed"
        assert "accuracy" not in row
        assert "a=1" in json.dumps(row["response"]) or "b=2" in json.dumps(row["response"])
    assert rows[0]["failure_reason"] == rows[1]["failure_reason"]
    return rows[0]["failure_reason"]


NOW = 1_000_000.0


def reset_text(retry_after_seconds: int) -> str:
    """The UTC reset time a quota reason names for a Retry-After of this many seconds, at NOW."""
    return datetime.fromtimestamp(NOW + retry_after_seconds, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


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
            with grader_failed(asyncio.TimeoutError):
                await server.verify(_make_verify_request("```python\nx=1\n```", problem_id="p1"))
            mock_request.assert_not_called()
        finally:
            _stop_patches(patches)

    @pytest.mark.asyncio
    async def test_a_partial_batch_times_out_typed_with_the_pending_count_and_sends_nothing(self):
        """A re-judge of fewer than batch_size saved answers cannot be graded alone: every waiter of the short batch
        fails as a masked, re-judgeable judge_failed naming the pending count, with no AA call and no padding."""
        server = _make_server(_make_config(batch_size=5, verify_timeout_seconds=0.1))
        mock_request, patches = _mock_api({"accuracy": 0.5, "timeout_rate": 0.0})
        try:
            fence = "```"
            bodies = [
                _make_verify_request(fence + "python" + chr(10) + f"x={i}" + chr(10) + fence, problem_id=f"p{i}")
                for i in range(3)
            ]
            rows = [
                json.loads(r.body)
                for r in await asyncio.gather(*(judge_failsafe(server.verify)(body=b) for b in bodies))
            ]
            mock_request.assert_not_called()
        finally:
            _stop_patches(patches)
        for row in rows:
            assert row["mask_sample"] is True and row["failure_kind"] == "judge_failed"
            assert row["failure_reason"].startswith(
                "3 saved answers pending, fewer than batch_size (5); grade them with replay"
            )
            assert row["response"]
        assert server._batches == []  # the stale short batch does not absorb later rows

    @pytest.mark.asyncio
    async def test_a_full_batch_still_fires_under_a_short_verify_timeout(self):
        server = _make_server(_make_config(batch_size=2, verify_timeout_seconds=5.0))
        mock_request, patches = _mock_api({"accuracy": 0.5, "timeout_rate": 0.0, "judge_error_count": 0})
        try:
            results = await asyncio.gather(
                server.verify(_make_verify_request("```python\na=1\n```", problem_id="p1")),
                server.verify(_make_verify_request("```python\nb=2\n```", problem_id="p2")),
            )
            assert mock_request.call_count == 1
        finally:
            _stop_patches(patches)
        assert {r.reward for r in results} == {0.5} and not any(r.mask_sample for r in results)

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
    async def test_smoke_mode_never_calls_aa_and_masks_every_row(self, tmp_path, caplog):
        """fire_after=2 + batch_size=5: the batch fires after 2 real submissions without any AA call. Every row is
        masked as not graded with a reason naming smoke mode, and the submissions are still cached."""
        server = _make_server(_make_config(batch_size=5, fire_after=2, cache_dir=tmp_path, unique_cache_per_run=False))

        mock_request, patches = _mock_api({"accuracy": 1.0, "timeout_rate": 0.0})
        try:
            with patch.object(server, "_call_aa_with_rotation", new_callable=AsyncMock) as mock_aa:
                with caplog.at_level("WARNING"):
                    results = await asyncio.gather(
                        server.verify(_make_verify_request("```python\na=1\n```", problem_id="Challenge_1_main")),
                        server.verify(_make_verify_request("```python\nb=2\n```", problem_id="Challenge_2_main")),
                    )
            assert mock_aa.await_count == 0
            assert mock_request.call_count == 0
        finally:
            _stop_patches(patches)
        assert "smoke mode: batch of 2 submissions not sent to AA" in caplog.text
        for r in results:
            assert r.mask_sample is True
            assert r.reward == 0.0
            assert r.accuracy is None and r.timeout_rate is None
            assert r.failure_kind == SMOKE_NOT_GRADED
            assert "smoke mode" in r.failure_reason and "not sent to the grading API" in r.failure_reason
        # Nothing was written as an accuracy, and the answers are cached for a later replay.
        assert len(_read_jsonl(tmp_path / "submissions.jsonl")) == 2
        assert not (tmp_path / "aa_responses.jsonl").exists()

    def test_smoke_rows_are_excluded_from_metrics(self):
        server = _make_server(_make_config())
        masked = {"problem_id": "p1", "accuracy": None, "mask_sample": True}
        assert server._critpt_score_fn(masked) == {}
        assert server._critpt_score_fn({"problem_id": "p1", "accuracy": 0.5}) == {"accuracy": 0.5}

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
            with patch("resources_servers.critpt.app.time.time", return_value=NOW):
                results = await verify_pair(server)
            reason = assert_terminal_stop(results, PROVIDER_QUOTA_EXHAUSTED)
            # The reset time comes from the last attempt's Retry-After (30s), not the reset header.
            assert reset_text(30) in reason
            assert mock_request.call_count == 3
        finally:
            _stop_patches(patches)

    @pytest.mark.asyncio
    async def test_single_key_429_no_retry(self):
        """One configured key, 429: raises immediately (one attempt, no rotation)."""
        server = _make_server(_make_config(batch_size=2, api_key=["solo-key"]))  # pragma: allowlist secret

        mock_request, patches = _mock_api_sequence([{"status": 429, "headers": {"Retry-After": "60"}, "body": "rl"}])
        try:
            with patch("resources_servers.critpt.app.time.time", return_value=NOW):
                results = await verify_pair(server)
            assert_terminal_stop(results, PROVIDER_QUOTA_EXHAUSTED)
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


class TestAccessDenied:
    """HTTP 403 from AA: the key authenticated but its account is not approved
    for CritPt grading.

    Distinct from 429 (approved, out of quota): AA rejects at the auth layer, so
    nothing is graded and no quota is spent. Before this was typed, a pool of
    unapproved keys turned into a bare RuntimeError, a 500 on /verify, and an
    aborted run.
    """

    _DENIED_BODY = (
        '{"error":"CritPt grading requires approved access. To protect benchmark integrity '
        "and ensure server capacity, the grading endpoint is limited to approved labs and "
        'researchers; access remains free."}'
    )

    @pytest.mark.asyncio
    async def test_403_raises_typed_access_denied_not_bare_runtime_error(self):
        """A 403 must surface as CritPtAccessDenied carrying AA's body."""
        server = _make_server(_make_config(batch_size=2, api_key=["k0"]))

        mock_request, patches = _mock_api_sequence([{"status": 403, "body": self._DENIED_BODY}])
        try:
            with patch("resources_servers.critpt.app.time.time", return_value=NOW):
                results = await verify_pair(server)
            reason = assert_terminal_stop(results, PROVIDER_CREDENTIALS_REJECTED)
            # The reason tells the operator what to actually do.
            assert "HTTP 403" in reason and "use an approved key" in reason
            assert mock_request.call_count == 1
        finally:
            _stop_patches(patches)

    @pytest.mark.asyncio
    async def test_403_is_not_confused_with_quota(self):
        """A 403 must NOT be reported as a rate limit. The two have different fixes:
        429 means wait, 403 means get the account approved."""
        server = _make_server(_make_config(batch_size=2, api_key=["k0"]))

        mock_request, patches = _mock_api_sequence([{"status": 403, "body": self._DENIED_BODY}])
        try:
            with patch("resources_servers.critpt.app.time.time", return_value=NOW):
                results = await verify_pair(server)
            assert_terminal_stop(results, PROVIDER_CREDENTIALS_REJECTED)
            assert mock_request.call_count == 1
        finally:
            _stop_patches(patches)

    @pytest.mark.asyncio
    async def test_rotates_past_denied_key_to_approved_key(self):
        """Mixed pool: k0's account is unapproved, k1's is approved.

        Rotation is what makes a mixed pool recover. A 403 costs no quota, so
        trying the next key is free.
        """
        server = _make_server(_make_config(batch_size=2, api_key=["k0", "k1"]))

        mock_request, patches = _mock_api_sequence(
            [
                {"status": 403, "body": self._DENIED_BODY},
                {"status": 200, "json": {"accuracy": 0.5, "timeout_rate": 0.0}},
            ]
        )
        try:
            results = await asyncio.gather(
                server.verify(_make_verify_request("```python\na=1\n```", problem_id="p1")),
                server.verify(_make_verify_request("```python\nb=2\n```", problem_id="p2")),
            )
            sent_keys = [call.kwargs["headers"]["x-api-key"] for call in mock_request.call_args_list]
            assert sent_keys == ["k0", "k1"]
            for r in results:
                assert r.reward == 0.5
            # Cursor sticks to the approved key so later batches skip the dead one.
            assert server._key_index == 1
        finally:
            _stop_patches(patches)

    @pytest.mark.asyncio
    async def test_all_keys_403_raises_after_full_cycle(self):
        """Every key denied: one attempt per key, then CritPtAccessDenied.

        Without rotation on 403 the server gave up after the first key.
        """
        server = _make_server(_make_config(batch_size=2, api_key=["k0", "k1", "k2"]))

        mock_request, patches = _mock_api_sequence(
            [
                {"status": 403, "body": self._DENIED_BODY},
                {"status": 403, "body": self._DENIED_BODY},
                {"status": 403, "body": self._DENIED_BODY},
            ]
        )
        try:
            with patch("resources_servers.critpt.app.time.time", return_value=NOW):
                results = await verify_pair(server)
            assert_terminal_stop(results, PROVIDER_CREDENTIALS_REJECTED)
            assert mock_request.call_count == 3
        finally:
            _stop_patches(patches)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("denied_first", [False, True], ids=["429-then-403", "403-then-429"])
    async def test_mixed_rate_limited_and_denied_pool_reports_the_rate_limit(self, denied_first: bool):
        """An approved key out of quota and an unapproved key: the error is the rate limit, with the approved key's
        reset, whichever key rotation tried last. Waiting helps; approval is not the only fix."""
        server = _make_server(_make_config(batch_size=2, api_key=["k0", "k1"]))
        limited = {"status": 429, "headers": {"Retry-After": "60", "X-Ratelimit-Reset": "111"}, "body": "rl"}
        denied = {"status": 403, "body": self._DENIED_BODY}

        mock_request, patches = _mock_api_sequence([denied, limited] if denied_first else [limited, denied])
        try:
            with patch("resources_servers.critpt.app.time.time", return_value=NOW):
                results = await verify_pair(server)
            reason = assert_terminal_stop(results, PROVIDER_QUOTA_EXHAUSTED)
            assert reset_text(60) in reason
            assert "1 of 2 key(s) were denied" in reason
            assert mock_request.call_count == 2
        finally:
            _stop_patches(patches)

    @pytest.mark.asyncio
    async def test_non_403_non_429_client_error_still_bare_runtime_error(self):
        """A 400 keeps the old fail-fast behaviour; only 403 gained a type."""
        server = _make_server(_make_config(batch_size=2, api_key=["k0"]))

        mock_request, patches = _mock_api_sequence([{"status": 400, "body": "bad payload"}])
        try:
            with grader_failed(RuntimeError) as exc_info:
                await asyncio.gather(
                    server.verify(_make_verify_request("```python\na=1\n```", problem_id="p1")),
                    server.verify(_make_verify_request("```python\nb=2\n```", problem_id="p2")),
                )
            assert not isinstance(exc_info.value, CritPtAccessDenied)
            assert not isinstance(exc_info.value, CritPtRateLimitExceeded)
            assert mock_request.call_count == 1
        finally:
            _stop_patches(patches)


class TestKeyFingerprint:
    """Fingerprints let us tell from a log which AA account a run used.

    $ARTIFICIAL_ANALYSIS_API_KEY usually comes from the launching user's
    environment, so it is absent from the run config.
    """

    def test_is_stable_and_short(self):
        assert _key_fingerprint("k0") == _key_fingerprint("k0")
        assert len(_key_fingerprint("k0")) == 8

    def test_distinguishes_keys(self):
        assert _key_fingerprint("k0") != _key_fingerprint("k1")

    def test_never_reveals_the_key(self):
        secret = "aa_averysecretkeyvalue0123456789"  # pragma: allowlist secret
        fp = _key_fingerprint(secret)
        assert secret not in fp
        assert fp not in secret


class TestTerminalAaErrors:
    """A quota or key error ends the batch as masked, unverified rows that keep their saved answer."""

    @pytest.mark.asyncio
    async def test_quota_reason_names_the_reset_time_from_retry_after(self):
        server = _make_server(_make_config(batch_size=2, api_key=["k0"]))
        mock_request, patches = _mock_api_sequence(
            [{"status": 429, "headers": {"Retry-After": "3600", "X-Ratelimit-Reset": "1"}, "body": "rl"}]
        )
        try:
            with patch("resources_servers.critpt.app.time.time", return_value=NOW):
                results = await verify_pair(server)
        finally:
            _stop_patches(patches)
        reason = assert_terminal_stop(results, PROVIDER_QUOTA_EXHAUSTED)
        assert (
            reason
            == f"AA quota exhausted, resets at {reset_text(3600)}; grade with replay or resubmit after the reset"
        )

    @pytest.mark.asyncio
    async def test_quota_reason_falls_back_to_the_reset_header(self):
        server = _make_server(_make_config(batch_size=2, api_key=["k0"]))
        mock_request, patches = _mock_api_sequence(
            [{"status": 429, "headers": {"X-Ratelimit-Reset": "86400"}, "body": "rl"}]
        )
        try:
            results = await verify_pair(server)
        finally:
            _stop_patches(patches)
        assert "resets at 1970-01-02 00:00 UTC" in assert_terminal_stop(results, PROVIDER_QUOTA_EXHAUSTED)

    @pytest.mark.asyncio
    async def test_401_is_a_rejected_key_not_a_judge_failure(self):
        server = _make_server(_make_config(batch_size=2, api_key=["k0"]))
        mock_request, patches = _mock_api_sequence([{"status": 401, "body": "bad key"}])
        try:
            results = await verify_pair(server)
            assert mock_request.call_count == 1
        finally:
            _stop_patches(patches)
        reason = assert_terminal_stop(results, PROVIDER_CREDENTIALS_REJECTED)
        assert "HTTP 401" in reason

    @pytest.mark.asyncio
    async def test_terminal_batch_keeps_its_submissions_in_the_cache(self, tmp_path):
        server = _make_server(
            _make_config(batch_size=2, api_key=["k0"], cache_dir=tmp_path, unique_cache_per_run=False)
        )
        mock_request, patches = _mock_api_sequence([{"status": 429, "headers": {"Retry-After": "60"}, "body": "rl"}])
        try:
            results = await verify_pair(server)
        finally:
            _stop_patches(patches)
        assert_terminal_stop(results, PROVIDER_QUOTA_EXHAUSTED)
        assert len(_read_jsonl(tmp_path / "submissions.jsonl")) == 2
        assert not (tmp_path / "aa_responses.jsonl").exists()

    @pytest.mark.asyncio
    async def test_server_error_stays_a_masked_judge_failure(self):
        server = _make_server(_make_config(batch_size=2, api_key=["k0"], api_max_retries=1))
        mock_request, patches = _mock_api_sequence([{"status": 500, "body": "boom"}])
        try:
            rows = await verify_pair(server)
        finally:
            _stop_patches(patches)
        assert {row["failure_kind"] for row in rows} == {"judge_failed"}


class TestPartialGraderErrors:
    """Up to max_judge_errors grader errors are tolerated and the accuracy rescaled to the graded items."""

    async def _verify_batch(self, server, aa_response: dict) -> list:
        mock_request, patches = _mock_api(aa_response)
        try:
            fence = "```"
            bodies = [
                _make_verify_request(fence + "python" + chr(10) + f"x={i}" + chr(10) + fence, problem_id=f"p{i}")
                for i in range(server.config.batch_size)
            ]
            return list(await asyncio.gather(*(judge_failsafe(server.verify)(body=b) for b in bodies)))
        finally:
            _stop_patches(patches)

    def test_default_threshold_is_three(self):
        assert _make_config().max_judge_errors == 3

    @pytest.mark.asyncio
    async def test_errors_at_the_threshold_are_accepted_and_rescaled(self):
        server = _make_server(_make_config(batch_size=10, max_judge_errors=3))
        results = await self._verify_batch(server, {"accuracy": 0.5, "timeout_rate": 0.1, "judge_error_count": 2})
        for r in results:
            assert not isinstance(r, JSONResponse), "an accepted batch is a scored response"
            assert r.mask_sample is False and r.failure_kind is None
            # 0.5 of the whole batch is 5 correct items; over the 8 graded items that is 0.625.
            assert r.accuracy == pytest.approx(0.625) and r.reward == pytest.approx(0.625)
            assert r.judge_error_count == 2 and r.graded_items == 8
            assert r.timeout_rate == 0.1

    @pytest.mark.asyncio
    async def test_rescaled_accuracy_is_clamped_to_one(self):
        server = _make_server(_make_config(batch_size=10, max_judge_errors=3))
        results = await self._verify_batch(server, {"accuracy": 0.9, "timeout_rate": 0.0, "judge_error_count": 3})
        assert {r.accuracy for r in results} == {1.0}
        assert {r.graded_items for r in results} == {7}

    @pytest.mark.asyncio
    async def test_errors_above_the_threshold_fail_the_batch_as_judge_failed(self):
        server = _make_server(_make_config(batch_size=10, max_judge_errors=3))
        results = await self._verify_batch(server, {"accuracy": 0.5, "timeout_rate": 0.0, "judge_error_count": 4})
        for resp in results:
            data = json.loads(resp.body)
            assert data["mask_sample"] is True and data["failure_kind"] == "judge_failed"
            assert "judge_error_count=4" in data["failure_reason"] and "max_judge_errors=3" in data["failure_reason"]

    @pytest.mark.asyncio
    async def test_threshold_is_configurable(self):
        server = _make_server(_make_config(batch_size=10, max_judge_errors=0))
        results = await self._verify_batch(server, {"accuracy": 0.5, "timeout_rate": 0.0, "judge_error_count": 1})
        assert all(json.loads(r.body)["failure_kind"] == "judge_failed" for r in results)

    @pytest.mark.asyncio
    async def test_accepted_batch_counts_as_scored_in_the_cache(self, tmp_path):
        server = _make_server(_make_config(batch_size=10, cache_dir=tmp_path, unique_cache_per_run=False))
        await self._verify_batch(server, {"accuracy": 0.5, "timeout_rate": 0.0, "judge_error_count": 2})
        partial = json.loads((tmp_path / "partial_metrics.json").read_text())
        assert partial["scored_submissions"] == 10 and partial["pending_submissions"] == 0
        assert partial["mean_accuracy_over_scored"] == pytest.approx(0.625)

    def test_graded_accuracy_edge_cases(self):
        assert graded_accuracy({"accuracy": 0.3}, 70, 3) == 0.3
        assert graded_accuracy({"accuracy": 0.3, "judge_error_count": 70}, 70, 100) is None


class TestGraderFailureTyping:
    @pytest.mark.asyncio
    async def test_judge_error_count_is_judge_failed_not_a_score(self):
        """AA reporting judge_error_count > 0 makes the batch accuracy unusable: masked, never a scored reward."""
        server = _make_server(_make_config(batch_size=2, max_judge_errors=0))
        mock_request, patches = _mock_api({"accuracy": 0.5, "timeout_rate": 0.0, "judge_error_count": 1})
        try:
            requests = [
                _make_verify_request("```python\na=1\n```", problem_id="p1"),
                _make_verify_request("```python\nb=2\n```", problem_id="p2"),
            ]
            results = await asyncio.gather(*(judge_failsafe(server.verify)(body=r) for r in requests))
        finally:
            _stop_patches(patches)
        for resp in results:
            data = json.loads(resp.body)
            assert data["mask_sample"] is True
            assert data["failure_kind"] == "judge_failed"
            assert "judge_error_count=1" in data["failure_reason"]
            assert data["problem_id"] in {"p1", "p2"}

    @pytest.mark.asyncio
    async def test_rejected_grader_error_batch_is_not_counted_as_scored_in_the_cache(self, tmp_path):
        server = _make_server(
            _make_config(batch_size=2, max_judge_errors=0, cache_dir=tmp_path, unique_cache_per_run=False)
        )
        mock_request, patches = _mock_api({"accuracy": 0.5, "timeout_rate": 0.0, "judge_error_count": 1})
        try:
            requests = [
                _make_verify_request("```python\na=1\n```", problem_id="p1"),
                _make_verify_request("```python\nb=2\n```", problem_id="p2"),
            ]
            results = await asyncio.gather(*(judge_failsafe(server.verify)(body=r) for r in requests))
        finally:
            _stop_patches(patches)
        assert all(json.loads(r.body)["mask_sample"] is True for r in results)
        partial = json.loads((tmp_path / "partial_metrics.json").read_text())
        assert partial["scored_submissions"] == 0
        assert partial["pending_submissions"] == 2
        assert partial["mean_accuracy_over_scored"] is None

    @pytest.mark.asyncio
    async def test_aa_http_failure_is_masked_by_judge_failsafe(self):
        server = _make_server(_make_config(batch_size=2, api_key=["k0"]))
        mock_request, patches = _mock_api_sequence([{"status": 400, "body": "bad payload"}])
        try:
            requests = [
                _make_verify_request("```python\na=1\n```", problem_id="p1"),
                _make_verify_request("```python\nb=2\n```", problem_id="p2"),
            ]
            results = await asyncio.gather(*(judge_failsafe(server.verify)(body=r) for r in requests))
        finally:
            _stop_patches(patches)
        for resp in results:
            data = json.loads(resp.body)
            assert data["mask_sample"] is True and data["failure_kind"] == "judge_failed"
            assert data["reward"] == 0.0

    @pytest.mark.asyncio
    async def test_clean_batch_and_model_timeouts_are_still_scored(self):
        """judge_error_count == 0 keeps the AA accuracy as reward, including a high timeout_rate."""
        server = _make_server(_make_config(batch_size=2))
        mock_request, patches = _mock_api({"accuracy": 0.25, "timeout_rate": 0.5, "judge_error_count": 0})
        try:
            results = await asyncio.gather(
                server.verify(_make_verify_request("```python\na=1\n```", problem_id="p1")),
                server.verify(_make_verify_request("```python\nb=2\n```", problem_id="p2")),
            )
        finally:
            _stop_patches(patches)
        for r in results:
            assert r.reward == 0.25 and r.timeout_rate == 0.5


@pytest.mark.asyncio
async def test_one_saved_answer_alone_cannot_fill_a_batch_so_replay_is_not_declared():
    assert CritPtResourcesServerConfig.REVERIFY_MODE.value == "unknown"
    server = _make_server(_make_config(batch_size=2, verify_timeout_seconds=0.05))

    with pytest.raises(Exception) as stopped:
        await server.verify(_make_verify_request("```python\nx=1\n```", problem_id="p1"))

    assert "fewer than batch_size" in str(stopped.value)
