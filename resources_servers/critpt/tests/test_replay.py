# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the CritPt replay tool's env-parsing + rotation helpers.

The integration of replay vs. a real cache_dir is covered by the live
server's test suite (see test_app.py::TestKeyRotation); this file pins the
pure parsing/rotation logic so a regression in the env-shape contract is
caught fast without spinning up an httpx mock.
"""

from unittest.mock import AsyncMock, patch

import pytest

from resources_servers.critpt.app import _ALL_PROBLEM_IDS, CritPtRateLimitExceeded
from resources_servers.critpt.replay import (
    _call_api_with_rotation,
    _load_api_keys,
    _pack_into_batches,
    _pad_to_batch_size,
    _parse_api_keys_env,
)


class TestParseApiKeysEnv:
    def test_single_key(self):
        assert _parse_api_keys_env("aa-xyz") == ["aa-xyz"]

    def test_single_key_with_whitespace(self):
        assert _parse_api_keys_env("  aa-xyz  ") == ["aa-xyz"]

    def test_bracketed_list(self):
        assert _parse_api_keys_env("[k1,k2,k3]") == ["k1", "k2", "k3"]

    def test_bracketed_list_with_spaces(self):
        assert _parse_api_keys_env("[ k1 , k2 ,  k3 ]") == ["k1", "k2", "k3"]

    def test_bracketed_list_with_quoted_items(self):
        """Tolerate the common shape of an exported `.env` line whose value
        itself contains quoted comma-separated strings.
        """
        assert _parse_api_keys_env('["k1","k2",\'k3\']') == ["k1", "k2", "k3"]

    def test_bracketed_list_dedupes_preserving_order(self):
        assert _parse_api_keys_env("[k1,k2,k1,k3,k2]") == ["k1", "k2", "k3"]

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError):
            _parse_api_keys_env("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValueError):
            _parse_api_keys_env("   ")

    def test_empty_list_rejected(self):
        with pytest.raises(ValueError):
            _parse_api_keys_env("[]")

    def test_list_of_empties_rejected(self):
        with pytest.raises(ValueError):
            _parse_api_keys_env("[,, ,]")


class TestPadToBatchSize:
    def _real(self, pid: str) -> dict:
        return {"problem_id": pid, "generated_code": "```python\nx=1\n```", "model": "m"}

    def test_pads_short_batch_up_to_batch_size(self):
        payload = [self._real("Challenge_1_main"), self._real("Challenge_2_main")]
        padded = _pad_to_batch_size(payload, batch_size=70)
        assert len(padded) == 70
        # Real submissions are preserved unchanged at the front.
        assert padded[:2] == payload

    def test_padding_uses_missing_problem_ids_with_empty_code(self):
        payload = [self._real("Challenge_1_main")]
        padded = _pad_to_batch_size(payload, batch_size=70)
        padding_entries = padded[1:]
        assert all(d["generated_code"] == "```python\n```" for d in padding_entries)
        # No padding entry reuses a real problem_id, and together they cover the canonical set.
        assert "Challenge_1_main" not in {d["problem_id"] for d in padding_entries}
        assert {p["problem_id"] for p in padded} == set(_ALL_PROBLEM_IDS)

    def test_already_full_batch_unchanged(self):
        payload = [self._real(pid) for pid in _ALL_PROBLEM_IDS]
        padded = _pad_to_batch_size(payload, batch_size=70)
        assert padded == payload


class TestPackIntoBatches:
    def _sub(self, pid: str) -> dict:
        return {"submission_id": pid, "submission": {"problem_id": pid}}

    def test_unique_problem_ids_share_one_batch(self):
        subs = [self._sub("p1"), self._sub("p2"), self._sub("p3")]
        batches = _pack_into_batches(subs, batch_size=70)
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_duplicate_problem_id_opens_new_batch(self):
        subs = [self._sub("p1"), self._sub("p1"), self._sub("p2")]
        batches = _pack_into_batches(subs, batch_size=70)
        # The second p1 cannot join the first batch (already has p1), so a new one opens.
        assert len(batches) == 2
        assert len(batches[0]) == 2  # p1, p2
        assert len(batches[1]) == 1  # duplicate p1


class TestLoadApiKeys:
    def test_unset_returns_empty(self, monkeypatch):
        monkeypatch.delenv("ARTIFICIAL_ANALYSIS_API_KEY", raising=False)
        assert _load_api_keys() == []

    def test_single_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ARTIFICIAL_ANALYSIS_API_KEY", "aa-from-env")
        assert _load_api_keys() == ["aa-from-env"]

    def test_bracketed_list_from_env(self, monkeypatch):
        monkeypatch.setenv("ARTIFICIAL_ANALYSIS_API_KEY", "[k1,k2,k3]")
        assert _load_api_keys() == ["k1", "k2", "k3"]


class TestCallApiWithRotation:
    @pytest.mark.asyncio
    async def test_first_key_succeeds(self):
        """Sticky: cursor stays on the working key (no +1 on success)."""
        with patch("resources_servers.critpt.replay._call_api", new_callable=AsyncMock) as call_api:
            call_api.return_value = {"accuracy": 0.7, "timeout_rate": 0.0}

            response, next_idx = await _call_api_with_rotation(
                api_keys=["k0", "k1"],
                api_url="https://example/api",
                submissions=[{"problem_id": "p1"}],
                max_retries=1,
                backoff_seconds=0.0,
                key_index_in=0,
            )

            assert response["accuracy"] == 0.7
            assert next_idx == 0
            assert call_api.await_count == 1
            assert call_api.await_args.kwargs["api_key"] == "k0"

    @pytest.mark.asyncio
    async def test_rotates_past_first_429(self):
        """On 429: advance, retry, and stick to the key that worked."""
        rate_limit = CritPtRateLimitExceeded(retry_after_seconds=60, reset_unix=1, body="rl")
        ok = {"accuracy": 0.9, "timeout_rate": 0.0}
        with patch("resources_servers.critpt.replay._call_api", new_callable=AsyncMock) as call_api:
            call_api.side_effect = [rate_limit, ok]

            response, next_idx = await _call_api_with_rotation(
                api_keys=["k0", "k1"],
                api_url="https://example/api",
                submissions=[{"problem_id": "p1"}],
                max_retries=1,
                backoff_seconds=0.0,
                key_index_in=0,
            )

            assert response == ok
            assert next_idx == 1
            assert call_api.await_count == 2
            sent_keys = [c.kwargs["api_key"] for c in call_api.await_args_list]
            assert sent_keys == ["k0", "k1"]

    @pytest.mark.asyncio
    async def test_all_keys_429_raises(self):
        last = CritPtRateLimitExceeded(retry_after_seconds=99, reset_unix=2, body="rl")
        with patch("resources_servers.critpt.replay._call_api", new_callable=AsyncMock) as call_api:
            call_api.side_effect = [
                CritPtRateLimitExceeded(retry_after_seconds=10, reset_unix=1, body="rl"),
                CritPtRateLimitExceeded(retry_after_seconds=20, reset_unix=2, body="rl"),
                last,
            ]

            with pytest.raises(CritPtRateLimitExceeded) as exc_info:
                await _call_api_with_rotation(
                    api_keys=["k0", "k1", "k2"],
                    api_url="https://example/api",
                    submissions=[{"problem_id": "p1"}],
                    max_retries=1,
                    backoff_seconds=0.0,
                    key_index_in=0,
                )
            assert exc_info.value.retry_after_seconds == 99
            assert call_api.await_count == 3

    @pytest.mark.asyncio
    async def test_starts_from_provided_cursor(self):
        """Sticky: starting on k2 and succeeding keeps the cursor on k2."""
        with patch("resources_servers.critpt.replay._call_api", new_callable=AsyncMock) as call_api:
            call_api.return_value = {"accuracy": 0.4, "timeout_rate": 0.0}

            response, next_idx = await _call_api_with_rotation(
                api_keys=["k0", "k1", "k2"],
                api_url="https://example/api",
                submissions=[{"problem_id": "p1"}],
                max_retries=1,
                backoff_seconds=0.0,
                key_index_in=2,
            )

            assert next_idx == 2
            assert call_api.await_args.kwargs["api_key"] == "k2"
            assert response["accuracy"] == 0.4
