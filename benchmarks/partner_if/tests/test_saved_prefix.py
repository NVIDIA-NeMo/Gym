# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A recovery uses exact saved bytes and refuses drift before any provider call."""

import json

import pytest

from responses_api_agents.ngc_recorded_context.saved_prefix import SavedPolicyPrefix, prefix_digest


def saved(tmp_path):
    values = {
        "identity.json": {"model": "test"},
        "call-01.request.json": {"model": "test", "messages": [{"role": "user", "content": "original"}]},
        "call-01.response.json": {"id": "saved", "model": "test"},
        "call-01.receipt.json": {"http_status": 200},
    }
    for name, value in values.items():
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / "api-order.jsonl").write_text(json.dumps({"index": 1, "request": "call-01.request.json"}) + "\n")
    return values


def test_prefix_returns_exact_response_once_then_allows_new_calls(tmp_path):
    values = saved(tmp_path)
    prefix = SavedPolicyPrefix(tmp_path, prefix_digest(tmp_path), "test")
    assert prefix.take(values["call-01.request.json"]) == (200, values["call-01.response.json"])
    assert prefix.take({"a_new_request": True}) is None
    assert prefix.consumed == 1


def test_changed_request_is_not_a_cache_hit(tmp_path):
    saved(tmp_path)
    prefix = SavedPolicyPrefix(tmp_path, prefix_digest(tmp_path), "test")
    with pytest.raises(ValueError, match="request differs"):
        prefix.take({"model": "test", "messages": []})
    assert prefix.consumed == 0


def test_changed_prefix_bytes_or_model_are_rejected(tmp_path):
    saved(tmp_path)
    digest = prefix_digest(tmp_path)
    with pytest.raises(ValueError, match="model mismatch"):
        SavedPolicyPrefix(tmp_path, digest, "other")
    (tmp_path / "call-01.response.json").write_text('{"model":"test","id":"altered"}')
    with pytest.raises(ValueError, match="checksum mismatch"):
        SavedPolicyPrefix(tmp_path, digest, "test")
