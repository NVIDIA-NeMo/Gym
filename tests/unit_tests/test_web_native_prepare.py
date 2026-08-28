# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from benchmarks.webvoyager import prepare_native_v3 as webvoyager_prepare


def test_webvoyager_native_prepare_uses_environment_source(monkeypatch, tmp_path) -> None:
    source = tmp_path / "webvoyager.jsonl"
    output = tmp_path / "prepared.jsonl"
    calls = []
    monkeypatch.setenv("WEBVOYAGER_SOURCE_JSONL", str(source))
    monkeypatch.setattr(
        webvoyager_prepare,
        "prepare_native",
        lambda source_path, output_path: calls.append((source_path, output_path)) or output_path,
    )

    assert webvoyager_prepare.prepare(output=output) == output
    assert calls == [(source, output)]


def test_native_prepare_fails_closed_without_source(monkeypatch) -> None:
    monkeypatch.delenv("WEBVOYAGER_SOURCE_JSONL", raising=False)

    with pytest.raises(RuntimeError, match="WEBVOYAGER_SOURCE_JSONL"):
        webvoyager_prepare.prepare()
