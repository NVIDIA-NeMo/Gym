# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from benchmarks.webvoyager import prepare_native_v3 as webvoyager_prepare


def test_webvoyager_native_prepare_delegates_optional_source(monkeypatch, tmp_path) -> None:
    output = tmp_path / "prepared.jsonl"
    calls = []
    monkeypatch.setattr(
        webvoyager_prepare,
        "prepare_native",
        lambda source_path, output_path: calls.append((source_path, output_path)) or output_path,
    )

    assert webvoyager_prepare.prepare(output=output) == output
    assert calls == [(None, output)]


def test_webvoyager_native_prepare_delegates_explicit_source(monkeypatch, tmp_path) -> None:
    source = tmp_path / "webvoyager.jsonl"
    output = tmp_path / "prepared.jsonl"
    calls = []
    monkeypatch.setattr(
        webvoyager_prepare,
        "prepare_native",
        lambda source_path, output_path: calls.append((source_path, output_path)) or output_path,
    )

    assert webvoyager_prepare.prepare(source=source, output=output) == output
    assert calls == [(source, output)]
