# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    ("module_name", "split", "artifact", "output_name"),
    [
        ("environments.calendar.prepare", "train", "train.jsonl", "train.jsonl"),
        ("environments.calendar_v2.prepare", "validation", "validation.jsonl", "validation.jsonl"),
        (
            "environments.code_gen.prepare",
            "validation",
            "validation.jsonl",
            "livecodebench_v5_2024-07-01_2025-02-01_validation.jsonl",
        ),
    ],
)
def test_huggingface_prepare_copies_without_consuming_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    split: str,
    artifact: str,
    output_name: str,
) -> None:
    source = tmp_path / "cache" / artifact
    source.parent.mkdir()
    source.write_text('{"task": 1}\n', encoding="utf-8")
    download = Mock(return_value=str(source))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "environment" / "prepare.py"))
    output = tmp_path / "environment" / "data" / output_name

    module.prepare(split)

    assert output.read_bytes() == source.read_bytes()
    assert source.read_text(encoding="utf-8") == '{"task": 1}\n'
    source.write_text('{"task": 2}\n', encoding="utf-8")

    module.prepare(split)

    assert output.read_text(encoding="utf-8") == '{"task": 2}\n'
    assert source.read_text(encoding="utf-8") == '{"task": 2}\n'
    assert download.call_count == 2
    download.assert_called_with(repo_id=module.REPO_ID, filename=artifact, repo_type="dataset")
