# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned upstream prompt extraction must never execute downloaded code."""

from io import BytesIO
from unittest.mock import Mock

import pytest

from resources_servers.genrm_eval import upstream_prompt as module


@pytest.mark.parametrize(
    "definition",
    [
        'judge_prompt = "Test prompt {question}"',
        'def judge_prompt():\n    return "Test prompt {question}"',
    ],
)
def test_download_extract_and_cache(tmp_path, monkeypatch, definition):
    source = 'raise RuntimeError("must never execute")\n' + definition
    download = Mock(return_value=BytesIO(source.encode()))
    monkeypatch.setattr(module, "urlopen", download)
    kwargs = dict(
        repo="example/repo", revision="pinned-revision", filename="prompts.py", name="judge_prompt", cache_dir=tmp_path
    )
    assert module.download_prompt(**kwargs) == "Test prompt {question}"
    assert module.download_prompt(**kwargs) == "Test prompt {question}"
    download.assert_called_once_with(
        "https://raw.githubusercontent.com/example/repo/pinned-revision/prompts.py", timeout=60
    )


@pytest.mark.parametrize(
    "source",
    [
        'other_prompt = "test"',
        'judge_prompt = ""',
        "judge_prompt = 1",
        "judge_prompt = dangerous_call()",
        "def judge_prompt():\n    return dangerous_call()",
    ],
)
def test_bad_sources_fail_without_caching(tmp_path, monkeypatch, source):
    monkeypatch.setattr(module, "urlopen", Mock(return_value=BytesIO(source.encode())))
    with pytest.raises(ValueError):
        module.download_prompt(
            repo="example/repo", revision="pin", filename="prompt.py", name="judge_prompt", cache_dir=tmp_path
        )
    assert not (tmp_path / "pin/prompt.py").exists()


def test_download_failure_has_no_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "urlopen", Mock(side_effect=OSError("offline")))
    with pytest.raises(OSError, match="offline"):
        module.download_prompt(
            repo="example/repo", revision="pin", filename="prompt.py", name="judge_prompt", cache_dir=tmp_path
        )


def test_benchmark_sources_use_existing_revision_pins(monkeypatch):
    from benchmarks.genrm_mmrb1 import build_validation as v1
    from benchmarks.genrm_mmrb1.prepare import UPSTREAM_REVISION
    from benchmarks.genrm_mmrb2 import build_validation as v2

    for benchmark, revision, name in [
        (v1, UPSTREAM_REVISION, "judge_prompt"),
        (v2, v2.UPSTREAM_REVISION, "get_reasoning_prompt"),
    ]:
        download = Mock(return_value="Synthetic prompt")
        monkeypatch.setattr(benchmark, "download_prompt", download)
        assert benchmark.load_prompt() == "Synthetic prompt"
        assert download.call_args.kwargs["revision"] == revision
        assert download.call_args.kwargs["name"] == name
