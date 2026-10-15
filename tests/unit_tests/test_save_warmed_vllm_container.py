# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock
from urllib.error import URLError

import pytest


@pytest.fixture
def helper() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "benchmarks/nemotron_3.5_super/save_warmed_vllm_container.py"
    spec = importlib.util.spec_from_file_location("save_warmed_vllm_container", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_waits_for_all_replicas(helper: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    response = MagicMock()
    response.__enter__.return_value.status = 200
    opener = MagicMock()
    opener.open.side_effect = [response, URLError("still loading"), response, response]
    monkeypatch.setattr(helper.urllib.request, "build_opener", lambda *args: opener)
    monkeypatch.setattr(helper.time, "sleep", lambda seconds: None)
    helper.wait_for_engines(["http://node:8001", "http://node:8002"], 10)
    assert [call.args[0] for call in opener.open.call_args_list] == [
        "http://node:8001/health",
        "http://node:8002/health",
        "http://node:8001/health",
        "http://node:8002/health",
    ]


def test_readiness_timeout(helper: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helper.time, "monotonic", iter([0, 11]).__next__)
    with pytest.raises(TimeoutError, match="startup timed out"):
        helper.wait_for_engines(["http://node:8001"], 10)


def test_collect_requires_current_warmup(helper: ModuleType, tmp_path: Path) -> None:
    marker = tmp_path / ".warmup-start"
    marker.touch()
    os.utime(marker, (100, 100))
    directory = tmp_path / ("a" * 64)
    directory.mkdir()
    path = directory / "autotune_configs.json"
    contents = json.dumps({"_metadata": {"gpu": "GB200"}, "ssu|shape": ["runner", 1]})
    path.write_text(contents)
    os.utime(path, (90, 90))
    with pytest.raises(ValueError, match="No autotuning caches were written"):
        helper.collect_caches(tmp_path, allow_empty=False)
    os.utime(path, (110, 110))
    assert helper.collect_caches(tmp_path, allow_empty=False) == {"a" * 64: contents}


def test_headless_node_may_have_no_cache(helper: ModuleType, tmp_path: Path) -> None:
    (tmp_path / ".warmup-start").touch()
    assert helper.collect_caches(tmp_path, allow_empty=True) == {}
    with pytest.raises(ValueError, match="No autotuning caches"):
        helper.collect_caches(tmp_path, allow_empty=False)


def test_install_combines_roles_and_preserves_complete_files(helper: ModuleType, tmp_path: Path) -> None:
    prefill = json.dumps({"_metadata": {"gpu": "GB200"}, "prefill|shape": ["runner", 1]})
    decode = json.dumps({"_metadata": {"gpu": "GB200"}, "decode|shape": ["runner", 2]})
    another_replica = json.dumps({"_metadata": {"gpu": "GB200"}, "prefill|shape": ["runner", 3]})
    helper.install_caches(
        tmp_path,
        [json.dumps({"a" * 64: prefill}), json.dumps({"b" * 64: decode}), json.dumps({"a" * 64: another_replica})],
    )
    assert (tmp_path / ("a" * 64) / "autotune_configs.json").read_text() == prefill
    assert (tmp_path / ("b" * 64) / "autotune_configs.json").read_text() == decode
    assert (tmp_path / ".baked").is_file()


@pytest.mark.parametrize("contents", ["{}", '{"_metadata": {}}', "[]", "invalid json"])
def test_rejects_invalid_cache(helper: ModuleType, tmp_path: Path, contents: str) -> None:
    with pytest.raises(ValueError):
        helper.install_caches(tmp_path, [json.dumps({"a" * 64: contents})])
    assert not (tmp_path / ".baked").exists()


def test_rejects_path_traversal_and_empty_collection(helper: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid vLLM cache configuration hash"):
        helper.install_caches(tmp_path, [json.dumps({"../outside": '{"op": 1}'})])
    with pytest.raises(ValueError, match="No autotuning caches collected"):
        helper.install_caches(tmp_path, ["{}"])


def test_rejects_incompatible_replica_entries(helper: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="incompatible cache entry sets"):
        helper.install_caches(tmp_path, [json.dumps({"a" * 64: '{"op1": 1}'}), json.dumps({"a" * 64: '{"op2": 2}'})])
    assert not (tmp_path / ".baked").exists()
