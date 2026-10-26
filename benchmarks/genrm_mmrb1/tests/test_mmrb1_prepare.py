# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest

from benchmarks.genrm_mmrb1 import prepare as module


def test_use_converted_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(module, "__file__", str(tmp_path / "prepare.py"))
    source = tmp_path / "source.jsonl"
    source.write_text('{"id":"test","verifier_metadata":{"judge_mode":"llm_as_a_judge"}}\n')
    output = tmp_path / "data/genrm_mmrb1_llm_as_a_judge_validation.jsonl"
    assert module.prepare(source, judge_mode="llm_as_a_judge") == output
    assert output.read_bytes() == source.read_bytes()
    assert module.prepare(judge_mode="llm_as_a_judge") == output
    assert module.prepare(output, judge_mode="llm_as_a_judge") == output


def test_missing_explicit_source(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(module, "__file__", str(tmp_path / "prepare.py"))
    with pytest.raises(FileNotFoundError):
        module.prepare(tmp_path / "missing.jsonl")


@pytest.mark.parametrize("mode", ["genrm", "llm_as_a_judge"])
@pytest.mark.parametrize("same_path", [False, True])
def test_import_rejects_wrong_mode_without_overwrite(tmp_path, monkeypatch, mode, same_path):
    monkeypatch.setattr(module, "__file__", str(tmp_path / "prepare.py"))
    output = tmp_path / "data" / f"genrm_mmrb1_{mode}_validation.jsonl"
    output.parent.mkdir()
    other = "genrm" if mode == "llm_as_a_judge" else "llm_as_a_judge"
    original = json.dumps({"id": "test", "verifier_metadata": {"judge_mode": other}}) + "\n"
    output.write_text(original)
    source = output if same_path else tmp_path / "source.jsonl"
    source.write_text(original)
    with pytest.raises(ValueError, match="selected judge_mode"):
        module.prepare(source, judge_mode=mode)
    assert output.read_text() == original


def test_empty_import_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "__file__", str(tmp_path / "prepare.py"))
    source = tmp_path / "empty.jsonl"
    source.write_text("")
    with pytest.raises(ValueError, match="selected judge_mode"):
        module.prepare(source)


@pytest.mark.parametrize("mode", ["genrm", "llm_as_a_judge"])
def test_download_build_and_reuse(tmp_path: Path, monkeypatch, mode: str) -> None:
    from benchmarks.genrm_mmrb1 import build_validation

    monkeypatch.setattr(module, "__file__", str(tmp_path / "prepare.py"))
    urls = []

    def download(url, *, timeout):
        urls.append(url)
        assert timeout == 60
        return BytesIO(b"# upstream script" if url.endswith(".py") else b"[]")

    monkeypatch.setattr(module, "urlopen", download)

    def run(command, *, cwd, check):
        assert command == [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--python",
            module.sys.executable,
            "--with",
            "datasets==3.1.0",
            "--with",
            "Pillow",
            "--with",
            "tqdm",
            "python",
            str((tmp_path / "run_downloader.py").resolve()),
            "scripts/0_download_data.py",
        ]
        assert check is True
        assert (cwd / command[-1]).is_file()
        assert (cwd / "data/all_data_release.json").is_file()
        (cwd / "data/all_data.json").write_text(json.dumps([{}] * 4711))

    runner = Mock(side_effect=run)
    monkeypatch.setattr(module.subprocess, "run", runner)

    def build(source, output, *, genrm):
        assert source == tmp_path / ".cache" / module.UPSTREAM_REVISION
        assert genrm == (mode == "genrm")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"id":"converted"}\n')

    builder = Mock(side_effect=build)
    monkeypatch.setattr(build_validation, "prepare", builder)
    output = module.prepare(judge_mode=mode)
    assert output.read_text() == '{"id":"converted"}\n'
    assert module.prepare(judge_mode=mode) == output
    assert len(urls) == 2
    assert all(module.UPSTREAM_REVISION in url for url in urls)
    assert runner.call_count == builder.call_count == 1
    # A missing converted file reuses completed source data without downloading again.
    output.unlink()
    module.prepare(judge_mode=mode)
    assert runner.call_count == 1
    assert builder.call_count == 2


def test_incomplete_source_is_not_published(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(module, "__file__", str(tmp_path / "prepare.py"))
    completed = tmp_path / ".cache" / module.UPSTREAM_REVISION / "data/all_data.json"
    completed.parent.mkdir(parents=True)
    completed.write_text("[]")
    with pytest.raises(ValueError, match="4,711"):
        module.prepare()
    assert not (tmp_path / "data/genrm_mmrb1_genrm_validation.jsonl").exists()


@pytest.mark.parametrize(
    "content",
    [
        '{"id":"x","verifier_metadata":null}\n',
        '{"id":"x","verifier_metadata":["unexpected"]}\n',
        '{"id":"x","verifier_metadata":"unexpected"}\n',
        '{"id":"x","verifier_metadata":42}\n',
        "not json\n",
        "[1]\n",
    ],
)
def test_import_rejects_malformed_rows(tmp_path, monkeypatch, content):
    monkeypatch.setattr(module, "__file__", str(tmp_path / "prepare.py"))
    source = tmp_path / "bad.jsonl"
    source.write_text(content)
    with pytest.raises(ValueError, match="selected judge_mode"):
        module.prepare(source)
    assert not (tmp_path / "data/genrm_mmrb1_genrm_validation.jsonl").exists()
