# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The conversion inspector must run end to end on the judge's real conversion path."""

import sys

import pytest

from resources_servers.gdpval import inspect_conversion


fitz = pytest.importorskip("fitz")


def _write_pdf(path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def _run(monkeypatch, *arguments: str) -> int:
    monkeypatch.setattr(sys, "argv", ["inspect_conversion", *arguments])
    return inspect_conversion.main()


def test_directory_mode_writes_pages_and_skips_run_state(tmp_path, monkeypatch):
    task = tmp_path / "task"
    task.mkdir()
    _write_pdf(task / "report.pdf", "Quarterly revenue grew")
    (task / "finish_params.json").write_text("{}")
    out = tmp_path / "preview"

    assert _run(monkeypatch, "--input", str(task), "--out", str(out), "--dpi", "72") == 0

    summary = (out / "SUMMARY.txt").read_text()
    assert "report.pdf" in summary
    assert "finish_params" not in summary
    assert sorted(path.name for path in (out / "report").glob("page_*.png")) == ["page_001.png"]
    assert (out / "report" / "MANIFEST.txt").is_file()


def test_single_file_mode_flags_run_state_files(tmp_path, monkeypatch, capsys):
    run_state = tmp_path / "finish_params.json"
    run_state.write_text("{}")

    assert _run(monkeypatch, "--input", str(run_state), "--out", str(tmp_path / "preview")) == 0

    assert "is in IGNORE_FILES and would be skipped by the judge" in capsys.readouterr().out


def test_missing_input_fails(tmp_path, monkeypatch):
    assert _run(monkeypatch, "--input", str(tmp_path / "absent.pdf"), "--out", str(tmp_path / "preview")) == 2
