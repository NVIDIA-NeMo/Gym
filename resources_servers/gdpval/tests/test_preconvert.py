# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
from pathlib import Path

import pytest

from resources_servers.gdpval import preconvert as pcv


class TestNeedsConversion:
    def test_office_without_pdf_needs_conversion(self, tmp_path: Path) -> None:
        f = tmp_path / "a.docx"
        f.write_text("x")
        assert pcv.needs_conversion(f) is True

    def test_office_with_sibling_pdf_does_not(self, tmp_path: Path) -> None:
        f = tmp_path / "a.docx"
        f.write_text("x")
        (tmp_path / "a.pdf").write_text("p")
        assert pcv.needs_conversion(f) is False

    def test_non_office_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("x")
        assert pcv.needs_conversion(f) is False


class TestFindConvertibleFiles:
    def test_finds_recursively_and_sorted(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        (tmp_path / "b.docx").write_text("b")
        (tmp_path / "sub" / "a.xlsx").write_text("a")
        (tmp_path / "ignore.txt").write_text("t")
        # already-converted should be skipped
        (tmp_path / "c.pptx").write_text("c")
        (tmp_path / "c.pdf").write_text("p")
        files = pcv.find_convertible_files(str(tmp_path))
        # Sorted by full path: top-level b.docx < sub/a.xlsx; c.pptx skipped (sibling .pdf exists).
        assert [p.name for p in files] == ["b.docx", "a.xlsx"]


class TestConvertToPdfErrors:
    def test_returns_message_when_libreoffice_not_found(self, tmp_path: Path, monkeypatch) -> None:
        f = tmp_path / "a.docx"
        f.write_text("x")

        def _raise(*_a, **_kw):
            raise FileNotFoundError("libreoffice")

        monkeypatch.setattr(subprocess, "run", _raise)
        path, ok, msg = pcv.convert_to_pdf(f)
        assert ok is False
        assert "libreoffice not found" in msg

    def test_returns_message_when_libreoffice_runs_but_no_pdf(self, tmp_path: Path, monkeypatch) -> None:
        f = tmp_path / "a.docx"
        f.write_text("x")

        class _CompletedNoPdf:
            returncode = 1
            stdout = ""
            stderr = "Some libreoffice error"

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _CompletedNoPdf())
        path, ok, msg = pcv.convert_to_pdf(f)
        assert ok is False
        assert "did not produce" in msg
        assert "Some libreoffice error" in msg

    def test_passes_user_installation_flag(self, tmp_path: Path, monkeypatch) -> None:
        f = tmp_path / "a.docx"
        f.write_text("x")
        captured: list[list[str]] = []

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        def _run(cmd, *_a, **_kw):
            captured.append(cmd)
            f.with_suffix(".pdf").write_bytes(b"%PDF-1.4 fake\n")
            return _Completed()

        monkeypatch.setattr(subprocess, "run", _run)
        path, ok, _ = pcv.convert_to_pdf(f)
        assert ok is True
        assert len(captured) == 1
        env_flags = [arg for arg in captured[0] if arg.startswith("-env:UserInstallation=")]
        assert len(env_flags) == 1, f"expected one -env:UserInstallation flag, got {env_flags}"
        assert env_flags[0].startswith("-env:UserInstallation=file://")
        # The path should be a unique tempdir (one per call); just sanity-check it points to a path.
        assert "/lo-profile-" in env_flags[0]


class TestPreconvertDirSurfacesFailures:
    def test_returns_error_messages(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "a.docx").write_text("x")
        (tmp_path / "b.xlsx").write_text("x")

        def _convert(path: Path) -> tuple[Path, bool, str]:
            return path, False, f"forced fail on {path.name}"

        monkeypatch.setattr(pcv, "convert_to_pdf", _convert)

        ok, fail, errors = pcv.preconvert_dir(str(tmp_path))
        assert ok == 0
        assert fail == 2
        assert sorted(errors) == sorted(["forced fail on a.docx", "forced fail on b.xlsx"])

    def test_empty_dir_returns_zeros(self, tmp_path: Path) -> None:
        ok, fail, errors = pcv.preconvert_dir(str(tmp_path))
        assert (ok, fail, errors) == (0, 0, [])

    def test_success_path_returns_no_errors(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "a.docx").write_text("x")

        def _convert(path: Path) -> tuple[Path, bool, str]:
            (path.with_suffix(".pdf")).write_text("p")
            return path, True, "ok"

        monkeypatch.setattr(pcv, "convert_to_pdf", _convert)

        ok, fail, errors = pcv.preconvert_dir(str(tmp_path))
        assert (ok, fail, errors) == (1, 0, [])


@pytest.mark.asyncio
async def test_preconvert_dir_async_propagates_results(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.docx").write_text("x")

    monkeypatch.setattr(pcv, "convert_to_pdf", lambda p: (p, False, "boom"))
    ok, fail, errors = await pcv.preconvert_dir_async(str(tmp_path))
    assert (ok, fail) == (0, 1)
    assert errors == ["boom"]
