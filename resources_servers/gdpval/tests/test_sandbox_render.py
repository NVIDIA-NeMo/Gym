# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Office files rendered to PDF sidecars by LibreOffice inside the task sandbox."""

import json
import shlex
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openpyxl import Workbook

from nemo_gym.client import ServerClient
from nemo_gym.sandbox import SandboxExecResult
from resources_servers.gdpval import setup_libreoffice, task_sandbox
from resources_servers.gdpval.app import GDPValResourcesServer, GDPValResourcesServerConfig
from resources_servers.gdpval.preconvert import resolve_pdf_provenance


class FakeSandbox:
    """Sandbox paths live under ``root``; the LibreOffice command writes a PDF unless told to fail."""

    def __init__(
        self,
        root: Path,
        failures: int = 0,
        inconclusive: int = 0,
        upload_error: bool = False,
        cleanup_error: bool = False,
    ):
        self.root = root
        self.failures = failures
        # Conversions that write a PDF but end without a conclusive result, as on a timeout.
        self.inconclusive = inconclusive
        self.upload_error = upload_error
        self.cleanup_error = cleanup_error
        self.conversions: list[list[str]] = []
        self.conversion_timeouts: list[float] = []

    def local(self, remote: str) -> Path:
        return self.root / remote.lstrip("/")

    async def exec(self, command: str, *, timeout_s: float, **_) -> SandboxExecResult:
        argv = shlex.split(command)
        if argv[:2] == ["mkdir", "-p"]:
            self.local(argv[2]).mkdir(parents=True)
        elif argv[:2] == ["rm", "-rf"]:
            if self.cleanup_error:
                raise ConnectionResetError("stream closed")
            shutil.rmtree(self.local(argv[2]), ignore_errors=True)
        else:
            self.conversions.append(argv)
            self.conversion_timeouts.append(timeout_s)
            if self.failures:
                self.failures -= 1
                return SandboxExecResult(stdout="", stderr="source file could not be loaded", return_code=0)
            outdir, source = self.local(argv[argv.index("--outdir") + 1]), Path(argv[-1])
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / f"{source.stem}.pdf").write_bytes(b"%PDF-1.7 " + source.name.encode())
            if self.inconclusive:
                self.inconclusive -= 1
                return SandboxExecResult(stdout=None, stderr=None, return_code=125, error_type="timeout")
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def upload(self, local_path: Path, remote_path: str) -> None:
        if self.upload_error:
            raise OSError("connection reset")
        shutil.copy2(local_path, self.local(remote_path))

    async def download(self, remote_path: str, local_path: Path) -> None:
        shutil.copy2(self.local(remote_path), local_path)


def _workbook(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    workbook.active["A1"] = 1
    workbook.save(path)
    return path


async def test_every_office_file_gets_a_sidecar_that_provenance_trusts(tmp_path):
    staging = tmp_path / "staging"
    report = _workbook(staging / "P&L Report.xlsx")
    reference = _workbook(staging / "reference_files" / "abc" / "Ref File.xlsx")
    (staging / "notes.txt").write_text("not office")
    sandbox = FakeSandbox(tmp_path / "sandbox")

    await task_sandbox.render_office_in_sandbox(sandbox, staging, ["soffice"])

    assert (staging / "P&L Report.xlsx.pdf").read_bytes() == b"%PDF-1.7 P&L_Report.xlsx"
    assert (staging / "reference_files" / "abc" / "Ref File.xlsx.pdf").exists()
    assert not (staging / "notes.txt.pdf").exists()
    assert all(argv[0] == "soffice" and argv[-1].endswith(".xlsx") for argv in sandbox.conversions)
    assert not any(" " in Path(argv[-1]).name for argv in sandbox.conversions), "LibreOffice drops spaced names"
    assert resolve_pdf_provenance(staging.iterdir()).office_pdfs == {report: staging / "P&L Report.xlsx.pdf"}
    assert reference.exists()
    assert not any((tmp_path / "sandbox" / "tmp").iterdir()), "the sandbox scratch directories are removed"


async def test_an_existing_sidecar_is_not_rendered_again(tmp_path):
    staging = tmp_path / "staging"
    _workbook(staging / "plan.xlsx")
    (staging / "plan.xlsx.pdf").write_bytes(b"%PDF earlier")
    sandbox = FakeSandbox(tmp_path / "sandbox")

    await task_sandbox.render_office_in_sandbox(sandbox, staging, ["libreoffice"])

    assert sandbox.conversions == []
    assert (staging / "plan.xlsx.pdf").read_bytes() == b"%PDF earlier"


async def test_a_rejected_workbook_is_retried_once_from_a_round_trip_copy(tmp_path):
    staging = tmp_path / "staging"
    _workbook(staging / "plan.xlsx")
    sandbox = FakeSandbox(tmp_path / "sandbox", failures=1)

    await task_sandbox.render_office_in_sandbox(sandbox, staging, ["libreoffice"])

    assert len(sandbox.conversions) == 2
    assert (staging / "plan.xlsx.pdf").exists()


async def test_a_file_that_never_renders_keeps_no_sidecar(tmp_path, caplog):
    staging = tmp_path / "staging"
    _workbook(staging / "plan.xlsx")
    sandbox = FakeSandbox(tmp_path / "sandbox", failures=2)

    await task_sandbox.render_office_in_sandbox(sandbox, staging, ["libreoffice"])

    assert not (staging / "plan.xlsx.pdf").exists()
    assert "Could not render plan.xlsx in the sandbox" in caplog.text


@pytest.mark.parametrize("inconclusive", [1, 2])
async def test_a_pdf_from_an_inconclusive_conversion_is_never_kept(tmp_path, inconclusive):
    staging = tmp_path / "staging"
    _workbook(staging / "plan.xlsx")
    sandbox = FakeSandbox(tmp_path / "sandbox", inconclusive=inconclusive)

    await task_sandbox.render_office_in_sandbox(sandbox, staging, ["libreoffice"])

    # A workbook gets one retry: it renders when that retry completes, and keeps no sidecar otherwise.
    assert len(sandbox.conversions) == 2
    assert (staging / "plan.xlsx.pdf").exists() == (inconclusive == 1)


async def test_a_sandbox_operation_that_raises_leaves_the_file_unrendered(tmp_path, caplog):
    staging = tmp_path / "staging"
    _workbook(staging / "plan.xlsx")
    sandbox = FakeSandbox(tmp_path / "sandbox", upload_error=True)

    await task_sandbox.render_office_in_sandbox(sandbox, staging, ["libreoffice"])

    assert (staging / "plan.xlsx").exists()
    assert not (staging / "plan.xlsx.pdf").exists()
    assert "sandbox render failed" in caplog.text


async def test_a_failed_scratch_cleanup_keeps_the_render(tmp_path):
    staging = tmp_path / "staging"
    _workbook(staging / "plan.xlsx")
    sandbox = FakeSandbox(tmp_path / "sandbox", cleanup_error=True)

    await task_sandbox.render_office_in_sandbox(sandbox, staging, ["libreoffice"])

    assert (staging / "plan.xlsx.pdf").exists()


async def test_files_left_when_the_render_budget_runs_out_are_not_rendered(tmp_path, monkeypatch, caplog):
    staging = tmp_path / "staging"
    _workbook(staging / "a.xlsx")
    _workbook(staging / "b.xlsx")
    sandbox = FakeSandbox(tmp_path / "sandbox")
    # Each reading of the clock is 100 s later: the first file renders, then the 250 s budget is gone.
    clock = iter(range(0, 10_000, 100))
    monkeypatch.setattr(task_sandbox, "time", SimpleNamespace(monotonic=lambda: next(clock)))

    await task_sandbox.render_office_in_sandbox(sandbox, staging, ["libreoffice"], budget_s=250)

    assert (staging / "a.xlsx.pdf").exists()
    assert not (staging / "b.xlsx.pdf").exists()
    # The run got only what was left of the budget (250 - 200 s), not the full per-run cap.
    assert sandbox.conversion_timeouts == [50]
    assert "budget of 250s used up; not rendering ['b.xlsx']" in caplog.text


async def test_a_run_gets_no_more_than_the_budget_left(tmp_path):
    staging = tmp_path / "staging"
    _workbook(staging / "plan.xlsx")
    sandbox = FakeSandbox(tmp_path / "sandbox")

    await task_sandbox.render_office_in_sandbox(sandbox, staging, ["libreoffice"], budget_s=30)

    assert (staging / "plan.xlsx.pdf").exists()
    assert 0 < sandbox.conversion_timeouts[0] <= 30


@pytest.mark.parametrize("render", [None, ["libreoffice"]])
async def test_collect_deliverables_renders_only_when_asked(tmp_path, render):
    sandbox = FakeSandbox(tmp_path / "sandbox")
    _workbook(sandbox.local(f"{task_sandbox.WORKDIR}/out/plan.xlsx"))
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    task_dir = tmp_path / "persist" / "task_t" / "repeat_0"

    await task_sandbox.collect_deliverables(
        sandbox,
        {"tool": "finish", "paths": ["out/plan.xlsx"], "reason": "done"},
        reference_dir,
        task_dir,
        render_libreoffice_command=render,
    )

    assert (task_dir / "plan.xlsx").exists()
    assert (task_dir / "plan.xlsx.pdf").exists() == (render is not None)
    assert json.loads((task_dir / "finish_params.json").read_text())["paths"] == ["out/plan.xlsx"]


def test_a_sandbox_command_skips_the_host_libreoffice_install_even_in_comparison_mode(tmp_path, monkeypatch):
    ensure = MagicMock(return_value=False)
    monkeypatch.setattr(setup_libreoffice, "ensure_libreoffice", ensure)
    refs = tmp_path / "refs"
    refs.mkdir()
    config = GDPValResourcesServerConfig(
        host="127.0.0.1",
        port=8000,
        name="resources",
        entrypoint="app.py",
        judge_model_server={"type": "responses_api_models", "name": "judge"},
        tavily_api_key="test-key",
        sandbox_provider="test",
        sandbox_config={},
        persist_deliverables_dir=str(tmp_path / "persist"),
        reward_mode="comparison",
        reference_deliverables_dir=str(refs),
        sandbox_libreoffice_command=["libreoffice"],
    )

    GDPValResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    ensure.assert_not_called()
