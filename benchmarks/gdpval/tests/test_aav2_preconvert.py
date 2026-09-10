# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from benchmarks.gdpval.hsg.aav2 import preconvert


@pytest.fixture
def campaign(tmp_path):
    run_dir = tmp_path / "run"
    candidate = run_dir / "deliverables/task_one/repeat_0"
    candidate.mkdir(parents=True)
    (candidate / "report.docx").write_bytes(b"original candidate Office document")
    (candidate / "finish_params.json").write_text("{}")
    with zipfile.ZipFile(candidate / "source.zip", "w") as archive:
        archive.writestr("nested/input.docx", b"original ZIP Office document")
        archive.writestr("readme.txt", "Keep every member")
    reference = tmp_path / "reference/task_one/repeat_0"
    reference.mkdir(parents=True)
    (reference / "finish_params.json").write_text("{}")
    (reference / "selection.pdf").write_bytes(b"%PDF existing reference render")
    (run_dir / "judge.yaml").write_text(
        "gdpval_resources_server:\n  resources_servers:\n    gdpval:\n      reference_models:\n"
        f"        reference:\n          deliverables_dir: {tmp_path / 'reference'}\n          elo: 1223\n"
    )
    dataset = run_dir / "dataset.jsonl"
    dataset.write_text('{"task_id":"one"}\n')
    (run_dir / "run.json").write_text(
        json.dumps({"RUN_DIR": str(run_dir), "JUDGE_CONFIG": str(run_dir / "judge.yaml"), "DATASET": str(dataset)})
    )
    return run_dir


@pytest.fixture
def office_converter(monkeypatch):
    calls = []

    def convert(root, max_concurrent):
        root = Path(root)
        calls.append(root)
        assert max_concurrent == 4
        documents = list(root.rglob("*.docx"))
        for document in documents:
            document.with_name(document.name + ".pdf").write_bytes(b"%PDF-1.4\nrendered sidecar\n")
        return len(documents), 0, []

    monkeypatch.setattr(preconvert, "preconvert_dir", convert)
    return calls


def files(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_preconvert_copies_only_candidate_and_keeps_existing_references(campaign, office_converter, monkeypatch):
    source = campaign / "deliverables"
    references = campaign.parent / "reference"
    before = files(source)
    refs_before = files(references)
    judge = campaign / "judge.yaml"
    config_before = judge.read_bytes()
    real_open, real_scandir = Path.open, os.scandir

    def open_candidate(path, *args, **kwargs):
        assert path != judge and not path.is_relative_to(references), "reference inputs must not be opened"
        return real_open(path, *args, **kwargs)

    def scan_candidate(path):
        if not isinstance(path, int):
            assert not Path(path).is_relative_to(references), "reference trees must not be inventoried"
        return real_scandir(path)

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", open_candidate)
        guard.setattr(os, "scandir", scan_candidate)
        output = preconvert.prepare(campaign)
    assert files(source) == before
    assert files(references) == refs_before
    assert judge.read_bytes() == config_before
    assert (output / "candidate/task_one/repeat_0/report.docx.pdf").is_file()
    with zipfile.ZipFile(output / "candidate/task_one/repeat_0/source.zip") as archive:
        assert archive.read("nested/input.docx") == b"original ZIP Office document"
        assert archive.read("readme.txt") == b"Keep every member"
        assert archive.read("nested/input.docx.pdf").startswith(b"%PDF")
    assert not (output / "references").exists()
    assert not (output / "judge.yaml").exists()
    assert not (output / "manifest.json").exists()
    assert all(not root.is_relative_to(source) for root in office_converter)


@pytest.mark.parametrize("salvaged", [False, True])
def test_preconvert_reuses_completed_candidate_without_reconversion(campaign, office_converter, salvaged):
    output = campaign / "prepared"
    if salvaged:
        shutil.copytree(campaign / "deliverables", output / "candidate")
    else:
        preconvert.prepare(campaign)
    calls = len(office_converter)
    cache = output / "candidate/task_one/repeat_0_verify_response_0123456789ab.json"
    cache.write_text("{}")
    before = files(output)
    assert preconvert.prepare(campaign) == output
    assert len(office_converter) == calls
    assert files(output) == before


def test_incomplete_candidate_is_not_reused(campaign, office_converter):
    output = campaign / "prepared"
    shutil.copytree(campaign / "deliverables", output / "candidate")
    (output / "candidate/task_one/repeat_0/finish_params.json").unlink()
    with pytest.raises(ValueError, match="no finish marker"):
        preconvert.prepare(campaign)
    assert not office_converter


def test_preconvert_scope_excludes_unselected_candidate_tasks(campaign, office_converter):
    unrelated = campaign / "deliverables/task_other/repeat_0"
    unrelated.mkdir(parents=True)
    (unrelated / "missing-source").symlink_to(campaign / "does-not-exist")
    output = preconvert.prepare(campaign)
    assert not (output / "candidate/task_other").exists()
    assert (unrelated / "missing-source").is_symlink()


def test_container_entrypoint_needs_no_python_site_packages(campaign):
    candidate = campaign / "deliverables/task_one/repeat_0"
    (candidate / "report.docx").unlink()
    (candidate / "source.zip").unlink()
    subprocess.run(
        [sys.executable, "-S", preconvert.__file__, str(campaign)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3])},
    )
    assert (campaign / "prepared/candidate/task_one/repeat_0/finish_params.json").is_file()


def test_office_failure_is_logged_and_preparation_continues(campaign, monkeypatch, capsys):
    monkeypatch.setattr(preconvert, "preconvert_dir", lambda *_args, **_kwargs: (0, 1, ["converter failed"]))
    output = preconvert.prepare(campaign)
    assert (output / "candidate/task_one/repeat_0/report.docx").read_bytes() == b"original candidate Office document"
    assert "Office render skipped: converter failed" in capsys.readouterr().out
    assert not list(campaign.glob(".preparing-*"))


def test_zip_office_hook_cannot_modify_original_members(campaign, monkeypatch):
    before = files(campaign / "deliverables")

    def bad_conversion(root, **_kwargs):
        for document in Path(root).rglob("input.docx"):
            document.write_bytes(b"modified source")
            document.with_name(document.name + ".pdf").write_bytes(b"%PDF")
        return 1, 0, []

    monkeypatch.setattr(preconvert, "preconvert_dir", bad_conversion)
    with pytest.raises(ValueError, match="modified an original member"):
        preconvert.prepare(campaign)
    assert files(campaign / "deliverables") == before
    assert not (campaign / "prepared").exists()
