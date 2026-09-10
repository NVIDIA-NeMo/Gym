# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.gdpval.hsg.aav2 import preconvert


@pytest.fixture
def campaign(tmp_path):
    run_dir = tmp_path / "run"
    candidate = run_dir / "deliverables/task_one/repeat_0"
    candidate.mkdir(parents=True)
    (candidate / "report.docx").write_bytes(b"original candidate Office document")
    (candidate / "finish_params.json").write_text("{}")
    reference = tmp_path / "reference/task_one/repeat_0"
    reference.mkdir(parents=True)
    (reference / "finish_params.json").write_text("{}")
    (reference / "selection.txt").write_text("Reference selection")
    nested = reference / "reference_files/asset"
    nested.mkdir(parents=True)
    with zipfile.ZipFile(nested / "source.zip", "w") as archive:
        archive.writestr("nested/reference.docx", b"original ZIP Office document")
        archive.writestr("readme.txt", "Keep every member")
    judge = run_dir / "judge.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "gdpval_resources_server": {
                    "resources_servers": {
                        "gdpval": {
                            "reference_models": {
                                "reference": {"deliverables_dir": str(tmp_path / "reference"), "elo": 1223}
                            }
                        }
                    }
                },
                "judge_model": {"api_key": "${oc.env:JUDGE_API_KEY}"},
            }
        ),
        judge,
    )
    dataset = run_dir / "dataset.jsonl"
    dataset.write_text('{"task_id":"one"}\n')
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "RUN_DIR": str(run_dir),
                "JUDGE_CONFIG": str(judge),
                "DATASET": str(dataset),
            }
        )
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


def test_preconvert_publishes_separate_trees_and_office_zip_sidecars(campaign, office_converter, monkeypatch):
    before = preconvert.inventory(campaign / "deliverables")
    refs = campaign.parent / "reference"
    refs_before = preconvert.inventory(refs)
    monkeypatch.setenv("JUDGE_API_KEY", "must-not-be-resolved")
    output = preconvert.prepare(campaign)
    assert preconvert.inventory(campaign / "deliverables") == before
    assert preconvert.inventory(refs) == refs_before
    assert (output / "candidate/task_one/repeat_0/report.docx.pdf").is_file()
    with zipfile.ZipFile(
        output / "references/reference/task_one/repeat_0/reference_files/asset/source.zip"
    ) as archive:
        assert archive.read("nested/reference.docx") == b"original ZIP Office document"
        assert archive.read("readme.txt") == b"Keep every member"
        assert archive.read("nested/reference.docx.pdf").startswith(b"%PDF")
    config = OmegaConf.load(output / "judge.yaml")
    reference = OmegaConf.select(config, preconvert.REFERENCE_KEY)["reference"]
    assert reference.deliverables_dir == str(output / "references/reference")
    assert reference.elo == 1223
    assert "${oc.env:JUDGE_API_KEY}" in (output / "judge.yaml").read_text()
    assert "must-not-be-resolved" not in (output / "judge.yaml").read_text()
    assert all(root != campaign / "deliverables" and root != refs for root in office_converter)
    media = json.loads((output / "references/reference.media.json").read_text())
    members = next(entry["members"] for entry in media["entries"] if entry["kind"] == "zip")
    generated = [member for member in members if member.get("generated")]
    assert [member["output"] for member in generated] == ["nested/reference.docx.pdf"]


def test_preconvert_reuses_only_matching_sources_and_outputs(campaign, office_converter):
    output = preconvert.prepare(campaign)
    calls = len(office_converter)
    cache = output / "candidate/task_one/repeat_0_verify_response_0123456789ab.json"
    cache.write_text("{}")
    assert preconvert.prepare(campaign) == output
    assert len(office_converter) == calls
    (output / "candidate/task_one/repeat_0/report.docx.pdf").write_bytes(b"changed")
    with pytest.raises(ValueError, match="input/output changed"):
        preconvert.prepare(campaign)


def test_preconvert_rejects_changed_originals(campaign, office_converter):
    preconvert.prepare(campaign)
    (campaign / "deliverables/task_one/repeat_0/report.docx").write_bytes(b"different original")
    with pytest.raises(ValueError, match="input/output changed"):
        preconvert.prepare(campaign)


def test_preconvert_scope_excludes_unselected_reference_tasks(campaign, office_converter):
    unrelated = campaign.parent / "reference/task_other/repeat_0"
    unrelated.mkdir(parents=True)
    (unrelated / "unsupported.psd").write_bytes(b"unsupported source")
    output = preconvert.prepare(campaign)
    assert not (output / "references/reference/task_other").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["inputs"]["task_ids"] == ["one"]
    assert all(
        "task_other" not in row["path"] for row in manifest["inputs"]["sources"]["references/reference"]["files"]
    )
    (unrelated / "unsupported.psd").write_bytes(b"changed unrelated source")
    assert preconvert.prepare(campaign) == output
    (campaign / "dataset.jsonl").write_text('{"task_id":"one","changed":true}\n')
    with pytest.raises(ValueError, match="input/output changed"):
        preconvert.prepare(campaign)


def test_absent_reference_task_stays_absent(campaign, office_converter):
    empty = campaign.parent / "empty_reference"
    empty.mkdir()
    judge = OmegaConf.load(campaign / "judge.yaml")
    OmegaConf.select(judge, preconvert.REFERENCE_KEY)["reference"].deliverables_dir = str(empty)
    OmegaConf.save(judge, campaign / "judge.yaml")
    output = preconvert.prepare(campaign)
    assert list((output / "references/reference").iterdir()) == []


def test_unrecognized_cache_file_invalidates_prepared_output(campaign, office_converter):
    output = preconvert.prepare(campaign)
    (output / "candidate/task_one/repeat_0_verify_response_0123456789abc.json").write_text("{}")
    with pytest.raises(ValueError, match="input/output changed"):
        preconvert.prepare(campaign)


def test_container_json_input_needs_no_python_site_packages(campaign):
    (campaign / "deliverables/task_one/repeat_0/report.docx").unlink()
    (campaign.parent / "reference/task_one/repeat_0/reference_files/asset/source.zip").unlink()
    config = OmegaConf.to_container(OmegaConf.load(campaign / "judge.yaml"), resolve=False)
    config_json = campaign / "judge.json"
    config_json.write_text(json.dumps(config))
    subprocess.run(
        [sys.executable, "-S", preconvert.__file__, str(campaign), "--config-json", str(config_json)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3])},
    )
    assert (campaign / "prepared/manifest.json").is_file()


def test_reference_mirror_is_materialized_with_link_provenance(campaign, office_converter):
    original = campaign.parent / "reference"
    mirror = campaign.parent / "mirror"
    task = mirror / "task_one/repeat_0"
    task.mkdir(parents=True)
    (original / "task_one/repeat_0/omitted.psd").write_bytes(b"excluded intermediate")
    (task / "selection.txt").symlink_to(original / "task_one/repeat_0/selection.txt")
    (task / "finish_params.json").symlink_to(original / "task_one/repeat_0/finish_params.json")
    (mirror / "FILTER_MANIFEST.txt").write_text("Visible files only")
    config = OmegaConf.load(campaign / "judge.yaml")
    OmegaConf.select(config, preconvert.REFERENCE_KEY)["reference"].deliverables_dir = str(mirror)
    OmegaConf.save(config, campaign / "judge.yaml")
    output = preconvert.prepare(campaign)
    copied = output / "references/reference"
    assert (copied / "task_one/repeat_0/selection.txt").read_text() == "Reference selection"
    assert not any(path.is_symlink() for path in copied.rglob("*"))
    assert not (copied / "task_one/repeat_0/omitted.psd").exists()
    assert (copied / "FILTER_MANIFEST.txt").read_text() == "Visible files only"
    manifest = json.loads((output / "manifest.json").read_text())
    links = manifest["reference_copies"]["references/reference"]["links"]
    assert {link["path"] for link in links} == {
        "task_one/repeat_0/finish_params.json",
        "task_one/repeat_0/selection.txt",
    }


@pytest.mark.parametrize("count", [0, 2])
def test_reference_repeat_count_fails_before_conversion(campaign, office_converter, count):
    task = campaign.parent / "reference/task_one"
    if count == 0:
        (task / "repeat_0/finish_params.json").unlink()
    else:
        (task / "repeat_1").mkdir()
        (task / "repeat_1/finish_params.json").write_text("{}")
    with pytest.raises(ValueError, match="exactly one completed reference repeat"):
        preconvert.prepare(campaign)
    assert not office_converter
    assert not (campaign / "prepared").exists()


@pytest.mark.parametrize("prefix", ["", "reference_files/"])
@pytest.mark.parametrize("legacy", [False, True])
def test_expected_reference_inputs_accept_current_and_legacy_layout(campaign, office_converter, prefix, legacy):
    (campaign / "dataset.jsonl").write_text(
        json.dumps({"task_id": "one", "reference_files": [prefix + "asset/source.zip"]}) + "\n"
    )
    refs = campaign.parent / "reference/task_one/repeat_0/reference_files"
    if legacy:
        (refs / "reference_files").mkdir()
        (refs / "asset").rename(refs / "reference_files/asset")
    assert preconvert.prepare(campaign).is_dir()


def test_missing_dataset_reference_input_fails_before_publication(campaign, office_converter):
    (campaign / "dataset.jsonl").write_text('{"task_id":"one","reference_files":["reference_files/missing.txt"]}\n')
    with pytest.raises(ValueError, match="missing dataset reference file"):
        preconvert.prepare(campaign)
    assert not (campaign / "prepared").exists()


def test_office_failure_is_logged_and_preparation_continues(campaign, monkeypatch, capsys):
    monkeypatch.setattr(preconvert, "preconvert_dir", lambda *_args, **_kwargs: (0, 1, ["converter failed"]))
    output = preconvert.prepare(campaign)
    assert (output / "manifest.json").is_file()
    assert (output / "candidate/task_one/repeat_0/report.docx").read_bytes() == b"original candidate Office document"
    assert "Office render skipped: converter failed" in capsys.readouterr().out
    assert not list(campaign.glob(".preparing-*"))


def test_zip_office_hook_cannot_modify_original_members(campaign, monkeypatch):
    def bad_conversion(root, **_kwargs):
        for document in Path(root).rglob("reference.docx"):
            document.write_bytes(b"modified source")
            document.with_name(document.name + ".pdf").write_bytes(b"%PDF")
        for document in Path(root).rglob("report.docx"):
            document.with_name(document.name + ".pdf").write_bytes(b"%PDF")
        return 1, 0, []

    monkeypatch.setattr(preconvert, "preconvert_dir", bad_conversion)
    with pytest.raises(ValueError, match="modified an original member"):
        preconvert.prepare(campaign)
    assert not (campaign / "prepared").exists()
