# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from benchmarks.gdpval.hsg.checkpoint_e2e import preconvert_closure as closure


def _write_ooxml(path: Path, members: list[tuple[str, bytes]] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = members or [
        ("[Content_Types].xml", b"<Types />"),
        ("word/document.xml", b"<document />"),
    ]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in rows:
            archive.writestr(name, data)
    return path


def _write_fake_preconvert(path: Path, *, valid_pdf: bool = True, assert_unique_members: bool = False) -> Path:
    unique_check = ""
    if assert_unique_members:
        unique_check = """
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate member reached converter")
"""
    payload = "b'%PDF-1.7 fake\\n'" if valid_pdf else "b'not-a-pdf'"
    path.write_text(
        """
import os
import zipfile
from pathlib import Path

def convert_to_pdf(path):
    path = Path(path)
%s
    log = os.environ.get("FAKE_PRECONVERT_LOG")
    if log:
        Path(log).write_text(str(path), encoding="utf-8")
    path.with_suffix(".pdf").write_bytes(%s)
    return path, True, "fake converted"
"""
        % (unique_check, payload),
        encoding="utf-8",
    )
    return path


def test_inventory_distinguishes_produced_reference_and_ready_office(tmp_path: Path) -> None:
    deliverables = tmp_path / "deliverables"
    produced = _write_ooxml(deliverables / "task_a" / "repeat_0" / "answer.docx")
    reference = _write_ooxml(deliverables / "task_a" / "repeat_0" / "reference_files" / "input.xlsx")
    ready = _write_ooxml(deliverables / "task_b" / "repeat_0" / "slides.pptx")
    ready.with_suffix(".pdf").write_bytes(b"%PDF-ready\n")

    inventory = closure.build_inventory(deliverables)

    assert inventory["status"] == "OPEN"
    assert inventory["counts"] == {
        "office_total": 3,
        "ready": 1,
        "produced_missing": 1,
        "reference_exceptions": 1,
        "destination_collisions": 0,
    }
    assert [row["source"] for row in inventory["produced_missing"]] == [str(produced)]
    assert [row["source"] for row in inventory["reference_exceptions"]] == [str(reference)]


def test_inventory_command_emits_json_without_closure_failure(tmp_path: Path, capsys) -> None:
    deliverables = tmp_path / "deliverables"
    _write_ooxml(deliverables / "answer.docx")

    returncode = closure.main(["inventory", "--root", str(deliverables), "--json"])
    output = json.loads(capsys.readouterr().out)

    assert returncode == 0
    assert output["schema"] == closure.INVENTORY_SCHEMA
    assert output["counts"]["produced_missing"] == 1
    assert output["remaining_produced"] == 1


def test_inventory_treats_invalid_existing_pdf_as_remaining(tmp_path: Path) -> None:
    deliverables = tmp_path / "deliverables"
    produced = _write_ooxml(deliverables / "answer.docx")
    produced.with_suffix(".pdf").write_bytes(b"not-a-pdf")

    inventory = closure.build_inventory(deliverables)

    assert inventory["remaining_produced"] == 1
    assert inventory["produced_missing"][0]["pdf_error"] == "sibling PDF does not start with %PDF"
    assert inventory["produced_pairs"] == []


def test_convert_imports_module_stages_source_and_leaves_reference_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deliverables = tmp_path / "deliverables"
    produced = _write_ooxml(deliverables / "task_a" / "repeat_0" / "answer.docx")
    reference = _write_ooxml(deliverables / "task_a" / "repeat_0" / "reference_files" / "input.xlsx")
    converter = _write_fake_preconvert(tmp_path / "preconvert.py")
    call_log = tmp_path / "converter-path.txt"
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("FAKE_PRECONVERT_LOG", str(call_log))
    source_sha256 = hashlib.sha256(produced.read_bytes()).hexdigest()

    receipt = closure.convert_closure(deliverables, converter, workers=2, scratch_root=scratch)

    expected_pdf = produced.with_suffix(".pdf")
    assert receipt["status"] == "PASS"
    assert receipt["remaining_produced"] == []
    assert receipt["counts"]["converted"] == 1
    assert receipt["counts"]["reference_exceptions"] == 1
    assert receipt["reference_exceptions"][0]["source"] == str(reference)
    assert not reference.with_suffix(".pdf").exists()
    assert expected_pdf.read_bytes().startswith(b"%PDF")
    assert receipt["converted"][0]["publish_method"] == "hardlink"
    assert receipt["produced_pairs"][0]["source"] == str(produced)
    assert len(receipt["closure_fingerprint"]) == 64
    assert hashlib.sha256(produced.read_bytes()).hexdigest() == source_sha256
    staged_path = Path(call_log.read_text(encoding="utf-8"))
    assert closure._STAGE_PREFIX in staged_path.parent.name
    assert scratch in staged_path.parents
    assert staged_path != produced
    assert not staged_path.exists()


def test_identical_duplicate_ooxml_members_are_deduplicated_before_conversion(tmp_path: Path) -> None:
    deliverables = tmp_path / "deliverables"
    with pytest.warns(UserWarning, match="Duplicate name"):
        produced = _write_ooxml(
            deliverables / "answer.docx",
            [
                ("[Content_Types].xml", b"same"),
                ("[Content_Types].xml", b"same"),
                ("word/document.xml", b"doc"),
            ],
        )
    converter = _write_fake_preconvert(
        tmp_path / "preconvert.py",
        assert_unique_members=True,
    )

    receipt = closure.convert_closure(deliverables, converter, workers=1)

    assert receipt["status"] == "PASS"
    audit = receipt["normalization_audit"][0]
    assert audit["normalized"] is True
    assert audit["conflicting_members"] == []
    assert audit["duplicate_members"] == [
        {
            "name": "[Content_Types].xml",
            "occurrences": 2,
            "content_sha256": hashlib.sha256(b"same").hexdigest(),
        }
    ]
    assert produced.with_suffix(".pdf").read_bytes().startswith(b"%PDF")


def test_conflicting_duplicate_ooxml_members_fail_closed(tmp_path: Path) -> None:
    deliverables = tmp_path / "deliverables"
    with pytest.warns(UserWarning, match="Duplicate name"):
        produced = _write_ooxml(
            deliverables / "answer.xlsx",
            [
                ("[Content_Types].xml", b"first"),
                ("[Content_Types].xml", b"second"),
            ],
        )
    converter = _write_fake_preconvert(tmp_path / "preconvert.py")

    receipt = closure.convert_closure(deliverables, converter, workers=1)

    assert receipt["status"] == "INCOMPLETE"
    assert [row["source"] for row in receipt["remaining_produced"]] == [str(produced)]
    assert "non-identical duplicate members" in receipt["failures"][0]["error"]
    audit = receipt["normalization_audit"][0]
    assert audit["normalized"] is False
    assert [row["name"] for row in audit["conflicting_members"]] == ["[Content_Types].xml"]
    assert not produced.with_suffix(".pdf").exists()


def test_invalid_converter_output_keeps_produced_gap_and_main_exits_nonzero(tmp_path: Path, capsys) -> None:
    deliverables = tmp_path / "deliverables"
    produced = _write_ooxml(deliverables / "answer.pptx")
    converter = _write_fake_preconvert(tmp_path / "preconvert.py", valid_pdf=False)
    receipt_path = tmp_path / "receipt.json"

    returncode = closure.main(
        [
            "convert",
            "--root",
            str(deliverables),
            "--preconvert-py",
            str(converter),
            "--workers",
            "1",
            "--scratch",
            str(tmp_path / "scratch"),
            "--receipt",
            str(receipt_path),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert returncode == 1
    assert output["status"] == "INCOMPLETE"
    assert output["remaining_produced"][0]["source"] == str(produced)
    assert "valid PDF" in output["failures"][0]["error"]
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == output
    assert not produced.with_suffix(".pdf").exists()


def test_source_hash_change_prevents_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deliverables = tmp_path / "deliverables"
    produced = _write_ooxml(deliverables / "answer.docx")
    converter = tmp_path / "preconvert.py"
    converter.write_text(
        """
import os
from pathlib import Path

def convert_to_pdf(path):
    source = Path(os.environ["FAKE_LIVE_SOURCE"])
    source.write_bytes(source.read_bytes() + b"changed")
    path.with_suffix(".pdf").write_bytes(b"%PDF-1.7 fake\\n")
    return path, True, "fake converted"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("FAKE_LIVE_SOURCE", str(produced))

    receipt = closure.convert_closure(deliverables, converter, workers=1)

    assert receipt["status"] == "INCOMPLETE"
    assert "source hash changed during conversion" in receipt["failures"][0]["error"]
    assert not produced.with_suffix(".pdf").exists()


def test_same_stem_sources_get_distinct_injective_pdf_sidecars(tmp_path: Path) -> None:
    deliverables = tmp_path / "deliverables"
    docx = _write_ooxml(deliverables / "answer.docx")
    xlsx = _write_ooxml(deliverables / "answer.xlsx")
    converter = _write_fake_preconvert(tmp_path / "preconvert.py")

    receipt = closure.convert_closure(deliverables, converter, workers=2)

    docx_pdf = docx.with_name("answer.docx.pdf")
    xlsx_pdf = xlsx.with_name("answer.xlsx.pdf")
    assert receipt["status"] == "PASS"
    assert receipt["remaining_produced"] == []
    assert receipt["failures"] == []
    assert {row["expected_pdf"] for row in receipt["converted"]} == {
        str(docx_pdf),
        str(xlsx_pdf),
    }
    assert docx_pdf.read_bytes().startswith(b"%PDF")
    assert xlsx_pdf.read_bytes().startswith(b"%PDF")
    assert not docx.with_suffix(".pdf").exists()
    inventory = closure.build_inventory(deliverables)
    assert inventory["status"] == "CLOSED"
    assert inventory["destination_collisions"] == []
    assert {row["pdf"] for row in inventory["produced_pairs"]} == {
        str(docx_pdf),
        str(xlsx_pdf),
    }


def test_ambiguous_plain_pdf_is_not_shared_between_same_stem_sources(tmp_path: Path) -> None:
    deliverables = tmp_path / "deliverables"
    docx = _write_ooxml(deliverables / "answer.docx")
    xlsx = _write_ooxml(deliverables / "answer.xlsx")
    docx.with_suffix(".pdf").write_bytes(b"%PDF-ambiguous\n")

    inventory = closure.build_inventory(deliverables)

    assert inventory["status"] == "OPEN"
    assert inventory["remaining_produced"] == 2
    assert {row["expected_pdf"] for row in inventory["produced_missing"]} == {
        str(docx.with_name("answer.docx.pdf")),
        str(xlsx.with_name("answer.xlsx.pdf")),
    }
    assert inventory["destination_collisions"] == []


def test_existing_injective_sidecar_wins_for_unique_source(tmp_path: Path) -> None:
    deliverables = tmp_path / "deliverables"
    source = _write_ooxml(deliverables / "answer.docx")
    sidecar = source.with_name("answer.docx.pdf")
    sidecar.write_bytes(b"%PDF-injective\n")
    source.with_suffix(".pdf").write_bytes(b"%PDF-legacy\n")

    inventory = closure.build_inventory(deliverables)

    assert inventory["status"] == "CLOSED"
    assert inventory["produced_pairs"][0]["pdf"] == str(sidecar)


def test_produced_office_symlink_is_not_followed_or_converted(tmp_path: Path) -> None:
    deliverables = tmp_path / "deliverables"
    target = _write_ooxml(tmp_path / "outside.docx")
    deliverables.mkdir()
    source = deliverables / "answer.docx"
    source.symlink_to(target)
    converter = _write_fake_preconvert(tmp_path / "preconvert.py")

    receipt = closure.convert_closure(deliverables, converter, workers=1)

    assert receipt["status"] == "INCOMPLETE"
    assert receipt["failures"][0]["error"].endswith(f"is a symlink: {source}")
    assert not source.with_suffix(".pdf").exists()


def test_reference_only_gap_does_not_require_converter_or_block_closure(tmp_path: Path) -> None:
    deliverables = tmp_path / "deliverables"
    reference = _write_ooxml(deliverables / "task" / "reference_files" / "input.docx")

    receipt = closure.convert_closure(deliverables, tmp_path / "missing-preconvert.py", workers=1)

    assert receipt["status"] == "PASS"
    assert receipt["converted"] == []
    assert receipt["remaining_produced"] == []
    assert receipt["reference_exceptions"][0]["source"] == str(reference)
    assert not reference.with_suffix(".pdf").exists()


def test_timeout_proxy_replaces_converter_timeout_with_900_seconds() -> None:
    captured: dict[str, object] = {}

    class Wrapped:
        TimeoutExpired = subprocess.TimeoutExpired

        @staticmethod
        def run(*args, **kwargs):
            captured["args"] = args
            captured["timeout"] = kwargs["timeout"]
            return object()

    proxy = closure._TimeoutSubprocessProxy(Wrapped, closure.CONVERSION_TIMEOUT_SECONDS)

    proxy.run(["libreoffice"], timeout=120)

    assert captured["timeout"] == 900
    assert proxy.TimeoutExpired is subprocess.TimeoutExpired


def test_loaded_preconvert_module_receives_timeout_proxy(tmp_path: Path) -> None:
    module_path = tmp_path / "preconvert.py"
    module_path.write_text(
        """
import subprocess

def convert_to_pdf(path):
    return path, False, "unused"
""",
        encoding="utf-8",
    )

    module, module_sha256 = closure._load_preconvert_module(module_path)

    assert isinstance(module.subprocess, closure._TimeoutSubprocessProxy)
    assert module.subprocess._timeout_seconds == 900
    assert module_sha256 == hashlib.sha256(module_path.read_bytes()).hexdigest()
