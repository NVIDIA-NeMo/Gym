# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest

from resources_servers.gdpval import preconvert


INPUT = "reference_files/hash/input.wav"


@pytest.fixture
def prepared(tmp_path):
    source, candidate = tmp_path / "source", tmp_path / "candidate"
    relative = Path("task_one/repeat_0") / INPUT
    original, proxy = source / relative, candidate / relative.with_suffix(".wav.flac")
    for path, content in ((original, b"original benchmark input"), (proxy, b"prepared proxy")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    receipt = tmp_path / "candidate.media.json"
    receipt.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "source": relative.as_posix(),
                        "output": proxy.relative_to(candidate).as_posix(),
                        "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                        "output_sha256": hashlib.sha256(proxy.read_bytes()).hexdigest(),
                    }
                ]
            }
        )
    )
    return source, candidate, original, proxy, receipt


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("name", [INPUT, "hash/input.wav"])
def test_input_receipt_maps_original_name_to_proxy(prepared, serialized, name):
    source, candidate, original, proxy, _ = prepared
    inputs = json.dumps([name]) if serialized else [name]
    preconvert.check_reference_inputs([{"task_id": "one", "reference_files": inputs}], source, candidate)
    assert original.read_bytes() == b"original benchmark input"
    assert proxy.read_bytes() == b"prepared proxy"


@pytest.mark.parametrize("file_index", [2, 3], ids=["source", "prepared"])
def test_input_hash_tampering_is_rejected(prepared, file_index):
    source, candidate, *_ = prepared
    prepared[file_index].write_bytes(b"changed after preparation")
    with pytest.raises(ValueError, match="benchmark input changed since preparation"):
        preconvert.check_reference_inputs([{"task_id": "one", "reference_files": [INPUT]}], source, candidate)


@pytest.mark.parametrize(
    "file_index,error",
    [(4, "no conversion receipt"), (2, "missing prepared benchmark input"), (3, "missing prepared benchmark input")],
    ids=["receipt", "source", "prepared"],
)
def test_missing_receipt_or_input_is_rejected(prepared, file_index, error):
    source, candidate, *_ = prepared
    prepared[file_index].unlink()
    with pytest.raises(ValueError, match=error):
        preconvert.check_reference_inputs([{"task_id": "one", "reference_files": [INPUT]}], source, candidate)


def test_only_allowlisted_missing_tasks_skip_input_checks(tmp_path):
    rows = [{"task_id": "missing", "reference_files": [INPUT]}]
    source, candidate = tmp_path / "source", tmp_path / "candidate"
    preconvert.check_reference_inputs(rows, source, candidate, {"missing"})
    with pytest.raises(ValueError, match="no conversion receipt"):
        preconvert.check_reference_inputs(rows, source, candidate, {"other"})


@pytest.mark.parametrize("name", ["/absolute/input.wav", "../input.wav", "reference_files/../input.wav"])
def test_unsafe_declared_input_path_is_rejected(prepared, name):
    source, candidate, *_ = prepared
    with pytest.raises(ValueError, match="invalid benchmark input path"):
        preconvert.check_reference_inputs([{"task_id": "one", "reference_files": [name]}], source, candidate)


def test_office_failure_is_logged_and_source_is_retained(tmp_path, monkeypatch, capsys):
    original = tmp_path / "report.docx"
    original.write_bytes(b"original Office document")

    def failed_conversion(root, max_concurrent):
        assert root == tmp_path and max_concurrent == 4
        return 0, 1, ["converter failed"]

    monkeypatch.setattr(preconvert, "preconvert_dir", failed_conversion)
    preconvert.office(tmp_path)
    assert original.read_bytes() == b"original Office document"
    output = capsys.readouterr().out
    assert "Office: converted=0, failed=1" in output
    assert "Office render skipped: converter failed" in output
