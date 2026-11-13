# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public category prompts, harness compatibility and fail-closed dataset preparation."""

import hashlib
import json
from copy import deepcopy
from unittest.mock import patch

import pytest

from benchmarks.cvdp_v110 import prepare as benchmark_prepare
from resources_servers.cvdp.app import _parse_model_response


@pytest.fixture
def raw_task():
    return {
        "id": "public-task",
        "categories": ["cid003"],
        "difficulty": "easy",
        "input": {"prompt": "Implement a latch.", "context": {"rtl/latch.sv": "module latch; endmodule", "empty": ""}},
        "output": {"context": {"rtl/latch.sv": "secret-reference-must-not-be-exported"}},
        "harness": {
            "files": {
                "src/lint.vlt": "keep\r\n  lint_off -rule PROCASSINIT\r\nlint_off -rule WIDTH\r\n",
                "src/test.py": "# lint_off -rule PROCASSINIT\n",
            }
        },
    }


def test_category_prompt_preserves_context_and_references_stay_out_of_model_input(raw_task):
    original = deepcopy(raw_task)
    row = benchmark_prepare.convert_task(raw_task)
    system, user = row["responses_create_params"]["input"]
    assert system["role"] == "system"
    assert "Specification to RTL Translation" in system["content"]
    assert "<answer>" not in json.dumps(row)
    assert _parse_model_response("module latch; endmodule", ["rtl/latch.sv"]) == {
        "rtl/latch.sv": "module latch; endmodule"
    }
    assert (
        "Consider the following content for the file rtl/latch.sv:\n```\nmodule latch; endmodule\n```"
        in user["content"]
    )
    assert user["content"].endswith("Your response will be saved directly to: rtl/latch.sv.\n")
    assert "secret-reference" not in json.dumps(row)
    # Target-path context is required by the simulator; empty files can be omitted.
    assert row["verifier_metadata"]["context_files"] == {"rtl/latch.sv": "module latch; endmodule"}
    assert row["verifier_metadata"]["harness_files"] == {
        "src/lint.vlt": "keep\r\nlint_off -rule WIDTH\r\n",
        "src/test.py": "# lint_off -rule PROCASSINIT\n",
    }
    assert raw_task == original


def test_multifile_instruction_preserves_target_order_and_omits_empty_context(raw_task):
    raw_task["input"]["context"] = {}
    raw_task["output"]["context"] = {"rtl/z.sv": "reference", "rtl/a.sv": "reference"}
    row = benchmark_prepare.convert_task(raw_task)
    assert row["responses_create_params"]["input"][1]["content"].endswith(
        "Generate the following files: rtl/z.sv, rtl/a.sv.\n"
    )
    assert "JSON schemas" in row["responses_create_params"]["input"][0]["content"]
    answer = {"code": [{"rtl/z.sv": "module z; endmodule"}, {"rtl/a.sv": "module a; endmodule"}]}
    assert _parse_model_response(json.dumps(answer), row["verifier_metadata"]["target_files"]) == {
        "rtl/z.sv": "module z; endmodule",
        "rtl/a.sv": "module a; endmodule",
    }
    assert row["verifier_metadata"]["target_files"] == ["rtl/z.sv", "rtl/a.sv"]
    assert "context_files" not in row["verifier_metadata"]


@pytest.mark.parametrize("category", ["cid006", "cid012", "cid999"])
def test_unexpected_categories_are_rejected_instead_of_silently_changing_the_split(raw_task, category):
    raw_task["categories"] = [category]
    with pytest.raises(ValueError, match="unsupported"):
        benchmark_prepare.convert_task(raw_task)


@pytest.fixture
def preparation_input(tmp_path, monkeypatch, raw_task):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(raw_task) + "\n")
    monkeypatch.setattr(benchmark_prepare, "RAW_SHA256", hashlib.sha256(raw.read_bytes()).hexdigest())
    monkeypatch.setattr(benchmark_prepare, "TASK_COUNT", 1)
    monkeypatch.setattr(benchmark_prepare, "convert_task", lambda task: {"task": task["id"]})
    expected = b'{"task": "public-task"}\n'
    monkeypatch.setattr(benchmark_prepare, "OUTPUT_SHA256", hashlib.sha256(expected).hexdigest())
    return raw, tmp_path / "output", expected


def test_public_download_is_pinned_and_needs_no_external_converter(preparation_input):
    raw, directory, expected = preparation_input
    with patch("huggingface_hub.hf_hub_download", return_value=str(raw)) as download:
        result = benchmark_prepare.prepare(output_dir=directory)
    download.assert_called_once_with(
        "nvidia/cvdp-benchmark-dataset",
        benchmark_prepare.RAW_FILE,
        repo_type="dataset",
        revision=benchmark_prepare.DATASET_REVISION,
    )
    assert result.read_bytes() == expected
    assert sorted(path.name for path in directory.iterdir()) == ["cvdp_v110.jsonl"]


def test_offline_preparation_does_not_contact_hugging_face(preparation_input):
    raw, directory, expected = preparation_input
    with patch("huggingface_hub.hf_hub_download") as download:
        result = benchmark_prepare.prepare(raw_path=raw, output_dir=directory)
    download.assert_not_called()
    assert result.read_bytes() == expected


@pytest.mark.parametrize("mismatch", ["input", "output", "count"])
def test_checksum_or_coverage_failure_preserves_previous_dataset(preparation_input, monkeypatch, mismatch):
    raw, directory, _ = preparation_input
    directory.mkdir()
    existing = directory / "cvdp_v110.jsonl"
    existing.write_bytes(b"previous verified dataset")
    if mismatch == "input":
        raw.write_bytes(b"changed raw input")
    elif mismatch == "output":
        monkeypatch.setattr(benchmark_prepare, "OUTPUT_SHA256", "0" * 64)
    else:
        monkeypatch.setattr(benchmark_prepare, "TASK_COUNT", 2)
    with pytest.raises(ValueError):
        benchmark_prepare.prepare(raw_path=raw, output_dir=directory)
    assert existing.read_bytes() == b"previous verified dataset"
    assert list(directory.iterdir()) == [existing]


def test_cached_data_requires_the_pinned_output_checksum(tmp_path, monkeypatch):
    output = tmp_path / "cached.jsonl"
    assert not benchmark_prepare.is_prepared_data_current(output)
    output.write_bytes(b"known dataset")
    monkeypatch.setattr(benchmark_prepare, "OUTPUT_SHA256", hashlib.sha256(output.read_bytes()).hexdigest())
    assert benchmark_prepare.is_prepared_data_current(output)
    output.write_bytes(b"changed dataset")
    assert not benchmark_prepare.is_prepared_data_current(output)


def test_prepared_ids_match_source_and_verifier_ids(raw_task):
    rows = [benchmark_prepare.convert_task(raw_task | {"id": name}) for name in ["b", "a"]]
    assert [row["task_id"] for row in rows] == ["b", "a"]
    assert all(row["task_id"] == row["verifier_metadata"]["task_id"] for row in rows)


def test_recipe_manifest_retains_license_and_metadata():
    from pathlib import Path

    from nemo_gym.environment import load_manifest

    manifest = load_manifest(Path(benchmark_prepare.__file__).with_name("manifest.yaml"))
    assert manifest.licensing == "LicenseRef-CVDP-Upstream"
    assert manifest.version == "0.1.0" and manifest.experimental


@pytest.mark.parametrize("category", ["cid002", "cid003", "cid004", "cid007", "cid016"])
@pytest.mark.parametrize("multifile", [False, True])
def test_generated_rows_follow_owner_and_verifier_contracts(raw_task, category, multifile):
    from nemo_gym.data import normalize_task_fields
    from resources_servers.cvdp.app import CVDPVerifierMetadata
    from resources_servers.cvdp.task_data import TaskData

    raw_task["categories"] = [category]
    if multifile:
        raw_task["output"]["context"]["rtl/second.sv"] = "reference"
    row = benchmark_prepare.convert_task(raw_task)
    fields, conflicts = normalize_task_fields(row)
    assert not conflicts
    owner = TaskData.model_validate(fields)
    runtime = CVDPVerifierMetadata.model_validate(row["verifier_metadata"])
    assert owner.model_dump() == runtime.model_dump()
    assert owner.task_id == row["task_id"]
    assert owner.categories == raw_task["categories"]
    assert owner.difficulty == raw_task["difficulty"]
    assert owner.target_files == list(raw_task["output"]["context"])

    # Both declarations must evolve together, including absent optional fields.
    assert TaskData.model_fields.keys() == CVDPVerifierMetadata.model_fields.keys()
    for name, field in TaskData.model_fields.items():
        other = CVDPVerifierMetadata.model_fields[name]
        assert field.annotation == other.annotation, name
        assert field.is_required() == other.is_required(), name
        if not field.is_required():
            assert field.get_default(call_default_factory=True) == other.get_default(call_default_factory=True), name
