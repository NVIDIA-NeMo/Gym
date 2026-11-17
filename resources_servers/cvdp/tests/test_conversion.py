# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import sys
from unittest.mock import patch

import pytest

from resources_servers.cvdp.scripts import convert_to_gym as conversion
from resources_servers.cvdp.scripts.convert_to_gym import main


@pytest.mark.parametrize("source_format", ["published", "processed"])
def test_conversion_preserves_repair_inputs_without_reference_answers(tmp_path, monkeypatch, source_format) -> None:
    starting = {"rtl/top.sv": "module top; // buggy starting code\nendmodule", "docs/spec.md": "Repair top."}
    harness = {"docker-compose.yml": "services: {}"}
    reference = {"rtl/top.sv": "secret reference answer"}
    source = {"id": "repair", "categories": ["cid001"]}
    if source_format == "published":
        source.update(context=starting, patch=reference, harness=harness)
    else:
        source.update(input={"context": starting}, output={"context": reference}, harness={"files": harness})
    dataset = tmp_path / "source.jsonl"
    dataset.write_text(json.dumps(source) + "\n")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "repair", "system": "Repair RTL", "user": "Fix top"}) + "\n")
    output = tmp_path / "gym.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["convert_to_gym", "--dataset", str(dataset), "--prompts", str(prompts), "--output", str(output)]
    )

    main()

    row = json.loads(output.read_text())
    assert row["verifier_metadata"]["context_files"] == starting
    assert row["verifier_metadata"]["target_files"] == ["rtl/top.sv"]
    assert row["verifier_metadata"]["harness_files"] == harness
    assert "secret reference answer" not in output.read_text()


@pytest.fixture
def public_task():
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


def test_public_single_file_prompt_preserves_starting_rtl_and_excludes_reference(public_task):
    row = conversion.convert_public_task(public_task)
    system, user = row["responses_create_params"]["input"]
    assert "Specification to RTL Translation" in system["content"]
    assert "module latch; endmodule" in user["content"]
    assert user["content"].endswith("Your response will be saved directly to: rtl/latch.sv.\n")
    assert "secret-reference" not in json.dumps(row)
    assert row["task_id"] == row["verifier_metadata"]["task_id"] == "public-task"
    assert row["verifier_metadata"]["context_files"] == {"rtl/latch.sv": "module latch; endmodule"}
    assert row["verifier_metadata"]["harness_files"] == {
        "src/lint.vlt": "keep\r\nlint_off -rule WIDTH\r\n",
        "src/test.py": "# lint_off -rule PROCASSINIT\n",
    }


def test_public_multifile_prompt_requests_json_in_target_order(public_task):
    public_task["input"]["context"] = {}
    public_task["output"]["context"] = {"rtl/z.sv": "reference", "rtl/a.sv": "reference"}
    row = conversion.convert_public_task(public_task)
    system, user = row["responses_create_params"]["input"]
    assert "JSON schemas" in system["content"]
    assert user["content"].endswith("Generate the following files: rtl/z.sv, rtl/a.sv.\n")
    assert row["verifier_metadata"]["target_files"] == ["rtl/z.sv", "rtl/a.sv"]
    assert "context_files" not in row["verifier_metadata"]


@pytest.mark.parametrize("category", ["cid006", "cid012", "cid999"])
def test_public_preparation_rejects_unsupported_subsets(public_task, category):
    public_task["categories"] = [category]
    with pytest.raises(ValueError, match="unsupported"):
        conversion.convert_public_task(public_task)


@pytest.fixture
def public_input(tmp_path, public_task):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(public_task) + "\n")
    expected = (json.dumps(conversion.convert_public_task(public_task)) + "\n").encode()
    options = {
        "output": tmp_path / "out" / "gym.jsonl",
        "expected_raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "expected_output_sha256": hashlib.sha256(expected).hexdigest(),
        "expected_count": 1,
    }
    return raw, expected, options


def test_public_download_uses_pinned_defaults(public_input):
    raw, expected, options = public_input
    with patch("huggingface_hub.hf_hub_download", return_value=str(raw)) as download:
        output = conversion.prepare_public_dataset(**options)
    download.assert_called_once_with(
        "nvidia/cvdp-benchmark-dataset",
        conversion.RAW_FILE,
        repo_type="dataset",
        revision=conversion.DATASET_REVISION,
    )
    assert output.read_bytes() == expected
    assert list(output.parent.iterdir()) == [output]


def test_public_cli_supports_offline_input_and_integrity_overrides(public_input, monkeypatch):
    raw, expected, options = public_input
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_to_gym",
            "--from-hf",
            "--dataset",
            str(raw),
            "--output",
            str(options["output"]),
            "--expected-sha256",
            options["expected_raw_sha256"],
            "--expected-output-sha256",
            options["expected_output_sha256"],
            "--expected-count",
            "1",
        ],
    )
    with patch("huggingface_hub.hf_hub_download") as download:
        main()
    download.assert_not_called()
    assert options["output"].read_bytes() == expected


def test_public_cli_allows_a_reviewed_new_revision_and_file(public_input, monkeypatch):
    raw, expected, options = public_input
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_to_gym",
            "--from-hf",
            "--output",
            str(options["output"]),
            "--hf-revision",
            "reviewed-commit",
            "--hf-file",
            "compatible.jsonl",
            "--expected-sha256",
            options["expected_raw_sha256"],
            "--expected-output-sha256",
            options["expected_output_sha256"],
            "--expected-count",
            "1",
        ],
    )
    with patch("huggingface_hub.hf_hub_download", return_value=str(raw)) as download:
        main()
    download.assert_called_once_with(
        "nvidia/cvdp-benchmark-dataset", "compatible.jsonl", repo_type="dataset", revision="reviewed-commit"
    )
    assert options["output"].read_bytes() == expected


@pytest.mark.parametrize("mismatch", ["input", "output", "count"])
def test_public_integrity_failure_preserves_previous_output(public_input, mismatch):
    raw, _, options = public_input
    output = options["output"]
    output.parent.mkdir()
    output.write_bytes(b"previous verified dataset")
    if mismatch == "input":
        raw.write_bytes(b"changed input")
    elif mismatch == "output":
        options["expected_output_sha256"] = "0" * 64
    else:
        options["expected_count"] = 2
    with pytest.raises(ValueError, match="expected SHA256|Expected 2"):
        conversion.prepare_public_dataset(raw_path=raw, **options)
    assert output.read_bytes() == b"previous verified dataset"
    assert list(output.parent.iterdir()) == [output]


@pytest.mark.parametrize(
    "arguments", [[], ["--prompts", "prompts.jsonl"], ["--from-hf", "--prompts", "prompts.jsonl"]]
)
def test_cli_rejects_missing_or_conflicting_inputs(tmp_path, monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["convert_to_gym", "--output", str(tmp_path / "out.jsonl"), *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
