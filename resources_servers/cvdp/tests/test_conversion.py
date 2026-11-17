# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys

import pytest

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
