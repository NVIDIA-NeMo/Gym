# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from benchmarks.partner_if import prepare as prepare_module


ROWS = [
    {"responses_create_params": {"input": "Say hello."}, "verifier_metadata": {"partner_if": {"task_id": "a"}}},
    {
        "responses_create_params": {"input": "Edit the file."},
        "verifier_metadata": {"partner_if": {"task_id": "b"}},
        "tool_backend": {"entrypoint": "tools:Backend", "files": {"tools.py": ""}},
    },
]
CONTENT = "".join(json.dumps(row) + "\n" for row in ROWS)


def download_writing(content):
    def download(config):
        Path(config.output_fpath).write_text(content)

    return download


@pytest.fixture
def prepared(tmp_path):
    with (
        patch.object(prepare_module, "DATA_DIR", tmp_path),
        patch.object(prepare_module, "OUTPUT_FPATH", tmp_path / "partner_if_benchmark.jsonl"),
        patch.object(prepare_module, "TOOLS_FPATH", tmp_path / "partner_if_tools.jsonl"),
        patch.object(prepare_module, "SHA256", hashlib.sha256(CONTENT.encode()).hexdigest()),
        patch.object(prepare_module, "ROW_COUNT", len(ROWS)),
    ):
        yield tmp_path


def test_prepare_writes_the_pinned_rows_and_their_tool_subset(prepared):
    with patch.object(prepare_module, "download_jsonl_dataset", download_writing(CONTENT)):
        result = prepare_module.prepare()
    assert result.read_text() == CONTENT
    assert (prepared / "partner_if_tools.jsonl").read_text() == json.dumps(ROWS[1]) + "\n"
    assert prepare_module.is_prepared_data_current(result)


@pytest.mark.parametrize(
    "content,error",
    [
        # An error body from the registry, or any other content, fails the digest check.
        ('{"message": "401 Unauthorized"}', "sha256"),
        (CONTENT + json.dumps(ROWS[0]) + "\n", "sha256"),
    ],
)
def test_prepare_refuses_content_other_than_the_pinned_release(prepared, content, error):
    (prepared / "partner_if_benchmark.jsonl").write_text("previous")
    with patch.object(prepare_module, "download_jsonl_dataset", download_writing(content)):
        with pytest.raises(ValueError, match=error):
            prepare_module.prepare()
    # The previous output survives a failed download.
    assert (prepared / "partner_if_benchmark.jsonl").read_text() == "previous"
    assert not prepare_module.is_prepared_data_current(prepared / "partner_if_benchmark.jsonl")


def test_routes_cover_the_release_and_name_its_environments():
    config = yaml.safe_load((prepare_module.BENCHMARK_DIR / "config.yaml").read_text())
    routes = config["partner_if_dispatcher"]["responses_api_agents"]["partner_if_dispatcher"]["task_routes"]
    assert list(routes) == [f"pif_{index:03d}" for index in range(1, prepare_module.ROW_COUNT + 1)]
    environments = {
        name for name, server in config.items() if isinstance(server, dict) and "environment_servers" in server
    }
    assert {route["name"] for route in routes.values()} <= environments
    # The tool environments' server loads the rows that prepare.py splits off for it.
    tools = config["partner_if_tools"]["resources_servers"]["partner_if"]
    assert tools["tasks_path"] == f"benchmarks/partner_if/data/{prepare_module.TOOLS_FPATH.name}"


def test_rows_split_on_newlines_only():
    # A raw U+0085 (next line) inside a row is not a row boundary, though str.splitlines() treats it as one.
    text = json.dumps({"text": "a\x85b"}, ensure_ascii=False) + "\n" + json.dumps({"text": "c"}) + "\n"
    assert [json.loads(line)["text"] for line in prepare_module.jsonl_lines(text)] == ["a\x85b", "c"]


def test_prepare_checks_the_row_count(prepared):
    with (
        patch.object(prepare_module, "ROW_COUNT", len(ROWS) + 1),
        patch.object(prepare_module, "download_jsonl_dataset", download_writing(CONTENT)),
    ):
        with pytest.raises(ValueError, match="rows"):
            prepare_module.prepare()
