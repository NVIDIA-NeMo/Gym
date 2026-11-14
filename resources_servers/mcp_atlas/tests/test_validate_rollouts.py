# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from resources_servers.mcp_atlas.credentials import SERVICE_ENV_NAMES
from resources_servers.mcp_atlas.validate_rollouts import main, scan_rollout_credentials


def test_scanner_checks_decoded_tool_and_model_strings_without_returning_values(tmp_path) -> None:
    path = tmp_path / "rollouts.jsonl"
    sentinel = "fake credential sentinel"
    escaped = "".join(f"\\u{ord(character):04x}" for character in sentinel)
    path.write_text('{"response":{"output":[{"text":"' + escaped + '"}]}}\n')
    found = scan_rollout_credentials([path], {"NOTION_TOKEN": sentinel, "EXA_API_KEY": "other fake value"})
    assert len(found) == 1
    assert found[0].path == path
    assert found[0].line == 1
    assert found[0].credential_names == ("NOTION_TOKEN",)
    assert sentinel not in repr(found)


def test_scanner_accepts_clean_rows_and_does_not_treat_empty_values_as_leaks(tmp_path) -> None:
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps({"response": {"output": "ordinary task answer"}}) + "\n")
    assert scan_rollout_credentials([path], {"NOTION_TOKEN": "", "EXA_API_KEY": "  "}) == []


def test_scanner_cli_rejects_leaks_without_printing_sentinel(monkeypatch, tmp_path, capsys) -> None:
    for name in SERVICE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    sentinel = "fake private sentinel for test"
    monkeypatch.setenv("NOTION_TOKEN", sentinel)
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps({"tool_output": sentinel}) + "\n")
    assert main([str(path)]) == 1
    output = capsys.readouterr()
    assert "NOTION_TOKEN" in output.out and f"{path}:1" in output.out
    assert sentinel not in output.out + output.err


def test_scanner_does_not_claim_to_have_scanned_a_missing_artifact(monkeypatch, tmp_path, capsys) -> None:
    for name in SERVICE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path / "absent.jsonl")])
    assert error.value.code == 2
    assert "checked_files" not in capsys.readouterr().out
