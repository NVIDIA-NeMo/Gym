# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from benchmarks.mcp_atlas.prepare import build_row, enabled_tool_names, extract_claims, load_catalog


SERVERS = ["calculator", "whois"]
CATALOG = {
    "calculator_calculate": {"name": "calculator_calculate", "server": "calculator", "description": "Calc"},
    "whois_whois_domain": {
        "name": "whois_whois_domain",
        "server": "whois",
        "description": "Whois",
        "inputSchema": {"type": "object", "properties": {"domain": {"type": "string"}}},
    },
}


def test_claims_parse_python_and_json_lists_and_clean_them() -> None:
    assert extract_claims("['The repo was created in 2013.', \"It’s 7 years.\"]") == [
        "The repo was created in 2013.",
        "It's 7 years.",
    ]
    assert extract_claims('["1. First claim here", "- second claim"]') == ["First claim here", "second claim"]
    # Neither JSON nor Python: upstream falls back to separator splitting.
    assert extract_claims("[claim one is here; claim two is here]") == ["[claim one is here", "claim two is here]"]
    assert extract_claims("") == []


def test_enabled_tools_accept_names_and_objects() -> None:
    blob = json.dumps(["calculator_calculate", {"name": "whois_whois_domain", "requiredParams": []}])
    assert enabled_tool_names(blob) == ["calculator_calculate", "whois_whois_domain"]


def test_row_offers_only_served_tools_and_keeps_claims() -> None:
    record = {
        "TASK": "689f",
        "PROMPT": "How old is the domain?",
        "ENABLED_TOOLS": json.dumps(["whois_whois_domain", "anili_search", "calculator_calculate"]),
        "GTFA_CLAIMS": "['The domain was registered in 2006.']",
    }
    row = build_row(record, CATALOG, SERVERS)
    assert row["task_id"] == "689f"
    assert row["responses_create_params"]["input"] == [{"role": "user", "content": "How old is the domain?"}]
    tools = row["responses_create_params"]["tools"]
    # Catalog order, the order the sandbox lists them.
    assert [tool["name"] for tool in tools] == ["calculator_calculate", "whois_whois_domain"]
    assert tools[0]["parameters"] == {"type": "object", "properties": {}}
    assert tools[1]["parameters"]["properties"] == {"domain": {"type": "string"}}
    meta = row["verifier_metadata"]
    assert meta["servers"] == ["calculator", "whois"]
    assert meta["dropped_tools"] == ["anili_search"]
    assert meta["claims"] == ["The domain was registered in 2006."]


def test_a_server_missing_from_the_catalog_fails_the_task() -> None:
    record = {
        "TASK": "t2",
        "PROMPT": "p",
        "ENABLED_TOOLS": json.dumps(["calculator_calculate", "mongodb_find"]),
        "GTFA_CLAIMS": "['c is a claim']",
    }
    with pytest.raises(ValueError, match="no tools of \\['mongodb'\\]"):
        build_row(record, CATALOG, SERVERS + ["mongodb"])


def test_catalog_names_carry_their_server_prefix() -> None:
    catalog, servers = load_catalog()
    assert len(servers) == 36
    assert all(tool["server"] in servers and name.startswith(f"{tool['server']}_") for name, tool in catalog.items())
