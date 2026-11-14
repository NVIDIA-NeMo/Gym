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
    with pytest.raises(ValueError, match="missing tools \\['mongodb_find'\\] of supported servers \\['mongodb'\\]"):
        build_row(record, CATALOG, SERVERS + ["mongodb"])


def test_a_partial_server_catalog_cannot_drop_requested_tools() -> None:
    record = {
        "TASK": "t3",
        "PROMPT": "Look up both the domain and top-level domain.",
        "ENABLED_TOOLS": json.dumps(["whois_whois_domain", "whois_whois_tld"]),
        "GTFA_CLAIMS": "['The registry is identified.']",
    }
    with pytest.raises(ValueError, match="task t3: .*missing tools \\['whois_whois_tld'\\]"):
        build_row(record, CATALOG, SERVERS)

    complete = {
        **CATALOG,
        "whois_whois_tld": {
            "name": "whois_whois_tld",
            "server": "whois",
            "inputSchema": {"type": "object", "properties": {"tld": {"type": "string"}}},
        },
    }
    row = build_row(record, complete, SERVERS)
    tools = row["responses_create_params"]["tools"]
    assert [tool["name"] for tool in tools] == ["whois_whois_domain", "whois_whois_tld"]
    assert tools[1]["parameters"] == complete["whois_whois_tld"]["inputSchema"]
    assert row["verifier_metadata"]["dropped_tools"] == []


def test_the_pinned_task_keeps_its_row_and_audits_the_unavailable_clinical_tool() -> None:
    task_id = "6887c6d4614f06dc73af2411"
    unavailable = "clinicaltrialsgov-mcp-server_clinicaltrials_search_studies"
    available = "clinicaltrialsgov-mcp-server_clinicaltrials_get_study"
    record = {
        "TASK": task_id,
        "PROMPT": "Identify the clinical study.",
        "ENABLED_TOOLS": json.dumps([available, unavailable]),
        "GTFA_CLAIMS": '["The study is identified."]',
    }
    catalog = {available: {"name": available, "server": "clinicaltrialsgov-mcp-server"}}
    row = build_row(record, catalog, ["clinicaltrialsgov-mcp-server"])
    assert row["task_id"] == task_id
    assert row["responses_create_params"]["input"] == [{"role": "user", "content": record["PROMPT"]}]
    assert [tool["name"] for tool in row["responses_create_params"]["tools"]] == [available]
    assert row["verifier_metadata"] == {
        "task_id": task_id,
        "enabled_tools": [available],
        "servers": ["clinicaltrialsgov-mcp-server"],
        "claims": ["The study is identified."],
        "dropped_tools": [unavailable],
    }


@pytest.mark.parametrize(
    ("task_id", "missing_tool"),
    [
        ("6887c6d4614f06dc73af2412", "clinicaltrialsgov-mcp-server_clinicaltrials_search_studies"),
        ("6887c6d4614f06dc73af2411", "clinicaltrialsgov-mcp-server_clinicaltrials_search_study"),
        ("6887c6d4614f06dc73af2411", "clinicaltrialsgov-mcp-server_clinicaltrials_get_study"),
        ("6887c6d4614f06dc73af2411", "slack_channels_list"),
    ],
)
def test_the_unavailable_clinical_tool_exception_does_not_hide_other_missing_exports(task_id, missing_tool) -> None:
    unavailable = "clinicaltrialsgov-mcp-server_clinicaltrials_search_studies"
    record = {
        "TASK": task_id,
        "PROMPT": "Identify the clinical study.",
        "ENABLED_TOOLS": json.dumps(sorted({unavailable, missing_tool})),
        "GTFA_CLAIMS": '["The study is identified."]',
    }
    with pytest.raises(ValueError, match=f"missing tools.*{missing_tool}"):
        build_row(record, {}, ["clinicaltrialsgov-mcp-server", "slack"])


def test_catalog_names_carry_their_server_prefix() -> None:
    catalog, servers = load_catalog()
    assert len(servers) == 36
    assert all(tool["server"] in servers and name.startswith(f"{tool['server']}_") for name, tool in catalog.items())
