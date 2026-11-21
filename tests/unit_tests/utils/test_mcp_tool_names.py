# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from nemo_gym.utils.mcp_tool_names import (
    HERMES_MCP_SERVERS_METADATA_KEY,
    hermes_mcp_metadata,
    normalize_hermes_mcp_tool_name,
)


def test_hermes_metadata_encodes_complete_grants_without_mutating_them():
    names = ["store", "store_b", "my-store.v1"]
    metadata = hermes_mcp_metadata(names)
    assert json.loads(metadata[HERMES_MCP_SERVERS_METADATA_KEY]) == names
    assert names == ["store", "store_b", "my-store.v1"]
    assert hermes_mcp_metadata([]) == {}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mcp_store_append", "mcp__store__append"),
        ("mcp_store_b_append", "mcp__store_b__append"),
        ("mcp_my_store_v1_append__item", "mcp__my-store.v1__append__item"),
        ("mcp_unknown_append", "mcp_unknown_append"),
        ("mcp__store__append", "mcp__store__append"),
        ("append", "append"),
    ],
)
def test_normalize_hermes_names_uses_all_grants_and_preserves_suffix(name, expected):
    assert normalize_hermes_mcp_tool_name(name, hermes_mcp_metadata(["store", "store_b", "my-store.v1"])) == expected


@pytest.mark.parametrize("metadata", [None, {}, {"unrelated": '["store"]'}])
def test_names_without_explicit_hermes_provenance_are_unchanged(metadata):
    assert normalize_hermes_mcp_tool_name("mcp_store_append", metadata) == "mcp_store_append"


@pytest.mark.parametrize("names", [("store",), "store", [""], [1], ["store", "store"], ["a-b", "a.b"]])
def test_metadata_rejects_invalid_or_ambiguous_grants(names):
    with pytest.raises(ValueError):
        hermes_mcp_metadata(names)


@pytest.mark.parametrize("encoded", ["not-json", "{}", '"store"', '["store","store"]', '["a-b","a.b"]', 1])
def test_normalization_rejects_malformed_metadata(encoded):
    with pytest.raises(ValueError):
        normalize_hermes_mcp_tool_name("mcp_store_append", {HERMES_MCP_SERVERS_METADATA_KEY: encoded})


def test_metadata_limit_applies_to_encoded_bytes_at_both_boundaries():
    names = ["a" * 508]
    assert len(hermes_mcp_metadata(names)[HERMES_MCP_SERVERS_METADATA_KEY].encode()) == 512
    for names in [["a" * 509], ["\u00e9" * 100]]:
        with pytest.raises(ValueError, match="512"):
            hermes_mcp_metadata(names)
    with pytest.raises(ValueError, match="512"):
        normalize_hermes_mcp_tool_name("name", {HERMES_MCP_SERVERS_METADATA_KEY: " " * 513})
