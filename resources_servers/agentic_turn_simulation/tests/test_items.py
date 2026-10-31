# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers over Responses items."""

import pytest

from resources_servers.agentic_turn_simulation.items import parse_args


class TestParseArgs:
    @pytest.mark.parametrize(
        "arguments, expected",
        [
            ('{"a": 1}', {"a": 1}),
            ("", {}),
            (None, {}),
            ("{bad", None),
            ("[1, 2]", None),
            ("3", None),
            ('{"offset": ' + "1" * 5000 + "}", None),  # past Python's integer-string limit: ValueError
            ('{"command": ' + "[" * 100_000, None),  # RecursionError while decoding
        ],
        ids=["object", "empty", "none", "broken", "array", "number", "huge-integer", "deep-nesting"],
    )
    def test_parse_args(self, arguments, expected) -> None:
        assert parse_args({"arguments": arguments}) == expected
