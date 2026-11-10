# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
from hydra.core.override_parser.overrides_parser import OverridesParser

from nemo_gym.utils.text import hydra_literal, hydra_quote


def _parsed(text: str):
    return OverridesParser.create().parse_overrides([f"+setting={text}"])[0].value()


@pytest.mark.parametrize(
    "value",
    [
        "1",
        "1.0",
        "true",
        "null",
        "nightly(run)",
        "o'brien",
        "nightly run",
        " padded ",
        "",
        "a,b",
        "x]y",
        "C:\\\\path\\\\",
        "caf\u00e9",
        "${oc.env:SOME_URL}",
        "plain",
        3,
        2.5,
        True,
        False,
        None,
        ["1", "true", "a b", "x,y", "o'brien"],
        [1, 2.5, True, None, "x"],
        [],
    ],
)
def test_hydra_reads_a_literal_back_as_the_value_it_was_made_from(value) -> None:
    parsed = _parsed(hydra_literal(value))

    assert parsed == value
    assert type(parsed) is type(value)


def test_a_string_that_hydra_would_read_as_something_else_is_quoted_and_a_plain_one_is_not() -> None:
    assert hydra_literal("1") == hydra_quote("1")
    assert hydra_literal("nightly(run)") == hydra_quote("nightly(run)")
    assert hydra_literal("plain") == "plain"
    assert hydra_literal("${oc.env:SOME_URL}") == "${oc.env:SOME_URL}"


def test_numbers_booleans_and_null_are_written_in_their_own_syntax() -> None:
    assert [hydra_literal(v) for v in (3, 2.5, True, False, None)] == ["3", "2.5", "true", "false", "null"]


@pytest.mark.parametrize("value", [{"null": 1}, [{"1": "x"}], object()])
def test_a_value_hydra_cannot_carry_is_refused(value) -> None:
    with pytest.raises(ValueError, match="cannot be written as a Hydra override value"):
        hydra_literal(value)
