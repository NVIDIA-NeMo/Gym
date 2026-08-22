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
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from rich.console import Console
from rich.table import Table

from nemo_gym.cli.utils import (
    exit_unknown_component,
    fuzzy_matches,
    print_rich_table,
    render_component_inspection,
)


# A cell value wider than Rich's 80-col non-TTY default, so a truncated render would ellipsize it.
_LONG_NAME = "aalcr_benchmark_" + "x" * 120


def _long_table() -> Table:
    table = Table(title="t")
    table.add_column("Benchmark name")
    table.add_row(_LONG_NAME)
    return table


class TestPrintRichTable:
    def test_not_truncated_when_piped(self, capsys) -> None:
        with patch.object(Console, "is_terminal", new_callable=PropertyMock, return_value=False):
            print_rich_table(_long_table())
        out = capsys.readouterr().out
        assert _LONG_NAME in out
        assert "…" not in out

    def test_uses_default_console_on_tty(self) -> None:
        fake_console = MagicMock()
        fake_console.is_terminal = True
        # `Console` is imported inside the function from `rich.console`, so patch it there.
        with patch("rich.console.Console", return_value=fake_console) as console_cls:
            table = _long_table()
            print_rich_table(table)

        # On a real terminal we keep the single auto-sized console (no width override) and just print.
        console_cls.assert_called_once_with()
        fake_console.print.assert_called_once_with(table)

    def test_not_truncated_regardless_of_ambient_width(self, capsys, monkeypatch) -> None:
        # Rich derives a non-TTY console's width from COLUMNS (falling back to 80). Whatever that
        # ambient width is, the table must render at its natural width, never truncated to fit.
        monkeypatch.setenv("COLUMNS", "10")
        print_rich_table(_long_table())
        out = capsys.readouterr().out
        assert _LONG_NAME in out
        assert "…" not in out


class TestFuzzyMatches:
    def test_substring_matches(self) -> None:
        assert fuzzy_matches("math", "math_with_judge")

    def test_token_typo_matches(self) -> None:
        # `aimee` is a near-miss for the `aime` token in `aime24`.
        assert fuzzy_matches("aimee", "aime24")

    def test_matches_any_field(self) -> None:
        assert fuzzy_matches("judge", "aime24", "math_with_judge_agent")

    def test_skips_empty_fields(self) -> None:
        assert not fuzzy_matches("math", "", None)

    def test_no_match(self) -> None:
        assert not fuzzy_matches("zzznomatch", "aime24", "math_with_judge")


class TestExitUnknownComponent:
    def test_exits_nonzero_with_did_you_mean(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            exit_unknown_component("calndr", ["calendar", "arc_agi"], "environment")
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "Unknown environment 'calndr'" in out and "calendar" in out


class TestRenderComponentInspection:
    def test_full_text_view(self, capsys) -> None:
        render_component_inspection(
            json_output=False,
            name="calendar",
            type_noun="environment",
            domain="agent",
            description="A calendar env.\nValue: Improve scheduling",  # value folded in by the caller
            details={"config": "/abs/config.yaml", "agent": "simple_agent"},
            usage="gym env start --environment calendar --model-type vllm_model",
        )
        out = capsys.readouterr().out
        assert "The calendar environment (domain: agent)" in out
        assert "A calendar env." in out and "Value: Improve scheduling" in out
        assert "Details:\nconfig: /abs/config.yaml\nagent: simple_agent" in out
        assert "Usage example:\ngym env start --environment calendar --model-type vllm_model" in out

    def test_omits_empty_sections(self, capsys) -> None:
        # A thin view (model): no domain suffix, no description block, no usage.
        render_component_inspection(
            json_output=False, name="vllm_model", type_noun="model", details={"path": "/abs/vllm_model"}
        )
        out = capsys.readouterr().out
        assert "The vllm_model model" in out and "path: /abs/vllm_model" in out
        assert "(domain" not in out and "Usage example:" not in out
