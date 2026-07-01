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

from rich.console import Console
from rich.table import Table

from nemo_gym.cli.utils import print_rich_table


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
