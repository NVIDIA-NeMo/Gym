# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the resumable Slurm launcher."""

import unittest
from unittest.mock import patch

from launch import accounting_states, slurm_options


class SlurmOptionTests(unittest.TestCase):
    def test_options_preserve_supported_manifest_order(self):
        self.assertEqual(
            slurm_options(
                {
                    "partition": "batch",
                    "account": "research",
                    "time": "02:00:00",
                    "constraint": "H100",
                }
            ),
            [
                "--account=research",
                "--partition=batch",
                "--constraint=H100",
                "--time=02:00:00",
            ],
        )


class AccountingTests(unittest.TestCase):
    @patch("launch.subprocess.check_output", return_value="101|FAILED+\n102|OUT_OF_MEMORY  \n")
    def test_normalizes_terminal_state_suffix(self, check_output):
        states = accounting_states({"a": {"job_id": "101"}, "b": {"job_id": "102"}})
        self.assertEqual(states, {"101": "FAILED", "102": "OUT_OF_MEMORY"})
        check_output.assert_called_once()

    @patch("launch.subprocess.check_output")
    def test_empty_ledger_does_not_call_slurm(self, check_output):
        self.assertEqual(accounting_states({}), {})
        check_output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
