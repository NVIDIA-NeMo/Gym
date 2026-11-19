# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the `sys.stdin` mock used when grading submitted code."""

from __future__ import annotations

from lcb_integration.testing_util import MockStdinWithBuffer


INPUTS = "2\n10 20 30\n40 50 60\n"


class TestMockBufferIsStateful:
    """`sys.stdin.buffer` must behave like a real byte stream, not replay its first line."""

    def test_readline_advances(self):
        buffer = MockStdinWithBuffer(INPUTS).buffer

        assert buffer.readline() == b"2\n"
        assert buffer.readline() == b"10 20 30\n"
        assert buffer.readline() == b"40 50 60\n"
        assert buffer.readline() == b""

    def test_readlines_returns_all_lines(self):
        buffer = MockStdinWithBuffer(INPUTS).buffer

        assert buffer.readlines() == [b"2\n", b"10 20 30\n", b"40 50 60\n"]

    def test_iteration_yields_each_line_once(self):
        buffer = MockStdinWithBuffer(INPUTS).buffer

        assert list(buffer) == [b"2\n", b"10 20 30\n", b"40 50 60\n"]

    def test_read_returns_whole_payload(self):
        buffer = MockStdinWithBuffer(INPUTS).buffer

        assert buffer.read() == INPUTS.encode("utf-8")

    def test_read_after_readline_returns_remainder(self):
        buffer = MockStdinWithBuffer(INPUTS).buffer

        assert buffer.readline() == b"2\n"
        assert buffer.read() == b"10 20 30\n40 50 60\n"

    def test_seek_rewinds(self):
        buffer = MockStdinWithBuffer(INPUTS).buffer

        assert buffer.readline() == b"2\n"
        buffer.seek(0)
        assert buffer.readline() == b"2\n"
