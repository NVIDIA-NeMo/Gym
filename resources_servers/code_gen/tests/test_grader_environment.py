# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the environment submitted code is graded in: stdout capture and interpreter limits."""

from __future__ import annotations

import sys

import pytest
from lcb_integration.testing_util import (
    Capturing,
    grade_call_based,
    import_string,
    values_are_close,
)


class TestStdoutCaptureSupportsBuffer:
    """Submitted code may use the `sys.stdout.buffer.write` fast-output idiom."""

    def test_buffer_write_is_captured(self):
        with Capturing() as captured:
            sys.stdout.buffer.write(b"42\n")
        assert captured[0] == "42\n"

    def test_text_writes_are_captured(self):
        with Capturing() as captured:
            print("hello")
        assert captured[0] == "hello\n"

    def test_text_and_buffer_writes_keep_order(self):
        with Capturing() as captured:
            print("a")
            sys.stdout.buffer.write(b"b\n")
            print("c")
        assert captured[0] == "a\nb\nc\n"

    def test_closing_stdout_does_not_lose_output(self):
        with Capturing() as captured:
            print("before")
            sys.stdout.close()
            print("after")
        assert captured[0] == "before\nafter\n"

    def test_stdout_is_restored(self):
        original = sys.stdout
        with Capturing():
            print("ignored")
        assert sys.stdout is original


class TestInterpreterLimits:
    """Competitive answers can be very large integers, which Python refuses to print by default."""

    def test_import_string_raises_the_int_to_str_limit(self):
        previous = sys.get_int_max_str_digits()
        try:
            exec(import_string, {})
            assert sys.get_int_max_str_digits() >= 1 << 20
            assert len(str(10**10000)) == 10001
        finally:
            sys.set_int_max_str_digits(previous)

    def test_default_limit_would_reject_a_large_answer(self):
        previous = sys.get_int_max_str_digits()
        try:
            sys.set_int_max_str_digits(4300)
            with pytest.raises(ValueError):
                str(10**10000)
        finally:
            sys.set_int_max_str_digits(previous)


class TestCallBasedFloatLists:
    """A returned list of floats within tolerance must not be scored Wrong Answer."""

    def test_float_list_within_tolerance_passes(self):
        code = "class Solution:\n    def f(self, n):\n        return [0.1000000001, 0.2]\n"
        results, _metadata = grade_call_based(
            code=code, all_inputs=["[1]"], all_outputs=["[0.1, 0.2]"], fn_name="f", timeout=10
        )
        assert results == [True]

    def test_float_list_outside_tolerance_fails(self):
        code = "class Solution:\n    def f(self, n):\n        return [0.5, 0.2]\n"
        results, _metadata = grade_call_based(
            code=code, all_inputs=["[1]"], all_outputs=["[0.1, 0.2]"], fn_name="f", timeout=10
        )
        assert results != [True]

    def test_float_list_of_different_length_fails(self):
        code = "class Solution:\n    def f(self, n):\n        return [0.1, 0.2, 0.3]\n"
        results, _metadata = grade_call_based(
            code=code, all_inputs=["[1]"], all_outputs=["[0.1, 0.2]"], fn_name="f", timeout=10
        )
        assert results != [True]

    def test_exact_int_list_still_passes(self):
        code = "class Solution:\n    def f(self, n):\n        return [1, 2, 3]\n"
        results, _metadata = grade_call_based(
            code=code, all_inputs=["[1]"], all_outputs=["[1, 2, 3]"], fn_name="f", timeout=10
        )
        assert results == [True]


class TestCallBasedIntegersStayExact:
    """Tolerance must never apply to integers, in either the scalar or the list case."""

    def test_large_int_list_off_by_one_fails(self):
        code = "class Solution:\n    def f(self, n):\n        return [50000000000000001]\n"
        results, _metadata = grade_call_based(
            code=code,
            all_inputs=["[1]"],
            all_outputs=["[50000000000000000]"],
            fn_name="f",
            timeout=10,
        )
        assert results != [True]

    def test_large_scalar_int_off_by_one_fails(self):
        code = "class Solution:\n    def f(self, n):\n        return 50000000000000001\n"
        results, _metadata = grade_call_based(
            code=code,
            all_inputs=["[1]"],
            all_outputs=["50000000000000000"],
            fn_name="f",
            timeout=10,
        )
        assert results != [True]

    def test_small_int_list_off_by_one_fails(self):
        code = "class Solution:\n    def f(self, n):\n        return [1, 2, 4]\n"
        results, _metadata = grade_call_based(
            code=code, all_inputs=["[1]"], all_outputs=["[1, 2, 3]"], fn_name="f", timeout=10
        )
        assert results != [True]

    def test_values_are_close_gates_on_float_type(self):
        assert not values_are_close(50000000000000000, 50000000000000001)
        assert not values_are_close(41, 42)
        assert values_are_close(50000000000000000, 50000000000000000)
        assert values_are_close(0.1, 0.1000000001)
        assert not values_are_close(0.1, 0.5)

    def test_booleans_never_get_tolerance(self):
        assert not values_are_close(True, 0.9999999999)
        assert not values_are_close(1.0, False)
        assert values_are_close(True, True)
        assert values_are_close(1.0, True)
