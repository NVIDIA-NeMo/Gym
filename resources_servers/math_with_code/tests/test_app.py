# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import MagicMock

import pytest
from app import (
    PythonExecutorResourcesServer,
    PythonExecutorResourcesServerConfig,
    _answers_match,
    _extract_boxed_answer,
    _normalize_answer,
)

from nemo_gym.server_utils import ServerClient


class TestApp:
    """Tests for the Python Executor server."""

    SERVER_NAME = "math_with_code"

    def test_sanity(self) -> None:
        """Basic instantiation test - always runs."""
        config = PythonExecutorResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        PythonExecutorResourcesServer(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"The answer is \boxed{42}.", "42"),
            (r"First \boxed{41}, then \boxed{42}.", "42"),
            (r"The answer is \boxed{\frac{1}{2}}.", r"\frac{1}{2}"),
            (r"The answer is \boxed{}.", None),
            (r"The answer is \boxed{42", None),
            ("The answer is 42.", None),
        ],
        ids=[
            "single-boxed-answer",
            "last-boxed-answer",
            "nested-braces",
            "empty-box",
            "unclosed-box",
            "missing-box",
        ],
    )
    def test_extract_boxed_answer(
        self,
        text: str,
        expected: str | None,
    ) -> None:
        assert _extract_boxed_answer(text) == expected

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("42", "42"),
            ("  42  ", "42"),
            (r"\(42\)", "42"),
            ("$42$", "42"),
            (r"\text{42}", "42"),
            ("1   2", "1 2"),
        ],
        ids=[
            "plain",
            "whitespace",
            "latex-parentheses",
            "dollar-delimiters",
            "text-wrapper",
            "internal-whitespace",
        ],
    )
    def test_normalize_answer(
        self,
        answer: str,
        expected: str,
    ) -> None:
        assert _normalize_answer(answer) == expected

    @pytest.mark.parametrize(
        ("actual", "expected", "matches"),
        [
            ("42", "42", True),
            (r"\(42\)", "42", True),
            ("42", "42.0", True),
            ("0.5", ".5", True),
            ("43", "42", False),
            ("not-a-number", "42", False),
            (None, "42", False),
        ],
        ids=[
            "exact-match",
            "normalized-exact-match",
            "numeric-equivalent",
            "decimal-equivalent",
            "numeric-mismatch",
            "non-numeric-mismatch",
            "missing-answer",
        ],
    )
    def test_answers_match(
        self,
        actual: str | None,
        expected: str,
        matches: bool,
    ) -> None:
        assert _answers_match(actual, expected) is matches
