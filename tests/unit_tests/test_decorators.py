# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.decorators import experimental


def test_experimental_warns_then_returns_result(capsys) -> None:
    @experimental
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    assert add.__name__ == "add"
    assert add.__doc__ == "Add two integers."
    assert add(2, 3) == 5
    assert "add is experimental and may change or be removed without notice" in capsys.readouterr().out
