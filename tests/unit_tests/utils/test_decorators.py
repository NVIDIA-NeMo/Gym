# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.utils.decorators import experimental


def test_experimental_warns_on_every_call_by_default(capsys):
    @experimental
    def answer():
        return 42

    assert answer() == 42
    assert answer() == 42

    assert capsys.readouterr().err.count("Warning:") == 2


def test_experimental_once_warns_the_first_time_in_a_process_and_runs_the_function_each_time(capsys):
    @experimental(once=True)
    def answer():
        return 42

    assert answer() == 42
    assert answer() == 42

    err = capsys.readouterr().err
    assert err.count("Warning:") == 1
    assert "answer" in err
