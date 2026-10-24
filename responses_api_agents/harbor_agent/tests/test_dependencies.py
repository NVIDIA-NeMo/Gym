# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke checks for dependencies installed in the Harbor agent server venv."""

import importlib.util

import pytest


def test_harbor_job_imports() -> None:
    if importlib.util.find_spec("harbor") is None:
        pytest.skip("Harbor is installed only in the agent server venv")

    # The root nemo-gym override excludes pathspec unless this server restores it.
    from harbor.job import Job

    assert Job is not None
