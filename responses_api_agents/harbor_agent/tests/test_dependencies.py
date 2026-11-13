# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependencies the Harbor agent server venv must provide."""


def test_harbor_job_imports() -> None:
    # Harbor imports pathspec through dirhash and scantree; Gym's default environment excludes it.
    from harbor.job import Job

    assert Job.__name__ == "Job"
