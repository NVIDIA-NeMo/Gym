# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Pytest/coverage discover pyproject.toml themselves; Pants rewrites the latter
# in its sandbox, so adding the original as a file dependency would collide.
files(name="project-config", sources=["uv.lock"])
files(name="pytest-config", sources=["pyproject.toml"])
