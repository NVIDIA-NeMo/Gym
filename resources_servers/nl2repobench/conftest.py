# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# `data/cache/source/` is a gitignored local checkout of upstream NL2RepoBench task sources (used
# by prepare.py), not test code - it has its own test_*.py files with unresolved local imports
# (e.g. `from logging_config import get_logger`) that pytest otherwise tries to collect and fails
# to import when `gym env test` runs pytest from this directory.
collect_ignore_glob = ["data/**"]
