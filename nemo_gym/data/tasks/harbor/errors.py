# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The error a Harbor task raises when it cannot be loaded or its records cannot be read.

HarborTaskError is defined apart from the loader so that the record readers, which the loader
imports, can raise it without importing the loader back. task.py imports it, so discover_tasks
still catches a record failure as a task failure.
"""


class HarborTaskError(ValueError):
    """The folder is not a runnable Harbor task."""
