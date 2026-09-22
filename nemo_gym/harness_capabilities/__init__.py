# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable artifact contracts for harness observability.

Readers supply native Gym JSON;
artifact conformance is deliberately separate from behavioral qualification.
"""

from .checker import inspect_record


__all__ = ["inspect_record"]
__version__ = "0.1.1"
