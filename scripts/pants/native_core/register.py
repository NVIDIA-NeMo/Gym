# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register only the native batch planning goal; test execution stays upstream."""

from scripts.pants.native_core.batching_goal import rules


__all__ = ["rules"]
