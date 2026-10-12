# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Example session providers for `backend: remote_cdp`.

Each subpackage supplies remote browsers from one service. They are imported
lazily by name (see `browser/registry.py`), so an unselected provider never
imports its SDK — the environment itself depends on none of them.
"""
