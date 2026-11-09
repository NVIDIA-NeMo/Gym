# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The public replay suite uses only its committed synthetic fixtures."""

from responses_api_agents.ngc_recorded_context.setup_ripgrep import ensure_ripgrep


def pytest_configure() -> None:
    ensure_ripgrep()
