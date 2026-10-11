# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluator-facing view of public benchmark account credentials."""

from __future__ import annotations

from resources_servers.webarena_browser.site_auth import benchmark_credentials


def _credential(site: str) -> dict[str, str]:
    credentials = benchmark_credentials(site)
    if credentials is None:
        raise RuntimeError(f"site {site!r} has no benchmark credentials")
    return {"username": credentials[0], "password": credentials[1]}


# The reference helpers currently consume this mapping for Magento API auth.
# Values are the benchmark-image accounts (or WA_* process overrides), never
# user credentials.
DEFAULT_CREDENTIALS = {
    site: _credential(site) for site in ("shopping", "shopping_admin", "reddit", "gitlab", "classifieds")
}
