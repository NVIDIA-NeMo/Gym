# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WebArena-family URL resolution and login for the visual-browser driver."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from nemo_gym.web.models import WebTask


LOG = logging.getLogger("nemo_gym.resources_servers.webarena_browser")

SITE_URL_ENVS = {
    "shopping": "WA_SHOPPING",
    "shopping_admin": "WA_SHOPPING_ADMIN",
    "reddit": "WA_REDDIT",
    "gitlab": "WA_GITLAB",
    "wikipedia": "WA_WIKIPEDIA",
    "map": "WA_MAP",
    "classifieds": "WA_CLASSIFIEDS",
    "homepage": "WA_HOMEPAGE",
}
URL_PLACEHOLDERS = {
    "__SHOPPING__": "shopping",
    "__SHOPPING_ADMIN__": "shopping_admin",
    "__REDDIT__": "reddit",
    "__GITLAB__": "gitlab",
    "__WIKIPEDIA__": "wikipedia",
    "__MAP__": "map",
    "__CLASSIFIEDS__": "classifieds",
    "__HOMEPAGE__": "homepage",
}

# These are the public benchmark accounts bundled with the reference site
# images, not personal credentials. Deployments can override every value with
# the corresponding WA_<SITE>_{USERNAME,PASSWORD} environment variables.
DEFAULT_BENCHMARK_CREDENTIALS = {
    "shopping": ("emma.lopez@gmail.com", "Password.123"),
    "shopping_admin": ("admin", "admin1234"),
    "reddit": ("MarvelsGrantMan136", "test1234"),
    "gitlab": ("byteblaze", "hello1234"),
    "classifieds": ("blake.sullivan@gmail.com", "Password.123"),
}


def configured_site_urls(task: WebTask) -> dict[str, str]:
    """Resolve only the site URLs required by this task."""

    urls: dict[str, str] = {}
    missing: list[str] = []
    for site in sorted(_required_sites(task)):
        env_name = SITE_URL_ENVS.get(site)
        if env_name is None:
            continue
        value = os.environ.get(env_name, "").strip().rstrip("/")
        if value:
            urls[site] = value
        else:
            missing.append(env_name)
    if missing:
        raise ValueError(f"missing WebArena site URL environment variables: {', '.join(sorted(set(missing)))}")
    return urls


def _required_sites(task: WebTask) -> set[str]:
    required = set(task.sites)

    def inspect(value: Any) -> None:
        if isinstance(value, str):
            for placeholder, site in URL_PLACEHOLDERS.items():
                if placeholder in value:
                    required.add(site)
            if "__GITLAB_SSH__" in value:
                required.add("gitlab")
        elif isinstance(value, dict):
            for key, item in value.items():
                inspect(key)
                inspect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                inspect(item)

    inspect(task.start_urls)
    inspect(task.original_metadata)
    return required


def resolve_site_templates(value: Any, site_urls: dict[str, str]) -> Any:
    """Recursively substitute deployment URLs in an immutable task payload."""

    if isinstance(value, str):
        resolved = value
        for placeholder, site in URL_PLACEHOLDERS.items():
            if placeholder not in resolved:
                continue
            base_url = site_urls.get(site)
            if not base_url:
                raise ValueError(f"{placeholder} has no configured site URL")
            resolved = resolved.replace(placeholder, base_url)
        if "__GITLAB_SSH__" in resolved:
            gitlab_url = site_urls.get("gitlab")
            if not gitlab_url:
                raise ValueError("__GITLAB_SSH__ has no configured GitLab URL")
            parsed = urlparse(gitlab_url)
            resolved = resolved.replace("__GITLAB_SSH__", f"{parsed.hostname}:2222")
        return resolved
    if isinstance(value, dict):
        return {
            resolve_site_templates(key, site_urls): resolve_site_templates(item, site_urls)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [resolve_site_templates(item, site_urls) for item in value]
    if isinstance(value, tuple):
        return tuple(resolve_site_templates(item, site_urls) for item in value)
    return value


def resolve_start_urls(task: WebTask, site_urls: dict[str, str]) -> list[str]:
    """Substitute deployment URLs without mutating the immutable task row."""

    return [str(resolve_site_templates(source_url, site_urls)) for source_url in task.start_urls]


def benchmark_credentials(site: str) -> tuple[str, str] | None:
    """Return public benchmark credentials with deployment-scoped overrides."""

    defaults = DEFAULT_BENCHMARK_CREDENTIALS.get(site)
    if defaults is None:
        return None
    prefix = f"WA_{site.upper()}"
    return (
        os.environ.get(f"{prefix}_USERNAME", defaults[0]),
        os.environ.get(f"{prefix}_PASSWORD", defaults[1]),
    )


def login_sites(
    task: WebTask,
    *,
    context: Any,
    site_urls: dict[str, str],
    goto: Callable[[Any, str], None],
) -> None:
    """Log into required local benchmark sites using a disposable page."""

    for site in task.sites:
        credentials = benchmark_credentials(site)
        if credentials is None:
            continue
        base_url = site_urls.get(site)
        if not base_url:
            raise ValueError(f"site {site!r} requires login but has no configured URL")
        for attempt in range(3):
            try:
                _login_site(
                    site,
                    base_url=base_url,
                    username=credentials[0],
                    password=credentials[1],
                    context=context,
                    goto=goto,
                )
                break
            except Exception:
                if attempt == 2:
                    raise
                LOG.warning("event=webarena_site_login_retry site=%s attempt=%d/3", site, attempt + 1)
                time.sleep(2**attempt)


def _login_site(
    site: str,
    *,
    base_url: str,
    username: str,
    password: str,
    context: Any,
    goto: Callable[[Any, str], None],
) -> None:
    page = context.new_page()
    try:
        if site == "reddit":
            goto(page, base_url)
            page.get_by_role("link", name="Log in").click()
            page.get_by_label("Username").fill(username)
            page.get_by_label("Password").fill(password)
            page.get_by_role("button", name="Log in").click()
        elif site == "gitlab":
            goto(page, f"{base_url}/users/sign_in")
            page.get_by_label("Username or email").fill(username)
            page.get_by_label("Password").fill(password)
            page.get_by_role("button", name="Sign in").click()
        elif site == "shopping":
            goto(page, f"{base_url}/customer/account/login/")
            page.get_by_label("Email", exact=True).fill(username)
            page.get_by_label("Password", exact=True).fill(password)
            page.get_by_role("button", name="Sign In").click()
        elif site == "shopping_admin":
            goto(page, base_url)
            page.get_by_label("Username").fill(username)
            page.get_by_label("Password").fill(password)
            page.get_by_role("button", name="Sign in").click()
        elif site == "classifieds":
            goto(page, f"{base_url}/index.php?page=login")
            page.locator("#email").fill(username)
            page.locator("#password").fill(password)
            page.get_by_role("button", name="Log in").click()
        time.sleep(2)
        LOG.info("event=webarena_site_login_complete site=%s", site)
    finally:
        page.close()
