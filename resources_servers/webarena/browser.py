# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from urllib.parse import urlparse

from resources_servers.webarena.schemas import WebArenaTaskRow


logger = logging.getLogger(__name__)
_INITIAL_NAVIGATION_TIMEOUT_MS = 35_000

URL_PLACEHOLDERS = {
    "__GITLAB__": "gitlab",
    "__REDDIT__": "reddit",
    "__SHOPPING__": "shopping",
    "__SHOPPING_ADMIN__": "shopping_admin",
    "__WIKIPEDIA__": "wikipedia",
    "__MAP__": "map",
    "__CLASSIFIEDS__": "classifieds",
}

CHROME_ARGS = [
    "--window-position=0,0",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-background-networking",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-breakpad",
    "--disable-component-extensions-with-background-pages",
    "--disable-dev-shm-usage",
    "--disable-features=TranslateUI",
    "--disable-ipc-flooding-protection",
    "--disable-renderer-backgrounding",
    "--force-color-profile=srgb",
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
    "--mute-audio",
    "--no-sandbox",
    "--disable-gpu",
    "--disable-quic",
    "--disable-http2",
]


def substitute_site_urls(task: WebArenaTaskRow, urls: dict[str, str]) -> WebArenaTaskRow:
    payload = json.dumps(task.model_dump(mode="json"))
    for placeholder, site in URL_PLACEHOLDERS.items():
        if site in urls:
            payload = payload.replace(placeholder, urls[site].rstrip("/"))
    if "__GITLAB_SSH__" in payload and "gitlab" in urls:
        parsed = urlparse(urls["gitlab"])
        authority = parsed.hostname or parsed.path.split("/", 1)[0]
        payload = payload.replace("__GITLAB_SSH__", authority)
    return WebArenaTaskRow.model_validate_json(payload)


async def _dismiss_dialog(dialog: Any) -> None:
    try:
        await dialog.accept()
    except Exception:
        pass


def _install_dialog_handler(page: Any) -> None:
    page.on("dialog", lambda dialog: asyncio.ensure_future(_dismiss_dialog(dialog)))


async def login_site(
    *,
    site: str,
    url: str,
    context: Any,
    credentials: dict[str, dict[str, str]],
) -> None:
    creds = credentials.get(site)
    if not creds:
        return
    page = await context.new_page()
    try:
        if site == "reddit":
            await page.goto(url, wait_until="domcontentloaded")
            await page.get_by_role("link", name="Log in").click()
            await page.get_by_label("Username").fill(creds["username"])
            await page.get_by_label("Password").fill(creds["password"])
            await page.get_by_role("button", name="Log in").click()
        elif site == "gitlab":
            await page.goto(f"{url.rstrip('/')}/users/sign_in", wait_until="domcontentloaded")
            await page.get_by_label("Username or email").fill(creds["username"])
            await page.get_by_label("Password").fill(creds["password"])
            await page.get_by_role("button", name="Sign in").click()
        elif site == "shopping":
            await page.goto(f"{url.rstrip('/')}/customer/account/login/", wait_until="domcontentloaded")
            await page.get_by_label("Email", exact=True).fill(creds["username"])
            await page.get_by_label("Password", exact=True).fill(creds["password"])
            await page.get_by_role("button", name="Sign In").click()
        elif site == "shopping_admin":
            await page.goto(url, wait_until="domcontentloaded")
            await page.get_by_label("Username").fill(creds["username"])
            await page.get_by_label("Password").fill(creds["password"])
            await page.get_by_role("button", name="Sign in").click()
        elif site == "classifieds":
            await page.goto(f"{url.rstrip('/')}/index.php?page=login", wait_until="domcontentloaded")
            await page.locator("#email").fill(creds["username"])
            await page.locator("#password").fill(creds["password"])
            await page.get_by_role("button", name="Log in").click()
        await asyncio.sleep(2)
    finally:
        await page.close()


async def login_sites(
    *,
    sites: list[str],
    urls: dict[str, str],
    context: Any,
    credentials: dict[str, dict[str, str]],
) -> None:
    for site in sites:
        if site not in urls:
            continue
        for attempt in range(3):
            try:
                await login_site(
                    site=site,
                    url=urls[site],
                    context=context,
                    credentials=credentials,
                )
                break
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(2**attempt)


async def _navigate_initial_page(page: Any, url: str) -> None:
    for attempt in range(3):
        try:
            await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=_INITIAL_NAVIGATION_TIMEOUT_MS,
            )
            return
        except Exception:
            if attempt == 2:
                raise
            logger.warning(
                "Initial task navigation to %s failed on attempt %d/3; retrying",
                url,
                attempt + 1,
                exc_info=True,
            )
            await asyncio.sleep(2**attempt)


async def setup_task_context(
    *,
    browser: Any,
    task: WebArenaTaskRow,
    urls: dict[str, str],
    credentials: dict[str, dict[str, str]],
    viewport_width: int,
    viewport_height: int,
) -> tuple[WebArenaTaskRow, Any, Any]:
    missing_sites = [site for site in task.sites if not urls.get(site)]
    if missing_sites:
        raise ValueError("Missing WebArena site URL configuration for: " + ", ".join(sorted(missing_sites)))
    resolved_task = substitute_site_urls(task, urls)
    context = await browser.new_context(viewport={"width": viewport_width, "height": viewport_height})
    context.set_default_timeout(45_000)
    context.set_default_navigation_timeout(45_000)
    page = await context.new_page()
    _install_dialog_handler(page)
    try:
        await login_sites(
            sites=resolved_task.sites,
            urls=urls,
            context=context,
            credentials=credentials,
        )
        for index, start_url in enumerate(resolved_task.start_urls):
            if index:
                page = await context.new_page()
                _install_dialog_handler(page)
            await _navigate_initial_page(page, start_url)
        if resolved_task.start_urls:
            await asyncio.sleep(2)
        return resolved_task, context, page
    except Exception:
        await context.close()
        raise


async def tab_context(page: Any | None) -> dict[str, Any] | None:
    if page is None:
        return None
    pages = list(page.context.pages)
    tabs = []
    for index, tab in enumerate(pages):
        try:
            title = await tab.title()
        except Exception:
            title = "unknown"
        tabs.append({"tab_id": index, "title": title, "url": tab.url})
    try:
        current = pages.index(page)
    except ValueError:
        current = -1
    return {"current_tab_id": current, "tab_count": len(pages), "tabs": tabs}
