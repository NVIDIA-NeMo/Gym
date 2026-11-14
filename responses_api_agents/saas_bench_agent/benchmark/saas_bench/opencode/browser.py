# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Episode-local UI and task-file MCP tools; no shell or arbitrary JavaScript."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import re
import sys
from contextlib import asynccontextmanager
from functools import wraps
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from mcp.server.fastmcp import FastMCP, Image
from playwright.async_api import Error as PlaywrightError, async_playwright


def origin(url: str) -> str:
    parts = urlsplit(url)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
    ):
        raise ValueError("Only assigned HTTP(S) app URLs are allowed")
    return f"{parts.scheme}://{parts.hostname}:{parts.port or (443 if parts.scheme == 'https' else 80)}"


class BrowserTools:
    """Own an isolated context and constrain navigation/uploads to task inputs."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.allowed = {origin(url) for url in config["allowed_origins"]}
        self.inputs = {
            Path(path).name: Path(path).resolve(strict=True)
            for path in config["input_files"]
        }
        if len(self.inputs) != len(config["input_files"]):
            raise ValueError("Task inputs must have unique basenames")
        self.artifacts = Path(config["artifacts"])
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.files = Path(config.get("files_dir", self.artifacts / "files")).resolve()
        self.files.mkdir(parents=True, exist_ok=True)
        self.sequence = 0
        self.page = None
        self.lock = asyncio.Lock()
        self.download_tasks = set()
        self.download_errors = []
        self.dialog_events = []
        self.dialog_choice = None
        self.refresh_server = None
        self.refresh_tasks = set()
        self.redirect_guards = {}

    def scope(self, frame_path: list[str] | None = None):
        """Select nested UI frames by selectors, without granting scripting access."""
        scope = self.page
        for selector in frame_path or []:
            scope = scope.frame_locator(selector)
        return scope

    async def save_download(self, page, download) -> None:
        """Persist app downloads under collision-safe, scoped basenames."""
        try:
            if (
                origin(page.url) not in self.allowed
                or origin(download.url.removeprefix("blob:")) not in self.allowed
            ):
                raise ValueError("Download is outside assigned apps")
            name = re.sub(
                r"[^\w. -]",
                "_",
                download.suggested_filename.replace("\\", "/").split("/")[-1],
            )
            if name in {"", ".", ".."}:
                name = "download"
            for number in range(10000):
                candidate = (
                    name
                    if number == 0
                    else f"{Path(name).stem}-{number}{Path(name).suffix}"
                )
                path = self.workspace_file(candidate)
                try:
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                except FileExistsError:
                    continue
                os.close(fd)
                try:
                    await download.save_as(str(path))
                except (Exception, asyncio.CancelledError):
                    path.unlink(missing_ok=True)
                    raise
                return
            raise ValueError("Too many downloads with the same filename")
        except Exception as exc:  # noqa: BLE001 -- report capture/download failures without repeating UI side effects
            self.download_errors.append(str(exc)[:1000])

    def on_download(self, page, download) -> None:
        task = asyncio.create_task(self.save_download(page, download))
        self.download_tasks.add(task)
        task.add_done_callback(self.download_tasks.discard)

    def resolve_input(self, filename: str) -> Path:
        """Resolve a basename or declared path; never open the requested path itself."""
        requested = Path(filename)
        if ".." in requested.parts:
            raise ValueError("Only declared task input files can be accessed")
        if filename in self.inputs:
            return self.inputs[filename]
        root = Path(self.config.get("input_root", Path.cwd()))
        candidate = (root / requested).resolve()
        if candidate in self.inputs.values():
            return candidate
        raise ValueError("Only declared task input files can be accessed")

    def workspace_file(self, filename: str) -> Path:
        """Resolve working files without exposing OpenCode config/session or repo files."""
        path = (self.files / filename).resolve()
        if not path.is_relative_to(self.files) or path == self.files:
            raise ValueError("File must be inside the task working-files directory")
        return path

    async def start(self) -> None:
        self.playwright = await async_playwright().start()
        # The rootless publishers bind IPv4 loopback. Keep localhost browser
        # resolution on that same address even when the host also supplies ::1.
        self.browser = await self.playwright.chromium.launch(
            headless=True, args=["--host-resolver-rules=MAP localhost 127.0.0.1"]
        )
        self.context = await self.browser.new_context(
            viewport=self.config.get("viewport", {"width": 1280, "height": 960}),
            accept_downloads=True,
        )
        self.context.set_default_timeout(15000)

        async def guard(route):
            request = route.request
            try:
                # Enforce app origins on documents (including redirects and popups).
                # Remote fonts/images may load, but never other local app slots.
                parsed = urlsplit(request.url)
                local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
                if (request.is_navigation_request() or local) and origin(
                    request.url
                ) not in self.allowed:
                    await route.abort()
                    return
            except ValueError:
                await route.abort()
                return
            if request.is_navigation_request():
                # Popup requests can precede Playwright's frame/page event.
                # Its initial URL has already passed the origin check above.
                try:
                    page = request.frame.page
                except PlaywrightError:
                    page = None
                if page in self.redirect_guards:
                    await self.redirect_guards[page]
            await route.continue_()

        await self.context.route("**/*", guard)
        self.context.on("page", self._new_page)
        self.page = await self.context.new_page()
        await self.redirect_guards[self.page]
        if self.config.get("refresh_socket"):
            path = Path(self.config["refresh_socket"])
            if path.exists():
                raise RuntimeError("Browser refresh socket already exists")
            self.refresh_server = await asyncio.start_unix_server(
                self.refresh_connection, path=str(path), limit=4096
            )
            path.chmod(0o600)

    async def refresh_connection(self, reader, writer) -> None:
        """Supply current visual evidence to the trusted native compaction hook."""
        task = asyncio.current_task()
        self.refresh_tasks.add(task)
        try:
            request = json.loads(await asyncio.wait_for(reader.readline(), timeout=5))
            if (
                request.get("operation") != "refresh"
                or not isinstance(request.get("summary_id"), str)
                or len(request["summary_id"]) > 200
            ):
                raise ValueError("Unsupported browser refresh request")
            async with self.lock:
                result = await asyncio.wait_for(self.observe(timeout=3000), timeout=20)
                state = json.loads(result[0])
                if not state.get("screenshot_available"):
                    raise RuntimeError("Could not capture fresh browser evidence")
                png = self.artifacts / (state["observation_id"] + ".png")
                reply = {
                    "state": state,
                    "image_url": "data:image/png;base64,"
                    + base64.b64encode(png.read_bytes()).decode(),
                }
                with (self.artifacts / "refresh_events.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "summary_id": request["summary_id"],
                                "observation_id": state["observation_id"],
                                "reason": "post_compaction",
                            }
                        )
                        + "\n"
                    )
            writer.write(json.dumps(reply).encode() + b"\n")
            await writer.drain()
        except Exception as exc:  # noqa: BLE001 -- return an explicit refresh failure, never stale visual evidence
            writer.write(json.dumps({"error": str(exc)[:1000]}).encode() + b"\n")
            try:
                await writer.drain()
            except (ConnectionError, OSError):
                pass
        finally:
            writer.close()
            self.refresh_tasks.discard(task)

    def _new_page(self, page) -> None:
        self.page = page
        self.redirect_guards[page] = asyncio.create_task(self.guard_redirects(page))
        page.on("download", lambda download: self.on_download(page, download))
        page.on("dialog", lambda dialog: self.handle_dialog(page, dialog))

    async def guard_redirects(self, page) -> None:
        """Inspect browser redirect responses without refetching UI requests."""
        from urllib.parse import urljoin

        client = await self.context.new_cdp_session(page)

        async def paused(event):
            try:
                location = next(
                    (
                        header["value"]
                        for header in event.get("responseHeaders", [])
                        if header["name"].lower() == "location"
                    ),
                    None,
                )
                if 300 <= event.get("responseStatusCode", 0) < 400 and location:
                    try:
                        permitted = (
                            origin(urljoin(event["request"]["url"], location))
                            in self.allowed
                        )
                    except ValueError:
                        permitted = False
                    if not permitted:
                        await client.send(
                            "Fetch.failRequest",
                            {
                                "requestId": event["requestId"],
                                "errorReason": "BlockedByClient",
                            },
                        )
                        return
                await client.send(
                    "Fetch.continueRequest", {"requestId": event["requestId"]}
                )
            except Exception:
                if page.is_closed():
                    return
                await page.close()
                raise

        client.on("Fetch.requestPaused", paused)
        await client.send(
            "Fetch.enable",
            {"patterns": [{"resourceType": "Document", "requestStage": "Response"}]},
        )

    async def handle_dialog(self, page, dialog) -> None:
        """Acknowledge requested UI confirmations and retain the actual response."""
        choice = self.dialog_choice
        explicit = choice is not None and choice[0] is page
        response = choice[1] if explicit else "default"
        text = choice[2] if explicit else ""
        if response == "default":
            response = "dismiss" if dialog.type == "prompt" else "accept"
        record = {
            "url": page.url,
            "type": dialog.type,
            "message": dialog.message[:2000],
            "response": response,
            "prompt_text_supplied": bool(text) and dialog.type == "prompt",
        }
        try:
            if response == "accept":
                await dialog.accept(prompt_text=text if dialog.type == "prompt" else "")
            else:
                await dialog.dismiss()
            record["handled"] = True
        except Exception as exc:  # noqa: BLE001 -- retain a closed/failed dialog without losing the UI result
            record.update(handled=False, error=str(exc)[:1000])
        self.dialog_events.append(record)
        with (self.artifacts / "dialogs.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def select_app(self, expected_url: str, *, required: bool = True) -> None:
        """Resolve the intended app under the caller's action lock."""
        if required and len(self.allowed) > 1 and not expected_url:
            raise ValueError(
                "Multiple apps are open: set expected_url to the intended app URL"
            )
        if expected_url:
            if origin(expected_url) not in self.allowed:
                raise ValueError("expected_url is outside assigned apps")
            target = next(
                (
                    p
                    for p in [self.page, *self.context.pages]
                    if p is not None
                    and not p.is_closed()
                    and p.url.startswith(("http:", "https:"))
                    and origin(p.url) == origin(expected_url)
                ),
                None,
            )
            if target is None:
                raise ValueError(
                    "Open the intended app with navigate before interacting"
                )
            self.page = target

    async def close(self) -> None:
        for task in self.redirect_guards.values():
            if not task.done():
                task.cancel()
        if self.redirect_guards:
            await asyncio.gather(*self.redirect_guards.values(), return_exceptions=True)
        if self.refresh_server is not None:
            self.refresh_server.close()
            await self.refresh_server.wait_closed()
            Path(self.config["refresh_socket"]).unlink(missing_ok=True)
        pending = list(self.refresh_tasks)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for task in list(self.download_tasks):
            task.cancel()
        if self.download_tasks:
            await asyncio.gather(*list(self.download_tasks), return_exceptions=True)
        if hasattr(self, "browser"):
            await self.browser.close()
        if hasattr(self, "playwright"):
            await self.playwright.stop()

    async def observe(self, timeout: int = 15000) -> list:
        """Capture evidence without turning a successful UI action into a retry."""
        try:
            return await self._capture_observation(timeout)
        except Exception as exc:  # noqa: BLE001 -- report capture/download failures without repeating UI side effects
            return [
                json.dumps(
                    {
                        "observation_id": f"{self.sequence:05d}",
                        "url": self.page.url,
                        "snapshot": "",
                        "screenshot_available": False,
                        "observation_errors": {"capture": str(exc)[:1000]},
                    }
                )
            ]

    async def _capture_observation(self, timeout: int = 15000) -> list:
        """Return accessibility state and a real screenshot, retaining both."""
        self.sequence += 1
        stem = f"{self.sequence:05d}"
        page = self.page
        if page.is_closed():
            page = next(p for p in self.context.pages if not p.is_closed())
            self.page = page
        errors = {}
        try:
            snapshot = await page.locator("body").aria_snapshot(timeout=timeout)
        except Exception as exc:  # noqa: BLE001 -- report capture/download failures without repeating UI side effects
            errors["snapshot"] = str(exc)[:1000]
            snapshot = ""
            try:
                snapshot = (
                    await page.locator("body").text_content(timeout=min(timeout, 2000))
                ) or ""
            except Exception as fallback:  # noqa: BLE001 -- retain DOM fallback failure
                errors["dom_fallback"] = str(fallback)[:1000]
        frames = []

        async def collect_frames(parent, path):
            for child in parent.child_frames:
                if len(frames) >= 12:
                    return
                try:
                    element = await child.frame_element()
                    index = await element.evaluate(
                        "el => Array.from(el.ownerDocument.querySelectorAll('iframe, frame')).indexOf(el)"
                    )
                    if index < 0:
                        continue
                except Exception as exc:  # noqa: BLE001 -- a detached frame must not discard other page evidence
                    errors.setdefault("frame_inventory", str(exc)[:500])
                    continue
                frame_path = [*path, f"iframe, frame >> nth={index}"]
                entry = {"frame_path": frame_path, "url": child.url, "name": child.name}
                frames.append(entry)
                try:
                    text = await child.locator("body").aria_snapshot(
                        timeout=min(timeout, 1000)
                    )
                    nonlocal snapshot
                    snapshot += "\nFrame " + json.dumps(frame_path) + ":\n" + text
                except Exception as exc:  # noqa: BLE001 -- report capture/download failures without repeating UI side effects
                    entry["observation_error"] = str(exc)[:500]
                await collect_frames(child, frame_path)

        await collect_frames(page.main_frame, [])
        png = None
        screenshot_source = None
        try:
            png = await page.screenshot(timeout=timeout)
            screenshot_source = "playwright"
            (self.artifacts / f"{stem}.png").write_bytes(png)
        except Exception as exc:  # noqa: BLE001 -- report capture/download failures without repeating UI side effects
            errors["screenshot"] = str(exc)[:1000]
            session = None
            try:
                session = await self.context.new_cdp_session(page)
                capture = await asyncio.wait_for(
                    session.send(
                        "Page.captureScreenshot",
                        {"format": "png", "captureBeyondViewport": False},
                    ),
                    timeout=min(timeout / 1000, 3),
                )
                png = base64.b64decode(capture["data"], validate=True)
                (self.artifacts / f"{stem}.png").write_bytes(png)
                screenshot_source = "cdp_fallback"
            except Exception as fallback:  # noqa: BLE001 -- retain capture failures separately from action execution
                errors["screenshot_fallback"] = str(fallback)[:1000]
            finally:
                if session is not None:
                    try:
                        await session.detach()
                    except Exception:  # noqa: BLE001 -- closing the page can invalidate its CDP session
                        pass
        try:
            title = await page.title()
        except Exception as exc:  # noqa: BLE001 -- report capture/download failures without repeating UI side effects
            errors["title"] = str(exc)[:1000]
            title = ""
        if self.download_tasks:
            await asyncio.wait(list(self.download_tasks), timeout=5)
        state = {
            "observation_id": stem,
            "url": page.url,
            "page_id": str(id(page)),
            "viewport": page.viewport_size,
            "title": title,
            "snapshot": snapshot,
            "frames": frames,
            "observation_errors": errors,
            "download_errors": list(self.download_errors),
            "downloads_pending": len(self.download_tasks),
            "dialogs": self.dialog_events[-20:],
            "screenshot_available": png is not None,
            "screenshot_source": screenshot_source,
            "inputs": sorted(self.inputs),
            "working_files": [
                str(p.relative_to(self.files))
                for p in sorted(self.files.rglob("*"))
                if p.is_file() and p.resolve().is_relative_to(self.files)
            ][:200],
            "tabs": [p.url for p in self.context.pages],
        }
        (self.artifacts / f"{stem}.json").write_text(
            json.dumps(state, ensure_ascii=False)
        )
        state["snapshot"] = snapshot[:6000]
        state["snapshot_truncated"] = len(snapshot) > 6000
        state["snapshot_total_chars"] = len(snapshot)
        state["snapshot_next_offset"] = 6000 if len(snapshot) > 6000 else None
        state["screenshot_file"] = f"{stem}.png" if png else None
        state["selector_help"] = (
            'Use role="button", name="Save", or selector=\'text="Save"\'. '
            'Accessibility text such as button "Save" is not a CSS selector. '
            "Use find_elements for attributes and unique selectors; never guess icon names. "
            "For iframe content, pass frame_path from the frames inventory to DOM tools. "
            "Read more snapshot text using observe(observation_id=..., offset=...). Recover archived screenshots with include_screenshot=True."
        )
        result = [json.dumps(state, ensure_ascii=False)]
        if png:
            result.append(Image(data=png, format="png"))
        return result


def create_server(config: dict, *, evidence_factory=None) -> FastMCP:
    """Expose browser actions and scoped task files using native MCP schemas."""
    tools = BrowserTools(config)
    # Optional SDG observer. Ordinary benchmark sessions retain the 2.1.1 path.
    evidence = evidence_factory(tools) if evidence_factory is not None else None

    @asynccontextmanager
    async def lifespan(server):
        try:
            await tools.start()
            yield tools
        finally:
            await tools.close()

    server = FastMCP("saas-browser", lifespan=lifespan)

    def serialized(fn):
        """Keep actions and their evidence atomic, including failed actions."""

        if fn.__name__ not in {"navigate", "inspect_input", "switch_tab", "close_tab"}:
            fn.__doc__ = (
                (fn.__doc__ or "")
                + "\nFor multiple apps, pass expected_url to select the intended app atomically."
            )

        page_tools = {
            "interact",
            "interact_at",
            "upload",
            "go_back",
            "wait",
            "dropdown_options",
            "extract",
            "find_elements",
            "find_text",
            "search_page",
            "save_as_pdf",
            "observe",
        }
        if len(tools.allowed) > 1 and fn.__name__ in page_tools:
            signature = inspect.signature(fn, eval_str=True)
            fn.__signature__ = signature.replace(
                parameters=[
                    parameter.replace(
                        kind=inspect.Parameter.KEYWORD_ONLY,
                        default=inspect.Parameter.empty
                        if parameter.name == "expected_url"
                        else parameter.default,
                    )
                    for parameter in signature.parameters.values()
                ]
            )

        @wraps(fn)
        async def wrapped(*args, **kwargs):
            async with tools.lock:
                capture = None
                try:
                    if evidence is not None:
                        capture = await evidence.begin(fn, args, kwargs)
                        capture['execution_phase'] = 'dispatch_validation'
                    if (
                        fn.__name__
                        in {
                            "interact",
                            "interact_at",
                            "upload",
                            "go_back",
                            "wait",
                            "dropdown_options",
                            "extract",
                            "find_elements",
                            "find_text",
                            "search_page",
                            "save_as_pdf",
                        }
                        or fn.__name__ == "observe"
                        and not kwargs.get("observation_id")
                    ):
                        tools.select_app(kwargs.get("expected_url", ""))
                    tools.dialog_choice = (
                        tools.page,
                        kwargs.get("dialog_response", "default"),
                        kwargs.get("prompt_text", ""),
                    )
                    if capture is not None:
                        capture['execution_phase'] = 'tool_execution'
                    result = await fn(*args, **kwargs)
                    if fn.__name__ in {
                        "interact",
                        "interact_at",
                        "upload",
                        "navigate",
                        "go_back",
                        "close_tab",
                        "switch_tab",
                    } and (
                        isinstance(result, list)
                        and result
                        and isinstance(result[0], str)
                    ):
                        state = json.loads(result[0])
                        state["action_status"] = "succeeded"
                        state["observation_status"] = (
                            "partial" if state.get("observation_errors") else "complete"
                        )
                        result[0] = json.dumps(state, ensure_ascii=False)
                    if evidence is not None:
                        return await evidence.finish(capture, result=result)
                    return result
                except Exception as exc:
                    if evidence is not None:
                        detail = await evidence.fail(capture, exc)
                        raise RuntimeError("\n" + detail) from exc
                    # Keep the original exception/error semantics. Retain fresh
                    # screenshot evidence and include bounded state in the error.
                    detail = None
                    if tools.page is not None and not tools.page.is_closed():
                        try:
                            observation = await tools.observe(timeout=2000)
                            state = json.loads(observation[0])
                            state["snapshot"] = state["snapshot"][:2500]
                            detail = json.dumps(
                                {"error": str(exc)[:1500], "observation": state}
                            )
                        except Exception as capture_exc:  # noqa: BLE001 -- retain original action error
                            print(
                                f"Error observation failed: {capture_exc}",
                                file=sys.stderr,
                            )
                    if detail is not None:
                        raise type(exc)(detail) from exc
                    raise
                finally:
                    tools.dialog_choice = None

        return wrapped

    @server.tool(structured_output=False if evidence is not None else None)
    @serialized
    async def navigate(url: str, new_tab: bool = False) -> list:
        """Open one of the assigned SaaS app URLs; returns screenshot and accessibility tree."""
        if origin(url) not in tools.allowed:
            raise ValueError("URL is outside the assigned SaaS app origins")
        if tools.config.get("ui_only_navigation") and (
            urlsplit(url).path not in {"", "/"}
            or urlsplit(url).query
            or urlsplit(url).fragment
        ):
            raise ValueError(
                "navigate only opens assigned app roots; follow internal links with interact"
            )
        # Different apps get separate tabs even when OpenCode batches navigations.
        # Revisiting an app selects its existing tab instead of replacing another app.
        target = next(
            (
                p
                for p in tools.context.pages
                if not p.is_closed()
                and p.url.startswith(("http:", "https:"))
                and origin(p.url) == origin(url)
            ),
            None,
        )
        if target is not None and not new_tab:
            tools.page = target
        elif (
            new_tab
            or tools.page is None
            or tools.page.is_closed()
            or (
                tools.page.url != "about:blank"
                and (
                    not tools.page.url.startswith(("http:", "https:"))
                    or origin(tools.page.url) != origin(url)
                )
            )
        ):
            tools.page = await tools.context.new_page()
        await tools.page.goto(url, wait_until="domcontentloaded")
        return await tools.observe()

    @server.tool(structured_output=False if evidence is not None else None)
    @serialized
    async def observe(
        observation_id: str = "",
        offset: int = 0,
        expected_url: str = "",
        include_screenshot: bool = False,
    ) -> list:
        """Observe the active page, or page through a saved snapshot in 6000-character chunks.

        Pass the returned observation_id and snapshot_next_offset to read more.
        Historical reads do not take another screenshot. Set include_screenshot=True
        to retrieve that observation's retained PNG, including failed-action evidence.
        """
        if observation_id:
            if not re.fullmatch(r"[0-9]{5,}", observation_id) or offset < 0:
                raise ValueError("Invalid observation_id or offset")
            state = json.loads((tools.artifacts / f"{observation_id}.json").read_text())
            snapshot = state["snapshot"]
            state.update(
                snapshot=snapshot[offset : offset + 6000],
                snapshot_total_chars=len(snapshot),
                snapshot_next_offset=offset + 6000
                if offset + 6000 < len(snapshot)
                else None,
            )
            result = [json.dumps(state, ensure_ascii=False)]
            if (
                include_screenshot
                and state.get("screenshot_available", True)
                and (tools.artifacts / f"{observation_id}.png").is_file()
            ):
                result.append(
                    Image(
                        data=(tools.artifacts / f"{observation_id}.png").read_bytes(),
                        format="png",
                    )
                )
            return result
        return await tools.observe()

    @server.tool(structured_output=False if evidence is not None else None)
    @serialized
    async def interact(
        action: Literal[
            "click", "fill", "press", "select", "check", "uncheck", "hover", "scroll"
        ],
        selector: str = "body",
        value: str = "",
        role: str = "",
        name: str = "",
        expected_url: str = "",
        button: Literal["left", "right", "middle"] = "left",
        frame_path: list[str] | None = None,
        dialog_response: Literal["default", "accept", "dismiss"] = "default",
        prompt_text: str = "",
    ) -> list:
        """Use a CSS selector, or role and exact accessible name, to interact through the UI.

        fill types value; press sends a key (e.g. Enter); select chooses an option value;
        scroll moves vertically by value pixels (default 600). button supports context menus.
        For multiple apps, expected_url must be the app URL you intend to act on;
        the matching tab is selected atomically. Use role="textbox", name="Username"
        or selector='text="Save"'; never selector='button "Save"' or 'text: Save'.
        frame_path is an ordered list of iframe selectors, outermost first.
        Confirmations/alerts are accepted and recorded by default; prompts are dismissed.
        Set dialog_response to override the response for this action, and prompt_text
        to supply text when accepting a JavaScript prompt.
        No JavaScript execution.
        """
        if not role and (
            re.match(r'^\s*(button|textbox|heading|link|checkbox)\s+"', selector)
            or re.match(r"^\s*\.?text\s*:", selector)
        ):
            raise ValueError(
                'Invalid selector syntax. Use role="button", name="Save" or selector=\'text="Save"\'; accessibility descriptions are not selectors'
            )
        locator = (
            tools.scope(frame_path).get_by_role(role, name=name, exact=True)
            if role
            else tools.scope(frame_path).locator(selector)
        )
        if action == "scroll":
            amount = max(-3000, min(3000, int(value or "600")))
            if selector in {"body", "html"} and not role:
                # Hovering a tall body can reset its scroll position; target its
                # document explicitly using fixed adapter code, including frames.
                await locator.evaluate(
                    "(el, amount) => el.ownerDocument.scrollingElement.scrollBy(0, amount)",
                    amount,
                )
            else:
                await locator.hover()
                await tools.page.mouse.wheel(0, amount)
            await tools.page.wait_for_timeout(100)
        elif action in {"fill", "press"}:
            await getattr(locator, action)(value)
        elif action == "click":
            await locator.click(button=button)
        elif action == "select":
            await locator.select_option(value)
        else:
            await getattr(locator, action)()
        return await tools.observe()

    @server.tool(structured_output=False if evidence is not None else None)
    @serialized
    async def interact_at(
        action: Literal["click", "hover", "scroll"],
        x: float,
        y: float,
        observation_id: str,
        expected_url: str = "",
        button: Literal["left", "right", "middle"] = "left",
        delta_y: int = 600,
        dialog_response: Literal["default", "accept", "dismiss"] = "default",
        prompt_text: str = "",
    ) -> list:
        """Act on a visible canvas using viewport coordinates from the latest screenshot.

        Use DOM selectors when available. Coordinates include iframe offsets already;
        observation_id must refer to the latest observation of the selected tab.
        dialog_response and prompt_text control dialogs opened by this action.
        """
        if observation_id != f"{tools.sequence:05d}":
            raise ValueError("Coordinates require the latest observation_id")
        state = json.loads((tools.artifacts / f"{observation_id}.json").read_text())
        if (
            state["url"] != tools.page.url
            or state.get("page_id") != str(id(tools.page))
            or state.get("viewport") != tools.page.viewport_size
            or not state.get("screenshot_available")
        ):
            raise ValueError(
                "Get a fresh screenshot of the intended tab before coordinate actions"
            )
        viewport = tools.page.viewport_size
        if not viewport or not (
            0 <= x < viewport["width"] and 0 <= y < viewport["height"]
        ):
            raise ValueError("Coordinates are outside the viewport")
        await tools.page.mouse.move(x, y)
        if action == "click":
            await tools.page.mouse.click(x, y, button=button)
        elif action == "scroll":
            await tools.page.mouse.wheel(0, max(-3000, min(3000, delta_y)))
            await tools.page.wait_for_timeout(100)
        return await tools.observe()

    @server.tool(structured_output=False if evidence is not None else None)
    @serialized
    async def upload(
        selector: str,
        filename: str,
        expected_url: str = "",
        frame_path: list[str] | None = None,
    ) -> list:
        """Upload a declared input or generated task working file to a file input."""
        try:
            path = tools.resolve_input(filename)
        except ValueError:
            try:
                path = tools.workspace_file(filename)
                if not path.is_file():
                    raise ValueError("File does not exist")
            except ValueError as exc:
                raise ValueError(
                    "Only declared inputs or task working files can be uploaded"
                ) from exc
        await tools.scope(frame_path).locator(selector).set_input_files(str(path))
        return await tools.observe()

    @server.tool(structured_output=False if evidence is not None else None)
    @serialized
    async def inspect_input(filename: str, page: int = 1) -> list:
        """View a declared or working-file image/PDF page (from 1)."""
        try:
            path = tools.resolve_input(filename)
        except ValueError as exc:
            try:
                path = tools.workspace_file(filename)
                if not path.is_file():
                    raise ValueError("File does not exist")
            except ValueError:
                raise ValueError(
                    "Only declared inputs or task working files can be previewed"
                ) from exc
        if path.suffix.lower() == ".pdf":
            import pymupdf

            with pymupdf.open(path) as document:
                if not 1 <= page <= len(document):
                    raise ValueError("PDF page is out of range")
                page_count = len(document)
                png = (
                    document[page - 1]
                    .get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5))
                    .tobytes("png")
                )
            return [
                json.dumps(
                    {
                        "filename": filename,
                        "page": page,
                        "pages": page_count,
                        "next_page": page + 1 if page < page_count else None,
                    }
                ),
                Image(data=png, format="png"),
            ]
        formats = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp"}
        if path.suffix.lower() not in formats:
            raise ValueError(
                "Input preview supports images and PDF; use upload for other file types"
            )
        return [
            filename,
            Image(data=path.read_bytes(), format=formats[path.suffix.lower()]),
        ]

    @server.tool(structured_output=False if evidence is not None else None)
    @serialized
    async def switch_tab(index: int) -> list:
        """Switch to a zero-based tab index listed by observe."""
        if not 0 <= index < len(tools.context.pages):
            raise ValueError("Invalid tab index")
        tools.page = tools.context.pages[index]
        await tools.page.bring_to_front()
        return await tools.observe()

    @server.tool(structured_output=False if evidence is not None else None)
    @serialized
    async def go_back(expected_url: str = "") -> list:
        """Go back in the current tab's browser history and observe the page."""
        await tools.page.go_back(wait_until="domcontentloaded")
        return await tools.observe()

    @server.tool(structured_output=False if evidence is not None else None)
    @serialized
    async def close_tab(index: int | None = None) -> list:
        """Close the current tab, or a zero-based tab index; keep one usable tab."""
        pages = tools.context.pages
        page = tools.page
        if index is not None:
            if not 0 <= index < len(pages):
                raise ValueError("Invalid tab index")
            page = pages[index]
        await page.close()
        if tools.page == page:
            tools.page = (
                tools.context.pages[-1]
                if tools.context.pages
                else await tools.context.new_page()
            )
        return await tools.observe()

    @server.tool(structured_output=False if evidence is not None else None)
    @serialized
    async def wait(
        seconds: float = 1,
        selector: str = "",
        state: Literal["visible", "hidden", "attached", "detached"] = "visible",
        expected_url: str = "",
        frame_path: list[str] | None = None,
    ) -> list:
        """Wait up to 30 seconds for a selector state, or pause for seconds."""
        if not 0 <= seconds <= 30:
            raise ValueError("seconds must be between 0 and 30")
        if selector:
            await (
                tools.scope(frame_path)
                .locator(selector)
                .wait_for(state=state, timeout=max(1, seconds * 1000))
            )
        else:
            await asyncio.sleep(seconds)
        return await tools.observe()

    @server.tool(structured_output=False if evidence is not None else None)
    @serialized
    async def dropdown_options(
        selector: str,
        expected_url: str = "",
        offset: int = 0,
        frame_path: list[str] | None = None,
    ) -> list[dict]:
        """List up to ten HTML select options. Increment offset by ten for more."""
        if offset < 0:
            raise ValueError("offset must be nonnegative")
        options = tools.scope(frame_path).locator(selector).locator("option")
        return [
            {
                "text": (await option.inner_text())[:200],
                "value": (await option.get_attribute("value") or "")[:200],
                "selected": await option.evaluate("element => element.selected"),
                "disabled": await option.is_disabled(),
            }
            for option in (await options.all())[offset : offset + 10]
        ]

    @server.tool(structured_output=False if evidence is not None else None)
    @serialized
    async def extract(
        query: str = "",
        selector: str = "body",
        extract_links: bool = False,
        extract_images: bool = False,
        element_offset: int = 0,
        offset: int = 0,
        expected_url: str = "",
        frame_path: list[str] | None = None,
    ) -> dict:
        """Read rendered page text, optionally links/images, for the caller to analyze.

        query records the desired focus; this tool returns DOM content rather than
        invoking another model. Use find_text/search_page for filtered matches.
        """
        if offset < 0 or element_offset < 0:
            raise ValueError("offsets must be nonnegative")
        locator = tools.scope(frame_path).locator(selector)
        text = await locator.evaluate("el => el.innerText ?? el.textContent ?? ''")
        result = {
            "url": tools.page.url,
            "title": await tools.page.title(),
            "query": query,
            "text": text[offset : offset + 6000],
            "truncated": offset + 6000 < len(text),
            "next_offset": offset + 6000 if offset + 6000 < len(text) else None,
        }
        for enabled, name, tag, attr in [
            (extract_links, "links", "a", "href"),
            (extract_images, "images", "img", "src"),
        ]:
            if enabled:
                items = await locator.locator(tag).all()
                result[name + "_next_offset"] = (
                    element_offset + 10 if element_offset + 10 < len(items) else None
                )
                result[name] = [
                    {
                        attr: (await item.get_attribute(attr) or "")[:500],
                        "text": (
                            await item.evaluate(
                                "el => el.innerText ?? el.textContent ?? ''"
                            )
                        )[:200],
                        "alt": (await item.get_attribute("alt") or "")[:200],
                    }
                    for item in items[element_offset : element_offset + 10]
                ]
        return result

    @server.tool(structured_output=False if evidence is not None else None)
    @serialized
    async def find_elements(
        selector: str,
        attributes: list[str] | None = None,
        offset: int = 0,
        expected_url: str = "",
        frame_path: list[str] | None = None,
    ) -> dict:
        """Find DOM elements by CSS selector; return text, attributes and bounds."""
        locator = tools.scope(frame_path).locator(selector)
        count = await locator.count()
        elements = []
        if offset < 0 or len(attributes or []) > 8:
            raise ValueError("offset must be nonnegative; request at most 8 attributes")
        for item in (await locator.all())[offset : offset + 10]:
            elements.append(
                {
                    "text": (
                        await item.evaluate(
                            "el => el.innerText ?? el.textContent ?? ''"
                        )
                    )[:200],
                    "attributes": {
                        attr: (await item.get_attribute(attr) or "")[:200]
                        for attr in (
                            attributes
                            or ["id", "role", "aria-label", "title", "data-type"]
                        )
                    },
                    "bounds": await item.bounding_box(),
                }
            )
        return {
            "count": count,
            "elements": elements,
            "truncated": offset + 10 < count,
            "next_offset": offset + 10 if offset + 10 < count else None,
        }

    @server.tool(structured_output=False if evidence is not None else None)
    @serialized
    async def find_text(
        text: str,
        exact: bool = False,
        offset: int = 0,
        expected_url: str = "",
        frame_path: list[str] | None = None,
    ) -> dict:
        """Locate elements containing visible text; returns text and screen bounds."""
        if offset < 0:
            raise ValueError("offset must be nonnegative")
        locator = tools.scope(frame_path).get_by_text(text, exact=exact)
        count = await locator.count()
        return {
            "count": count,
            "matches": [
                {
                    "text": (
                        await item.evaluate(
                            "el => el.innerText ?? el.textContent ?? ''"
                        )
                    )[:200],
                    "bounds": await item.bounding_box(),
                }
                for item in (await locator.all())[offset : offset + 10]
            ],
            "truncated": offset + 10 < count,
            "next_offset": offset + 10 if offset + 10 < count else None,
        }

    @server.tool(structured_output=False if evidence is not None else None)
    @serialized
    async def search_page(
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = False,
        expected_url: str = "",
        offset: int = 0,
        frame_path: list[str] | None = None,
    ) -> dict:
        """Search page text; use next_offset to paginate ten matches at a time."""
        if offset < 0:
            raise ValueError("offset must be nonnegative")
        source = await tools.scope(frame_path).locator("body").inner_text()
        text = source[:100000]
        expression = re.compile(
            pattern if regex else re.escape(pattern),
            flags=0 if case_sensitive else re.IGNORECASE,
        )
        matches = []
        truncated = False
        for index, match in enumerate(expression.finditer(text)):
            if index < offset:
                continue
            if len(matches) == 10:
                truncated = True
                break
            matches.append(
                {
                    "text": match.group()[:200],
                    "start": match.start(),
                    "end": match.end(),
                    "context": text[max(0, match.start() - 100) : match.start() + 200],
                }
            )
        return {
            "url": tools.page.url,
            "matches": matches,
            "next_offset": offset + 10 if truncated else None,
            "source_truncated": len(source) > len(text),
        }

    @server.tool(structured_output=False if evidence is not None else None)
    @serialized
    async def save_as_pdf(filename: str = "page.pdf", expected_url: str = "") -> dict:
        """Export the current page as a PDF into task working files."""
        path = tools.workspace_file(filename)
        if path.suffix.lower() != ".pdf":
            raise ValueError("PDF filename must end in .pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        await tools.page.pdf(path=str(path), print_background=True)
        return {
            "filename": str(path.relative_to(tools.files)),
            "bytes": path.stat().st_size,
        }

    return server


if __name__ == "__main__":
    create_server(json.loads(Path(sys.argv[1]).read_text())).run(transport="stdio")
