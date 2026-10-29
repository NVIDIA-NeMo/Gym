# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""VAB-CSS from VisualAgentBench (arXiv:2408.06327) as a Gymnasium-style environment.

The model sees a target screenshot, the page's HTML and the current (corrupted) screenshot, and fixes the CSS
with text actions ("Action: edit_rule('.menu li', 'float', 'left')"), one per turn, for at most ``max_rounds``
turns. Each edit returns a new screenshot. The episode ends when the model says "I have fixed the css style" or
the rounds run out; then the last screenshot is compared with the target. Reward is 1.0 when
``1 - SSIM < success_distance`` (0.1 upstream). The loop follows upstream's src/server/tasks/css_agent/task.py.
"""

import asyncio
import base64
import logging
import re
import shlex
import shutil
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlsplit

from fastapi import FastAPI, HTTPException, Request
from pydantic import Field, PrivateAttr

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.gymnasium import (
    EnvStepRequest,
    EnvStepResponse,
    GymnasiumServer,
    Observation,
    extract_text,
)
from resources_servers.interactive_browser.browser._chromium import ensure_chromium
from resources_servers.vab_css.chromium_host import ensure_chromium_host_libraries
from resources_servers.vab_css.css_tools import CSSInteractor, compare_images_ssim_gray


LOGGER = logging.getLogger(__name__)

FIXED_PHRASE = "I have fixed the css style"
DEFAULT_INSTRUCTION = (
    "Before taking any actions, please carefully compare the two screenshots to find the difference in layout."
)
# Upstream also lists insert_rule, but its call always raises (it receives a string), so it is left out here.
ACTIONS = ("get_selectors_by_html_elements", "select_rule", "edit_rule", "revert_last_edit")
SCREENSHOT_NAME = "new_screenshot.png"


class ScreenshotError(RuntimeError):
    """The browser could not render a page: an infrastructure failure, not a mistake of the model."""


class VabCssConfig(BaseResourcesServerConfig):
    data_dir: Optional[str] = Field(
        default=None,
        description="Directory of task folders (upstream data/css_dataset); default: this server's data/example_tasks.",
    )
    work_dir: Optional[str] = Field(default=None, description="Where per-episode task copies go; default: temp dir.")
    max_rounds: int = Field(default=10, ge=1, description="Model turns per episode (upstream round: 10).")
    success_distance: float = Field(default=0.1, description="Success when 1 - SSIM(target, final) is below this.")
    improve_margin: float = Field(default=0.05, description="'improved' when the distance drops by more than this.")
    max_concurrent_screenshots: int = Field(default=8, ge=1)
    install_browser: bool = Field(
        default=True, description="Install Playwright Chromium, and its system libraries as root, if missing."
    )


class VabCssEnv(GymnasiumServer):
    config: VabCssConfig
    _browser: Any = PrivateAttr(default=None)
    _playwright: Any = PrivateAttr(default=None)
    _browser_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    _screenshot_semaphore: Optional[asyncio.Semaphore] = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._screenshot_semaphore = asyncio.Semaphore(self.config.max_concurrent_screenshots)
        if self.config.install_browser:
            ensure_chromium()
            ensure_chromium_host_libraries()

    @property
    def data_dir(self) -> Path:
        return Path(self.config.data_dir) if self.config.data_dir else Path(__file__).parent / "data" / "example_tasks"

    async def _get_browser(self) -> Any:
        async with self._browser_lock:
            if self._browser is None or not self._browser.is_connected():
                from playwright.async_api import async_playwright

                if self._playwright is None:
                    self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch()
            return self._browser

    async def take_screenshot(self, html_path: Path, output_path: Path) -> None:
        """Full-page screenshot at Playwright's default 1280x720 viewport, as upstream's screenshot.py takes it.

        The page's CSS is model-written, so it may load only its own task's files (see _request_allowed).
        """
        async with self._screenshot_semaphore:
            try:
                browser = await self._get_browser()
                page = await browser.new_page()
                try:
                    await page.route("**/*", partial(_route_request, task_dir=html_path.parent.resolve()))
                    await page.goto(html_path.resolve().as_uri(), timeout=60000)
                    await page.screenshot(path=str(output_path), full_page=True, animations="disabled", timeout=60000)
                finally:
                    await page.close()
            except Exception as e:
                raise ScreenshotError(f"could not render {html_path}: {e}") from e

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[Observation], dict]:
        task_id = metadata.get("task_id")
        source = self.data_dir / str(task_id)
        if not task_id or not (source / "index.html").is_file():
            raise HTTPException(status_code=400, detail=f"no VAB-CSS task {task_id!r} under {self.data_dir}")
        self._cleanup_session(session_id)  # a repeated reset replaces the session's earlier episode
        work_root = Path(tempfile.mkdtemp(prefix="vab_css_", dir=self.config.work_dir))
        task_dir = work_root / source.name
        copy = asyncio.ensure_future(asyncio.to_thread(shutil.copytree, source, task_dir))
        try:
            await asyncio.shield(copy)
            interactor = await asyncio.to_thread(CSSInteractor, task_dir / "index.html")
            initial_distance = float((task_dir / "corruption" / "distance.txt").read_text().strip())
            instruction_file = task_dir / "corruption" / "instruction.txt"
            instruction = instruction_file.read_text() if instruction_file.is_file() else DEFAULT_INSTRUCTION
            observation = [
                _text("Here is the target screenshot."),
                _image(task_dir / "index.png"),
                _text("Here is the html content."),
                _text((task_dir / "index.html").read_text()),
                _text("Here is the current screenshot."),
                _image(task_dir / "corruption" / "index_corrupted.png"),
                _text(instruction),
            ]
        except BaseException:
            if copy.done():
                shutil.rmtree(work_root, ignore_errors=True)
            else:  # a cancelled reset leaves the copy thread running: remove the tree once it stops writing
                copy.add_done_callback(lambda _: shutil.rmtree(work_root, ignore_errors=True))
            raise
        self.session_state[session_id] = {
            "work_root": work_root,
            "task_dir": task_dir,
            "interactor": interactor,
            "rounds": 0,
            "actions": [],
            "initial_distance": initial_distance,
            "step_lock": asyncio.Lock(),
            "steps": {},
            "running": set(),
            "finished": False,
        }
        # gymnasium_agent then sends a request id with each step and calls /close when the rollout ends.
        return observation, {"task_id": task_id, "supports_step_idempotency": True, "supports_explicit_close": True}

    async def _step_endpoint(self, body: EnvStepRequest, request: Request) -> EnvStepResponse:
        """Run each step once; a resend with the same ``_ng_step_request_id`` gets the first attempt's result.

        ServerClient resends a request whose connection dropped, and a CSS edit applied twice would replace its undo
        baseline. The step runs shielded, so a dropped connection cannot stop an edit halfway, and the episode's
        state, terminal response included, stays until /close.
        """
        session_id = request.session.get(SESSION_ID_KEY)
        state = self.session_state.get(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="no VAB-CSS episode for this session; call /reset first")
        request_id = (body.model_extra or {}).get("_ng_step_request_id")
        async with state["step_lock"]:
            step = state["steps"].get(request_id) if request_id is not None else None
            if step is None:
                if state["finished"]:
                    raise HTTPException(status_code=409, detail="the VAB-CSS episode has finished; call /close")
                step = asyncio.ensure_future(self._run_step(body, state, session_id))
                state["running"].add(step)
                step.add_done_callback(state["running"].discard)
                if request_id is not None:
                    state["steps"][request_id] = step
            result = await asyncio.shield(step)
        return result.model_copy(deep=True)

    async def _run_step(self, body: EnvStepRequest, state: dict, session_id: Optional[str]) -> EnvStepResponse:
        observation, reward, terminated, truncated, info = await self.step(
            body.response, body.model_extra or {}, session_id
        )
        state["finished"] = terminated or truncated
        return EnvStepResponse(
            observation=observation, reward=reward, terminated=terminated, truncated=truncated, info=info
        )

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[Observation], float, bool, bool, dict]:
        state = self.session_state.get(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="no VAB-CSS episode for this session; call /reset first")
        state["rounds"] += 1
        message = extract_text(action).split("Observation:")[0].strip().replace("Action:\n", "Action:")
        if FIXED_PHRASE in message:
            return await self._finish(state, terminated=True)

        observation = await self._execute(state, message)
        if state["rounds"] >= self.config.max_rounds:
            return await self._finish(state, terminated=False)
        return observation, 0.0, False, False, {}

    async def _execute(self, state: dict, message: str) -> Observation:
        """Run the first Action line of the message, as upstream does, and return what the model sees next."""
        interactor: CSSInteractor = state["interactor"]
        for line in message.split("\n"):
            line = line.strip()
            if not re.match(r"Action.*?:", line):
                continue
            function_names = re.findall(r"(\w+)\(", line)
            if "revert_last_edit" not in function_names and "revert_last_edit" in line:
                function_names.append("revert_last_edit")
            if not function_names:
                continue
            parts: list[dict] = []
            for function_name in function_names:
                arguments: list[str] = []
                try:
                    if function_name == "revert_last_edit":
                        done = interactor.revert_last_edit()
                        state["actions"].append("revert_last_edit()")
                        parts.append(
                            _text("You have successfully reverted the last edit." if done else "No edit to revert.")
                        )
                        break
                    if function_name not in ACTIONS:
                        raise AttributeError(f"'CSSInteractor' object has no attribute '{function_name}'")
                    arguments = _parse_arguments(function_name, line)
                    parts.extend(await self._run_action(state, function_name, arguments))
                    state["actions"].append(f"{function_name}({', '.join(arguments)})")
                except ScreenshotError:
                    raise  # the browser failed, not the model: fail the rollout instead of blaming the action
                except Exception as e:  # the model sees its own mistake, as upstream reports it
                    parts.append(
                        _text(
                            f"{function_name}({', '.join(arguments)}) is not valid due to exception {e}. "
                            "You may make a mistake and need to fix it."
                        )
                    )
                    break
            return parts
        return "No executable function found! Need to recheck the action. Please stick to the format of Action: ..."

    async def _run_action(self, state: dict, function_name: str, arguments: list[str]) -> list[dict]:
        interactor: CSSInteractor = state["interactor"]
        if function_name == "get_selectors_by_html_elements":
            elements, selectors = await asyncio.to_thread(interactor.get_selectors_by_html_elements, arguments[0])
            if not elements:
                return [
                    _text(
                        "No html elements found based on the input. Please specify a different html elements "
                        "specification."
                    )
                ]
            if not selectors:
                return [
                    _text(
                        "No matched selectors found based on the html elements. Please specify a different html elements."
                    )
                ]
            listing = "".join(f"{i + 1}. '{selector}'\n" for i, selector in enumerate(selectors))
            return [
                _text(
                    f"The following {len(selectors)} selectors are matched based on the returned html elements:\n{listing}"
                )
            ]
        if function_name == "select_rule":
            rule, _ = interactor.select_rule(arguments[0])
            if rule is None:
                return [_text(f"No rule found based on selector {arguments[0]}. Please specify a different selector.")]
            properties = {name: rule.style[name] for name in rule.style.keys()}
            return [
                _text(
                    f"Here is the rule founded based on selector {arguments[0]}: {properties}. "
                    "Is this the rule you want to edit?"
                )
            ]
        # edit_rule
        if not await asyncio.to_thread(interactor.edit_rule, *arguments):
            return [
                _text(
                    f"No rule found based on selector {arguments[0]}. Please specify a different selector or "
                    "consider inserting a new rule."
                )
            ]
        screenshot = await self._fresh_screenshot(state)
        return [
            _text(
                "You have successfully edited the rule. Here is the updated screenshot. If you think this edit is not "
                "useful, you can revert the last edit by calling revert_last_edit()."
            ),
            _image(screenshot),
        ]

    async def _fresh_screenshot(self, state: dict) -> Path:
        path = state["task_dir"] / SCREENSHOT_NAME
        path.unlink(missing_ok=True)
        await self.take_screenshot(state["task_dir"] / "index.html", path)
        return path

    async def _finish(self, state: dict, *, terminated: bool) -> tuple[None, float, bool, bool, dict]:
        """Score the last screenshot taken; like upstream, a revert does not take a new one."""
        final = state["task_dir"] / SCREENSHOT_NAME
        if not final.exists():
            await self.take_screenshot(state["task_dir"] / "index.html", final)
        distance = 1 - await asyncio.to_thread(compare_images_ssim_gray, state["task_dir"] / "index.png", final)
        success = distance < self.config.success_distance
        info = {
            "success": success,
            "distance": distance,
            "initial_distance": state["initial_distance"],
            "improved": distance < state["initial_distance"] - self.config.improve_margin,
            "rounds": state["rounds"],
            "fixed_claimed": terminated,
            "actions": state["actions"],
        }
        return None, 1.0 if success else 0.0, terminated, not terminated, info

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/close")(self._close_endpoint)
        return app

    async def _close_endpoint(self, request: Request) -> dict[str, bool]:
        await self.close_session(request.session.get(SESSION_ID_KEY))
        self.checkpoint_session_ended(request)
        return {"ok": True}

    def _cleanup_session(self, session_id: Optional[str]) -> None:
        """Forget the session now; remove its tree once no step it started is still running (and writing)."""
        state = self.session_state.pop(session_id, None)
        if state is None:
            return
        running = set(state["running"])
        if not running:
            shutil.rmtree(state["work_root"], ignore_errors=True)
            return

        def remove_after_last(task: asyncio.Future) -> None:
            running.discard(task)
            if not running:
                shutil.rmtree(state["work_root"], ignore_errors=True)

        for task in running:
            task.add_done_callback(remove_after_last)


async def _route_request(route: Any, *, task_dir: Path) -> None:
    if _request_allowed(route.request.url, task_dir):
        await route.continue_()
    else:
        LOGGER.warning("Blocked a VAB-CSS page request to %s", route.request.url)
        await route.abort("blockedbyclient")


def _request_allowed(url: str, task_dir: Path) -> bool:
    """Files under the task folder and inline data only.

    The CSS is model-written, and a DNS check here cannot bind Chromium's own connection, so no network request
    is allowed; the warning above names any remote asset a task tries to load.
    """
    parts = urlsplit(url)
    if parts.scheme == "file":
        return Path(unquote(parts.path)).resolve().is_relative_to(task_dir)
    return parts.scheme in ("data", "blob")


def _parse_arguments(function_name: str, line: str) -> list[str]:
    """Upstream's parsing: shlex for most actions, the raw text for get_selectors_by_html_elements."""
    if function_name == "get_selectors_by_html_elements":
        arguments = re.findall(r"get_selectors_by_html_elements\((.+?)\)", line)[0]
        if arguments.startswith("'") and arguments.endswith("'"):
            arguments = arguments[1:-1]
        if arguments.startswith('"') and arguments.endswith('"'):
            arguments = arguments[1:-1]
        return [arguments]
    matches = re.findall(r"{}\((.*?)\)".format(re.escape(function_name)), line)
    return [arg.rstrip(",") for arg in shlex.split(matches[0])]


def _text(text: str) -> dict:
    return {"type": "input_text", "text": text}


def _image(path: Path) -> dict:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "input_image", "image_url": f"data:image/png;base64,{data}", "detail": "high"}


if __name__ == "__main__":
    VabCssEnv.run_webserver()
