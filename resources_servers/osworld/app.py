# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""OSWorld resources server (computer-use benchmark) on the ``nemo_gym.sandbox`` SDK.

Per rollout (session), this server:

1. ``/seed_session`` — allocates an OSWorld desktop VM through ``AsyncSandbox``
   (OpenSandbox KVM pool or AgentEnv snapshot), resolves the guest ``:5000``
   control endpoint via ``AsyncSandbox.endpoint(5000)``, waits for the desktop to render, and runs
   the task's setup with the OFFICIAL harness semantics — a ``eval_task.py --phase setup``
   subprocess that imports the pinned ``osworld`` fork and calls ``DesktopEnv.reset``.
   With AgentEnv fork oversampling, setup runs once and the active generation
   siblings are cloned from that exact initialized state.
2. Exposes the two agent-facing tools: ``/screenshot`` (pixel observation) and ``/execute``
   (shell / python command in the guest — the OSWorld action modality).
3. ``/verify`` — scores with the COMPLETE upstream evaluator (``eval_task.py --phase
   evaluate`` -> ``DesktopEnv.evaluate()``; the caller always evaluates, including at step
   exhaustion), then **always** releases the sandbox.

Setup/evaluate run in subprocesses because the fork's remote-provider addressing is
env-var-global (``OSWORLD_CONTROL_SERVER_URL``/``OSWORLD_REMOTE_ADDR``); concurrent
sessions must not share a process. Guest HTTP goes through NeMo Gym's global aiohttp
client; the ``_guest_request`` / ``_run_task_phase`` seams are what unit tests mock.
"""

import asyncio
import base64
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from nemo_gym.server_utils import request as gym_request


logger = logging.getLogger("nemo_gym.osworld.resources_server")

# eval_task.py prints exactly one sentinel line to stdout per invocation.
_TASK_SENTINEL = "__NEMO_GYM_OSWORLD__"
_EVAL_TASK_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_task.py")

# Guest-environment prep run once at boot. OSWorld's fine-interaction tasks (LibreOffice
# context menus, menu->dialog flows) fail systematically when a focus-stealing popup —
# Ubuntu's "Software Updates Ready to Install" (update-notifier) or a notification banner —
# grabs focus and dismisses the just-opened menu/dialog (a documented OSWorld gotcha class,
# cf. the Snap-Store popup). Kill update-notifier, silence notification banners, disable the
# screensaver/idle lock, and press Escape to clear any already-open popup. Best-effort.
# After a task's setup opens an office app, the window's bottom edge (LibreOffice sheet tabs +
# '+' add-sheet button + status bar) falls below the 1080 screen (default geometry
# 70,101/1850x1016), blocking tab-bar access. Fit it via the EWMH maximize STATE:
# `wmctrl -e` (move/resize) is a verified no-op on these windows under this WM (probe3 manual,
# v9 immediate, v9b delayed double-tap — live mid-rollout geometry unchanged every time), while
# `-b add,maximized_*` provably changes the window (v5). Its two historical side effects are
# both independently fixed: the transient blank view is absorbed by the 10s post-setup settle
# that runs AFTER this script, and the Get-Involved/Donate info bars are disabled by the
# FirstRun=false boot prep. Double-tap to cover LibreOffice's late geometry restore while the
# document finishes loading. `pkill -f` uses the [u] bracket so the pattern can't match this
# script's own `sh -c` cmdline and kill it (the unbracketed form self-terminated at line 1).
_MAXIMIZE_OFFICE_SCRIPT = (
    "pkill -f '[u]pdate-notifier' 2>/dev/null; "
    "export DISPLAY=:0; "
    "fit() { wmctrl -lG 2>/dev/null | grep -iE 'libreoffice|calc|writer|impress|draw|gimp' | "
    "while read wid rest; do "
    'wmctrl -i -r "$wid" -b add,maximized_vert,maximized_horz 2>/dev/null; done; }; '
    "sleep 4; fit; sleep 3; fit; sleep 2; "
    "echo WINDOW_FITTED; wmctrl -lG 2>/dev/null | grep -iE 'libreoffice|calc|writer|impress|draw|gimp'"
)

# Always run (independent of the OOD popup/window prep): disable LibreOffice's "Keep Current
# Format?" alien-format warning so the evaluator postconfig's ctrl+s saves xlsx SILENTLY. Without
# it the modal blocks the save, the agent's number-FORMAT changes (0.00, currency, dates) never
# reach disk, and the CSV the evaluator compares shows raw values → format/display tasks score 0
# despite being visually correct (root-caused by trajectory replay vs Jeff's per-task rewards;
# the guest qcow2 == official image, which ships this warning ON). Pure save-correctness, not an
# agent-visible change. Live-verified: dialog gone, 0.00 format persists to the saved xlsx.
_SAVE_CONFIG_SCRIPT = (
    "export DISPLAY=:0; "
    "XCU=$HOME/.config/libreoffice/4/user/registrymodifications.xcu; "
    'mkdir -p "$(dirname "$XCU")"; '
    '[ -f "$XCU" ] || printf \'<?xml version="1.0" encoding="UTF-8"?>\\n'
    '<oor:items xmlns:oor="http://openoffice.org/2001/registry">\\n</oor:items>\\n\' > "$XCU"; '
    'grep -q WarnAlienFormat "$XCU" || sed -i '
    '\'s#</oor:items>#<item oor:path="/org.openoffice.Office.Common/Save/Document">'
    '<prop oor:name="WarnAlienFormat" oor:op="fuse"><value>false</value></prop></item>'
    '</oor:items>#\' "$XCU"; '
    # LibreOffice reads registrymodifications only at process start. A boot-time quickstarter
    # (or any soffice already up) would keep the warning ON in memory, so kill it — the task's
    # own open_file then launches a fresh instance that picks up WarnAlienFormat=false.
    "pkill -x soffice.bin 2>/dev/null; pkill -f 'oosplash' 2>/dev/null; echo SAVE_CONFIG_DONE"
)

# An agent can legitimately finish with the cursor at (0, 0). Evaluator postconfig
# commands commonly invoke pyautogui next (for example, ctrl+s), whose default
# fail-safe then aborts the evaluator and incorrectly converts the sample to reward
# zero. Reset only the pointer immediately before evaluation; disabling FAILSAFE is
# scoped to this short-lived helper process and does not affect agent actions.
_CENTER_POINTER_SCRIPT = (
    'python3 -c "import pyautogui; pyautogui.FAILSAFE=False; '
    'width, height = pyautogui.size(); pyautogui.moveTo(width // 2, height // 2)"'
)

_GUEST_PREP_SCRIPT = (
    "pkill -f '[u]pdate-notifier' 2>/dev/null; "
    "pkill -f '[u]pdate-manager' 2>/dev/null; "
    "export DISPLAY=:0; export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus; "
    "gsettings set org.gnome.desktop.notifications show-banners false 2>/dev/null; "
    "gsettings set org.gnome.desktop.screensaver lock-enabled false 2>/dev/null; "
    "gsettings set org.gnome.desktop.session idle-delay 0 2>/dev/null; "
    # Disable LibreOffice's first-run / Get-Involved-Donate info bars + Tip-of-the-Day, which
    # otherwise appear at the top of the doc, shift the whole grid down, and misalign the
    # agent's learned cell coordinates. Written before any task opens a doc; append-if-absent
    # to the existing registrymodifications so other settings are preserved.
    "XCU=$HOME/.config/libreoffice/4/user/registrymodifications.xcu; "
    'mkdir -p "$(dirname "$XCU")"; '
    '[ -f "$XCU" ] || printf \'<?xml version="1.0" encoding="UTF-8"?>\\n'
    '<oor:items xmlns:oor="http://openoffice.org/2001/registry">\\n</oor:items>\\n\' > "$XCU"; '
    # FirstRun/TipOfTheDay do NOT control the "Get Involved" / "Donate" info bars (live-verified
    # in calc-v10: bars present with FirstRun=false written). Those are governed by the
    # LastTimeGetInvolvedShown / LastTimeDonateShown epochs — set them far in the future
    # (2100-01-01) so the bars never fire. The bars animate in a few seconds after the document
    # opens, shifting the grid ~80px mid-think and desyncing the agent's click coordinates.
    'grep -q \'"FirstRun"\' "$XCU" || sed -i '
    '\'s#</oor:items>#<item oor:path="/org.openoffice.Office.Common/Misc">'
    '<prop oor:name="FirstRun" oor:op="fuse"><value>false</value></prop></item>'
    '<item oor:path="/org.openoffice.Office.Common/Misc"><prop oor:name="ShowTipOfTheDay" '
    'oor:op="fuse"><value>false</value></prop></item>'
    '<item oor:path="/org.openoffice.Office.Common/Misc"><prop oor:name="LastTimeGetInvolvedShown" '
    'oor:op="fuse"><value>4102444800</value></prop></item>'
    '<item oor:path="/org.openoffice.Office.Common/Misc"><prop oor:name="LastTimeDonateShown" '
    'oor:op="fuse"><value>4102444800</value></prop></item>'
    '</oor:items>#\' "$XCU"; '
    "python3 -c \"import pyautogui; pyautogui.FAILSAFE=False; pyautogui.press('escape')\" 2>/dev/null; "
    "echo GUEST_PREPPED"
)


class OSWorldResourcesServerConfig(BaseResourcesServerConfig):
    # A sandbox block name resolved from the merged global config (e.g. "sandbox" from
    # nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml), or an inline
    # single-key provider mapping ({provider_name: {...}}).
    sandbox_provider: Optional[Union[str, Dict[str, Any]]] = None
    pool_ref: str = "osworld-kvm"
    # OpenSandbox SDK 0.1.15 requires an image even when poolRef selects the
    # prebuilt OSWorld KVM image. The pool, not this placeholder, supplies the VM.
    sandbox_image: str = Field(default="busybox:1.36", min_length=1)
    sandbox_ttl_s: int = 7200
    boot_wait_s: int = 600
    # Retry seed_session with a fresh sandbox on boot/setup failure (a single VM occasionally
    # fails to boot in time under concurrency; without retry that 500 crashes the whole run).
    alloc_retries: int = 3
    poll_interval_s: float = 10.0
    # The desktop renders once the screenshot PNG grows past ~500KB (a black, un-rendered
    # screen is a tiny ~6KB PNG). Validated on the OpenSandbox runc-kvm OSWorld image.
    screenshot_min_bytes: int = 500_000
    client_password: str = "password"
    request_retries: int = 5
    # Silence guest focus-stealers (update-notifier popup / notification banners) at boot.
    # DEFAULT OFF for reference parity: neither Yi's runner, the fork, nor upstream OSWorld
    # suppresses these artifacts — the model was evidently trained/evaluated with them present,
    # and calc-v10 (interventions ON) collapsed versus the intervention-free runs. Note the
    # script also self-terminated (pkill self-match, fixed in 264ae2c9) in every run before
    # 2026-07-16, so all historical results — including the in-band os/vs_code/gimp domains —
    # were produced WITHOUT any guest prep.
    # Make xlsx saves silent (WarnAlienFormat off) so the agent's formatting reaches disk before
    # evaluation. Save-correctness, not agent-visible — default ON.
    save_config_fix: bool = True
    prepare_guest_environment: bool = False
    # Settle time between task setup and the first agent observation. The reference kimi runner
    # sleeps 10s; other upstream OSWorld runners sleep 20-60s. On OUR (slower, 4-core KVM) VMs a
    # timing probe shows the LibreOffice layout reaches its stable state (document painted,
    # toolbars loaded, Get-Involved/Donate info bars animated in) at t≈11-14s after setup's
    # open_file — 10s sat exactly on the transition boundary, so first screenshots randomly
    # caught either layout and the agent's coordinates went stale mid-think. 20s = measured
    # stabilization + margin, within the upstream-sanctioned range.
    post_setup_settle_s: float = 20.0
    # Fit the task's office window to the screen post-setup (sheet tabs/status bar on-screen).
    # DEFAULT OFF for reference parity: the reference stack does no window management, and
    # calc-v10 (maximize ON) collapsed — the maximized layout is out-of-distribution for the
    # model. Kept as an opt-in experiment knob; see _MAXIMIZE_OFFICE_SCRIPT.
    maximize_office_windows: bool = False
    # Setup/evaluate subprocess knobs (the pinned `osworld` fork lives in this server's venv).
    cache_dir: str = "/tmp/osworld_eval_cache"
    setup_timeout_s: int = 900
    eval_timeout_s: int = 1200
    # Reference parity: the runner sleeps 10s between the agent's last action and
    # env.evaluate() (lib_run_single "Wait for the environment to settle").
    pre_verify_settle_s: float = 10.0
    # AgentEnv oversampling: initialize one task, then fork the active logical
    # owners from that exact state. This does not create extra trajectories.
    fork_oversampling: bool = False
    fork_claim_timeout_s: float = 7200.0


class OSWorldSeedSessionRequest(BaseSeedSessionRequest):
    # The agent forwards the whole dataset row to /seed_session, so accept extras and
    # pull ``verifier_metadata`` (id / config / evaluator / instruction) from it.
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Optional[Dict[str, Any]] = None


class OSWorldSeedSessionResponse(BaseSeedSessionResponse):
    model_config = ConfigDict(extra="allow")
    sandbox_id: Optional[str] = None
    screen: Optional[Dict[str, int]] = None


class ExecuteToolRequest(BaseModel):
    command: Union[str, List[str]]
    shell: bool = True


class ScreenshotResponse(BaseModel):
    image_base64: str


class OSWorldVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Optional[Dict[str, Any]] = None
    # The agent's faithful OSWorld action history (the evaluator inspects the LAST entry:
    # "FAIL" drives the `infeasible` metric and short-circuits others).
    action_history: Optional[List[str]] = None


class OSWorldVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    # Non-None when the 0 reward is (or may be) an infrastructure artifact rather than a
    # judged failure: evaluate-phase error, scoring exception, or a session that never got
    # a sandbox. Lets a post-run pass identify poisoned rows and re-run exactly those.
    verify_error: Optional[str] = None


@dataclass
class _HTTPResult:
    status: int
    content: bytes
    content_type: str = ""

    def json(self) -> Any:
        return json.loads(self.content) if self.content else None

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


@dataclass
class _ForkGroup:
    active_indices: tuple[int, ...]
    preparation: asyncio.Task[Dict[int, dict]]
    states: Optional[Dict[int, dict]] = None
    expiry: Optional[asyncio.Task[None]] = None


class OSWorldResourcesServer(SimpleResourcesServer):
    config: OSWorldResourcesServerConfig
    # Per-session state: session_id -> {"sandbox": AsyncSandbox, "sandbox_id", "control_url",
    # "headers", "proxied", "screen"}.
    session_id_to_sandbox: Dict[str, dict] = Field(default_factory=dict)
    _provider_config: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _provider_metadata: Optional[Dict[str, str]] = PrivateAttr(default=None)
    _fork_groups: Dict[str, _ForkGroup] = PrivateAttr(default_factory=dict)
    _fork_groups_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/screenshot")(self.screenshot)
        app.post("/execute")(self.execute)
        app.post("/release")(self.release)

        return app

    # ---------------------------------------------------------------------------------
    # Sandbox provider plumbing (nemo_gym.sandbox SDK)
    # ---------------------------------------------------------------------------------

    def _resolved_provider(self) -> tuple[Dict[str, Any], Dict[str, str]]:
        """Resolve ``config.sandbox_provider`` once (name reference needs the global config)."""
        if self._provider_config is None:
            if self.config.sandbox_provider is None:
                raise ValueError("osworld resources server requires `sandbox_provider`")
            named_configs: Optional[Dict[str, Any]] = None
            if isinstance(self.config.sandbox_provider, str):
                named_configs = ServerClient.load_from_global_config().global_config_dict
            self._provider_config = resolve_provider_config(self.config.sandbox_provider, named_configs)
            self._provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, named_configs)
        return self._provider_config, self._provider_metadata or {}

    def _sandbox_spec(self) -> SandboxSpec:
        provider_config, provider_metadata = self._resolved_provider()
        provider_name = next(iter(provider_config))
        if provider_name == "agentenv":
            return SandboxSpec(
                ttl_s=self.config.sandbox_ttl_s,
                ports=(5000, 9222, 8080),
                metadata={**provider_metadata, "purpose": "osworld"},
            )
        return SandboxSpec(
            image=self.config.sandbox_image,
            ttl_s=self.config.sandbox_ttl_s,
            ports=(5000,),
            metadata={**provider_metadata, "purpose": "osworld"},
            provider_options={"extensions": {"poolRef": self.config.pool_ref}},
        )

    async def _sandbox_state(self, sandbox: AsyncSandbox) -> dict:
        """Resolve the service routes needed by the guest API and evaluator."""
        provider_name = sandbox._require_handle().provider_name
        ports = (5000, 9222, 8080) if provider_name == "agentenv" else (5000,)
        endpoints = {port: await sandbox.endpoint(port) for port in ports}
        control = endpoints[5000]
        path = urlsplit(control.endpoint).path.rstrip("/")
        proxied = provider_name == "agentenv" or "/proxy/" in path or path.endswith("/proxy")
        state = {
            "sandbox": sandbox,
            "sandbox_id": sandbox._require_handle().sandbox_id,
            "control_url": control.endpoint.rstrip("/"),
            "headers": dict(control.headers),
            "proxied": proxied,
        }
        if len(endpoints) > 1:
            state["service_endpoints"] = {
                str(port): {
                    "endpoint": endpoint.endpoint.rstrip("/"),
                    "headers": dict(endpoint.headers),
                }
                for port, endpoint in endpoints.items()
            }
        return state

    async def _allocate_sandbox(self) -> dict:
        """Allocate one pool VM through the SDK and resolve its guest :5000 endpoint."""
        provider_config, _ = self._resolved_provider()
        sandbox = AsyncSandbox(provider_config, self._sandbox_spec())
        await sandbox.start()
        try:
            return await self._sandbox_state(sandbox)
        except Exception:
            await self._release({"sandbox": sandbox, "sandbox_id": "?"})
            raise

    async def _release(self, sandbox_state: dict) -> None:
        try:
            await sandbox_state["sandbox"].stop()
        except Exception as e:  # noqa: BLE001 - releasing must never mask the real result
            logger.warning("Failed to release OSWorld sandbox %s: %r", sandbox_state.get("sandbox_id"), e)

    async def _release_session(self, session_id: str) -> bool:
        """Idempotently remove and release one session-owned sandbox."""

        sandbox_state = self.session_id_to_sandbox.pop(session_id, None)
        if sandbox_state is None:
            return False
        await self._release(sandbox_state)
        return True

    # ---------------------------------------------------------------------------------
    # Guest :5000 HTTP (single mockable seam)
    # ---------------------------------------------------------------------------------

    async def _guest_request(
        self,
        sandbox_state: dict,
        method: str,
        path: str,
        *,
        json_body: Optional[Any] = None,
    ) -> _HTTPResult:
        """Guest control-API request with a transient-failure retry guard.

        Retries on (a) transient network exceptions (e.g. a local ``kubectl port-forward``
        tunnel dropping mid-run), (b) an empty response body, and (c) 5xx from the proxy.
        """
        url = f"{sandbox_state['control_url']}{path}"
        route_headers = dict(sandbox_state.get("headers") or {})

        result: Optional[_HTTPResult] = None
        last_exc: Optional[Exception] = None
        retries = max(1, self.config.request_retries)
        for attempt in range(retries):
            # Fresh kwargs per attempt: gym_request mutates headers (adds Content-Type for json).
            kwargs: Dict[str, Any] = {}
            if route_headers:
                kwargs["headers"] = dict(route_headers)
            if json_body is not None:
                kwargs["json"] = json_body
            try:
                resp = await gym_request(method, url, **kwargs)
                content = await resp.read()
                result = _HTTPResult(
                    status=resp.status,
                    content=content,
                    content_type=resp.headers.get("Content-Type", ""),
                )
            except Exception as e:  # noqa: BLE001 - transient tunnel/proxy failures are retryable
                last_exc = e
                logger.warning("guest request %s %s failed (attempt %d/%d): %r", method, url, attempt + 1, retries, e)
                await asyncio.sleep(min(2.0 * (attempt + 1), 8.0))
                continue
            if result.content and result.status < 500:
                return result
            await asyncio.sleep(min(2.0 * (attempt + 1), 8.0))
        if result is not None:
            return result
        raise last_exc if last_exc is not None else RuntimeError(f"guest request failed: {method} {url}")

    async def _guest_execute(self, sandbox_state: dict, command: Any, shell: bool) -> Dict[str, Any]:
        res = await self._guest_request(
            sandbox_state, "POST", "/execute", json_body={"command": command, "shell": shell}
        )
        try:
            return res.json() or {"output": res.text}
        except (json.JSONDecodeError, ValueError):
            return {"output": res.text}

    async def _guest_screen_size(self, sandbox_state: dict) -> Dict[str, int]:
        try:
            res = await self._guest_request(sandbox_state, "POST", "/screen_size", json_body={})
            data = res.json()
            if isinstance(data, dict):
                return {
                    "width": int(data.get("width") or 1920),
                    "height": int(data.get("height") or 1080),
                }
        except Exception as e:  # noqa: BLE001 - cosmetic; default is the OSWorld standard
            logger.warning("Could not read guest screen size, defaulting to 1920x1080: %r", e)
        return {"width": 1920, "height": 1080}

    async def _wait_for_boot(self, sandbox_state: dict) -> None:
        # 1) Wait for the guest control API (:5000) to answer.
        deadline = time.monotonic() + self.config.boot_wait_s
        platform_up = False
        while time.monotonic() < deadline:
            try:
                res = await self._guest_request(sandbox_state, "GET", "/platform")
                if res.status == 200:
                    platform_up = True
                    break
            except Exception:  # noqa: BLE001 - keep polling until the deadline
                pass
            await asyncio.sleep(self.config.poll_interval_s)
        if not platform_up:
            raise RuntimeError(f"OSWorld guest :5000 not reachable within {self.config.boot_wait_s}s")

        # 2) Wait for the desktop to actually render (screenshot grows past the threshold).
        deadline = time.monotonic() + self.config.boot_wait_s
        rendered = False
        while time.monotonic() < deadline:
            try:
                res = await self._guest_request(sandbox_state, "GET", "/screenshot")
                if len(res.content) > self.config.screenshot_min_bytes:
                    rendered = True
                    break
            except Exception:  # noqa: BLE001 - keep polling until the deadline
                pass
            await asyncio.sleep(self.config.poll_interval_s)
        if not rendered:
            logger.warning(
                "OSWorld desktop did not exceed %d bytes; proceeding anyway", self.config.screenshot_min_bytes
            )

        # 3) Always: make xlsx saves silent so the agent's formatting persists (see
        #    _SAVE_CONFIG_SCRIPT). Independent of the OOD popup/window prep below.
        if self.config.save_config_fix:
            try:
                res = await self._guest_execute(sandbox_state, _SAVE_CONFIG_SCRIPT, shell=True)
                logger.info("Save-config applied (WarnAlienFormat off): %r", str(res)[:80])
            except Exception as e:  # noqa: BLE001 - best-effort; never fail the rollout
                logger.warning("Could not apply save-config (proceeding): %r", e)

        # 4) Optional (default off, OOD): silence focus-stealers / popups.
        if self.config.prepare_guest_environment:
            await self._prepare_guest_environment(sandbox_state)

    async def _prepare_guest_environment(self, sandbox_state: dict) -> None:
        try:
            res = await self._guest_execute(sandbox_state, _GUEST_PREP_SCRIPT, shell=True)
            logger.info("Guest environment prepped (focus-stealers silenced): %r", str(res)[:120])
        except Exception as e:  # noqa: BLE001 - best-effort; never fail the rollout
            logger.warning("Could not prep guest environment (proceeding): %r", e)

    async def _maximize_office_windows(self, sandbox_state: dict) -> None:
        try:
            res = await self._guest_execute(sandbox_state, _MAXIMIZE_OFFICE_SCRIPT, shell=True)
            # The script echoes WINDOW_FITTED + the post-move `wmctrl -lG` line(s): keep them in
            # the log as per-seed evidence the fit actually took effect.
            logger.info("Office window fit post-setup: %r", str(res)[:240])
        except Exception as e:  # noqa: BLE001 - best-effort; never fail the rollout
            logger.warning("Could not fit office windows (proceeding): %r", e)

    async def _center_pointer_for_evaluation(self, sandbox_state: dict) -> None:
        """Keep evaluator pyautogui postconfig from inheriting a fail-safe corner."""
        try:
            result = await self._guest_execute(sandbox_state, _CENTER_POINTER_SCRIPT, shell=True)
            returncode = result.get("returncode")
            if returncode not in (None, 0):
                logger.warning(
                    "Could not center pointer before evaluation (returncode=%s): %r",
                    returncode,
                    str(result)[:240],
                )
        except Exception as e:  # noqa: BLE001 - best-effort; evaluation still owns scoring
            logger.warning("Could not center pointer before evaluation (proceeding): %r", e)

    # ---------------------------------------------------------------------------------
    # Setup/evaluate subprocesses (official harness semantics via the pinned fork)
    # ---------------------------------------------------------------------------------

    async def _run_task_phase(
        self,
        phase: str,
        task_config: Dict[str, Any],
        sandbox_state: dict,
        *,
        action_history: Optional[List[str]] = None,
        timeout_s: float = 900.0,
    ) -> Dict[str, Any]:
        """Run one ``eval_task.py`` phase; returns the parsed sentinel dict."""
        os.makedirs(self.config.cache_dir, exist_ok=True)
        screen = sandbox_state.get("screen") or {"width": 1920, "height": 1080}
        fd, cfg_path = tempfile.mkstemp(suffix=".json", prefix=f"osw_{phase}_")
        with os.fdopen(fd, "w") as f:
            json.dump(task_config, f)
        headers_fd, headers_path = tempfile.mkstemp(suffix=".json", prefix="osw_headers_")
        os.chmod(headers_path, 0o600)
        with os.fdopen(headers_fd, "w") as f:
            json.dump(sandbox_state.get("headers") or {}, f)
        service_endpoints_path: Optional[str] = None
        if sandbox_state.get("service_endpoints"):
            service_fd, service_endpoints_path = tempfile.mkstemp(suffix=".json", prefix="osw_service_endpoints_")
            os.chmod(service_endpoints_path, 0o600)
            with os.fdopen(service_fd, "w") as f:
                json.dump(sandbox_state["service_endpoints"], f)
        try:
            cmd = [
                sys.executable,
                _EVAL_TASK_SCRIPT,
                "--phase",
                phase,
                "--config",
                cfg_path,
                "--control-url",
                sandbox_state["control_url"],
                "--headers-file",
                headers_path,
                "--cache",
                self.config.cache_dir,
                "--action-history",
                json.dumps(action_history or []),
                "--screen-w",
                str(int(screen.get("width", 1920))),
                "--screen-h",
                str(int(screen.get("height", 1080))),
                "--client-password",
                self.config.client_password,
            ]
            if service_endpoints_path is not None:
                cmd.extend(
                    [
                        "--service-endpoints-file",
                        service_endpoints_path,
                    ]
                )
            if sandbox_state.get("proxied"):
                cmd.append("--proxied")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return {"ok": False, "score": None, "error": f"{phase}_timeout_{timeout_s:g}s"}

            for line in out.decode("utf-8", errors="replace").splitlines():
                if line.startswith(_TASK_SENTINEL):
                    result = json.loads(line[len(_TASK_SENTINEL) :].strip())
                    if not result.get("ok"):
                        # Surface the subprocess's actual traceback — a bare
                        # "{phase}_exception" has proven undiagnosable from the run log.
                        tail = err.decode("utf-8", errors="replace")[-1200:]
                        logger.warning(
                            "eval_task %s failed (task=%s, error=%r); stderr tail:\n%s",
                            phase,
                            task_config.get("id", "?"),
                            result.get("error"),
                            tail,
                        )
                    return result
            tail = err.decode("utf-8", errors="replace")[-800:]
            # returncode disambiguates a crash (<0 = signal: -9 OOM-kill, -11 segfault)
            # from a clean exit that skipped the sentinel.
            logger.warning(
                "eval_task %s produced no sentinel line (returncode=%s); stderr tail:\n%s",
                phase,
                proc.returncode,
                tail,
            )
            return {"ok": False, "score": None, "error": f"{phase}_no_sentinel_rc{proc.returncode}"}
        finally:
            try:
                os.unlink(cfg_path)
            except OSError:
                pass
            try:
                os.unlink(headers_path)
            except OSError:
                pass
            if service_endpoints_path is not None:
                try:
                    os.unlink(service_endpoints_path)
                except OSError:
                    pass

    # ---------------------------------------------------------------------------------
    # Session helpers
    # ---------------------------------------------------------------------------------

    def _get_session_sandbox(self, request: Request) -> dict:
        session_id = request.session[SESSION_ID_KEY]
        sandbox_state = self.session_id_to_sandbox.get(session_id)
        if sandbox_state is None:
            raise HTTPException(
                status_code=400, detail="No OSWorld sandbox for this session; call /seed_session first."
            )
        return sandbox_state

    async def _initialize_sandbox(self, verifier_metadata: Dict[str, Any]) -> dict:
        """Allocate and initialize one task, retrying fresh infrastructure."""
        last_err: Optional[Exception] = None
        attempts = max(1, self.config.alloc_retries)
        for attempt in range(attempts):
            sandbox_state: Optional[dict] = None
            try:
                sandbox_state = await self._allocate_sandbox()
                logger.info(
                    "Allocated OSWorld sandbox %s (attempt %d/%d)",
                    sandbox_state["sandbox_id"],
                    attempt + 1,
                    attempts,
                )
                await self._wait_for_boot(sandbox_state)
                sandbox_state["screen"] = await self._guest_screen_size(sandbox_state)
                if verifier_metadata:
                    setup = await self._run_task_phase(
                        "setup",
                        verifier_metadata,
                        sandbox_state,
                        timeout_s=self.config.setup_timeout_s,
                    )
                    if not setup.get("ok"):
                        if attempt + 1 < attempts:
                            raise RuntimeError(f"task setup failed: {setup.get('error')!r}")
                        logger.warning(
                            "Task setup failed on the final attempt (%r); proceeding un-set-up",
                            setup.get("error"),
                        )
                if self.config.maximize_office_windows:
                    await self._maximize_office_windows(sandbox_state)
                if self.config.post_setup_settle_s > 0:
                    await asyncio.sleep(self.config.post_setup_settle_s)
                return sandbox_state
            except Exception as e:  # noqa: BLE001 - boot/setup can fail transiently
                last_err = e
                if sandbox_state is not None:
                    await self._release(sandbox_state)
                logger.warning(
                    "OSWorld initialization attempt %d/%d failed (%r); retrying fresh",
                    attempt + 1,
                    attempts,
                    e,
                )
        raise RuntimeError(f"seed_session failed after {attempts} attempts: {last_err!r}")

    async def _prepare_fork_group(
        self,
        verifier_metadata: Dict[str, Any],
        active_indices: tuple[int, ...],
    ) -> Dict[int, dict]:
        """Initialize once and fork every other active logical owner."""
        parent = await self._initialize_sandbox(verifier_metadata)
        states = [parent]
        try:
            if len(active_indices) > 1:
                children = await parent["sandbox"].fork(
                    len(active_indices) - 1,
                    ttl_s=self.config.sandbox_ttl_s,
                )
                for child in children:
                    state = await self._sandbox_state(child)
                    state["screen"] = dict(parent.get("screen") or {})
                    # A fork is expected to be immediately runnable, but verify
                    # the control service before handing it to a rollout.
                    platform = await self._guest_request(state, "GET", "/platform")
                    if platform.status != 200:
                        raise RuntimeError(
                            f"forked OSWorld sandbox {state['sandbox_id']} returned {platform.status} from /platform"
                        )
                    states.append(state)
        except Exception:
            await asyncio.gather(*(self._release(state) for state in states), return_exceptions=True)
            raise
        return dict(zip(active_indices, states, strict=True))

    async def _expire_fork_group(self, key: str, group: _ForkGroup) -> None:
        await asyncio.sleep(self.config.fork_claim_timeout_s)
        async with self._fork_groups_lock:
            if self._fork_groups.get(key) is not group:
                return
            self._fork_groups.pop(key, None)
            unclaimed = list((group.states or {}).values())
            group.states = {}
        if unclaimed:
            logger.warning("Releasing %d unclaimed AgentEnv forks for group %s", len(unclaimed), key)
            await asyncio.gather(*(self._release(state) for state in unclaimed), return_exceptions=True)

    async def _claim_forked_sandbox(
        self,
        body: OSWorldSeedSessionRequest,
        verifier_metadata: Dict[str, Any],
    ) -> dict:
        extras = body.model_extra or {}
        group_id = extras.get("_ng_group_id")
        group_attempt = extras.get("_ng_group_attempt", 0)
        rollout_index = extras.get("_ng_rollout_index")
        raw_indices = extras.get("_ng_active_generation_indices")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("fork oversampling requires _ng_group_id")
        if not isinstance(group_attempt, int) or isinstance(group_attempt, bool) or group_attempt < 0:
            raise ValueError("fork oversampling requires a non-negative _ng_group_attempt")
        if not isinstance(rollout_index, int) or isinstance(rollout_index, bool) or rollout_index < 0:
            raise ValueError("fork oversampling requires a non-negative _ng_rollout_index")
        if (
            not isinstance(raw_indices, list)
            or not raw_indices
            or any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in raw_indices)
        ):
            raise ValueError("fork oversampling requires _ng_active_generation_indices")
        active_indices = tuple(sorted(set(raw_indices)))
        if len(active_indices) != len(raw_indices) or rollout_index not in active_indices:
            raise ValueError("fork oversampling generation indices must be unique and contain this rollout")

        task_id = str(verifier_metadata.get("id") or "unknown")
        key = f"{group_id}:{group_attempt}:{task_id}"
        async with self._fork_groups_lock:
            group = self._fork_groups.get(key)
            if group is None:
                preparation = asyncio.create_task(self._prepare_fork_group(verifier_metadata, active_indices))
                group = _ForkGroup(active_indices=active_indices, preparation=preparation)
                self._fork_groups[key] = group
            elif group.active_indices != active_indices:
                raise ValueError(f"fork oversampling group {key} changed its active generation set")

        try:
            prepared = await asyncio.shield(group.preparation)
        except Exception:
            async with self._fork_groups_lock:
                if self._fork_groups.get(key) is group:
                    self._fork_groups.pop(key, None)
            raise

        async with self._fork_groups_lock:
            if group.states is None:
                group.states = dict(prepared)
            sandbox_state = group.states.pop(rollout_index, None)
            if sandbox_state is None:
                raise RuntimeError(f"fork oversampling generation {rollout_index} was already claimed for group {key}")
            if group.states:
                if group.expiry is None:
                    group.expiry = asyncio.create_task(self._expire_fork_group(key, group))
            else:
                self._fork_groups.pop(key, None)
                if group.expiry is not None:
                    group.expiry.cancel()
        return sandbox_state

    # ---------------------------------------------------------------------------------
    # Endpoints
    # ---------------------------------------------------------------------------------

    async def seed_session(self, request: Request, body: OSWorldSeedSessionRequest) -> OSWorldSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        verifier_metadata = body.verifier_metadata or {}
        provider_name = next(iter(self._resolved_provider()[0]))
        if self.config.fork_oversampling and provider_name == "agentenv":
            sandbox_state = await self._claim_forked_sandbox(body, verifier_metadata)
        else:
            sandbox_state = await self._initialize_sandbox(verifier_metadata)
        previous = self.session_id_to_sandbox.pop(session_id, None)
        if previous is not None:
            await self._release(previous)
        self.session_id_to_sandbox[session_id] = sandbox_state
        logger.info("Assigned OSWorld sandbox %s to session %s", sandbox_state["sandbox_id"], session_id)
        return OSWorldSeedSessionResponse(sandbox_id=sandbox_state["sandbox_id"], screen=sandbox_state.get("screen"))

    async def screenshot(self, request: Request) -> ScreenshotResponse:
        sandbox_state = self._get_session_sandbox(request)
        retries = max(1, self.config.request_retries)
        last_error = "unknown screenshot response"
        for attempt in range(retries):
            res = await self._guest_request(sandbox_state, "GET", "/screenshot")
            is_png = res.content.startswith(b"\x89PNG\r\n\x1a\n")
            if res.status == 200 and is_png:
                return ScreenshotResponse(image_base64=base64.b64encode(res.content).decode())
            last_error = (
                f"status={res.status} content_type={res.content_type!r} "
                f"bytes={len(res.content)} png_signature={is_png}"
            )
            logger.warning(
                "invalid OSWorld screenshot response (attempt %d/%d): %s",
                attempt + 1,
                retries,
                last_error,
            )
            if attempt + 1 < retries:
                await asyncio.sleep(min(2.0 * (attempt + 1), 8.0))
        raise RuntimeError(f"OSWorld guest returned no valid PNG screenshot after {retries} attempts: {last_error}")

    async def execute(self, request: Request, body: ExecuteToolRequest) -> Dict[str, Any]:
        sandbox_state = self._get_session_sandbox(request)
        return await self._guest_execute(sandbox_state, body.command, body.shell)

    async def release(self, request: Request) -> Dict[str, bool]:
        """Release without evaluating, for agent failures before the verifier call."""

        return {"released": await self._release_session(request.session[SESSION_ID_KEY])}

    async def verify(self, request: Request, body: OSWorldVerifyRequest) -> OSWorldVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        sandbox_state = self.session_id_to_sandbox.get(session_id)
        verifier_metadata = body.verifier_metadata or {}
        reward = 0.0
        verify_error: Optional[str] = None

        try:
            if sandbox_state is None:
                logger.warning("verify called without a seeded OSWorld sandbox for session %s", session_id)
                verify_error = "no_seeded_sandbox"
            elif not verifier_metadata.get("evaluator"):
                logger.warning("verify called without an evaluator in verifier_metadata")
                verify_error = "no_evaluator_in_verifier_metadata"
            else:
                # Reference parity: the runner sleeps 10s between the last agent action and
                # env.evaluate() ("Wait for the environment to settle",
                # lib_run_single.run_single_example_kimi). Without it, the evaluator's
                # postconfig (activate window -> ctrl+s -> convert) can capture the document
                # mid-commit (dialog closing, formula recalc) and score correct work as 0.
                if self.config.pre_verify_settle_s > 0:
                    await asyncio.sleep(self.config.pre_verify_settle_s)
                await self._center_pointer_for_evaluation(sandbox_state)
                result = await self._run_task_phase(
                    "evaluate",
                    verifier_metadata,
                    sandbox_state,
                    action_history=body.action_history or [],
                    timeout_s=self.config.eval_timeout_s,
                )
                if result.get("error"):
                    logger.warning(
                        "evaluate reported error %r (task=%s, score=%s)",
                        result["error"],
                        verifier_metadata.get("id", "?"),
                        result.get("score"),
                    )
                    verify_error = str(result["error"])
                reward = float(result.get("score") or 0.0)
        except Exception as e:  # noqa: BLE001 - scoring failures must not leak the sandbox
            logger.warning("OSWorld verify failed, scoring 0: %r", e)
            reward = 0.0
            verify_error = repr(e)
        finally:
            await self._release_session(session_id)

        return OSWorldVerifyResponse(**body.model_dump(), reward=float(reward), verify_error=verify_error)


if __name__ == "__main__":
    OSWorldResourcesServer.run_webserver()
