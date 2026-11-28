# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schemas for the CUA-Gym-Hub resources server.

The server speaks the exact seed/step/verify/close protocol that
``responses_api_agents/webarena_agent`` already uses against the WebArena server, so
all request/response models are reused from ``resources_servers.webarena``.
Task rows therefore validate as ``WebArenaTaskRow`` (the ``eval`` block is an
inert stub — verification is the row's inlined reward program) and carry their
CUA-Gym specifics in the extra ``cuagym`` field.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from resources_servers.cuagym.egress import normalize_host
from resources_servers.webarena.schemas import (
    WebArenaResourcesServerConfig,
    WebArenaTaskRow,
)


# Inert eval stub for rows: webarena_agent validates task rows as WebArenaTaskRow,
# whose eval.eval_types must be a known Classic WebArena evaluator. The cuagym
# server never reads it; reward comes from the row's inlined reward program.
CUAGYM_EVAL_STUB: dict[str, Any] = {
    "eval_types": ["string_match"],
    "reference_answers": None,
    "note": "unused — CUA-Gym reward code is authoritative",
}


class CuaGymTaskInfo(BaseModel):
    """The `cuagym` extra field carried inside a task row.

    ``initial_setup`` and ``eval_reward_code`` are self-contained Python
    programs produced by the dataset converter: sid appears only as the
    ``__CUA_GYM_SID__`` placeholder and hub endpoints as
    ``__CUA_GYM_<APP>_URL__`` placeholders, both filled by the server at
    episode time. ``bundle_id`` is provenance only.

    ``external_base_url`` lets a row name the deployment of an app that is not
    part of the hub (e.g. ``webarena_map_mock``). It is only a fallback: an
    ``endpoints`` entry in ``$CUA_GYM_ENDPOINTS_YAML`` always wins, so a moved
    deployment is a config change and never a data reconversion.
    """

    bundle_id: str
    app_dir: str
    eval_reward_code: str
    initial_setup: str | None = None
    external_base_url: str | None = None


class CuaGymResourcesServerConfig(WebArenaResourcesServerConfig):
    """WebArena browser-pool config plus CUA-Gym-Hub wiring.

    ``site_urls``/``credentials``/``judge_*`` fields are inherited but unused —
    the hub needs no login and rewards come from row-inlined programs.
    ``hub_base_url`` is substituted into those programs at episode time, so
    rotating the (ephemeral) hub IP is a one-line config change and never
    requires reconverting data.
    """

    hub_base_url: str = "http://localhost"
    hub_base_port: int = Field(default=8000, ge=1)
    node_placement_strategy: Literal["spread_all", "spread_gen"] = Field(
        default="spread_all",
        description=(
            "spread_all distributes browser actors across every Ray node; "
            "spread_gen restricts them to runtime generation nodes"
        ),
    )
    # Defaults must stay strictly inside the inherited browser budgets
    # (browser_operation_timeout_seconds=180, browser_evaluation_timeout_seconds=300);
    # see normalize_hub_config.
    setup_timeout_seconds: float = Field(default=90.0, gt=0)
    reward_timeout_seconds: float = Field(default=120.0, gt=0)
    terminate_properly_reward: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Reward bonus added when the agent ends the trajectory with a valid "
            "terminate tool call. Trajectories without valid termination skip "
            "reward evaluation and receive zero, except that webarena_agent scores a "
            "turn truncated at max output tokens as -terminate_properly_reward."
        ),
    )
    egress_allowlist: bool = Field(
        default=True,
        description=(
            "Launch Chromium with a host-resolver rule that fails DNS for every host except the hub "
            "(hub_base_url and every $CUA_GYM_ENDPOINTS_YAML endpoint), loopback, the map site's public "
            "dependencies and egress_allow_extra_hosts (see egress.py). A navigate/tabs_create to a blocked "
            "host returns the browser's error as a recoverable tool message. false = open internet (pre-2026-10-17)."
        ),
    )
    egress_allow_extra_hosts: list[str] = Field(
        default_factory=list,
        description="Extra hosts or patterns (e.g. '*.example.org', 'http://10.0.0.5:3000') for the egress allowlist.",
    )
    egress_exclude_endpoint_apps: list[str] = Field(
        default_factory=lambda: ["webarena_judge"],
        description=(
            "$CUA_GYM_ENDPOINTS_YAML entries that are not websites (reward-program services such as the judge "
            "proxy on the Ray head node) and stay out of the browser egress allowlist."
        ),
    )
    typing_backend: Literal["xkeymap", "pyautogui"] = Field(
        default="xkeymap",
        description=(
            "How the computer tool's `type` action sends text. xkeymap: keycode and Shift come from the same "
            "keymap binding (webarena/x11_typing.py; '<' is no longer typed as '>') and held modifiers are "
            "released first. pyautogui: pyautogui.write as before 2026-10-17 (the real eval harness still "
            "types this way)."
        ),
    )
    typing_unmapped_chars: Literal["insert_text", "remap", "drop"] = Field(
        default="insert_text",
        description=(
            "xkeymap only: characters with no binding in the Xvfb keymap (é, ™, —, curly quotes, CJK, emoji). "
            "insert_text: Playwright keyboard.insert_text into the focused element of the current page; the "
            "browser UI (omnibox) never gets these characters. remap: bind a spare keycode to the character's "
            "keysym (xdotool's technique); the training image's Chromium does not use re-bound keycodes, so "
            "the characters are lost (browser check 2026-10-17). drop: skip them silently as pyautogui does."
        ),
    )
    normalize_key_names: bool = Field(
        default=True,
        description=(
            "key_press: lower-case key names and map cmd/command/control -> ctrl, option -> alt, return -> enter, "
            "escape -> esc, as the real eval harness does (webarena/key_names.py). pyautogui drops cmd/command/"
            "control/option on X11, so ['cmd', 't'] typed a plain 't'. false = names passed through (pre-2026-10-17)."
        ),
    )
    block_mock_internals: bool = Field(
        default=True,
        description=(
            "Block top-level navigations to the hub's state endpoints (/go, /state, /post), its data bundles "
            "(/assets/*.js), view-source: of hub pages and file: URLs (internals_guard.py); the apps' own "
            "requests are untouched. A blocked navigation returns a recoverable net::ERR_BLOCKED_BY_CLIENT tool "
            "message. false = no guard (pre-2026-10-17)."
        ),
    )
    propagate_sid_to_new_tabs: bool = Field(
        default=True,
        description=(
            "Install a per-episode browser-context init script that gives every tab on a hub origin the "
            "episode sid (sessionStorage keys + ?sid= when missing; session_sid.py), so tabs the agent opens "
            "itself stay in the scored session. false = sid only on the start URL (pre-2026-10-17)."
        ),
    )
    bring_to_front: Literal["page_change", "every_computer_call"] = Field(
        default="page_change",
        description=(
            "When the worker calls Playwright bring_to_front (CDP Page.bringToFront: activates the tab and gives "
            "the keyboard focus to its page content). page_change: only when the worker changes the active page "
            "(episode start, navigate, tabs_create, tabs_focus), as the real eval harness does, so the address "
            "bar, the find bar or a new tab's omnibox keeps the keyboard focus into the next computer call; "
            "insert_text then goes to the tab in front (Shift-tap probe, as in the harness). every_computer_call: "
            "before every computer call and on tabs_focus (pre-2026-10-18; text typed in the call after a click "
            "on the address bar or Ctrl+F went to the page, and a tab switch by key or click was undone)."
        ),
    )

    @model_validator(mode="after")
    def normalize_hub_config(self) -> Self:
        self.hub_base_url = self.hub_base_url.rstrip("/")
        # Fail at config time, not at the first browser launch, on an entry that cannot be a host pattern.
        self.egress_allow_extra_hosts = [normalize_host(entry) for entry in self.egress_allow_extra_hosts]
        # Setup runs inside /seed_session and reward inside /verify, both of
        # which the base server bounds with its own browser budgets. If the
        # inner budget is not strictly smaller, the outer wait_for fires first
        # and the episode fails as a generic timeout instead of reporting the
        # program's own failure (and the browser lease is retired needlessly).
        if self.setup_timeout_seconds >= self.browser_operation_timeout_seconds:
            raise ValueError(
                "setup_timeout_seconds must be < browser_operation_timeout_seconds "
                f"({self.setup_timeout_seconds} >= {self.browser_operation_timeout_seconds})"
            )
        if self.reward_timeout_seconds >= self.browser_evaluation_timeout_seconds:
            raise ValueError(
                "reward_timeout_seconds must be < browser_evaluation_timeout_seconds "
                f"({self.reward_timeout_seconds} >= {self.browser_evaluation_timeout_seconds})"
            )
        return self


def task_info_from_row(task: WebArenaTaskRow) -> CuaGymTaskInfo:
    extra = getattr(task, "cuagym", None)
    if extra is None:
        extra = (task.model_extra or {}).get("cuagym")
    if extra is None:
        raise ValueError(f"task {task.task_id} has no `cuagym` field")
    if isinstance(extra, CuaGymTaskInfo):
        return extra
    return CuaGymTaskInfo.model_validate(extra)
