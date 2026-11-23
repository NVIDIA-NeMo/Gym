# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the 2026-10-17 environment-parity fixes.

Pure Python: no browser, no Ray, no X server, no third-party imports. Runs under
pytest or as a plain script (login-node friendly)::

    python3 -I resources_servers/cuagym/tests/test_env_parity.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resources_servers.cuagym.egress import (  # noqa: E402
    LOOPBACK_HOSTS,
    MAP_DEPENDENCY_HOSTS,
    egress_allow_hosts,
    host_allowed,
    host_resolver_rules_arg,
    normalize_host,
)
from resources_servers.cuagym.internals_guard import (  # noqa: E402
    VIEW_SOURCE_REASON,
    blocked_message,
    blocked_navigation,
    computer_call_may_open_internals,
)
from resources_servers.cuagym.session_sid import (  # noqa: E402
    HUB_SID_STORAGE_KEYS,
    hub_scope,
    origin_of,
    sid_init_script,
)
from resources_servers.webarena.key_names import HARNESS_KEY_ALIASES, normalize_key_names  # noqa: E402


# endpoints block of projects/ccv2/diag3/kit/configs/webarena_endpoints_88c91a.yaml (new hub instance)
KIT_ENDPOINTS = {
    "webarena_classifieds_mock": "http://198.51.100.10:8000",
    "webarena_gitlab_mock": "http://198.51.100.10:8001",
    "webarena_reddit_mock": "http://198.51.100.10:8002",
    "webarena_shopping_admin_mock": "http://198.51.100.10:8003",
    "webarena_shopping_mock": "http://198.51.100.10:8004",
    "webarena_map_mock": "http://198.51.100.20:3000",
    "reddit_mock": "http://198.51.100.30:8073",
    "notion_mock": "http://198.51.100.30:8066",
}
# The real-eval harness default (WA_EGRESS_ALLOW in webarena_eval/.../common/browser.py) minus its benchmark host.
HARNESS_DEFAULT_ALLOW = (
    "localhost,127.0.0.1,tile.openstreetmap.org,*.tile.openstreetmap.org,*.openstreetmap.de,"
    "graphhopper.com,*.graphhopper.com,*.memomaps.de"
).split(",")


def _raises(exc_type: type[BaseException], fn, *args) -> None:
    try:
        fn(*args)
    except exc_type:
        return
    raise AssertionError(f"{fn.__name__}{args} did not raise {exc_type.__name__}")


# --------------------------------------------------------------------------- egress


def test_egress_allowlist_covers_hub_endpoints_loopback_and_map_dependencies() -> None:
    allow = egress_allow_hosts(hub_base_url="http://localhost", endpoints=KIT_ENDPOINTS)
    for host in ("198.51.100.10", "198.51.100.20", "198.51.100.30", *LOOPBACK_HOSTS, *MAP_DEPENDENCY_HOSTS):
        assert host in allow, host
    assert len(allow) == len(set(allow)), "allowlist must be de-duplicated"
    assert allow[0] == "localhost", "hub hosts come first"


def test_egress_allowlist_is_a_superset_of_the_real_harness_list() -> None:
    allow = egress_allow_hosts(hub_base_url="http://localhost", endpoints=KIT_ENDPOINTS)
    assert set(HARNESS_DEFAULT_ALLOW) <= set(allow)
    assert "overpass-api.de" in allow  # the mock map's "Query features" endpoint (not in the harness list)


def test_judge_proxy_endpoint_stays_out_of_the_browser_allowlist() -> None:
    # judge_proxy_start.sh adds the head-node proxy to the run's endpoints YAML (sessions: none).
    endpoints = {**KIT_ENDPOINTS, "webarena_judge": "http://10.65.12.34:41234"}
    allow = egress_allow_hosts(hub_base_url="http://localhost", endpoints=endpoints)
    assert not host_allowed("10.65.12.34", allow)
    assert host_allowed("198.51.100.10", allow)
    opened = egress_allow_hosts(hub_base_url="http://localhost", endpoints=endpoints, exclude_apps=())
    assert host_allowed("10.65.12.34", opened)


def test_host_resolver_rule_has_the_harness_format() -> None:
    arg = host_resolver_rules_arg(["198.51.100.40", "localhost", "*.memomaps.de"])
    expected = "MAP * ~NOTFOUND, EXCLUDE 198.51.100.40, EXCLUDE localhost, EXCLUDE *.memomaps.de"
    assert arg == "--host-resolver-rules=" + expected


def test_host_allowed_follows_chromium_pattern_semantics() -> None:
    allow = egress_allow_hosts(hub_base_url="http://localhost", endpoints=KIT_ENDPOINTS, extra=["10.0.0.5:3000"])
    for host in (
        "198.51.100.10",
        "198.51.100.20",
        "localhost",
        "tile.openstreetmap.org",
        "a.tile.openstreetmap.org",
        "gps.tile.openstreetmap.org",
        "valhalla1.openstreetmap.de",
        "tileserver.memomaps.de",
        "overpass-api.de",
        "10.0.0.5",
        "Tile.OpenStreetMap.org",
    ):
        assert host_allowed(host, allow), host
    # Hosts seen in our checkpoints' and Qwen's off-mock navigations, plus near-misses of allowed patterns.
    for host in (
        "github.com",
        "raw.githubusercontent.com",
        "gitlab.com",
        "www.google.com",
        "maps.google.com",
        "www.openstreetmap.org",
        "openstreetmap.de",
        "memomaps.de",
        "198.51.100.11",
        "evil-overpass-api.de",
        "overpass-api.de.evil.com",
        "",
    ):
        assert not host_allowed(host, allow), host


def test_normalize_host_accepts_urls_ports_and_patterns() -> None:
    assert normalize_host("http://203.0.113.10/:8000") == "203.0.113.10"  # old kit YAML typo
    assert normalize_host("HTTP://Example.ORG:8080/x?y=1") == "example.org"
    assert normalize_host("10.0.0.5:3000") == "10.0.0.5"
    assert normalize_host(" *.Example.org ") == "*.example.org"
    for bad in ("", "a b", "x,y", "http://", "host;rm", "-bad.org"):
        _raises(ValueError, normalize_host, bad)


PARITY_KNOBS = (
    "egress_allowlist: true",
    "egress_allow_extra_hosts: []",
    "egress_exclude_endpoint_apps: [webarena_judge]",
    "propagate_sid_to_new_tabs: true",
    "typing_backend: xkeymap",
    "typing_unmapped_chars: insert_text",
    "normalize_key_names: true",
    "block_mock_internals: true",
    "bring_to_front: page_change",
)


def test_cuagym_configs_expose_the_parity_knobs() -> None:
    for name in ("cuagym", "cuagym_cc_authority", "cuagym_cc_authority_n2k2", "cuagym_cc_authority_n2k4"):
        text = (ROOT / "resources_servers/cuagym/configs" / f"{name}.yaml").read_text()
        for knob in PARITY_KNOBS:
            assert f"      {knob}\n" in text, (name, knob)


def _schema_defaults() -> dict:
    """Field defaults of CuaGymResourcesServerConfig, read from the source (no pydantic import)."""
    import ast

    tree = ast.parse((ROOT / "resources_servers/cuagym/schemas.py").read_text())
    config = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "CuaGymResourcesServerConfig")
    defaults = {}
    for stmt in config.body:
        if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.value, ast.Call)):
            continue
        for kw in stmt.value.keywords:
            if kw.arg == "default":
                defaults[stmt.target.id] = ast.literal_eval(kw.value)
            elif kw.arg == "default_factory":
                factory = kw.value
                if isinstance(factory, ast.Name) and factory.id == "list":
                    defaults[stmt.target.id] = []
                elif isinstance(factory, ast.Lambda):
                    defaults[stmt.target.id] = ast.literal_eval(factory.body)
    return defaults


def _yaml_scalar(value: str):
    if value in ("true", "false"):
        return value == "true"
    if value.startswith("[") and value.endswith("]"):
        return [item.strip() for item in value[1:-1].split(",") if item.strip()]
    return value


def test_schema_defaults_match_the_config_knobs() -> None:
    defaults = _schema_defaults()
    for knob in PARITY_KNOBS:
        name, _, value = knob.partition(": ")
        assert defaults[name] == _yaml_scalar(value), (name, defaults.get(name), value)
    assert defaults["typing_unmapped_chars"] == "insert_text"


# --------------------------------------------------------------------------- sid in every tab

KIT_SESSIONS = {"webarena_map_mock": "none"}
SID = "d2c196dd-38d7-4c03-8e92-0a190154754d"
WEBARENA_SID_KEYS = ("gitlab_sid", "reddit_sid", "webarena_shopping_sid", "shopping_admin_sid", "classifieds_sid")
HUB_CHECKOUT = ROOT.parent / "CUA-Gym-Hub"  # $WS/projects/CUA-Gym-Hub next to this worktree


def _script_cfg(script: str) -> dict:
    import json

    head = "const cfg = "
    start = script.index(head) + len(head)
    return json.loads(script[start : script.index(";\n", start)])


def _kit_scope(base_url: str = "http://198.51.100.10:8001", mode: str = "sid"):
    return hub_scope(
        base_url=base_url,
        base_session_mode=mode,
        hub_base_url="http://localhost",
        endpoints=KIT_ENDPOINTS,
        sessions=KIT_SESSIONS,
    )


def test_origin_of_matches_window_location_origin() -> None:
    assert origin_of("http://198.51.100.10:8001/a/b?sid=1#x") == "http://198.51.100.10:8001"
    assert origin_of("HTTP://Host.Example:80/") == "http://host.example"
    assert origin_of("https://h.example:443/x") == "https://h.example"
    for url in ("about:blank", "view-source:http://h/", "file:///etc/passwd", "http://h:99999/", ""):
        assert origin_of(url) is None, url


def test_hub_scope_covers_sid_apps_and_excludes_the_map() -> None:
    scope = _kit_scope()
    for port in (8000, 8001, 8002, 8003, 8004):
        assert f"http://198.51.100.10:{port}" in scope.origins
    assert "http://198.51.100.20:3000" in scope.excluded_origins
    assert "http://198.51.100.20:3000" not in scope.origins
    assert scope.contains("http://198.51.100.10:8002/forums/all?x=1")
    assert scope.contains("http://localhost:7999/")  # base_port + index app on the hub host
    assert not scope.contains("http://198.51.100.20:3000/directions")
    assert not scope.contains("https://github.com/")
    assert not scope.contains("about:blank")


def test_hub_scope_of_a_map_start_episode_still_covers_the_other_apps() -> None:
    scope = _kit_scope(base_url="http://203.0.113.30:3000", mode="none")  # vendor map via row external_base_url
    assert "http://203.0.113.30:3000" in scope.excluded_origins
    assert not scope.contains("http://203.0.113.30:3000/search?query=x")
    assert scope.contains("http://198.51.100.10:8001/")


def test_sid_init_script_carries_the_episode_scope() -> None:
    scope = _kit_scope()
    script = sid_init_script(SID, scope)
    cfg = _script_cfg(script)
    assert cfg["sid"] == SID
    assert cfg["origins"] == list(scope.origins)
    assert cfg["hosts"] == ["localhost"]
    assert cfg["excluded"] == ["http://198.51.100.20:3000"]
    for key in WEBARENA_SID_KEYS:
        assert key in cfg["keys"], key
    # Fill-only semantics: an existing sessionStorage key or URL sid is never overwritten,
    # and the URL is rewritten in top-level http(s) documents only.
    assert "if (!store.getItem(key)) store.setItem(key, cfg.sid);" in script
    assert 'if (!params.get("sid"))' in script
    assert 'window.top === window && (loc.protocol === "http:" || loc.protocol === "https:")' in script
    assert script.count("%") == 0


def test_sid_init_script_rejects_a_sid_that_could_break_out_of_the_script() -> None:
    for bad in ("", 'x"; alert(1); "', "a b", "a/b"):
        _raises(ValueError, sid_init_script, bad, _kit_scope())


def test_sid_keys_match_the_webarena_mock_sources() -> None:
    import re

    if not HUB_CHECKOUT.is_dir():
        print(f"  (skip: no CUA-Gym-Hub checkout at {HUB_CHECKOUT})")
        return
    literal = re.compile(
        r"""sessionStorage\.(?:get|set)Item\(\s*['"]([A-Za-z0-9_]+)['"]"""
        r"""|SID_KEY\s*=\s*['"]([A-Za-z0-9_]+)['"]"""
    )
    for app in ("gitlab", "reddit", "shopping", "shopping_admin", "classifieds"):
        text = (HUB_CHECKOUT / "websites" / f"webarena_{app}_mock" / "src" / "utils" / "dataManager.js").read_text()
        found = {a or b for a, b in literal.findall(text)}
        assert found, app
        assert found <= set(HUB_SID_STORAGE_KEYS), (app, found)


# --------------------------------------------------------------------------- mock internals


def test_internals_guard_blocks_state_endpoints_bundles_view_source_and_files() -> None:
    scope = _kit_scope()
    hub = "http://198.51.100.10:8001"
    blocked = (
        f"{hub}/state?sid={SID}",
        f"{hub}/STATE?sid=x",  # connect matches routes case-insensitively
        f"{hub}/state.json",  # ... and with a "." or "/" after the route
        f"{hub}/go?sid={SID}",
        f"{hub}/go/",
        f"{hub}/post?sid=x",
        f"{hub}/state-inspector",
        f"{hub}/_cua_session?token=x",
        f"{hub}/%73tate?sid=x",
        f"{hub}/assets/seed-issues_index-Do-OE0Lz.js",
        "http://198.51.100.10:8004/assets/seed-BKL9wgOZ.js",
        "http://198.51.100.10:8002/assets/index-pY0-y-wD.js",  # the reddit bundle embeds the whole corpus
        f"{hub}/@fs/etc/passwd",
        f"{hub}/src/data/issues.json",
        "http://localhost:7999/state?sid=x",  # base_port + index app on the hub host
        f"view-source:{hub}/byteblaze/gimmiethat.space",
        "view-source:http://198.51.100.10:8004/",
        "file:///scratch/fsw/portfolios/nemotron/webarena_e26_train.jsonl",
        "FILE:///etc/passwd",
        "view-source:file:///etc/passwd",
    )
    allowed = (
        f"{hub}/",
        f"{hub}/byteblaze/gimmiethat.space/-/raw/main/index.html?sid={SID}",
        f"{hub}/goals",
        f"{hub}/states",
        f"{hub}/postmill",
        f"{hub}/assets/index-4EBUeqcj.css",
        f"{hub}/explore/projects?sort=stars",
        "http://198.51.100.10:8002/f/gaming/123/some-post",
        "http://198.51.100.10:8003/admin/sales/order/",
        "http://198.51.100.20:3000/assets/application.debug-66c6.js",  # the map is the real OSM app: out of scope
        "http://198.51.100.20:3000/api/0.6/way/209051849",
        "https://github.com/state",  # off-hub URLs are the egress rule's business
        "view-source:https://github.com/",
        "about:blank",
        "chrome://newtab/",
    )
    for url in blocked:
        assert blocked_navigation(url, scope), url
    for url in allowed:
        assert blocked_navigation(url, scope) is None, url
    # The worker gives the same reason when the document shows a hub page's source (the tab reports the inner URL).
    assert blocked_navigation(f"view-source:{hub}/explore", scope) == VIEW_SOURCE_REASON


def test_internals_prefilter_flags_only_calls_that_can_open_internals() -> None:
    def call(*actions):
        return {"actions": list(actions)}

    def flagged(arguments, history=False):
        return computer_call_may_open_internals(arguments, blocked_page_in_history=history)

    omnibox = call(
        {"action": "left_click", "coordinate": [0.3, 0.05]},
        {"action": "type", "text": "198.51.100.10:8001/state?sid=x\n"},
    )
    assert flagged(omnibox)
    assert flagged(call({"action": "type", "text": "view-source:x"}))
    assert flagged(call({"action": "type", "text": "file:///etc/passwd"}))
    assert flagged(call({"action": "key_press", "keys": ["CTRL", "U"]}))  # view source
    back = call({"action": "key_press", "keys": ["alt", "left"]})
    assert not flagged(back) and flagged(back, history=True)
    for text in ("hello world", "/google maps", "states of the union", "@go team", "ctrl u"):
        assert not flagged(call({"action": "type", "text": text}), history=True), text
    assert not flagged(call({"action": "key_press", "keys": ["ctrl", "a"]}), history=True)


def test_blocked_message_reads_like_a_browser_navigation_error() -> None:
    url = "http://198.51.100.10:8001/state?sid=x"
    assert blocked_message(url) == f"Tool error: Error: Page.goto: net::ERR_BLOCKED_BY_CLIENT at {url}"


# --------------------------------------------------------------------------- key_press names

HARNESS_AGENT = ROOT.parent / "webarena_eval/osworld_internal/webarena/nvidia/nemotron_toolcall_agent.py"


def test_key_names_are_normalized_like_the_real_harness() -> None:
    assert normalize_key_names(["cmd", "t"]) == ["ctrl", "t"]  # pyautogui alone drops "cmd": a plain "t"
    assert normalize_key_names(["Control", "A"]) == ["ctrl", "a"]  # not Ctrl+Shift+A
    assert normalize_key_names(["command", "Left"]) == ["ctrl", "left"]
    assert normalize_key_names(["option", "Return", "Escape"]) == ["alt", "enter", "esc"]
    assert normalize_key_names(["shift", "End"]) == ["shift", "end"]
    assert normalize_key_names(["super"]) == ["super"]  # passed through (and dropped by pyautogui) in both


def test_key_aliases_match_the_harness_source() -> None:
    import ast

    if not HARNESS_AGENT.is_file():
        print(f"  (skip: no harness copy at {HARNESS_AGENT})")
        return
    tree = ast.parse(HARNESS_AGENT.read_text())
    funcs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_normalize_key"]
    assert len(funcs) == 1
    aliases = next(
        ast.literal_eval(node.value)
        for node in ast.walk(funcs[0])
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "aliases"
    )
    assert aliases == HARNESS_KEY_ALIASES


def _main() -> int:
    failures = 0
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failures += 1
            traceback.print_exc()
            print(f"FAIL {name}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
