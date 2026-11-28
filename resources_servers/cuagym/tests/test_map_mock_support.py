# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""External (non-hub) apps such as ``webarena_map_mock``: endpoint precedence, session
modes, start-URL construction, and the reward-program environment."""

from __future__ import annotations

import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from resources_servers.cuagym import hub_apps
from resources_servers.cuagym.browser_worker import compute_start_url, reward_env, snapshot_urls
from resources_servers.cuagym.episode_code import (
    app_session,
    app_url,
    placeholder_values,
    run_code,
    substitute,
)
from resources_servers.cuagym.hub_apps import SESSION_MODE_NONE, SESSION_MODE_SID, endpoint_overrides
from resources_servers.cuagym.schemas import CuaGymTaskInfo


MAP = "webarena_map_mock"
MAP_URL = "http://203.0.113.30:3000"
GITLAB = "webarena_gitlab_mock"
GITLAB_URL = "http://203.0.113.10:8001"
HUB = "http://hub.example"
PORT = 8000

ENDPOINTS_YAML = f"""
endpoints:
  {GITLAB}: {GITLAB_URL}
  {MAP}: {MAP_URL}/
placeholders:
  __CUA_GYM_WEBARENA_GITLAB_URL__: {{app_dir: {GITLAB}, kind: url}}
  __CUA_GYM_WEBARENA_MAP_URL__: {{app_dir: {MAP}, kind: url}}
sessions:
  {MAP}: none
"""


@pytest.fixture
def endpoints_yaml(tmp_path, monkeypatch):
    def _write(text: str) -> str:
        path = tmp_path / "endpoints.yaml"
        path.write_text(text)
        monkeypatch.setenv(hub_apps.ENDPOINTS_YAML_ENV, str(path))
        hub_apps._OVERRIDES_CACHE = None
        hub_apps._OVERRIDES_CACHE_PATH = None
        return str(path)

    yield _write
    hub_apps._OVERRIDES_CACHE = None
    hub_apps._OVERRIDES_CACHE_PATH = None


@pytest.fixture
def no_endpoints_yaml(monkeypatch):
    monkeypatch.delenv(hub_apps.ENDPOINTS_YAML_ENV, raising=False)
    hub_apps._OVERRIDES_CACHE = None
    hub_apps._OVERRIDES_CACHE_PATH = None
    yield
    hub_apps._OVERRIDES_CACHE = None
    hub_apps._OVERRIDES_CACHE_PATH = None


# --- endpoint precedence -----------------------------------------------------------


def test_yaml_endpoint_wins_over_row_fallback(endpoints_yaml):
    endpoints_yaml(ENDPOINTS_YAML)
    assert app_url(MAP, HUB, PORT, fallback="http://stale.example:3000") == MAP_URL  # trailing slash stripped
    assert app_url(GITLAB, HUB, PORT) == GITLAB_URL


def test_row_fallback_used_when_yaml_has_no_entry(no_endpoints_yaml):
    assert app_url(MAP, HUB, PORT, fallback=f"{MAP_URL}/") == MAP_URL


def test_hub_port_math_applies_to_builtin_apps_without_row_url(no_endpoints_yaml):
    assert app_url("gitlab_mock", HUB, PORT) == f"{HUB}:{PORT + hub_apps.APP_DIRS.index('gitlab_mock')}"
    # A row that names its own deployment wins over the hub convention; only a YAML entry outranks it.
    assert app_url("gitlab_mock", HUB, PORT, fallback="http://row.example:9000/") == "http://row.example:9000"


def test_unknown_app_without_fallback_raises(no_endpoints_yaml):
    with pytest.raises(ValueError, match="unknown CUA-Gym-Hub app dir"):
        app_url(MAP, HUB, PORT)
    with pytest.raises(ValueError, match="absolute URL"):
        app_url(MAP, HUB, PORT, fallback="203.0.113.30:3000")


def test_task_info_accepts_external_base_url():
    info = CuaGymTaskInfo.model_validate(
        {"bundle_id": "b", "app_dir": MAP, "eval_reward_code": "print('REWARD: 0')", "external_base_url": MAP_URL}
    )
    assert info.external_base_url == MAP_URL
    assert (
        CuaGymTaskInfo.model_validate({"bundle_id": "b", "app_dir": MAP, "eval_reward_code": ""}).external_base_url
        is None
    )


def test_placeholder_resolves_to_external_endpoint(endpoints_yaml):
    endpoints_yaml(ENDPOINTS_YAML)
    values = placeholder_values(HUB, PORT)
    assert values["__CUA_GYM_WEBARENA_MAP_URL__"] == MAP_URL
    assert substitute("x = '__CUA_GYM_WEBARENA_MAP_URL__/directions'", sid="s", hub_base_url=HUB, base_port=PORT) == (
        f"x = '{MAP_URL}/directions'"
    )


# --- session modes -------------------------------------------------------------------


def test_session_mode_from_yaml_defaults_to_sid(endpoints_yaml):
    endpoints_yaml(ENDPOINTS_YAML)
    assert app_session(MAP) == SESSION_MODE_NONE
    assert app_session(GITLAB) == SESSION_MODE_SID  # YAML endpoint without a sessions entry keeps isolation
    assert app_session("gitlab_mock") == SESSION_MODE_SID
    assert endpoint_overrides()["sessions"] == {MAP: SESSION_MODE_NONE}


def test_session_mode_without_yaml_is_sid(no_endpoints_yaml):
    assert app_session(MAP) == SESSION_MODE_SID
    assert endpoint_overrides()["sessions"] == {}


def test_sessions_block_is_validated(endpoints_yaml):
    endpoints_yaml(f"endpoints:\n  {MAP}: {MAP_URL}\nsessions:\n  {MAP}: sometimes\n")
    with pytest.raises(ValueError, match="sessions.webarena_map_mock must be one of"):
        endpoint_overrides()
    endpoints_yaml(f"endpoints:\n  {MAP}: {MAP_URL}\nsessions:\n  webarena_typo_mock: none\n")
    with pytest.raises(ValueError, match="unknown app_dir"):
        endpoint_overrides()


# --- start URL -----------------------------------------------------------------------


def test_start_url_hub_convention_matches_previous_behaviour():
    assert compute_start_url(GITLAB_URL, "S", [""], SESSION_MODE_SID) == f"{GITLAB_URL}/?sid=S"
    assert compute_start_url(GITLAB_URL, "S", ["/admin/x/"], SESSION_MODE_SID) == f"{GITLAB_URL}/admin/x/?sid=S"
    assert compute_start_url(GITLAB_URL, "S", ["a/b?p=1"], SESSION_MODE_SID) == f"{GITLAB_URL}/a/b?p=1&sid=S"
    assert (
        compute_start_url(GITLAB_URL, "S", ["http://h/x?sid=__CUA_GYM_SID__"], SESSION_MODE_SID) == "http://h/x?sid=S"
    )


def test_start_url_keeps_sid_before_fragment():
    assert compute_start_url(MAP_URL, "S", ["/#map=15/43.66/-70.25"], SESSION_MODE_SID) == (
        f"{MAP_URL}/?sid=S#map=15/43.66/-70.25"
    )
    assert compute_start_url(MAP_URL, "S", ["/directions?engine=x#map=1/2/3"], SESSION_MODE_SID) == (
        f"{MAP_URL}/directions?engine=x&sid=S#map=1/2/3"
    )


def test_start_url_stateless_app_gets_no_sid():
    assert compute_start_url(MAP_URL, "S", [""], SESSION_MODE_NONE) == f"{MAP_URL}/"
    assert compute_start_url(MAP_URL, "S", ["/directions?engine=x#map=1/2/3"], SESSION_MODE_NONE) == (
        f"{MAP_URL}/directions?engine=x#map=1/2/3"
    )


# --- reward environment --------------------------------------------------------------


def test_snapshot_urls_keeps_tab_order_and_drops_blank_tabs():
    context = SimpleNamespace(
        pages=[
            SimpleNamespace(url="http://a/1"),
            SimpleNamespace(url="about:blank"),
            SimpleNamespace(url="http://a/2?sid=x"),
        ]
    )
    assert snapshot_urls(context) == ["http://a/1", "http://a/2?sid=x"]
    assert snapshot_urls(None) == []


def test_reward_env_exports_answer_status_and_json_urls():
    env = reward_env(status="success", answer="4.6 km", final_urls=["http://a/1", "http://a/2"])
    assert env == {
        "CUA_GYM_AGENT_ANSWER": "4.6 km",
        "CUA_GYM_AGENT_STATUS": "success",
        "CUA_GYM_FINAL_URLS": '["http://a/1", "http://a/2"]',
    }
    assert reward_env(status=None, answer=None, final_urls=[]) == {
        "CUA_GYM_AGENT_ANSWER": "",
        "CUA_GYM_AGENT_STATUS": "",
        "CUA_GYM_FINAL_URLS": "[]",
    }


REWARD_PROGRAM = """
import json, os
answer = os.environ.get("CUA_GYM_AGENT_ANSWER", "")
urls = json.loads(os.environ.get("CUA_GYM_FINAL_URLS") or "[]")
worked = any("/directions" in u for u in urls)
answered = worked and answer.strip() == "4.6 km"
print(f"REWARD: {(0.4 if worked else 0.0) + (0.6 if answered else 0.0)}")
"""


@pytest.mark.parametrize(
    ("urls", "answer", "expected"),
    [
        ([], "4.6 km", "REWARD: 0.0"),
        (["http://m/directions?engine=bike&route=1,2;3,4"], "", "REWARD: 0.4"),
        (["http://m/directions?engine=bike&route=1,2;3,4"], "wrong", "REWARD: 0.4"),
        (["http://m/", "http://m/directions?engine=bike&route=1,2;3,4"], "4.6 km", "REWARD: 1.0"),
    ],
)
def test_reward_program_sees_the_exported_environment(urls, answer, expected):
    result = run_code(
        REWARD_PROGRAM,
        timeout_seconds=30,
        extra_env=reward_env(status="success", answer=answer, final_urls=urls),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == expected


def test_run_code_environment_is_allowlisted_plus_reward_env(monkeypatch):
    monkeypatch.setenv("NEMO_GYM_CONFIG_DICT", "secret")
    result: subprocess.CompletedProcess[str] = run_code(
        "import json, os; print(json.dumps(sorted(k for k in os.environ if k.startswith(('CUA_GYM_', 'NEMO_')))))",
        timeout_seconds=30,
        extra_env=reward_env(status="failure", answer="", final_urls=["http://m/"]),
    )
    assert json.loads(result.stdout.strip()) == ["CUA_GYM_AGENT_ANSWER", "CUA_GYM_AGENT_STATUS", "CUA_GYM_FINAL_URLS"]
    assert "NEMO_GYM_CONFIG_DICT" not in os.environ or True  # the parent keeps it; the child never saw it
