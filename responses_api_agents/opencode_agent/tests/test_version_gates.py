# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-release behavior of a sandbox session: installer digests, request sampling and the untested-release warning.

Each case sets its release explicitly: the shared fake sandbox (`setup`) runs the default, while most shipped configs
pin 1.17.11.
"""

import shlex
import shutil
from contextlib import nullcontext
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from responses_api_agents.opencode_agent.runtime import OPENCODE_RELEASE_SHA256
from responses_api_agents.opencode_agent.tests.test_sandbox_extensions import episode, index, launch_config, record
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import close_body, seed, setup  # noqa: F401


LINUX_ASSETS = {"x64-baseline", "x64-baseline-musl", "arm64", "arm64-musl"}
# The supported releases, written out rather than read from app.py or runtime.py so that dropping one fails here.
VALIDATED_RELEASES = ("1.17.11", "1.19.6", "2.2.7", "2.3.2")
# OpenCode 1.17.11 and 1.19.6 built-in agents (README "Request sampling"); stated here as the independent oracle.
BUILTIN_AGENTS = {"build", "plan", "general", "explore", "compaction", "title", "summary"}


def _opencode2_store(version):
    """Copy the fake store instead of translating it, as test_opencode2_session's fixture does, for 2.x versions."""
    if not version.startswith("2."):
        return nullcontext()

    def translate(source, target, *, labels=None):
        shutil.copy(source, target)
        return []

    return patch("responses_api_agents.opencode_agent.sandbox.translate_store", translate)


@pytest.mark.parametrize("version", VALIDATED_RELEASES)
async def test_installer_receives_the_release_digests_pinned_for_the_configured_version(setup, version):
    agent, sandbox = setup
    agent.config.opencode_version = version
    events = record(sandbox)
    with _opencode2_store(version):
        response, _ = await episode(agent, sandbox)
    assert response.status == "completed"
    install = shlex.split(events[index(events, "install_opencode_runtime.sh")])
    # runtime, version, executable path, its digest, PATH prefix, then the release digests.
    assert install[2:-1] == [f"/tmp/nemo-gym-opencode-runtime-{version}", version, "", "", ""]
    digests = dict(word.split("=", 1) for word in install[-1].split())
    assert digests.keys() == LINUX_ASSETS
    assert digests == OPENCODE_RELEASE_SHA256[version]


async def test_a_release_without_pinned_digests_downloads_unchecked_on_opencode_1(setup):
    agent, sandbox = setup
    agent.config.opencode_version = "1.19.4"
    events = record(sandbox)
    response, _ = await episode(agent, sandbox)
    assert response.status == "completed"
    assert shlex.split(events[index(events, "install_opencode_runtime.sh")])[-1] == ""


@pytest.mark.parametrize("version", ["1.17.11", "1.19.6"])
async def test_request_sampling_reaches_every_builtin_agent_on_each_validated_opencode_1(setup, version):
    agent, sandbox = setup
    agent.config.opencode_version = version
    response, _ = await episode(agent, sandbox, temperature=0.7, top_p=0.9)
    assert response.status == "completed"
    _, config = launch_config(sandbox)
    assert config["agent"] == {name: {"temperature": 0.7, "top_p": 0.9} for name in BUILTIN_AGENTS}
    assert config["provider"]["nemo_gym"]["models"]["dummy_model"]["temperature"] is True


@pytest.mark.parametrize("version", ["1.19.4", "1.15.13"])
def test_request_sampling_is_refused_for_an_opencode_1_outside_the_validated_releases(setup, version):
    agent, sandbox = setup
    agent.config.opencode_version = version
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        refused = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task", "temperature": 0.6})
    assert refused.status_code == 422
    # Names both validated releases, so a gate that dropped one of them fails here.
    assert "applies request temperature only with OpenCode 1.17.11, 1.19.6," in refused.json()["detail"]
    sandbox.launch.assert_not_awaited()


@pytest.mark.parametrize(
    ("version", "warn"),
    [*((version, False) for version in VALIDATED_RELEASES), ("1.19.4", True)],
)
def test_only_a_release_outside_the_validated_ones_logs_the_untested_warning(setup, caplog, version, warn):
    agent, sandbox = setup
    agent.config.opencode_version = version
    with _opencode2_store(version), TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        created.raise_for_status()
        client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"])).raise_for_status()
    assert (f"OpenCode {version} is untested" in caplog.text) is warn
