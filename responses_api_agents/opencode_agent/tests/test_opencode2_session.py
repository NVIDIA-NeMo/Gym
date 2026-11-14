# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox sessions that run OpenCode 2 (``opencode_version: 2.x``) through the shared session lifecycle."""

import hashlib
import json
import shlex
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from responses_api_agents.opencode_agent.opencode2 import PLUGIN_SOURCE
from responses_api_agents.opencode_agent.opencode2.config import UNRESTRICTED_MEDIA
from responses_api_agents.opencode_agent.runtime import OPENCODE_RELEASE_SHA256, OPENCODE_V2_VERSION
from responses_api_agents.opencode_agent.tests.test_mcp_grants import grant
from responses_api_agents.opencode_agent.tests.test_sandbox_extensions import (
    episode,
    index,
    launch_config,
    positional,
    record,
    seed_status,
    session_store,
)
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import (  # noqa: F401
    JPEG,
    JPEG_URI,
    PNG,
    PNG_URI,
    capture_observations,
    close_body,
    seed,
    setup,
)


RUNTIME = f"/tmp/nemo-gym-opencode-runtime-{OPENCODE_V2_VERSION}"


@pytest.fixture
def opencode2(setup):  # noqa: F811
    """The shared fake sandbox with OpenCode 2 selected; the fake store is copied rather than translated."""
    agent, sandbox = setup
    agent.config.opencode_version = OPENCODE_V2_VERSION
    translations = []

    def translate(source, target, *, labels=None):
        translations.append(labels)
        shutil.copy(source, target)

    with patch("responses_api_agents.opencode_agent.sandbox.translate_store", translate):
        sandbox.translations = translations
        yield agent, sandbox


async def test_a_session_installs_the_pinned_release_and_runs_opencode_2_standalone(opencode2):
    agent, sandbox = opencode2
    events = record(sandbox)
    response, closed = await episode(agent, sandbox)
    assert response.status == "completed"
    assert response.metadata["opencode_version"] == OPENCODE_V2_VERSION
    install = shlex.split(events[index(events, "install_opencode_runtime.sh")])
    # The release download is checked against the digests pinned for this version.
    digests = " ".join(f"{arch}={digest}" for arch, digest in OPENCODE_RELEASE_SHA256[OPENCODE_V2_VERSION].items())
    assert install[2:] == [RUNTIME, OPENCODE_V2_VERSION, "", "", "", digests]
    directory = sandbox.directory
    # One plugin directory replaces OpenCode 1's plugin files.
    assert sandbox.files[f"{directory}/nemo-gym-plugin/index.js"] == PLUGIN_SOURCE.read_text()
    assert not [path for path in sandbox.files if path.endswith((".mjs", "assistant_message_header.js"))]
    params, config = launch_config(sandbox)
    assert params["command"] == [
        f"{RUNTIME}/opencode",
        "run",
        "--standalone",
        "--format",
        "json",
        "--title",
        "NeMo Gym",
        "--agent",
        "build",
        "--thinking",
    ]
    assert params["env"]["OPENCODE_DISABLE_FREE_MODELS"] == "1"
    assert "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX" not in params["env"]
    assert config["model"] == "nemo_gym/dummy_model"
    model = config["providers"]["nemo_gym"]["models"]["dummy_model"]
    assert model["limit"] == {"context": 32000, "input": 32000, "output": 4096}
    # OpenCode 2 assumes image input for an unknown model; a text-only session declares text only.
    assert model["capabilities"] == {"input": ["text"], "output": ["text"]}
    assert "media" not in config
    assert "execute" not in config["permission"]
    (plugin,) = config["plugins"]
    assert plugin["package"] == f"file://{directory}/nemo-gym-plugin"
    assert plugin["options"] == {
        "taskEnvironmentFile": f"{directory}/task-env.json",
        "readyFile": f"{directory}/task-environment.ready",
    }
    # The store is read through the OpenCode 1 tables.
    assert sandbox.translations == [{}]


def test_prompt_images_declare_image_input_and_turn_off_image_processing(opencode2):
    agent, sandbox = opencode2
    prompt = [
        {"type": "input_text", "text": "What is shown?"},
        {"type": "input_image", "image_url": PNG_URI, "detail": "auto"},
        {"type": "input_image", "image_url": JPEG_URI, "detail": "auto"},
    ]
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        result = client.post(
            "/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": [{"role": "user", "content": prompt}]}
        )
        assert result.status_code == 200, result.text
        params, config = launch_config(sandbox)
        assert params["prompt"] == "What is shown?"
        command = params["command"]
        paths = [command[position + 1] for position, arg in enumerate(command) if arg == "--file"]
        assert paths == [
            f"/tmp/nemo-gym-media/{hashlib.sha256(PNG).hexdigest()}.png",
            f"/tmp/nemo-gym-media/{hashlib.sha256(JPEG).hexdigest()}.jpg",
        ]
        assert [sandbox.files[path] for path in paths] == [PNG, JPEG]
        model = config["providers"]["nemo_gym"]["models"]["dummy_model"]
        assert model["capabilities"]["input"] == ["text", "image"]
        assert config["media"] == UNRESTRICTED_MEDIA
        assert UNRESTRICTED_MEDIA["image"]["auto_resize"] is False
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text


@pytest.mark.parametrize("attachment", [None, {"image": {"max_width": 1024}}])
async def test_configured_image_input_and_limits_reach_opencode_2(opencode2, attachment):
    agent, sandbox = opencode2
    agent.config.image_input = True
    if attachment is not None:
        agent.config.opencode_config = {**agent.config.opencode_config, "attachment": attachment}
    response, _ = await episode(agent, sandbox)
    assert response.status == "completed"
    _, config = launch_config(sandbox)
    assert config["providers"]["nemo_gym"]["models"]["dummy_model"]["capabilities"]["input"] == ["text", "image"]
    # OpenCode 2 calls OpenCode 1's `attachment` limits `media`; a configured value replaces the lifted limits.
    assert config["media"] == (attachment or UNRESTRICTED_MEDIA)
    assert "attachment" not in config


async def test_sampling_steps_and_a_replaced_prompt_reach_opencode_2(opencode2):
    agent, sandbox = opencode2
    agent.config.system_prompt = "Be {careful}."
    agent.config.system_prompt_mode = "replace"
    agent.config.opencode_config = {
        "permission": {"*": "allow", "execute": "deny"},
        "agent": {"build": {"steps": 5}},
        "compaction": {"auto": False},
    }
    response, _ = await episode(agent, sandbox, temperature=0.25, top_p=0.5)
    assert response.status == "completed"
    params, config = launch_config(sandbox)
    # OpenCode 2 does not apply per-agent sampling; the plugin sets it on every model call.
    assert config["plugins"][0]["options"]["sampling"] == {"temperature": 0.25, "top_p": 0.5}
    assert config["providers"]["nemo_gym"]["models"]["dummy_model"]["capabilities"]["temperature"] is True
    assert config["agent"] == {"build": {"steps": 5, "prompt": "Be {careful}."}}
    assert config["compaction"] == {"auto": False}
    assert config["permission"] == {"*": "allow", "execute": "deny"}
    # The prompt's braces never reach OpenCode's {env:} and {file:} substitution.
    assert "{careful}" not in params["env"]["OPENCODE_CONFIG_CONTENT"]
    assert "instructions" not in config


def test_mcp_grants_are_direct_servers_that_the_plugin_checks(opencode2):
    agent, sandbox = opencode2
    agent.config.mcp_tool_timeout_seconds = 900
    body = seed().model_dump(mode="json")
    body["tool_accesses"] = [grant("task_tools", required=True), grant("extra", required=False)]
    wait_exit = sandbox.wait_exit

    async def plugin_then_exit():
        assert json.loads(sandbox.files[f"{sandbox.directory}/mcp-status.json"]) is None
        sandbox.files[f"{sandbox.directory}/mcp-status.json"] = json.dumps(
            {"task_tools": {"status": "failed", "error": "connection refused"}, "extra": {"status": "connected"}}
        )
        return await wait_exit()

    sandbox.wait_exit = plugin_then_exit
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=body)
        assert created.status_code == 200, created.text
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        _, config = launch_config(sandbox)
        assert config["mcp"]["servers"]["task_tools"] == {
            "type": "remote",
            "url": "http://resources.example:8000/task_tools/mcp",
            "oauth": False,
            "codemode": False,
            "headers": {"x-session": "token"},
            "timeout": {"startup": 900000, "catalog": 900000, "execution": 900000},
        }
        options = config["plugins"][0]["options"]
        assert options["mcpGrants"] == {"required": ["task_tools"], "optional": ["extra"]}
        assert options["mcpStatusFile"] == f"{sandbox.directory}/mcp-status.json"
        assert options["mcpStartupMs"] == 900000
        assert not [path for path in sandbox.files if path.endswith("required_mcp.mjs")]
        assert result.status_code == 502
        assert result.json()["detail"] == (
            "Required Gym MCP tools could not be initialized: task_tools: failed (connection refused)"
        )
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        (
            {"sandbox_plugin_dependencies_path": "/opt/deps.tar", "sandbox_plugin_dependencies_sha256": "a" * 64},
            "plugin dependencies",
        ),
        ({"opencode_version": "2.9.0"}, "no pinned release digest"),
    ],
)
def test_settings_opencode_2_cannot_use_refuse_the_seed(opencode2, setting, message):
    agent, sandbox = opencode2
    for key, value in setting.items():
        setattr(agent.config, key, value)
    refused = seed_status(agent)
    assert refused.status_code == 422
    assert message in refused.text
    sandbox.exec.assert_not_awaited()


async def test_an_unpinned_version_runs_from_a_pinned_binary(opencode2):
    agent, sandbox = opencode2
    agent.config.opencode_version = "2.9.0"
    agent.config.sandbox_executable_path = "/opt/opencode"
    agent.config.sandbox_executable_sha256 = "b" * 64
    events = record(sandbox)
    response, _ = await episode(agent, sandbox)
    assert response.status == "completed"
    install = shlex.split(events[index(events, "install_opencode_runtime.sh")])
    # No release is downloaded, so there is no digest to check it against.
    assert install[3:] == ["2.9.0", "/opt/opencode", "b" * 64, "", ""]


async def test_owner_plugins_are_directories_loaded_after_gyms(opencode2):
    agent, sandbox = opencode2
    agent.config.plugins = ["/opt/owner/plugin-a", "/opt/owner/plugin-b"]
    events = record(sandbox)
    response, _ = await episode(agent, sandbox)
    assert response.status == "completed"
    check = events[index(events, "/opt/owner/plugin-a")]
    assert '[ -d "$path" ]' in check and '[ -f "$path" ]' not in check
    _, config = launch_config(sandbox)
    assert config["plugins"][1:] == ["file:///opt/owner/plugin-a", "file:///opt/owner/plugin-b"]


async def test_exports_use_opencode_2_session_export(opencode2, tmp_path):
    agent, sandbox = opencode2
    agent.config.export_dir = "/exports"
    capture_observations(sandbox, tmp_path)
    events = record(sandbox)
    response, _ = await episode(agent, sandbox)
    assert response.status == "completed"
    (exported,) = [event for event in events if '"$1" session export --standalone "$3"' in event]
    assert positional(exported)[:3] == [f"{RUNTIME}/opencode", "/exports", "root"]


def test_an_import_uses_session_import_into_the_task_workdir(opencode2, tmp_path):
    agent, sandbox = opencode2
    agent.config.session_import_path = "/staged/session.json"
    imported = session_store(tmp_path / "imported.db", roots=["ses_prefix"], messages=["prefix-user"])
    original = sandbox.download

    async def download(source, destination):
        if source.endswith("/opencode.db"):
            Path(destination).write_bytes(imported.read_bytes())
        else:
            await original(source, destination)

    sandbox.download = download
    events = record(sandbox)
    assert seed_status(agent).status_code == 200
    (command,) = [event for event in events if "/opencode session import" in event]
    # OpenCode 2 binds an imported session to the directory it is given.
    assert f"{RUNTIME}/opencode session import --standalone --directory /app /staged/session.json" in command


async def test_labels_the_plugin_recorded_name_the_replies(opencode2):
    agent, sandbox = opencode2
    original = sandbox.download

    async def download(source, destination):
        if source.endswith("/model-call-labels.jsonl"):
            Path(destination).write_text(json.dumps({"label": "msg_gym_1", "session": "ses_1", "message": "msg_1"}))
        else:
            await original(source, destination)

    sandbox.download = download
    response, _ = await episode(agent, sandbox)
    assert response.status == "completed"
    assert sandbox.translations == [{("ses_1", "msg_1"): "msg_gym_1"}]


async def test_model_call_capture_asks_the_plugin_for_labels(opencode2):
    agent, sandbox = opencode2
    with patch.object(type(agent), "_model_call_capture_enabled", return_value=True):
        response, _ = await episode(agent, sandbox)
    assert response.status == "completed"
    _, config = launch_config(sandbox)
    assert config["plugins"][0]["options"]["labelFile"] == f"{sandbox.directory}/model-call-labels.jsonl"


async def test_a_local_call_runs_opencode_2_standalone_and_reads_its_store(opencode2, tmp_path):
    from responses_api_agents.opencode_agent.tests.test_opencode2_store import ROOT, assistant, store, user

    agent, _ = opencode2
    log = tmp_path / "calls.json"
    source = store(
        tmp_path / "fixture.db",
        [(ROOT, "msg_user", *user("task", 1)), (ROOT, "msg_a1", *assistant(2, [{"type": "text", "text": "Done"}]))],
    )
    binary = tmp_path / "opencode"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, shutil, sys\n"
        f"json.dump({{'argv': sys.argv[1:], 'pwd': os.environ.get('PWD'), "
        f"'config': json.loads(os.environ['OPENCODE_CONFIG_CONTENT'])}}, open({str(log)!r}, 'w'))\n"
        "store = os.path.join(os.environ['XDG_DATA_HOME'], 'opencode')\n"
        "os.makedirs(store, exist_ok=True)\n"
        f"shutil.copy({str(source)!r}, os.path.join(store, 'opencode.db'))\n"
    )
    binary.chmod(0o755)
    agent.config.command = str(binary)
    agent.config.workspace_root = str(tmp_path / "workspaces")
    with patch("responses_api_agents.opencode_agent.app.ensure_opencode2") as install:
        output, usage, _, _ = await agent._run_opencode("task", None)
    install.assert_not_called()
    call = json.loads(log.read_text())
    assert call["argv"][:4] == ["run", "--standalone", "-m", "nemo/" + agent.config.model]
    assert call["argv"][-1] == "task"
    assert Path(call["pwd"]).name.startswith("opencode_")
    model = call["config"]["provider"]["nemo"]["models"][agent.config.model]
    assert model["modalities"] == {"input": ["text"], "output": ["text"]}
    assert call["config"]["plugins"][-1]["package"] == PLUGIN_SOURCE.parent.as_uri()
    assert "plugin" not in call["config"]
    assert output[-1].content[0].text == "Done"
    assert usage["output_tokens"] > 0


def test_token_capture_knows_the_step_limit_note_of_the_pinned_opencode_2():
    from nemo_gym.models import HARNESS_NOTES

    notes = [note for note in HARNESS_NOTES if note.harness == "opencode"]
    assert notes and all(OPENCODE_V2_VERSION in note.checked_versions for note in notes)
