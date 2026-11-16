# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox sessions that run OpenCode 2 (``opencode_version: 2.x``) through the shared session lifecycle."""

import hashlib
import json
import shlex
import shutil
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from responses_api_agents.opencode_agent.opencode2 import PLUGIN_SOURCE
from responses_api_agents.opencode_agent.opencode2 import config as opencode2_config
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
        return []

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
    # Code Mode is off: its `execute` tool is denied after every other rule.
    assert list(config["permission"])[-1] == "execute" and config["permission"]["execute"] == "deny"
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
        "permission": {"execute": "allow", "*": "allow"},
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
    # A configured rule cannot re-enable Code Mode.
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
    with patch.object(type(agent), "_model_call_capture_enabled", return_value=True):
        response, _ = await episode(agent, sandbox)
    assert response.status == "completed"
    assert sandbox.translations == [{("ses_1", "msg_1"): "msg_gym_1"}]


async def test_without_capture_no_label_file_is_read(opencode2):
    agent, sandbox = opencode2
    original = sandbox.download
    read = []

    async def download(source, destination):
        read.append(source)
        await original(source, destination)

    sandbox.download = download
    response, closed = await episode(agent, sandbox)
    assert response.status == "completed"
    assert not [source for source in read if source.endswith("/model-call-labels.jsonl")]
    assert sandbox.translations == [{}]
    assert "model_call_labels_unavailable" not in {gap.code for gap in closed.agent_observations.gaps}


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
        f"json.dump({{'argv': sys.argv[1:], 'pwd': os.environ.get('PWD'), 'env': dict(os.environ), "
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
    # A fixed title: OpenCode 2 would otherwise ask a model for one outside the rollout.
    assert call["argv"][:6] == ["run", "--standalone", "--title", "NeMo Gym", "-m", "nemo/" + agent.config.model]
    assert call["argv"][-1] == "task"
    assert Path(call["pwd"]).name.startswith("opencode_")
    model = call["config"]["provider"]["nemo"]["models"][agent.config.model]
    assert model["modalities"] == {"input": ["text"], "output": ["text"]}
    assert call["config"]["plugins"][-1]["package"] == PLUGIN_SOURCE.parent.as_uri()
    assert call["config"]["permission"]["execute"] == "deny"
    # The sandbox sessions' isolation: no project config, updates, model catalog or hosted free models.
    assert {key: call["env"][key] for key in opencode2_config.ISOLATION} == opencode2_config.ISOLATION
    assert "plugin" not in call["config"]
    assert output[-1].content[0].text == "Done"
    assert usage["output_tokens"] > 0


def test_token_capture_knows_the_step_limit_note_of_the_pinned_opencode_2():
    from nemo_gym.models import HARNESS_NOTES

    notes = [note for note in HARNESS_NOTES if note.harness == "opencode"]
    assert notes and all(OPENCODE_V2_VERSION in note.checked_versions for note in notes)


class _StoreSession:
    """The state ``OpenCodeSandboxSession.readable_store`` reads, around a sandbox whose label file can be swapped."""

    def __init__(self, labels_text):
        self.opencode2 = True
        self.model_call_labels = True
        self.persistent_directory = "/session"
        self.message_ids = []
        self._labels = {}
        self._capture_gaps = []
        self.labels_text = labels_text
        self.session = SimpleNamespace(sandbox=object())


async def _capture(state, directory):
    """One capture's translated message IDs (in order) and gap codes."""
    from responses_api_agents.opencode_agent.sandbox import OpenCodeSandboxSession

    async def read_text(sandbox, *, path):
        assert path == "/session/model-call-labels.jsonl"
        if state.labels_text is None:
            raise RuntimeError("download failed")
        return state.labels_text

    with patch("responses_api_agents.opencode_agent.sandbox.read_text", read_text):
        database = await OpenCodeSandboxSession.readable_store(state, directory, capture=True)
    with sqlite3.connect(database) as con:
        ids = [row[0] for row in con.execute("select id from message order by time_created, id")]
    return ids, [gap.code for gap in state._capture_gaps]


async def test_a_failed_label_read_never_renames_an_earlier_reply(tmp_path):
    """Earlier replies keep their IDs across captures, so the snapshot never returns them as new output."""
    from responses_api_agents.opencode_agent.tests.test_opencode2_store import ROOT, assistant, store, user

    def label(n):
        return json.dumps({"label": f"msg_gym_{n}", "session": ROOT, "message": f"msg_a{n}"}) + "\n"

    turns = []
    for n in (1, 2, 3):
        turns += [(ROOT, f"msg_u{n}", *user(f"turn {n}", 10 * n)), (ROOT, f"msg_a{n}", *assistant(10 * n + 1, []))]
    state = _StoreSession(label(1))
    store(tmp_path / "opencode.db", turns[:2])
    ids, gaps = await _capture(state, tmp_path)
    assert (ids, gaps) == (["msg_u1", "msg_gym_1"], [])
    state.message_ids = ids

    # The label file cannot be read: the earlier labels still apply, and the capture says its calls may not join.
    state.labels_text = None
    (tmp_path / "opencode.db").unlink()
    store(tmp_path / "opencode.db", turns[:4])
    ids, gaps = await _capture(state, tmp_path)
    assert (ids, gaps) == (["msg_u1", "msg_gym_1", "msg_u2", "msg_a2"], ["model_call_labels_unavailable"])
    state.message_ids = ids

    # Readable again: the reply that was captured unlabeled keeps its ID; only the new reply takes its label.
    state.labels_text = label(1) + label(2) + label(3)
    (tmp_path / "opencode.db").unlink()
    store(tmp_path / "opencode.db", turns)
    ids, gaps = await _capture(state, tmp_path)
    assert (ids, gaps) == (["msg_u1", "msg_gym_1", "msg_u2", "msg_a2", "msg_u3", "msg_gym_3"], [])


async def test_records_the_translation_cannot_carry_are_a_capture_gap(tmp_path):
    from responses_api_agents.opencode_agent.tests.test_opencode2_store import ROOT, store, user

    state = _StoreSession("")
    state.model_call_labels = False
    store(tmp_path / "opencode.db", [(ROOT, "msg_u1", *user("go", 1)), (ROOT, "msg_x", "shell", {"command": "ls"})])
    ids, gaps = await _capture(state, tmp_path)
    assert (ids, gaps) == (["msg_u1"], ["harness_records_untranslated"])
    assert state._capture_gaps[0].detail == "message type 'shell' x1"


PLUGIN_HARNESS = """
import { readFileSync, writeFileSync } from "node:fs"
const [plugin, options] = [(await import(process.argv[2])).default, JSON.parse(process.argv[3])]
const hooks = {}
const transforms = []
const steps = []
let wake
const ctx = {
  options,
  shell: { hook: async (name, callback) => { hooks[`shell.${name}`] = callback } },
  session: { hook: async (name, callback) => { (hooks[`session.${name}`] ??= []).push(callback) } },
  tool: { transform: async (callback) => { transforms.push(callback) } },
  mcp: {
    list: async () => ({
      data: [
        { name: "required", status: { status: "failed", error: "refused" } },
        { name: "optional", status: { status: "connected" } },
      ],
    }),
  },
  event: {
    subscribe: (filter) => {
      hooks.subscribed = filter.types
      return (async function* () {
        while (true) {
          if (!steps.length) await new Promise((resolve) => { wake = resolve })
          yield steps.shift()
        }
      })()
    },
  },
}
await plugin.setup(ctx)
const result = { registered: Object.keys(hooks).sort(), subscribed: hooks.subscribed }

const shellEvent = { env: { HOME: "/root", PATH: "/bin" } }
hooks["shell.create.before"](shellEvent)
result.shellEnv = shellEvent.env

const tools = new Map([
  ["shell", { id: "shell", options: { codemode: false } }],
  ["opencode_worktree_remove", { id: "opencode_worktree_remove", options: { namespace: "opencode", codemode: true, permission: "edit" } }],
  ["opencode_models", { id: "opencode_models", options: { namespace: "opencode" } }],
  ["weather_forecast", { id: "weather_forecast", options: { namespace: "weather", codemode: false } }],
])
for (const transform of transforms) transform({ list: () => [...tools.values()], remove: (id) => tools.delete(id) })
result.tools = [...tools.keys()]

result.sampled = []
for (const name of ["session.context", "session.compaction"]) {
  const request = { options: { maxOutputTokens: 9 } }
  for (const callback of hooks[name].slice(0, 1)) await callback(request)
  result.sampled.push(request.options)
}

const headers = []
for (const kind of ["primary", "primary", "compaction"]) {
  const event = { sessionID: "ses_1", kind, headers: { "x-session-id": "upstream" } }
  for (const callback of hooks["session.model.request"]) await callback(event)
  headers.push(event.headers)
}
result.headers = headers
steps.push({ type: "session.step.started", data: { sessionID: "ses_1", assistantMessageID: "msg_1" } })
wake?.()
await new Promise((resolve) => setTimeout(resolve, 50))
const next = { sessionID: "ses_1", kind: "primary", headers: {} }
for (const callback of hooks["session.model.request"]) await callback(next)
result.nextLabel = next.headers["X-OpenCode-Assistant-Message-Id"]
result.labels = readFileSync(options.labelFile, "utf8").trim().split("\\n").map((line) => JSON.parse(line))

try {
  for (const callback of hooks["session.prompt"]) await callback({ sessionID: "ses_1" })
  result.prompt = "accepted"
} catch (error) {
  result.prompt = error.message
}
result.mcpStatus = JSON.parse(readFileSync(options.mcpStatusFile, "utf8"))
result.ready = readFileSync(options.readyFile, "utf8")
console.log(JSON.stringify(result))
process.exit(0)
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for the OpenCode 2 plugin")
def test_the_plugin_registers_its_hooks_and_acts_on_them(tmp_path):
    """Runs the plugin against a stand-in for the OpenCode 2.2.7 plugin context.

    The hook and domain names (``shell.hook("create.before")``, ``session.hook`` for ``context``, ``compaction``,
    ``prompt`` and ``model.request``, ``tool.transform``, ``mcp.list``, ``event.subscribe``) are those of
    ``packages/plugin/src/promise`` at 2.2.7; the real-model runs in the PR check that OpenCode calls them.
    """
    harness = tmp_path / "harness.mjs"
    harness.write_text(PLUGIN_HARNESS)
    environment = {key: f"/task/{key.lower()}" for key in ("HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME")}
    environment |= {key: f"/task/{key.lower()}" for key in ("XDG_CACHE_HOME", "XDG_STATE_HOME")}
    (tmp_path / "task-env.json").write_text(json.dumps(environment))
    options = {
        "taskEnvironmentFile": str(tmp_path / "task-env.json"),
        "readyFile": str(tmp_path / "ready"),
        "sampling": {"temperature": 0.25, "top_p": 0.5},
        "mcpGrants": {"required": ["required"], "optional": ["optional"]},
        "mcpStatusFile": str(tmp_path / "mcp-status.json"),
        "mcpStartupMs": 1000,
        "activation": "agent-session:activation-2",
        "labelFile": str(tmp_path / "labels.jsonl"),
    }
    run = subprocess.run(
        [shutil.which("node"), str(harness), PLUGIN_SOURCE.as_uri(), json.dumps(options)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    result = json.loads(run.stdout)
    assert result["registered"] == [
        "session.compaction",
        "session.context",
        "session.model.request",
        "session.prompt",
        "shell.create.before",
        "subscribed",
    ]
    assert result["subscribed"] == ["session.step.started", "session.step.failed"]
    assert result["shellEnv"] == {**environment, "PATH": "/bin"}
    # Only tools offered outside Code Mode remain: OpenCode's own tools and the MCP grants.
    assert result["tools"] == ["shell", "weather_forecast"]
    assert result["sampled"] == [{"maxOutputTokens": 9, "temperature": 0.25, "topP": 0.5}] * 2
    first, second, compaction = result["headers"]
    assert first["X-Session-Id"] == "agent-session:activation-2:ses_1" and "x-session-id" not in first
    # Retries before the step starts reuse the label; compaction calls carry none.
    assert first["X-OpenCode-Assistant-Message-Id"] == second["X-OpenCode-Assistant-Message-Id"]
    assert "X-OpenCode-Assistant-Message-Id" not in compaction
    assert result["labels"] == [
        {"label": first["X-OpenCode-Assistant-Message-Id"], "session": "ses_1", "message": "msg_1"}
    ]
    assert result["nextLabel"] not in (None, first["X-OpenCode-Assistant-Message-Id"])
    assert result["prompt"] == "Required Gym MCP tools could not be initialized: required"
    assert result["mcpStatus"] == {
        "required": {"status": "failed", "error": "refused"},
        "optional": {"status": "connected"},
    }
    assert result["ready"] == "ready\n"
