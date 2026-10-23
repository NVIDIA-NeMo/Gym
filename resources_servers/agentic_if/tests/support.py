# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared builders and doubles for these tests.

OpenCode exports and boundary call records, sandbox transports (a scripted double and a local subprocess one),
sealed task rows, and a ServerClient double that plays the task backend and the judge model server.
"""

import asyncio
import hashlib
import json
import os
import shlex
import signal
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from aiohttp import ClientResponseError

from nemo_gym.base_resources_server import ResourcesSeedSessionResponse
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_correlation import current_rollout_id
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.providers import SandboxExecResult
from resources_servers.agentic_if.episode.session_evidence import row_prompt_texts
from resources_servers.agentic_if.if_grading.contracts import SessionInfo
from resources_servers.agentic_if.task_variants.builder import digest
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec, seal_runtime_row


json_module = json

# -------------------------------------------------------------------------------------- OpenCode exports and records


def message(identity, role, *parts, session="root", created=1):
    return {
        "info": {"id": identity, "role": role, "sessionID": session, "time": {"created": created}},
        "parts": list(parts),
    }


def text(value):
    return {"type": "text", "text": value}


def call(identity, tool="read", *, status="completed", end=3, **state):
    return {
        "type": "tool",
        "id": "part-" + identity,
        "callID": identity,
        "tool": tool,
        "state": {
            "status": status,
            "input": {"filePath": "/app/a.py"},
            "output": "contents",
            "time": {"start": 2, "end": end},
            **state,
        },
    }


def export(*messages, session="root", parent=None):
    return {"info": {"id": session, "parentID": parent}, "messages": list(messages)}


def observation(session="root", identity="a", *, tool="RunCommand", arguments='{ "command": "pwd" }'):
    return {
        "session_id": session,
        "assistant_message_id": identity,
        "execution_tools": {tool: "bash"},
        "request": {"messages": [{"role": "system", "content": "Explain each command."}]},
        "response": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Inspect",
                        "tool_calls": [{"id": "c1", "function": {"name": tool, "arguments": arguments}}],
                    }
                }
            ]
        },
    }


def session_export(session="root", parent=None):
    return export(
        message("u", "user", text("Fix it"), session=session),
        message(
            "a",
            "assistant",
            text("Inspect"),
            call("c1", "bash", input={"command": "pwd"}, output="/app", end=4),
            session=session,
            created=2,
        ),
        session=session,
        parent=parent,
    )


# The replay receipt seed records for a continuation that replayed no commands and matched its recorded diff.
REPLAY_RECEIPT = {
    "mode": "restore",
    "replayed_commands": 0,
    "command_output_mismatch_indices": [],
    "workspace_matches_diff": True,
}


def answered_export(session="root", parent=None):
    """The OpenCode session through the final reply that follows the tool result."""
    source = session_export(session, parent)
    source["messages"].append(message("z", "assistant", text("Done"), session=session, created=5))
    return source


def tool_observation(session="root", value="/app"):
    """The next actor request: it carries the tool result, and its reply ends the session."""
    return {
        "source": "model",
        "session_id": session,
        "assistant_message_id": "z",
        "execution_tools": {},
        "request": {"messages": [{"role": "tool", "tool_call_id": "c1", "content": value}]},
        "response": {"choices": [{"message": {"role": "assistant", "content": "Done"}}]},
    }


def exported(session, text="Done"):
    return {
        "info": {"id": session, "parentID": None},
        "messages": [
            {
                "info": {"id": "u", "role": "user", "sessionID": session, "time": {"created": 1}},
                "parts": [{"type": "text", "text": "Fix"}],
            },
            {
                "info": {
                    "id": "a",
                    "role": "assistant",
                    "sessionID": session,
                    "finish": "stop",
                    "time": {"created": 2, "completed": 3},
                },
                "parts": [{"type": "reasoning", "text": "Thinking"}, {"type": "text", "text": text}],
            },
        ],
    }


def finished_session(answer="DONE", *, session="ses_root", finish="stop", error=None, task="Fix it"):
    """A plain root session: the task, then one final reply, as OpenCode exports it."""
    info = {
        "id": "a1",
        "role": "assistant",
        "sessionID": session,
        "finish": finish,
        "time": {"created": 2, "completed": 3},
    }
    if error is not None:
        info["error"] = error
    return {
        "info": {"id": session, "parentID": None, "directory": "/app"},
        "messages": [
            {
                "info": {"id": "u1", "role": "user", "sessionID": session, "time": {"created": 1}},
                "parts": [{"type": "text", "text": task}],
            },
            {"info": info, "parts": [{"type": "text", "text": answer}]},
        ],
    }


def final_call_record(answer="DONE", *, session="ses_root", message_id="a1"):
    """The boundary's record of the model call that produced the final reply."""
    return {
        "source": "model",
        "session_id": session,
        "assistant_message_id": message_id,
        "request": {"messages": [{"role": "user", "content": "Fix it"}]},
        "execution_tools": {},
        "response": {"choices": [{"message": {"role": "assistant", "content": answer}}]},
    }


# ---------------------------------------------------------------------------------------------- sandbox doubles


class Transport:
    """A connected task sandbox: records every command and answers the identity and cwd probes; anything else (the Git
    baseline included) prints nothing."""

    def __init__(self):
        self.calls = []
        self.uploads = {}
        self.uid = "1001"
        self.stops = self.disconnects = 0
        self.fail = {}

    async def exec(self, command=None, **kwargs):
        self.calls.append({"command": command, **kwargs})
        for fragment, result in self.fail.items():
            if fragment in command:
                return result
        output = ""
        if command == "id -u":
            output = "0\n"
        elif command == "id -u && id -g":
            output = f"{self.uid}\n1001\n"
        elif "getent passwd" in command:
            output = "/home/agent\n/bin/bash\n"
        elif command.startswith("pwd -P &&"):
            # `pwd -P && (cd -- <expected> && pwd -P)`: this sandbox has no symlinks, so the path resolves to itself.
            expected = shlex.split(command.split("cd -- ", 1)[1].rsplit(" &&", 1)[0])[0]
            output = f"{kwargs.get('cwd') or '/app'}\n{expected}\n"
        return SandboxExecResult(stdout=output, stderr="", return_code=0)

    async def upload(self, source, destination):
        self.uploads[destination] = Path(source).read_bytes()

    async def download(self, source, destination):
        Path(destination).write_bytes(self.uploads[source])

    async def stop(self):
        self.stops += 1

    async def disconnect(self):
        self.disconnects += 1


class LocalTransport:
    """Runs commands on this host, as the current user, so Git and file effects are real."""

    def __init__(self):
        self.calls = []

    async def exec(self, command=None, *, user=None, cwd=None, timeout_s=None, env=None, **kwargs):
        self.calls.append((command, user))
        proc = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command,
            cwd=cwd,
            env=os.environ | (env or {}),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        communication = asyncio.create_task(proc.communicate())
        try:
            stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), timeout_s)
            code = proc.returncode
        except TimeoutError:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = await communication
            code = 124
        return SandboxExecResult(stdout=stdout.decode(), stderr=stderr.decode(), return_code=code)

    async def upload(self, source, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(Path(source).read_bytes())


def seeded(workdir="/app", *, session_id="attempt-a", sandbox_id="sandbox-a"):
    """The typed seed reply of a task backend that owns the sandbox."""
    return ResourcesSeedSessionResponse(
        resources_session_id=session_id,
        sandbox_access=SandboxAccess(
            connection=DirectSandboxConnection(provider_config_ref="sandbox", descriptor={"sandbox_id": sandbox_id}),
            workdir=workdir,
        ),
    ).model_dump(mode="json")


def response(status="completed", **fields):
    return NeMoGymResponse(
        id="actual-response",
        created_at=0,
        model="policy",
        object="response",
        output=[],
        tools=[],
        tool_choice="auto",
        parallel_tool_calls=False,
        status=status,
        **fields,
    )


# --------------------------------------------------------------------------------------------------- sealed rows

EXACTLY_DONE = [{"id": "end", "kind": "exact", "parameters": {"value": "DONE"}, "scope": {"places": ["final"]}}]
JUDGED = [{"id": "literal-rubric", "kind": "judge", "rubric": "Finish with DONE.", "scope": {"places": ["final"]}}]


def sealed(fields, *, backend="swe", text="Fix it", system=None, **configuration):
    """A sealed flat row, with its content-digest ``task_id``."""
    messages = [{"role": "user", "content": text}]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})
    configuration.setdefault("checks", EXACTLY_DONE)
    row = seal_runtime_row(
        dict(fields) | {"responses_create_params": {"input": messages}},
        RuntimeSpec(harness="opencode", task_backend=backend, **configuration),
    )
    return {"task_id": digest(row)} | row


def swe_row(**configuration):
    fields = {"instance_id": "task-a", "workdir": "/app", "image_url": "registry/task-a:1"}
    return sealed(fields | configuration.pop("fields", {}), backend="swe", **configuration)


def tb21_row(**configuration):
    fields = {"task_name": "terminal-bench/a", "docker_image": "registry/a:1", "task_folder": "tasks/a"}
    return sealed(fields | configuration.pop("fields", {}), backend="tb21", **configuration)


def task_data(row):
    """What single_agent_turn seeds with: the row without the actor's request."""
    return {key: value for key, value in row.items() if key != "responses_create_params"}


def verify_body(row, answer_response=None, **extra):
    """What single_agent_turn verifies with: the task data, the actor's request and the harness's response."""
    return (
        task_data(row)
        | {
            "responses_create_params": row["responses_create_params"],
            "response": (answer_response or response()).model_dump(mode="json"),
        }
        | extra
    )


# ---------------------------------------------------------------------------------------------- server doubles


def http_error(status):
    return ClientResponseError(request_info=SimpleNamespace(real_url="http://backend"), history=(), status=status)


class Backends:
    """The task backend and the judge model server, as ServerClient.post reaches them."""

    def __init__(self):
        self.calls = []
        self.seed_error = None
        self.verify_error = None
        self.close_error = None
        self.verdict = {"reward": 1.0, "evaluation_completed": True}
        # An exception every judge call raises, or a list of them (None answers) taken one per call.
        self.judge_error = None
        self.judge_text = '{"answer": "yes", "why": "The visible answer ends in DONE."}'
        # The token usage each judge reply carries (None: a reply without usage); replies are numbered from 1.
        self.judge_usage = JUDGE_USAGE
        self.judge_replies = 0
        self.global_config_dict = {"sandbox": {"opensandbox": {}}}

    async def post(self, *, server_name, url_path, json=None, cookies=None, **kwargs):
        self.calls.append(
            {
                "server_name": server_name,
                "url_path": url_path,
                "json": deepcopy(json),
                "cookies": cookies,
                # The rollout a real ServerClient would attribute this call to.
                "rollout_id": current_rollout_id(),
            }
        )
        if server_name == "judge":
            error = (
                (self.judge_error.pop(0) if self.judge_error else None)
                if isinstance(self.judge_error, list)
                else (self.judge_error)
            )
            if error is not None:
                raise error
            self.judge_replies += 1
            value = judge_reply(
                json["model"], self.judge_text, response_id=f"judge-reply-{self.judge_replies}", usage=self.judge_usage
            )
        elif url_path == "/seed_session":
            if self.seed_error is not None:
                raise self.seed_error
            value = seeded(session_id=json["resources_session_id"], sandbox_id=json["resources_session_id"])
        elif url_path == "/verify":
            if self.verify_error is not None:
                raise self.verify_error
            value = {key: item for key, item in json.items()} | self.verdict
        else:
            if self.close_error is not None:
                raise self.close_error
            value = {"resources_session_id": json["resources_session_id"]}

        async def read():
            return json_module.dumps(value).encode()

        return SimpleNamespace(ok=True, read=read, cookies={"backend": "cookie"}, status=200)

    def to(self, url_path):
        return [call for call in self.calls if call["url_path"] == url_path and call["server_name"] != "judge"]


JUDGE_USAGE = {
    "input_tokens": 1200,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 300,
    "output_tokens_details": {"reasoning_tokens": 120},
    "total_tokens": 1500,
}


def judge_reply(model, text, *, response_id="judge-reply", usage=None):
    return {
        "id": response_id,
        "object": "response",
        "created_at": 0,
        "model": model,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "id": "message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    } | ({"usage": usage} if usage is not None else {})


def bundle_outputs(
    exports=(),
    *,
    hook_receipts=None,
    event_records=(),
    call_records=(),
    gaps=(),
    boundary_issues=(),
    boundary_receipts=(),
):
    """What the plugin bundle reports at verify: exports keyed by session ID, their session inventory, the receipts,
    and the raw files as read."""
    exports = {document["info"]["id"]: document for document in exports}
    sessions = [
        SessionInfo(id=session_id, parent_id=document["info"].get("parentID"))
        for session_id, document in exports.items()
    ]
    hook_receipts = [{"kind": "loaded", "profile_sha256": "p" * 64}] if hook_receipts is None else list(hook_receipts)
    files = {f"exports/{session_id}.json": json.dumps(document).encode() for session_id, document in exports.items()}
    files |= {
        f"receipts/{name}": "".join(json.dumps(row) + "\n" for row in rows).encode()
        for name, rows in (
            ("hook-applied.jsonl", hook_receipts),
            ("hook-events-fired.jsonl", event_records),
            ("model-calls.jsonl", call_records),
        )
    }
    return SimpleNamespace(
        exports=exports,
        sessions=sessions,
        hook_receipts=hook_receipts,
        event_records=list(event_records),
        call_records=list(call_records),
        boundary_issues=list(boundary_issues),
        boundary_receipts=[dict(row) for row in boundary_receipts],
        gaps=list(gaps),
        files=files,
    )


def presented(row):
    """The bundle outputs of a run that showed the model the row as written, then answered at once.

    The boundary was configured with the row's system text, the model call carried it as its only system message, the
    root session started from the row's user text, and the plugins loaded the row's profile and planted events.
    """
    system, task = row_prompt_texts(row["responses_create_params"]["input"])
    messages = ([{"role": "system", "content": system}] if system is not None else []) + [
        {"role": "user", "content": task}
    ]
    record = final_call_record() | {"request": {"messages": messages}}
    configured = {
        "kind": "configured",
        "system_sha256": None if system is None else hashlib.sha256(system.encode()).hexdigest(),
    }
    events = row["agentic_if"].get("events") or []
    return bundle_outputs(
        [finished_session(task=task)],
        call_records=[record],
        boundary_receipts=[configured],
        event_records=[{"kind": "loaded", "plants": len(events)}] if events else (),
    )
