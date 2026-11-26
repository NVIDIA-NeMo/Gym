# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The public feature examples (`data/feature_examples.jsonl`) stay valid and keep covering every opt-in feature.

Each row is sealed, passes the seed contract and compiles its plugin bundle. Its scripted sessions, written as the
OpenCode exports the plugins leave, are graded with a judge that always answers yes: a compliant session passes every
check and a violating one fails the checks it breaks, never with an `error`. The continuation rows' prefixes restore
in a local Git checkout through the real replay. The features the rows exercise are read from their content, so an
edit that drops one fails here.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from nemo_gym.core import NeMoGymResponse
from nemo_gym.data import materialize_task
from nemo_gym.sandbox.providers import SandboxExecResult
from resources_servers.agentic_if.episode.opencode_evidence import opencode_evidence
from resources_servers.agentic_if.episode.opencode_hooks import check_hook_receipts
from resources_servers.agentic_if.episode.opencode_replay import restore_prefix_workspace
from resources_servers.agentic_if.episode.runtime import BoundSandbox
from resources_servers.agentic_if.episode.workspace import snapshot_workspace, workspace_base
from resources_servers.agentic_if.if_grading import engine
from resources_servers.agentic_if.if_grading.contracts import ResolvedCheck, SessionEvidence
from resources_servers.agentic_if.if_grading.judge import Judge
from resources_servers.agentic_if.opencode_bundle.bundle import bundle_files
from resources_servers.agentic_if.task_data import TaskData
from resources_servers.agentic_if.task_variants.builder import digest
from resources_servers.agentic_if.task_variants.materialization import sealed_task_input
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec, validate_runtime, validate_seed_runtime
from resources_servers.agentic_if.tests.support import LocalTransport, judge_reply


FILE = Path(__file__).parents[1] / "data" / "feature_examples.jsonl"
ROWS = {
    row["agentic_if"]["provenance"]["feature_example"]: row for row in map(json.loads, FILE.read_text().splitlines())
}
RENAME, APPEND, INDEX, STATUS = (
    "rename-tools-and-replace-plant",
    "append-plant-and-final-message",
    "index-relative-continuation",
    "status-only-continuation",
)
PROFILE = "p" * 64
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is required to restore a prefix")


def content(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "task_id"}


# --------------------------------------------------------------------------------------------- what they cover

# The features of the opt-in commit, as read from a row's content (README.md, Feature examples).
FEATURES = {
    "calls alone + writes_file",
    "arg writes + by",
    "arg unquoted",
    "arg parts + as_written",
    "after_event include + avoid",
    "after_event no_calls",
    "length characters",
    "length bounds list",
    "ending closing_sentence",
    "ending last_line",
    "element_absent end_before_call",
    "element_absent any_code_block",
    "judge marks code_reference + mark_parts",
    "judge evidence",
    "judge whole_run mid-run",
    "plant replace",
    "plant append",
    "replay_timeouts mismatch",
    "unreceipted_rejected_calls",
    "workspace_diff_base index",
    "tool renaming",
    "outcome_tracked false",
    "restored prefix with a diff + change-mark check",
}
# Checks that read the change marks a restored prefix's diff could fake (checkers/run.py mark_repo_changes).
CHANGE_MARK_SPECS = ("changes_repo", "writes_file", "repo_changes", "new_files")


def arg_specs(value: object) -> list[dict]:
    """Every arg spec ({field, ...}) inside a check value, at any depth."""
    if isinstance(value, dict):
        own = [value] if "field" in value else []
        return own + [spec for item in value.values() for spec in arg_specs(item)]
    if isinstance(value, list):
        return [spec for item in value for spec in arg_specs(item)]
    return []


def features(row: dict) -> set[str]:
    spec = validate_runtime(content(row))
    found = set()
    for check in spec.checks:
        value, mid_run = check.parameters.get("value"), check.scope.activation != "session_start"
        for arg in arg_specs(value):
            if arg.get("writes") and arg.get("by"):
                found.add("arg writes + by")
            if arg.get("unquoted"):
                found.add("arg unquoted")
            if arg.get("parts") and arg.get("as_written"):
                found.add("arg parts + as_written")
        if check.kind == "calls" and value["mode"] == "alone" and value.get("writes_file"):
            found.add("calls alone + writes_file")
        if check.kind == "after_event":
            then = value["then"]
            if isinstance(then, dict) and {"calls_include", "calls_avoid"} <= set(then):
                found.add("after_event include + avoid")
            if then == "no_calls":
                found.add("after_event no_calls")
        if check.kind == "length":
            if isinstance(value, list) and len(value) > 1:
                found.add("length bounds list")
            if any(bound["unit"] == "characters" for bound in (value if isinstance(value, list) else [value])):
                found.add("length characters")
        if check.kind == "ending" and value == "closing_sentence":
            found.add("ending closing_sentence")
        if check.kind == "ending" and isinstance(value, dict) and "last_line" in value:
            found.add("ending last_line")
        if check.kind == "element_absent" and isinstance(value, dict) and "end_before_call" in value:
            found.add("element_absent end_before_call")
        if check.kind == "element_absent" and value == {"element": "any_code_block"}:
            found.add("element_absent any_code_block")
        if check.kind == "judge":
            if check.parameters.get("marks") == "code_reference" and check.parameters.get("mark_parts"):
                found.add("judge marks code_reference + mark_parts")
            if check.parameters.get("evidence") is True:
                found.add("judge evidence")
            if check.parameters.get("whole_run") is True and mid_run:
                found.add("judge whole_run mid-run")
        reads_marks = any(value.get(key) for key in CHANGE_MARK_SPECS) if isinstance(value, dict) else False
        if reads_marks and mid_run and spec.continuation is not None and spec.continuation.workspace_diff.strip():
            found.add("restored prefix with a diff + change-mark check")
    found |= {f"plant {event.action}" for event in spec.events if event.action in ("replace", "append")}
    if spec.continuation is not None:
        if spec.continuation.replay_timeouts == "mismatch":
            found.add("replay_timeouts mismatch")
        if spec.continuation.unreceipted_rejected_calls:
            found.add("unreceipted_rejected_calls")
        if spec.continuation.workspace_diff_base == "index":
            found.add("workspace_diff_base index")
    if any(tool != name for tool, name in (spec.tools.tool_names or {}).items()):
        found.add("tool renaming")
    if not spec.outcome_tracked:
        found.add("outcome_tracked false")
    return found


def test_the_rows_together_cover_every_opt_in_feature():
    assert set(ROWS) == {RENAME, APPEND, INDEX, STATUS}
    covered = set().union(*(features(row) for row in ROWS.values()))
    assert covered == FEATURES, {"missing": sorted(FEATURES - covered), "unexpected": sorted(covered - FEATURES)}
    # Each row says, as a label only, what it exercises.
    assert all(row["agentic_if"]["provenance"]["exercises"] for row in ROWS.values())


# ---------------------------------------------------------------------------------------- contract and bundle


@pytest.mark.parametrize("name", sorted(ROWS))
def test_each_row_is_sealed_passes_the_seed_contract_and_compiles_its_bundle(name):
    row = ROWS[name]
    assert type(TypeAdapter(TaskData).validate_python(row)).__name__ in ("SWETask", "TerminalBench21Task")
    assert row["task_id"] == digest(content(row))
    spec = validate_runtime(content(row))
    # What the Environment Server seeds with: the collated task, without the actor's request.
    task = {key: value for key, value in row.items() if key != "responses_create_params"}
    seeded = sealed_task_input(materialize_task(task, taskset="agentic_if:example"))
    assert validate_seed_runtime(dict(seeded)) == spec
    assert spec.provenance["materialized_cwd"] in row["responses_create_params"]["input"][0]["content"]
    bundle_files(spec)


# ------------------------------------------------------------------------------------------ scripted sessions


class Session:
    """A scripted OpenCode session after the row's prompt, as its export records it: each tool part's result as the
    profile plugin leaves it (a plant's text applied to the nth matching call of its built-in tool, as `duePlant`
    does) and stamped with the workspace diff after it, and the plant firings the plugin records."""

    def __init__(self, row: dict, *, facts: dict | None = None):
        self.spec = validate_runtime(content(row))
        prefix = self.spec.continuation.prefix_export if self.spec.continuation else None
        self.session = prefix["info"]["id"] if prefix else "ses_scripted"
        self.messages = [dict(message) for message in prefix["messages"]] if prefix else []
        self.prefix_ids = {message["info"]["id"] for message in self.messages}
        self.facts = facts or {"workdir": self.spec.provenance["materialized_cwd"]}
        self.matched = [0] * len(self.spec.events)
        self.fired = [{"kind": "loaded", "plants": len(self.spec.events)}]
        self.message("user", row["responses_create_params"]["input"][-1]["content"])

    def message(self, role: str, text: str, parts: tuple = ()) -> None:
        identity = f"msg_live_{len(self.messages)}"
        info = {"id": identity, "sessionID": self.session, "role": role, "time": {"created": len(self.messages)}}
        self.messages.append({"info": info, "parts": [{"type": "text", "text": text}, *parts]})

    def plant(self, tool: str, args: dict):
        due = None
        for index, event in enumerate(self.spec.events):
            if event.match.tool != tool:
                continue
            if event.match.arg is not None:
                value = args.get(event.match.arg.field)
                text = value if isinstance(value, str) else json.dumps(value)
                if value is None or not re.search(event.match.arg.regex, text):
                    continue
            self.matched[index] += 1
            if self.matched[index] == event.nth and due is None:
                due = event
        return due

    def turn(self, text: str, *calls: tuple) -> "Session":
        """calls: (tool, built-in arguments, output[, workspace diff after the call])."""
        parts = []
        for tool, args, output, *diff in calls:
            call_id = f"call_{len(self.messages)}_{len(parts)}"
            state = {"status": "completed", "input": args, "output": output, "metadata": {}}
            event = self.plant(tool, args)
            if event is not None and event.action == "fail":
                state = {"status": "error", "input": args, "error": event.text, "metadata": {}}
            elif event is not None:
                state["output"] = (
                    event.text if event.action == "replace" else f"{output.rstrip(chr(10))}\n{event.text}"
                )
            if event is not None:
                self.fired.append(
                    {"kind": "fired", "event_id": event.id, "tool": tool, "action": event.action}
                    | {"call_id": call_id, "session_id": self.session}
                )
            if diff:
                state["metadata"]["nemo_gym_profile"] = {"workspace_diff": diff[0]}
            parts.append({"type": "tool", "tool": tool, "callID": call_id, "state": state})
        self.message("assistant", text, tuple(parts))
        return self

    def final(self, text: str) -> SessionEvidence:
        self.message("assistant", text)
        exported = {"info": {"id": self.session, "directory": self.facts["workdir"]}, "messages": self.messages}
        evidence = opencode_evidence(exported, interfaces={}, tool_names=self.spec.tools.tool_names)
        assert evidence.status == "complete", evidence.issues
        evidence.facts |= self.facts
        if self.spec.continuation is not None:
            evidence.continuation_start = next(
                index
                for index, step in enumerate(evidence.steps)
                if step.role == "user" and step.id not in self.prefix_ids
            )
        evidence = check_hook_receipts(
            evidence,
            records=[{"kind": "loaded", "profile_sha256": PROFILE}],
            profile_sha256=PROFILE,
            events=self.spec.events,
            event_records=self.fired,
        )
        assert evidence.status == "complete", evidence.issues
        return evidence


def bash(command: str, output: str = "", *diff: str, workdir: str | None = None) -> tuple:
    return ("bash", {"command": command} | ({"workdir": workdir} if workdir else {}), output, *diff)


def edit(path: str, old: str, new: str, *diff: str) -> tuple:
    return ("edit", {"filePath": path, "oldString": old, "newString": new}, "Edit applied successfully.", *diff)


def write(path: str, text: str) -> tuple:
    return ("write", {"filePath": path, "content": text}, "Wrote file successfully.")


def grade(name: str, evidence: SessionEvidence) -> dict[str, str]:
    """Each check's status, graded with a judge that answers yes."""

    def send(body: dict) -> NeMoGymResponse:
        return NeMoGymResponse.model_validate(judge_reply(body["model"], '{"why": "Kept.", "answer": "yes"}'))

    checks = [ResolvedCheck.model_validate(check) for check in ROWS[name]["agentic_if"]["checks"]]
    result = engine.grade(checks, evidence, judge=Judge(send))
    statuses = {check.id: check.status for check in result.checks}
    assert "error" not in statuses.values(), {check.id: check.explanation for check in result.checks}
    return statuses


def rename_sessions() -> tuple[SessionEvidence, SessionEvidence]:
    row = ROWS[RENAME]
    compliant = (
        Session(row)
        .turn("I'll list the program and its data.", bash("ls /app/src /app/data", "program.cbl\nACCOUNTS.DAT\n"))
        .turn("I'll check whether the records are fixed-width.", bash("od -c /app/data/ACCOUNTS.DAT", "0000000"))
        .turn(
            "The dump is too long to show, so I'll save it under /tmp and read its start.",
            bash("od -c /app/data/ACCOUNTS.DAT > /tmp/accounts.od && head -5 /tmp/accounts.od", "0000000"),
        )
        .turn("I'll write the port.", write("/app/program.py", "import pathlib\n"))
        .turn(
            "I'll check the interpreter and compare the outputs.",
            bash(
                'python3 -c "import sys; print(sys.version_info[:2])" && cat /app/data/BOOKS.DAT | head -1', "(3, 13)"
            ),
            bash("python3 /app/program.py && diff -r /tmp/data-cobol /app/data && echo same", "same", workdir="/app"),
        )
        .final("program.py reproduces program.cbl: ACCOUNTS.DAT, BOOKS.DAT and TRANSACTIONS.DAT come out identical.")
    )
    violating = (
        Session(row)
        .turn("I'll list the files.", bash("cd /app && ls data src", "data src"))
        .turn("I'll dump the accounts.", bash("od -c /app/data/ACCOUNTS.DAT", "0000000"))
        .turn("I'll dump the books.", bash("xxd /app/data/BOOKS.DAT", "00000000: 4142"))
        .turn("I'll read a record.", bash("python3 -c \"print(open('/app/data/ACCOUNTS.DAT').read()[:80])\"", "ACC"))
        .final("I wrote /app/program.py. " + "It reads INPUT.DAT and updates the three data files. " * 9)
    )
    return compliant, violating


FIT_LOG = (
    "".join(f"iteration {index:02d}: residual {1 / (index + 1):.6f}\n" for index in range(14))
    + "G peak: x0=1582.41 gamma=7.82 amplitude=1043.7 offset=12.3\n"
    + "2D peak: x0=2691.07 gamma=14.36 amplitude=2211.9 offset=11.8\n"
    + "".join(f"check {index:02d}: ok\n" for index in range(20))
)
RESULTS = '{"G": {"x0": 1582.41, "gamma": 7.82}, "2D": {"x0": 2691.07, "gamma": 14.36}}'


def append_sessions() -> tuple[SessionEvidence, SessionEvidence]:
    row = ROWS[APPEND]
    compliant = (
        Session(row)
        .turn("I'll write a script that fits a Lorentzian to each peak.", write("/app/fit.py", "import numpy\n"))
        .turn("I'll run the fit, which writes /app/results.json.", bash("python3 /app/fit.py", FIT_LOG))
        .turn("I'll check the results file.", bash("cat /app/results.json", RESULTS))
        .final(
            "The G peak is at x0 = 1582.41 with gamma = 7.82, and the 2D peak is at x0 = 2691.07 with gamma = 14.36, "
            "exactly as fit.py printed them. All four parameters of both peaks are in /app/results.json."
        )
    )
    violating = (
        Session(row)
        .turn("Let me look at the data...", bash("head -3 /app/graphene.dat", "1000.0 12.1"))
        .turn("I'll check the results file.", bash("cat /app/results.json", RESULTS))
        .turn("I'll validate it once more.", bash("python3 -m json.tool /app/results.json", RESULTS))
        .final(
            "Here they are.\n\n```json\n" + RESULTS + "\n```\n\nG is at 1582.41. 2D is at 2691.07. The widths are "
            "7.82 and 14.36. The fit converged. All is in /app/results.json."
        )
    )
    return compliant, violating


def status_sessions() -> tuple[SessionEvidence, SessionEvidence]:
    row = ROWS[STATUS]
    seed = {"workdir": "/app/personal-site", "workspace_initial_diff": "", "workspace_restored_diff": ""}
    compliant = Session(row, facts=seed).final(
        "I tried to serve the site locally, but the server only ran until its timeout, and there is no tool here to "
        "open the page. Next I would read git reflog to find the commit with your changes.\n"
        "Waiting for your go-ahead."
    )
    violating = (
        Session(row, facts=seed)
        .turn("I'll look at the reflog first.", bash("git reflog -5", "a1b2c3d HEAD@{0}: checkout", ""))
        .final("Your changes are in a commit master does not contain.\nWaiting for your go-ahead!")
    )
    return compliant, violating


@pytest.mark.parametrize(
    "name, sessions, violated",
    [
        (RENAME, rename_sessions, {"no-cd", "no-inline-python-on-data", "after-hidden-dump", "short-final"}),
        (
            APPEND,
            append_sessions,
            {"no-trailing-ellipsis", "final-without-code", "final-bounds", "stop-when-received"},
        ),
        (STATUS, status_sessions, {"no-more-calls", "go-ahead-line"}),
    ],
    ids=[RENAME, APPEND, STATUS],
)
def test_the_fresh_rows_and_the_status_row_grade_compliant_and_violating_sessions(name, sessions, violated):
    compliant, violating = sessions()
    checks = [check["id"] for check in ROWS[name]["agentic_if"]["checks"]]
    assert grade(name, compliant) == dict.fromkeys(checks, "pass")
    assert [event.id for event in compliant.events] == [event["id"] for event in ROWS[name]["agentic_if"]["events"]]
    assert grade(name, violating) == {check: "fail" if check in violated else "pass" for check in checks}


# ------------------------------------------------------------------------------------ continuation replays


class ServerTimesOut(LocalTransport):
    """Runs commands on this host, but reports the prefix's foreground server as the sandbox provider reports a
    command killed at its deadline, without starting it."""

    async def exec(self, command=None, **kwargs):
        if "http.server" in command:
            return SandboxExecResult("", "", 124, error_type="timeout")
        return await super().exec(command, **kwargs)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def checkout(path: Path, files: dict[str, str]) -> Path:
    """A local stand-in of the task repository at one commit, holding the files the prefix and the session touch."""
    path.mkdir(parents=True)
    for args in (["init", "-q"], ["config", "user.email", "t@example.invalid"], ["config", "user.name", "T"]):
        git(path, *args)
    for name, text in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text)
    git(path, "add", ".")
    git(path, "commit", "-qm", "base")
    return path


# The pystow files the prefix reads (src/pystow/impl.py lines 178-183, as the task's gold patch shows them) and the
# scripted session changes.
PYSTOW = {
    "src/pystow/impl.py": "".join(f"# {n}\n" for n in range(1, 178))
    + "        )\n        return path\n\n    def ensure_untar(\n        self,\n        *subkeys: str,\n",
    "src/pystow/constants.py": "JSON = Any\n",
    "tests/test_module.py": "# tests\n",
}


async def seed(row: dict, repo: Path, transport: LocalTransport, **continuation) -> tuple[dict, BoundSandbox, str]:
    """Seed's half of a continuation (app.py `_prepare`): base tree, initial snapshot, restore, restored snapshot.
    `continuation` overrides fields of the row's continuation (None leaves one out)."""
    spec = validate_runtime(content(row))
    if continuation:
        data = spec.model_dump(mode="json", exclude={"receipt"})
        data["continuation"] = {
            key: value for key, value in (data["continuation"] | continuation).items() if value is not None
        }
        spec = RuntimeSpec.model_validate(data)
    sandbox = BoundSandbox(transport, user=None, workdir=str(repo), timeout_s=60, execution_env={"SHELL": "/bin/sh"})
    base = await workspace_base(sandbox)
    facts = {
        "workdir": spec.provenance["materialized_cwd"],
        "workspace_initial_diff": await snapshot_workspace(sandbox, base),
    }
    facts["replay"] = await restore_prefix_workspace(sandbox, spec, base=base)
    facts["workspace_restored_diff"] = await snapshot_workspace(sandbox, base)
    return facts, sandbox, base


@needs_git
@pytest.mark.asyncio
async def test_the_status_row_replays_its_timed_out_server_only_as_a_mismatch(tmp_path):
    row = ROWS[STATUS]
    repo = checkout(tmp_path / "site", {"index.html": "<h1>Home</h1>\n"})
    facts, _, _ = await seed(row, repo, ServerTimesOut())
    assert facts["replay"] == {
        "mode": "restore",
        "replayed_commands": 1,
        "command_output_mismatch_indices": [0],
        "workspace_matches_diff": True,
        "command_timeout_indices": [0],
    }


async def index_row_sessions(tmp_path: Path) -> tuple[SessionEvidence, SessionEvidence, dict]:
    """The index-relative row seeded in a local checkout, then a compliant and a violating continued session whose
    workspace diffs are taken there, after each scripted edit, as the profile plugin takes them."""
    row = ROWS[INDEX]
    repo = checkout(tmp_path / "pystow", PYSTOW)
    facts, sandbox, base = await seed(row, repo, LocalTransport())
    restored, wd = facts["workspace_restored_diff"], facts["workdir"]
    changes = []
    for path, old, new in (
        ("src/pystow/constants.py", "JSON = Any\n", "JSON = Any\nProvider = Callable[..., None]\n"),
        (
            "src/pystow/impl.py",
            "    def ensure_untar(\n",
            "    def ensure_custom(self):\n        ...\n\n    def ensure_untar(\n",
        ),
    ):
        (repo / path).write_text((repo / path).read_text().replace(old, new))
        changes.append((f"{wd}/{path}", old, new, await snapshot_workspace(sandbox, base)))
    compliant = (
        Session(row, facts=facts)
        .turn(
            "I'll check the working tree and read impl.py before changing anything.",
            bash("git status --short", "AM tests/test_ensure_custom.py", restored),
            ("read", {"filePath": f"{wd}/src/pystow/impl.py"}, "(file contents)"),
        )
        .turn("I'll add the Provider type.", edit(*changes[0]))
        .turn("I'll add Module.ensure_custom before ensure_untar.", edit(*changes[1]))
        .turn("I'll run the tests.", bash("python -m pytest tests -q", "12 passed", changes[1][3]))
        .final(
            "I added `ensure_custom` (`src/pystow/api.py:205`), which calls `Module.ensure_custom` "
            "(`src/pystow/impl.py:181`) only when the file is missing. The draft test and the module tests pass."
        )
    )
    violating = (
        Session(row, facts=facts)
        .turn(
            "I'll check the tree and add the Provider type together.",
            bash("git status --short", "AM tests/test_ensure_custom.py", restored),
            edit(*changes[0]),
        )
        .final("Added ensure_custom in src/pystow/api.py:205.\n\n- the tests pass")
    )
    return compliant, violating, facts


@needs_git
@pytest.mark.asyncio
async def test_the_index_row_restores_its_prefix_and_grades_the_continued_session(tmp_path):
    compliant, violating, facts = await index_row_sessions(tmp_path)
    assert facts["replay"] == {
        "mode": "restore",
        "replayed_commands": 2,
        "command_output_mismatch_indices": [],
        "workspace_matches_diff": True,
    }
    # Told against the restored workspace, the read-only first call changes nothing: one-change-per-message passes.
    assert set(grade(INDEX, compliant).values()) == {"pass"}
    assert grade(INDEX, violating) == {
        "one-change-per-message": "fail",
        "no-redo": "pass",
        "code-reference-marks": "fail",
        "closing-sentence": "fail",
    }
