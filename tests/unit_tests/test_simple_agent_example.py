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
"""Tests for examples/simple_agent.py, using a fake `gym` executable for the process orchestration."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).parents[2]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("simple_agent_example", REPO_ROOT / "examples" / "simple_agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["simple_agent_example"] = module
    spec.loader.exec_module(module)
    return module


simple_agent_example = _load_script()


class TestRouteArgs:
    def test_splits_flags_between_start_and_collection(self) -> None:
        routed = simple_agent_example.route_args(
            [
                "--config", "a.yaml", "--config=b.yaml", "-m", "gpt",
                "-i", "in.jsonl", "-o", "out/../runs/r.jsonl", "--num-repeats", "2",
                "-v", "--search-dir", "/extra", "+responses_create_params={max_output_tokens: 10}",
            ],
            cwd=Path("/work"),
        )  # fmt: skip
        override = "+responses_create_params={max_output_tokens: 10}"
        assert routed.start == [
            "--config",
            "a.yaml",
            "--config",
            "b.yaml",
            "-m",
            "gpt",
            "-v",
            "--search-dir",
            "/extra",
            override,
        ]
        assert routed.collect == [
            "-i", "in.jsonl", "-o", "/work/runs/r.jsonl", "--num-repeats", "2", "-v", "--search-dir", "/extra", override,
        ]  # fmt: skip
        assert routed.output == Path("/work/runs/r.jsonl")
        assert routed.collect_dests == {"input", "output", "num_repeats"}

    def test_arguments_after_separator_go_to_collection_verbatim(self) -> None:
        routed = simple_agent_example.route_args(
            ["--config", "a.yaml", "--", "--config", "x", "-o", "rel.jsonl"], cwd=Path("/w")
        )
        assert routed.start == ["--config", "a.yaml"]
        assert routed.collect == ["--config", "x", "-o", "rel.jsonl"]
        assert "output" in routed.collect_dests and routed.output is None

    @pytest.mark.parametrize(
        "argv, message",
        [
            (["--no-serve"], "unsupported argument"),
            (["--bogus"], "unsupported argument"),
            (["--config"], "expects a value"),
        ],
    )
    def test_rejects(self, argv: list[str], message: str) -> None:
        with pytest.raises(SystemExit, match=message):
            simple_agent_example.route_args(argv, cwd=Path("/w"))

    def test_own_options_are_separated(self) -> None:
        own, rest = simple_agent_example.parse_args(
            ["--harness-cwd", "/repo", "--config", "c", "--", "--harness-cwd", "x"]
        )
        assert own.harness_cwd == "/repo" and own.gym_workdir is None
        assert rest == ["--config", "c", "--", "--harness-cwd", "x"]


def test_gym_workdir(tmp_path: Path) -> None:
    # This checkout is an editable install: Gym already works in the checkout.
    assert simple_agent_example.gym_workdir(None) == REPO_ROOT
    requested = tmp_path / "gym-work"
    assert simple_agent_example.gym_workdir(str(requested)) == requested and requested.is_dir()


def test_stop_servers_kills_the_whole_group_when_sigint_is_ignored() -> None:
    start = subprocess.Popen(
        ["bash", "-c", "trap '' INT; sleep 60 & echo $!; wait"],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    child = int(start.stdout.readline())
    simple_agent_example.stop_servers(start, grace_s=0.5)
    assert start.returncode == -signal.SIGKILL
    for _ in range(100):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail("a process in the server group survived teardown")


# A stand-in for the `gym` CLI. `env start` serves the head server's /readyz (503 until READY_AFTER
# seconds) and config on the ++head_server.port it is given, then waits for SIGINT; `eval run` records
# its arguments and exits with EVAL_EXIT_CODE.
FAKE_GYM = r"""#!{python}
import http.server, json, os, sys, threading, time
record = os.environ["FAKE_GYM_RECORD"]
with open(record, "a") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd(), "extra_roots": os.environ.get("NEMO_GYM_EXTRA_ROOTS")}) + "\n")
if sys.argv[1:3] == ["eval", "run"]:
    if os.environ.get("FAKE_ECHO_FILE"):  # stand in for simple_agent echoing items during collection
        with open(os.environ["FAKE_ECHO_FILE"], "a") as f:
            f.write(os.environ.get("FAKE_ECHO_TEXT", "[ab12:1] assistant\nhello\n\n"))
    print('{\n  "mean/reward": 1.0\n}', flush=True)  # the collection's own metrics output
    sys.exit(int(os.environ.get("EVAL_EXIT_CODE", "0")))
if os.environ.get("START_EXIT_CODE"):
    print(os.environ.get("START_MESSAGE", "server exploded"), flush=True)
    sys.exit(int(os.environ["START_EXIT_CODE"]))
port = int(next(a for a in sys.argv if a.startswith("++head_server.port=")).split("=")[1])
started = time.monotonic()
config = {"policy_model": {"responses_api_models": {}}}
config.update({name: {"responses_api_agents": {}} for name in os.environ.get("AGENTS", "").split(",") if name})
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/readyz":
            ready = time.monotonic() - started > float(os.environ.get("READY_AFTER", "0"))
            body, status = b"{}", 200 if ready else 503
        else:
            body, status = json.dumps(json.dumps(config)).encode(), 200
        self.send_response(status); self.end_headers(); self.wfile.write(body)
    def log_message(self, *args):
        pass
server = http.server.HTTPServer(("127.0.0.1", port), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    with open(record, "a") as f:
        f.write(json.dumps({"stopped": True}) + "\n")
"""


@pytest.fixture
def fake_gym(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    gym = tmp_path / "bin" / "gym"
    gym.parent.mkdir()
    gym.write_text(FAKE_GYM.replace("{python}", sys.executable))
    gym.chmod(0o755)
    record = tmp_path / "record.jsonl"
    monkeypatch.setenv("FAKE_GYM_RECORD", str(record))
    monkeypatch.setattr(simple_agent_example, "gym_executable", lambda: str(gym))
    monkeypatch.setattr(simple_agent_example, "gym_workdir", lambda requested: tmp_path / "gym-workdir")
    (tmp_path / "gym-workdir").mkdir()
    return record


def _records(record: Path) -> list[dict]:
    return [json.loads(line) for line in record.read_text().splitlines()]


def _main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    harness = tmp_path / "target-repo"
    harness.mkdir(exist_ok=True)
    monkeypatch.chdir(harness)
    try:
        return simple_agent_example.main(argv)
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> int:
    return _main(
        tmp_path, monkeypatch, ["--config", "c.yaml", "-i", "tasks.jsonl", "-o", "../out/rollouts.jsonl", *extra]
    )


def test_serves_collects_and_stops(fake_gym: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTS", "only_agent")
    monkeypatch.setenv("READY_AFTER", "1")
    monkeypatch.setenv("EVAL_EXIT_CODE", "3")

    assert _run(tmp_path, monkeypatch) == 3

    start, collect, stopped = _records(fake_gym)
    port = next(a for a in start["argv"] if a.startswith("++head_server.port="))
    harness = tmp_path / "target-repo"
    assert start["argv"][:4] == ["env", "start", "--config", "c.yaml"]
    assert f'+harness_cwd="{harness}"' in start["argv"]
    assert collect["argv"] == [
        "eval", "run", "--no-serve", "-i", "tasks.jsonl", "-o", str(tmp_path / "out" / "rollouts.jsonl"),
        "++head_server.host=127.0.0.1", port, "--agent", "only_agent",
    ]  # fmt: skip
    # Gym runs in its own working directory; relative inputs are still looked up in the harness cwd first.
    assert start["cwd"] == collect["cwd"] == str(tmp_path / "gym-workdir")
    assert collect["extra_roots"].split(os.pathsep)[0] == str(harness)
    assert stopped == {"stopped": True}  # stopped gracefully with SIGINT
    assert (tmp_path / "out" / "rollouts_servers.log").exists()


def test_explicit_agent_and_harness_cwd_are_kept(
    fake_gym: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTS", "a1,a2")
    assert _run(tmp_path, monkeypatch, "-a", "a2", "+harness_cwd=/elsewhere") == 0
    start, collect, _ = _records(fake_gym)
    assert "+harness_cwd=/elsewhere" in start["argv"] and not any(
        a.startswith('+harness_cwd="') for a in start["argv"]
    )
    assert collect["argv"].count("-a") == 1 and "--agent" not in collect["argv"]


def test_servers_stop_gracefully_even_if_launched_with_sigint_ignored(
    fake_gym: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Background jobs of non-interactive shells start with SIGINT ignored, and children inherit that.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    assert _run(tmp_path, monkeypatch, "-a", "agent") == 0
    assert _records(fake_gym)[-1] == {"stopped": True}


def test_startup_failure_reports_the_server_log(
    fake_gym: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv("START_EXIT_CODE", "1")
    assert _run(tmp_path, monkeypatch) == 1
    assert len(_records(fake_gym)) == 1  # no collection was attempted
    err = capsys.readouterr().err
    assert "exited with code 1 before the servers were ready" in err and "server exploded" in err


@pytest.mark.parametrize("argv, missing", [(["-i", "x.jsonl"], "-o/--output"), (["-o", "x.jsonl"], "-i/--input")])
def test_input_and_output_are_required(argv: list[str], missing: str) -> None:
    with pytest.raises(SystemExit, match=missing):
        simple_agent_example.main(argv)


def test_prompt_serves_codex_tools_on_a_generated_task(
    fake_gym: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTS", "codex_tools_simple_agent")
    argv = ["-m", "gpt", "--prompt", "Fix the tests.", "--check-command", "pytest -q", "-o", "../out/r.jsonl"]

    assert _main(tmp_path, monkeypatch, argv) == 0

    start, collect, _ = _records(fake_gym)
    task_path = tmp_path / "out" / "r_task.jsonl"
    assert start["argv"][2:6] == ["-m", "gpt", "--resources-server", "codex_tools"]
    assert collect["argv"][collect["argv"].index("--input") + 1] == str(task_path)
    (row,) = [json.loads(line) for line in task_path.read_text().splitlines()]
    assert row["responses_create_params"]["input"][-1] == {"role": "user", "content": "Fix the tests."}
    assert row["verifier_metadata"] == {"check_command": "pytest -q"}
    assert [tool["name"] for tool in row["responses_create_params"]["tools"]] == [
        "exec_command", "write_stdin", "apply_patch", "update_plan",
    ]  # fmt: skip


@pytest.mark.parametrize(
    "selection",
    [["--config", "c.yaml"], ["--resources-server", "mcqa"], ["--environment", "e"], ["+config_paths=[c.yaml]"]],
)
def test_explicit_environment_disables_the_default(selection: list[str]) -> None:
    routed = simple_agent_example.route_args([*selection, "-i", "x", "-o", "y"], cwd=Path("/w"))
    assert simple_agent_example.selects_environment(routed)
    assert not simple_agent_example.selects_environment(simple_agent_example.route_args(["-m", "gpt"], cwd=Path("/w")))


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--prompt", "p", "-i", "x.jsonl", "-o", "y.jsonl"], "either --prompt or -i/--input"),
        (["--check-command", "true", "-i", "x.jsonl", "-o", "y.jsonl"], "--check-command requires --prompt"),
        (["--prompt", "p"], "-o/--output is required"),
    ],
)
def test_prompt_argument_errors(argv: list[str], message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        simple_agent_example.main(argv)


def test_prompt_config_is_not_mistaken_for_prompt() -> None:
    own, rest = simple_agent_example.parse_args(["--prompt-config", "p.yaml", "-o", "y"])
    assert own.prompt is None and rest == ["--prompt-config", "p.yaml", "-o", "y"]


def test_prompt_echoes_pretty_items_by_default(
    fake_gym: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    echo_file = tmp_path / "out" / "r_items.log"
    monkeypatch.setenv("FAKE_ECHO_FILE", str(echo_file))
    monkeypatch.setenv("AGENTS", "codex_tools_simple_agent")

    assert _main(tmp_path, monkeypatch, ["--prompt", "Fix it.", "-o", "../out/r.jsonl"]) == 0

    start = _records(fake_gym)[0]
    assert "+simple_agent_echo_items=pretty" in start["argv"]
    assert f'+simple_agent_echo_file="{echo_file}"' in start["argv"]
    assert "[ab12:1] assistant\nhello\n\n" in capsys.readouterr().out
    assert echo_file.read_text() == "[ab12:1] assistant\nhello\n\n"


@pytest.mark.parametrize(
    "extra, expected",
    [
        ([], None),  # no --prompt: off by default
        (["--echo", "json"], "json"),
        (["--echo", "off"], None),
        (["--echo", "json", "+simple_agent_echo_items=pretty"], "pretty"),  # explicit overrides win
    ],
)
def test_echo_selection(
    fake_gym: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str], expected: str | None
) -> None:
    assert _run(tmp_path, monkeypatch, "-a", "agent", *extra) == 0
    start_argv = _records(fake_gym)[0]["argv"]
    echo_items = [a.split("=", 1)[1] for a in start_argv if a.startswith("+simple_agent_echo_items=")]
    assert echo_items == ([expected] if expected else [])
    echo_files = [a for a in start_argv if a.startswith("+simple_agent_echo_file=")]
    if expected == "json":
        assert echo_files == [f'+simple_agent_echo_file="{tmp_path / "out" / "rollouts_items.jsonl"}"']
    elif expected is None:
        assert echo_files == []


def test_json_echo_keeps_stdout_pure_json_lines(
    fake_gym: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture
) -> None:
    items = '{"type": "message", "role": "user", "content": "hi"}\n{"type": "function_call", "name": "x"}\n'
    monkeypatch.setenv("FAKE_ECHO_FILE", str(tmp_path / "out" / "rollouts_items.jsonl"))
    monkeypatch.setenv("FAKE_ECHO_TEXT", items)

    assert _run(tmp_path, monkeypatch, "-a", "agent", "--echo", "json") == 0

    out, err = capfd.readouterr()
    assert out == items
    assert '"mean/reward": 1.0' in err


@pytest.mark.parametrize(
    "uv_on_path, extra, expected",
    [
        (True, [], ["+skip_venv_if_present=true"]),  # default
        (False, [], ["+skip_venv_if_present=true"]),
        (True, ["--skip-venv-if-present"], ["+skip_venv_if_present=true"]),
        (True, ["--no-skip-venv-if-present"], ["+skip_venv_if_present=false"]),
        (False, ["--no-skip-venv-if-present"], ["+skip_venv_if_present=false"]),  # user's choice, with a warning
        (True, ["+skip_venv_if_present=false"], ["+skip_venv_if_present=false"]),  # explicit override wins
    ],
)
def test_skip_venv_if_present(
    fake_gym: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    uv_on_path: bool,
    extra: list[str],
    expected: list[str],
) -> None:
    monkeypatch.setattr(
        simple_agent_example.shutil, "which", lambda name, path=None: "/bin/uv" if uv_on_path else None
    )
    assert _run(tmp_path, monkeypatch, "-a", "agent", *extra) == 0
    start_argv = _records(fake_gym)[0]["argv"]
    assert [a for a in start_argv if a.lstrip("+").startswith("skip_venv_if_present=")] == expected
    warned = "which is not on PATH" in capsys.readouterr().err
    assert warned == (not uv_on_path and "--no-skip-venv-if-present" in extra)


def test_missing_uv_hint_on_startup_failure(
    fake_gym: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(simple_agent_example.shutil, "which", lambda name, path=None: None)
    monkeypatch.setenv("START_EXIT_CODE", "1")
    monkeypatch.setenv("START_MESSAGE", "(my_server) /bin/bash: line 1: uv: command not found")
    assert _run(tmp_path, monkeypatch) == 1
    assert "requires uv: install it" in capsys.readouterr().err
