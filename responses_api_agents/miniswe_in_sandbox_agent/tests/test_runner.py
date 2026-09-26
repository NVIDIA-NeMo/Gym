# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The stdlib runner reproduces mini-swe-agent 2.4.6 / Gym server-harness semantics."""

import importlib.util
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml
from jinja2 import StrictUndefined, Template

from responses_api_agents.miniswe_in_sandbox_agent.app import RUNNER_SCRIPT, build_vendor_zip


TEMPLATES = yaml.safe_load((RUNNER_SCRIPT.parent / "mini_2_4_6.yaml").read_text())


def load_runner():
    spec = importlib.util.spec_from_file_location("miniswe_runner", RUNNER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.Runner._backoff = staticmethod(lambda attempt: None)  # keep retry tests fast
    return module


def config(tmp_path, **overrides):
    values = {
        "session_id": "tb4-test",
        "model_url": "http://127.0.0.1:9/v1",
        "headers": {"x-session-id": "tb4-test"},
        "request_params": {"max_output_tokens": 4096},
        "task": "Say hello.",
        "workdir": str(tmp_path / "work"),
        "env": TEMPLATES["environment"]["env"],
        "templates": {
            "system_template": TEMPLATES["agent"]["system_template"],
            "instance_template": TEMPLATES["agent"]["instance_template"],
            "observation_template": TEMPLATES["model"]["observation_template"],
            "format_error_template": TEMPLATES["model"]["format_error_template"],
        },
        "step_limit": 0,
        "step_timeout_sec": 30,
        "max_consecutive_format_errors": 3,
        "http_retries": 0,
        "budget_sec": 0,
        "pids_file": str(tmp_path / "pids"),
        "output_dir": str(tmp_path / "out"),
        "vendor_zip": None,
    }
    values.update(overrides)
    (tmp_path / "work").mkdir(exist_ok=True)
    return values


def call(command, call_id="call_1", text="Running."):
    return {
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            },
            {
                "id": call_id,
                "type": "function_call",
                "call_id": call_id,
                "name": "bash",
                "arguments": json.dumps({"command": command}),
                "status": "completed",
            },
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def scripted(runner, responses):
    queue = list(responses)
    module = sys.modules[type(runner).__module__]

    def post(body):
        assert body["tools"] == [module.BASH_TOOL] and body["max_output_tokens"] == 4096
        post.bodies.append(body)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    post.bodies = []
    return post


def render(template, **variables):
    return Template(template, undefined=StrictUndefined).render(**variables)


def test_submission_records_match_the_server_harness(tmp_path):
    module = load_runner()
    runner = module.Runner(config(tmp_path))
    runner.post_responses = scripted(
        runner, [call("printf 'a\\nb'", "c1"), call("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho done", "c2")]
    )
    info = runner.run()
    assert info["exit_status"] == "Submitted" and info["submission"] == "done\n"
    trajectory = json.loads((tmp_path / "out/trajectory.json").read_text())
    assert trajectory["trajectory_format"] == "mini-swe-agent-1.1"
    assert trajectory["info"]["exit_status"] == "Submitted" and trajectory["info"]["model_stats"]["api_calls"] == 2
    roles = [m["role"] for m in trajectory["messages"]]
    # system, instance, assistant, tool, assistant, exit — the submitting command's observation is omitted (mini-SWE).
    assert roles == ["system", "user", "assistant", "tool", "assistant", "exit"]
    assert trajectory["messages"][1]["content"].startswith("Please solve this issue: Say hello.")
    observation = trajectory["messages"][3]
    expected = render(
        TEMPLATES["model"]["observation_template"],
        output={"output": "a\nb", "returncode": 0, "exception_info": None},
        **runner.template_vars(),
    )
    assert observation["content"] == expected and observation["tool_call_id"] == "c1"
    assert json.loads(observation["content"]) == {"returncode": 0, "output": "a\nb"}
    items = json.loads((tmp_path / "out/output_items.json").read_text())
    assert [i["type"] for i in items] == [
        "message",
        "function_call",
        "function_call_output",
        "message",
        "function_call",
        "function_call_output",
    ]
    assert items[2]["call_id"] == "c1" and items[5]["call_id"] == "c2"
    assert json.loads((tmp_path / "out/usages.json").read_text())[0]["input_tokens"] == 10
    result = json.loads((tmp_path / "out/result.json").read_text())
    assert (
        result["exit_status"] == "Submitted" and result["n_calls"] == 2 and result["steps"] == 2 and result["finished"]
    )
    # The second request replays native items: instruction, assistant output, tool output.
    second = runner.post_responses.bodies[1]["input"]
    assert [i.get("type", i.get("role")) for i in second] == [
        "system",
        "user",
        "message",
        "function_call",
        "function_call_output",
    ]
    assert second[-1] == {"type": "function_call_output", "call_id": "c1", "output": expected}
    assert (tmp_path / "pids").read_text().count("\n") == 2


def test_format_errors_end_after_three_and_map_output_token_limit(tmp_path):
    module = load_runner()
    runner = module.Runner(config(tmp_path))
    no_call = {
        "output": [
            {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "thinking", "annotations": []}],
            }
        ],
        "usage": None,
    }
    runner.post_responses = scripted(runner, [no_call, no_call, no_call])
    info = runner.run()
    assert info["exit_status"] == "RepeatedFormatError"
    messages = json.loads((tmp_path / "out/trajectory.json").read_text())["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "user", "user", "user", "exit"]
    assert "No tool calls found" in messages[2]["content"] and messages[2]["extra"]["interrupt_type"] == "FormatError"
    # All three cut by max_output_tokens -> OutputTokenLimitExceeded (server-harness mapping).
    runner = module.Runner(config(tmp_path, output_dir=str(tmp_path / "out2")))
    cut = dict(no_call, incomplete_details={"reason": "max_output_tokens"})
    runner.post_responses = scripted(runner, [cut, cut, cut])
    assert runner.run()["exit_status"] == "OutputTokenLimitExceeded"
    messages = json.loads((tmp_path / "out2/trajectory.json").read_text())["messages"]
    assert "reached the output token limit (finish_reason=length)" in messages[2]["content"]


def test_bad_tool_call_arguments_are_format_errors_then_recover(tmp_path):
    module = load_runner()
    runner = module.Runner(config(tmp_path))
    bad = call("x", "b1")
    bad["output"][1]["arguments"] = "{not json"
    other_tool = call("x", "b2")
    other_tool["output"][1]["name"] = "python"
    runner.post_responses = scripted(
        runner, [bad, other_tool, call("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "b3")]
    )
    assert runner.run()["exit_status"] == "Submitted"
    messages = json.loads((tmp_path / "out/trajectory.json").read_text())["messages"]
    assert "Error parsing tool call arguments" in messages[2]["content"]
    assert "Unknown tool 'python'" in messages[3]["content"] and "Missing 'command'" not in messages[3]["content"]


def test_step_limit_and_budget_exits(tmp_path):
    module = load_runner()
    runner = module.Runner(config(tmp_path, step_limit=1))
    runner.post_responses = scripted(runner, [call("true", "s1")])
    assert runner.run()["exit_status"] == "LimitsExceeded"
    runner = module.Runner(config(tmp_path, output_dir=str(tmp_path / "out2"), budget_sec=0.001))
    runner.started -= 10
    runner.post_responses = scripted(runner, [])
    assert runner.run()["exit_status"] == "TimeExceeded"


def test_context_overflow_ends_the_episode_normally(tmp_path):
    module = load_runner()
    runner = module.Runner(config(tmp_path))
    runner.post_responses = scripted(runner, [module.ContextOverflow("context length exceeded (131072)")])
    info = runner.run()
    assert info["exit_status"] == "ContextWindowExceeded"
    assert json.loads((tmp_path / "out/result.json").read_text())["n_calls"] == 1


def test_command_timeout_kills_the_group_and_reports_like_the_harness(tmp_path):
    module = load_runner()
    runner = module.Runner(config(tmp_path, step_timeout_sec=1))
    runner.post_responses = scripted(
        runner,
        [call("echo start; sleep 30; echo never", "t1"), call("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "t2")],
    )
    assert runner.run()["exit_status"] == "Submitted"
    observation = json.loads(json.loads((tmp_path / "out/trajectory.json").read_text())["messages"][3]["content"])
    assert observation["returncode"] == -1 and observation["exception_info"] == "Command timed out after 1 seconds."
    assert observation["output"].startswith("start")


def test_long_output_uses_the_elided_branch_with_htmlsafe_json(tmp_path):
    module = load_runner()
    runner = module.Runner(config(tmp_path))
    runner.post_responses = scripted(
        runner,
        [call("python3 -c \"print('<x>&' * 4000)\"", "l1"), call("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "l2")],
    )
    runner.run()
    observation = json.loads((tmp_path / "out/trajectory.json").read_text())["messages"][3]["content"]
    parsed = json.loads(observation)
    assert parsed["elided_chars"] == 4000 * 4 + 1 - 10000 and parsed["warning"] == "Output too long."
    assert len(parsed["output_head"]) == 5000 and len(parsed["output_tail"]) == 5000
    assert "\\u003c" in observation and "<" not in observation  # jinja2 tojson escaping preserved


def test_uncaught_error_is_recorded_and_re_raised(tmp_path):
    module = load_runner()
    runner = module.Runner(config(tmp_path))
    runner.post_responses = scripted(runner, [module.ModelServerError("model server HTTP 500: boom")])
    with pytest.raises(module.ModelServerError):
        runner.run()
    trajectory = json.loads((tmp_path / "out/trajectory.json").read_text())
    assert trajectory["info"]["exit_status"] == "ModelServerError"
    assert json.loads((tmp_path / "out/result.json").read_text())["finished"] is True  # the exit record was saved


def test_http_client_retries_then_maps_overflow_and_hard_errors(tmp_path):
    module = load_runner()
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            calls.append((self.path, self.headers.get("x-session-id"), body["input"][0]["role"]))
            n = len(calls)
            if n == 1:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"upstream busy")
            elif n == 2:
                payload = json.dumps(call("true", "h1")).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
            elif n == 3:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": {"message": "This model\'s maximum context length is 131072 tokens"}}')
            else:
                self.send_response(422)
                self.end_headers()
                self.wfile.write(b"bad request")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        runner = module.Runner(config(tmp_path, model_url=f"http://127.0.0.1:{server.server_port}/v1", http_retries=2))
        runner.messages = [{"role": "system", "content": "s"}]
        body = {"input": [{"role": "system", "content": "s"}], "tools": []}
        first = runner.post_responses(body)
        assert first["output"][1]["call_id"] == "h1" and calls[0][0] == "/v1/responses" and calls[0][1] == "tb4-test"
        assert runner.model_call_attempts[0]["request_id"].startswith("tb4-test-0-")
        assert len(calls) == 2 and runner.http_errors[0]["status"] == 503
        with pytest.raises(module.ContextOverflow):
            runner.post_responses(body)
        with pytest.raises(module.ModelServerError, match="HTTP 422"):
            runner.post_responses(body)
    finally:
        server.shutdown()


def test_escaped_descendant_holding_output_cannot_hang_a_step(tmp_path):
    module = load_runner()
    runner = module.Runner(config(tmp_path, step_timeout_sec=20))
    # A daemonized grandchild keeps running; upstream's pipe drain would block until it exits.
    runner.post_responses = scripted(
        runner,
        [
            call("setsid sleep 30 > /dev/null 2>&1 & echo started", "d1"),
            call("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "d2"),
        ],
    )
    started = module.now()
    assert runner.run()["exit_status"] == "Submitted"
    assert module.now() - started < 10
    observation = json.loads(json.loads((tmp_path / "out/trajectory.json").read_text())["messages"][3]["content"])
    assert observation["returncode"] == 0 and observation["output"] == "started\n"
    assert not list((tmp_path / "out").glob("step-*.out"))


def test_sigterm_saves_a_time_exceeded_exit(tmp_path):
    import os
    import signal

    module = load_runner()
    runner = module.Runner(config(tmp_path))

    def slow_post(body):
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("SIGTERM must interrupt the model call")

    runner.post_responses = slow_post
    try:
        info = runner.run()
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    assert info["exit_status"] == "TimeExceeded" and info["signal"] == "SIGTERM"
    result = json.loads((tmp_path / "out/result.json").read_text())
    assert result["terminated"] is True and result["finished"] is True


def test_grace_prevents_a_model_call_that_cannot_finish(tmp_path):
    module = load_runner()
    runner = module.Runner(config(tmp_path, budget_sec=100))
    runner.max_model_latency = 200.0  # observed slow calls -> grace 300 s > remaining
    runner.post_responses = scripted(runner, [])
    assert runner.run()["exit_status"] == "TimeExceeded"


def test_vendor_zip_provides_pure_python_jinja_on_any_interpreter(tmp_path):
    archive = build_vendor_zip(tmp_path / "vendor.zip")
    names = zipfile_names(archive)
    assert any(n.startswith("jinja2/") for n in names) and any(n.startswith("markupsafe/") for n in names)
    assert not any(n.endswith(".so") or "__pycache__" in n for n in names)
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); import jinja2, markupsafe; "
        "from jinja2 import Template, StrictUndefined; "
        "print(Template('{{ x | tojson }}', undefined=StrictUndefined).render(x=\"<a>&'b\"))"
    )
    for interpreter in {sys.executable, "/usr/bin/python3"}:
        if not Path(interpreter).exists():
            continue
        out = subprocess.run(
            [interpreter, "-S", "-c", script, str(archive)], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == '"\\u003ca\\u003e\\u0026\\u0027b"'


def zipfile_names(path):
    import zipfile

    with zipfile.ZipFile(path) as archive:
        return archive.namelist()
