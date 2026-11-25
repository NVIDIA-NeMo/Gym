# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple Agent's Python tools: a real worker process, run the way the sandboxed loop runs it."""

import json
import os
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from responses_api_agents.simple_agent import python_tool_client
from responses_api_agents.simple_agent.python_tool import PythonToolConfig, PythonToolSpec
from responses_api_agents.simple_agent.python_tool_client import PythonTools


WORKER = str(Path(python_tool_client.__file__).with_name("python_tool_worker.py"))
# The math_with_code tool's builtins.
MATH_BUILTINS = [
    "print",
    "len",
    "str",
    "int",
    "float",
    "list",
    "dict",
    "tuple",
    "set",
    "min",
    "max",
    "sum",
    "abs",
    "range",
    "enumerate",
    "zip",
    "__import__",
]


@asynccontextmanager
async def _tools(tmp_path: Path, **config: Any) -> AsyncIterator[PythonTools]:
    spec = PythonToolSpec(
        config=PythonToolConfig(**config), python=sys.executable, worker=WORKER, workdir=str(tmp_path)
    )
    tools = PythonTools({"execute_python": spec})
    try:
        yield tools
    finally:
        await tools.close()


async def _run(tools: PythonTools, code: str) -> dict[str, Any]:
    output, error_type = await tools.call("execute_python", {"code": code})
    assert error_type is None
    return json.loads(output)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def test_names_a_call_defines_stay_defined_for_the_next_call(tmp_path: Path) -> None:
    async with _tools(tmp_path) as tools:
        assert await _run(tools, "x = 6 * 7") == {
            "success": True,
            "stdout": "",
            "stderr": "",
            "error_message": None,
            "result": None,
        }
        # The output is the compact JSON the Resources Server's tool returned, field for field.
        output, _ = await tools.call(
            "execute_python", {"code": "import sys\nprint(x)\nprint('warn', file=sys.stderr)\nx"}
        )

    assert output == '{"success":true,"stdout":"42\\n","stderr":"warn\\n","error_message":null,"result":"42"}'


async def test_top_level_names_are_call_locals_as_in_the_resources_worker(tmp_path: Path) -> None:
    """Top-level definitions land in a locals mapping, so a function cannot see a sibling one, as before."""
    async with _tools(tmp_path) as tools:
        defined = await _run(tools, "def f(n):\n    return g(n)\ndef g(n):\n    return n + 1\nf(1)")
        # The last line is evaluated again for its value; that fails the same way and gives no result.
        assert defined["success"] is False
        assert defined["error_message"] == "name 'g' is not defined"
        assert (await _run(tools, "g(1)"))["result"] == "2"


async def test_the_tool_has_only_the_allowed_builtins_and_its_preloaded_modules(tmp_path: Path) -> None:
    async with _tools(tmp_path, builtins=MATH_BUILTINS, preload={"m": "math"}) as tools:
        assert (await _run(tools, "print(m.sqrt(16))"))["stdout"] == "4.0\n"
        refused = await _run(tools, "sorted([2, 1])")
        assert refused["success"] is False and refused["error_message"] == "name 'sorted' is not defined"
        # __import__ is allowed, so import statements work.
        assert (await _run(tools, "import fractions\nstr(fractions.Fraction(1, 3))"))["result"] == "1/3"


async def test_an_error_returns_its_message_and_drops_what_the_code_printed(tmp_path: Path) -> None:
    async with _tools(tmp_path) as tools:
        failed = await _run(tools, "print('partial')\n1 / 0")
        assert failed == {
            "success": False,
            "stdout": "",
            "stderr": "",
            "error_message": "division by zero",
            "result": None,
        }
        assert (await _run(tools, "1 +"))["error_message"].startswith("invalid syntax")


async def test_code_past_its_timeout_fails_and_the_worker_keeps_its_names(tmp_path: Path) -> None:
    async with _tools(tmp_path, timeout_seconds=1) as tools:
        await _run(tools, "kept = 1")
        assert (await _run(tools, "while True:\n    pass"))["error_message"] == "code timed-out"
        assert (await _run(tools, "kept"))["result"] == "1"


async def test_a_long_call_into_c_times_out_late_and_keeps_the_names(tmp_path: Path) -> None:
    """SIGALRM fires between bytecodes, so a long C call answers once it returns, and the worker lives on."""
    async with _tools(tmp_path, timeout_seconds=1) as tools:
        await _run(tools, "kept = 1")
        started = time.monotonic()
        assert (await _run(tools, "sum(range(4 * 10**8))"))["error_message"] == "code timed-out"
        assert time.monotonic() - started < 1 + python_tool_client.KILL_GRACE_SECONDS
        assert (await _run(tools, "kept"))["result"] == "1"


async def test_code_that_blocks_the_alarm_is_killed_and_the_next_call_starts_afresh(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(python_tool_client, "KILL_GRACE_SECONDS", 0.5)
    async with _tools(tmp_path, timeout_seconds=1) as tools:
        await _run(tools, "kept = 1")
        worker = tools._workers["execute_python"]
        blocked = "import signal\nsignal.signal(signal.SIGALRM, signal.SIG_IGN)\nwhile True:\n    pass"

        timed_out = await _run(tools, blocked)

        assert timed_out["error_message"] == f"code timed-out; {python_tool_client.RESTARTS}"
        assert worker.returncode is not None
        # The model was told: a new worker answers, without the old names.
        assert (await _run(tools, "kept"))["error_message"] == "name 'kept' is not defined"
        assert (await _run(tools, "1 + 1"))["result"] == "2"


async def test_writes_to_the_real_descriptors_cannot_corrupt_a_reply(tmp_path: Path) -> None:
    async with _tools(tmp_path) as tools:
        noisy = "import os\nos.write(1, b'\\x00\\x00\\x00\\x09garbage')\nos.write(2, b'noise')\nprint('ok')"
        assert await _run(tools, noisy) == {
            "success": True,
            "stdout": "ok\n",
            "stderr": "",
            "error_message": None,
            "result": None,
        }
        assert (await _run(tools, "1 + 1"))["result"] == "2"


async def test_a_reply_the_code_forges_stops_the_worker_instead_of_answering(tmp_path: Path) -> None:
    """Code that writes a frame into the reply channel (fd 4 in the worker) cannot answer a call or break the loop."""
    async with _tools(tmp_path) as tools:
        await _run(tools, "kept = 1")
        forged = await _run(tools, "import os\nos.write(4, b'\\x00\\x00\\x00\\x02{}')")
        assert forged["error_message"] == f"Python worker stopped; {python_tool_client.RESTARTS}"
        assert (await _run(tools, "kept"))["error_message"] == "name 'kept' is not defined"


async def test_code_that_ends_the_worker_reports_it_and_the_next_call_starts_afresh(tmp_path: Path) -> None:
    async with _tools(tmp_path) as tools:
        ended = await _run(tools, "import sys\nsys.exit(3)")
        assert ended["success"] is False
        assert ended["error_message"] == f"Python worker stopped; {python_tool_client.RESTARTS}"
        assert (await _run(tools, "1"))["result"] == "1"


async def test_code_larger_than_a_pipe_buffer_runs(tmp_path: Path) -> None:
    async with _tools(tmp_path) as tools:
        code = f"text = '{'a' * 300_000}'\nlen(text)"
        assert (await _run(tools, code))["result"] == "300000"
        assert (await _run(tools, "print('b' * 1_000_000)"))["stdout"] == "b" * 1_000_000 + "\n"


async def test_close_stops_what_the_code_forked(tmp_path: Path) -> None:
    async with _tools(tmp_path) as tools:
        forked = await _run(
            tools, "import os, time\npid = os.fork()\nif pid == 0:\n    time.sleep(120)\n    os._exit(0)\nprint(pid)"
        )
        child = int(forked["stdout"])
        assert _alive(child)
        started = time.monotonic()
        await tools.close()
        assert time.monotonic() - started < python_tool_client.REAP_TIMEOUT_SECONDS + 1
        for _ in range(50):
            if not _alive(child):
                break
            time.sleep(0.1)
        assert not _alive(child)


async def test_the_code_runs_in_the_task_working_directory(tmp_path: Path) -> None:
    async with _tools(tmp_path) as tools:
        await _run(tools, "open('note.txt', 'w').write('hi')")
    assert (tmp_path / "note.txt").read_text() == "hi"


@pytest.mark.parametrize("arguments", [{"cod": "1"}, {"code": 1}, ["1"]])
async def test_arguments_without_code_are_an_error_the_model_sees(tmp_path: Path, arguments: Any) -> None:
    async with _tools(tmp_path) as tools:
        output, error_type = await tools.call("execute_python", arguments)
        assert json.loads(output)["success"] is False and "code" in json.loads(output)["error_message"]
        # Recorded as a failed call, as the Resources Server's 422 was.
        assert error_type == "invalid_arguments"
        assert "execute_python" not in tools._workers


async def test_a_worker_whose_preload_fails_fails_the_call_and_starts_again_on_the_next(tmp_path: Path) -> None:
    async with _tools(tmp_path, preload={"x": "no_such_module_for_this_test"}) as tools:
        first = await _run(tools, "1")
        assert first["success"] is False
        assert first["error_message"].startswith("Python worker failed to start: ModuleNotFoundError")
        # Not stopped for good: the next call tries to start the worker again.
        assert (await _run(tools, "1"))["error_message"] == first["error_message"]


async def test_close_stops_the_workers(tmp_path: Path) -> None:
    async with _tools(tmp_path) as tools:
        await _run(tools, "1")
        worker = tools._workers["execute_python"]
        await tools.close()
        assert worker.returncode is not None
        assert tools._workers == {}


async def test_a_lone_surrogate_the_code_prints_reaches_the_model_as_utf8(tmp_path: Path) -> None:
    async with _tools(tmp_path) as tools:
        output, _ = await tools.call("execute_python", {"code": "print('\\ud800')"})
    output.encode("utf-8")
    assert json.loads(output)["stdout"] == "?\n"


async def test_answer_runs_its_own_tools_and_answers_unknown_names_only_without_a_grant(tmp_path: Path) -> None:
    async with _tools(tmp_path) as tools:
        output, error_type = await tools.answer("execute_python", {"code": "1 + 1"}, granted=False)
        assert json.loads(output)["result"] == "2" and error_type is None
        # Without a grant every tool of the task runs here: another name gets a Resources Server's 404 reply.
        assert await tools.answer("execute_code", {"code": "1"}, granted=False) == (
            '{"detail":"Not Found"}',
            "unknown_tool",
        )
        # With a grant the name may be the granted server's tool.
        assert await tools.answer("get_weather", {"city": "x"}, granted=True) is None


@pytest.mark.parametrize(
    "config",
    [
        {"preload": {"np": "numpy; import os"}},
        {"preload": {"1np": "numpy"}},
        {"builtins": ["print", "no_such"]},
        {"requirements": ["numpy"]},
        {"requirements": ["numpy>=2"]},
    ],
)
def test_a_config_that_cannot_build_the_namespace_is_refused(config: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        PythonToolConfig(**config)
