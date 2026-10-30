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
"""The rds tools over MCP, driven the way Claude Code drives them: seed, tools/list, tools/call, verify.

The calls run the real tools and the real scorer. Tests that fold need EternaFold and skip without it."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from resources_servers.rna_design.app import RnaDesignResourcesServer, RnaDesignResourcesServerConfig


TOKEN_HEADER = "X-NeMo-Gym-Session-Token"
RPC_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def _suite_target(name: str) -> dict:
    import yaml

    suite = (
        Path(__file__).resolve().parents[3]
        / "responses_api_agents/rna_design_agent/suites/rnainvbench_eterna100_v2.yaml"
    )
    return next(t for t in yaml.safe_load(suite.read_text())["targets"] if t["name"] == name)


TARGET = _suite_target("eterna100_v2_157")


def _have_eternafold() -> bool:
    try:
        import arnie  # noqa: F401
    except ImportError:
        return False
    return Path(os.environ.get("ARNIEFILE", "/nonexistent")).is_file()


needs_eternafold = pytest.mark.skipif(not _have_eternafold(), reason="EternaFold and arnie are not configured")


def _server(tmp_path: Path, **overrides) -> RnaDesignResourcesServer:
    values = dict(
        host="0.0.0.0", port=8080, entrypoint="", name="rna_design", expose_tools_over_mcp=True, work_dir=str(tmp_path)
    )
    values.update(overrides)
    return RnaDesignResourcesServer(
        config=RnaDesignResourcesServerConfig(**values), server_client=MagicMock(spec=ServerClient)
    )


def _client(tmp_path: Path, server: RnaDesignResourcesServer | None = None) -> TestClient:
    from nemo_gym.server_runtime.mcp_auto_exposure import maybe_auto_expose

    server = server or _server(tmp_path)
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    return TestClient(app, base_url="http://127.0.0.1:8000")


def _rpc(client: TestClient, method: str, params: dict | None = None, token: str | None = None, rid: int = 1) -> dict:
    headers = dict(RPC_HEADERS)
    if token:
        headers[TOKEN_HEADER] = token
    body = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        body["params"] = params
    return client.post("/mcp", headers=headers, json=body, follow_redirects=False).json()


SEED = {"responses_create_params": {"input": []}, "suite": "rnainvbench_eterna100_v2", "target": TARGET}


def _seed(client: TestClient) -> str:
    seeded = client.post("/seed_session", json=SEED)
    assert seeded.status_code == 200, seeded.text
    _rpc(
        client,
        "initialize",
        {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
    )
    client.post("/mcp", headers=RPC_HEADERS, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    return seeded.json()["mcp"]["headers"][TOKEN_HEADER]


def _call(client: TestClient, token: str, name: str, arguments: dict, rid: int) -> dict:
    result = _rpc(client, "tools/call", {"name": name, "arguments": arguments}, token=token, rid=rid)["result"]
    return json.loads(result["content"][0]["text"])


def _verify(client: TestClient, usage: dict | None = None, last_message: str | None = None) -> dict:
    body = {
        "responses_create_params": {"input": []},
        "response": {
            "id": "r",
            "created_at": 0,
            "model": "m",
            "object": "response",
            "output": [
                {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": last_message, "annotations": []}],
                }
            ]
            if last_message
            else [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            **({"usage": usage} if usage else {}),
        },
        "suite": "rnainvbench_eterna100_v2",
        "target": TARGET,
    }
    reply = client.post("/verify", json=body)
    assert reply.status_code == 200, reply.text
    return reply.json()


def test_every_rds_tool_is_listed_with_its_own_schema(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    with _client(tmp_path) as client:
        token = _seed(client)
        tools = {t["name"]: t for t in _rpc(client, "tools/list", {}, token=token, rid=2)["result"]["tools"]}
        _verify(client)
    assert {"generate_grnade", "score_fold_match", "score_constraints", "fold_eternafold"} <= set(tools)
    assert "sequences" in tools["score_fold_match"]["inputSchema"]["properties"]
    # operator knobs such as a write path stay off the schema, as they do for the agent's model
    assert "out_dir" not in tools["report_rules"]["inputSchema"]["properties"]
    # the suite withholds these, so the harness never sees them, as the agent's model never does
    assert not {"generate_inverse_fold", "pipeline_design_and_rank"} & set(tools)


def test_a_late_reply_from_an_abandoned_call_is_not_taken_for_the_next(tmp_path: Path) -> None:
    """A harness gave up on a slow call. Its reply arrives after the next request was sent and is skipped."""
    import asyncio

    from resources_servers.rna_design.app import ToolWorker

    script = tmp_path / "echo.py"
    script.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    r = json.loads(line)\n"
        "    if r['op'] == 'slow':\n"
        "        continue\n"
        "    if r['op'] == 'score':\n"
        "        print(json.dumps({'ok': True, 'result': 'late', 'id': r['id'] - 1}), flush=True)\n"
        "    print(json.dumps({'ok': True, 'result': r['op'], 'id': r['id']}), flush=True)\n"
    )

    async def scenario() -> dict:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        worker = ToolWorker(proc, timeout_s=0.5)
        with pytest.raises(asyncio.TimeoutError):
            await worker.ask({"op": "slow"})
        worker.timeout_s = 10
        try:
            return await worker.ask({"op": "score"})
        finally:
            await worker.close()

    assert asyncio.run(scenario())["result"] == "score"


@needs_eternafold
def test_a_tool_the_task_does_not_offer_is_refused(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    with _client(tmp_path) as client:
        token = _seed(client)
        # this suite withholds the CPU generator from every arm, so the session's MCP token refuses it
        reply = _rpc(
            client,
            "tools/call",
            {"name": "generate_inverse_fold", "arguments": {"n": 5, "target": TARGET["name"]}},
            token=token,
            rid=2,
        )
        _verify(client)
    assert reply["result"]["isError"] and "not allowed for this session" in reply["result"]["content"][0]["text"]


@needs_eternafold
def test_designs_written_by_the_harness_are_scored_like_the_agents(tmp_path: Path) -> None:
    """A harness may write its own sequences and score them. They enter the pool, and verify scores the pool."""
    pytest.importorskip("mcp")
    with _client(tmp_path) as client:
        token = _seed(client)
        state = _call(
            client,
            token,
            "score_fold_match",
            {"sequences": ["GGGGAAACCCCA", "AAAAAAAAAAAA"], "target": TARGET["name"]},
            rid=2,
        )
        _call(client, token, "score_constraints", {"sequences": "@all", "target": TARGET["name"]}, rid=3)
        result = _verify(client)
    assert state["population"]["n"] == 2
    assert result["pool_size"] == 2 and result["tool_calls"] == 2 and result["tool_failures"] == 0
    assert result["reward"] == (1.0 if result["solved"] else 0.0) and result["mask_sample"] is False


@needs_eternafold
def test_calls_that_fail_on_the_models_own_arguments_score_zero(tmp_path: Path) -> None:
    """Scoring an empty pool is the model's mistake, not the environment's, so it is measured as 0."""
    pytest.importorskip("mcp")
    with _client(tmp_path) as client:
        token = _seed(client)
        _call(client, token, "score_fold_match", {"sequences": "@unscored", "target": TARGET["name"]}, rid=2)
        result = _verify(client)
    assert result["tool_calls"] == 1 and result["tool_failures"] == 1 and result["environment_faults"] == 0
    assert result["mask_sample"] is False and result["reward"] == 0.0


def test_the_model_cannot_set_operator_fields_or_exceed_the_round_budget(tmp_path: Path) -> None:
    """Hidden from the schema is not enough: the worker refuses an operator field, and every call spends a round."""
    with _client(tmp_path) as client:
        assert client.post("/seed_session", json=SEED).status_code == 200
        refused = client.post("/report_rules", json={"rules": [], "out_dir": str(tmp_path / "escape")}).json()
        valid = {"sequences": ["G" * TARGET["length"]], "target": TARGET["name"]}
        replies = [client.post("/score_constraints", json=valid).json() for _ in range(12)]
        _verify(client)
    assert "out_dir is set by the operator" in refused["error"] and not (tmp_path / "escape").exists()
    # the refused call was round 1 of 12, so the twelfth call here is the thirteenth round
    assert "error" not in replies[10] and "rounds" in replies[11]["error"]


def test_a_reply_larger_than_64_kib_still_verifies(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.post("/seed_session", json=SEED).status_code == 200
        client.post("/report_rules", json={"rules": [], "notes": "x" * 70_000})
        result = _verify(client)
    assert result["tool_calls"] == 1


@pytest.mark.parametrize("worker_exits_first", [False, True])
def test_closing_a_worker_stops_the_processes_it_started(tmp_path: Path, worker_exits_first: bool) -> None:
    """gRNAde runs as the tool worker's child. Ending the session must end it too, even after the worker died."""
    import asyncio
    import time

    from resources_servers.rna_design.app import ToolWorker

    pid_file = tmp_path / "child.pid"
    script = tmp_path / "parent.py"
    script.write_text(
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen(['sleep', '60'])\nopen({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        + ("" if worker_exits_first else "time.sleep(60)\n")
    )

    async def scenario() -> int:
        proc = await asyncio.create_subprocess_exec(sys.executable, str(script), start_new_session=True)
        while not pid_file.exists() or not pid_file.read_text():
            await asyncio.sleep(0.05)
        if worker_exits_first:
            await proc.wait()
        await ToolWorker(proc, timeout_s=1).close()
        return int(pid_file.read_text())

    child = asyncio.run(scenario())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        # Killed but not yet reaped, as where PID 1 does not reap orphans.
        if subprocess.run(["ps", "-o", "stat=", "-p", str(child)], capture_output=True, text=True).stdout.startswith(
            "Z"
        ):
            return
        time.sleep(0.1)
    raise AssertionError(f"process {child} outlived its tool worker")


@needs_eternafold
def test_a_hidden_objective_shows_the_harness_an_alias_not_the_puzzle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataclasses

    from resources_servers.rna_design import tool_worker

    real = tool_worker.worker.find_campaign

    def hidden(suite_path, target_name):
        suite, campaign = real(suite_path, target_name)
        return suite, dataclasses.replace(campaign, objective_visibility="hidden")

    monkeypatch.setattr(tool_worker.worker, "find_campaign", hidden)
    for name in ("RDS_TARGET", "RDS_RUN_DIR"):  # Campaign sets these. Restore them for the other tests.
        monkeypatch.setenv(name, os.environ.get(name, ""))
    suite = (
        Path(__file__).resolve().parents[3]
        / "responses_api_agents/rna_design_agent/suites/rnainvbench_eterna100_v2.yaml"
    )
    campaign = tool_worker.Campaign(str(suite), "eterna100_v2_157", str(tmp_path / "run"))
    shown = json.dumps(campaign.objective)
    assert "eterna" not in shown and os.environ["RDS_TARGET"].startswith("target_")
    assert campaign.campaign.target_name == "eterna100_v2_157"  # the real campaign still judges the pool
    campaign.state.close()


def test_a_harness_that_gives_up_releases_the_rollout_by_the_seeded_id(tmp_path: Path) -> None:
    """The seed reply names the session, so a harness that abandons the rollout can close it at /close_session."""
    pytest.importorskip("mcp")
    server = _server(tmp_path)
    with _client(tmp_path, server) as client:
        sid = client.post("/seed_session", json=SEED).json()["resources_session_id"]
        worker = server.workers[sid]
        close = {"resources_session_id": sid, "episode_id": {"rollout_id": "abandoned"}}
        assert client.post("/close_session", json=close).status_code == 200
        assert sid not in server.workers and worker.proc.returncode is not None
        assert client.post("/close_session", json=close).status_code == 200  # closing twice is fine
        assert client.post("/seed_session", json=SEED).status_code == 409  # a late seed cannot reopen it


def test_reading_the_campaign_state_is_free(tmp_path: Path) -> None:
    """The agent's model sees the state every round for free. A harness reads it with campaign_state."""
    pytest.importorskip("mcp")
    with _client(tmp_path) as client:
        token = _seed(client)
        names = {t["name"] for t in _rpc(client, "tools/list", {}, token=token, rid=2)["result"]["tools"]}
        state = _call(client, token, "campaign_state", {}, rid=3)
        again = _call(client, token, "campaign_state", {}, rid=4)
        result = _verify(client)
    assert "campaign_state" in names
    assert state["objective"]["target"] == TARGET["name"]
    # wall_seconds is a clock reading, not a charge: it advances between any two snapshots. Every axis
    # a call could spend must be unchanged.
    free = lambda b: {k: v for k, v in b["spend"].items() if k != "wall_seconds"}  # noqa: E731
    assert free(state["budget"]) == free(again["budget"])
    assert state["budget"]["limits"] == again["budget"]["limits"]
    assert result["tool_calls"] == 0 and result["rounds"] == 0


def test_a_rollout_that_never_reached_the_model_is_masked_not_scored(tmp_path: Path) -> None:
    """A dead model endpoint fails the harness before any tool call. That is no measurement of the model, so it is
    masked. A model that did answer and then chose to stop is a real result and scores 0."""
    pytest.importorskip("mcp")
    answered = {
        "input_tokens": 1200,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 40,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 1240,
    }
    with _client(tmp_path) as client:
        _seed(client)
        unreached = _verify(client)
    with _client(tmp_path) as client:
        _seed(client)
        stopped = _verify(client, usage=answered)
    with _client(tmp_path) as client:
        _seed(client)
        # Claude Code's last message after the endpoint failed mid-rollout and its own retries ran out.
        cut = _verify(
            client, usage=answered, last_message="<think>\nStart.\n</think>\n\nAPI Error: 503 no live replica"
        )
    assert unreached["tool_calls"] == 0 and unreached["mask_sample"] is True
    assert unreached["failure_reason"] == "the harness made no model call and no tool call"
    assert stopped["tool_calls"] == 0 and stopped["mask_sample"] is False and stopped["reward"] == 0.0
    assert cut["mask_sample"] is True and cut["failure_reason"] == "the model endpoint failed during the rollout"


def test_a_session_that_never_reached_verify_is_closed_by_the_next(tmp_path: Path) -> None:
    """A rollout that dies before /verify never closes its worker, so the next seed closes stale ones."""
    server = _server(tmp_path, session_ttl_s=0.0)
    with _client(tmp_path, server) as client:
        assert client.post("/seed_session", json=SEED).status_code == 200
        (abandoned,) = server.workers.values()
        client.cookies.clear()  # the next rollout, in a session of its own
        assert client.post("/seed_session", json=SEED).status_code == 200
        assert abandoned.proc.returncode is not None
        assert len(server.workers) == 1 and abandoned not in server.workers.values()
        _verify(client)


def test_a_worker_that_fails_to_open_leaves_nothing_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tempfile

    import resources_servers.rna_design.app as app_module

    async def stuck(self, request):
        raise TimeoutError

    monkeypatch.setattr(app_module.ToolWorker, "ask", stuck)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    server = _server(tmp_path, work_dir="")
    with _client(tmp_path, server) as client, pytest.raises(TimeoutError):
        client.post("/seed_session", json=SEED)
    assert server.workers == {} and list((tmp_path / "tmp" / "rna_design").iterdir()) == []


def test_a_scoring_call_that_times_out_is_masked_not_a_server_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import resources_servers.rna_design.app as app_module

    real_ask = app_module.ToolWorker.ask

    async def ask(self, request):
        if request["op"] == "score":
            raise TimeoutError
        return await real_ask(self, request)

    monkeypatch.setattr(app_module.ToolWorker, "ask", ask)
    server = _server(tmp_path)
    with _client(tmp_path, server) as client:
        _seed(client)
        (worker,) = server.workers.values()
        verified = _verify(client)
    assert verified["reward"] == 0.0 and verified["mask_sample"] is True
    assert "scoring failed" in verified["failure_reason"] and worker.proc.returncode is not None


def test_a_tool_that_prints_to_stdout_does_not_break_the_replies(tmp_path: Path) -> None:
    import subprocess
    import sys

    worker_path = Path(__file__).resolve().parents[1] / "tool_worker.py"
    script = f"""
import os, runpy, sys
sys.argv = [{str(worker_path)!r}]
module = runpy.run_path({str(worker_path)!r}, run_name="tool_worker")
class Noisy:
    def __init__(self, *args):
        self.objective, self.allowed = "o", set()
    def score(self):
        print("a stray print from a tool")
        os.write(1, b"native output without a newline")
        return {{"solved": False}}
module["Campaign"] = Noisy
module["main"].__globals__["Campaign"] = Noisy
module["main"]()
"""
    requests = "\n".join(
        [
            '{{"op": "open", "suite_path": "s", "target_name": "t", "run_dir": "r", "id": 1}}'.format(),
            '{{"op": "score", "id": 2}}'.format(),
        ]
    )
    done = subprocess.run(
        [sys.executable, "-c", script], input=requests + "\n", capture_output=True, text=True, timeout=120
    )
    replies = [json.loads(line) for line in done.stdout.splitlines()]
    assert [r["id"] for r in replies] == [1, 2] and replies[1] == {"ok": True, "result": {"solved": False}, "id": 2}


def test_stopping_the_server_closes_its_workers(tmp_path: Path) -> None:
    server = _server(tmp_path)
    with _client(tmp_path, server) as client:
        assert client.post("/seed_session", json=SEED).status_code == 200
        (worker,) = server.workers.values()
    assert worker.proc.returncode is not None and server.workers == {}


def test_an_unknown_suite_starts_no_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import resources_servers.rna_design.app as app_module

    spawned = []
    monkeypatch.setattr(app_module.asyncio, "create_subprocess_exec", lambda *args, **kwargs: spawned.append(args))
    with _client(tmp_path) as client, pytest.raises(ValueError, match="unknown suite"):
        client.post("/seed_session", json={**SEED, "suite": "no_such_suite"})
    assert spawned == []


def test_a_temp_run_dir_is_removed_after_scoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no work_dir the run directory is scratch. A configured work_dir keeps it as the record."""
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    with _client(tmp_path, _server(tmp_path, work_dir="")) as client:
        assert client.post("/seed_session", json=SEED).status_code == 200
        assert len(list((tmp_path / "tmp" / "rna_design").iterdir())) == 1
        _verify(client)
    assert list((tmp_path / "tmp" / "rna_design").iterdir()) == []
    with _client(tmp_path) as client:
        assert client.post("/seed_session", json=SEED).status_code == 200
        _verify(client)
    assert len(list(tmp_path.glob("run_*"))) == 1


@needs_eternafold
def test_a_tool_called_without_target_uses_the_campaign(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    with _client(tmp_path) as client:
        token = _seed(client)
        _call(client, token, "score_constraints", {"sequences": ["G" * TARGET["length"]]}, rid=2)
        result = _verify(client)
    assert result["tool_calls"] == 1 and result["tool_failures"] == 0 and result["pool_size"] == 1
