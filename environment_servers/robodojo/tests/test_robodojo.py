# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for RoboDojo environment server, episode runner, pool, and OpenCode relay."""

import asyncio
import hashlib
import json
import os
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from benchmarks.robodojo.prepare import prepare
from environment_servers.robodojo.app import (
    RoboDojoEnvironmentConfig,
    RoboDojoEnvironmentServer,
    RoboDojoTaskInput,
)
from environment_servers.robodojo.cases import bind_case
from environment_servers.robodojo.episode import (
    CLEANUP_AUDIT,
    HORIZON_AUDIT,
    SUPERVISOR_PATH,
    Episode,
    EpisodeError,
    HorizonScores,
    read_budget_exit,
)
from environment_servers.robodojo.native_pool import NativePool
from environment_servers.robodojo.opencode_relay import (
    expand_responses_sse_lines,
    mount_opencode_relay,
    normalize_opencode_responses_body,
)
from nemo_gym.client import ServerClient
from nemo_gym.core import (
    AGENT_LIMIT,
    AGENT_TIMEOUT,
    ENVIRONMENT_SERVER_FAILED,
    HARNESS_TERMINATION_UNCONFIRMED,
    BaseEpisodeRequest,
    rollout_context,
)
from nemo_gym.core.failure_kinds import ENVIRONMENT_PROTOCOL_VIOLATION
from nemo_gym.server_runtime import CleanupContext
from nemo_gym.trajectory import grading_result_binding


CASE = "general_pickup__standard__g1__l0"


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _episode_request(case_id: str, task_index: int = 0) -> BaseEpisodeRequest[RoboDojoTaskInput]:
    return BaseEpisodeRequest[RoboDojoTaskInput].model_validate(
        {
            "episode_id": {"rollout_id": f"robodojo-{task_index}", "attempt": 0},
            "task": {
                "task_id": {"taskset": "robodojo", "task_id": f"case-{task_index}"},
                "task_input": {"case_id": case_id, "responses_create_params": {"input": []}},
            },
        }
    )


@pytest.fixture
def runtime(tmp_path: Path):
    asset = tmp_path / "layout.json"
    asset.write_text('{"objects":[]}')
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    manifest = tmp_path / "panel.json"
    write_json(
        manifest,
        {
            "native_source_files": [],
            "cases": [
                {
                    "case_id": CASE,
                    "runtime_task": "general_pickup",
                    "layout_id": 0,
                    "eval_seed": 1,
                    "native_step_limit": 200,
                    "layout": {"path": asset.name, "sha256": digest},
                }
            ],
        },
    )
    template = tmp_path / "bridge.json"
    write_json(
        template,
        {
            "integration": "robodojo",
            "simulator_backend": "isaaclab3",
            "robodojo": str(tmp_path),
            "state_access": "none",
            "sdk": {"execution": "ttc", "gates": False},
            "model": {},
        },
    )
    return template, manifest, asset, digest


def test_prepare_and_bind_the_same_frozen_layout(runtime, tmp_path: Path) -> None:
    template, manifest, asset, digest = runtime
    path = prepare(str(manifest), tmp_path / "prepared.jsonl")
    row = json.loads(path.read_text())
    assert row["case_id"] == CASE
    assert row["responses_create_params"] == {"input": []}
    with pytest.raises(ValueError, match="absent from the frozen panel"):
        prepare(str(manifest), path, case_id="missing")
    assert json.loads(path.read_text()) == row
    kwargs = dict(
        template_path=template,
        panel_path=manifest,
        validity_path=None,
        case_id=row["case_id"],
        horizon="native_x2",
        model_url="http://localhost:1234/v1",
        model_name="model",
    )
    config, case = bind_case(**kwargs)
    assert case.layout.sha256 == digest
    assert (config["eval_seed"], config["layout_id"], config["step_limit"]) == (1, 0, 400)
    assert config["model"] == {"id": "model", "base_url": "http://localhost:1234/v1"}
    asset.write_text('{"objects":["changed"]}')
    with pytest.raises(ValueError, match="RoboDojo input changed"):
        bind_case(**kwargs)


def test_pool_runs_two_cases_concurrently_and_exports_both(runtime, tmp_path: Path, monkeypatch) -> None:
    template, manifest, _, _ = runtime
    second = "general_pickup__standard__g1__l1"
    panel = json.loads(manifest.read_text())
    panel["panel_id"], panel["panel_sha256"] = "test", "a" * 64
    panel["cases"].append({**panel["cases"][0], "case_id": second, "layout_id": 1})
    write_json(manifest, panel)
    validity = tmp_path / "validity.json"
    write_json(
        validity,
        {
            "panel_id": "test",
            "panel_sha256": "a" * 64,
            "validity_sha256": "b" * 64,
            "assignment": [{"slot": case, "case_id": case} for case in (CASE, second)],
        },
    )
    selection = f"{CASE},{second}"
    monkeypatch.setenv("ROBODOJO_POOL_CASES", selection)
    inputs = prepare(str(manifest), tmp_path / "prepared.jsonl")
    assert [json.loads(line)["case_id"] for line in inputs.read_text().splitlines()] == [CASE, second]
    write_json(template, {**json.loads(template.read_text()), "simulator_backend": "isaacsim"})
    launcher = tmp_path / "pool.py"
    launcher.write_text(
        """
import json, socket, sys, time
from pathlib import Path
root = Path(sys.argv[1])
cases = json.loads((root / 'cases.json').read_text())
sockets = []
for index, bound in enumerate(cases.values()):
    listener = socket.socket(socket.AF_UNIX)
    bound['native_pool_socket'] = str(root / f'{index}.sock')
    bound['native_pool_evidence'] = str(root / f'case-{index}')
    listener.bind(bound['native_pool_socket'])
    listener.listen()
    sockets.append(listener)
(root / 'seats.json').write_text(json.dumps(cases))
time.sleep(60)
"""
    )
    config = RoboDojoEnvironmentConfig(
        name="robodojo_environment",
        host="127.0.0.1",
        port=8012,
        entrypoint="app.py",
        agent_server={"type": "responses_api_agents", "name": "robodojo"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        model_name="test",
        bridge_config=template,
        panel_manifest=manifest,
        case_validity=validity,
        bridge_python=Path(sys.executable),
        simulation_python=Path(sys.executable),
        output_root=tmp_path / "outputs",
        evidence_export_root=tmp_path / "evidence",
        pool_cases=selection,
        pool_launcher=launcher,
        pool_startup_seconds=5,
    )
    client = ServerClient(
        head_server_config={"host": "127.0.0.1", "port": 8000},
        global_config_dict=OmegaConf.create(
            {"policy_model": {"responses_api_models": {"vllm_model": {"host": "127.0.0.1", "port": 8011}}}}
        ),
    )
    server = RoboDojoEnvironmentServer(config=config, server_client=client)
    entered: set[int] = set()

    async def run():
        ready = asyncio.Event()

        async def episode(self, bound):
            layout = bound["layout_id"]
            entered.add(layout)
            if entered == {0, 1}:
                ready.set()
            await asyncio.wait_for(ready.wait(), 3)
            pool = Path(bound["native_pool_evidence"])
            pool.mkdir(exist_ok=True)
            write_json(pool / "reset.json", {"layout_id": layout})
            (pool / "sensors.mp4").write_bytes(bytes([layout]))
            (self.directory / "native").mkdir()
            (self.directory / "native" / "evaluation").symlink_to(pool, target_is_directory=True)
            return "scored", HorizonScores(
                within_native=bool(layout),
                within_budget=True,
                score_within_native=layout,
                score_within_budget=1,
                task_step_limit=200,
                control_budget=400,
            )

        monkeypatch.setattr(Episode, "run", episode)
        app = server.setup_webserver()
        async with app.router.lifespan_context(app):
            results = await asyncio.gather(
                *(
                    server.run(
                        req := _episode_request(case, idx),
                        CleanupContext(episode_id=req.episode_id, cleanup_timeout_seconds=10.0),
                    )
                    for idx, case in enumerate((CASE, second))
                )
            )
            # Re-acquiring a finished case's slot must succeed on retry / repeat.
            retry_req = _episode_request(CASE, 2)
            retry_result = await server.run(
                retry_req,
                CleanupContext(episode_id=retry_req.episode_id, cleanup_timeout_seconds=10.0),
            )
            assert retry_result.result["native_score"] == 0
            process = server._pool.process
            temporary = Path(server._pool.directory.name)
        assert process.returncode is not None
        assert not temporary.exists()
        return results

    with rollout_context("robodojo-pool-test"):
        results = asyncio.run(run())
    assert [r.result["native_score"] for r in results] == [0, 1]
    assert [Path(r.result["video_path"]).read_bytes() for r in results] == [b"\x00", b"\x01"]


@pytest.mark.parametrize("backend", ["isaaclab3", "isaacsim"])
@pytest.mark.parametrize("fault", ["success", "decision_budget", "deadline", "unscored", "unclean", "wrong_backend"])
def test_run_uses_native_horizon_and_masks_missing_measurement(
    runtime, tmp_path: Path, monkeypatch, backend: str, fault: str
) -> None:
    template, manifest, _, digest = runtime
    write_json(template, {**json.loads(template.read_text()), "simulator_backend": backend})
    native_metadata = {
        "isaaclab3": {"simulator": "Isaac Lab 3.0 / Newton", "physics_backend": "isaaclab3_newton"},
        "isaacsim": {"simulator": "RoboDojo"},
    }
    other_backend = "isaacsim" if backend == "isaaclab3" else "isaaclab3"
    commands = []

    async def engine(self, *args, timeout, log=None):
        commands.append(args)
        if SUPERVISOR_PATH in args:
            return b"/runtime/supervise.py\n"
        if "run" in args:
            config = json.loads(Path(args[args.index("--config") + 1]).read_text())
            assert config["task"] == "general_pickup"
            assert config["step_limit"] == 400
            assert config["simulator_backend"] == backend
            assert config["model"]["base_url"].startswith("http://127.0.0.1:8011/")
            assert config["model"]["base_url"].endswith("/v1")
            native = Path(args[args.index("--output") + 1])
            write_json(native / "result.json", {"sdk_summary": {"closed": fault != "unclean", "faults": []}})
            write_json(native / "outcome.json", {"schema": "ember_bridge.outcome.v1", "readback_verified": True})
            write_json(native / "actor" / "ttc-runtime.json", {"state": "closed"})
            recorded_backend = other_backend if fault == "wrong_backend" else backend
            write_json(native / "evaluation" / "reset.json", {"metadata": native_metadata[recorded_backend]})
            (native / "evaluation" / "sensors.mp4").touch()
            return b""
        if CLEANUP_AUDIT in args or "validate" in args:
            return b""
        if "result" in args:
            return json.dumps(
                {
                    "schema": "ember_bridge.evaluation_result.v1",
                    "counted": fault in ("decision_budget", "deadline", "success", "wrong_backend"),
                    "class": (
                        "scored"
                        if fault in ("success", "wrong_backend")
                        else f"policy_failure.{fault}"
                        if fault in ("decision_budget", "deadline")
                        else "infrastructure.worker"
                    ),
                }
            ).encode()
        if HORIZON_AUDIT in args:
            if fault == "success":
                assert args[-2:] == ("scored", "native_x2")
                return b'{"within_native":true,"within_budget":true,"score_within_native":1,"score_within_budget":1,"task_step_limit":200,"control_budget":400}'
            assert args[-2:] == (f"policy_failure.{fault}", "native_x2")
            return b'{"within_native":false,"within_budget":false,"score_within_native":0.3,"score_within_budget":0.4,"task_step_limit":200,"control_budget":400}'
        raise AssertionError(args)

    monkeypatch.setattr(Episode, "execute", engine)
    client = ServerClient(
        head_server_config={"host": "127.0.0.1", "port": 8000},
        global_config_dict=OmegaConf.create(
            {"policy_model": {"responses_api_models": {"vllm_model": {"host": "127.0.0.1", "port": 8011}}}}
        ),
    )
    config = RoboDojoEnvironmentConfig(
        name="robodojo_environment",
        host="127.0.0.1",
        port=8012,
        entrypoint="app.py",
        agent_server={"type": "responses_api_agents", "name": "robodojo"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        model_name="test",
        bridge_config=template,
        panel_manifest=manifest,
        bridge_python=Path(sys.executable),
        simulation_python=Path(sys.executable),
        output_root=tmp_path / "outputs",
    )
    server = RoboDojoEnvironmentServer(config=config, server_client=client)
    req = _episode_request(CASE)
    cleanup = CleanupContext(episode_id=req.episode_id, cleanup_timeout_seconds=10.0)
    with rollout_context("robodojo-test"):
        episode_response = asyncio.run(server.run(req, cleanup))
    result = episode_response.result
    assert result["case_id"] == result["assigned_case_id"] == CASE
    assert result["layout_sha256"] == digest
    assert result["simulator"] == ("isaacsim_physx" if backend == "isaacsim" else "isaaclab3_newton")
    assert result["reward"] == float(fault == "success")
    if fault in ("unscored", "unclean", "wrong_backend"):
        assert result["mask_sample"] is True
        assert result["native_score"] is None
        assert result.get("ng_verifier_invocations", []) == []
        assert (
            result["failure_kind"]
            == {
                "unscored": ENVIRONMENT_SERVER_FAILED,
                "unclean": HARNESS_TERMINATION_UNCONFIRMED,
                "wrong_backend": ENVIRONMENT_PROTOCOL_VIOLATION,
            }[fault]
        )
        if fault == "wrong_backend":
            assert all(HORIZON_AUDIT not in command for command in commands)
        return
    assert result["mask_sample"] is False
    assert (
        result["failure_kind"]
        == {
            "success": None,
            "decision_budget": AGENT_LIMIT,
            "deadline": AGENT_TIMEOUT,
        }[fault]
    )
    assert result["native_score"] == (1 if fault == "success" else 0.3)
    assert result["extended_score"] == (1 if fault == "success" else 0.4)
    assert Path(result["video_path"]).is_file()
    assert (Path(result["evidence_dir"]) / "horizon.json").is_file()
    (invocation,) = result["ng_verifier_invocations"]
    assert invocation["grading_record"]["verdict"] == "scored"
    assert invocation["grading_record"]["score"] == result["reward"]
    assert invocation["grading_record"]["result_binding_sha256"] == grading_result_binding(result)


def test_timeout_terminates_an_owned_subprocess(tmp_path: Path) -> None:
    episode = Episode(tmp_path, Path(sys.executable), Path(sys.executable), None, 5, 1)
    with pytest.raises(TimeoutError):
        asyncio.run(episode.execute(sys.executable, "-c", "import time; time.sleep(10)", timeout=0.1))
    with pytest.raises(EpisodeError, match="exited with status 9"):
        asyncio.run(episode.execute(sys.executable, "-c", "raise SystemExit(9)", timeout=5, log=tmp_path / "log"))
    assert asyncio.run(episode.execute(sys.executable, "-c", "print('closed')", timeout=5)) == b"closed\n"


def test_bind_case_routes_shared_planner_socket(runtime, tmp_path: Path, monkeypatch) -> None:
    template, manifest, _, _ = runtime
    table = tmp_path / "planner.json"
    sock = "/tmp/rd-planner-test.sock"
    write_json(table, {CASE: sock})
    monkeypatch.setenv("ROBODOJO_SHARED_PLANNER_TABLE", str(table))
    config, _ = bind_case(
        template_path=template,
        panel_path=manifest,
        validity_path=None,
        case_id=CASE,
        horizon="native_x2",
        model_url="http://localhost:1234/v1",
        model_name="model",
    )
    assert config["shared_planner_socket"] == sock

    write_json(table, {CASE: "relative.sock"})
    with pytest.raises(ValueError, match="valid shared planner socket"):
        bind_case(
            template_path=template,
            panel_path=manifest,
            validity_path=None,
            case_id=CASE,
            horizon="native_x2",
            model_url="http://localhost:1234/v1",
            model_name="model",
        )


def test_read_budget_exit_accepts_verified_decision_budget_and_exports_archive(tmp_path: Path, monkeypatch) -> None:
    evidence = tmp_path / "native"
    error = "Actor exhausted its decision budget after 120 calls"
    write_json(
        evidence / "outcome.json",
        {
            "schema": "ember_bridge.outcome.v1",
            "class": "policy_failure.decision_budget",
            "unknown_outcome": False,
            "policy_faults": [],
            "actor": {"unknown_outcome": False, "finished": False, "error": error},
            "errors": {
                "launcher": f"Harness failure: {error}; see log",
                "interrupted": None,
                "archive": None,
                "verification": "Native result is incomplete; no grade was produced",
            },
            "call_cap": {"cap": 120, "reached": True, "calls": 120, "charged_calls": 120},
        },
    )
    outcome = read_budget_exit(evidence, 120)
    assert outcome["class"] == "policy_failure.decision_budget"

    with pytest.raises(EpisodeError, match="verified decision-budget evidence"):
        read_budget_exit(evidence, 60)

    write_json(evidence / "sdk-close.json", {"closed": True, "faults": [], "open_scripts": []})
    write_json(evidence / "actor" / "ttc-runtime.json", {"state": "closed"})
    write_json(evidence / "evaluation" / "reset.json", {"metadata": {"simulator": "RoboDojo"}})
    write_json(evidence / "evaluation" / "summary.json", {"episode_id": "ep_000"})
    write_json(evidence / "evaluation" / "ep_000" / "action_0000.json", {"step": 0})
    write_json(evidence / "grading.json", {"score": 0.25})
    write_json(evidence / "evaluation" / "success_end.json", {"success": False})

    episode = Episode(tmp_path, Path(sys.executable), Path(sys.executable), None, 5, 1)

    async def fake_execute(self, *args, timeout, log=None):
        if SUPERVISOR_PATH in args:
            return b"/tmp/supervise.py\n"
        if "run" in args:
            raise EpisodeError("bridge exited with status 1")
        if CLEANUP_AUDIT in args or "validate" in args:
            return b""
        if "result" in args:
            return json.dumps(
                {
                    "schema": "ember_bridge.evaluation_result.v1",
                    "counted": True,
                    "class": "policy_failure.decision_budget",
                }
            ).encode()
        if HORIZON_AUDIT in args:
            return b'{"within_native":false,"within_budget":false,"score_within_native":0.25,"score_within_budget":0.25,"task_step_limit":200,"control_budget":400}'
        raise AssertionError(args)

    monkeypatch.setattr(Episode, "execute", fake_execute)
    classification, scores = asyncio.run(
        episode.run(
            {
                "simulator_backend": "isaacsim",
                "horizon_profile": "native_x2",
                "sdk": {"call_cap": 120},
            }
        )
    )
    assert classification == "policy_failure.decision_budget"
    assert scores.score_within_native == 0.25

    exported = episode.export_evidence(tmp_path / "export")
    assert (exported / "native/sdk-close.json").is_file()
    assert (exported / "native/grading.json").is_file()
    assert not (exported / "native/evaluation/ep_000/action_0000.json").exists()
    archive = exported / "native/evaluation/actions.tar.gz"
    assert archive.is_file()
    with tarfile.open(archive, "r:gz") as bundle:
        assert bundle.getnames() == ["ep_000/action_0000.json"]


def test_native_pool_shared_startup_gate_verifies_head_rgb(tmp_path: Path, monkeypatch) -> None:
    gate = tmp_path / "gate"
    gate.mkdir()
    launcher = tmp_path / "fake_pool.py"
    launcher.write_text(
        "import json, socket, sys, time\n"
        "from pathlib import Path\n"
        "d = Path(sys.argv[1])\n"
        "cases = json.loads((d / 'cases.json').read_text())\n"
        "socks = []\n"
        "seats = {}\n"
        "for i, cid in enumerate(cases):\n"
        "    sp = d / f'{i}.sock'\n"
        "    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "    s.bind(str(sp))\n"
        "    s.listen(1)\n"
        "    socks.append(s)\n"
        "    ev = d / f'case-{i}'\n"
        "    ev.mkdir()\n"
        "    seats[cid] = {'native_pool_socket': str(sp), 'native_pool_evidence': str(ev)}\n"
        "(d / 'seats.json').write_text(json.dumps(seats))\n"
        "if sys.argv[2:] or True:\n"
        "    print('[head_rgb] initial-ready slots=' + str(list(range(len(cases)))), flush=True)\n"
        "time.sleep(30)\n"
    )
    monkeypatch.setenv("ROBODOJO_SHARED_STARTUP_GATE", str(gate))
    monkeypatch.setenv("ROBODOJO_SHARED_STARTUP_GROUP", "3")

    async def run_gate():
        pool = NativePool()

        async def release_when_ready():
            marker = gate / "ready-3.json"
            while not marker.is_file():
                await asyncio.sleep(0.05)
            data = json.loads(marker.read_text())
            assert data == {"group": 3, "cases": ["c0", "c1"]}
            (gate / "release").write_text("go\n")

        task = asyncio.create_task(release_when_ready())
        try:
            await pool.start(
                Path(sys.executable),
                launcher,
                {"c0": {}, "c1": {}},
                {},
                tmp_path / "pool.log",
                10,
            )
            await task
            assert set(pool.seats) == {"c0", "c1"}
        finally:
            await pool.close(5)

    asyncio.run(run_gate())

    bad_launcher = tmp_path / "bad_pool.py"
    bad_launcher.write_text(launcher.read_text().replace("initial-ready", "black-frame"))
    (gate / "release").write_text("go\n")
    # Reuse a log that already contains a stale ready line from a prior pool.
    stale_log = tmp_path / "stale-pool.log"
    stale_log.write_text("[head_rgb] initial-ready slots=[0]\n")

    async def run_bad():
        pool = NativePool()
        try:
            with pytest.raises(ValueError, match="initial head capture is missing valid pixels"):
                await pool.start(
                    Path(sys.executable),
                    bad_launcher,
                    {"c0": {}},
                    {},
                    stale_log,
                    10,
                )
        finally:
            await pool.close(5)

    asyncio.run(run_bad())

    # Startup gate timeout must raise RuntimeError (evaluation-side), not bare TimeoutError.
    (gate / "release").unlink()

    async def run_gate_timeout():
        pool = NativePool()
        try:
            with pytest.raises(RuntimeError, match="timed out waiting for the shared startup gate"):
                await pool.start(
                    Path(sys.executable),
                    launcher,
                    {"c0": {}},
                    {},
                    tmp_path / "timeout-pool.log",
                    0.5,
                )
        finally:
            await pool.close(5)

    asyncio.run(run_gate_timeout())


def test_opencode_relay_normalizes_input_images_and_expands_sse_over_aiohttp() -> None:
    body = {
        "model": "gpt-6.1-sol",
        "stream": True,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "obs"},
                    {"type": "input_image", "image_url": "data:image/jpeg;base64,abc"},
                    {"type": "input_image", "image_url": "data:image/jpeg;base64,def", "detail": "high"},
                ],
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [{"type": "input_image", "image_url": "data:image/jpeg;base64,ghi"}],
            },
        ],
    }
    normalized = normalize_opencode_responses_body(body)
    assert normalized["input"][0]["content"][1]["detail"] == "auto"
    assert normalized["input"][0]["content"][2]["detail"] == "high"
    assert normalized["input"][1]["output"][0]["detail"] == "auto"
    assert "detail" not in body["input"][0]["content"][1]

    raw_sse = [
        ": keepalive",
        "",
        "event: response.created",
        'data: {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress", "output": []}}',
        "",
        "event: response.output_item.done",
        'data: {"type": "response.output_item.done", "output_index": 0, "item": {"id": "rs_1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "think"}]}}',
        "",
        "event: response.output_item.done",
        'data: {"type": "response.output_item.done", "output_index": 1, "item": {"id": "fc_1", "type": "function_call", "call_id": "c1", "name": "bash", "arguments": "{\\"cmd\\":\\"ls\\"}", "status": "completed"}}',
        "",
        "event: response.output_item.added",
        'data: {"type": "response.output_item.added", "output_index": 2, "item": {"id": "msg_1", "type": "message", "role": "assistant", "content": [], "status": "in_progress"}}',
        "",
        "event: response.output_item.done",
        'data: {"type": "response.output_item.done", "output_index": 2, "item": {"id": "msg_1", "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}], "status": "completed"}}',
        "",
        "event: response.completed",
        'data: {"type": "response.completed", "response": {"id": "resp_1", "status": "completed", "output": []}}',
        "",
    ]

    async def _aiter(items: list[str]):
        for item in items:
            yield item

    chunks = asyncio.run(_collect_async(expand_responses_sse_lines(_aiter(raw_sse))))
    assert ": keepalive\n\n" in "".join(chunks)

    captured: dict[str, object] = {}

    class _MockUpstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            captured["path"] = self.path
            captured["body"] = json.loads(self.rfile.read(length))
            payload = ("\n".join(raw_sse) + "\n").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    srv = HTTPServer(("127.0.0.1", 0), _MockUpstream)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        app = FastAPI()
        mount_opencode_relay(app, {"tok1": f"http://127.0.0.1:{port}/ng-rollout/0-0/v1"})
        client = TestClient(app)

        assert client.post("/_opencode_relay/missing/v1/responses", json=body).status_code == 404
        resp = client.post("/_opencode_relay/tok1/v1/responses", json=body)
        assert resp.status_code == 200
        assert captured["path"] == "/ng-rollout/0-0/v1/responses"
        assert captured["body"]["input"][0]["content"][1]["detail"] == "auto"
        assert ": keepalive\n\n" in resp.text
        events = [json.loads(line[5:].strip()) for line in resp.text.splitlines() if line.startswith("data:")]
        types_by_idx = [(e["type"], e.get("output_index")) for e in events]
        assert ("response.output_item.added", 0) in types_by_idx
        assert ("response.reasoning_summary_text.delta", 0) in types_by_idx
        assert ("response.output_item.added", 1) in types_by_idx
        assert ("response.function_call_arguments.delta", 1) in types_by_idx
        assert types_by_idx.count(("response.output_item.added", 2)) == 1
        assert types_by_idx.index(("response.output_item.added", 1)) < types_by_idx.index(
            ("response.output_item.done", 1)
        )
    finally:
        srv.shutdown()
        srv.server_close()


async def _collect_async(stream):
    return [chunk async for chunk in stream]


def test_agent_seed_request_and_pool_recovery_and_clean_cancel(runtime, tmp_path: Path, monkeypatch) -> None:
    template, manifest, _, digest = runtime
    second = "general_pickup__standard__g1__l1"
    panel = json.loads(manifest.read_text())
    panel["panel_id"], panel["panel_sha256"] = "test", "a" * 64
    panel["cases"].append({**panel["cases"][0], "case_id": second, "layout_id": 1})
    write_json(manifest, panel)
    validity = tmp_path / "validity.json"
    write_json(
        validity,
        {
            "panel_id": "test",
            "panel_sha256": "a" * 64,
            "validity_sha256": "b" * 64,
            "assignment": [{"slot": case, "case_id": case} for case in (CASE, second)],
        },
    )
    launcher = tmp_path / "pool.py"
    child_pid_file = tmp_path / "pool-child.pid"
    cohort_urls_file = tmp_path / "cohort-urls.json"
    launcher.write_text(
        "import json, socket, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "root = Path(sys.argv[1])\n"
        "cases = json.loads((root / 'cases.json').read_text())\n"
        f"Path({str(cohort_urls_file)!r}).write_text(json.dumps({{k: v['model']['base_url'] for k, v in cases.items()}}))\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"Path({str(child_pid_file)!r}).write_text(str(child.pid))\n"
        "sockets = []\n"
        "for idx, bound in enumerate(cases.values()):\n"
        "    listener = socket.socket(socket.AF_UNIX)\n"
        "    bound['native_pool_socket'] = str(root / f'{idx}.sock')\n"
        "    bound['native_pool_evidence'] = str(root / f'case-{idx}')\n"
        "    bound['step_limit'] = 999999\n"
        "    listener.bind(bound['native_pool_socket'])\n"
        "    listener.listen()\n"
        "    sockets.append(listener)\n"
        "(root / 'seats.json').write_text(json.dumps(cases))\n"
        "time.sleep(60)\n"
    )
    config = RoboDojoEnvironmentConfig(
        name="robodojo_environment",
        host="127.0.0.1",
        port=8012,
        entrypoint="app.py",
        agent_server={"type": "responses_api_agents", "name": "robodojo"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        model_name="test",
        bridge_config=template,
        panel_manifest=manifest,
        case_validity=validity,
        bridge_python=Path(sys.executable),
        simulation_python=Path(sys.executable),
        output_root=tmp_path / "outputs",
        evidence_export_root=tmp_path / "evidence",
        pool_cases=f"{CASE},{second}",
        pool_launcher=launcher,
        pool_startup_seconds=5,
    )
    client = ServerClient(
        head_server_config={"host": "127.0.0.1", "port": 8000},
        global_config_dict=OmegaConf.create(
            {"policy_model": {"responses_api_models": {"vllm_model": {"host": "127.0.0.1", "port": 8011}}}}
        ),
    )
    server = RoboDojoEnvironmentServer(config=config, server_client=client)
    seed_req = server.agent_seed_request(_episode_request(CASE, 0))
    assert seed_req.agent_context.values == {
        "case_id": CASE,
        "assigned_case_id": CASE,
        "layout_sha256": digest,
    }

    seen_urls: list[str] = []
    seen_limits: list[int] = []
    call_count = 0
    old_ep_entered = asyncio.Event()
    release_old_ep = asyncio.Event()
    new_ep_entered = asyncio.Event()
    release_new_ep = asyncio.Event()

    async def run():
        async def fake_episode(self, bound):
            nonlocal call_count
            call_count += 1
            seen_urls.append(bound["model"]["base_url"])
            seen_limits.append(bound["step_limit"])
            if call_count == 1:
                self.cleanup_confirmed = True
                raise asyncio.CancelledError()
            if call_count == 2:
                old_ep_entered.set()
                await release_old_ep.wait()
                raise EpisodeError("pool crashed")
            if call_count == 3:
                new_ep_entered.set()
                await release_new_ep.wait()
            return "scored", HorizonScores(
                within_native=True,
                within_budget=True,
                score_within_native=1.0,
                score_within_budget=1.0,
                task_step_limit=200,
                control_budget=400,
            )

        def broken_export(self, root):
            raise TypeError("malformed summary")

        monkeypatch.setattr(Episode, "run", fake_episode)
        monkeypatch.setattr(Episode, "export_evidence", broken_export)
        app = server.setup_webserver()
        async with app.router.lifespan_context(app):
            # 1. Clean cancel after cleanup_confirmed=True must NOT poison server.
            req0 = _episode_request(CASE, 0)
            with pytest.raises(asyncio.CancelledError):
                await server.run(req0, CleanupContext(episode_id=req0.episode_id, cleanup_timeout_seconds=5.0))
            assert server._cleanup_unconfirmed is False
            # Cohort model URL must be unprefixed (no rollout-specific prefix).
            cohort_urls = json.loads(cohort_urls_file.read_text())
            assert set(cohort_urls.values()) == {"http://127.0.0.1:8011/v1"}

            # 2. Start episode on pool 1, crash pool 1's leader while child process survives.
            req_old = _episode_request(CASE, 1)
            old_task = asyncio.create_task(
                server.run(req_old, CleanupContext(episode_id=req_old.episode_id, cleanup_timeout_seconds=5.0))
            )
            await asyncio.wait_for(old_ep_entered.wait(), 5.0)
            first_pid = server._pool.process.pid
            first_child_pid = int(child_pid_file.read_text())
            os.kill(first_child_pid, 0)  # child is alive before crash recovery
            server._pool.process.kill()
            await server._pool.process.wait()

            # 3. New request restarts pool 2, kills old pool 1's orphaned child, and claims CASE on pool 2.
            req_new = _episode_request(CASE, 2)
            new_task = asyncio.create_task(
                server.run(req_new, CleanupContext(episode_id=req_new.episode_id, cleanup_timeout_seconds=5.0))
            )
            await asyncio.wait_for(new_ep_entered.wait(), 5.0)
            assert server._pool.process.pid != first_pid
            with pytest.raises(ProcessLookupError):
                os.kill(first_child_pid, 0)

            # 4. Old episode finishing on pool 1 must NOT release pool 2's active claim for CASE.
            release_old_ep.set()
            old_res = await old_task
            assert old_res.result["mask_sample"] is True
            assert CASE in server._pool.active_cases

            # A concurrent duplicate request for CASE on pool 2 must be rejected as already running.
            req_dup = _episode_request(CASE, 3)
            dup_res = await server.run(
                req_dup, CleanupContext(episode_id=req_dup.episode_id, cleanup_timeout_seconds=5.0)
            )
            assert dup_res.result["mask_sample"] is True

            release_new_ep.set()
            res_new = await new_task
            assert res_new.result["reward"] == 1.0
            assert res_new.result["mask_sample"] is False
            assert len(res_new.result["ng_verifier_invocations"]) == 1
            assert seen_limits == [400, 400, 400]
            assert len(set(seen_urls)) == 3

            # 5. Unconfirmed cancel (cleanup_confirmed=False) MUST mark _cleanup_unconfirmed=True.
            async def unconfirmed_cancel(self, bound):
                raise asyncio.CancelledError()

            monkeypatch.setattr(Episode, "run", unconfirmed_cancel)
            req_bad_cancel = _episode_request(CASE, 4)
            with pytest.raises(asyncio.CancelledError):
                await server.run(
                    req_bad_cancel,
                    CleanupContext(episode_id=req_bad_cancel.episode_id, cleanup_timeout_seconds=5.0),
                )
            assert server._cleanup_unconfirmed is True

    with rollout_context("robodojo-recovery-test"):
        asyncio.run(run())
