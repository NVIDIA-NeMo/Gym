# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.robodojo.prepare import prepare
from nemo_gym.client import ServerClient
from responses_api_agents.robodojo_agent.app import RoboDojoAgent, RoboDojoAgentConfig, RoboDojoRequest
from responses_api_agents.robodojo_agent.cases import bind_case
from responses_api_agents.robodojo_agent.episode import CLEANUP_AUDIT, HORIZON_AUDIT, SUPERVISOR_PATH, Episode


CASE = "general_pickup__standard__g1__l0"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def runtime(tmp_path):
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


def test_prepare_and_bind_the_same_frozen_layout(runtime, tmp_path):
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


@pytest.mark.parametrize("fault", ["success", None, "unscored", "unclean"])
def test_run_uses_native_horizon_and_masks_missing_measurement(runtime, tmp_path, monkeypatch, fault):
    template, manifest, _, digest = runtime
    commands = []

    async def engine(self, *args, timeout, log=None):
        commands.append(args)
        if SUPERVISOR_PATH in args:
            return b"/runtime/supervise.py\n"
        if "run" in args:
            config = json.loads(Path(args[args.index("--config") + 1]).read_text())
            assert config["task"] == "general_pickup"
            assert config["step_limit"] == 400
            assert config["model"]["base_url"] == "http://127.0.0.1:8011/v1"
            native = Path(args[args.index("--output") + 1])
            write_json(native / "result.json", {"sdk_summary": {"closed": fault != "unclean", "faults": []}})
            write_json(native / "outcome.json", {"schema": "ember_bridge.outcome.v1", "readback_verified": True})
            write_json(native / "actor" / "ttc-runtime.json", {"state": "closed"})
            (native / "evaluation").mkdir()
            (native / "evaluation" / "sensors.mp4").touch()
            return b""
        if CLEANUP_AUDIT in args or "validate" in args:
            return b""
        if "result" in args:
            return json.dumps(
                {
                    "schema": "ember_bridge.evaluation_result.v1",
                    "counted": fault in (None, "success"),
                    "class": (
                        "scored"
                        if fault == "success"
                        else "policy_failure.deadline"
                        if fault is None
                        else "infrastructure.worker"
                    ),
                }
            ).encode()
        if HORIZON_AUDIT in args:
            if fault == "success":
                assert args[-2:] == ("scored", "native_x2")
                return b'{"within_native":true,"within_budget":true,"score_within_native":1,"score_within_budget":1,"task_step_limit":200,"control_budget":400}'
            assert args[-2:] == ("policy_failure.deadline", "native_x2")
            return b'{"within_native":false,"within_budget":false,"score_within_native":0.3,"score_within_budget":0.4,"task_step_limit":200,"control_budget":400}'
        raise AssertionError(args)

    monkeypatch.setattr(Episode, "execute", engine)
    client = ServerClient(
        head_server_config={"host": "127.0.0.1", "port": 8000},
        global_config_dict=OmegaConf.create(
            {"policy_model": {"responses_api_models": {"vllm_model": {"host": "127.0.0.1", "port": 8011}}}}
        ),
    )
    config = RoboDojoAgentConfig(
        name="robodojo",
        host="127.0.0.1",
        port=8012,
        entrypoint="app.py",
        model_server={"type": "responses_api_models", "name": "policy_model"},
        model_name="test",
        bridge_config=template,
        panel_manifest=manifest,
        bridge_python=Path(sys.executable),
        simulation_python=Path(sys.executable),
        output_root=tmp_path / "outputs",
    )
    agent = RoboDojoAgent(config=config, server_client=client)
    row = RoboDojoRequest(case_id=CASE, responses_create_params={"input": []})
    result = asyncio.run(agent.run(row))
    assert result.case_id == result.assigned_case_id == CASE
    assert result.layout_sha256 == digest
    assert result.reward == float(fault == "success")
    if fault in ("unscored", "unclean"):
        assert result.mask_sample is True
        assert result.native_score is None
        assert not any(HORIZON_AUDIT in command for command in commands)
        return
    assert result.mask_sample is False
    assert result.native_score == (1 if fault == "success" else 0.3)
    assert result.extended_score == (1 if fault == "success" else 0.4)
    assert Path(result.video_path).is_file()
    assert (Path(result.evidence_dir) / "horizon.json").is_file()


def test_timeout_terminates_an_owned_subprocess(tmp_path):
    episode = Episode(tmp_path, Path(sys.executable), Path(sys.executable), None, 5, 1)
    with pytest.raises(TimeoutError):
        asyncio.run(episode.execute(sys.executable, "-c", "import time; time.sleep(10)", timeout=0.1))
    assert asyncio.run(episode.execute(sys.executable, "-c", "print('closed')", timeout=5)) == b"closed\n"
