# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for RoboLab prepare, terminal auditing, OpenCode relay, and environment server."""

import asyncio
import hashlib
import json
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from benchmarks.robolab.prepare import load_case_manifest, prepare
from environment_servers.robolab.app import (
    RoboLabEnvironmentConfig,
    RoboLabEnvironmentServer,
    RoboLabTaskInput,
)
from environment_servers.robolab.opencode_relay import (
    mount_opencode_relay,
    normalize_opencode_responses_body,
)
from environment_servers.robolab.terminal import read_condition
from nemo_gym.client import ServerClient
from nemo_gym.config import AggregateMetricsRequest
from nemo_gym.core import AGENT_LIMIT, ENVIRONMENT_SERVER_FAILED, BaseEpisodeRequest, rollout_context
from nemo_gym.server_runtime import CleanupContext
from nemo_gym.trajectory import grading_result_binding


SOURCE_SHA = "b" * 64
OFFICIAL_SHA = "c" * 64


def _make_manifest(
    *,
    sdk_version: str = "1.5",
    sdk_bundle: str | None = None,
    model_id: str = "openai/openai/gpt-6.1-sol",
    arm: str = "sdk_v15",
) -> dict[str, object]:
    vision = "rgbd" if sdk_version == "open_model" else "rgb"
    sdk: dict[str, object] = {
        "version": sdk_version,
        "execution": "ttc",
        "gates": False,
        "call_cap": 120,
        "profile": {
            "vision": vision,
            "oracle": "none",
            "contact": False,
            "perception": False,
        },
    }
    if sdk_bundle is not None:
        sdk["bundle"] = sdk_bundle
    return {
        "schema": "robolab.efb-candidate.v1",
        "official_manifest_sha256": OFFICIAL_SHA,
        "case": {
            "id": "GrabABagelTask@run-010",
            "task": "GrabABagelTask",
            "run": 10,
            "native_horizon": 450,
            "official_bundle": "6924e138fa53",
        },
        "native_run_plan": {
            "schema": "robolab.native-sequence.v2",
            "arm": arm,
            "task": "GrabABagelTask",
            "run_indices": [10],
            "reset_only_prefix": 10,
        },
        "bridge_config": {
            "integration": "robolab",
            "policies": [],
            "episode_seconds": 300,
            "harness_command": [
                "/opt/robodojo-ora/bin/node",
                "/opt/robodojo-ora/source/ember_bridge/harnesses/opencode/main.mts",
                "--opencode",
                "/osmo/run/workspace/envsdk/native-programs/b34c68ea6b93/opencode/runtime",
            ],
            "native": {"task": "GrabABagelTask", "seed": 0, "startup_seconds": 60},
            "sdk": sdk,
            "model": {"id": model_id, "reasoning": "xhigh", "base_url": "https://inference-api.nvidia.com/v1"},
        },
        "harness": "opencode",
        "tools": ["observe"],
        "score_source": "native_summary",
    }


def _write_staged_manifest(root: Path, data: dict[str, object]) -> tuple[Path, str]:
    raw = (json.dumps(data, indent=2) + "\n").encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    path = root / "staged-files" / digest / "robolab-case.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path, digest


def _write_condition_receipts(
    condition: Path,
    *,
    arm: str = "sdk_v15",
    scored: bool = True,
    score: float = 1.0,
    audit_passed: bool = True,
    tamper_summary: bool = False,
) -> None:
    ep = condition / "runs" / "run-010"
    ep.mkdir(parents=True, exist_ok=True)
    (condition / "official").mkdir(parents=True, exist_ok=True)
    summary = {"run": 10, "score": score, "success": bool(score >= 1.0)}
    summary_sha = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if tamper_summary:
        summary["score"] = 0.0
    (condition / "official" / "episode_results.jsonl").write_text(json.dumps(summary) + "\n")
    derived = {
        "schema": "robolab.full-suite-episode.v2",
        "identity": {"arm": arm, "task": "GrabABagelTask", "run_index": 10, "seed": 0},
        "attempted": True,
        "scored": scored,
        "native_complete": scored,
        "native_task": "GrabABagelTask" if scored else None,
        "native_score": score if scored else None,
        "native_horizon": 450,
        "success": bool(score >= 1.0) if scored else None,
        "outcome_class": "scored" if scored else "policy_failure.decision_budget",
        "bindings": {"source_manifest_sha256": SOURCE_SHA},
        "evidence": {
            "archive_audit_passed": audit_passed,
            "official_summary_sha256": summary_sha,
        },
    }
    (ep / "episode-result.json").write_text(json.dumps(derived) + "\n")
    (ep / "terminal-record.json").write_text(
        json.dumps({"schema": "robolab.sealed-terminal.v1", "derived_episode_result": derived}) + "\n"
    )
    (condition / "condition-result.json").write_text(
        json.dumps(
            {
                "task": "GrabABagelTask",
                "arm": arm,
                "requested_runs": [10],
                "attempted_runs": [10],
                "process": {"returncode": 0, "forced": False, "live_owned_after_cleanup": []},
                "error": None,
                "interrupted_signal": None,
            }
        )
        + "\n"
    )


@pytest.mark.parametrize(
    ("sdk_version", "sdk_bundle", "model_id", "arm"),
    [
        ("1.5", None, "openai/openai/gpt-6.1-sol", "sdk_v15"),
        ("open_model", "assisted", "openai/openai/gpt-6.1-sol", "sdk_v15"),
        ("open_model", "core", "nvidia/nvidia/nemotron-3.5-super-vl", "open_model"),
    ],
)
def test_prepare_accepts_sdk15_open_model_and_multiple_models(
    tmp_path: Path, sdk_version: str, sdk_bundle: str | None, model_id: str, arm: str
) -> None:
    manifest_path, digest = _write_staged_manifest(
        tmp_path, _make_manifest(sdk_version=sdk_version, sdk_bundle=sdk_bundle, model_id=model_id, arm=arm)
    )
    out = prepare(manifest_path, tmp_path / "case.jsonl")
    row = json.loads(out.read_text())
    assert row["case_id"] == "GrabABagelTask@run-010"
    assert row["verifier_metadata"]["manifest_sha256"] == digest
    assert row["verifier_metadata"]["native_run_plan"]["arm"] == arm


def test_prepare_rejects_mismatched_vision_or_staged_digest(tmp_path: Path) -> None:
    bad = _make_manifest(sdk_version="open_model", sdk_bundle="assisted")
    bad["bridge_config"]["sdk"]["profile"]["vision"] = "rgb"  # type: ignore[index]
    bad_path, _ = _write_staged_manifest(tmp_path, bad)
    with pytest.raises(ValueError, match="SDK 1.5 RGB or open_model RGB-D"):
        load_case_manifest(bad_path)

    tampered = tmp_path / "staged-files" / ("f" * 64) / "robolab-case.json"
    tampered.parent.mkdir(parents=True)
    tampered.write_text(json.dumps(_make_manifest()))
    with pytest.raises(ValueError, match="does not match its staged SHA-256"):
        load_case_manifest(tampered)


def test_read_condition_audits_scored_unscored_and_tampered_receipts(tmp_path: Path) -> None:
    manifest_path, digest = _write_staged_manifest(
        tmp_path, _make_manifest(sdk_version="open_model", sdk_bundle="core")
    )
    manifest, _ = load_case_manifest(manifest_path)

    cond_ok = tmp_path / "cond-ok"
    _write_condition_receipts(cond_ok, scored=True, score=1.0)
    res = read_condition(cond_ok, manifest, digest, SOURCE_SHA)
    assert res["reward"] == 1.0
    assert res["native_success"] is True
    assert res["cleanup_confirmed"] is True

    cond_unscored = tmp_path / "cond-unscored"
    _write_condition_receipts(cond_unscored, scored=False)
    res_unscored = read_condition(cond_unscored, manifest, digest, SOURCE_SHA)
    assert res_unscored["native_score"] is None
    assert res_unscored["native_outcome"] == "policy_failure.decision_budget"
    assert "reward" not in res_unscored

    cond_tampered = tmp_path / "cond-tampered"
    _write_condition_receipts(cond_tampered, scored=True, score=1.0, tamper_summary=True)
    with pytest.raises(ValueError, match="sealed official summary"):
        read_condition(cond_tampered, manifest, digest, SOURCE_SHA)


def test_environment_server_runs_mock_backend_and_aggregates_metrics(tmp_path: Path) -> None:
    manifest_path, digest = _write_staged_manifest(
        tmp_path, _make_manifest(sdk_version="open_model", sdk_bundle="assisted")
    )
    backend = tmp_path / "mock-robolab-condition"
    backend.write_text(
        f"#!{sys.executable}\n"
        "import hashlib, json, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "cond = Path(args[args.index('--condition-output') + 1])\n"
        "ep = cond / 'runs' / 'run-010'\n"
        "ep.mkdir(parents=True, exist_ok=True)\n"
        "(cond / 'official').mkdir(parents=True, exist_ok=True)\n"
        "summary = {'run': 10, 'score': 1.0, 'success': True}\n"
        "sha = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(',', ':')).encode()).hexdigest()\n"
        "(cond / 'official' / 'episode_results.jsonl').write_text(json.dumps(summary) + '\\n')\n"
        "derived = {\n"
        "  'schema': 'robolab.full-suite-episode.v2',\n"
        "  'identity': {'arm': 'sdk_v15', 'task': 'GrabABagelTask', 'run_index': 10, 'seed': 0},\n"
        "  'attempted': True, 'scored': True, 'native_complete': True,\n"
        "  'native_task': 'GrabABagelTask', 'native_score': 1.0, 'native_horizon': 450,\n"
        "  'success': True, 'outcome_class': 'scored',\n"
        f"  'bindings': {{'source_manifest_sha256': '{SOURCE_SHA}'}},\n"
        "  'evidence': {'archive_audit_passed': True, 'official_summary_sha256': sha},\n"
        "}\n"
        "(ep / 'episode-result.json').write_text(json.dumps(derived) + '\\n')\n"
        "(ep / 'terminal-record.json').write_text(json.dumps({'schema': 'robolab.sealed-terminal.v1', 'derived_episode_result': derived}) + '\\n')\n"
        "(cond / 'condition-result.json').write_text(json.dumps({\n"
        "  'task': 'GrabABagelTask', 'arm': 'sdk_v15', 'requested_runs': [10], 'attempted_runs': [10],\n"
        "  'process': {'returncode': 0, 'forced': False, 'live_owned_after_cleanup': []},\n"
        "  'error': None, 'interrupted_signal': None,\n"
        "}) + '\\n')\n"
    )
    backend.chmod(backend.stat().st_mode | stat.S_IXUSR)

    config = RoboLabEnvironmentConfig(
        name="robolab_environment",
        host="127.0.0.1",
        port=8015,
        entrypoint="app.py",
        agent_server={"type": "responses_api_agents", "name": "robolab"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        case_manifest=manifest_path,
        backend=backend,
        output_root=tmp_path / "out",
        native_source_sha256=SOURCE_SHA,
        use_policy_model_relay=True,
    )
    client = ServerClient(
        head_server_config={"host": "127.0.0.1", "port": 8000},
        global_config_dict=OmegaConf.create(
            {"policy_model": {"responses_api_models": {"vllm_model": {"host": "127.0.0.1", "port": 8011}}}}
        ),
    )
    server = RoboLabEnvironmentServer(config=config, server_client=client)
    server.setup_webserver()

    req = BaseEpisodeRequest[RoboLabTaskInput].model_validate(
        {
            "episode_id": {"rollout_id": "robolab-0", "attempt": 0},
            "task": {
                "task_id": {"taskset": "robolab", "task_id": "case-0"},
                "task_input": {
                    "case_id": "GrabABagelTask@run-010",
                    "verifier_metadata": {"seed": 0, "manifest_sha256": digest},
                },
            },
        }
    )
    cleanup = CleanupContext(episode_id=req.episode_id, cleanup_timeout_seconds=10.0)
    with rollout_context("robolab-test"):
        resp = asyncio.run(server.run(req, cleanup))
    assert resp.result is not None
    assert resp.result["reward"] == 1.0
    assert resp.result["native_score"] == 1.0
    assert resp.result["native_success"] is True
    assert resp.result["mask_sample"] is False
    (invocation,) = resp.result["ng_verifier_invocations"]
    assert invocation["grading_record"]["verdict"] == "scored"
    assert invocation["grading_record"]["score"] == resp.result["reward"]
    assert invocation["grading_record"]["result_binding_sha256"] == grading_result_binding(resp.result)

    agg = asyncio.run(server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[resp.result])))
    assert agg.key_metrics["num/scored_cases"] == 1
    assert agg.key_metrics["mean/native_score"] == 1.0


def test_robolab_opencode_relay_streams_via_aiohttp() -> None:
    body = {
        "model": "gpt-6.1-sol",
        "stream": True,
        "input": [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/jpeg;base64,AAA"}]}],
    }
    assert normalize_opencode_responses_body(body)["input"][0]["content"][0]["detail"] == "auto"
    captured: dict[str, object] = {}

    class _MockUpstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            captured["body"] = json.loads(self.rfile.read(length))
            sse = (
                "event: response.output_item.done\n"
                'data: {"type":"response.output_item.done","output_index":0,"item":{"id":"fc1","type":"function_call","call_id":"c1","name":"bash","arguments":"{}","status":"completed"}}\n\n'
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(sse)))
            self.end_headers()
            self.wfile.write(sse)

    srv = HTTPServer(("127.0.0.1", 0), _MockUpstream)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        app = FastAPI()
        mount_opencode_relay(app, {"rl1": f"http://127.0.0.1:{port}/v1"})
        client = TestClient(app)
        assert client.post("/_opencode_relay/missing/v1/responses", json=body).status_code == 404
        resp = client.post("/_opencode_relay/rl1/v1/responses", json=body)
        assert resp.status_code == 200
        assert captured["body"]["input"][0]["content"][0]["detail"] == "auto"
        assert "response.output_item.added" in resp.text
        assert "response.function_call_arguments.delta" in resp.text
    finally:
        srv.shutdown()
        srv.server_close()


def test_suite_manifest_scores_policy_failure_as_unmasked_zero_and_masks_infra(tmp_path: Path) -> None:
    base = _make_manifest(sdk_version="open_model", sdk_bundle="core", arm="open_model")
    suite = {
        "schema": "robolab.efb-suite.v1",
        "official_manifest_sha256": OFFICIAL_SHA,
        "arm": "open_model",
        "cases": [
            {
                "id": "GrabABagelTask@run-010",
                "task": "GrabABagelTask",
                "run": 10,
                "native_horizon": 450,
                "official_bundle": "6924e138fa53",
            },
            {
                "id": "GrabABagelTask@run-011",
                "task": "GrabABagelTask",
                "run": 11,
                "native_horizon": 450,
                "official_bundle": "6924e138fa53",
            },
            {
                "id": "GrabABagelTask@run-012",
                "task": "GrabABagelTask",
                "run": 12,
                "native_horizon": 450,
                "official_bundle": "6924e138fa53",
            },
        ],
        "bridge_config": base["bridge_config"],
        "harness": "opencode",
        "tools": ["observe"],
        "score_source": "native_summary",
    }
    manifest_path, digest = _write_staged_manifest(tmp_path, suite)
    prepared = prepare(manifest_path, tmp_path / "suite.jsonl")
    rows = [json.loads(line) for line in prepared.read_text().splitlines()]
    assert [r["case_id"] for r in rows] == [
        "GrabABagelTask@run-010",
        "GrabABagelTask@run-011",
        "GrabABagelTask@run-012",
    ]

    backend = tmp_path / "mock-suite-backend"
    backend.write_text(
        f"#!{sys.executable}\n"
        "import hashlib, json, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "case_cfg = json.loads(Path(args[args.index('--case-manifest') + 1]).read_text())\n"
        "run_idx = case_cfg['case']['run']\n"
        "cond = Path(args[args.index('--condition-output') + 1])\n"
        "ep = cond / 'runs' / f'run-{run_idx:03d}'\n"
        "ep.mkdir(parents=True, exist_ok=True)\n"
        "(cond / 'official').mkdir(parents=True, exist_ok=True)\n"
        "scored = (run_idx == 10)\n"
        "outcome = 'scored' if run_idx == 10 else ('policy_failure.decision_budget' if run_idx == 11 else 'infrastructure.worker')\n"
        "summary = {'run': run_idx, 'score': 1.0, 'success': True}\n"
        "sha = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(',', ':')).encode()).hexdigest()\n"
        "(cond / 'official' / 'episode_results.jsonl').write_text(json.dumps(summary) + '\\n')\n"
        "derived = {\n"
        "  'schema': 'robolab.full-suite-episode.v2',\n"
        "  'identity': {'arm': 'open_model', 'task': 'GrabABagelTask', 'run_index': run_idx, 'seed': 0},\n"
        "  'attempted': True, 'scored': scored, 'native_complete': scored,\n"
        "  'native_task': 'GrabABagelTask' if scored else None,\n"
        "  'native_score': 1.0 if scored else None, 'native_horizon': 450,\n"
        "  'success': True if scored else None, 'outcome_class': outcome,\n"
        f"  'bindings': {{'source_manifest_sha256': '{SOURCE_SHA}'}},\n"
        "  'evidence': {'archive_audit_passed': True, 'official_summary_sha256': sha},\n"
        "}\n"
        "(ep / 'episode-result.json').write_text(json.dumps(derived) + '\\n')\n"
        "(ep / 'terminal-record.json').write_text(json.dumps({'schema': 'robolab.sealed-terminal.v1', 'derived_episode_result': derived}) + '\\n')\n"
        "(cond / 'condition-result.json').write_text(json.dumps({\n"
        "  'task': 'GrabABagelTask', 'arm': 'open_model', 'requested_runs': [run_idx], 'attempted_runs': [run_idx],\n"
        "  'process': {'returncode': 0, 'forced': False, 'live_owned_after_cleanup': []},\n"
        "  'error': None, 'interrupted_signal': None,\n"
        "}) + '\\n')\n"
    )
    backend.chmod(backend.stat().st_mode | stat.S_IXUSR)

    config = RoboLabEnvironmentConfig(
        name="robolab_environment",
        host="127.0.0.1",
        port=8016,
        entrypoint="app.py",
        agent_server={"type": "responses_api_agents", "name": "robolab"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        case_manifest=manifest_path,
        backend=backend,
        output_root=tmp_path / "suite-out",
        native_source_sha256=SOURCE_SHA,
    )
    client = ServerClient(
        head_server_config={"host": "127.0.0.1", "port": 8000},
        global_config_dict=OmegaConf.create(
            {"policy_model": {"responses_api_models": {"vllm_model": {"host": "127.0.0.1", "port": 8011}}}}
        ),
    )
    server = RoboLabEnvironmentServer(config=config, server_client=client)
    server.setup_webserver()

    results = []
    with rollout_context("robolab-suite-test"):
        for idx, row in enumerate(rows):
            req = BaseEpisodeRequest[RoboLabTaskInput].model_validate(
                {
                    "episode_id": {"rollout_id": f"robolab-suite-{idx}", "attempt": 0},
                    "task": {
                        "task_id": {"taskset": "robolab", "task_id": f"case-{idx}"},
                        "task_input": {
                            "case_id": row["case_id"],
                            "verifier_metadata": {"seed": 0, "manifest_sha256": digest},
                        },
                    },
                }
            )
            cleanup = CleanupContext(episode_id=req.episode_id, cleanup_timeout_seconds=10.0)
            results.append(asyncio.run(server.run(req, cleanup)).result)

    # run-010: scored 1.0; run-011: decision_budget unmasked 0.0; run-012: infra masked
    assert results[0]["reward"] == 1.0 and results[0]["mask_sample"] is False
    assert results[1]["reward"] == 0.0
    assert results[1]["native_score"] == 0.0
    assert results[1]["native_success"] is False
    assert results[1]["mask_sample"] is False
    assert results[1]["failure_kind"] == AGENT_LIMIT
    assert len(results[1]["ng_verifier_invocations"]) == 1
    assert results[1]["ng_verifier_invocations"][0]["grading_record"][
        "result_binding_sha256"
    ] == grading_result_binding(results[1])
    assert results[2]["mask_sample"] is True
    assert results[2]["failure_kind"] == ENVIRONMENT_SERVER_FAILED
    assert results[2].get("ng_verifier_invocations", []) == []

    agg = asyncio.run(server.aggregate_metrics(AggregateMetricsRequest(verify_responses=results)))
    assert agg.key_metrics["num/scored_cases"] == 2
    assert agg.key_metrics["num/unscored_cases"] == 1
    assert agg.key_metrics["mean/native_score"] == pytest.approx(0.5)
    assert agg.key_metrics["mean/native_success"] == pytest.approx(0.5)
