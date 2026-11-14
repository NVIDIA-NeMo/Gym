# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check capture boundaries/masking; mock grades are not semantic calibration."""

import copy
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import SecretStr
from starlette.requests import Request

from benchmarks.partner_if.tests.fixture_data import PROBE, RUNTIME, SOURCE
from resources_servers.partner_if.native_evidence import live_agent_evidence
from responses_api_agents.ngc_recorded_context import app as replay_app
from responses_api_agents.ngc_recorded_context import replay


TASK = json.loads((SOURCE / "tasks/ExampleProject/pif_029.json").read_text())
BUNDLE = json.loads((SOURCE / "assets/ngc-recorded-context/bundle.json").read_text()) | {"runtime": RUNTIME}
PARAMS = TASK["responses_create_params"]


def call(name, args, identity="c"):
    return {"id": identity, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def probe(identity="probe", uri=None):
    uri = uri or BUNDLE["sibling"]["input_uri"]
    return call("Bash", {"command": f"python3 -I {PROBE} --uri '{uri}'"}, identity)


def draft(text="Publish a new artifact or choose another version.", identity="draft"):
    return call("Write", {"file_path": "/workspace/logs/answer.txt", "content": text}, identity)


def response(*calls, content=None, finish=None):
    return {
        "id": "test_" + uuid4().hex,
        "model": "test",
        "choices": [
            {
                "message": {"role": "assistant", "content": content, "tool_calls": list(calls)},
                "finish_reason": finish or ("tool_calls" if calls else "stop"),
            }
        ],
    }


async def episode(tmp_path, responses):
    queue = list(responses)

    async def request_api(payload, timeout):
        if not queue:
            raise AssertionError("Unexpected retry")
        result = queue.pop(0)
        if isinstance(result, Exception):
            raise result
        return 200, result

    config = replay.ReplayConfig(package_dir=SOURCE / "assets", artifacts_dir=tmp_path)
    return await replay.execute(config, PARAMS, model="test", secret="private-key", request_api=request_api)


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["before", "same", "absent"])
async def test_turn_receipts_distinguish_prior_evidence_from_same_batch_or_no_probe(tmp_path, order):
    responses = (
        [response(probe()), response(draft())]
        if order == "before"
        else [response(probe(), draft())]
        if order == "same"
        else [response(draft())]
    )
    root, output, capture, diagnostics = await episode(tmp_path, responses)
    assert diagnostics["evaluation_completed"]
    receipt = json.loads(capture.artifacts["replay_receipts"]["text"])
    events = receipt["new_events"]
    probes = [e for e in events if e["verified_probe"].get("namespace") == "sibling"]
    drafts = [e for e in events if e["kind"] == "answer_draft"]
    assert bool(probes) == (order != "absent")
    assert bool(probes and probes[0]["turn"] < drafts[0]["turn"]) == (order == "before")
    if order == "same":
        assert [item["type"] for item in output] == [
            "function_call",
            "function_call",
            "function_call_output",
            "function_call_output",
        ]
    evidence = live_agent_evidence(PARAMS, output, capture)
    assert len(evidence.context) == len(PARAMS["input"])
    assert all(event.call_id != PARAMS["input"][-1].get("call_id") for event in evidence.trajectory)
    assert "Publish" in evidence.artifact_text("final_reply")
    assert (root / "call-01.request.json").is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    [
        "filesystem",
        "wrong_version",
        "output_limit",
        "refusal",
        "provider",
        "invalid_tool",
        "duplicate",
        "empty",
        "wrong_model",
    ],
)
async def test_incomplete_attempts_are_explicit_and_never_become_scores(tmp_path, kind):
    first = {
        "filesystem": response(call("Read", {"file_path": "/not-in-fixture"})),
        "wrong_version": response(probe(uri=BUNDLE["sibling"]["input_uri"] + "different")),
        "output_limit": response(content="partial", finish="length"),
        "refusal": response(content="refused", finish="content_filter"),
        "provider": RuntimeError("private-key provider down"),
        "invalid_tool": response(call("unregistered", {})),
        "duplicate": response(probe("same"), draft(identity="same")),
        "empty": response(),
        "wrong_model": response(content="answer") | {"model": "another-model"},
    }[kind]
    root, _, _, diagnostics = await episode(tmp_path, [first])
    assert diagnostics["evaluation_completed"] is False
    assert diagnostics["model_calls"] == 1
    assert "reward" not in diagnostics
    assert "private-key" not in "".join(p.read_text() for p in root.glob("*.json"))


@pytest.mark.asyncio
async def test_mention_and_inherited_actions_do_not_manufacture_new_probe_receipts(tmp_path):
    _, _, capture, diagnostics = await episode(
        tmp_path, [response(content="The sibling URI is " + BUNDLE["sibling"]["input_uri"])]
    )
    assert diagnostics["new_sibling_probe_count"] == 0
    assert json.loads(capture.artifacts["replay_receipts"]["text"])["new_events"] == []


@pytest.mark.parametrize("fault", ["script", "compound", "result", "duplicate_uri"])
def test_unexecuted_or_misattributed_probe_cannot_earn_trusted_receipt(fault):
    fn = probe()["function"]
    if fault == "script":
        fn["arguments"] = fn["arguments"].replace(PROBE, "/somewhere/ngc_readonly_probe.py")
    if fault == "compound":
        args = json.loads(fn["arguments"])
        args["command"] += "; echo pretend"
        fn["arguments"] = json.dumps(args)
    if fault == "duplicate_uri":
        args = json.loads(fn["arguments"])
        args["command"] += " --uri another"
        fn["arguments"] = json.dumps(args)
    event = {"kind": "sibling_probe", "function": fn, "result": json.dumps(BUNDLE["sibling"])}
    if fault == "result":
        event["result"] = "fictional result"
    with pytest.raises(ValueError):
        replay.probe_identity(event, BUNDLE)


def test_denied_exact_probe_is_valid_investigation_not_a_correctness_failure():
    bundle = copy.deepcopy(BUNDLE)
    bundle["sibling"] = {"input_uri": BUNDLE["sibling"]["input_uri"], "http_status": 403, "exists": False}
    event = {"kind": "sibling_probe", "function": probe()["function"], "result": json.dumps(bundle["sibling"])}
    assert replay.probe_identity(event, bundle)["returned_result"]["http_status"] == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["completed", "gap", "judge_down"])
async def test_gym_grades_only_complete_capture_and_retains_answer_when_judge_fails(tmp_path, monkeypatch, state):
    raw = response(draft()) if state != "gap" else response(call("Read", {"file_path": "/missing"}))
    root, output, captured, diagnostics = await episode(tmp_path, [raw])

    async def execute(*args, **kwargs):
        return root, output, captured, diagnostics

    posts = []

    async def post(**kwargs):
        posts.append(kwargs)
        if state == "judge_down":
            raise RuntimeError("judge unavailable")
        return kwargs["json"]

    async def success(result):
        pass

    async def decoded(result):
        # This mock proves routing only; it is NOT a semantic verdict.
        return result | {"reward": 0.0, "mask_sample": False, "evaluation_completed": True}

    monkeypatch.setattr(replay_app, "execute", execute)
    monkeypatch.setattr(replay_app, "raise_for_status", success)
    monkeypatch.setattr(replay_app, "get_response_json", decoded)
    config = replay_app.ReplayAgentConfig.model_construct(
        num_workers=1,
        model="test",
        inference_api_key=SecretStr("private-key"),
        resources_server=SimpleNamespace(name="partner_if"),
        replay=replay.ReplayConfig(package_dir=SOURCE / "assets", artifacts_dir=tmp_path),
    )
    agent = replay_app.ReplayAgent.model_construct(config=config, server_client=SimpleNamespace(post=post))
    result = await agent.run(
        Request({"type": "http", "headers": []}), replay_app.ReplayRunRequest.model_validate(TASK)
    )
    assert len(posts) == (state != "gap")
    assert result.mask_sample is (state != "completed")
    assert result.evaluation_completed is (state == "completed")
    if state == "completed":
        assert result.reward == 0.0
    assert (root / "grading-result.json").exists()


@pytest.mark.asyncio
async def test_audit_entrypoint_refuses_fresh_policy_requests(tmp_path, monkeypatch):
    attempted = []

    async def execute(*args, **kwargs):
        attempted.append(True)
        await kwargs["request_api"]({"model": "test", "messages": []}, 1)
        raise AssertionError("The audit endpoint allowed a fresh policy request")

    monkeypatch.setattr(replay_app, "execute", execute)
    config = replay_app.ReplayAgentConfig.model_construct(
        num_workers=1,
        model="test",
        inference_api_key=SecretStr("private-key"),
        resources_server=SimpleNamespace(name="partner_if"),
        replay=replay.ReplayConfig(package_dir=SOURCE / "assets", artifacts_dir=tmp_path),
    )
    agent = replay_app.ReplayAgent.model_construct(config=config, server_client=SimpleNamespace())
    result = await agent.run(
        Request({"type": "http", "headers": []}), replay_app.ReplayRunRequest.model_validate(TASK)
    )
    assert attempted == [True]
    assert result.mask_sample and not result.evaluation_completed
    assert "Live actor calls are disabled" in result.failure_reason


def test_replay_config_routes_only_private_profile_without_a_container():
    from omegaconf import OmegaConf

    from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig, get_first_server_config_dict

    cfg = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {"config_paths": ["benchmarks/partner_if/config_ngc_replay.yaml"], "policy_api_key": "dummy"},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    block = OmegaConf.to_container(get_first_server_config_dict(cfg, "partner_if_ngc_replay_agent"), resolve=True)
    from responses_api_agents.ngc_recorded_context.native_agent import NativeReplayAgentConfig

    config = NativeReplayAgentConfig.model_validate(block | {"name": "partner_if_ngc_replay_agent"})
    assert not hasattr(config, "harness")
    assert config.replay.max_new_turns == 6
    assert config.replay.max_policy_calls == 8
    assert config.replay.evaluation_boundary == "first_answer"
    assert config.model_server.name == "partner_if_ngc_replay_model"
    from responses_api_agents.partner_if_dispatcher.app import PartnerIFDispatcherConfig

    dispatcher_block = OmegaConf.to_container(get_first_server_config_dict(cfg, "partner_if_dispatcher"), resolve=True)
    dispatcher = PartnerIFDispatcherConfig.model_validate(dispatcher_block | {"name": "partner_if_dispatcher"})
    assert set(dispatcher.task_routes) == {"pif_029"}
    destination = dispatcher.task_routes["pif_029"].name
    assert destination == "partner_if_ngc_replay_environment_server"
    environment = OmegaConf.to_container(get_first_server_config_dict(cfg, destination), resolve=True)
    assert set(cfg[destination]["environment_servers"]) == {"single_agent_turn"}
    assert environment["verifier_continuation"]["max_activations"] == config.replay.max_new_turns
    assert environment["agent_server"]["name"] == config.name
    assert environment["resources_server"]["name"] == dispatcher.resources_server.name == "partner_if"
    assert dispatcher_block["datasets"][0]["prepare_script"] == "benchmarks/partner_if/prepare_ngc_replay.py"
    assert "datasets" not in block
    from pathlib import Path

    from nemo_gym.environment import discover_environment_catalog, load_manifest, resolve_manifest_config_path
    from nemo_gym.environment.validation import _mirror_differences, _resolve_manifest_composition

    manifest = Path(__file__).resolve().parents[1] / "config_ngc_replay" / "manifest.yaml"
    declared = load_manifest(manifest)
    resolved = resolve_manifest_config_path(manifest, declared)
    assert _mirror_differences(declared, _resolve_manifest_composition(resolved)) == {}
    entry = next(e for e in discover_environment_catalog() if e.name == "partner_if/config_ngc_replay")
    assert entry.config_path == resolved


def test_three_matched_review_requests_validate_and_keep_expected_labels_out_of_judge_input():
    from benchmarks.partner_if.replay_examples import examples
    from resources_servers.partner_if.app import VerifyRequest
    from resources_servers.partner_if.llm import payload

    controls = examples(TASK, BUNDLE, RUNTIME)
    assert [row["expected_if_passed"] for row in controls] == [True, False, True]
    for row in controls:
        body = VerifyRequest.model_validate(row["request"])
        assert body.verifier_metadata.partner_if.data_kind == "synthetic_smoke"
        evidence = live_agent_evidence(PARAMS, row["request"]["response"]["output"], body.verification_input)
        judge = payload(body.verifier_metadata.partner_if.rubrics, evidence)
        assert "expected_if_passed" not in json.dumps(judge)
        assert evidence.artifact_text("final_reply")


def test_paths_resolve_from_checkout_when_server_changes_working_directory(tmp_path, monkeypatch):
    from responses_api_agents.ngc_recorded_context.paths import GYM_ROOT

    monkeypatch.chdir(tmp_path)
    config = replay.ReplayConfig(package_dir=SOURCE / "assets", artifacts_dir="results/partner_if/path-test")
    assert config.artifacts_dir == GYM_ROOT / "results/partner_if/path-test"
    assert not hasattr(config, "agentic_if_root")
    case, _ = replay.load_inputs(config, PARAMS)
    assert len(case["messages"]) == 6


@pytest.mark.parametrize("change", ["completed_metadata_only", "prompt", "tool_arguments", "unfinished_status"])
def test_transport_normalization_accepts_only_nonoperative_completed_status(change, tmp_path):
    delivered = copy.deepcopy(PARAMS)
    for item in delivered["input"]:
        if item.get("type") == "message" and item.get("status") == "completed":
            item.pop("status")
    if change == "prompt":
        delivered["input"][0]["content"] += " changed prompt"
    elif change == "tool_arguments":
        next(item for item in delivered["input"] if item["type"] == "function_call")["arguments"] = "{}"
    elif change == "unfinished_status":
        next(item for item in delivered["input"] if item.get("role") == "assistant")["status"] = "incomplete"
    config = replay.ReplayConfig(package_dir=SOURCE / "assets", artifacts_dir=tmp_path)
    if change == "completed_metadata_only":
        replay.load_inputs(config, delivered)
    else:
        with pytest.raises(ValueError, match="pinned actor"):
            replay.load_inputs(config, delivered)
