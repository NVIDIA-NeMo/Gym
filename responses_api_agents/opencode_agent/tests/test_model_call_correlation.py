# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from nemo_gym.base_responses_api_model import (
    CaptureStore,
    merge_model_call_capture_into_record,
    read_model_call_records,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.rollout_collection import _build_trajectory_record
from nemo_gym.rollout_observability import (
    AgentObservationBundle,
    ContextCompactionObservation,
    TrajectoryRecord,
    join_assistant_message_calls,
    join_model_call_observations,
)
from responses_api_agents.opencode_agent.app import _parse_opencode_session
from responses_api_agents.opencode_agent.tests.test_app import _session_db
from responses_api_agents.opencode_sandboxed_agent.app import parse_opencode_observations


POLICY = ModelServerRef(type="responses_api_models", name="policy")
ROW = {"_ng_task_index": 0, "_ng_rollout_index": 0}


@pytest.fixture(params=[_parse_opencode_session, parse_opencode_observations], ids=["local", "sandboxed"])
def parse(request):
    return request.param


def _policy(*, text="same answer", **message):
    return (
        {"role": "assistant", "time": {"created": 1000, "completed": 2000}, **message},
        [{"type": "step-start"}, {"type": "text", "text": text}, {"type": "step-finish"}],
    )


def _observations(parse, db):
    trajectory = TrajectoryRecord(task_id="0", rollout_id="0-0")
    observations = parse(db, "fallback", trajectory, model_ref=POLICY)
    return observations, trajectory


def _capture(store, call_id, message_id, *, session="root", **updates):
    exchange = {
        "model_call_id": call_id,
        "client_session_id": session,
        "model_ref": POLICY.model_dump(),
        "dialect": "responses",
        "status_code": 200,
        "request": {"input": "same request"},
        "response": {"id": f"response-{call_id}", "status": "completed", "output": []},
        **updates,
    }
    if message_id is not None:
        exchange["client_assistant_message_id"] = message_id
    store.record("0-0", exchange)


def _merge_record(store, observations, trajectory):
    record = {
        **ROW,
        "ng_agent_observations": observations.model_dump(mode="json"),
        "ng_trajectory": trajectory.model_dump(mode="json"),
    }
    merge_model_call_capture_into_record(record, [store.root], include_payloads=True)
    return record


def _merge(store, observations, trajectory):
    record = _merge_record(store, observations, trajectory)
    bundle = AgentObservationBundle.model_validate(record["ng_agent_observations"])
    return record, bundle, _build_trajectory_record(ROW, record)


def _ids(owner):
    return [ref.model_call_id for ref in owner.model_calls]


def _message_gaps(bundle):
    return [gap for gap in bundle.gaps if gap.code.startswith("assistant_message_call_")]


def test_capture_merge_links_retries_child_turns_and_compaction_by_ids(tmp_path, parse):
    db = _session_db(
        tmp_path,
        [
            _policy(),
            ("child", *_policy()),
            ("user", [{"type": "compaction", "auto": True}]),
            _policy(summary=True, parentID="m2", text="saved summary"),
            _policy(),
        ],
        sessions=[("root", None), ("child", "root")],
    )
    observations, trajectory = _observations(parse, db)
    store = CaptureStore(tmp_path / "capture")
    _capture(store, "retry", "m0", status_code=503, error_category="upstream_error")
    _capture(store, "success", "m0")
    _capture(store, "child", "m1", session="child")
    _capture(store, "summary", "m3")
    _capture(store, "next-turn", "m4")
    _capture(store, "title", None)

    record, bundle, canonical = _merge(store, observations, trajectory)

    turns = {turn.source_message_id: turn for turn in canonical.turns}
    assert set(turns) == {"m0", "m1", "m4"}
    assert _ids(turns["m0"]) == ["retry", "success"]
    assert _ids(turns["m1"]) == ["child"]
    assert _ids(turns["m4"]) == ["next-turn"]
    assert turns["m1"].invocation_id == "child"
    assert all(turn.source_model_ref == POLICY for turn in turns.values())
    [compaction] = [item for item in bundle.records if isinstance(item, ContextCompactionObservation)]
    assert compaction.source_message_ids == ["m3"]
    assert compaction.source_model_ref == POLICY
    assert compaction.summary == "saved summary"
    assert _ids(compaction) == ["summary"]
    assert compaction.before_model_call is compaction.after_model_call is None
    invocations = {item.invocation_id: item for item in canonical.invocations}
    assert set(_ids(invocations["root"])) == {"retry", "success", "summary", "next-turn", "title"}
    assert _ids(invocations["child"]) == ["child"]
    expected = {
        "retry": ("root", "m0"),
        "success": ("root", "m0"),
        "child": ("child", "m1"),
        "summary": ("root", "m3"),
        "next-turn": ("root", "m4"),
        "title": ("root", None),
    }
    assert {
        call.model_call_id: (call.client_session_id, call.client_assistant_message_id)
        for call in canonical.model_calls
    } == expected
    assert len(record["ng_model_call_capture"]["calls"]) == len(expected)
    assert not _message_gaps(bundle)


@pytest.mark.parametrize("error_category,status_code", [("upstream_error", 503), ("cancelled", None)])
def test_missing_pre_stream_turn_keeps_failed_call_invocation_owned(tmp_path, parse, error_category, status_code):
    db = _session_db(tmp_path, [({"role": "assistant", "error": {"name": "Aborted"}}, [])])
    observations, trajectory = _observations(parse, db)
    store = CaptureStore(tmp_path / "capture")
    _capture(store, "failed", "m0", status_code=status_code, response=None, error_category=error_category)

    record, bundle, canonical = _merge(store, observations, trajectory)

    assert canonical.turns == []
    assert _ids(canonical.invocations[0]) == ["failed"]
    [call] = canonical.model_calls
    assert (call.client_session_id, call.client_assistant_message_id) == ("root", "m0")
    assert call.response_metadata.error_category == error_category
    assert record["ng_model_call_capture"]["calls"][0]["error_category"] == error_category
    assert [(gap.code, gap.invocation_id, gap.detail) for gap in _message_gaps(bundle)] == [
        ("assistant_message_call_unmatched", "root", "m0:failed")
    ]


@pytest.mark.parametrize(
    "model_ref,expected_gap",
    [
        ({"type": "responses_api_models", "name": "other"}, "assistant_message_call_unmatched"),
        (None, "assistant_message_call_unmatched"),
    ],
)
def test_wrong_or_missing_model_scope_never_links_turn(tmp_path, parse, model_ref, expected_gap):
    observations, trajectory = _observations(parse, _session_db(tmp_path, [_policy()]))
    store = CaptureStore(tmp_path / "capture")
    _capture(store, "call", "m0", model_ref=model_ref)

    _, bundle, canonical = _merge(store, observations, trajectory)

    assert canonical.turns[0].model_calls == []
    assert _ids(canonical.invocations[0]) == ["call"]
    assert [gap.code for gap in _message_gaps(bundle)] == [expected_gap]


def test_wrong_rollout_identity_never_links_turn(tmp_path, parse):
    observations, trajectory = _observations(parse, _session_db(tmp_path, [_policy()]))
    trajectory = trajectory.model_copy(
        update={
            "rollout_id": "another-rollout",
            "turns": [turn.model_copy(update={"rollout_id": "another-rollout"}) for turn in trajectory.turns],
        }
    )
    store = CaptureStore(tmp_path / "capture")
    _capture(store, "call", "m0")

    _, bundle, canonical = _merge(store, observations, trajectory)

    assert canonical.turns[0].model_calls == []
    assert _ids(canonical.invocations[0]) == ["call"]
    assert [gap.code for gap in _message_gaps(bundle)] == ["assistant_message_call_rollout_mismatch"]
    assert "producer_trajectory_identity_mismatch" in {gap.code for gap in canonical.gaps}


@pytest.mark.parametrize("duplicate", ["target", "capture"])
def test_ambiguous_identity_never_links_turn(tmp_path, parse, duplicate):
    observations, trajectory = _observations(parse, _session_db(tmp_path, [_policy()]))
    store = CaptureStore(tmp_path / "capture")
    _capture(store, "call", "m0")
    if duplicate == "target":
        trajectory.turns.append(trajectory.turns[0].model_copy(update={"turn_no": 2}, deep=True))
    else:
        _capture(store, "call", "m0")

    record = _merge_record(store, observations, trajectory)
    bundle = AgentObservationBundle.model_validate(record["ng_agent_observations"])
    projected = TrajectoryRecord.model_validate(record["ng_trajectory"])

    assert all(not turn.model_calls for turn in projected.turns)
    assert [gap.code for gap in _message_gaps(bundle)] == ["assistant_message_call_ambiguous"]
    assert len(record["ng_model_call_capture"]["calls"]) == (2 if duplicate == "capture" else 1)


def test_legacy_calls_remain_invocation_owned_without_guessing_turn(tmp_path, parse):
    observations, trajectory = _observations(parse, _session_db(tmp_path, [_policy()]))
    store = CaptureStore(tmp_path / "capture")
    _capture(store, "legacy", None)

    _, bundle, canonical = _merge(store, observations, trajectory)

    assert canonical.turns[0].model_calls == []
    assert _ids(canonical.invocations[0]) == ["legacy"]
    assert canonical.model_calls[0].client_assistant_message_id is None
    assert not _message_gaps(bundle)


def test_association_is_idempotent_and_does_not_mutate_inputs(tmp_path, parse):
    observations, trajectory = _observations(
        parse,
        _session_db(tmp_path, [_policy(), ("user", [{"type": "compaction"}]), _policy(summary=True, parentID="m1")]),
    )
    store = CaptureStore(tmp_path / "capture")
    _capture(store, "turn", "m0")
    _capture(store, "summary", "m2")
    calls = read_model_call_records(store, "0-0")
    observations = join_model_call_observations(observations, calls)
    before = deepcopy((observations, trajectory, calls))

    first = join_assistant_message_calls(observations, trajectory, calls)
    second = join_assistant_message_calls(*first, calls)

    assert first == second
    assert (observations, trajectory, calls) == before
    assert _ids(first[1].turns[0]) == ["turn"]
    [compaction] = [item for item in first[0].records if isinstance(item, ContextCompactionObservation)]
    assert _ids(compaction) == ["summary"]
    record, _, _ = _merge(store, observations, trajectory)
    original_record = deepcopy(record)
    merge_model_call_capture_into_record(record, [store.root], include_payloads=True)
    assert record == original_record


def test_compaction_parent_membership_is_scoped_to_session_and_preserves_all_summary_ids(tmp_path, parse):
    db = _session_db(
        tmp_path,
        [
            ("user", [{"type": "compaction", "auto": True}]),
            _policy(summary=True, parentID="m0", text="first summary"),
            _policy(summary=True, parentID="m0", text="second summary"),
            ("child", *_policy(summary=True, parentID="m0", text="foreign summary")),
        ],
        sessions=[("root", None), ("child", "root")],
    )
    observations, trajectory = _observations(parse, db)
    store = CaptureStore(tmp_path / "capture")
    _capture(store, "first", "m1")
    _capture(store, "second", "m2")
    _capture(store, "foreign", "m3", session="child")

    _, bundle, canonical = _merge(store, observations, trajectory)

    assert canonical.turns == []
    [compaction] = [item for item in bundle.records if isinstance(item, ContextCompactionObservation)]
    assert compaction.source_message_ids == ["m1", "m2"]
    assert compaction.summary is None
    assert _ids(compaction) == ["first", "second"]
    assert "compaction_summary_ambiguous" in {gap.code for gap in bundle.gaps}
    assert [(gap.code, gap.invocation_id, gap.detail) for gap in _message_gaps(bundle)] == [
        ("assistant_message_call_unmatched", "child", "m3:foreign")
    ]
    child = next(item for item in canonical.invocations if item.invocation_id == "child")
    assert _ids(child) == ["foreign"]


@pytest.mark.parametrize("failure", ["invalid_trajectory", "helper_exception"])
def test_message_join_failure_preserves_invocation_ownership(tmp_path, parse, monkeypatch, failure):
    observations, trajectory = _observations(parse, _session_db(tmp_path, [_policy()]))
    store = CaptureStore(tmp_path / "capture")
    _capture(store, "captured", "m0")
    record = {
        **ROW,
        "ng_agent_observations": observations.model_dump(mode="json"),
        "ng_trajectory": trajectory.model_dump(mode="json"),
    }
    if failure == "invalid_trajectory":
        record["ng_trajectory"]["turns"] = "invalid"
    else:

        def fail_association(*_args, **_kwargs):
            raise RuntimeError("message association failed")

        monkeypatch.setattr("nemo_gym.base_responses_api_model.join_assistant_message_calls", fail_association)

    merge_model_call_capture_into_record(record, [store.root], include_payloads=True)

    bundle = AgentObservationBundle.model_validate(record["ng_agent_observations"])
    assert [gap.code for gap in _message_gaps(bundle)] == ["assistant_message_call_join_failed"]
    canonical = _build_trajectory_record(ROW, record)
    assert _ids(canonical.invocations[0]) == ["captured"]
    assert len(canonical.model_calls) == 1
    assert canonical.model_calls[0].client_assistant_message_id == "m0"
    assert all(not turn.model_calls for turn in canonical.turns)
    assert "agent_observation_join_failed" not in {gap.code for gap in canonical.gaps}
    if failure == "invalid_trajectory":
        assert "producer_trajectory_invalid" in {gap.code for gap in canonical.gaps}


@pytest.mark.parametrize("agent", ["opencode_agent", "opencode_sandboxed_agent"])
def test_harness_property_reaches_model_capture_and_links_message(tmp_path, parse, agent):
    from fastapi.testclient import TestClient
    from omegaconf import OmegaConf

    from nemo_gym.base_responses_api_model import BaseResponsesAPIModelConfig, SimpleResponsesAPIModel
    from nemo_gym.server_utils import ServerClient

    class Model(SimpleResponsesAPIModel):
        async def chat_completions(self, body):
            return {"choices": []}

        async def responses(self, body):
            return {"output": []}

    client = ServerClient(
        head_server_config={"host": "localhost", "port": 0},
        global_config_dict=OmegaConf.create(
            {
                "observability_enabled": True,
                "model_call_capture_dir": str(tmp_path / "capture"),
                "agent": {
                    "responses_api_agents": {
                        agent: {
                            "model_server": {"type": "responses_api_models", "name": "policy"},
                        }
                    }
                },
            }
        ),
    )
    model = Model(
        config=BaseResponsesAPIModelConfig(host="localhost", port=0, name="policy", entrypoint="app.py"),
        server_client=client,
    )
    app = model.setup_webserver()
    with TestClient(app) as client:
        assert (
            client.post(
                "/ng-rollout/0-0/v1/chat/completions",
                json={"model": "policy", "messages": []},
                headers={"X-Session-Id": "root", "X-OpenCode-Assistant-Message-Id": "m0"},
            ).status_code
            == 200
        )
    store = CaptureStore(tmp_path / "capture")
    [call] = read_model_call_records(store, "0-0")
    assert call.client_assistant_message_id == "m0"
    observations, trajectory = _observations(parse, _session_db(tmp_path, [_policy()]))
    _, bundle, joined = _merge(store, observations, trajectory)
    assert _ids(joined.turns[0]) == [call.model_call_id]
    assert not _message_gaps(bundle)
