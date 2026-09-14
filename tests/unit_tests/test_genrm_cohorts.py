# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from copy import deepcopy

import pytest
from omegaconf import OmegaConf

from nemo_gym.config_types import ConfigError
from nemo_gym.genrm_cohorts import collection_admission, prepare_genrm_collection, reject_genrm_reverification


def configuration(size=4):
    return {
        "judge": {"resources_servers": {"genrm_compare": {"num_rollouts_per_prompt": size}}},
        **{
            name: {"responses_api_agents": {"simple_agent": {"resources_server": {"name": "judge"}}}}
            for name in ("a", "b")
        },
    }


def rows(count=8, agent="a", offset=0):
    return [{"_ng_task_index": 0, "_ng_rollout_index": i + offset, "agent_ref": {"name": agent}} for i in range(count)]


def test_fanout_and_repeats_make_separate_complete_groups():
    inputs = rows() + rows(agent="b", offset=8)
    assert prepare_genrm_collection(inputs, configuration(), 4)
    assert [r["_ng_rollout_index"] for r in inputs] == list(range(16))
    assert [r["_ng_group_member_index"] for r in inputs] == list(range(4)) * 4
    assert len({r["_ng_group_id"] for r in inputs}) == 4
    assert all(r["_ng_cohort_failure_mode"] == "row" for r in inputs)
    fresh = rows()
    prepare_genrm_collection(fresh, configuration(), 4)
    assert {r["_ng_group_id"] for r in fresh}.isdisjoint(r["_ng_group_id"] for r in inputs)


@pytest.mark.parametrize("count, concurrency", [(3, 4), (4, 3)])
def test_incomplete_or_under_concurrent_groups_fail_before_mutation(count, concurrency):
    inputs = rows(count)
    original = deepcopy(inputs)
    with pytest.raises(ConfigError):
        prepare_genrm_collection(inputs, configuration(), concurrency)
    assert inputs == original


def test_resume_advances_complete_group_together():
    inputs = rows(4)
    prepare_genrm_collection(inputs, configuration(), 4)
    ids = [r["_ng_group_id"] for r in inputs]
    # Per-rollout attempts differ; these must not define the group attempt.
    for i, row in enumerate(inputs):
        row["_ng_attempt_index"] = i
    prepare_genrm_collection(inputs, configuration(), 4, resume=True)
    assert [r["_ng_group_id"] for r in inputs] == ids
    assert [r["_ng_group_attempt"] for r in inputs] == [1] * 4


def test_resume_rejects_partial_group_without_modifying_remaining_rows():
    inputs = rows(4)
    prepare_genrm_collection(inputs, configuration(), 4)
    pending = inputs[2:]
    original = deepcopy(pending)
    with pytest.raises(ConfigError, match="Partial-group resume"):
        prepare_genrm_collection(pending, configuration(), 4, resume=True)
    assert pending == original


def test_resume_rejects_legacy_cache():
    with pytest.raises(ConfigError, match="lacks group identities"):
        prepare_genrm_collection(rows(4), configuration(), 4, resume=True)


@pytest.mark.parametrize("force, judge_failed_only", [(True, False), (False, True), (True, True)])
async def test_reverification_rejects_before_creating_output(tmp_path, monkeypatch, force, judge_failed_only):
    import nemo_gym.rollout_reverification as module

    monkeypatch.setattr(module, "get_global_config_dict", configuration)
    config = module.RolloutReverificationConfig(
        materialized_inputs_jsonl_fpath=str(tmp_path / "input.jsonl"),
        rollouts_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
        output_jsonl_fpath=str(tmp_path / "output.jsonl"),
        force=force,
        judge_failed_only=judge_failed_only,
    )
    with pytest.raises(ConfigError, match="GenRM cohort reverification"):
        await module.RolloutReverificationHelper().run_from_config(config)
    assert not list(tmp_path.iterdir())


def test_other_environments_are_unaffected():
    inputs = rows(3)
    assert not prepare_genrm_collection(inputs, {}, 1)
    reject_genrm_reverification({})
    reject_genrm_reverification(configuration(size=1))


async def test_interleaved_groups_fill_with_concurrency_equal_to_group_size():
    inputs = rows(8)
    prepare_genrm_collection(inputs, configuration(), 4)
    semaphore = asyncio.Semaphore(4)
    admission = collection_admission(inputs, semaphore)
    arrived = {}
    active = 0
    maximum = 0

    async def run(row):
        nonlocal active, maximum
        async with admission(row):
            active += 1
            maximum = max(maximum, active)
            group = row["_ng_group_id"]
            count, event = arrived.setdefault(group, [0, asyncio.Event()])
            arrived[group][0] = count + 1
            if count + 1 == 4:
                event.set()
            await event.wait()
            active -= 1

    order = [0, 4, 1, 5, 2, 6, 3, 7]
    await asyncio.wait_for(asyncio.gather(*(run(inputs[i]) for i in order)), timeout=1)
    assert maximum == 4 and semaphore._value == 4


async def test_cancelled_pending_member_releases_reserved_permit():
    inputs = rows(4)
    prepare_genrm_collection(inputs, configuration(), 4)
    semaphore = asyncio.Semaphore(4)
    await semaphore.acquire()  # an ordinary rollout is already active
    admission = collection_admission(inputs, semaphore)

    async def run(row):
        async with admission(row):
            await asyncio.sleep(0)

    leader = asyncio.create_task(run(inputs[0]))
    await asyncio.sleep(0)
    follower = asyncio.create_task(run(inputs[1]))
    await asyncio.sleep(0)
    follower.cancel()
    with pytest.raises(asyncio.CancelledError):
        await follower
    semaphore.release()
    await asyncio.gather(leader, run(inputs[2]), run(inputs[3]))
    await asyncio.sleep(0)
    assert semaphore._value == 4


async def test_file_collector_saves_failed_answers_and_resumes_only_complete_group(tmp_path, monkeypatch):
    import json as jsonlib
    from unittest.mock import AsyncMock

    import nemo_gym.rollout_collection as collection
    from nemo_gym.reward_profile import compute_aggregate_metrics
    from tests.unit_tests.test_rollout_collection import FakeResponse, install_fake_server_client

    cfg = configuration()
    monkeypatch.setattr(collection, "get_global_config_dict", lambda: cfg)
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "4")
    source = tmp_path / "input.jsonl"
    source.write_text(jsonlib.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "a"}}) + "\n")
    output = tmp_path / "output.jsonl"
    attempts = []
    aggregated = []

    async def post(server_name, url_path, json, **kwargs):
        if url_path == "/run":
            row = deepcopy(json)
            attempts.append(row)
            assert row["_ng_cohort_failure_mode"] == "row"
            persisted = [jsonlib.loads(line) for line in config.materialized_jsonl_fpath.read_text().splitlines()]
            match = next(r for r in persisted if r["_ng_rollout_index"] == row["_ng_rollout_index"])
            assert match["_ng_group_attempt"] == row["_ng_group_attempt"]
            failed = server_name == "a" and row["_ng_group_attempt"] == 0
            result = row | {
                "response": {
                    "id": f"{server_name}-{row['_ng_group_attempt']}-{row['_ng_rollout_index']}",
                    "usage": {},
                },
                "reward": 0.0 if failed else 1.0,
            }
            if failed:
                result |= {"_ng_failure_class": "judge_failed", "_ng_failure_kind": "cohort_incomplete"}
            return FakeResponse(200, result)
        assert url_path == "/aggregate_metrics"
        aggregated.extend(json.verify_responses)
        return FakeResponse(200, compute_aggregate_metrics(json.verify_responses).model_dump())

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    client.global_config_dict = OmegaConf.create(cfg)
    config = collection.RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(output),
        num_repeats=4,
        fan_out={"a": ["a", "b"]},
        num_samples_in_parallel=4,
        disable_health_check=True,
        upload_rollouts=False,
    )
    helper = collection.RolloutCollectionHelper()
    await helper.run_from_config(config)
    successes = [jsonlib.loads(line) for line in output.read_text().splitlines()]
    failures = [jsonlib.loads(line) for line in collection.failures_path_for(output).read_text().splitlines()]
    assert len(successes) == len(failures) == 4
    assert all(row["agent_ref"]["name"] == "b" for row in successes)
    assert all(row["response"]["id"].startswith("a-0-") for row in failures)
    assert len(aggregated) == 4 and all(row["reward"] == 1.0 for row in aggregated)
    assert not config.route_failures_to_sidecar  # tagged judge rows route without the generic opt-in
    old_id = failures[0]["_ng_group_id"]
    attempts.clear()
    aggregated.clear()
    config.resume_from_cache = True
    await helper.run_from_config(config)
    resumed = [jsonlib.loads(line) for line in output.read_text().splitlines()]
    assert len(resumed) == 8 and len(attempts) == 4
    assert all(row["_ng_group_attempt"] == 1 and row["_ng_group_id"] == old_id for row in attempts)
    assert [row for row in resumed if row["agent_ref"]["name"] == "b"] == successes
    assert len(aggregated) == 8 and all(row["reward"] == 1.0 for row in aggregated)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"_ng_group_attempt": -1}, "nonnegative integer"),
        ({"_ng_group_attempt": True}, "nonnegative integer"),
        ({"_ng_group_attempt": 2}, "nonnegative integer"),
        ({"_ng_group_member_index": 1}, "every persisted member"),
    ],
)
def test_resume_rejects_corrupted_identity_without_mutation(changes, message):
    inputs = rows(4)
    prepare_genrm_collection(inputs, configuration(), 4)
    inputs[0].update(changes)
    original = deepcopy(inputs)
    with pytest.raises(ConfigError, match=message):
        prepare_genrm_collection(inputs, configuration(), 4, resume=True)
    assert inputs == original


@pytest.mark.parametrize("identity", ["", "x" * 257, 123])
def test_invalid_explicit_group_identity_is_rejected(identity):
    inputs = [row | {"_ng_group_id": identity} for row in rows(4)]
    with pytest.raises(ConfigError, match="nonempty strings"):
        prepare_genrm_collection(inputs, configuration(), 4)


def test_group_identity_cannot_be_shared_across_agents():
    inputs = [row | {"_ng_group_id": "shared"} for row in rows(4) + rows(4, agent="b", offset=4)]
    with pytest.raises(ConfigError, match="reused across tasks or agents"):
        prepare_genrm_collection(inputs, configuration(), 4)
    assert not any("_ng_group_member_index" in row for row in inputs)


async def test_cancelled_reservation_leader_releases_permits_and_settles_followers():
    inputs = rows(4)
    cfg = configuration() | {"global_option": True}
    prepare_genrm_collection(inputs, cfg, 4)
    semaphore = asyncio.Semaphore(4)
    await semaphore.acquire()
    admission = collection_admission(inputs, semaphore)

    async def run(row):
        async with admission(row):
            pytest.fail("Cancelled group must not dispatch")

    leader = asyncio.create_task(run(inputs[0]))
    await asyncio.sleep(0)
    follower = asyncio.create_task(run(inputs[1]))
    await asyncio.sleep(0)
    leader.cancel()
    results = await asyncio.wait_for(asyncio.gather(leader, follower, return_exceptions=True), 1)
    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    assert semaphore._value == 3
    semaphore.release()
    # A non-cohort request can still use all the capacity after cancellation.
    async with admission({}):
        assert semaphore._value == 3
    assert semaphore._value == 4
