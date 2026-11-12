# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stale multi-stage journal names the fingerprint components that differ (a diagnostic only)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Tuple

import orjson
import pytest

import nemo_gym.rollouts.rollout_collection as rollout_collection_module
import resources_servers.gdpval.multistage_orchestrator as multistage_module
from nemo_gym.artifacts import NG_FAILURE_CLASS_KEY
from nemo_gym.config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.rollouts import NG_ENVIRONMENT_SERVER_KEY
from resources_servers.gdpval.multistage_elo import StageSpec
from resources_servers.gdpval.multistage_orchestrator import (
    FINGERPRINT_COMPONENTS_KEY,
    MultiStageRunConfig,
    _pending_restart_stage,
    _prepare_resume,
    build_file_resume,
    compute_fingerprint,
    compute_fingerprint_with_components,
    describe_fingerprint_component_changes,
    journal_path_for,
    load_latest_attempt_dispositions,
    read_journal,
    read_journal_fingerprint_components,
    run_e2e_multistage,
)


# ``compute_fingerprint`` of ``_fixture()``, computed before the component digests existed.
PINNED_FINGERPRINT = "b6aa4f2214e9208ee727dfc35486e2a787887549a38a27b69ec09d9f62f7c804"  # pragma: allowlist secret

UNKNOWN = "components unknown (the journal predates component digests)"


def _fixture() -> Tuple[MultiStageRunConfig, Dict[str, float], Dict[str, Any], Dict[str, Any]]:
    """Inputs that reach every payload entry of the fingerprint."""
    cfg = MultiStageRunConfig(
        enabled=True,
        stages=[StageSpec(num_tasks=3), StageSpec(num_tasks=5, num_models=2, seed=11)],
        column=["occupation", "sector"],
        nested_tasks=True,
        seed=7,
        reuse_cached_deliverables=False,
        replace_transport_ineligible_tasks=True,
    )
    reference_elos = {"ref-b": 1200.0, "ref-a": 1000.0}
    distribution = {
        "grp-b": {"percentage": 0.25, "task_ids": ["t2"]},
        "grp-a": {"percentage": 0.75, "task_ids": ["t1", "t0"]},
    }
    rows = [
        {"_ng_task_index": index, "_ng_rollout_index": 0, "task_id": task_id}
        for index, task_id in enumerate(["t0", "t1", "t2"])
    ]
    run_config = {
        "agent_name": "example_agent",
        "input_jsonl_fpath": "inputs/tasks.jsonl",
        "limit": None,
        "num_repeats": 1,
        "num_repeats_add_seed": False,
        "responses_create_params": {"temperature": 0.0},
        "prompt_config": None,
        "skills": None,
    }
    runtime = {
        "policy_model": {
            "responses_api_models": {
                "vllm_model": {
                    "model": "example-policy",
                    "base_url": "http://policy.example.invalid:8000/v1",
                    "api_key": "dummy",
                    "host": "policy.example.invalid",
                    "port": 8000,
                }
            }
        },
        "judge_model": {"responses_api_models": {"openai_model": {"openai_model": "example-judge"}}},
        "gdpval_resources_server": {"resources_servers": {"gdpval": {"num_comparison_trials": 2}}},
        "operational_logging": {"level": "INFO"},
        "multistage": {"transport_assignment_repair": {"enabled": False, "max_rounds": 2}},
    }
    inputs = {"materialized_rows": rows, "rollout_collection_config": run_config, "resolved_global_config": runtime}
    return cfg, reference_elos, distribution, inputs


def _digests(mutate: Optional[Callable[[Dict[str, Any]], None]] = None) -> Tuple[str, Dict[str, str]]:
    cfg, reference_elos, distribution, inputs = _fixture()
    if mutate is not None:
        mutate(inputs)
    return compute_fingerprint_with_components(cfg, reference_elos, distribution, **inputs)


def _change_judge(inputs: Dict[str, Any]) -> None:
    inputs["resolved_global_config"]["judge_model"]["responses_api_models"]["openai_model"]["openai_model"] = (
        "example-judge-2"
    )


def _journal(tmp_path: Path, fingerprint: str, components: Optional[Dict[str, str]]) -> Tuple[Path, Path]:
    """A one-stage journal; without components its lines are exactly what the driver wrote before digests."""
    out = tmp_path / "rollouts.jsonl"
    out.write_text("")
    journal = journal_path_for(out)
    records = [
        {"stage_index": 0, "status": "planned", "reference_ids": ["ref-a"], "task_ids": ["t0"]},
        {"stage_index": 0, "status": "complete"},
    ]
    if components is None:
        journal.write_bytes(
            b"".join(orjson.dumps({**record, "fingerprint": fingerprint}) + b"\n" for record in records)
        )
    else:
        resume = build_file_resume(out, journal, fingerprint, fingerprint_components=components)
        resume.on_plan(0, records[0])
        resume.on_outcome(0, records[1])
    return out, journal


def _resume(out: Path, journal: Path, fingerprint: str, components: Dict[str, str]):
    return _prepare_resume(
        SimpleNamespace(resume_from_cache=True), out, journal, fingerprint, fingerprint_components=components
    )


def test_the_fingerprint_is_byte_identical_to_the_pinned_value() -> None:
    cfg, reference_elos, distribution, inputs = _fixture()

    assert compute_fingerprint(cfg, reference_elos, distribution, **inputs) == PINNED_FINGERPRINT
    assert compute_fingerprint_with_components(cfg, reference_elos, distribution, **inputs)[0] == PINNED_FINGERPRINT


def test_components_are_the_hashed_inputs_with_one_entry_per_server_block() -> None:
    _, components = _digests()

    assert sorted(components) == [
        "column",
        "distribution",
        "materialized_rows",
        "nested_tasks",
        "reference_elos",
        "replace_transport_ineligible_tasks",
        "reuse_cached_deliverables",
        "rollout_collection_config",
        "runtime_components.gdpval_resources_server",
        "runtime_components.judge_model",
        "runtime_components.policy_model",
        "seed",
        "stages",
        "transport_assignment_repair",
    ]
    assert components["seed"] == hashlib.sha256(orjson.dumps(7, option=orjson.OPT_SORT_KEYS)).hexdigest()

    def rebind_policy(inputs: Dict[str, Any]) -> None:
        server = inputs["resolved_global_config"]["policy_model"]["responses_api_models"]["vllm_model"]
        server.update(host="other.example.invalid", port=9000, base_url="http://other.example.invalid:9000/v1")

    # The digests hash the same payload: fields the fingerprint ignores change no digest either.
    assert _digests(rebind_policy) == _digests()


def test_one_changed_server_block_is_named_alone() -> None:
    old_fingerprint, old = _digests()
    new_fingerprint, new = _digests(_change_judge)

    assert old_fingerprint != new_fingerprint
    assert describe_fingerprint_component_changes(old, new) == "components differ: runtime_components.judge_model"


def test_added_removed_and_changed_components_are_listed_in_sorted_order() -> None:
    def change(inputs: Dict[str, Any]) -> None:
        runtime = inputs["resolved_global_config"]
        runtime["extra_model"] = {"responses_api_models": {"vllm_model": {"model": "example-extra"}}}
        del runtime["multistage"]
        runtime["policy_model"]["responses_api_models"]["vllm_model"]["model"] = "example-policy-2"
        inputs["materialized_rows"] = inputs["materialized_rows"][:2]

    _, old = _digests()
    _, new = _digests(change)

    assert describe_fingerprint_component_changes(old, new) == (
        "components differ: materialized_rows, runtime_components.extra_model, runtime_components.policy_model, "
        "transport_assignment_repair"
    )
    assert describe_fingerprint_component_changes(new, old) == describe_fingerprint_component_changes(old, new)
    assert describe_fingerprint_component_changes(None, new) == UNKNOWN


def test_a_stale_resume_names_the_changed_server_block(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    old_fingerprint, old = _digests()
    new_fingerprint, new = _digests(_change_judge)
    out, journal = _journal(tmp_path, old_fingerprint, old)

    resume = _resume(out, journal, new_fingerprint, new)

    assert (
        f"[multistage-elo] starting fresh: journal STALE (fingerprint {old_fingerprint} != {new_fingerprint}): "
        "components differ: runtime_components.judge_model\n"
    ) in capsys.readouterr().err
    # The verdict is unchanged: the journal is set aside and the run starts fresh.
    assert resume.plans == {} and resume.outcomes == {}
    assert len(list(tmp_path.glob(journal.name + ".stale.*"))) == 1


def test_the_same_config_names_nothing_and_resumes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    fingerprint, components = _digests()
    out, journal = _journal(tmp_path, fingerprint, components)

    resume = _resume(out, journal, fingerprint, components)

    err = capsys.readouterr().err
    assert "resuming multi-stage run from cache (fingerprint match)" in err
    assert "STALE" not in err and "components" not in err
    assert set(resume.plans) == {0} and set(resume.outcomes) == {0}


def test_a_journal_without_digests_resumes_on_a_matching_fingerprint(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    fingerprint, components = _digests()
    out, journal = _journal(tmp_path, fingerprint, None)
    assert read_journal_fingerprint_components(journal) is None

    resume = _resume(out, journal, fingerprint, components)

    assert "resuming multi-stage run from cache (fingerprint match)" in capsys.readouterr().err
    assert set(resume.plans) == {0} and set(resume.outcomes) == {0}


def test_a_stale_journal_without_digests_says_the_components_are_unknown(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    old_fingerprint, _ = _digests()
    new_fingerprint, new = _digests(_change_judge)
    out, journal = _journal(tmp_path, old_fingerprint, None)

    resume = _resume(out, journal, new_fingerprint, new)

    assert (
        f"[multistage-elo] starting fresh: journal STALE (fingerprint {old_fingerprint} != {new_fingerprint}): "
        f"{UNKNOWN}\n"
    ) in capsys.readouterr().err
    assert resume.outcomes == {}


def test_every_record_carries_the_digests_and_readers_that_predate_them_ignore_them(tmp_path: Path) -> None:
    fingerprint, components = _digests()
    out = tmp_path / "rollouts.jsonl"
    journal = journal_path_for(out)
    resume = build_file_resume(out, journal, fingerprint, fingerprint_components=components)
    resume.on_plan(0, {"stage_index": 0, "status": "planned", "reference_ids": ["ref-a"], "task_ids": ["t0"]})
    failed = {"stage_index": 0, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "task_id": "t0"}
    resume.on_rows(0, [{**failed, NG_FAILURE_CLASS_KEY: "agent_timeout", NG_ENVIRONMENT_SERVER_KEY: "gdpval"}])
    resume.on_restart(0)
    # A replayed plan brings the digests it was written with; the record keeps only this run's.
    replayed = {"stage_index": 0, "status": "planned", "reference_ids": ["ref-a"], "task_ids": ["t0"]}
    resume.on_plan(0, {**replayed, "fingerprint": "f" * 64, FINGERPRINT_COMPONENTS_KEY: {"seed": "0" * 64}})
    resume.on_outcome(0, {"stage_index": 0, "status": "complete"})

    records = [orjson.loads(line) for line in journal.read_bytes().splitlines()]
    assert [record["status"] for record in records] == [
        "planned",
        "attempt_dispositions",
        "restart_from_stage",
        "restart_cleanup_complete",
        "planned",
        "complete",
    ]
    assert all(record["fingerprint"] == fingerprint for record in records)
    assert all(record[FINGERPRINT_COMPONENTS_KEY] == components for record in records)
    assert read_journal_fingerprint_components(journal) == components

    # The journal readers below are the base commit's, unchanged. They read the journal exactly as the same
    # journal without the new key, which is how a run at the base commit would read it.
    stripped = tmp_path / "stripped" / journal.name
    stripped.parent.mkdir()
    stripped.write_bytes(
        b"".join(
            orjson.dumps({k: v for k, v in record.items() if k != FINGERPRINT_COMPONENTS_KEY}) + b"\n"
            for record in records
        )
    )

    def without_key(stage_records: Dict[int, dict]) -> Dict[int, dict]:
        return {
            index: {k: v for k, v in record.items() if k != FINGERPRINT_COMPONENTS_KEY}
            for index, record in stage_records.items()
        }

    plans, outcomes, journal_fingerprint = read_journal(journal)
    base_plans, base_outcomes, base_fingerprint = read_journal(stripped)
    assert (without_key(plans), without_key(outcomes), journal_fingerprint) == (
        base_plans,
        base_outcomes,
        base_fingerprint,
    )
    assert journal_fingerprint == fingerprint and set(plans) == {0} and set(outcomes) == {0}
    assert load_latest_attempt_dispositions(journal) == load_latest_attempt_dispositions(stripped)
    assert _pending_restart_stage(journal) is None and _pending_restart_stage(stripped) is None


async def test_the_driver_journals_the_digests_and_names_a_changed_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    reference_elos = {"ref-a": 1000.0, "ref-b": 1200.0}
    rows = [
        {
            TASK_INDEX_KEY_NAME: index,
            ROLLOUT_INDEX_KEY_NAME: 0,
            AGENT_REF_KEY_NAME: {"name": "example_agent"},
            "task_id": task_id,
            "responses_create_params": {"input": [], "metadata": {"task_id": task_id}},
        }
        for index, task_id in enumerate(["t0", "t1"])
    ]

    class FakeHelper:
        def _preprocess_rows_from_config(self, config):
            return rows

        def _run_examples_with_metadata(self, rows, **kwargs):
            async def done(row):
                ref = row["reference_ids"][0]
                per_reference = {ref: {"wins": 1, "losses": 0, "ties": 0, "reference_elo": reference_elos[ref]}}
                result = {"task_id": row["task_id"], "per_reference": per_reference}
                return rollout_collection_module._CompletedRollout(row=row, result=result, rollout_latency_ms=None)

            return [done(row) for row in rows]

        async def _call_aggregate_metrics(self, results, rows, output_fpath):
            return tmp_path / "aggregate.json"

    monkeypatch.setattr(rollout_collection_module, "RolloutCollectionHelper", FakeHelper)
    distribution = {"grp": {"percentage": 1.0, "task_ids": ["t0", "t1"]}}
    monkeypatch.setattr(multistage_module, "ensure_distribution", lambda *args, **kwargs: (distribution, None))
    config = rollout_collection_module.RolloutCollectionConfig(
        input_jsonl_fpath=str(tmp_path / "input.jsonl"),
        output_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
        require_complete=True,
        route_failures_to_sidecar=True,
    )
    references = {key: {"elo": elo} for key, elo in reference_elos.items()}
    global_config = {
        "multistage": {"enabled": True, "stages": ["2"], "seed": 0},
        "gdpval": {"resources_servers": {"gdpval": {"reference_models": references}}},
        "policy_model": {"responses_api_models": {"vllm_model": {"model": "example-policy"}}},
    }
    journal = journal_path_for(Path(config.output_jsonl_fpath))

    await run_e2e_multistage(config, global_config)
    first = [orjson.loads(line) for line in journal.read_bytes().splitlines()]
    assert first and all(FINGERPRINT_COMPONENTS_KEY in record for record in first)
    assert {"runtime_components.gdpval", "runtime_components.policy_model"} <= set(
        first[0][FINGERPRINT_COMPONENTS_KEY]
    )

    global_config["policy_model"]["responses_api_models"]["vllm_model"]["model"] = "example-policy-2"
    config.resume_from_cache = True
    capsys.readouterr()
    await run_e2e_multistage(config, global_config)

    second = [orjson.loads(line) for line in journal.read_bytes().splitlines()]
    old_fingerprint, new_fingerprint = first[-1]["fingerprint"], second[-1]["fingerprint"]
    assert old_fingerprint != new_fingerprint
    assert all(record["fingerprint"] == new_fingerprint for record in second)
    assert (
        f"[multistage-elo] starting fresh: journal STALE (fingerprint {old_fingerprint} != {new_fingerprint}): "
        "components differ: runtime_components.policy_model\n"
    ) in capsys.readouterr().err


def _persist_dir(path: str) -> Callable[[Dict[str, Any]], None]:
    def mutate(inputs: Dict[str, Any]) -> None:
        inputs["resolved_global_config"]["gdpval_resources_server"]["resources_servers"]["gdpval"][
            "persist_deliverables_dir"
        ] = path

    return mutate


def _two_names(tmp_path: Path) -> Tuple[Path, Path]:
    real = tmp_path / "physical" / "run" / "bench"
    real.mkdir(parents=True)
    (tmp_path / "linked").symlink_to(tmp_path / "physical", target_is_directory=True)
    return real, tmp_path / "linked" / "run" / "bench"


def test_two_names_of_one_directory_give_one_fingerprint(tmp_path: Path) -> None:
    # A run directory reached through a symlinked mount in one segment and its physical path in the next.
    real, linked = _two_names(tmp_path)

    assert _digests(_persist_dir(str(real))) == _digests(_persist_dir(str(linked)))


def test_a_different_directory_or_a_content_change_still_differs(tmp_path: Path) -> None:
    real, _ = _two_names(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    fingerprint, components = _digests(_persist_dir(str(real)))

    other_fingerprint, other_components = _digests(_persist_dir(str(other)))
    assert other_fingerprint != fingerprint
    assert [k for k in components if components[k] != other_components[k]] == [
        "runtime_components.gdpval_resources_server"
    ]
    assert _digests(_change_judge)[0] != _digests()[0]


def test_a_journal_written_with_the_raw_path_resumes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    real, linked = _two_names(tmp_path)
    cfg, reference_elos, distribution, inputs = _fixture()
    _persist_dir(str(linked))(inputs)
    legacy = compute_fingerprint(cfg, reference_elos, distribution, **inputs, canonical_paths=False)
    _persist_dir(str(real))(inputs)
    fingerprint, components = compute_fingerprint_with_components(cfg, reference_elos, distribution, **inputs)
    assert legacy != fingerprint
    run = tmp_path / "journal"
    run.mkdir()
    out, journal = _journal(run, legacy, None)

    resume = _prepare_resume(
        SimpleNamespace(resume_from_cache=True),
        out,
        journal,
        fingerprint,
        fingerprint_components=components,
        accepted_fingerprints=(legacy,),
    )

    assert "resuming multi-stage run from cache (fingerprint match)" in capsys.readouterr().err
    assert set(resume.plans) == {0}


def test_a_resume_through_the_other_name_of_the_run_directory_replays_the_journal(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    # One segment sees the physical run directory as its working directory, the next (a requeue on another node or
    # a resubmit) sees the symlinked one: the journal written by the first must be replayed by the second.
    real, linked = _two_names(tmp_path)
    cfg, reference_elos, distribution, inputs = _fixture()
    _persist_dir(str(real))(inputs)
    first, first_components = compute_fingerprint_with_components(cfg, reference_elos, distribution, **inputs)
    run = tmp_path / "journal"
    run.mkdir()
    out, journal = _journal(run, first, first_components)

    _persist_dir(str(linked))(inputs)
    second, second_components = compute_fingerprint_with_components(cfg, reference_elos, distribution, **inputs)
    resume = _resume(out, journal, second, second_components)

    assert second == first
    err = capsys.readouterr().err
    assert "resuming multi-stage run from cache (fingerprint match)" in err and "STALE" not in err
    assert set(resume.plans) == {0} and set(resume.outcomes) == {0}


def _reference_and_deliverables(reference: str, deliverables: str) -> Callable[[Dict[str, Any]], None]:
    def mutate(inputs: Dict[str, Any]) -> None:
        gdpval = inputs["resolved_global_config"]["gdpval_resources_server"]["resources_servers"]["gdpval"]
        gdpval["reference_deliverables_dir"] = reference
        gdpval["persist_deliverables_dir"] = deliverables

    return mutate


def test_a_deliverables_directory_created_between_segments_keeps_the_fingerprint(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    # Both directories are named through a symlinked mount; the deliverables one is created by the first segment.
    _, linked = _two_names(tmp_path)
    reference = linked / "reference"
    reference.mkdir()
    deliverables = linked / "deliverables"
    cfg, reference_elos, distribution, inputs = _fixture()
    _reference_and_deliverables(str(reference), str(deliverables))(inputs)
    first, first_components = compute_fingerprint_with_components(cfg, reference_elos, distribution, **inputs)
    run = tmp_path / "journal"
    run.mkdir()
    out, journal = _journal(run, first, first_components)

    deliverables.mkdir()
    second, second_components = compute_fingerprint_with_components(cfg, reference_elos, distribution, **inputs)
    legacy = compute_fingerprint(cfg, reference_elos, distribution, **inputs, canonical_paths=False)
    resume = _prepare_resume(
        SimpleNamespace(resume_from_cache=True),
        out,
        journal,
        second,
        fingerprint_components=second_components,
        accepted_fingerprints=(legacy,),
    )

    assert second == first
    err = capsys.readouterr().err
    assert "resuming multi-stage run from cache (fingerprint match)" in err and "STALE" not in err
    assert set(resume.plans) == {0} and set(resume.outcomes) == {0}


def test_a_missing_path_under_a_symlinked_mount_is_named_by_its_real_location(tmp_path: Path) -> None:
    real, linked = _two_names(tmp_path)

    assert _digests(_persist_dir(str(real / "not-yet"))) == _digests(_persist_dir(str(linked / "not-yet")))
