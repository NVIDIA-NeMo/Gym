# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control-plane tests over the scripted fake provider: nothing is executed or compared on the host.

Result files use the real runner envelope, the codec's wire envelope and the comparator's verdict shape, written by
hand: no value is encoded, decoded or compared here.
"""

import asyncio
import json
import logging
from itertools import permutations
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from nemo_gym.core import failure_kinds
from resources_servers.critpt_custom_grader import execution, reference_cache, runner, sandbox_domain, task_data
from resources_servers.critpt_custom_grader.execution import (
    DOMAINS,
    ExecutionPolicy,
    Grader,
    _correlate_result,
    _parse_result,
    _reported_exit,
    _Unscorable,
    build_compare_job,
    build_run_job,
    validate_candidate_source,
)
from resources_servers.critpt_custom_grader.reference_cache import CachedReference, ReferenceCache
from resources_servers.critpt_custom_grader.runner import MAX_CODE_LENGTH, MAX_NONCE_LENGTH, RESULT_BYTES_LIMIT
from resources_servers.critpt_custom_grader.task_data import (
    MAX_CASES,
    MAX_JSON_NODES,
    MAX_TEXT_BYTES,
    MAX_WIRE_BYTES,
    WIRE_FORMAT,
    TaskData,
)
from resources_servers.critpt_custom_grader.tests.support import (
    CANDIDATE,
    ENCODING_ERROR,
    EXCEPTION,
    MISSING_FILE,
    VALUE,
    ScriptedProvider,
    compare_result,
    exited,
    happy_script,
    make_grader,
    make_policy,
    make_spec,
    make_task,
    run_result,
    runner_execs,
    value_outcome,
    verdict,
    wire,
)


LEGACY_VALUE = {"kind": "value", "value": {"format": WIRE_FORMAT, "type": "int", "value": "2"}}


def nested(levels):
    node = ["int", "1"]
    for _ in range(levels):
        node = ["list", [node]]
    return node


async def hang():
    await asyncio.sleep(10)


def entered_hang():
    """A ``hang`` that sets the returned event on entry, so a test waits for the event instead of a sleep margin."""
    entered = asyncio.Event()

    async def hanging():
        entered.set()
        await asyncio.sleep(10)

    return hanging, entered


async def until_entered(event):
    """Wait until a scripted step is entered. Ten seconds without it fails the test instead of hanging it."""
    await asyncio.wait_for(event.wait(), timeout=10)


def assert_job_cleaned(result, provider, *created):
    """Every created domain was stopped exactly once and the result's cleanup says so."""
    assert result.cleanup == {domain: "deleted" if domain in created else "not_created" for domain in DOMAINS}
    assert sorted(spec.metadata["ng-grader-domain"] for spec in provider.created) == sorted(created)
    assert sorted(closed_domains(provider)) == sorted(created)
    for domain in created:
        assert len(runner_execs(provider, domain)) <= (2 if domain == "comparator" else 1)  # no resubmission
    assert not stray_tasks()


async def cancel_self():
    asyncio.current_task().cancel()
    await asyncio.sleep(10)


def stray_tasks():
    return [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()]


def created_domains(provider):
    return [spec.metadata["ng-grader-domain"] for spec in provider.created]


def closed_domains(provider):
    """The domain of every stop the provider completed, one entry per stop."""
    return [provider.sandboxes[sandbox_id]["domain"] for sandbox_id in provider.closed]


def domain_files(provider, domain):
    """What the grader uploaded to that domain's sandbox, by remote path (one sandbox per domain in these tests)."""
    (box,) = [box for box in provider.sandboxes.values() if box["domain"] == domain]
    return box["files"]


def domain_job(provider, domain):
    """The job the domain's runner read last: ``job.json`` in the working directory its runner exec used."""
    (cwd,) = {entry["cwd"] for entry in runner_execs(provider, domain)}
    return json.loads(domain_files(provider, domain)[f"{cwd}/job.json"])


def result_reads(provider, domain):
    """Result-file reads in one domain. Each starts with the size probe, which runs only behind a result exit."""
    return sum(
        1 for entry in provider.exec_calls if entry["domain"] == domain and entry["command"].startswith("test -f")
    )


def fail_downloads(provider, domain, outcome):
    """Raise or await ``outcome`` in every result download of ``domain``, after the size probe found the file."""
    download = provider.download_file

    async def download_file(handle, source_path, target_path):
        if provider.sandboxes[handle.sandbox_id]["domain"] == domain:
            if isinstance(outcome, BaseException):
                raise outcome
            await outcome()
        await download(handle, source_path, target_path)

    provider.download_file = download_file


# --- policy ------------------------------------------------------------------------------------- #


def test_policy_defaults_and_derived_deadlines():
    policy = ExecutionPolicy()
    assert policy.exec_user is None and policy.workdir == "/work" and policy.max_concurrent_jobs == 4
    assert policy.compare_deadline_s(1) == 50 and policy.compare_deadline_s(64) == 600  # cases*20+30, capped
    assert policy.minimum_job_timeout_s() > 2 * policy.suite_timeout_s
    assert policy.effective_job_timeout_s() == policy.minimum_job_timeout_s()


@pytest.mark.parametrize(
    "fields",
    [
        {"exec_user": "root"},
        {"exec_user": 0},
        {"exec_user": -1},
        {"exec_user": 2**31},
        {"interpreter": "python3"},
        {"workdir": 'a"b'},
        {"exec_timeout_margin_s": 1.0},
        {"cpu_time_limit_s": 5000},
        {"snapshot": "x"},
        {"os_user": "grader"},
        {"journal_dir": "/j"},
    ],
)
def test_policy_rejects_root_unsafe_paths_short_margins_and_removed_fields(fields):
    with pytest.raises((ValidationError, ValueError)):
        ExecutionPolicy(**fields)


@pytest.mark.parametrize("uid", [1, 1000, 2**31 - 1])
def test_policy_accepts_a_non_root_uid_up_to_the_largest_32_bit_one(uid):
    assert ExecutionPolicy(exec_user=uid).exec_user == uid


def test_compare_job_carries_the_per_case_limit():
    job = build_compare_job(make_task(), "candidate", [VALUE, VALUE], "n" * 16, make_policy(), per_case_s=20)
    assert job["limits"]["per_case_s"] == 20 and job["limits"]["suite_deadline_s"] == make_policy().compare_deadline_s(
        2
    )
    assert runner.validate_job(job, "compare") is None
    run_job = build_run_job(make_task(), "f", "n" * 16, make_policy())
    assert "per_case_s" not in run_job["limits"] and runner.validate_job(run_job, "run") is None
    assert runner.validate_job(dict(run_job, limits=dict(run_job["limits"], per_case_s=5)), "run") == "invalid_job"


def test_task_fixture_validates_without_legacy_conversion(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("legacy conversion is not part of control-plane tests"))
    monkeypatch.setattr(task_data, "_legacy_wire", forbidden)
    monkeypatch.setattr(runner, "_input_literal", forbidden)
    task = make_task()
    assert len(task.test_cases) == 2
    assert task.test_cases[1].expected.model_dump(mode="json") == value_outcome(["int", "3"])
    forbidden.assert_not_called()


@pytest.mark.parametrize("field", ["suite_timeout_s", "create_timeout_s", "cleanup_timeout_s", "memory_limit_mib"])
def test_policy_limits_must_be_positive(field):
    with pytest.raises(ValidationError):
        make_policy(**{field: 0})


@pytest.mark.parametrize("value", ["grader", "_svc-user_2"])
def test_policy_accepts_well_formed_identifiers(value):
    assert make_policy(exec_user=value).exec_user == value


@pytest.mark.parametrize("value", ["!grader", "Grader", "grader!", "grader\n"])
def test_policy_identifiers_are_anchored_at_both_ends(value):
    # Without a start anchor a search would accept the bad prefix. ``$`` alone would accept the trailing newline.
    with pytest.raises(ValidationError):
        make_policy(exec_user=value)


def test_policy_exec_user_cannot_start_with_a_digit():
    with pytest.raises(ValidationError):
        make_policy(exec_user="1grader")


def test_policy_rejects_task_selected_runtime_fields():
    for field in ("image", "mounts", "provider_options", "timeouts", "network_allow_list"):
        with pytest.raises(ValidationError):
            make_policy(**{field: "x"})
    with pytest.raises(ValidationError):
        make_policy(interpreter='/usr/bin/python3"; rm -rf /')


def test_suite_timeout_default_preserves_documented_budget():
    assert make_policy().suite_timeout_s == 1800


def test_job_deadline_budget_counts_every_awaited_bound():
    fields = {
        "suite_timeout_s": 100,
        "compare_timeout_s": 10,
        # At the runner cleanup-budget floor (CLEANUP_BUDGET_S = 15 s), so two margins are 30 s.
        "exec_timeout_margin_s": 15.0,
        "cpu_time_limit_s": 50,
        "create_timeout_s": 2.0,
        "transfer_timeout_s": 3.0,
        "cleanup_timeout_s": 4.0,
    }
    policy = make_policy(**fields)
    # Two runs and two comparisons at deadline plus two margins, three creates, the 27 file transfers (uploads, size
    # probes and downloads) at transfer plus grace, and three stops.
    expected = 2 * (100 + 30) + 2 * (10 + 30) + 3 * 2 + 27 * (3 + 5) + 3 * 4
    assert execution._JOB_TRANSFERS == 27
    assert policy.minimum_job_timeout_s() == expected == 574
    assert policy.effective_job_timeout_s() == expected
    assert make_policy(job_timeout_s=float(expected), **fields).effective_job_timeout_s() == expected
    with pytest.raises(ValidationError):
        make_policy(job_timeout_s=expected - 1.0, **fields)


# --- job flow ----------------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_happy_path_passes_and_cleans_all_three_domains():
    grader, provider = make_grader(happy_script())
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("passed", 1.0, "passed")
    assert result.cases_equal == 2 and result.case_count == 2
    assert_job_cleaned(result, provider, "reference", "comparator", "candidate")
    assert set(result.sandboxes) == set(DOMAINS) and result.sandboxes["candidate"]["outcome"] == "completed"
    # the three domains used distinct per-domain workdirs under the spec's base
    workdirs = {call["cwd"] for call in provider.exec_calls if "runner.py" in call["command"]}
    assert len(workdirs) == 3 and all(w.startswith("/work/") for w in workdirs)
    # every runner exec ran as the exec user with an explicit timeout
    assert all(
        call["user"] == "grader" and call["timeout_s"] >= 60
        for call in provider.exec_calls
        if "runner.py" in call["command"]
    )


@pytest.mark.asyncio
async def test_two_concurrent_jobs_have_distinct_workdirs_and_both_complete():
    grader, provider = make_grader(
        {
            **happy_script(),
            ("result", "reference"): [run_result("completed", [VALUE, VALUE])] * 2,
            ("result", "comparator"): [
                compare_result(["equal", "equal"], side="reference"),
                compare_result(["equal", "equal"]),
            ]
            * 2,
            ("result", "candidate"): [run_result("completed", [VALUE, VALUE])] * 2,
        },
        max_concurrent_jobs=2,
    )
    results = await asyncio.gather(
        grader.grade(make_task(), CANDIDATE, job_id="aaaaaaaaaaaa"),
        grader.grade(make_task(), CANDIDATE, job_id="bbbbbbbbbbbb"),
    )
    assert [r.category for r in results] == ["passed", "passed"]
    cwds = {call["cwd"] for call in provider.exec_calls if "runner.py" in call["command"]}
    assert {"/work/aaaaaaaaaaaa-candidate", "/work/bbbbbbbbbbbb-candidate"} <= cwds


@pytest.mark.asyncio
async def test_semaphore_bounds_concurrency():
    gate, entered = asyncio.Event(), asyncio.Event()

    async def wait_for_gate():
        entered.set()
        await gate.wait()
        return exited(0)

    script = {**happy_script(), ("exec", "reference"): [wait_for_gate, wait_for_gate]}
    grader, provider = make_grader(script, max_concurrent_jobs=1)
    first = asyncio.create_task(grader.grade(make_task(), CANDIDATE))
    second = asyncio.create_task(grader.grade(make_task(), CANDIDATE))
    await until_entered(entered)  # the first job holds the semaphore inside its reference run
    assert len(provider.created) == 1  # the second job waits for the semaphore
    gate.set()
    await asyncio.gather(first, second)


@pytest.mark.asyncio
async def test_draining_grader_refuses_new_jobs_and_stops_live_domains():
    candidate_run, entered = entered_hang()
    grader, provider = make_grader(
        {**happy_script(), ("exec", "candidate"): [candidate_run]}, exec_timeout_margin_s=20.0
    )
    job = asyncio.create_task(grader.grade(make_task(), CANDIDATE))
    await until_entered(entered)  # the job is inside the candidate run, so every domain exists
    await grader.shutdown()
    # shutdown itself stopped the job's live sandboxes, while the candidate exec still hangs
    assert sorted(closed_domains(provider)) == sorted(DOMAINS)
    assert not any(box["live"] for box in provider.sandboxes.values()) and not job.done()
    result = await grader.grade(make_task(), CANDIDATE)
    assert result.category == "server_draining" and result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE
    job.cancel()
    with pytest.raises(asyncio.CancelledError):
        await job
    assert len(provider.closed) == 3  # shutdown stopped the live sandboxes of the cancelled job


def test_outcome_categories_separate_observations_from_scored_contracts():
    assert set(execution.FAILED_CATEGORIES) == {
        "candidate_mismatch",
        "candidate_source_limit",
        "candidate_invalid_output",
        "candidate_source_error",
        "candidate_timeout",
        "candidate_exception",
        "candidate_encoding_error",
        "candidate_worker_abort",
    }
    # The candidate-attributable faults observed after a completed preflight are scored, not unscorable.
    assert set(execution.CANDIDATE_FAULT_CATEGORIES) == {
        "candidate_source_error",
        "candidate_timeout",
        "candidate_exception",
        "candidate_encoding_error",
        "candidate_worker_abort",
    }
    assert set(execution.CANDIDATE_FAULT_CATEGORIES) <= set(execution.FAILED_CATEGORIES)
    assert set(execution.FAILED_CATEGORIES).isdisjoint(execution.UNSCORABLE_CATEGORIES)
    terminal = {category for category, (_, terminal) in execution.UNSCORABLE_CATEGORIES.items() if terminal}
    assert terminal == {"task_invalid", "reference_mismatch"}
    for category in terminal:
        assert execution.UNSCORABLE_CATEGORIES[category] == (failure_kinds.VERIFIER_ERROR, True)
    for category in execution.UNSCORABLE_CATEGORIES.keys() - terminal:
        assert execution.UNSCORABLE_CATEGORIES[category] == (failure_kinds.PROVIDER_UNAVAILABLE, False)


async def test_passing_candidate_scores_one_and_cleans_all_domains():
    grader, provider = make_grader(happy_script())
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("passed", 1.0, "passed")
    assert result.cases_attempted == result.cases_equal == 2
    assert result.cleanup == {"reference": "deleted", "comparator": "deleted", "candidate": "deleted"}
    assert not any(box["live"] for box in provider.sandboxes.values())
    # Each domain is stopped exactly once: the final sweep is a no-op after a mid-job stop.
    assert sorted(provider.closed) == sorted(provider.sandboxes) and len(provider.closed) == 3


async def test_transfer_budget_matches_the_transfers_a_job_performs():
    grader, provider = make_grader(happy_script())
    provider.upload_file = Mock(wraps=provider.upload_file)
    provider.download_file = Mock(wraps=provider.download_file)
    assert (await grader.grade(make_task(), CANDIDATE)).reward == 1.0
    per_domain = {domain: result_reads(provider, domain) for domain in DOMAINS}  # the size probes
    for transfer in provider.upload_file.call_args_list + provider.download_file.call_args_list:
        per_domain[provider.sandboxes[transfer.args[0].sandbox_id]["domain"]] += 1
    assert per_domain == {"reference": 8, "comparator": 11, "candidate": 8}
    assert sum(per_domain.values()) == execution._JOB_TRANSFERS


@pytest.mark.parametrize("code", runner.SOURCE_ERROR_CODES)
async def test_preflight_failure_never_allocates_candidate(code):
    script = happy_script()
    script[("result", "reference")] = run_result("source_error", [], code=code)
    script[("exec", "reference")] = exited(runner.EXIT_SOURCE_ERROR)
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "reference_error")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cases_attempted == result.cases_equal == 0 and result.first_failed_case is None
    assert_job_cleaned(result, provider, "reference")


@pytest.mark.parametrize("observation", [EXCEPTION, ENCODING_ERROR], ids=["exception", "encoding_error"])
@pytest.mark.parametrize("statuses", [["equal", "equal"], ["invalid_reference", "equal"]])
async def test_reference_non_value_observations_never_reach_preflight(observation, statuses):
    script = happy_script()
    script[("result", "reference")] = run_result("completed", [VALUE, observation])
    preflight = compare_result(statuses, side="reference")
    script[("result", "comparator")] = [preflight]
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "reference_error")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cases_attempted == result.cases_equal == 0 and result.first_failed_case is None
    assert script[("result", "comparator")] == [preflight]  # neither a pass nor an earlier defect may be used
    assert_job_cleaned(result, provider, "reference")


async def test_reference_mismatch_is_unscorable_not_candidate_fault():
    script = happy_script()
    script[("result", "comparator")] = [compare_result(["equal", "invalid_reference"], side="reference")]
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert result.category == "reference_mismatch" and result.reward is None
    assert result.failure_class == failure_kinds.VERIFIER_ERROR and result.terminal is True
    assert_job_cleaned(result, provider, "reference", "comparator")


@pytest.mark.parametrize(
    ("status", "category", "terminal"),
    [
        ("invalid_reference", "reference_mismatch", True),
        ("invalid_expected", "task_invalid", True),
        ("uncertain", "comparator_uncertain", False),
        ("mismatch", "result_invalid", False),  # candidate-role verdicts cannot judge the reference
        ("invalid_candidate", "result_invalid", False),
    ],
)
async def test_reference_preflight_maps_every_comparator_verdict(status, category, terminal):
    script = happy_script()
    script[("result", "comparator")] = [compare_result(["equal", status], side="reference")]
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category, result.terminal) == (
        "unscorable",
        None,
        category,
        terminal,
    )
    assert result.failure_class == (failure_kinds.VERIFIER_ERROR if terminal else failure_kinds.PROVIDER_UNAVAILABLE)
    assert result.cases_attempted == result.cases_equal == 0 and result.first_failed_case is None
    assert_job_cleaned(result, provider, "reference", "comparator")
    # The preflight judges observed reference outcomes against the stored expectations, never the other way round.
    compare_job = domain_job(provider, "comparator")
    task = make_task()
    assert compare_job["observed_role"] == "reference"
    assert [case["observed"] for case in compare_job["cases"]] == [VALUE, VALUE]
    assert [case["expected"] for case in compare_job["cases"]] == [
        case.expected.model_dump(mode="json") for case in task.test_cases
    ]


@pytest.mark.parametrize("role", ["reference", "candidate"])
@pytest.mark.parametrize("status", runner.VERDICT_STATUSES)
@pytest.mark.parametrize(
    "side", ["reference", "candidate", "expected", "input", "request", "comparator", None, "other"]
)
def test_verdict_batch_role_status_side_contract(role, status, side):
    # Hand-transcribed comparator._result call sites. No comparator import or scientific work.
    valid = {
        "reference": {
            ("equal", "reference"),
            ("invalid_reference", "reference"),
            ("invalid_expected", "expected"),
            ("uncertain", "reference"),
            ("uncertain", "comparator"),
            ("uncertain", "request"),
        },
        "candidate": {
            ("equal", "candidate"),
            ("mismatch", "candidate"),
            ("invalid_candidate", "candidate"),
            ("invalid_expected", "expected"),
            ("uncertain", "candidate"),
            ("uncertain", "comparator"),
            ("uncertain", "request"),
        },
    }
    results = [verdict(status, side=side)]
    if (status, side) in valid[role]:
        assert execution._validate_verdict_batch(results, role, attempted=2) is None
    else:
        with pytest.raises(_Unscorable) as info:
            execution._validate_verdict_batch(results, role, attempted=2)
        assert (info.value.category, info.value.attempted) == ("result_invalid", 2)


@pytest.mark.parametrize(
    ("role", "defect", "invalid"),
    [
        ("reference", "invalid_reference", "mismatch"),
        ("reference", "invalid_reference", "invalid_candidate"),
        ("reference", "invalid_expected", "mismatch"),
        ("reference", "invalid_expected", "invalid_candidate"),
        ("candidate", "mismatch", "invalid_reference"),
        ("candidate", "invalid_candidate", "invalid_reference"),
        ("candidate", "invalid_expected", "invalid_reference"),
    ],
)
@pytest.mark.parametrize("order", list(permutations(range(3))))
async def test_invalid_comparator_role_precedes_disposition_in_every_order(role, defect, invalid, order):
    data = make_task().model_dump(mode="json")
    data["test_cases"].append(data["test_cases"][-1])
    task = TaskData.model_validate(data)
    script = happy_script()
    script[("result", "reference")] = script[("result", "candidate")] = run_result("completed", [VALUE] * 3)
    statuses = ["equal", defect, invalid]
    report = compare_result([statuses[index] for index in order], side=role)
    script[("result", "comparator")] = (
        [report] if role == "reference" else [compare_result(["equal"] * 3, side="reference"), report]
    )
    grader, provider = make_grader(script)
    result = await grader.grade(task, CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "result_invalid")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cases_attempted == (0 if role == "reference" else 3)
    assert result.cases_equal == 0 and result.first_failed_case is None
    assert_job_cleaned(result, provider, *(DOMAINS if role == "candidate" else ("reference", "comparator")))


@pytest.mark.parametrize("role", ["reference", "candidate"])
@pytest.mark.parametrize("side", ["other_role", "expected", "input", "request", "comparator", None])
@pytest.mark.parametrize("wrong_index", [0, 1, None], ids=["first", "last", "all"])
async def test_all_equal_comparator_reports_require_the_observed_role(role, side, wrong_index):
    if side == "other_role":
        side = "candidate" if role == "reference" else "reference"
    report = compare_result(["equal", "equal"], side=role)
    for index in range(2):
        if wrong_index is None or wrong_index == index:
            report["results"][index]["side"] = side
    script = happy_script()
    if role == "reference":
        script[("result", "comparator")] = [report]
    else:
        script[("result", "comparator")][1] = report
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "result_invalid")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cases_attempted == (0 if role == "reference" else 2)
    assert result.cases_equal == 0 and result.first_failed_case is None
    assert_job_cleaned(result, provider, *(DOMAINS if role == "candidate" else ("reference", "comparator")))


async def test_domain_separation_and_expected_metadata_projection():
    grader, provider = make_grader(happy_script())
    await grader.grade(make_task(), CANDIDATE)
    assert sorted(created_domains(provider)) == sorted(DOMAINS) and len(provider.sandboxes) == 3
    (workdir,) = {entry["cwd"] for entry in runner_execs(provider, "candidate")}
    candidate_files = domain_files(provider, "candidate")
    assert f"{workdir}/critpt_grader_rt/comparator.py" not in candidate_files
    candidate_job = domain_job(provider, "candidate")
    assert set(candidate_job) == {
        "version",
        "mode",
        "nonce",
        "entrypoint",
        "source_path",
        "input_conversions",
        "cases",
        "limits",
    }
    assert candidate_job["input_conversions"] == []  # make_task declares none, so the key is always present
    assert candidate_job["source_path"] == f"{workdir}/source.py"  # the job names the domain's own workdir
    assert all(set(case) == {"args", "kwargs"} for case in candidate_job["cases"])
    assert candidate_files[f"{workdir}/source.py"] == CANDIDATE.encode()
    assert "source.py" not in "".join(domain_files(provider, "comparator"))
    compare_job = domain_job(provider, "comparator")
    task = make_task()
    assert compare_job["observed_role"] == "candidate"
    assert compare_job["cases"][1]["expected"] == task.test_cases[1].expected.model_dump(mode="json")
    assert compare_job["cases"][1]["policy"] == task.comparison_policy(1)
    for domain in DOMAINS:
        command = runner_execs(provider, domain)[0]["command"]
        assert 'runner.py" ' in command
        assert command.startswith('"/usr/local/bin/python3" -I -B')


@pytest.mark.asyncio
async def test_configured_server_default_tolerance_reaches_the_comparator():
    # The operator-set silent-leaf default is plumbed app.py -> Grader -> build_compare_job ->
    # comparison_policy, so a non-default yaml value reaches the comparator inside the per-case policy.
    comparator_jobs = []

    async def run_comparator():  # each comparator invocation records the job it reads when its runner starts
        comparator_jobs.append(domain_job(provider, "comparator"))
        return exited(runner.EXIT_COMPLETED)

    ScriptedProvider.reset()
    provider = ScriptedProvider({**happy_script(), ("exec", "comparator"): run_comparator})
    grader = Grader(make_policy(), provider=provider, spec=make_spec(), default_rtol="2e-4", default_atol="3e-9")
    await grader.grade(make_task(), CANDIDATE)
    task = make_task()
    assert [job["observed_role"] for job in comparator_jobs] == ["reference", "candidate"]
    for compare_job in comparator_jobs:
        for index, case in enumerate(compare_job["cases"]):
            assert case["policy"]["default_rtol"] == "2e-4"
            assert case["policy"]["default_atol"] == "3e-9"
            assert case["policy"] == task.comparison_policy(index, "2e-4", "3e-9")


async def test_run_command_runs_as_the_configured_exec_user():
    # The runner refuses root, so every exec in every domain runs as the policy exec_user, which the provider applies.
    # Nothing in the command switches users. This asserts the configured user reaches the provider, not the default.
    grader, provider = make_grader(happy_script(), exec_user="_svc-user_2")
    await grader.grade(make_task(), CANDIDATE)
    assert sorted(created_domains(provider)) == sorted(DOMAINS)
    assert provider.exec_calls and all(entry["user"] == "_svc-user_2" for entry in provider.exec_calls)
    for domain in DOMAINS:
        command = runner_execs(provider, domain)[0]["command"]
        assert "runuser" not in command and command.endswith(">/dev/null 2>&1")
        assert command.startswith('"/usr/local/bin/python3" -I -B')


@pytest.mark.parametrize("value", ["-grader", "gra der", "grader;ls", "grader$x", "grader|x"])
def test_policy_rejects_shell_unsafe_exec_user(value):
    # Providers that switch users through a shell wrapper embed exec_user, so a leading hyphen or any shell
    # metacharacter must be rejected.
    with pytest.raises(ValidationError):
        make_policy(exec_user=value)


async def test_candidate_mismatch_scores_zero_with_first_failed_case():
    script = happy_script()
    script[("result", "reference")] = run_result("completed", [VALUE, value_outcome(["int", "3"])])
    script[("result", "candidate")] = run_result("completed", [VALUE, value_outcome(["int", "4"])])
    script[("result", "comparator")] = [
        compare_result(["equal", "equal"], side="reference"),
        compare_result(["equal", "mismatch"]),
    ]
    grader, provider = make_grader(script)
    source = "def f(x):\n    return x + (1 if x == 1 else 2)\n"
    result = await grader.grade(make_task(), source)
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, "candidate_mismatch")
    assert result.first_failed_case == result.cases_equal == 1 and result.cases_attempted == 2
    assert result.failure_class is None and result.terminal is None
    compare_job = domain_job(provider, "comparator")
    assert compare_job["cases"][1]["observed"] == value_outcome(["int", "4"])
    assert compare_job["cases"][1]["expected"] == {"kind": "value", "value": wire(["int", "3"])}
    assert_job_cleaned(result, provider, *DOMAINS)


async def test_comparator_validated_invalid_candidate_value_still_scores_zero():
    script = happy_script()
    invalid_value = value_outcome(["int", "not-an-integer"])
    script[("result", "candidate")] = run_result("completed", [VALUE, invalid_value])
    script[("result", "comparator")][1] = compare_result(["equal", "invalid_candidate"])
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, "candidate_invalid_output")
    assert result.first_failed_case == result.cases_equal == 1 and result.cases_attempted == 2
    assert result.failure_class is None and result.terminal is None
    compare_job = domain_job(provider, "comparator")
    assert compare_job["cases"][1]["observed"] == invalid_value  # the host forwards it without decoding
    assert_job_cleaned(result, provider, *DOMAINS)


@pytest.mark.parametrize(
    ("status", "outcome", "category", "first"),
    [
        ("invalid_candidate", "failed", "candidate_invalid_output", 1),
        ("uncertain", "unscorable", "comparator_uncertain", None),
        ("invalid_expected", "unscorable", "comparator_uncertain", None),
        # A reference-role verdict cannot judge the candidate.
        ("invalid_reference", "unscorable", "result_invalid", None),
    ],
)
async def test_candidate_verdicts_map_to_bounded_categories(status, outcome, category, first):
    script = happy_script()
    script[("result", "comparator")] = [
        compare_result(["equal", "equal"], side="reference"),
        compare_result(["equal", status]),
    ]
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.category, result.first_failed_case) == (outcome, category, first)
    assert result.reward == (0.0 if outcome == "failed" else None)
    assert result.failure_class == (None if outcome == "failed" else failure_kinds.PROVIDER_UNAVAILABLE)
    assert result.terminal is (None if outcome == "failed" else False)
    assert result.cases_attempted == 2
    assert result.cases_equal == (1 if outcome == "failed" else 0)
    assert_job_cleaned(result, provider, *DOMAINS)


@pytest.mark.parametrize(
    ("observation", "category"),
    [(EXCEPTION, "candidate_exception"), (ENCODING_ERROR, "candidate_encoding_error")],
    ids=["exception", "encoding_error"],
)
@pytest.mark.parametrize(
    "statuses",
    [["equal", "equal"], ["equal", "mismatch"], ["equal", "invalid_candidate"], ["mismatch", "equal"]],
)
async def test_candidate_non_value_observations_score_zero(observation, category, statuses):
    script = happy_script()
    script[("result", "candidate")] = run_result("completed", [VALUE, observation])
    judged = compare_result(statuses)
    script[("result", "comparator")][1] = judged
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    # A candidate that raises or returns an unencodable object after a completed preflight is a
    # candidate-attributable fault, scored 0 at the first offending case, never an authenticated pass.
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, category)
    assert result.failure_class is None and result.terminal is None
    assert result.cases_attempted == 2 and result.cases_equal == 0 and result.first_failed_case == 1
    assert script[("result", "comparator")] == [judged]  # no verdict on the candidate was consumed
    assert result_reads(provider, "comparator") == 1
    assert_job_cleaned(result, provider, *DOMAINS)


def _task_with_exception_case():
    # A two-case task whose second case expects an exception: the code must raise, and neither the exception type
    # nor its message is compared.
    data = make_task().model_dump(mode="json")
    data["test_cases"][1]["expected"] = {"kind": "exception"}
    return TaskData.model_validate(data)


@pytest.mark.parametrize("domain", ["reference", "candidate"])
async def test_expected_exception_case_admits_a_raised_observation(domain):
    # An exception observed where the stored expectation is an exception is admitted to comparison, not refused.
    # The verdicts are scripted equal, so this is control-flow evidence that the observation reaches comparison,
    # not a qualification of the comparator's own exception rule.
    task = _task_with_exception_case()
    script = happy_script()
    script[("result", domain)] = run_result("completed", [VALUE, EXCEPTION])
    grader, provider = make_grader(script)
    result = await grader.grade(task, CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("passed", 1.0, "passed")
    assert result.cases_attempted == 2 and result.cases_equal == 2 and result.first_failed_case is None
    # Every scripted verdict was consumed: preflight over the reference and the judge over the candidate both ran.
    assert script[("result", "comparator")] == []
    assert_job_cleaned(result, provider, *DOMAINS)


async def test_expected_exception_reference_that_does_not_raise_is_a_reference_defect():
    # A reference that returns a value where the case expects an exception is a reference defect. The value is
    # admitted to preflight, and the comparator's invalid_reference verdict maps to the terminal reference_mismatch,
    # the same as any reference that contradicts its stored expectation. No candidate is created.
    task = _task_with_exception_case()
    script = happy_script()  # reference returns [VALUE, VALUE]: it did not raise on the exception case
    script[("result", "comparator")][0] = compare_result(["equal", "invalid_reference"], side="reference")
    grader, provider = make_grader(script)
    result = await grader.grade(task, CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "reference_mismatch")
    assert result.failure_class == failure_kinds.VERIFIER_ERROR and result.terminal is True
    assert result.cases_attempted == 0 and result.cases_equal == 0 and result.first_failed_case is None
    assert_job_cleaned(result, provider, "reference", "comparator")


async def test_expected_exception_candidate_that_does_not_raise_is_a_wrong_answer():
    # A candidate that returns a value where the case expects an exception did not raise. The value is admitted to
    # the judge, and the comparator's mismatch verdict scores it 0 as candidate_mismatch at that case.
    task = _task_with_exception_case()
    script = happy_script()
    script[("result", "reference")] = run_result("completed", [VALUE, EXCEPTION])  # reference raises as required
    script[("result", "candidate")] = run_result("completed", [VALUE, VALUE])  # candidate did not raise
    script[("result", "comparator")][1] = compare_result(["equal", "mismatch"])
    grader, provider = make_grader(script)
    result = await grader.grade(task, CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, "candidate_mismatch")
    assert result.failure_class is None and result.terminal is None
    assert result.cases_attempted == 2 and result.cases_equal == 1 and result.first_failed_case == 1
    assert_job_cleaned(result, provider, *DOMAINS)


@pytest.mark.parametrize("code", runner.SOURCE_ERROR_CODES)
async def test_candidate_source_error_reports_are_not_host_source_contracts(code):
    script = happy_script()
    script[("result", "candidate")] = run_result("source_error", [], code=code)
    script[("exec", "candidate")] = exited(runner.EXIT_SOURCE_ERROR)
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    # A source-error report after a completed preflight scores 0 with the category preserved. The bounded code
    # and correlated exit still do not authenticate the cause, so a host source contract is not implied.
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, "candidate_source_error")
    assert result.failure_class is None and result.terminal is None
    assert result.cases_attempted == result.cases_equal == 0 and result.first_failed_case is None
    assert result_reads(provider, "comparator") == 1
    assert_job_cleaned(result, provider, *DOMAINS)


async def test_candidate_timeout_correlation_is_not_fault_attribution():
    script = happy_script()
    script[("result", "candidate")] = run_result("timeout", [VALUE], timed_out=1)
    script[("exec", "candidate")] = exited(runner.EXIT_TIMEOUT)
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    # A suite timeout after a completed preflight scores 0 with the category preserved. Correlation is not
    # authentication, so the category names what the candidate domain reported, not a proven cause.
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, "candidate_timeout")
    assert result.failure_class is None and result.terminal is None
    assert result.cases_attempted == 1 and result.cases_equal == 0 and result.first_failed_case is None
    assert result_reads(provider, "comparator") == 1
    assert_job_cleaned(result, provider, *DOMAINS)
    # The same payload behind the provider's exit 0 disagrees with its own status: uncorrelated, never scored.
    script = happy_script()
    script[("result", "candidate")] = run_result("timeout", [VALUE], timed_out=1)
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category, result.terminal) == (
        "unscorable",
        None,
        "result_invalid",
        False,
    )
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE
    assert_job_cleaned(result, provider, *DOMAINS)


@pytest.mark.parametrize("exit_code", sorted(runner.RESULT_EXIT_CODES))
async def test_missing_candidate_result_is_infrastructure_uncertainty(exit_code):
    # Behind every result exit, EXIT_TIMEOUT included, an absent file is not a candidate timeout or any verdict.
    script = happy_script()
    script[("result", "candidate")] = MISSING_FILE
    script[("exec", "candidate")] = exited(exit_code)
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert result.category == "candidate_no_result" and result.reward is None
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result_reads(provider, "candidate") == 1


@pytest.mark.parametrize(
    ("failure", "category"),
    [
        (b"x" * (RESULT_BYTES_LIMIT + 1), "transfer_limit"),
        (ConnectionResetError("reset"), "candidate_no_result"),
        (asyncio.TimeoutError(), "candidate_no_result"),
        (hang, "candidate_no_result"),
    ],
    ids=["limit", "transport", "timeout", "hang"],
)
async def test_result_transfer_failures_are_bounded_and_never_scored(monkeypatch, failure, category):
    # An oversized file is refused at the size probe. A download that fails or hangs after the probe found the file
    # proves nothing about the candidate: it is a missing result, bounded by the transfer timeout.
    monkeypatch.setattr(sandbox_domain, "TRANSFER_GRACE_S", 0.01)
    script = happy_script()
    if type(failure) is bytes:
        script[("result", "candidate")] = failure
    grader, provider = make_grader(script, transfer_timeout_s=0.05)
    if type(failure) is not bytes:
        fail_downloads(provider, "candidate", failure)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, category)
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cleanup == {domain: "deleted" for domain in DOMAINS} and not stray_tasks()


async def test_malformed_result_files_are_result_invalid_and_stop_the_job():
    script = happy_script()
    script[("result", "reference")] = run_result("completed", [LEGACY_VALUE, LEGACY_VALUE])
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.category, result.terminal) == ("unscorable", "result_invalid", False)
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE
    assert created_domains(provider) == ["reference"] and result.cleanup["reference"] == "deleted"

    script = happy_script()
    script[("result", "candidate")] = b"\xff{not json"
    grader, _ = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.category, result.cases_attempted, result.cases_equal) == ("result_invalid", 0, 0)

    script = happy_script()
    script[("result", "candidate")] = run_result("completed", [VALUE])  # one outcome for two cases
    grader, _ = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.category, result.cases_attempted) == ("result_invalid", 1)
    assert result.cleanup == {domain: "deleted" for domain in DOMAINS}


async def test_comparator_uncertain_is_unscorable():
    script = happy_script()
    script[("result", "comparator")] = [
        compare_result(["equal", "equal"], side="reference"),
        compare_result(["uncertain", "equal"]),
    ]
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "comparator_uncertain")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cases_attempted == 2 and result.cases_equal == 0 and result.first_failed_case is None
    assert_job_cleaned(result, provider, *DOMAINS)


# A preflight timeout (reference role) or a comparator runner error on either side stays unattributed
# comparator uncertainty. The candidate judge-stage timeout is the one runtime report that becomes a candidate
# fault, covered separately in test_candidate_judge_timeout_scores_invalid_output.
@pytest.mark.parametrize(
    ("role", "status", "code", "exit_code"),
    [
        ("reference", "timeout", None, runner.EXIT_TIMEOUT),
        ("reference", "runner_error", "limits", runner.EXIT_RUNNER_ERROR),
        ("candidate", "runner_error", "limits", runner.EXIT_RUNNER_ERROR),
    ],
)
async def test_comparator_runtime_reports_are_nonterminal(role, status, code, exit_code):
    script = happy_script()
    report = {**compare_result([]), "status": status, "code": code}
    if role == "reference":
        script[("result", "comparator")] = [report]
        script[("exec", "comparator")] = exited(exit_code)
    else:
        script[("result", "comparator")][1] = report
        script[("exec", "comparator")] = [exited(runner.EXIT_COMPLETED), exited(exit_code)]
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "comparator_uncertain")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cases_attempted == (0 if role == "reference" else 2)
    assert result.cases_equal == 0 and result.first_failed_case is None
    assert result_reads(provider, "comparator") == (1 if role == "reference" else 2)
    if role == "reference":
        assert_job_cleaned(result, provider, "reference", "comparator")
    else:
        assert_job_cleaned(result, provider, *DOMAINS)


@pytest.mark.parametrize("timed_out_case", [1, None], ids=["reported_case", "no_case"])
async def test_candidate_judge_timeout_scores_invalid_output(timed_out_case):
    # A judge-stage timeout after a completed preflight is candidate-attributable: the comparator resolved the
    # reference/expected side cleanly at preflight, so a candidate value it cannot resolve within the CPU budget
    # scores 0 as candidate_invalid_output, at the case the runner reported it timed out on when one is given.
    script = happy_script()
    script[("result", "comparator")][1] = {**compare_result([]), "status": "timeout", "timed_out_case": timed_out_case}
    script[("exec", "comparator")] = [exited(runner.EXIT_COMPLETED), exited(runner.EXIT_TIMEOUT)]
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, "candidate_invalid_output")
    assert result.failure_class is None and result.terminal is None
    assert result.cases_attempted == 2 and result.cases_equal == 0 and result.first_failed_case == timed_out_case
    assert result_reads(provider, "comparator") == 2
    assert_job_cleaned(result, provider, *DOMAINS)


@pytest.mark.parametrize(
    ("role", "first", "uncertain"),
    [
        ("reference", "invalid_reference", "uncertain"),
        ("reference", "invalid_expected", "uncertain"),
        # A decided candidate failure (a mismatch or an invalid output) outranks uncertainty instead:
        # test_decided_candidate_failure_outranks_candidate_side_uncertainty.
        ("candidate", "mismatch", "invalid_expected"),
    ],
)
@pytest.mark.parametrize("reverse", [False, True], ids=["defect_first", "uncertainty_first"])
@pytest.mark.parametrize("uncertain_side", ["observed_role", "comparator", "request"])
async def test_later_comparator_uncertainty_cannot_be_hidden_by_a_defect(
    role, first, uncertain, reverse, uncertain_side
):
    script = happy_script()
    report = compare_result([first, uncertain], side=role)
    if uncertain == "uncertain":
        report["results"][1]["side"] = role if uncertain_side == "observed_role" else uncertain_side
    if reverse:
        report["results"].reverse()
    if role == "reference":
        script[("result", "comparator")] = [report]
    else:
        script[("result", "comparator")][1] = report
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "comparator_uncertain")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cases_attempted == (0 if role == "reference" else 2)
    assert result.cases_equal == 0 and result.first_failed_case is None
    if role == "reference":
        assert_job_cleaned(result, provider, "reference", "comparator")
    else:
        assert_job_cleaned(result, provider, *DOMAINS)


# Uncertain verdicts the comparator emits on each side the candidate role contract admits: a per-case timeout, an
# undecided comparison, a malformed observation and a malformed request.
CANDIDATE_UNCERTAINTY = [
    pytest.param("comparator", "comparator_timeout", id="per_case_timeout"),
    pytest.param("comparator", "symbolic_undecided", id="undecided_comparison"),
    pytest.param("candidate", "invalid_observation_protocol", id="observed_role"),
    pytest.param("request", "invalid_request", id="request"),
]


def _judged_job(statuses, side="comparator", code="comparator_timeout"):
    """A task with one case per status, a clean preflight, and a candidate judge reporting ``statuses`` in order.

    Every ``uncertain`` verdict carries ``side`` and ``code``; every other verdict carries its usual side.
    """
    count = len(statuses)
    data = make_task().model_dump(mode="json")
    data["test_cases"] = [data["test_cases"][index % 2] for index in range(count)]
    report = compare_result(statuses)
    for item in report["results"]:
        if item["status"] == "uncertain":
            item.update(side=side, code=code)
    script = happy_script()
    script[("result", "reference")] = script[("result", "candidate")] = run_result("completed", [VALUE] * count)
    script[("result", "comparator")] = [compare_result(["equal"] * count, side="reference"), report]
    return TaskData.model_validate(data), script


@pytest.mark.parametrize(("side", "code"), CANDIDATE_UNCERTAINTY)
@pytest.mark.parametrize(
    ("statuses", "category", "first", "equal"),
    [
        (["mismatch", "uncertain"], "candidate_mismatch", 0, 0),
        (["uncertain", "mismatch"], "candidate_mismatch", 1, 0),
        (["equal", "uncertain", "mismatch"], "candidate_mismatch", 2, 1),
        (["invalid_candidate", "uncertain"], "candidate_invalid_output", 0, 0),
        (["uncertain", "invalid_candidate"], "candidate_invalid_output", 1, 0),
        # The first decided failure names the category, whatever follows it.
        (["invalid_candidate", "uncertain", "mismatch"], "candidate_invalid_output", 0, 0),
        (["uncertain", "invalid_candidate", "mismatch"], "candidate_invalid_output", 1, 0),
    ],
    ids=[
        "mismatch_first",
        "uncertainty_first",
        "equal_uncertain_mismatch",
        "invalid_output_first",
        "uncertainty_before_invalid_output",
        "invalid_output_before_mismatch",
        "uncertain_invalid_output_mismatch",
    ],
)
async def test_decided_candidate_failure_outranks_candidate_side_uncertainty(
    statuses, category, first, equal, side, code
):
    # A candidate that fails a case outright and times out on a later case is failed, not unscorable: the first
    # decided failure (a mismatch or an invalid output) names the category and the failed case, and an uncertain case
    # is passed over and never counted equal.
    task, script = _judged_job(statuses, side, code)
    grader, provider = make_grader(script)
    result = await grader.grade(task, CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, category)
    assert result.failure_class is None and result.terminal is None  # a scored row, never re-graded
    assert result.first_failed_case == first and result.cases_equal == equal
    assert result.cases_attempted == len(statuses)
    assert_job_cleaned(result, provider, *DOMAINS)


@pytest.mark.parametrize(("side", "code"), CANDIDATE_UNCERTAINTY)
async def test_uncertainty_on_every_case_stays_unscorable(side, code):
    task, script = _judged_job(["uncertain", "uncertain"], side, code)
    grader, provider = make_grader(script)
    result = await grader.grade(task, CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "comparator_uncertain")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cases_attempted == 2 and result.cases_equal == 0 and result.first_failed_case is None
    assert_job_cleaned(result, provider, *DOMAINS)


@pytest.mark.parametrize(
    ("defect", "category"),
    [
        ("invalid_reference", "result_invalid"),  # a reference-role verdict cannot judge the candidate
        ("invalid_expected", "comparator_uncertain"),
    ],
)
@pytest.mark.parametrize("order", list(permutations(range(3))))
async def test_reference_or_expectation_defect_outranks_mismatch_and_uncertainty(defect, category, order):
    statuses = [defect, "mismatch", "uncertain"]
    task, script = _judged_job([statuses[index] for index in order])
    grader, provider = make_grader(script)
    result = await grader.grade(task, CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, category)
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cases_attempted == 3 and result.cases_equal == 0 and result.first_failed_case is None
    assert_job_cleaned(result, provider, *DOMAINS)


async def test_candidate_source_contract_checked_before_allocation():
    grader, provider = make_grader(happy_script())
    policy = grader.policy
    result = await grader.grade(make_task(), "x" * (policy.max_candidate_source_bytes + 1))
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, "candidate_source_limit")
    assert result.failure_class is None and result.terminal is None and result.cleanup == {}
    assert provider.created == []
    assert validate_candidate_source("def f():\n\x00", policy) == "candidate_source_limit"
    assert validate_candidate_source(CANDIDATE, policy) is None


@pytest.mark.parametrize(
    "source",
    [None, b"source", "", " \n", "x\x00", "\ud800", "é" * 5],
    ids=["not-text", "bytes", "empty", "blank", "nul", "invalid-utf8", "utf8-byte-limit"],
)
async def test_host_source_contract_violations_remain_scored_without_allocation(source):
    grader, provider = make_grader(happy_script(), max_candidate_source_bytes=8)
    policy = grader.policy
    assert validate_candidate_source(source, policy) == "candidate_source_limit"
    result = await grader.grade(make_task(), source)
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, "candidate_source_limit")
    assert result.failure_class is None and result.terminal is None
    assert result.cases_attempted == result.cases_equal == 0 and result.first_failed_case is None
    assert result.cleanup == {} and provider.created == []
    assert validate_candidate_source("é" * 4, policy) is None  # exactly eight bytes, and source is never parsed here


async def test_cancellation_cleans_created_domains():
    script = happy_script()
    script[("exec", "candidate")] = asyncio.CancelledError()
    grader, provider = make_grader(script)
    with pytest.raises(asyncio.CancelledError):
        await grader.grade(make_task(), CANDIDATE)
    assert not any(box["live"] for box in provider.sandboxes.values())
    assert sorted(provider.closed) == sorted(provider.sandboxes) and len(provider.closed) == 3


# --- cleanup, deadline, drain and the reference cache ------------------------------------------- #


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("status", ["equal", "mismatch"])
async def test_a_failed_stop_is_retried_by_the_final_sweep_and_the_verdict_stands(domain, status):
    script = happy_script()
    script[("result", "comparator")][1] = compare_result(["equal", status])
    script[("stop", domain)] = [RuntimeError("synthetic stop failure")]  # the domain's mid-job stop fails once
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    expected = ("passed", 1.0, "passed") if status == "equal" else ("failed", 0.0, "candidate_mismatch")
    assert (result.outcome, result.reward, result.category) == expected
    assert result.cases_attempted == 2 and result.cases_equal == (2 if status == "equal" else 1)
    assert_job_cleaned(result, provider, *DOMAINS)  # the final sweep's retry stopped it: deleted, stopped once


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("status", ["equal", "mismatch"])
async def test_a_stop_that_keeps_failing_is_reported_and_the_verdict_stands(domain, status):
    script = happy_script()
    script[("result", "comparator")][1] = compare_result(["equal", status])
    script[("stop", domain)] = RuntimeError("synthetic stop failure")  # every stop of this domain fails
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    expected = ("passed", 1.0, "passed") if status == "equal" else ("failed", 0.0, "candidate_mismatch")
    assert (result.outcome, result.reward, result.category) == expected
    assert result.failure_class is None and result.terminal is None
    assert result.cleanup == {name: "delete_failed" if name == domain else "deleted" for name in DOMAINS}
    assert sorted(closed_domains(provider)) == sorted(name for name in DOMAINS if name != domain)
    assert [box["domain"] for box in provider.sandboxes.values() if box["live"]] == [domain]
    assert not stray_tasks()


async def test_a_cancelled_stop_is_reraised_only_after_every_domain_stopped(monkeypatch):
    # The candidate worker aborts, so the comparator and the candidate are both first stopped by the final sweep.
    script = happy_script()
    script[("exec", "candidate")] = exited(runner.EXIT_WORKER_ABORTED)
    script[("stop", "comparator")] = [cancel_self]
    close_domains = Grader._close_domains
    recorded = {}

    async def recording_close_domains(self, domains, statuses):
        try:
            await close_domains(self, domains, statuses)
        finally:
            recorded.update(statuses)

    monkeypatch.setattr(Grader, "_close_domains", recording_close_domains)
    grader, provider = make_grader(script)
    with pytest.raises(asyncio.CancelledError):
        await grader.grade(make_task(), CANDIDATE)
    assert recorded == {"reference": "deleted", "comparator": "delete_failed", "candidate": "deleted"}
    assert sorted(closed_domains(provider)) == ["candidate", "reference"]  # the candidate stop ran regardless
    assert not grader._live and not stray_tasks()


async def test_a_failed_create_stops_the_job_and_creates_nothing_after_it():
    grader, provider = make_grader({**happy_script(), ("create", "comparator"): RuntimeError("synthetic create")})
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "provider_create")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert_job_cleaned(result, provider, "reference")  # the comparator and the candidate stay not_created


async def test_a_job_past_its_deadline_is_unscorable_and_every_domain_is_stopped(monkeypatch):
    candidate_run, entered = entered_hang()
    monkeypatch.setattr(ExecutionPolicy, "effective_job_timeout_s", lambda self: 2.0)
    grader, provider = make_grader({**happy_script(), ("exec", "candidate"): candidate_run})
    result = await grader.grade(make_task(), CANDIDATE)
    assert entered.is_set()  # the deadline fell inside the candidate run, after every domain was created
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "job_deadline")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert_job_cleaned(result, provider, *DOMAINS)


async def test_shutdown_stops_an_in_flight_job_from_creating_another_sandbox():
    gate, entered = asyncio.Event(), asyncio.Event()

    async def held():
        entered.set()
        await gate.wait()

    grader, provider = make_grader(happy_script())
    fail_downloads(provider, "reference", held)  # the job waits after the reference ran, before the comparator
    job = asyncio.create_task(grader.grade(make_task(), CANDIDATE))
    await until_entered(entered)  # the job holds inside the reference result download
    await grader.shutdown()
    gate.set()
    result = await job
    assert (result.outcome, result.category) == ("unscorable", "server_draining")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert_job_cleaned(result, provider, "reference")  # no sandbox was created once the drain began
    assert not grader._live


async def test_a_job_cancelled_during_its_final_stops_leaves_nothing_tracked():
    comparator_stop, entered = entered_hang()
    script = happy_script()
    script[("result", "candidate")] = run_result(
        "timeout", [VALUE], timed_out=1
    )  # the comparator is left to the sweep
    script[("exec", "candidate")] = exited(runner.EXIT_TIMEOUT)
    script[("stop", "comparator")] = [comparator_stop]
    grader, _ = make_grader(script)
    job = asyncio.create_task(grader.grade(make_task(), CANDIDATE))
    await until_entered(entered)  # the job is in its final sweep, waiting on the comparator stop
    job.cancel()
    with pytest.raises(asyncio.CancelledError):
        await job
    assert not grader._live and not stray_tasks()


@pytest.mark.parametrize("refusal", ["no_such_category", "provider_exec"])
async def test_a_cached_refusal_that_is_not_terminal_here_is_a_miss(refusal):
    cache = ReferenceCache()
    grader, provider = make_grader(happy_script(), cache=cache)
    digest = execution._runtime_digest(make_spec())
    provider_class = reference_cache.provider_class_name(provider)
    key = ReferenceCache.key(
        make_task(), grader.policy, digest, make_spec().image, None, None, provider_class=provider_class
    )
    cache.put(key, CachedReference(refusal=refusal))  # an entry this build never writes
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.category) == ("passed", "passed")
    assert len(runner_execs(provider, "reference")) == 1  # the reference ran: the entry was not trusted
    assert cache.get(key) == CachedReference(refusal=None)  # and the passed preflight replaced it


@pytest.mark.parametrize("category", ["passed", "reference_mismatch"])
async def test_a_failing_cache_write_never_replaces_the_outcome(monkeypatch, caplog, category):
    script = happy_script()
    if category == "reference_mismatch":
        script[("result", "comparator")] = [compare_result(["equal", "invalid_reference"], side="reference")]
    cache = ReferenceCache()
    monkeypatch.setattr(cache, "put", Mock(side_effect=OSError("synthetic cache write failure")))
    grader, _ = make_grader(script, cache=cache)
    with caplog.at_level(logging.DEBUG, logger=execution.__name__):
        result = await grader.grade(make_task(), CANDIDATE)
    assert result.category == category and cache.put.call_count == 1
    assert "OSError" in caplog.text and "synthetic cache write failure" not in caplog.text


# --- result shape validation (host-side, values never decoded) ---------------------------------- #


def parse(payload, mode="run"):
    data = payload if type(payload) is bytes else json.dumps(payload).encode()
    return _parse_result(data, mode)


def rejected(payload, mode="run"):
    with pytest.raises(_Unscorable) as info:
        parse(payload, mode)
    return info.value.category == "result_invalid"


def mutate(base, **changes):
    return {**base, **changes}


def without(base, key):
    return {name: value for name, value in base.items() if name != key}


GOOD_RUN = run_result("completed", [VALUE, EXCEPTION, ENCODING_ERROR])
GOOD_COMPARE = compare_result(
    ["equal", "mismatch", "uncertain", "invalid_candidate", "invalid_expected", "invalid_reference"]
)

MALFORMED_RUN = [
    pytest.param(b"{", id="not json"),
    pytest.param(b"\xff\xfe", id="not utf-8"),
    pytest.param(b"[" * 100_000, id="nested past the parser's recursion limit"),
    pytest.param(b"[]", id="not an object"),
    pytest.param(mutate(GOOD_RUN, extra=1), id="extra envelope key"),
    pytest.param(without(GOOD_RUN, "code"), id="missing envelope key"),
    pytest.param(mutate(without(GOOD_RUN, "outcomes"), results=[]), id="results key in run mode"),
    pytest.param(mutate(GOOD_RUN, version=2), id="wrong version"),
    pytest.param(mutate(GOOD_RUN, version="1"), id="text version"),
    pytest.param(mutate(GOOD_RUN, version=True), id="boolean version"),
    pytest.param(mutate(GOOD_RUN, mode="compare"), id="mode mismatch"),
    pytest.param(mutate(GOOD_RUN, nonce="a" * (MAX_NONCE_LENGTH + 1)), id="nonce too long"),
    pytest.param(mutate(GOOD_RUN, nonce=1), id="nonce not text"),
    pytest.param(mutate(GOOD_RUN, status="crashed"), id="unknown status"),
    pytest.param(mutate(GOOD_RUN, code="syntax_error"), id="code on completed"),
    pytest.param(mutate(GOOD_RUN, status="source_error", outcomes=[]), id="source_error without code"),
    pytest.param(
        mutate(GOOD_RUN, status="source_error", code="limits", outcomes=[]), id="source_error with runner code"
    ),
    pytest.param(
        mutate(GOOD_RUN, status="runner_error", code="syntax_error", outcomes=[]),
        id="runner_error with source code",
    ),
    pytest.param(
        mutate(GOOD_RUN, status="runner_error", code="x" * (MAX_CODE_LENGTH + 1), outcomes=[]), id="code too long"
    ),
    pytest.param(mutate(GOOD_RUN, timed_out_case=0), id="timed_out_case on completed"),
    pytest.param(mutate(GOOD_RUN, status="timeout", timed_out_case=MAX_CASES), id="timed_out_case past the cap"),
    pytest.param(mutate(GOOD_RUN, status="timeout", timed_out_case=-1), id="negative timed_out_case"),
    pytest.param(mutate(GOOD_RUN, status="timeout", timed_out_case=False), id="boolean timed_out_case"),
    pytest.param(mutate(GOOD_RUN, outcomes={}), id="outcomes not a list"),
    pytest.param(mutate(GOOD_RUN, outcomes=[VALUE] * (MAX_CASES + 1)), id="too many outcomes"),
    pytest.param(mutate(GOOD_RUN, outcomes=["value"]), id="outcome not an object"),
    pytest.param(mutate(GOOD_RUN, outcomes=[{"kind": "stdout", "value": "2"}]), id="unknown outcome kind"),
    pytest.param(mutate(GOOD_RUN, outcomes=[{"kind": 1}]), id="kind not text"),
    pytest.param(mutate(GOOD_RUN, outcomes=[LEGACY_VALUE]), id="legacy typed wire shape"),
    pytest.param(mutate(GOOD_RUN, outcomes=[{**VALUE, "stdout": ""}]), id="value outcome with extra field"),
    pytest.param(mutate(GOOD_RUN, outcomes=[{"kind": "value"}]), id="value outcome without a wire"),
    pytest.param(
        mutate(GOOD_RUN, outcomes=[{"kind": "value", "value": ["int", "2"]}]), id="bare node without envelope"
    ),
    pytest.param(
        mutate(GOOD_RUN, outcomes=[{"kind": "value", "value": {**wire(["int", "2"]), "type": "int"}}]),
        id="wire with extra key",
    ),
    pytest.param(
        mutate(GOOD_RUN, outcomes=[{"kind": "value", "value": {"format": "critpt-value-v0", "value": ["int", "2"]}}]),
        id="wire format mismatch",
    ),
    pytest.param(
        mutate(GOOD_RUN, outcomes=[{"kind": "value", "value": {"format": WIRE_FORMAT, "value": {"int": "2"}}}]),
        id="wire node not a list",
    ),
    pytest.param(mutate(GOOD_RUN, outcomes=[value_outcome(nested(60))]), id="wire past the depth cap"),
    pytest.param(mutate(GOOD_RUN, outcomes=[value_outcome(["int", 2**64])]), id="untagged large integer"),
    pytest.param(
        mutate(GOOD_RUN, outcomes=[value_outcome(["str", "x" * (MAX_WIRE_BYTES + 1)])]),
        id="one value past its byte cap",
    ),
    pytest.param(
        mutate(GOOD_RUN, outcomes=[value_outcome(["list", [["int", "1"]] * (MAX_JSON_NODES // 3 + 1)])]),
        id="one value past its node cap",
    ),
    pytest.param(mutate(GOOD_RUN, outcomes=[{**EXCEPTION, "traceback": ""}]), id="exception with unknown field"),
    pytest.param(mutate(GOOD_RUN, outcomes=[{"kind": "exception", "type": 1}]), id="exception type not text"),
    pytest.param(
        mutate(GOOD_RUN, outcomes=[{"kind": "exception", "message": "m" * (MAX_TEXT_BYTES + 1)}]),
        id="exception message too long",
    ),
    pytest.param(mutate(GOOD_RUN, outcomes=[{"kind": "encoding_error"}]), id="encoding_error without code"),
    pytest.param(
        mutate(GOOD_RUN, outcomes=[{"kind": "encoding_error", "code": None}]), id="encoding_error code not text"
    ),
    pytest.param(
        mutate(GOOD_RUN, outcomes=[{"kind": "encoding_error", "code": "x" * (MAX_CODE_LENGTH + 1)}]),
        id="encoding_error code too long",
    ),
    pytest.param(mutate(GOOD_RUN, outcomes=[{**ENCODING_ERROR, "path": ""}]), id="encoding_error with extra field"),
]

MALFORMED_COMPARE = [
    pytest.param(mutate(GOOD_COMPARE, mode="run"), id="mode mismatch"),
    pytest.param(mutate(without(GOOD_COMPARE, "results"), outcomes=[]), id="outcomes key in compare mode"),
    pytest.param(
        mutate(GOOD_COMPARE, status="source_error", code="syntax_error", results=[]),
        id="source_error is not a compare status",
    ),
    pytest.param(
        mutate(GOOD_COMPARE, results=[mutate(verdict("uncertain"), equal=False)]), id="legacy boolean on uncertain"
    ),
    pytest.param(
        mutate(GOOD_COMPARE, results=[mutate(verdict("mismatch"), equal=None)]), id="mismatch without its boolean"
    ),
    pytest.param(
        mutate(GOOD_COMPARE, results=[mutate(verdict("equal"), equal=1)]), id="integer stands in for the boolean"
    ),
    pytest.param(mutate(GOOD_COMPARE, results=[verdict("close_enough")]), id="unknown verdict status"),
    pytest.param(mutate(GOOD_COMPARE, results=[mutate(verdict("equal"), status=None)]), id="verdict status not text"),
    pytest.param(mutate(GOOD_COMPARE, results=[mutate(verdict("equal"), version=2)]), id="verdict version"),
    pytest.param(mutate(GOOD_COMPARE, results=[mutate(verdict("equal"), observed={})]), id="verdict with extra key"),
    pytest.param(mutate(GOOD_COMPARE, results=[without(verdict("equal"), "path")]), id="verdict missing key"),
    pytest.param(mutate(GOOD_COMPARE, results=[verdict("equal", path="p" * 1025)]), id="path too long"),
    pytest.param(mutate(GOOD_COMPARE, results=[mutate(verdict("equal"), path=None)]), id="path not text"),
    pytest.param(mutate(GOOD_COMPARE, results=[verdict("equal", side="s" * 257)]), id="side too long"),
    pytest.param(mutate(GOOD_COMPARE, results=[mutate(verdict("equal"), code=3)]), id="code not text"),
    pytest.param(mutate(GOOD_COMPARE, results=["equal"]), id="verdict not an object"),
    pytest.param(mutate(GOOD_COMPARE, results=[verdict("equal")] * (MAX_CASES + 1)), id="too many verdicts"),
]


@pytest.mark.parametrize("payload", MALFORMED_RUN)
def test_parse_result_rejects_malformed_run_results(payload):
    assert rejected(payload, "run")


@pytest.mark.parametrize("payload", MALFORMED_COMPARE)
def test_parse_result_rejects_malformed_compare_results(payload):
    assert rejected(payload, "compare")


def test_parse_result_accepts_every_documented_envelope():
    assert parse(GOOD_RUN)["outcomes"] == [VALUE, EXCEPTION, ENCODING_ERROR]
    diagnostics = [{"kind": "exception"}, {"kind": "exception", "type": None, "message": "m"}]
    assert parse(run_result("completed", diagnostics))["outcomes"] == diagnostics
    assert parse(run_result("timeout", [VALUE], timed_out=1))["timed_out_case"] == 1
    assert parse(run_result("timeout", []))["status"] == "timeout"
    for code in runner.SOURCE_ERROR_CODES:
        assert parse(run_result("source_error", [], code=code))["code"] == code
    for code in runner.RUNNER_ERROR_CODES:
        assert parse(run_result("runner_error", [], code=code))["code"] == code
    assert parse(run_result("completed", [VALUE] * MAX_CASES))["outcomes"] == [VALUE] * MAX_CASES
    assert parse(run_result("completed", [value_outcome(nested(30))]))["status"] == "completed"
    assert parse(GOOD_COMPARE, "compare")["results"] == GOOD_COMPARE["results"]


def test_parse_result_holds_each_value_to_its_own_caps_not_the_suite():
    # Each value stays under the per-value node and byte caps on its own, but together they exceed both. Applying
    # either cap to the whole suite would refuse a legal result. The counts are derived from the live caps, so
    # this stays a suite-vs-value check if the caps move again.
    per_value_items = 18_000
    heavy = value_outcome(["list", [["int", "1"]] * per_value_items])  # 3 + 3*items JSON nodes, under both caps
    heavy_nodes = 3 + 3 * per_value_items
    count = MAX_JSON_NODES // heavy_nodes + 1  # enough node-heavy values to exceed the node cap in aggregate
    long_text = value_outcome(["str", "x" * (MAX_WIRE_BYTES - 128)])  # just under the byte cap on its own
    outcomes = [heavy] * count + [long_text] * 2
    assert count + 2 <= MAX_CASES  # the whole result still fits the case-count cap
    assert heavy_nodes < MAX_JSON_NODES  # each node-heavy value is under the per-value node cap
    assert count * heavy_nodes > MAX_JSON_NODES  # together they exceed it
    assert 2 * (MAX_WIRE_BYTES - 128) > MAX_WIRE_BYTES  # two long values exceed the byte cap in aggregate
    result = parse(run_result("completed", outcomes))
    assert result["outcomes"] == outcomes  # returned exactly as observed, never decoded


def test_parse_result_enforces_the_total_byte_cap():
    data = json.dumps(run_result("completed", [VALUE])).encode()
    assert parse(data + b" " * (RESULT_BYTES_LIMIT - len(data)))["status"] == "completed"
    assert rejected(data + b" " * (RESULT_BYTES_LIMIT + 1 - len(data)))


# --- provider exit and result correlation (necessary checks that do not authenticate a file) -- #


NON_RESULT_EXITS = [
    pytest.param(exited(runner.EXIT_CONTAINMENT_FAILED), id="containment_failed"),
    pytest.param(exited(runner.EXIT_WORKER_ABORTED), id="worker_aborted"),
    pytest.param(exited(runner.EXIT_PROTOCOL_VIOLATION), id="protocol_violation"),
    pytest.param(exited(runner.EXIT_USAGE), id="usage"),
    pytest.param(exited(137), id="unknown"),
    pytest.param(exited(-9), id="negative"),
]
NO_EXIT_REPORTS = [
    pytest.param(exited(124, error_type="timeout"), id="provider timeout sentinel"),
    pytest.param(exited(125, error_type="sandbox"), id="provider sandbox sentinel"),
    pytest.param(exited(0, error_type="sandbox"), id="sentinel behind a zero code"),
    pytest.param(exited(True), id="boolean"),
    pytest.param(exited(None), id="missing"),
    pytest.param(exited("0"), id="text"),
    pytest.param(exited(0.0), id="float"),
]
# Shape-valid payloads behind a result exit other than the one their status pairs with.
MISPAIRED = [
    pytest.param(runner.EXIT_SOURCE_ERROR, run_result("completed", [VALUE, VALUE]), id="completed behind exit 1"),
    pytest.param(runner.EXIT_RUNNER_ERROR, run_result("completed", [VALUE, VALUE]), id="completed behind exit 2"),
    pytest.param(runner.EXIT_TIMEOUT, run_result("completed", [VALUE, VALUE]), id="completed behind exit 3"),
    pytest.param(
        runner.EXIT_COMPLETED, run_result("source_error", [], code="syntax_error"), id="source_error behind exit 0"
    ),
    pytest.param(
        runner.EXIT_TIMEOUT, run_result("source_error", [], code="syntax_error"), id="source_error behind exit 3"
    ),
    pytest.param(runner.EXIT_COMPLETED, run_result("timeout", [VALUE], timed_out=1), id="timeout behind exit 0"),
    pytest.param(
        runner.EXIT_COMPLETED, run_result("runner_error", [], code="limits"), id="runner_error behind exit 0"
    ),
]


def test_status_exit_pairing_mirrors_the_runner_protocol():
    assert execution._STATUS_EXIT_CODES == runner._STATUS_EXIT_CODES
    assert set(execution._STATUS_EXIT_CODES) == set(runner.RUN_STATUSES) >= set(runner.COMPARE_STATUSES)
    assert set(execution._STATUS_EXIT_CODES.values()) == runner.RESULT_EXIT_CODES
    # One exit per status.
    assert len(set(execution._STATUS_EXIT_CODES.values())) == len(execution._STATUS_EXIT_CODES)
    assert runner.RESULT_EXIT_CODES.isdisjoint(runner.NO_RESULT_EXIT_CODES)
    assert runner.EXIT_USAGE not in runner.RESULT_EXIT_CODES


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (exited(0), 0),
        (exited(runner.EXIT_TIMEOUT), runner.EXIT_TIMEOUT),
        (exited(137), 137),  # reported, but outside the result set: the domain refuses it
        (exited(-9), -9),
        (exited(True), None),  # not an exit, although True == EXIT_SOURCE_ERROR
        (exited(False), None),
        (exited(None), None),
        (exited("0"), None),
        (exited(0.0), None),
        (exited(124, error_type="timeout"), None),
        (exited(125, error_type="sandbox"), None),
        (exited(0, error_type="sandbox"), None),  # a set error_type is never an exit, whatever the code says
        (None, None),
        (SimpleNamespace(error_type=None), None),  # no return_code at all
    ],
)
def test_reported_exit_is_an_exact_integer_process_exit_or_nothing(report, expected):
    assert _reported_exit(report) == expected


def test_correlate_result_requires_the_paired_exit_and_the_job_nonce():
    nonce = "a" * 32
    for status, exit_code in execution._STATUS_EXIT_CODES.items():
        code = "syntax_error" if status == "source_error" else "limits" if status == "runner_error" else None
        result = parse(mutate(run_result(status, [], code=code), nonce=nonce))
        assert _correlate_result(result, exit_code, nonce) is None
        for other in sorted(runner.RESULT_EXIT_CODES - {exit_code}):
            with pytest.raises(_Unscorable) as info:
                _correlate_result(result, other, nonce)
            assert info.value.category == "result_invalid"
        with pytest.raises(_Unscorable):
            _correlate_result(result, exit_code, "b" * 32)
        with pytest.raises(_Unscorable):
            _correlate_result(result, bool(exit_code), nonce)  # True == 1 must never stand in for exit 1
    # A runner that could not read its job writes an empty nonce: shape-valid, but uncorrelated with this job.
    orphan = parse(run_result("runner_error", [], code="invalid_job"))
    assert orphan["nonce"] == ""
    with pytest.raises(_Unscorable) as info:
        _correlate_result(orphan, runner.EXIT_RUNNER_ERROR, nonce)
    assert info.value.category == "result_invalid"
    compared = parse(mutate(compare_result(["equal"]), nonce=nonce), "compare")
    assert _correlate_result(compared, runner.EXIT_COMPLETED, nonce) is None
    with pytest.raises(_Unscorable):
        _correlate_result(compared, runner.EXIT_TIMEOUT, nonce)


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("report", NON_RESULT_EXITS)
async def test_non_result_exits_never_read_the_file_and_attribute_nothing(domain, report):
    script = happy_script()  # every scripted file would pass: none may be read behind these exits
    script[("exec", domain)] = report
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    if domain == "candidate" and report.return_code in runner.WORKER_ABORT_EXIT_CODES:
        # A runner-authenticated candidate worker abort is a scored 0 fault, not a no-result. The file is
        # still never read. Pinned in test_candidate_worker_abort_scores_zero_and_is_not_recoverable.
        assert (result.outcome, result.reward, result.category) == ("failed", 0.0, "candidate_worker_abort")
        assert result.failure_class is None and result.terminal is None
    else:
        assert (result.outcome, result.reward, result.category) == ("unscorable", None, f"{domain}_no_result")
        assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result_reads(provider, domain) == 0 and result.cleanup[domain] == "deleted"
    if domain != "candidate":
        assert "candidate" not in created_domains(provider) and result.cleanup["candidate"] == "not_created"


@pytest.mark.parametrize(
    "code", [runner.EXIT_WORKER_ABORTED, runner.EXIT_PROTOCOL_VIOLATION], ids=["worker_aborted", "protocol_violation"]
)
async def test_candidate_worker_abort_scores_zero_and_is_not_recoverable(code):
    # After a completed preflight the candidate suite exit is a runner-authenticated abort: the supervisor ran
    # to completion and reported the worker died or broke the protocol. That is a candidate fault scored 0 with
    # the category preserved, not a provider-ambiguous no-result. No result file is read, no case is attempted,
    # and the row is a real grade (scored, no failure class), so reverify never re-selects it.
    script = happy_script()
    script[("exec", "candidate")] = exited(code)
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, "candidate_worker_abort")
    assert result.failure_class is None and result.terminal is None
    assert result.cases_attempted == 0 and result.cases_equal == 0 and result.first_failed_case is None
    assert "candidate_worker_abort" in execution.CANDIDATE_FAULT_CATEGORIES
    assert result_reads(provider, "candidate") == 0  # the file behind a no-result exit is never read
    assert_job_cleaned(result, provider, *DOMAINS)


async def test_reference_and_comparator_worker_abort_stay_unscorable():
    # Only the candidate domain attributes a worker abort to the candidate. A reference or comparator abort has
    # no candidate to blame, so it stays a provider-ambiguous no-result, unscorable and non-terminal.
    for domain in ("reference", "comparator"):
        script = happy_script()
        script[("exec", domain)] = exited(runner.EXIT_WORKER_ABORTED)
        grader, _ = make_grader(script)
        result = await grader.grade(make_task(), CANDIDATE)
        assert (result.outcome, result.reward, result.category) == ("unscorable", None, f"{domain}_no_result")
        assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False


@pytest.mark.parametrize("report", NO_EXIT_REPORTS)
async def test_reports_without_a_process_exit_are_provider_failures(report):
    script = happy_script()
    script[("exec", "candidate")] = report
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "provider_exec")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result_reads(provider, "candidate") == 0
    assert result.cleanup == {domain: "deleted" for domain in DOMAINS}


@pytest.mark.parametrize(("exit_code", "payload"), MISPAIRED)
async def test_mispaired_exit_and_status_is_result_invalid_for_the_candidate(exit_code, payload):
    script = happy_script()
    script[("exec", "candidate")] = exited(exit_code)
    script[("result", "candidate")] = payload
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category, result.terminal) == (
        "unscorable",
        None,
        "result_invalid",
        False,
    )
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE
    assert result_reads(provider, "candidate") == 1  # read behind a result exit, refused after the shape check
    assert result.cleanup == {domain: "deleted" for domain in DOMAINS}


@pytest.mark.parametrize(("exit_code", "payload"), MISPAIRED)
async def test_mispaired_exit_and_status_is_result_invalid_for_the_reference(exit_code, payload):
    # Never a terminal reference defect: the file and the exit disagree, so neither says anything about the task.
    script = happy_script()
    script[("exec", "reference")] = exited(exit_code)
    script[("result", "reference")] = payload
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category, result.terminal) == (
        "unscorable",
        None,
        "result_invalid",
        False,
    )
    assert "candidate" not in created_domains(provider) and result.cleanup["candidate"] == "not_created"


@pytest.mark.parametrize(
    ("exit_code", "payload", "outcome", "category", "reward"),
    [
        (runner.EXIT_COMPLETED, run_result("completed", [VALUE, VALUE]), "passed", "passed", 1.0),
        (
            runner.EXIT_SOURCE_ERROR,
            run_result("source_error", [], code="syntax_error"),
            "failed",
            "candidate_source_error",
            0.0,
        ),
        (runner.EXIT_TIMEOUT, run_result("timeout", [VALUE], timed_out=1), "failed", "candidate_timeout", 0.0),
        (runner.EXIT_RUNNER_ERROR, run_result("runner_error", [], code="limits"), "unscorable", "provider_exec", None),
    ],
    ids=["completed", "source_error", "timeout", "runner_error"],
)
async def test_paired_exit_and_status_reach_the_candidate_mapping(exit_code, payload, outcome, category, reward):
    # Passing the correlation gate does not authenticate a source, timeout or runtime-error report. After a
    # completed preflight a candidate source error or suite timeout scores 0 with the category preserved. A
    # runtime error is unattributed provider uncertainty.
    script = happy_script()
    script[("exec", "candidate")] = exited(exit_code)
    script[("result", "candidate")] = payload
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.category, result.reward) == (outcome, category, reward)
    assert result.cases_attempted == len(payload["outcomes"]) and result.first_failed_case is None
    if outcome == "passed":
        assert result.failure_class is None and result.terminal is None and result.cases_equal == 2
        assert result_reads(provider, "comparator") == 2  # preflight and candidate judge both ran
    elif outcome == "failed":
        # A whole-run candidate fault scores 0 without a candidate value comparison ever running.
        assert result.failure_class is None and result.terminal is None and result.cases_equal == 0
        assert result_reads(provider, "comparator") == 1
    else:
        assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
        assert result.cases_equal == 0 and result_reads(provider, "comparator") == 1
    assert result_reads(provider, "candidate") == 1
    assert_job_cleaned(result, provider, *DOMAINS)


@pytest.mark.parametrize(
    ("exit_code", "payload", "category"),
    [
        (runner.EXIT_SOURCE_ERROR, run_result("source_error", [], code="import_error"), "reference_error"),
        (runner.EXIT_TIMEOUT, run_result("timeout", [VALUE], timed_out=1), "reference_timeout"),
        (runner.EXIT_RUNNER_ERROR, run_result("runner_error", [], code="limits"), "reference_error"),
    ],
    ids=["source_error", "timeout", "runner_error"],
)
async def test_paired_exit_and_status_reach_the_reference_mapping(exit_code, payload, category):
    # Even an accepted exit and the correct job nonce cannot attribute the report to a reference defect.
    script = happy_script()
    script[("exec", "reference")] = exited(exit_code)
    script[("result", "reference")] = payload
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, category)
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cases_attempted == result.cases_equal == 0 and result.first_failed_case is None
    assert result_reads(provider, "reference") == 1 and "candidate" not in created_domains(provider)
    assert_job_cleaned(result, provider, "reference")


async def test_comparator_no_result_exit_after_a_completed_candidate_scores_nothing():
    script = happy_script()
    script[("exec", "comparator")] = [exited(runner.EXIT_COMPLETED), exited(runner.EXIT_WORKER_ABORTED)]
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "comparator_no_result")
    assert result.cases_attempted == 2 and result.cases_equal == 0
    assert result_reads(provider, "comparator") == 1  # the preflight verdicts only
    assert result.cleanup == {domain: "deleted" for domain in DOMAINS}


async def test_invocation_nonces_are_distinct_and_match_uploaded_jobs(monkeypatch):
    nonces = [character * 32 for character in "abcd"]
    token = Mock(side_effect=nonces)
    monkeypatch.setattr(execution.secrets, "token_hex", token)
    original_run = execution._run_domain
    seen = []

    async def recording_run(domain, mode, deadline_s, nonce):
        job = json.loads(provider.sandboxes[domain.sandbox_id]["files"][f"{domain.workdir}/job.json"])
        assert job["nonce"] == nonce and job["mode"] == mode
        seen.append((domain.name, job.get("observed_role"), nonce))
        result = await original_run(domain, mode, deadline_s, nonce)
        assert result["nonce"] == nonce
        return result

    monkeypatch.setattr(execution, "_run_domain", recording_run)
    grader, provider = make_grader(happy_script())
    result = await grader.grade(make_task(), CANDIDATE, job_id="12345678")
    assert result.reward == 1.0
    assert [invocation.args for invocation in token.call_args_list] == [(16,)] * 4
    assert seen == [
        ("reference", None, nonces[0]),
        ("comparator", "reference", nonces[1]),
        ("candidate", None, nonces[2]),
        ("comparator", "candidate", nonces[3]),
    ]
    assert len({nonce for _, _, nonce in seen}) == 4
    assert_job_cleaned(result, provider, *DOMAINS)


@pytest.mark.parametrize(("origin", "target"), [("reference", "candidate"), ("comparator", "comparator")])
async def test_stale_download_bytes_cannot_replay_across_invocations(monkeypatch, origin, target):
    nonces = [character * 32 for character in "abcd"]  # reference, preflight, candidate and judge invocations
    monkeypatch.setattr(execution.secrets, "token_hex", Mock(side_effect=nonces))
    script = happy_script()
    # The bytes the origin invocation read: its scripted payload behind its own nonce, as the provider wrote them.
    origin_index = 0 if origin == "reference" else 1
    stale = json.dumps(dict(script[("result", origin)][0], nonce=nonces[origin_index])).encode()
    if target == "comparator":
        script[("result", target)][1] = stale
    else:
        script[("result", target)] = stale  # Bytes, not a dict: the fake must not substitute the current nonce.
    correlate = Mock(wraps=execution._correlate_result)
    monkeypatch.setattr(execution, "_correlate_result", correlate)
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE, job_id="12345678")
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "result_invalid")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    accepted = correlate.call_args_list[origin_index].args  # the origin invocation accepted exactly these bytes
    assert accepted[0] == json.loads(stale) and accepted[2] == nonces[origin_index]
    payload, exit_code, nonce = correlate.call_args.args
    assert payload["nonce"] == json.loads(stale)["nonce"] != nonce
    assert nonce == domain_job(provider, target)["nonce"] and exit_code == runner.EXIT_COMPLETED
    assert result.cases_equal == 0 and result.first_failed_case is None
    assert_job_cleaned(result, provider, *DOMAINS)


@pytest.mark.parametrize("target", ["reference", "preflight", "candidate", "comparison"])
async def test_previous_candidate_download_bytes_cannot_replay_in_a_new_job(monkeypatch, target):
    grader, provider = make_grader(happy_script())
    assert (await grader.grade(make_task(), CANDIDATE, job_id="12345678")).reward == 1.0
    domain = "comparator" if target in {"preflight", "comparison"} else target
    origin = "comparator" if domain == "comparator" else "candidate"
    # The bytes the previous job's last read in the origin domain returned: its scripted payload behind its nonce.
    last_read = (
        compare_result(["equal", "equal"]) if origin == "comparator" else run_result("completed", [VALUE, VALUE])
    )
    stale = json.dumps(dict(last_read, nonce=domain_job(provider, origin)["nonce"])).encode()
    script = happy_script()
    if domain == "comparator":
        script[("result", domain)][0 if target == "preflight" else 1] = stale
    else:
        script[("result", domain)] = stale
    correlate = Mock(wraps=execution._correlate_result)
    monkeypatch.setattr(execution, "_correlate_result", correlate)
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE, job_id="87654321")
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "result_invalid")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    payload, exit_code, nonce = correlate.call_args.args
    assert payload["nonce"] == json.loads(stale)["nonce"] != nonce
    assert nonce == domain_job(provider, domain)["nonce"] and exit_code == runner.EXIT_COMPLETED
    created = ("reference",) if target == "reference" else ("reference", "comparator")
    assert_job_cleaned(result, provider, *(DOMAINS if target in {"candidate", "comparison"} else created))


async def test_result_must_echo_the_job_nonce():
    # A shape-valid passing file that echoes another nonce is uncorrelated with this invocation and never scored.
    script = happy_script()
    script[("result", "candidate")] = json.dumps(
        dict(run_result("completed", [VALUE, VALUE]), nonce="f" * 32)
    ).encode()
    grader, _ = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category, result.terminal) == (
        "unscorable",
        None,
        "result_invalid",
        False,
    )
    # The runner writes an empty nonce when it cannot read its job: shape-valid, uncorrelated, and never a terminal
    # reference defect.
    script = happy_script()
    script[("exec", "reference")] = exited(runner.EXIT_RUNNER_ERROR)
    script[("result", "reference")] = json.dumps(run_result("runner_error", [], code="invalid_job")).encode()
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.category, result.terminal) == ("result_invalid", False) and "candidate" not in created_domains(
        provider
    )
    # Verdict files are correlated the same way.
    script = happy_script()
    script[("result", "comparator")] = [
        json.dumps(dict(compare_result(["equal", "equal"], side="reference"), nonce="f" * 32)).encode()
    ]
    grader, provider = make_grader(script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert result.category == "result_invalid" and "candidate" not in created_domains(provider)


async def test_cancellation_during_the_result_download_propagates_and_cleans_up():
    grader, provider = make_grader(happy_script())
    fail_downloads(provider, "candidate", cancel_self)  # the exit was accepted, and the read itself is cancelled
    with pytest.raises(asyncio.CancelledError):
        await grader.grade(make_task(), CANDIDATE)
    asyncio.current_task().uncancel()
    assert result_reads(provider, "candidate") == 1
    assert sorted(provider.closed) == sorted(provider.sandboxes) and len(provider.closed) == 3
    assert not any(box["live"] for box in provider.sandboxes.values()) and not stray_tasks()


# --- runner static helpers (synthetic sources, never executed) ---------------------------------- #


def test_entrypoint_binding_fails_closed():
    tree, _ = runner.parse_source("def f(x):\n    return x\n")
    assert runner.check_entrypoint_binding(tree, "f") is None
    assert runner.check_entrypoint_binding(tree, "g") == "entrypoint_missing"
    tree, _ = runner.parse_source("def f(x):\n    return x\n\ndef f(y):\n    return y\n")
    assert runner.check_entrypoint_binding(tree, "f") == "entrypoint_conflict"
    tree, _ = runner.parse_source("def f(x):\n    return x\nf = 3\n")
    assert runner.check_entrypoint_binding(tree, "f") == "entrypoint_conflict"
    tree, _ = runner.parse_source("def f(x):\n    return x\nfrom os import *\n")
    assert runner.check_entrypoint_binding(tree, "f") == "entrypoint_conflict"
    tree, _ = runner.parse_source("async def f(x):\n    return x\n")
    assert runner.check_entrypoint_binding(tree, "f") == "entrypoint_missing"
    assert runner.parse_source("def f(:\n") == (None, "syntax_error")


def test_walrus_local_to_the_entrypoint_body_is_not_a_conflict():
    # A walrus inside the entrypoint body binds in the function's own scope, so it does not rebind the
    # module-level entrypoint.
    tree, error = runner.parse_source("def f(x):\n    return (f := x + 1)\n")
    assert error is None and runner.check_entrypoint_binding(tree, "f") is None
    # A walrus that binds the entrypoint name inside a different top-level function's body is also local.
    tree, _ = runner.parse_source("def f(x):\n    return x\ndef g(x):\n    return (f := x)\n")
    assert runner.check_entrypoint_binding(tree, "f") is None
    # A walrus binding a name other than the entrypoint, inside the entrypoint body, is likewise allowed.
    tree, _ = runner.parse_source("def f(x):\n    return (y := x + 1)\n")
    assert runner.check_entrypoint_binding(tree, "f") is None


def test_module_scope_walrus_rebinding_the_entrypoint_stays_a_conflict():
    # A walrus evaluated at module scope still rebinds the entrypoint. A bare module-level expression, and a
    # default argument or decorator expression on a top-level def, are all evaluated in the enclosing scope.
    tree, error = runner.parse_source("def f(x):\n    return x\n(f := 1)\n")
    assert error is None and runner.check_entrypoint_binding(tree, "f") == "entrypoint_conflict"
    tree, _ = runner.parse_source("def f(x):\n    return x\ndef g(y=(f := 1)):\n    return y\n")
    assert runner.check_entrypoint_binding(tree, "f") == "entrypoint_conflict"
    tree, _ = runner.parse_source(
        "def deco(fn):\n    return fn\ndef f(x):\n    return x\n@(f := deco)\ndef g():\n    return 1\n"
    )
    assert runner.check_entrypoint_binding(tree, "f") == "entrypoint_conflict"


def test_run_job_validation_rejects_unknown_shapes():
    job = build_compare_job(make_task(), "candidate", [VALUE, VALUE], "abc", make_policy())
    assert runner.validate_job(job, "compare") is None
    assert runner.validate_job(dict(job, observed_role="judge"), "compare") == "invalid_job"
    assert runner.validate_job(dict(job, mode="run"), "run") == "invalid_job"


# --- task data: extra keys, bounds, complex carriers, input conversions ------------------------- #


def _delivered_task(**overrides):
    """A minimal valid task dict in the canonical shape. Overrides replace top-level fields."""
    base = {
        "problem_id": "p1",
        "reference_source": "def f(x):\n    return x\n",
        "entrypoint": "f",
        "test_cases": [{"args": [wire(["int", "1"])], "expected": value_outcome(["int", "1"])}],
    }
    base.update(overrides)
    return base


def test_build_run_job_always_carries_input_conversions():
    policy = make_policy()
    # A task that declares none still puts the key in the job as an empty list.
    assert build_run_job(make_task(), "f", "abcd", policy)["input_conversions"] == []
    # A task that declares conversions transports them verbatim, aligned with the positional inputs.
    task = TaskData.model_validate(_delivered_task(input_conversions=[None, "symbol", "function"]))
    assert build_run_job(task, "f", "abcd", policy)["input_conversions"] == [None, "symbol", "function"]


@pytest.mark.parametrize(
    "conversions", [[], [None], ["symbol"], ["function"], [None, "symbol", "function"], [None] * 128]
)
def test_task_data_accepts_valid_input_conversions(conversions):
    assert TaskData.model_validate(_delivered_task(input_conversions=conversions)).input_conversions == conversions


@pytest.mark.parametrize(
    "conversions", [["symbolic"], ["Symbol"], [1], [True], "symbol", {"0": "symbol"}, [None] * 129]
)
def test_task_data_rejects_bad_input_conversions(conversions):
    with pytest.raises(ValidationError):
        TaskData.model_validate(_delivered_task(input_conversions=conversions))


def test_task_data_input_conversions_default_is_none():
    assert TaskData.model_validate(_delivered_task()).input_conversions is None


@pytest.mark.parametrize("comparison_type", ["numeric", "", "x" * 64])
def test_keyword_shape_accepts_and_drops_comparison_type(comparison_type):
    task = TaskData.model_validate(
        _delivered_task(
            test_cases=[{"inputs": {"v": "1"}, "expected_output": "2", "comparison_type": comparison_type}]
        )
    )
    # The key is dropped, and the case normalizes to the keyword shape unchanged.
    case = task.test_cases[0]
    assert case.kwargs["v"].value == ["legacy", "1"] and case.expected.value.value == ["legacy", "2"]


def test_keyword_shape_comparison_type_may_be_null():
    task = TaskData.model_validate(
        _delivered_task(test_cases=[{"inputs": {"v": "1"}, "expected_output": "2", "comparison_type": None}])
    )
    assert task.test_cases[0].kwargs["v"].value == ["legacy", "1"]


@pytest.mark.parametrize("comparison_type", [123, ["numeric"], "x" * 65])
def test_keyword_shape_rejects_invalid_comparison_type(comparison_type):
    with pytest.raises(ValidationError):
        TaskData.model_validate(
            _delivered_task(
                test_cases=[{"inputs": {"v": "1"}, "expected_output": "2", "comparison_type": comparison_type}]
            )
        )


def test_comparison_type_is_only_accepted_on_the_keyword_shape():
    # The positional shape does not take the extra key. No other unknown key is accepted anywhere.
    with pytest.raises(ValidationError):
        TaskData.model_validate(
            _delivered_task(test_cases=[{"input": [1], "output": 1, "comparison_type": "numeric"}])
        )
    with pytest.raises(ValidationError):
        TaskData.model_validate(_delivered_task(test_cases=[{"inputs": {"v": "1"}, "expected_output": "2", "z": 1}]))


def test_complex_carrier_becomes_a_complex_wire_node_on_args_and_expected():
    task = TaskData.model_validate(
        _delivered_task(test_cases=[{"input": [{"__complex__": [1, 2.5]}], "output": {"__complex__": [3, 4]}}])
    )
    case = task.test_cases[0]
    assert case.args[0].value == ["complex", ["int", "1"], ["float", (2.5).hex()]]
    assert case.expected.value.value == ["complex", ["int", "3"], ["int", "4"]]


def test_complex_carrier_nests_through_lists_and_maps():
    task = TaskData.model_validate(
        _delivered_task(
            test_cases=[{"input": [[{"__complex__": [0, -1]}]], "output": {"z": {"__complex__": [1.0, 0.0]}}}]
        )
    )
    case = task.test_cases[0]
    assert case.args[0].value == ["list", [["complex", ["int", "0"], ["int", "-1"]]]]
    assert case.expected.value.value == ["map", [["z", ["complex", ["float", (1.0).hex()], ["float", (0.0).hex()]]]]]


@pytest.mark.parametrize(
    "carrier",
    [
        {"__complex__": [1, 2, 3]},  # not exactly two parts
        {"__complex__": [True, 2]},  # a bool part is not a real
        {"__complex__": [float("inf"), 2]},  # a nonfinite part
        {"__complex__": [1, 2], "x": 1},  # an extra key
        {"__complex__": "nope"},  # not a list
    ],
)
def test_non_complex_dicts_stay_plain_maps(carrier):
    task = TaskData.model_validate(_delivered_task(test_cases=[{"input": [carrier], "expected_error": "boom"}]))
    assert task.test_cases[0].args[0].value[0] == "map"


def test_large_value_of_the_delivered_shape_validates_with_margin():
    # A dict holding a list of pair-lists of small ints: ~12.5k nodes at depth 4, 12 such cases in one task.
    grid = {"grid": [[[i % 1000, (i * 7) % 1000] for _ in range(65)] for i in range(64)]}
    task = TaskData.model_validate(
        _delivered_task(test_cases=[{"inputs": {"n": "1"}, "expected_output": grid} for _ in range(12)])
    )
    assert len(task.test_cases) == 12


# --- runtime probe ------------------------------------------------------------------------------ #

# ScriptedProvider answers the size probe (``test -f ... && stat -c %s ...``) and the download from the
# ``("result", domain)`` entry: with this entry the probe's upload check finds a one-byte file and reads it back.
MARKER_FOUND = {("result", "probe"): b"x"}


def answer_exec(provider, prefix, answer):
    """Replace the provider's answer to every exec whose command starts with ``prefix``; the call is still recorded.

    ``answer`` is an exception to raise, a coroutine function to await, or ``(stdout, return_code, error_type)``.
    """
    from nemo_gym.sandbox import SandboxExecResult

    scripted = provider.exec

    async def patched_exec(handle, command, **kwargs):
        result = await scripted(handle, command, **kwargs)
        if not command.startswith(prefix):
            return result
        if isinstance(answer, BaseException):
            raise answer
        if callable(answer):
            return await answer()
        stdout, code, error_type = answer
        return SandboxExecResult(stdout=stdout, stderr="", return_code=code, error_type=error_type)

    provider.exec = patched_exec
    return provider


def probe_warnings(caplog):
    """The WARNING messages the probe and its sandbox domain logged, in order."""
    names = (execution.__name__, sandbox_domain.__name__)
    return [
        record.getMessage() for record in caplog.records if record.name in names and record.levelno == logging.WARNING
    ]


@pytest.mark.asyncio
async def test_probe_passes_and_reports_the_uid():
    provider = ScriptedProvider({"uid": 1000, **MARKER_FOUND})
    info = await execution.probe_runtime(provider, make_spec(), make_policy())
    assert info["uid"] == "1000" and info["interpreter"] == "/usr/local/bin/python3"
    commands = [call["command"] for call in provider.exec_calls]
    assert commands[0].startswith("mkdir -p") and commands[1] == "id -u"
    assert commands[2] == '"/usr/local/bin/python3" -I -B -c "import pydantic, sympy, numpy, scipy"'
    assert all(call["user"] == "grader" for call in provider.exec_calls)
    assert provider.closed == ["fake-probe-1"]  # the probe sandbox is stopped


@pytest.mark.asyncio
async def test_probe_refuses_root_missing_dependency_and_create_failure():
    with pytest.raises(execution.RuntimeProbeError) as info:
        await execution.probe_runtime(ScriptedProvider({"uid": 0}), make_spec(), make_policy())
    assert info.value.category == "exec_user_is_root"
    with pytest.raises(execution.RuntimeProbeError) as info:
        await execution.probe_runtime(ScriptedProvider({"imports_ok": False}), make_spec(), make_policy())
    assert info.value.category == "interpreter_missing_dependency"
    from nemo_gym.sandbox import SandboxCreateError

    with pytest.raises(execution.RuntimeProbeError) as info:
        await execution.probe_runtime(
            ScriptedProvider({("create", "probe"): SandboxCreateError("quota")}), make_spec(), make_policy()
        )
    assert info.value.category == "provider_create"


@pytest.mark.asyncio
async def test_probe_refuses_when_provider_cannot_switch_user():
    class RefusingProvider(ScriptedProvider):
        async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
            if user is not None:
                raise ValueError("The local sandbox provider cannot run commands as another user")
            return await super().exec(handle, command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

    with pytest.raises(execution.RuntimeProbeError) as info:
        await execution.probe_runtime(RefusingProvider({}), make_spec(), make_policy(exec_user="grader"))
    assert info.value.category == "exec_user_unsupported"
    info = await execution.probe_runtime(
        RefusingProvider({"uid": 1000, **MARKER_FOUND}), make_spec(), make_policy(exec_user=None)
    )
    assert info["uid"] == "1000"


@pytest.mark.asyncio
async def test_probe_uploads_a_nested_marker_and_sizes_it_with_the_domain_size_probe():
    provider = ScriptedProvider({"uid": 1000, **MARKER_FOUND})
    downloads = []
    download = provider.download_file

    async def recorded_download(handle, source_path, target_path):
        downloads.append(source_path)
        await download(handle, source_path, target_path)

    provider.download_file = recorded_download
    info = await execution.probe_runtime(provider, make_spec(), make_policy())
    assert info == {
        "uid": "1000",
        "interpreter": "/usr/local/bin/python3",
        "sandbox_id": "fake-probe-1",
        "upload": "ok",
    }
    workdir = provider.exec_calls[1]["cwd"]
    marker = f"{workdir}/probe/nested/marker"
    files = provider.sandboxes["fake-probe-1"]["files"]
    assert workdir.startswith("/work/probe") and list(files) == [marker] and len(files[marker]) == 1
    # The probe's own size probe, then the domain's bounded download: its size probe and exactly one download.
    assert len(provider.exec_calls) == 5 and downloads == [marker]
    size_probe = provider.exec_calls[3]
    assert size_probe["command"] == f'test -f "{marker}" && stat -c %s "{marker}"'
    assert size_probe["cwd"] == workdir and size_probe["user"] == "grader"
    assert provider.exec_calls[4] == size_probe  # the download path sizes the file with the very same call
    # Exactly the call a grading domain makes before it reads a result file, on the same path.
    domain = sandbox_domain.SandboxDomain("probe", make_policy(), provider, make_spec(), job_id="drift1")
    await domain.create()
    await domain._download_bounded(marker)
    keys = ("command", "timeout_s", "user")
    assert [provider.exec_calls[-1][key] for key in keys] == [size_probe[key] for key in keys]
    await domain.close()


@pytest.mark.asyncio
async def test_probe_refuses_a_provider_whose_nested_upload_fails(caplog):
    provider = ScriptedProvider({"uid": 1000, **MARKER_FOUND})

    async def no_parent_directories(handle, source_path, target_path):
        raise FileNotFoundError(target_path)

    provider.upload_file = no_parent_directories
    with caplog.at_level(logging.WARNING), pytest.raises(execution.RuntimeProbeError) as info:
        await execution.probe_runtime(provider, make_spec(), make_policy())
    assert info.value.category == "upload_unsupported" and provider.closed == ["fake-probe-1"]
    assert not any(call["command"].startswith("test -f") for call in provider.exec_calls)
    assert probe_warnings(caplog) == [
        "sandbox domain probe: transfer failed (FileNotFoundError)",
        "runtime probe: upload failed (Unscorable)",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "script, logged",
    [
        (  # the size probe finds it; the transfer fails
            {("download", "probe"): RuntimeError("provider text 7f3a")},
            ["sandbox domain probe: download failed (RuntimeError)", "runtime probe: download failed (Unscorable)"],
        ),
        ({("result", "probe"): b"y"}, []),  # one byte, but not the one uploaded
    ],
    ids=["download_raises", "other_bytes"],
)
async def test_probe_refuses_a_marker_the_bounded_download_does_not_return(caplog, script, logged):
    provider = ScriptedProvider({"uid": 1000, **MARKER_FOUND, **script})
    with caplog.at_level(logging.WARNING), pytest.raises(execution.RuntimeProbeError) as info:
        await execution.probe_runtime(provider, make_spec(), make_policy())
    assert info.value.category == "upload_unsupported" and "7f3a" not in str(info.value)
    assert provider.closed == ["fake-probe-1"]
    assert probe_warnings(caplog) == logged and "7f3a" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        ("", 1, None),  # the file is not where the domain looks
        ("1\n", 0, "sandbox"),  # no process exit
        ("1 bytes\n", 0, None),  # decorated output the domain's parser refuses
        ("motd\n1\n", 0, None),
        ("\x1b[0m1\n", 0, None),
        ("\u00b9\n", 0, None),  # a digit, not an ASCII one
        ("9" * 21 + "\n", 0, None),  # longer than any 64-bit size
        ("", 0, None),
        RuntimeError("provider text 7f3a"),
        hang,  # bounded on the host
    ],
)
async def test_probe_refuses_a_size_probe_the_domain_would_not_trust(monkeypatch, caplog, answer):
    monkeypatch.setattr(execution, "TRANSFER_GRACE_S", 0.0)
    provider = answer_exec(ScriptedProvider({"uid": 1000, **MARKER_FOUND}), "test -f", answer)
    with caplog.at_level(logging.WARNING), pytest.raises(execution.RuntimeProbeError) as info:
        await execution.probe_runtime(provider, make_spec(), make_policy(transfer_timeout_s=0.05))
    assert info.value.category == "upload_unsupported" and "7f3a" not in str(info.value)
    assert provider.closed == ["fake-probe-1"]
    raised = "TimeoutError" if answer is hang else type(answer).__name__ if isinstance(answer, BaseException) else None
    assert probe_warnings(caplog) == ([f"runtime probe: size probe failed ({raised})"] if raised else [])
    assert "7f3a" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prefix, answer, category",
    [
        ("id -u", RuntimeError("provider text 7f3a"), "provider_exec"),
        ("id -u", ValueError("provider text 7f3a"), "exec_user_unsupported"),  # a refused user=
        ("id -u", NotImplementedError("provider text 7f3a"), "exec_user_unsupported"),
        ("id -u", ("uid=1000\n", 0, None), "exec_user_unsupported"),
        ("id -u", ("\u00b2\n", 0, None), "exec_user_unsupported"),  # str.isdigit() accepts it, int() does not
        ('"/usr/local/bin/python3" -I -B -c', RuntimeError("provider text 7f3a"), "provider_exec"),
        ('"/usr/local/bin/python3" -I -B -c', ("", 0, "sandbox"), "interpreter_missing_dependency"),
    ],
)
async def test_probe_maps_provider_answers_to_fixed_categories(caplog, prefix, answer, category):
    provider = answer_exec(ScriptedProvider({"uid": 1000, **MARKER_FOUND}), prefix, answer)
    with caplog.at_level(logging.WARNING), pytest.raises(execution.RuntimeProbeError) as info:
        await execution.probe_runtime(provider, make_spec(), make_policy())
    assert info.value.category == category and "7f3a" not in str(info.value)
    assert info.value.__cause__ is None  # provider text never leaves the probe, not even as a chained cause
    assert provider.closed == ["fake-probe-1"]
    stage = "id -u" if prefix == "id -u" else "import check"
    raised = type(answer).__name__ if isinstance(answer, BaseException) else None
    assert probe_warnings(caplog) == ([f"runtime probe: {stage} failed ({raised})"] if raised else [])
    assert "7f3a" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", [ValueError, NotImplementedError])
async def test_probe_without_a_requested_user_maps_a_refusal_class_to_provider_exec(caplog, refusal):
    provider = answer_exec(ScriptedProvider({"uid": 1000, **MARKER_FOUND}), "id -u", refusal("provider text 7f3a"))
    with caplog.at_level(logging.WARNING), pytest.raises(execution.RuntimeProbeError) as info:
        await execution.probe_runtime(provider, make_spec(), make_policy(exec_user=None))
    assert info.value.category == "provider_exec" and info.value.__cause__ is None
    assert probe_warnings(caplog) == [f"runtime probe: id -u failed ({refusal.__name__})"]
    assert "7f3a" not in caplog.text and provider.closed == ["fake-probe-1"]


# --- exec user refusal during a grade ----------------------------------------------------------- #


def refuse_first_comparator_setup(provider):
    """The comparator's first setup exec raises ValueError, as a provider that cannot run as ``exec_user`` does."""
    scripted = provider.exec
    refusals = [ValueError("cannot run commands as another user")]

    async def patched_exec(handle, command, **kwargs):
        if (
            refusals
            and provider.sandboxes[handle.sandbox_id]["domain"] == "comparator"
            and command.startswith("mkdir -p")
        ):
            raise refusals.pop()
        return await scripted(handle, command, **kwargs)

    provider.exec = patched_exec
    return provider


@pytest.mark.asyncio
async def test_exec_user_unsupported_during_a_grade_is_retryable():
    grader, provider = make_grader(happy_script())
    refuse_first_comparator_setup(provider)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.category, result.reward) == ("unscorable", "exec_user_unsupported", None)
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert not any(box["live"] for box in provider.sandboxes.values())


@pytest.mark.asyncio
async def test_a_failed_setup_whose_stop_failed_is_stopped_by_the_final_sweep():
    script = {**happy_script(), ("stop", "comparator"): [RuntimeError("synthetic stop failure")]}
    grader, provider = make_grader(script)
    refuse_first_comparator_setup(provider)  # the stop start_with_setup runs after the refusal fails once
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.category) == ("unscorable", "exec_user_unsupported")
    assert_job_cleaned(result, provider, "reference", "comparator")  # the sweep's retry stopped it: deleted, once


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [False, True])
async def test_reference_cache_stores_only_task_level_refusals(monkeypatch, terminal):
    from resources_servers.critpt_custom_grader.reference_cache import ReferenceCache

    if terminal:  # whether a refusal is cached does not follow the terminal flag
        monkeypatch.setitem(
            execution.UNSCORABLE_CATEGORIES, "exec_user_unsupported", (failure_kinds.VERIFIER_ERROR, True)
        )
    script = happy_script()
    script[("result", "reference")] = [run_result("completed", [VALUE, VALUE])] * 2
    grader, provider = make_grader(script, cache=ReferenceCache())
    refuse_first_comparator_setup(provider)
    first = await grader.grade(make_task(), CANDIDATE)
    assert (first.category, first.terminal) == ("exec_user_unsupported", terminal)
    assert created_domains(provider) == ["reference", "comparator"]
    second = await grader.grade(make_task(), CANDIDATE)
    assert second.category == "passed"  # nothing was cached: the reference and the preflight ran again
    assert created_domains(provider) == ["reference", "comparator", "reference", "comparator", "candidate"]
    assert execution.CACHED_REFUSALS == {"task_invalid", "reference_mismatch"}
