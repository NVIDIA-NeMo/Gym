# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os

import pytest

from resources_servers.critpt_custom_grader import reference_cache
from resources_servers.critpt_custom_grader.execution import Grader
from resources_servers.critpt_custom_grader.reference_cache import CachedReference, ReferenceCache
from resources_servers.critpt_custom_grader.runner import EXIT_COMPLETED, EXIT_TIMEOUT
from resources_servers.critpt_custom_grader.task_data import TaskData
from resources_servers.critpt_custom_grader.tests.support import (
    CANDIDATE,
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
    wire,
)


@pytest.fixture(autouse=True)
def fresh_warning_state(monkeypatch):
    # The write-failure warning is logged once per process; each test starts as if none had been logged yet.
    monkeypatch.setattr(reference_cache, "_persist_failure_logged", False, raising=False)


def other_task():
    return TaskData.model_validate(
        {
            "problem_id": "p2",
            "reference_source": "def f(x):\n    return x + 2\n",
            "entrypoint": "f",
            "test_cases": [{"args": [wire(["int", "1"])], "expected": value_outcome(["int", "3"])}],
        }
    )


def task_dict():
    return make_task().model_dump(mode="json")


def reversed_keys(node):
    if isinstance(node, dict):
        return {name: reversed_keys(value) for name, value in reversed(list(node.items()))}
    if isinstance(node, list):
        return [reversed_keys(item) for item in node]
    return node


PROVIDER = "pkg.Provider"


def key_of(task=None, policy=None, *, runtime="rt", image="img", rtol=None, atol=None, provider_class=PROVIDER):
    task = make_task() if task is None else task
    policy = make_policy() if policy is None else policy
    return ReferenceCache.key(task, policy, runtime, image, rtol, atol, provider_class=provider_class)


def test_key_covers_task_policy_runtime_image_provider_and_defaults():
    base = ReferenceCache.key(make_task(), make_policy(), "rt", "img", None, None, provider_class=PROVIDER)
    assert len(base) == 64
    assert base != ReferenceCache.key(other_task(), make_policy(), "rt", "img", None, None, provider_class=PROVIDER)
    assert base != ReferenceCache.key(
        make_task(), make_policy(suite_timeout_s=3600), "rt", "img", None, None, provider_class=PROVIDER
    )
    assert base != ReferenceCache.key(make_task(), make_policy(), "rt2", "img", None, None, provider_class=PROVIDER)
    assert base != ReferenceCache.key(make_task(), make_policy(), "rt", "img:2", None, None, provider_class=PROVIDER)
    assert base != ReferenceCache.key(make_task(), make_policy(), "rt", "img", None, None, provider_class="pkg.Other")
    assert base != ReferenceCache.key(make_task(), make_policy(), "rt", "img", "5e-12", "0", provider_class=PROVIDER)
    assert base == ReferenceCache.key(make_task(), make_policy(), "rt", "img", None, None, provider_class=PROVIDER)


def _change_expected_value(data):
    data["test_cases"][1]["expected"]["value"]["value"] = ["int", "4"]


def _change_reference_source(data):
    data["reference_source"] += "# edited\n"


def _add_task_leaf_tolerance(data):
    data["tolerances"] = [{"path": "root", "rtol": "1e-6"}]


def _add_case_leaf_tolerance(data):
    data["test_cases"][0]["tolerances"] = [{"path": "root", "atol": "1e-9"}]


def _change_statement(data):
    data["problem"] = "Report the result to six significant figures."


@pytest.mark.parametrize(
    "mutate",
    [
        _change_expected_value,
        _change_reference_source,
        _add_task_leaf_tolerance,
        _add_case_leaf_tolerance,
        _change_statement,
    ],
    ids=["expected_value", "reference_source", "task_leaf_tolerance", "case_leaf_tolerance", "statement"],
)
def test_each_task_field_changes_the_key(mutate):
    assert key_of(TaskData.model_validate(task_dict())) == key_of()  # the round trip alone changes nothing
    data = task_dict()
    mutate(data)
    assert key_of(TaskData.model_validate(data)) != key_of()


def test_key_does_not_depend_on_the_key_order_of_the_task_dict():
    reordered = reversed_keys(task_dict())
    assert list(reordered) != list(task_dict())  # the order really differs
    assert key_of(TaskData.model_validate(reordered)) == key_of()


KEYED_POLICY_FIELDS = [
    ("interpreter", "/usr/bin/python3"),
    ("exec_user", "other"),
    ("memory_limit_mib", 3072),
    ("cpu_time_limit_s", 1700),
    ("max_processes", 32),
    ("suite_timeout_s", 3600),
    ("compare_per_case_s", 30),
    ("compare_timeout_s", 900),
    ("exec_timeout_margin_s", 30.0),
]


@pytest.mark.parametrize(("name", "value"), KEYED_POLICY_FIELDS, ids=[name for name, _ in KEYED_POLICY_FIELDS])
def test_each_keyed_policy_field_changes_the_key_alone(name, value):
    assert key_of(policy=make_policy(**{name: value})) != key_of()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("create_timeout_s", 30.0),
        ("transfer_timeout_s", 30.0),
        ("cleanup_timeout_s", 30.0),
        ("max_concurrent_jobs", 2),
    ],
)
def test_policy_fields_that_do_not_reach_the_runner_leave_the_key_alone(name, value):
    changed = make_policy(**{name: value})
    assert getattr(changed, name) != getattr(make_policy(), name)  # the field really differs
    assert key_of(policy=changed) == key_of()


@pytest.mark.parametrize(("rtol", "atol"), [("5e-12", None), (None, "0")], ids=["rtol", "atol"])
def test_each_default_tolerance_changes_the_key_alone(rtol, atol):
    assert key_of(rtol=rtol, atol=atol) != key_of()


def test_default_tolerances_are_not_interchangeable():
    assert key_of(rtol="0") != key_of(atol="0")


def test_memory_cache_hit_miss_and_eviction():
    cache = ReferenceCache(max_entries=2)
    assert cache.get("a") is None
    cache.put("a", CachedReference(refusal=None))
    cache.put("b", CachedReference(refusal="task_invalid"))
    cache.put("c", CachedReference(refusal=None))
    assert cache.get("a") is None and cache.get("b").refusal == "task_invalid" and cache.get("c").refusal is None


def test_get_refreshes_recency():
    cache = ReferenceCache(max_entries=2)
    cache.put("a", CachedReference(refusal=None))
    cache.put("b", CachedReference(refusal=None))
    assert cache.get("a") is not None  # a is now the most recent
    cache.put("c", CachedReference(refusal=None))  # evicts b, the least recent
    assert cache.get("b") is None and cache.get("a") is not None and cache.get("c") is not None


def test_directory_cache_persists_atomically(tmp_path):
    cache = ReferenceCache(str(tmp_path / "refcache"))
    cache.put("k" * 64, CachedReference(refusal="reference_mismatch"))
    files = list((tmp_path / "refcache").iterdir())
    assert [f.name for f in files] == ["k" * 64 + ".json"] and oct(files[0].stat().st_mode)[-3:] == "600"
    assert json.loads(files[0].read_text()) == {"version": 1, "refusal": "reference_mismatch"}
    assert ReferenceCache(str(tmp_path / "refcache")).get("k" * 64).refusal == "reference_mismatch"
    (tmp_path / "refcache" / ("z" * 64 + ".json")).write_text("not json")
    assert ReferenceCache(str(tmp_path / "refcache")).get("z" * 64) is None  # a corrupt entry is a miss


def test_disk_entry_outlives_memory_eviction(tmp_path):
    cache = ReferenceCache(str(tmp_path / "refcache"), max_entries=1)
    cache.put("a" * 64, CachedReference(refusal="task_invalid"))
    cache.put("b" * 64, CachedReference(refusal=None))  # evicts a from memory only
    assert cache.get("a" * 64) == CachedReference(refusal="task_invalid")  # read back from disk


def test_a_passed_preflight_round_trips_through_disk(tmp_path):
    directory = tmp_path / "refcache"
    ReferenceCache(str(directory)).put("k" * 64, CachedReference(refusal=None))
    assert json.loads((directory / ("k" * 64 + ".json")).read_text()) == {"version": 1, "refusal": None}
    assert ReferenceCache(str(directory)).get("k" * 64) == CachedReference(refusal=None)  # a hit, not a miss


@pytest.mark.parametrize(
    "text",
    [
        pytest.param('{"version": 1}', id="no_refusal_key"),
        pytest.param('{"version": 2, "refusal": null}', id="other_version"),
        pytest.param('{"version": true, "refusal": null}', id="boolean_version"),
        pytest.param("[]", id="not_an_object"),
        pytest.param("null", id="null_document"),
        pytest.param("", id="empty_file"),
        pytest.param("[" * 100_000, id="nested_too_deep"),
    ],
)
def test_malformed_entries_are_misses(tmp_path, text):
    directory = tmp_path / "refcache"
    directory.mkdir()
    key = "m" * 64
    (directory / f"{key}.json").write_text(text)
    cache = ReferenceCache(str(directory))
    assert cache.get(key) is None
    cache.put(key, CachedReference(refusal="reference_mismatch"))  # the next write heals the entry
    assert ReferenceCache(str(directory)).get(key) == CachedReference(refusal="reference_mismatch")
    assert [f.name for f in directory.iterdir()] == [f"{key}.json"]


def test_put_failure_is_not_fatal(tmp_path, caplog):
    directory = tmp_path / "refcache"
    cache = ReferenceCache(str(directory))
    directory.rmdir()  # the directory disappears after construction
    with caplog.at_level(logging.WARNING, logger=reference_cache.__name__):
        cache.put("a" * 64, CachedReference(refusal="task_invalid"))  # does not raise
        cache.put("b" * 64, CachedReference(refusal=None))
    assert cache.get("a" * 64) == CachedReference(refusal="task_invalid")  # still served from memory
    assert cache.get("b" * 64) == CachedReference(refusal=None)
    records = [record for record in caplog.records if record.name == reference_cache.__name__]
    assert [record.levelno for record in records] == [logging.WARNING]  # once per process, not once per write
    assert "FileNotFoundError" in records[0].getMessage()
    assert str(tmp_path) not in records[0].getMessage()  # the exception class name only, no path


def test_failed_write_leaves_no_temporary_file(tmp_path, monkeypatch):
    directory = tmp_path / "refcache"
    cache = ReferenceCache(str(directory))

    def refuse(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", refuse)
    cache.put("k" * 64, CachedReference(refusal=None))  # best effort: does not raise
    assert list(directory.iterdir()) == []
    assert cache.get("k" * 64) == CachedReference(refusal=None)  # still served from memory


def test_only_oserror_is_swallowed_and_the_temporary_file_is_still_removed(tmp_path, monkeypatch):
    directory = tmp_path / "refcache"
    cache = ReferenceCache(str(directory))

    def broken(*_args, **_kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(RuntimeError):
        cache.put("k" * 64, CachedReference(refusal=None))
    assert list(directory.iterdir()) == []


@pytest.mark.asyncio
async def test_second_grade_of_the_same_task_skips_reference_and_preflight():
    script = {
        ("result", "reference"): [run_result("completed", [VALUE, VALUE])],
        ("result", "comparator"): [
            compare_result(["equal", "equal"], side="reference"),
            compare_result(["equal", "equal"]),
            compare_result(["equal", "equal"]),
        ],
        ("result", "candidate"): [run_result("completed", [VALUE, VALUE])] * 2,
    }
    grader, provider = make_grader(script, cache=ReferenceCache())
    first = await grader.grade(make_task(), CANDIDATE)
    second = await grader.grade(make_task(), CANDIDATE)
    assert first.category == second.category == "passed"
    assert len(runner_execs(provider, "reference")) == 1  # the reference ran once
    assert len(runner_execs(provider, "comparator")) == 3  # preflight once, judge twice
    assert second.cleanup == {"reference": "not_created", "comparator": "deleted", "candidate": "deleted"}


@pytest.mark.asyncio
async def test_terminal_refusals_are_cached_and_provider_faults_are_not():
    script = {
        ("result", "reference"): [run_result("completed", [VALUE, VALUE])] * 2,
        ("result", "comparator"): [compare_result(["invalid_reference", "equal"], side="reference")],
    }
    grader, provider = make_grader(script, cache=ReferenceCache())
    first = await grader.grade(make_task(), CANDIDATE)
    second = await grader.grade(make_task(), CANDIDATE)
    assert first.category == second.category == "reference_mismatch" and second.terminal is True
    assert len(runner_execs(provider, "reference")) == 1
    script = {
        **happy_script(),
        ("exec", "reference"): [exited(EXIT_TIMEOUT), exited(EXIT_COMPLETED)],
        ("result", "reference"): [run_result("timeout", [], timed_out=0), run_result("completed", [VALUE, VALUE])],
    }
    grader, provider = make_grader(script, cache=ReferenceCache())
    first = await grader.grade(make_task(), CANDIDATE)
    second = await grader.grade(make_task(), CANDIDATE)
    assert (first.category, first.terminal) == ("reference_timeout", False)
    assert second.category == "passed"
    assert len(runner_execs(provider, "reference")) == 2  # the timeout was not cached


class RecordingCache(ReferenceCache):
    """A memory cache that records every key the grader looks up."""

    def __init__(self):
        super().__init__()
        self.lookups = []

    def get(self, key):
        self.lookups.append(key)
        return super().get(key)


async def grader_key(**spec_fields):
    """The key the grader itself builds for make_task() under a sandbox spec with these fields."""
    cache = RecordingCache()
    grader, _ = make_grader(happy_script(), spec=make_spec(**spec_fields), cache=cache)
    await grader.grade(make_task(), CANDIDATE)
    (key,) = cache.lookups
    return key


OPERATOR_SPEC = {"files": {"/work/seed.txt": "seed"}, "env": {"SEED": "1"}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        {"files": {"/work/seed.txt": "seed2"}},
        {"files": {"/work/other.txt": "seed"}},
        {"files": {}},
        {"env": {"SEED": "2"}},
        {"env": {"OTHER": "1"}},
        {"env": {}},
    ],
    ids=["file_bytes", "file_name", "no_file", "env_value", "env_name", "no_env"],
)
async def test_an_operator_file_or_env_change_changes_the_key(changed):
    assert await grader_key(**{**OPERATOR_SPEC, **changed}) != await grader_key(**OPERATOR_SPEC)


@pytest.mark.asyncio
async def test_spec_fields_outside_the_key_leave_it_alone():
    base = await grader_key(**OPERATOR_SPEC)
    assert await grader_key(**OPERATOR_SPEC) == base  # the same spec, built again
    assert await grader_key(**OPERATOR_SPEC, ttl_s=900, metadata={"team": "x"}) == base


class OtherProvider(ScriptedProvider):
    """Another provider class over the same scripted plane, with the same ``name``: a second runtime."""


def disk_grader(directory, script, provider_class=ScriptedProvider, **spec_fields):
    """A grader over a new cache on ``directory``, as a later server process builds it."""
    provider = provider_class(script)
    cache = ReferenceCache(str(directory))
    return Grader(make_policy(), provider=provider, spec=make_spec(**spec_fields), reference_cache=cache), provider


def mismatch_script():
    return {
        ("result", "reference"): [run_result("completed", [VALUE, VALUE])],
        ("result", "comparator"): [compare_result(["invalid_reference", "equal"], side="reference")],
    }


def test_the_provider_is_named_by_its_class_not_its_name_attribute():
    named = reference_cache.provider_class_name
    assert OtherProvider.name == ScriptedProvider.name
    assert named(ScriptedProvider({})) == f"{ScriptedProvider.__module__}.ScriptedProvider"
    assert named(OtherProvider({})) != named(ScriptedProvider({}))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_class", "image"),
    [(OtherProvider, "example/grader:test"), (ScriptedProvider, "example/grader:other")],
    ids=["provider", "image"],
)
async def test_the_same_task_under_another_runtime_is_a_separate_persisted_entry(tmp_path, provider_class, image):
    grader, _ = disk_grader(tmp_path, happy_script())
    assert (await grader.grade(make_task(), CANDIDATE)).category == "passed"
    grader, provider = disk_grader(tmp_path, happy_script(), provider_class, image=image)
    assert (await grader.grade(make_task(), CANDIDATE)).category == "passed"
    assert len(runner_execs(provider, "reference")) == 1  # the first runtime's passed preflight was not read
    assert len(list(tmp_path.glob("*.json"))) == 2  # one persisted entry per runtime


@pytest.mark.asyncio
async def test_a_refusal_persisted_under_one_provider_is_never_read_under_another(tmp_path):
    grader, _ = disk_grader(tmp_path, mismatch_script())
    assert (await grader.grade(make_task(), CANDIDATE)).category == "reference_mismatch"
    grader, provider = disk_grader(tmp_path, happy_script(), OtherProvider)  # same task, image, policy and name
    result = await grader.grade(make_task(), CANDIDATE)
    assert result.category == "passed" and len(runner_execs(provider, "reference")) == 1
    grader, provider = disk_grader(tmp_path, {})  # the first provider again, in a later process
    assert (await grader.grade(make_task(), CANDIDATE)).category == "reference_mismatch"
    assert runner_execs(provider, "reference") == []  # its own entry was read: the reference did not run
