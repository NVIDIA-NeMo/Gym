# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import dataclasses
import json
import logging
from types import SimpleNamespace

import pytest

from nemo_gym.sandbox import SandboxCreateError, SandboxExecResult, SandboxResources, SandboxSpec
from resources_servers.critpt_custom_grader import sandbox_domain as sd
from resources_servers.critpt_custom_grader.runner import EXIT_COMPLETED, EXIT_WORKER_ABORTED, RESULT_BYTES_LIMIT
from resources_servers.critpt_custom_grader.tests.support import (
    MISSING_FILE,
    VALUE,
    ScriptedProvider,
    exited,
    make_spec,
    run_result,
)


def policy(**overrides):
    fields = {
        "interpreter": "/usr/local/bin/python3",
        "exec_user": "grader",
        "create_timeout_s": 5.0,
        "transfer_timeout_s": 5.0,
        "exec_timeout_margin_s": 20.0,
        "cleanup_timeout_s": 5.0,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def domain(provider, name="candidate", **overrides):
    return sd.SandboxDomain(name, policy(**overrides), provider, make_spec(), job_id="job1234")


async def hang():
    await asyncio.sleep(10)


@pytest.mark.asyncio
async def test_create_uses_the_base_workdir_and_makes_the_domain_dir_as_exec_user():
    provider = ScriptedProvider({})
    dom = domain(provider)
    await dom.create()
    spec = provider.created[0]
    assert (
        spec.workdir == "/work"
        and spec.metadata[sd.LABEL_DOMAIN] == "candidate"
        and spec.metadata[sd.LABEL_JOB] == "job1234"
    )
    assert any(key.endswith("THREADS") for key in spec.env)  # Gym's cpu cap env for cpu=1
    assert (
        dom.workdir == "/work/job1234-candidate" and dom.cleanup == "created" and dom.sandbox_id == "fake-candidate-1"
    )
    assert (
        provider.exec_calls[0]["command"] == 'mkdir -p "/work/job1234-candidate"'
        and provider.exec_calls[0]["user"] == "grader"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [SandboxCreateError("quota"), RuntimeError("sdk"), ValueError("decode")])
async def test_create_failure_is_provider_create(error):
    dom = domain(ScriptedProvider({("create", "candidate"): error}))
    with pytest.raises(sd.Unscorable) as info:
        await dom.create()
    assert info.value.category == "provider_create" and dom.cleanup == "not_created"


@pytest.mark.asyncio
async def test_upload_targets_the_domain_workdir():
    provider = ScriptedProvider({})
    dom = domain(provider)
    await dom.create()
    await dom.upload({"critpt_grader_rt/runner.py": b"print(1)\n", "job.json": b'{"nonce": "n"}'})
    files = provider.sandboxes["fake-candidate-1"]["files"]
    assert set(files) == {"/work/job1234-candidate/critpt_grader_rt/runner.py", "/work/job1234-candidate/job.json"}


@pytest.mark.asyncio
async def test_run_builds_the_fixed_command_without_runuser():
    provider = ScriptedProvider({("result", "candidate"): [run_result("completed", [VALUE])]})
    dom = domain(provider)
    await dom.create()
    await dom.upload({"job.json": b'{"nonce": "n1"}'})
    exit_code, data = await dom.run("run", 300)
    call = [c for c in provider.exec_calls if "runner.py" in c["command"]][0]
    assert call["command"] == (
        '"/usr/local/bin/python3" -I -B "/work/job1234-candidate/critpt_grader_rt/runner.py" run '
        '"/work/job1234-candidate/job.json" "/work/job1234-candidate/result.json" >/dev/null 2>&1'
    )
    assert "runuser" not in call["command"]
    assert call["cwd"] == "/work/job1234-candidate" and call["user"] == "grader" and call["timeout_s"] == 320
    assert call["env"]["HOME"] == "/work/job1234-candidate" and call["env"]["PYTHONHASHSEED"] == "0"
    assert exit_code == EXIT_COMPLETED and json.loads(data)["nonce"] == "n1"
    assert dom.last_exit == 0 and dom.last_error_type is None and dom.wall_time_s >= 0


@pytest.mark.asyncio
async def test_error_type_wins_over_return_code():
    dom = domain(ScriptedProvider({("exec", "candidate"): [exited(0, error_type="sandbox")]}))
    await dom.create()
    with pytest.raises(sd.Unscorable) as info:
        await dom.run("run", 300)
    assert info.value.category == "provider_exec" and dom.last_error_type == "sandbox"


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError("sdk"), RuntimeError("unreachable")])
async def test_raised_exec_error_is_provider_exec(error):
    dom = domain(ScriptedProvider({("exec", "candidate"): [error]}))
    await dom.create()
    with pytest.raises(sd.Unscorable) as info:
        await dom.run("run", 300)
    assert info.value.category == "provider_exec"


@pytest.mark.asyncio
async def test_unpaired_exit_is_no_result_and_candidate_worker_abort():
    # A valid result sits behind the unpaired exit: only the exit gate can refuse it, before any probe.
    provider = ScriptedProvider(
        {("exec", "reference"): [exited(64)], ("result", "reference"): [run_result("completed", [VALUE])]}
    )
    dom = domain(provider, name="reference")
    await dom.create()
    with pytest.raises(sd.Unscorable) as info:
        await dom.run("run", 300)
    assert info.value.category == "reference_no_result"
    assert not any(call["command"].startswith("test -f") for call in provider.exec_calls)
    dom = domain(ScriptedProvider({("exec", "candidate"): [exited(EXIT_WORKER_ABORTED)]}))
    await dom.create()
    with pytest.raises(sd.CandidateWorkerAbort):
        await dom.run("run", 300)


@pytest.mark.asyncio
async def test_oversized_result_is_transfer_limit():
    dom = domain(ScriptedProvider({("result", "candidate"): [b"x" * (RESULT_BYTES_LIMIT + 1)]}))
    await dom.create()
    with pytest.raises(sd.Unscorable) as info:
        await dom.run("run", 300)
    assert info.value.category == "transfer_limit"


@pytest.mark.asyncio
async def test_missing_result_and_download_exception_are_no_result():
    dom = domain(ScriptedProvider({("result", "candidate"): [MISSING_FILE]}))
    await dom.create()
    with pytest.raises(sd.Unscorable) as info:
        await dom.run("run", 300)
    assert info.value.category == "candidate_no_result"
    # The probe sees a result and reports its size; the transfer then fails: the download-raises branch.
    provider = ScriptedProvider(
        {
            ("result", "comparator"): [run_result("completed", [VALUE])],
            ("download", "comparator"): [RuntimeError("transport")],
        }
    )
    dom = domain(provider, name="comparator")
    await dom.create()
    with pytest.raises(sd.Unscorable) as info:
        await dom.run("compare", 60)
    assert info.value.category == "comparator_no_result"
    assert any(call["command"].startswith("test -f") for call in provider.exec_calls)
    assert (
        provider.script[("download", "comparator")] == []
    )  # the scripted failure was consumed: download() was reached


@pytest.mark.asyncio
async def test_close_statuses():
    dom = domain(ScriptedProvider({}))
    assert await dom.close() == "not_created"
    await dom.create()
    assert await dom.close() == "deleted"
    provider = ScriptedProvider({("stop", "candidate"): RuntimeError("stop failed")})
    dom = domain(provider)
    await dom.create()
    assert await dom.close() == "delete_failed" and dom.cleanup == "delete_failed"
    provider = ScriptedProvider({("stop", "candidate"): hang})
    dom = domain(provider, cleanup_timeout_s=0.05)
    await dom.create()
    assert await dom.close() == "delete_failed"


@pytest.mark.asyncio
async def test_cancellation_during_exec_propagates():
    async def cancel_self():
        asyncio.current_task().cancel()
        await asyncio.sleep(10)

    dom = domain(ScriptedProvider({("exec", "candidate"): [cancel_self]}))
    await dom.create()
    with pytest.raises(asyncio.CancelledError):
        await dom.run("run", 300)


@pytest.mark.asyncio
async def test_every_spec_field_reaches_the_provider():
    operator = SandboxSpec(
        image="example/grader:test",
        ttl_s=900,
        ready_timeout_s=45,
        workdir="/scratch/",
        env={"OMP_NUM_THREADS": "7", "FOO": "bar"},
        files={"/scratch/seed.txt": "seed"},
        metadata={"team": "x"},
        resources=SandboxResources(cpu=2, memory_mib=4096, disk_gib=10, gpu=1, gpu_type="a100"),
        entrypoint=["sleep", "infinity"],
        provider_options={"key": "value"},
        ports=(8080,),
    )
    provider = ScriptedProvider({})
    dom = sd.SandboxDomain("candidate", policy(), provider, operator, job_id="job1234")
    await dom.create()
    created = provider.created[0]
    handled = {"workdir", "env", "metadata", "provider_options"}
    for field in dataclasses.fields(SandboxSpec):
        if field.name not in handled:
            assert getattr(created, field.name) == getattr(operator, field.name), field.name
    assert created.workdir == "/scratch"  # the operator's base: the domain directory is made by the setup hook
    assert created.env["FOO"] == "bar" and created.env["OMP_NUM_THREADS"] == "7"  # the operator's entries win
    assert created.env["OPENBLAS_NUM_THREADS"] == "2"  # Gym's cpu cap env for cpu=2
    assert created.metadata == {"team": "x", sd.LABEL_JOB: "job1234", sd.LABEL_DOMAIN: "candidate"}
    assert created.provider_options == {"key": "value"} and created.provider_options is not operator.provider_options
    assert provider.sandboxes["fake-candidate-1"]["files"] == {"/scratch/seed.txt": b"seed"}  # uploaded at start
    assert operator.env == {"OMP_NUM_THREADS": "7", "FOO": "bar"} and operator.metadata == {"team": "x"}


@pytest.mark.asyncio
async def test_a_failed_run_does_not_keep_the_previous_runs_diagnostics():
    provider = ScriptedProvider(
        {
            ("exec", "comparator"): [exited(0), RuntimeError("unreachable")],
            ("result", "comparator"): [run_result("completed", [VALUE])],
        }
    )
    dom = domain(provider, name="comparator")
    await dom.create()
    await dom.run("compare", 60)
    assert dom.last_exit == 0
    with pytest.raises(sd.Unscorable) as info:
        await dom.run("compare", 60)
    assert info.value.category == "provider_exec" and dom.last_exit is None and dom.last_error_type is None


@pytest.mark.asyncio
async def test_a_timed_out_run_is_labelled_timeout():
    # The module's own wait_for expires first.
    dom = domain(ScriptedProvider({("exec", "candidate"): [hang]}), exec_timeout_margin_s=0.02)
    await dom.create()
    with pytest.raises(sd.Unscorable) as info:
        await dom.run("run", 0)
    assert info.value.category == "provider_exec" and dom.last_error_type == "timeout" and dom.last_exit is None
    # The provider raises its own TimeoutError.
    dom = domain(ScriptedProvider({("exec", "candidate"): [TimeoutError("sdk")]}))
    await dom.create()
    with pytest.raises(sd.Unscorable):
        await dom.run("run", 300)
    assert dom.last_error_type == "timeout"
    # Any other raised error carries no label.
    dom = domain(ScriptedProvider({("exec", "candidate"): [RuntimeError("unreachable")]}))
    await dom.create()
    with pytest.raises(sd.Unscorable):
        await dom.run("run", 300)
    assert dom.last_error_type is None


@pytest.mark.asyncio
async def test_concurrent_closes_reach_the_provider_once():
    stops = []

    async def slow_stop():
        stops.append(1)
        await asyncio.sleep(0.05)

    provider = ScriptedProvider({("stop", "candidate"): slow_stop})
    dom = domain(provider)
    await dom.create()
    assert await asyncio.gather(dom.close(), dom.close()) == ["deleted", "deleted"]
    assert stops == [1] and provider.closed == ["fake-candidate-1"]

    async def must_not_stop_again():
        raise AssertionError("a deleted domain must not reach the provider again")

    dom._sandbox.stop = must_not_stop_again
    assert await dom.close() == "deleted"


@pytest.mark.asyncio
async def test_a_failed_stop_is_logged_with_the_exception_class_only(caplog):
    dom = domain(ScriptedProvider({("stop", "candidate"): RuntimeError("provider text 7f3a")}))
    await dom.create()
    with caplog.at_level(logging.WARNING):
        assert await dom.close() == "delete_failed"
    records = [record for record in caplog.records if record.name == sd.__name__]
    assert len(records) == 1 and records[0].levelno == logging.WARNING and records[0].exc_info is None
    message = records[0].getMessage()
    assert "candidate" in message and "RuntimeError" in message and "7f3a" not in message


SECRET = "provider text 7f3a"


async def fail_create(error):
    await domain(ScriptedProvider({("create", "candidate"): error})).create()


async def fail_transfer(error):
    provider = ScriptedProvider({})

    async def refuse_upload(handle, source_path, target_path):
        raise error

    provider.upload_file = refuse_upload
    dom = domain(provider)
    await dom.create()
    await dom.upload({"job.json": b"{}"})


async def fail_run(error):
    dom = domain(ScriptedProvider({("exec", "candidate"): [error]}))
    await dom.create()
    await dom.run("run", 300)


async def fail_size_probe(error):
    provider = ScriptedProvider({("result", "candidate"): [run_result("completed", [VALUE])]})
    scripted = provider.exec

    async def refuse_size_probe(handle, command, **kwargs):
        if command.startswith("test -f"):
            raise error
        return await scripted(handle, command, **kwargs)

    provider.exec = refuse_size_probe
    dom = domain(provider)
    await dom.create()
    await dom.run("run", 300)


async def fail_download(error):
    dom = domain(
        ScriptedProvider(
            {("result", "candidate"): [run_result("completed", [VALUE])], ("download", "candidate"): [error]}
        )
    )
    await dom.create()
    await dom.run("run", 300)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage, fail, error",
    [
        ("create", fail_create, SandboxCreateError(SECRET)),
        ("transfer", fail_transfer, OSError(SECRET)),
        ("run", fail_run, TimeoutError(SECRET)),
        ("run", fail_run, ConnectionError(SECRET)),
        ("size probe", fail_size_probe, RuntimeError(SECRET)),
        ("download", fail_download, PermissionError(SECRET)),
    ],
    ids=["create", "transfer", "run_timeout", "run", "size_probe", "download"],
)
async def test_a_swallowed_provider_failure_is_logged_once_with_its_class_only(caplog, stage, fail, error):
    with caplog.at_level(logging.WARNING), pytest.raises(sd.Unscorable):
        await fail(error)
    records = [record for record in caplog.records if record.name == sd.__name__]
    assert [record.levelno for record in records] == [logging.WARNING] and records[0].exc_info is None
    message = records[0].getMessage()
    assert message == f"sandbox domain candidate: {stage} failed ({type(error).__name__})"
    assert "7f3a" not in message


@pytest.mark.asyncio
@pytest.mark.parametrize("size_text", ["not-a-number", "", "12abc", "1.5", "-5", "\u00b2", "9" * 21])
async def test_a_probe_that_is_not_a_plain_integer_is_no_result(size_text):
    provider = ScriptedProvider({("result", "candidate"): [run_result("completed", [VALUE])]})
    scripted_exec = provider.exec

    async def probe_says(handle, command, **kwargs):
        if command.startswith("test -f"):
            return SandboxExecResult(stdout=f"{size_text}\n", stderr="", return_code=0)
        return await scripted_exec(handle, command, **kwargs)

    provider.exec = probe_says
    dom = domain(provider)
    await dom.create()
    with pytest.raises(sd.Unscorable) as info:
        await dom.run("run", 300)
    assert info.value.category == "candidate_no_result"
    assert (
        len(provider.script[("result", "candidate")]) == 1
    )  # nothing trustworthy to download: nothing was downloaded


@pytest.mark.asyncio
async def test_a_download_that_leaves_no_file_is_no_result():
    provider = ScriptedProvider({("result", "candidate"): [run_result("completed", [VALUE])]})

    async def download_nothing(handle, source_path, target_path):
        return None

    provider.download_file = download_nothing
    dom = domain(provider)
    await dom.create()
    with pytest.raises(sd.Unscorable) as info:
        await dom.run("run", 300)
    assert info.value.category == "candidate_no_result"


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", [ValueError, NotImplementedError])
async def test_a_provider_that_cannot_run_as_the_exec_user_is_exec_user_unsupported(refusal):
    provider = ScriptedProvider({})
    scripted = provider.exec

    async def refuses_other_users(handle, command, *, user=None, **kwargs):
        if user is not None:
            raise refusal("cannot run commands as another user")
        return await scripted(handle, command, user=user, **kwargs)

    provider.exec = refuses_other_users
    dom = domain(provider)
    with pytest.raises(sd.Unscorable) as info:
        await dom.create()
    assert info.value.category == "exec_user_unsupported" and dom.cleanup == "not_created"
    assert provider.closed == ["fake-candidate-1"]  # start_with_setup stopped the sandbox
    dom = domain(provider, exec_user=None)  # the provider's own identity needs no switch
    await dom.create()
    assert dom.cleanup == "created"


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", [ValueError, NotImplementedError])
async def test_a_refusal_class_without_a_requested_user_is_provider_create(refusal):
    provider = ScriptedProvider({})

    async def refuses_every_command(handle, command, **kwargs):
        raise refusal("not supported")

    provider.exec = refuses_every_command
    dom = domain(provider, exec_user=None)  # no user was requested, so nothing can have been refused
    with pytest.raises(sd.Unscorable) as info:
        await dom.create()
    assert info.value.category == "provider_create" and dom.cleanup == "not_created"
    assert provider.closed == ["fake-candidate-1"]  # start_with_setup stopped the sandbox


SETUP_FAILURES = [(exited(1), "provider_create"), (ValueError("user"), "exec_user_unsupported")]


def fail_setup(provider, failure):
    """The setup hook's ``mkdir -p`` returns ``failure``, or raises it when it is an exception."""
    scripted = provider.exec

    async def setup_fails(handle, command, **kwargs):
        if not command.startswith("mkdir -p"):
            return await scripted(handle, command, **kwargs)
        if isinstance(failure, BaseException):
            raise failure
        return failure

    provider.exec = setup_fails


@pytest.mark.asyncio
@pytest.mark.parametrize("failure, category", SETUP_FAILURES)
async def test_a_failed_setup_whose_stop_also_fails_is_delete_failed(failure, category):
    stops = []

    async def failing_stop():
        stops.append(1)
        raise RuntimeError("stop failed")

    provider = ScriptedProvider({("stop", "candidate"): failing_stop})  # every stop fails
    fail_setup(provider, failure)
    dom = domain(provider)
    with pytest.raises(sd.Unscorable) as info:
        await dom.create()
    assert info.value.category == category and provider.closed == [] and stops == [1]
    assert dom.cleanup == "delete_failed" and dom.sandbox_id == "fake-candidate-1"
    assert await dom.close() == "delete_failed" and stops == [1, 1]  # close() retried the stop
    assert dom.sandbox_id == "fake-candidate-1" and provider.sandboxes["fake-candidate-1"]["live"]  # until the ttl


@pytest.mark.asyncio
@pytest.mark.parametrize("failure, category", SETUP_FAILURES)
async def test_close_retries_the_stop_of_a_failed_setup(failure, category):
    provider = ScriptedProvider({("stop", "candidate"): [RuntimeError("transient")]})  # only the first stop fails
    fail_setup(provider, failure)
    dom = domain(provider)
    with pytest.raises(sd.Unscorable) as info:
        await dom.create()
    assert info.value.category == category and provider.closed == []
    assert dom.cleanup == "delete_failed" and dom.sandbox_id == "fake-candidate-1"
    assert await asyncio.gather(dom.close(), dom.close()) == ["deleted", "deleted"]  # concurrent closers: one retry
    assert provider.closed == ["fake-candidate-1"] and dom.sandbox_id == "fake-candidate-1"
    assert await dom.close() == "deleted" and provider.closed == ["fake-candidate-1"]  # a deleted domain stays so


@pytest.mark.asyncio
async def test_a_cancelled_stop_of_a_failed_setup_is_retried_at_close():
    async def cancel_self():
        asyncio.current_task().cancel()
        await asyncio.sleep(10)

    provider = ScriptedProvider({("stop", "candidate"): [cancel_self]})  # the first stop is cancelled
    fail_setup(provider, exited(1))
    dom = domain(provider)
    with pytest.raises(asyncio.CancelledError):
        await dom.create()
    assert dom.cleanup == "delete_failed" and dom.sandbox_id == "fake-candidate-1"
    assert await dom.close() == "deleted" and provider.closed == ["fake-candidate-1"]


@pytest.mark.asyncio
async def test_the_sandbox_of_a_failed_setup_is_kept_for_close_alone():
    provider = ScriptedProvider({("stop", "candidate"): [RuntimeError("transient")]})
    fail_setup(provider, exited(1))
    dom = domain(provider)
    with pytest.raises(sd.Unscorable):
        await dom.create()
    with pytest.raises(RuntimeError):
        await dom.upload({"job.json": b"{}"})
    with pytest.raises(sd.Unscorable) as info:
        await dom.run("run", 300)
    assert info.value.category == "provider_exec"
    assert provider.sandboxes["fake-candidate-1"]["files"] == {}
    assert not any("runner.py" in call["command"] for call in provider.exec_calls)


@pytest.mark.asyncio
async def test_cancellation_during_create_propagates():
    async def cancel_self():
        asyncio.current_task().cancel()
        await asyncio.sleep(10)

    dom = domain(ScriptedProvider({("create", "candidate"): cancel_self}))
    with pytest.raises(asyncio.CancelledError):
        await dom.create()
    assert dom.cleanup == "not_created"


@pytest.mark.asyncio
async def test_the_observation_truncates_a_long_provider_sandbox_id():
    class LongIds(ScriptedProvider):
        async def create(self, spec):
            handle = await super().create(spec)
            long_id = f"{handle.sandbox_id}-{'x' * 1000}"
            self.sandboxes[long_id] = self.sandboxes.pop(handle.sandbox_id)
            return dataclasses.replace(handle, sandbox_id=long_id)

    dom = domain(LongIds({}))
    assert dom.observation("unknown")["sandbox_id"] is None  # nothing created yet
    await dom.create()
    assert len(dom.sandbox_id) > 1000  # the domain keeps the provider's id for its own calls
    assert dom.observation("completed")["sandbox_id"] == dom.sandbox_id[:256]
    assert await dom.close() == "deleted"
