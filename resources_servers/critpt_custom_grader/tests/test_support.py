# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from nemo_gym.sandbox import AsyncSandbox, SandboxCreateError, SandboxSpec
from nemo_gym.sandbox.providers.registry import list_providers
from resources_servers.critpt_custom_grader.tests.support import (
    MISSING_FILE,
    VALUE,
    ScriptedProvider,
    exited,
    run_result,
)


@pytest.mark.asyncio
async def test_scripted_provider_round_trips_files_and_fills_the_nonce(tmp_path):
    provider = ScriptedProvider({("result", "candidate"): [run_result("completed", [VALUE])]})
    box = AsyncSandbox(provider, owns_provider=False)
    await box.start(SandboxSpec(workdir="/work", metadata={"ng-grader-domain": "candidate"}))
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"nonce": "abc123"}))
    await box.upload(job, "/work/j1-candidate/job.json")
    result = await box.exec(
        '"/usr/local/bin/python3" -I -B runner.py run x y', cwd="/work/j1-candidate", user="grader"
    )
    assert result.return_code == 0 and result.error_type is None
    assert provider.exec_calls[-1]["user"] == "grader" and provider.exec_calls[-1]["cwd"] == "/work/j1-candidate"
    probe = await box.exec('test -f "/work/j1-candidate/result.json" && stat -c %s "/work/j1-candidate/result.json"')
    assert probe.return_code == 0 and int(probe.stdout.strip()) > 0
    target = tmp_path / "result.json"
    await box.download("/work/j1-candidate/result.json", target)
    assert json.loads(target.read_text())["nonce"] == "abc123"
    await box.stop()
    assert provider.closed == ["fake-candidate-1"] and provider.aclosed == 0


@pytest.mark.asyncio
async def test_scripted_failures_create_exec_missing_file(tmp_path):
    provider = ScriptedProvider(
        {
            ("create", "reference"): SandboxCreateError("quota"),
            ("exec", "comparator"): [exited(0, error_type="timeout")],
            ("result", "comparator"): [MISSING_FILE],
        }
    )
    with pytest.raises(SandboxCreateError):
        await AsyncSandbox(provider, owns_provider=False).start(
            SandboxSpec(workdir="/work", metadata={"ng-grader-domain": "reference"})
        )
    box = AsyncSandbox(provider, owns_provider=False)
    await box.start(SandboxSpec(workdir="/work", metadata={"ng-grader-domain": "comparator"}))
    assert (await box.exec("runner.py compare")).error_type == "timeout"
    probe = await box.exec('test -f "/work/x/result.json" && stat -c %s "/work/x/result.json"')
    assert probe.return_code == 1
    with pytest.raises(FileNotFoundError):
        await box.download("/work/x/result.json", tmp_path / "r.json")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("transport"), TimeoutError])
async def test_download_key_raises_for_its_domain_only_and_the_default_is_unchanged(tmp_path, failure):
    payload = run_result("completed", [VALUE])
    provider = ScriptedProvider(
        {
            ("download", "comparator"): failure,  # an exception instance or an exception class
            ("result", "comparator"): [payload],
            ("result", "candidate"): [payload],
        }
    )
    comparator = AsyncSandbox(provider, owns_provider=False)
    await comparator.start(SandboxSpec(workdir="/work", metadata={"ng-grader-domain": "comparator"}))
    probe = await comparator.exec('test -f "/work/x/result.json" && stat -c %s "/work/x/result.json"')
    assert (
        probe.return_code == 0 and int(probe.stdout.strip()) > 0
    )  # the probe still sees the result: only the transfer fails
    with pytest.raises(failure if isinstance(failure, type) else type(failure)):
        await comparator.download("/work/x/result.json", tmp_path / "comparator.json")
    assert not (tmp_path / "comparator.json").exists()
    candidate = AsyncSandbox(provider, owns_provider=False)
    await candidate.start(SandboxSpec(workdir="/work", metadata={"ng-grader-domain": "candidate"}))
    await candidate.download("/work/x/result.json", tmp_path / "candidate.json")
    assert json.loads((tmp_path / "candidate.json").read_text())["status"] == "completed"


def test_fake_is_registered_for_config_driven_tests():
    assert "critpt_fake" in list_providers()


@pytest.mark.asyncio
async def test_size_probe_is_answered_before_the_import_probe():
    # "stat -c" plus a path that contains "import" is still the size probe, so a missing file stays missing.
    provider = ScriptedProvider({("result", "candidate"): [MISSING_FILE]})
    box = AsyncSandbox(provider, owns_provider=False)
    await box.start(SandboxSpec(workdir="/work", metadata={"ng-grader-domain": "candidate"}))
    probe = await box.exec(
        'test -f "/work/import1-candidate/result.json" && stat -c %s "/work/import1-candidate/result.json"'
    )
    assert (probe.return_code, probe.stdout) == (1, "")


@pytest.mark.asyncio
async def test_exhausted_result_list_probes_as_absent(tmp_path):
    provider = ScriptedProvider({("result", "candidate"): [run_result("completed", [VALUE])]})
    box = AsyncSandbox(provider, owns_provider=False)
    await box.start(SandboxSpec(workdir="/work", metadata={"ng-grader-domain": "candidate"}))
    command = 'test -f "/work/j1-candidate/result.json" && stat -c %s "/work/j1-candidate/result.json"'
    assert (await box.exec(command)).return_code == 0
    await box.download("/work/j1-candidate/result.json", tmp_path / "result.json")
    probe = await box.exec(command)
    assert (probe.return_code, probe.stdout) == (1, "")
