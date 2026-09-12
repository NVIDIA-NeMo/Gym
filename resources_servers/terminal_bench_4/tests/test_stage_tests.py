# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tarfile

import pytest

from nemo_gym.sandbox import SandboxExecResult
from resources_servers.terminal_bench_4.stage_tests import REMOTE_TESTS_BUNDLE, pack_tests, stage_verifier_tests
from resources_servers.terminal_bench_4.tests.test_app import (
    FakeSandbox,
    make_server,
    make_task_dir,
    seed_and_verify,
    wire_sandboxes,
    write_reward,
)


def test_bundle_contains_only_relative_test_paths(tmp_path):
    source = tmp_path / "tests"
    (source / "nested").mkdir(parents=True)
    (source / "test.sh").write_text("python /tests/nested/check.py\n")
    (source / "nested/check.py").write_text("assert True\n")
    bundle = tmp_path / "tests.tgz"
    pack_tests(source, bundle, 1024)
    with tarfile.open(bundle) as archive:
        assert set(archive.getnames()) == {"tests/test.sh", "tests/nested/check.py"}


def test_missing_script_and_oversized_or_symlinked_bundles_fail(tmp_path):
    source = tmp_path / "tests"
    source.mkdir()
    bundle = tmp_path / "tests.tgz"
    with pytest.raises(ValueError, match="missing test.sh"):
        pack_tests(source, bundle, 1024)
    (source / "test.sh").write_text("123456")
    with pytest.raises(ValueError, match="max_bytes"):
        pack_tests(source, bundle, 5)
    (source / "leak").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symlink"):
        pack_tests(source, bundle, 1024)


def test_bundle_entry_limit(tmp_path, monkeypatch):
    source = tmp_path / "tests"
    source.mkdir()
    (source / "test.sh").write_text("echo test")
    monkeypatch.setattr(
        "resources_servers.terminal_bench_4.stage_tests.os.walk",
        lambda *args, **kwargs: iter([(str(source), [], ["absent"] * 10001)]),
    )
    with pytest.raises(ValueError, match="10,000"):
        pack_tests(source, tmp_path / "tests.tgz", 1024)


class UploadVerifier(FakeSandbox):
    async def exec(self, command, **kwargs):
        if command.startswith(": ng-tb4-install-tests;"):
            assert "mv /tests /tmp/nemo-gym-image-tests-" in command
            assert self.files[REMOTE_TESTS_BUNDLE]
            self._extract(self.files[REMOTE_TESTS_BUNDLE])
            return SandboxExecResult(stdout="", stderr="", return_code=0)
        return await super().exec(command, **kwargs)


async def test_adapted_tests_are_uploaded_only_to_fresh_verifier(tmp_path):
    task = make_task_dir(tmp_path, artifacts='["/app/answer.txt"]')
    agent = FakeSandbox(name="agent", files={"/app/answer.txt": b"42"})

    def grade(verifier):
        assert agent.stopped
        assert verifier.files["/app/answer.txt"] == b"42"
        assert verifier.files["/tests/test.sh"] == (task / "tests/test.sh").read_bytes()
        assert "/tests/test.sh" not in agent.files
        assert REMOTE_TESTS_BUNDLE not in agent.files
        return write_reward("1")(verifier)

    verifier = UploadVerifier(name="verifier", on_run_tests=grade)
    server = make_server(tmp_path, verifier_tests_from_task=True)
    created = wire_sandboxes(server, agent, verifier)
    result = await seed_and_verify(server, task)
    assert result.reward == 1 and result.evaluation_completed
    assert [x["role"] for x in created] == ["agent", "verifier"]
    assert verifier.stopped


async def test_failed_install_is_not_a_reward_zero(tmp_path):
    task = make_task_dir(tmp_path)
    sandbox = FakeSandbox(name="verifier", fail_steps={"install-tests": 1})
    with pytest.raises(RuntimeError, match="Installing tests"):
        await stage_verifier_tests(sandbox, task / "tests", 1024 * 1024)
