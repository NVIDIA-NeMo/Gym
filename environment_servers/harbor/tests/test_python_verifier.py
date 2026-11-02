# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Python verifier runs inside the sandbox through the test.sh shim the server uploads over /tests (M31)."""

import io
import json
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

from environment_servers.harbor.tests.conftest import TASK_TOML, AgentSandbox, FakeSandbox, make_server, run_episode
from environment_servers.harbor.tests.test_task_data import episode_with
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.tasks.harbor import load_task


VERIFIER = """
from nemo_gym.verifiers import final_text


def verify(workspace, task_data, response) -> float:
    return float(final_text(response).strip() == task_data.get("answer", ""))
"""


def archive_files(archive: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tar:
        return {
            member.name.removeprefix("./"): tar.extractfile(member).read()
            for member in tar.getmembers()
            if member.isfile()
        }


# Separate mode in another image, which the verifier sandbox fakes below.
SEPARATE_PYTHON_TOML = (
    TASK_TOML.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n')
    + '\n[verifier.environment]\ndocker_image = "org/verifier:1"\n'
)


@dataclass
class ShippedTestsSandbox(FakeSandbox):
    """A verifier image that bakes its own ``/tests/test.sh``, as a prebuilt verifier image usually does."""

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if command.startswith("if [ -d /tests/test.sh ]"):
            self.execs.append({"command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s, "user": user})
            return SandboxExecResult(stdout="file\n", stderr="", return_code=0)
        return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)


def shim_folders(server) -> list[str]:
    """The per-episode shim folders left under the server's snapshots directory."""
    return sorted(p.name for p in Path(server._snapshots.name).iterdir() if p.name.startswith("verifier-shim-"))


def wire_verifier_sandbox(server, verifier, monkeypatch):
    """Hand ``verifier`` out as the separate verifier sandbox, as the test_app separate-mode tests do."""
    monkeypatch.setattr(
        "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
    )

    async def create_verifier(task, *, episode_key=None):
        server._sandboxes[f"{episode_key}/verifier"] = verifier
        return verifier

    monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
    return verifier


def lose_snapshot_tests(server, monkeypatch) -> None:
    """Make the snapshot's ``tests/`` disappear under the episode: after the agent's turn, as verification starts (a
    host-side loss between seed and verification, before either mode decides what to upload)."""
    original = server._run_verifier

    async def without_tests(session, trajectory, **kwargs):
        shutil.rmtree(session.task.path / "tests")
        return await original(session, trajectory, **kwargs)

    monkeypatch.setattr(server, "_run_verifier", without_tests)


def make_python_verifier_server(tmp_path, monkeypatch, *, dataset_level: bool = False, sandbox=None, toml=TASK_TOML):
    """The usual server over ``datasets/ds/hello``, the task's test.sh replaced by a Python verifier."""
    server, task, sandbox, created = make_server(tmp_path, monkeypatch, sandbox, toml=toml)
    (task.path / "tests" / "test.sh").unlink()
    if dataset_level:
        (task.path / "tests").rmdir()
        (task.path.parent / "verifier.py").write_text(VERIFIER)
    else:
        (task.path / "tests" / "verifier.py").write_text(VERIFIER)
    (task.path / "task_data.json").write_text(json.dumps({"answer": "hi"}))
    task = load_task(task.path)
    server.config.tasksets["ds"].tasks["hello"] = task.digest
    return server, task, sandbox, created


class TestPythonVerifierShim:
    async def test_the_shim_is_uploaded_over_the_tasks_tests_and_graded_as_test_sh(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_python_verifier_server(tmp_path, monkeypatch)
        assert task.python_verifier == task.path / "tests" / "verifier.py"
        response = await server.run_request(episode_with(task, {"answer": "hi"}))
        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.verifier_mode == "shared"
        # Two uploads into /tests: the task's own tests/ (with verifier.py), then the shim over it.
        uploads = [archive_files(archive) for archive in sandbox.uploaded_archives]
        tests, shim = [files for files in uploads if "verifier.py" in files or "test.sh" in files]
        assert set(tests) == {"verifier.py"} and tests["verifier.py"].decode() == VERIFIER
        assert set(shim) == {
            "test.sh",
            ".ng/run_verifier.py",
            ".ng/task_data.json",
            ".ng/response.json",
            ".ng/nemo_gym/__init__.py",
            ".ng/nemo_gym/verifiers/__init__.py",
            ".ng/nemo_gym/verifiers/files.py",
            ".ng/nemo_gym/verifiers/runner.py",
        }
        script = shim["test.sh"].decode()
        # The verifier module is the one the task's tests/ carries; the workspace is the session's.
        assert "--verifier /tests/verifier.py" in script and "--workspace /app" in script
        assert "--response /tests/.ng/response.json" in script and "--reward-dir /logs/verifier" in script
        assert json.loads(shim[".ng/task_data.json"]) == {"answer": "hi"}
        # The agent's final response, as the verifier sees it.
        uploaded_response = json.loads(shim[".ng/response.json"])
        assert uploaded_response["id"] == "response" and uploaded_response["output"]
        # test.sh ran under the usual wrapper, as the verifier user, in the workspace.
        run = next(call for call in sandbox.execs if "test.sh" in call["command"] and "timeout" in call["command"])
        assert run["cwd"] == "/app" and "bash /tests/test.sh" in run["command"]
        chmod = next(call for call in sandbox.execs if call["command"] == "chmod +x /tests/test.sh")
        assert chmod is not None

    async def test_a_dataset_level_verifier_travels_in_the_shim_and_the_snapshot(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_python_verifier_server(tmp_path, monkeypatch, dataset_level=True)
        assert task.python_verifier == task.path.parent / "verifier.py" and not (task.path / "tests").exists()
        response = await server.run_request(episode_with(task, {"answer": "hi"}))
        assert response.failure is None, response.failure
        assert response.result.reward == 1.0
        uploads = [archive_files(archive) for archive in sandbox.uploaded_archives]
        (shim,) = [files for files in uploads if "test.sh" in files]
        assert shim[".ng/verifier.py"].decode() == VERIFIER
        assert "--verifier /tests/.ng/verifier.py" in shim["test.sh"].decode()
        # The server's snapshot carried verifier.py beside the task, so the digest it re-derived matched.
        assert not any("verifier.py" in files and "test.sh" not in files for files in uploads)

    async def test_an_edited_dataset_verifier_is_refused_by_the_digest(self, tmp_path, monkeypatch):
        server, task, _, created = make_python_verifier_server(tmp_path, monkeypatch, dataset_level=True)
        (task.path.parent / "verifier.py").write_text(VERIFIER + "\n# tampered\n")
        response = await server.run_request(episode_with(task, {"answer": "hi"}))
        assert response.failure is not None and "changed since materialization" in response.failure.failure_reason
        assert created == []

    async def test_a_test_sh_task_uploads_no_shim(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        assert task.python_verifier is None
        assert (await run_episode(server, task)).result.reward == 1.0
        uploads = [archive_files(archive) for archive in sandbox.uploaded_archives]
        assert not any(name.startswith(".ng/") for files in uploads for name in files)

    async def test_a_test_sh_task_with_a_verifier_py_helper_uploads_no_shim(self, tmp_path, monkeypatch):
        # The hub's helper shape (whole datasets ship it): tests/test.sh runs tests/verifier.py. The task is a
        # test.sh task; the helper travels in the task's tests/ upload and no shim is written over it.
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        (task.path / "tests" / "verifier.py").write_text("print(1.0)\n")
        (task.path / "tests" / "test.sh").write_text(
            "#!/bin/bash\npython3 /tests/verifier.py > /logs/verifier/reward.txt\n"
        )
        task = load_task(task.path)
        assert task.python_verifier is None
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        assert (await run_episode(server, task)).result.reward == 1.0
        uploads = [archive_files(archive) for archive in sandbox.uploaded_archives]
        (tests,) = [files for files in uploads if "test.sh" in files]
        assert sorted(tests) == ["test.sh", "verifier.py"] and tests["verifier.py"] == b"print(1.0)\n"
        assert not any(name.startswith(".ng/") for files in uploads for name in files)

    async def test_the_shim_folder_does_not_outlive_its_upload(self, tmp_path, monkeypatch):
        # The shim is per episode (it embeds the response); a long-running server must not keep one per episode.
        server, task, sandbox, _ = make_python_verifier_server(tmp_path, monkeypatch)
        for _ in range(3):
            response = await server.run_request(episode_with(task, {"answer": "hi"}))
            assert response.failure is None, response.failure
        assert sum(1 for archive in sandbox.uploaded_archives if "test.sh" in archive_files(archive)) == 3
        assert shim_folders(server) == []

    async def test_a_task_verifier_whose_snapshot_tests_vanished_is_an_infrastructure_failure(
        self, tmp_path, monkeypatch
    ):
        # The shim's test.sh names /tests/verifier.py, which only the task's own tests/ upload provides. A snapshot
        # whose tests/ disappeared under the episode used to be treated like a dataset-level verifier with no tests/
        # (the `is_dir()` check): the shim alone went up, the runner exited without a reward, and the episode was
        # scored as a zero. The upload is required by where the verifier is, so the loss masks the sample, as the
        # same loss does for a test.sh task.
        server, task, sandbox, _ = make_python_verifier_server(tmp_path, monkeypatch)
        assert task.python_verifier == task.path / "tests" / "verifier.py"
        lose_snapshot_tests(server, monkeypatch)
        sandbox.verifier_files = {}  # what the runner leaves when /tests/verifier.py is not there: no reward
        response = await server.run_request(episode_with(task, {"answer": "hi"}))
        assert response.failure is None, response.failure
        result = response.result
        assert result.mask_sample is True and result.failure_kind == "provider_unavailable", result
        assert "tests" in result.failure_reason
        # Nothing was graded: no shim went up, and test.sh never ran against a missing module.
        assert not any("test.sh" in archive_files(archive) for archive in sandbox.uploaded_archives)
        assert not any("test.sh" in c["command"] and "timeout" in c["command"] for c in sandbox.execs)
        # The control: a test.sh task losing its tests/ is masked the same way.
        plain_server, plain_task, plain_sandbox, _ = make_server(tmp_path / "plain", monkeypatch)
        lose_snapshot_tests(plain_server, monkeypatch)
        plain_sandbox.verifier_files = {}
        plain = (await run_episode(plain_server, plain_task)).result
        assert plain.mask_sample is True and plain.failure_kind == result.failure_kind


class TestSeparateModeShim:
    """Separate mode: the shim reaches the verifier sandbox, after the task's tests, over whatever the image ships."""

    def separate(self, tmp_path, monkeypatch, *, dataset_level=False, image_ships_tests=False):
        server, task, agent, _ = make_python_verifier_server(
            tmp_path, monkeypatch, dataset_level=dataset_level, sandbox=AgentSandbox(), toml=SEPARATE_PYTHON_TOML
        )
        verifier = wire_verifier_sandbox(
            server, ShippedTestsSandbox() if image_ships_tests else FakeSandbox(), monkeypatch
        )
        return server, task, agent, verifier

    @staticmethod
    def uploads_into(sandbox) -> list[dict[str, bytes]]:
        return [archive_files(archive) for archive in sandbox.uploaded_archives]

    @staticmethod
    def kinds(uploads) -> list[str]:
        return ["shim" if "test.sh" in files else "tests" if "verifier.py" in files else "other" for files in uploads]

    async def test_the_shim_reaches_the_verifier_sandbox_after_the_tasks_tests(self, tmp_path, monkeypatch):
        server, task, agent, verifier = self.separate(tmp_path, monkeypatch)
        response = await server.run_request(episode_with(task, {"answer": "hi"}))
        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.verifier_mode == "separate"
        uploads = self.uploads_into(verifier)
        kinds = self.kinds(uploads)
        assert kinds.count("tests") == 1 and kinds.count("shim") == 1 and kinds.index("tests") < kinds.index("shim")
        shim = uploads[kinds.index("shim")]
        script = shim["test.sh"].decode()
        assert "--verifier /tests/verifier.py" in script and "--response /tests/.ng/response.json" in script
        assert "--workspace /app" in script
        assert json.loads(shim[".ng/response.json"])["id"] == "response"
        assert json.loads(shim[".ng/task_data.json"]) == {"answer": "hi"}
        # Nothing of the grader went to the agent's sandbox, and test.sh ran in the verifier's.
        assert "shim" not in self.kinds(self.uploads_into(agent))
        run = next(c for c in verifier.execs if "test.sh" in c["command"] and "timeout" in c["command"])
        assert "bash /tests/test.sh" in run["command"]
        assert not any("test.sh" in c["command"] and "timeout" in c["command"] for c in agent.execs)
        assert verifier.stopped and agent.stopped and shim_folders(server) == []

    async def test_a_verifier_image_with_its_own_test_sh_still_gets_the_tests_and_the_shim(
        self, tmp_path, monkeypatch
    ):
        # A baked /tests/test.sh wins for a test.sh task (Harbor uploads nothing into a separate verifier); it is not
        # the grader of a task that declares a Python verifier, so the task's tests/ and then the shim go over it.
        server, task, _, verifier = self.separate(tmp_path, monkeypatch, image_ships_tests=True)
        response = await server.run_request(episode_with(task, {"answer": "hi"}))
        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.verifier_mode == "separate"
        kinds = self.kinds(self.uploads_into(verifier))
        assert kinds.count("tests") == 1 and kinds.count("shim") == 1 and kinds.index("tests") < kinds.index("shim")
        assert any(c["command"].startswith("if [ -d /tests/test.sh ]") for c in verifier.execs)
        # The image has tests, so it is not the one the warning is about.
        assert "org/verifier:1" not in server._verifier_images_without_tests
        # Contrast: a test.sh task over the same image uploads nothing into /tests, as before.
        plain_server, plain_task, _, _ = make_server(
            tmp_path / "plain", monkeypatch, AgentSandbox(), toml=SEPARATE_PYTHON_TOML
        )
        plain_verifier = wire_verifier_sandbox(plain_server, ShippedTestsSandbox(), monkeypatch)
        assert plain_task.python_verifier is None
        assert (await run_episode(plain_server, plain_task)).result.verifier_mode == "separate"
        assert set(self.kinds(self.uploads_into(plain_verifier))) <= {"other"}

    async def test_a_dataset_level_verifier_travels_in_the_shim_alone(self, tmp_path, monkeypatch):
        server, task, _, verifier = self.separate(tmp_path, monkeypatch, dataset_level=True)
        assert task.python_verifier == task.path.parent / "verifier.py" and not (task.path / "tests").exists()
        response = await server.run_request(episode_with(task, {"answer": "hi"}))
        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.verifier_mode == "separate"
        uploads = self.uploads_into(verifier)
        kinds = self.kinds(uploads)
        assert kinds.count("shim") == 1 and "tests" not in kinds
        shim = uploads[kinds.index("shim")]
        assert shim[".ng/verifier.py"].decode() == VERIFIER
        assert "--verifier /tests/.ng/verifier.py" in shim["test.sh"].decode()

    async def test_a_task_verifier_whose_snapshot_tests_vanished_is_an_infrastructure_failure(
        self, tmp_path, monkeypatch
    ):
        # The separate-mode branch of the same decision: the task's tests/ is required because the task's own
        # tests/verifier.py is the grader, so its loss under the episode masks the sample instead of uploading the
        # shim alone into the verifier sandbox and scoring the runner's missing-module exit as a zero.
        server, task, agent, verifier = self.separate(tmp_path, monkeypatch)
        lose_snapshot_tests(server, monkeypatch)
        verifier.verifier_files = {}
        response = await server.run_request(episode_with(task, {"answer": "hi"}))
        assert response.failure is None, response.failure
        result = response.result
        assert result.mask_sample is True and result.failure_kind == "provider_unavailable", result
        assert "shim" not in self.kinds(self.uploads_into(verifier))
        assert not any("test.sh" in c["command"] and "timeout" in c["command"] for c in verifier.execs)
        assert agent.stopped and verifier.stopped and shim_folders(server) == []
