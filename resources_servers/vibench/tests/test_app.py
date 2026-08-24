# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.vibench.app import (
    PlanResult,
    VibenchResourcesServer,
    VibenchResourcesServerConfig,
    VibenchVerifyRequest,
    add_evaluation_tags,
)


def make_server(tmp_path: Path, **overrides) -> VibenchResourcesServer:
    config = VibenchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="vibench_resources_server",
        vibench_repo_root=str(tmp_path),
        artifact_dir=str(tmp_path / "artifacts"),
        **overrides,
    )
    return VibenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def make_verify_request(**overrides) -> VibenchVerifyRequest:
    body = {
        "app": "notes",
        "artifact": "mvp",
        "prd_files": ["prds/notes/prd/mvp.txt"],
        "test_plans": ["prds/notes/tests/mvp/test1.txt", "prds/notes/tests/mvp/test2.txt"],
        "asset_dirs": [],
        "artifact_path": "/tmp/vibench-artifacts/app.tar",
        "test_assets_dir": None,
        "responses_create_params": {"input": [{"role": "user", "content": "build it"}]},
        "response": {
            "id": "resp_1",
            "created_at": 0,
            "model": "m",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        },
    }
    body.update(overrides)
    return VibenchVerifyRequest(**body)


class TestEvaluationTags:
    def test_adds_pass_and_comment_after_each_skippable(self):
        plan = "<step>do a thing<skippable>false</skippable></step>\n<step>b<skippable>true</skippable></step>"
        tagged = add_evaluation_tags(plan)
        assert tagged.count("<pass>Y/N</pass>") == 2
        assert tagged.count("<comment></comment>") == 2
        # The evaluation agent expects the tags immediately after the skippable block.
        assert "<skippable>false</skippable>\n<pass>Y/N</pass>\n<comment></comment>" in tagged

    def test_plan_without_skippable_is_unchanged(self):
        plan = "<step>just do it</step>"
        assert add_evaluation_tags(plan) == plan


class TestPathResolution:
    def test_resolves_relative_to_repo_root(self, tmp_path):
        server = make_server(tmp_path)
        assert server._resolve("prds/notes/prd/mvp.txt") == (tmp_path / "prds/notes/prd/mvp.txt").resolve()

    def test_rejects_escape_from_repo_root(self, tmp_path):
        server = make_server(tmp_path)
        # Dataset rows are untrusted input; they must not be able to read arbitrary host files.
        with pytest.raises(ValueError):
            server._resolve("../../etc/passwd")


class TestArtifactUnpacking:
    """The tarball comes out of a box the model controlled, so its members are untrusted."""

    def _tar_with(self, tmp_path: Path, arcname: str) -> Path:
        payload = tmp_path / "payload.txt"
        payload.write_text("x")
        archive = tmp_path / "artifacts" / "app.tar"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w") as tar:
            tar.add(payload, arcname=arcname)
        return archive

    def test_unpacks_normal_members(self, tmp_path):
        server = make_server(tmp_path)
        archive = self._tar_with(tmp_path, "package.json")
        dest = tmp_path / "app"

        server._unpack_artifact(archive, dest)

        assert (dest / "package.json").read_text() == "x"

    def test_rejects_member_escaping_the_app_dir(self, tmp_path):
        server = make_server(tmp_path)
        archive = self._tar_with(tmp_path, "../escaped.txt")
        dest = tmp_path / "app"

        with pytest.raises(ValueError):
            server._unpack_artifact(archive, dest)
        assert not (tmp_path / "escaped.txt").exists()

    @pytest.mark.asyncio
    async def test_rejected_artifact_path_is_not_deleted(self, tmp_path):
        """A rejected path must not be unlinked: deleting an unvalidated, agent-supplied
        path is arbitrary file deletion, and the rejection branch used to do exactly that."""
        server = make_server(tmp_path)
        victim = tmp_path / "IMPORTANT_FILE"
        victim.write_text("do not delete")

        response = await server.verify(_FakeRequest(), make_verify_request(artifact_path=str(victim)))

        assert response.build_failed is True
        assert victim.exists(), "verify deleted a file outside artifact_dir"

    def test_rejects_artifact_path_outside_artifact_dir(self, tmp_path):
        server = make_server(tmp_path)
        # A compromised agent must not be able to point the verifier at arbitrary host files.
        with pytest.raises(ValueError):
            server._resolve_artifact("/etc/passwd")


class _FakeProc:
    """Stand-in for asyncio.create_subprocess_exec's return value."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self._out = stdout.encode()
        self._err = stderr.encode()

    async def communicate(self):
        return self._out, self._err


def _patch_env_creator(monkeypatch, proc: _FakeProc) -> None:
    async def fake_exec(*a, **k):
        return proc

    monkeypatch.setattr("resources_servers.vibench.app.asyncio.create_subprocess_exec", fake_exec)


class TestGraderEnv:
    @pytest.mark.asyncio
    async def test_env_file_entries_are_loaded(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text('# comment\nPROVIDER_KEY="secret-value"\n\nBLANK\n')
        server = make_server(tmp_path, vibench_env_file=str(env_file))
        _patch_env_creator(monkeypatch, _FakeProc(0, "{}"))

        env = await server._grader_env()

        assert env["PROVIDER_KEY"] == "secret-value"
        assert "BLANK" not in env

    @pytest.mark.asyncio
    async def test_env_creator_output_is_merged(self, tmp_path, monkeypatch):
        """The grader agents get their model and tool list from env_creator, not the .env."""
        server = make_server(tmp_path)
        derived = {
            "AGENT_SEEDING_LLM_MODEL": "some/seeding-model",
            "AGENT_SEEDING_LLM_TOOLS": "TerminalTool,FileEditorTool",
            "AGENT_SEEDING_LLM_API_KEY": "k",
        }
        _patch_env_creator(monkeypatch, _FakeProc(0, json.dumps(derived)))

        env = await server._grader_env()

        assert env["AGENT_SEEDING_LLM_MODEL"] == "some/seeding-model"
        assert env["AGENT_SEEDING_LLM_TOOLS"] == "TerminalTool,FileEditorTool"

    @pytest.mark.asyncio
    async def test_unset_builder_slot_is_filled_from_seeding(self, tmp_path, monkeypatch):
        """ViBench validates the builder slot even though grading never uses it."""
        server = make_server(tmp_path)
        derived = {"AGENT_SEEDING_LLM_API_KEY": "k", "AGENT_SEEDING_LLM_MODEL": "m"}
        _patch_env_creator(monkeypatch, _FakeProc(0, json.dumps(derived)))

        env = await server._grader_env()

        assert env["AGENT_LLM_API_KEY"] == "k"
        assert env["AGENT_LLM_MODEL"] == "m"

    @pytest.mark.asyncio
    async def test_env_creator_failure_raises(self, tmp_path, monkeypatch):
        """Degrading to an empty env sends debugging to credentials instead of here."""
        server = make_server(tmp_path)
        _patch_env_creator(monkeypatch, _FakeProc(1, "", "no such model key"))

        with pytest.raises(RuntimeError, match="env_creator failed"):
            await server._grader_env()

    @pytest.mark.asyncio
    async def test_env_is_derived_once_and_cached(self, tmp_path, monkeypatch):
        server = make_server(tmp_path)
        calls = []

        async def fake_exec(*a, **k):
            calls.append(1)
            return _FakeProc(0, json.dumps({"AGENT_SEEDING_LLM_API_KEY": "k"}))

        monkeypatch.setattr("resources_servers.vibench.app.asyncio.create_subprocess_exec", fake_exec)

        await server._grader_env()
        await server._grader_env()

        # Six grading calls per rollout must not mean six subprocesses.
        assert len(calls) == 1


class TestVerifyRewardAggregation:
    """verify() is exercised with the sandbox and grading steps stubbed out; the Docker path
    is covered by the end-to-end smoke test in README.md, not by unit tests."""

    @pytest.mark.asyncio
    async def test_missing_artifact_scores_zero_and_flags_build_failure(self, tmp_path, monkeypatch):
        server = make_server(tmp_path)

        response = await server.verify(_FakeRequest(), make_verify_request(artifact_path=None))

        assert response.reward == 0.0
        assert response.build_failed is True
        assert response.test_plans_total == 2
        assert response.test_plans_graded == 0

    @pytest.mark.asyncio
    async def test_reward_is_mean_normalized_score_across_test_plans(self, tmp_path, monkeypatch):
        server = make_server(tmp_path)
        _stub_extraction(server, monkeypatch)

        scores = iter([1.0, 0.5])

        async def fake_grade(app_dir, test_plan_rel, work_dir, test_assets_dir):
            normalized = next(scores)
            return PlanResult(
                test_plan=Path(test_plan_rel).stem,
                score=normalized * 10,
                full_points=10,
                normalized_score=normalized,
                steps_total=5,
                steps_passed=int(5 * normalized),
                seeding_failed=False,
                duration_s=1.0,
            )

        monkeypatch.setattr(server, "_grade_one_test_plan", fake_grade)

        response = await server.verify(_FakeRequest(), make_verify_request())

        assert response.reward == pytest.approx(0.75)
        assert response.reward_components == {"test1": 1.0, "test2": 0.5}
        assert response.build_failed is False
        assert response.test_plans_graded == 2

    @pytest.mark.asyncio
    async def test_seeding_failure_counts_as_zero_not_as_a_dropped_plan(self, tmp_path, monkeypatch):
        server = make_server(tmp_path)
        _stub_extraction(server, monkeypatch)

        outcomes = iter([(1.0, False), (0.0, True)])

        async def fake_grade(app_dir, test_plan_rel, work_dir, test_assets_dir):
            normalized, seeding_failed = next(outcomes)
            return PlanResult(
                test_plan=Path(test_plan_rel).stem,
                score=normalized * 10,
                full_points=10 if not seeding_failed else 0,
                normalized_score=normalized,
                steps_total=0,
                steps_passed=0,
                seeding_failed=seeding_failed,
                duration_s=1.0,
            )

        monkeypatch.setattr(server, "_grade_one_test_plan", fake_grade)

        response = await server.verify(_FakeRequest(), make_verify_request())

        # A plan that could not be seeded drags the mean down rather than vanishing from it.
        assert response.reward == pytest.approx(0.5)
        assert response.seeding_failure_rate == pytest.approx(0.5)
        assert response.test_plans_graded == 1


class _FakeRequest:
    def __init__(self, session_id: str = "session-1"):
        self.session = {"session_id": session_id}


def _stub_extraction(server: VibenchResourcesServer, monkeypatch) -> None:
    """Make artifact unpacking produce a minimally valid app."""

    def fake_unpack(artifact: Path, dest: Path):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "package.json").write_text(json.dumps({"name": "app"}))

    monkeypatch.setattr(server, "_unpack_artifact", fake_unpack)
    monkeypatch.setattr(server, "_resolve_artifact", lambda p: Path(p))
