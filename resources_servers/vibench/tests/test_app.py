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
import asyncio
import inspect
import json
import signal
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


def _const_env(env: dict):
    async def _inner():
        return dict(env)

    return _inner


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


class TestRedaction:
    def test_grader_credentials_are_scrubbed_from_captured_output(self, tmp_path):
        """Captured output ships in the rollout JSONL, which gets committed."""
        server = make_server(tmp_path)
        env = {"AGENT_SEEDING_LLM_API_KEY": "super-secret-value", "AGENT_SEEDING_LLM_MODEL": "m"}

        out = server._redact("connecting with key=super-secret-value now", env)

        assert "super-secret-value" not in out
        assert "<redacted:AGENT_SEEDING_LLM_API_KEY>" in out

    def test_non_secret_values_are_left_alone(self, tmp_path):
        server = make_server(tmp_path)
        env = {"AGENT_SEEDING_LLM_MODEL": "anthropic/some-model"}

        assert server._redact("model anthropic/some-model", env) == "model anthropic/some-model"

    def test_short_values_are_not_scrubbed(self, tmp_path):
        """A short key would match everywhere and destroy the log's usefulness."""
        server = make_server(tmp_path)
        assert server._redact("the app is ok", {"AGENT_LLM_API_KEY": "ok"}) == "the app is ok"


class TestAggregateMetrics:
    """Signatures must match AggregateMetricsMixin: compute_metrics receives rollouts
    grouped by task, get_key_metrics receives agent_metrics and returns a dict."""

    def test_separates_the_three_causes_of_a_zero(self, tmp_path):
        server = make_server(tmp_path)
        # Grouped by task, as compute_aggregate_metrics passes it.
        tasks = [
            [
                {
                    "reward": 1.0,
                    "test_plans_total": 3,
                    "test_plans_graded": 3,
                    "build_failed": False,
                    "seeding_failure_rate": 0.0,
                }
            ],
            [
                {
                    "reward": 0.0,
                    "test_plans_total": 3,
                    "test_plans_graded": 0,
                    "build_failed": True,
                    "seeding_failure_rate": 0.0,
                }
            ],
            [
                {
                    "reward": 0.5,
                    "test_plans_total": 4,
                    "test_plans_graded": 4,
                    "build_failed": False,
                    "seeding_failure_rate": 0.25,
                }
            ],
        ]

        m = server.compute_metrics(tasks)

        assert m["mean_reward"] == pytest.approx(0.5)
        assert m["perfect_rate"] == pytest.approx(1 / 3)
        assert m["zero_rate"] == pytest.approx(1 / 3)
        assert m["build_failure_rate"] == pytest.approx(1 / 3)
        assert m["plans_graded_rate"] == pytest.approx(7 / 10)

    def test_multiple_rollouts_per_task_are_flattened(self, tmp_path):
        """num_repeats > 1 puts several rollouts in one task group."""
        server = make_server(tmp_path)
        tasks = [
            [
                {"reward": 1.0, "test_plans_total": 1, "test_plans_graded": 1},
                {"reward": 0.0, "test_plans_total": 1, "test_plans_graded": 1},
            ]
        ]

        m = server.compute_metrics(tasks)

        assert m["mean_reward"] == pytest.approx(0.5)
        assert m["perfect_rate"] == pytest.approx(0.5)

    def test_empty_input_is_not_a_division_error(self, tmp_path):
        assert make_server(tmp_path).compute_metrics([]) == {}
        assert make_server(tmp_path).compute_metrics([[]]) == {}

    def test_key_metrics_takes_agent_metrics_and_returns_a_dict(self, tmp_path):
        server = make_server(tmp_path)
        agent_metrics = {
            "mean_reward": 0.5,
            "plans_graded_rate": 0.9,
            "build_failure_rate": 0.1,
            "mean/foo": 1.0,
            "unrelated": 2.0,
        }

        selected = server.get_key_metrics(agent_metrics)

        assert isinstance(selected, dict)
        assert selected["plans_graded_rate"] == 0.9
        assert selected["mean/foo"] == 1.0, "the framework default (mean/*) must survive"
        assert "unrelated" not in selected

    def test_key_metrics_tolerates_absent_keys(self, tmp_path):
        assert make_server(tmp_path).get_key_metrics({}) == {}

    def test_signatures_match_the_framework_contract(self, tmp_path):
        """The bug this replaces was a signature mismatch that unit tests missed because they
        called these the way the code expected, not the way the framework does."""
        from nemo_gym.reward_profile import AggregateMetricsMixin

        server = make_server(tmp_path)
        for name in ("compute_metrics", "get_key_metrics"):
            mine = inspect.signature(getattr(server, name))
            base = inspect.signature(getattr(AggregateMetricsMixin, name))
            assert list(mine.parameters) == [q for q in base.parameters if q != "self"], name


class TestRunVibenchScript:
    @pytest.mark.asyncio
    async def test_returns_code_and_merged_output(self, tmp_path, monkeypatch):
        server = make_server(tmp_path)
        monkeypatch.setattr(server, "_grader_env", _const_env({}))
        _patch_env_creator(monkeypatch, _FakeProc(0, "hello"))

        code, log = await server._run_vibench_script(["/bin/true"], timeout_s=30)

        assert (code, log) == (0, "hello")

    @pytest.mark.asyncio
    async def test_credentials_are_scrubbed_from_the_returned_log(self, tmp_path, monkeypatch):
        """This log is stored on PlanResult.error and ships in the rollout JSONL."""
        server = make_server(tmp_path)
        monkeypatch.setattr(server, "_grader_env", _const_env({"AGENT_SEEDING_LLM_API_KEY": "leaky-secret-key"}))
        _patch_env_creator(monkeypatch, _FakeProc(0, "using leaky-secret-key here"))

        _, log = await server._run_vibench_script(["/bin/true"], timeout_s=30)

        assert "leaky-secret-key" not in log

    @pytest.mark.asyncio
    async def test_spawn_failure_is_a_failed_grade_not_an_exception(self, tmp_path, monkeypatch):
        server = make_server(tmp_path)
        monkeypatch.setattr(server, "_grader_env", _const_env({}))

        async def boom(*a, **k):
            raise OSError("cannot spawn")

        monkeypatch.setattr("resources_servers.vibench.app.asyncio.create_subprocess_exec", boom)

        code, log = await server._run_vibench_script(["/bin/true"], timeout_s=30)

        assert code == 1
        assert "cannot spawn" in log

    @pytest.mark.asyncio
    async def test_timeout_terminates_the_group_so_compose_cleanup_runs(self, tmp_path, monkeypatch):
        """SIGKILL would skip ViBench's finally-block `docker-compose down`, leaking the
        very stack this timeout exists to reap."""
        server = make_server(tmp_path, evaluation_timeout_s=1, cleanup_grace_s=0.05)
        monkeypatch.setattr(server, "_grader_env", _const_env({}))
        signals: list[int] = []

        class _Hanging:
            returncode = None
            pid = 4242

            async def communicate(self):
                raise asyncio.TimeoutError()

            async def wait(self):
                # Never exits on SIGTERM, so the escalation path is exercised too.
                await asyncio.sleep(10)

        async def fake_exec(*a, **k):
            assert k.get("start_new_session") is True, "no process group means no group signal"
            return _Hanging()

        monkeypatch.setattr("resources_servers.vibench.app.asyncio.create_subprocess_exec", fake_exec)
        monkeypatch.setattr("resources_servers.vibench.app.os.getpgid", lambda pid: pid)
        monkeypatch.setattr("resources_servers.vibench.app.os.killpg", lambda pid, sig: signals.append(sig))

        code, log = await server._run_vibench_script(["/bin/sleep", "99"], timeout_s=0.05)

        assert code == 1
        assert "timed out" in log
        # SIGTERM first so cleanup can run, SIGKILL only for what ignores it.
        # SIGINT first: ViBench cleans up in a finally, which SIGTERM does not unwind.
        assert signals == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]

    @pytest.mark.asyncio
    async def test_a_process_that_exits_on_sigint_is_not_escalated(self, tmp_path, monkeypatch):
        server = make_server(tmp_path, cleanup_grace_s=5)
        monkeypatch.setattr(server, "_grader_env", _const_env({}))
        signals: list[int] = []

        class _Polite:
            """Exits on the first signal, as a real process does: wait() returning means
            the process is gone, so returncode is set."""

            returncode = None
            pid = 99

            async def communicate(self):
                raise asyncio.TimeoutError()

            async def wait(self):
                self.returncode = -signal.SIGINT
                return self.returncode

        async def fake_exec(*a, **k):
            return _Polite()

        monkeypatch.setattr("resources_servers.vibench.app.asyncio.create_subprocess_exec", fake_exec)
        monkeypatch.setattr("resources_servers.vibench.app.os.getpgid", lambda pid: pid)
        monkeypatch.setattr("resources_servers.vibench.app.os.killpg", lambda pid, sig: signals.append(sig))

        await server._run_vibench_script(["/bin/sleep", "99"], timeout_s=0.05)

        assert signals == [signal.SIGINT]


class TestPlanFailureIsolation:
    """A bad plan must be a zeroed plan, never a 500 that loses the whole rollout."""

    @pytest.mark.asyncio
    async def test_truncated_scorecard_zeroes_only_that_plan(self, tmp_path, monkeypatch):
        server = make_server(tmp_path)
        plan = tmp_path / "prds" / "notes" / "tests" / "mvp" / "test1.txt"
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text("<step>x</step>")
        work = tmp_path / "work"

        async def fake_run(cmd, timeout_s=None):
            out = work / "test1"
            if "run-seed.py" in " ".join(cmd):
                (out / "seed" / "seeding").mkdir(parents=True, exist_ok=True)
            else:
                (out / "evaluation-finished.json").write_text('{"score": 30, "full_po')
            return 0, ""

        monkeypatch.setattr(server, "_run_vibench_script", fake_run)

        r = await server._grade_one_test_plan(tmp_path / "app", "prds/notes/tests/mvp/test1.txt", work, None)

        assert r.normalized_score == 0.0
        assert r.seeding_failed is False
        assert "JSONDecodeError" in (r.error or "")

    @pytest.mark.asyncio
    async def test_unreadable_plan_file_zeroes_only_that_plan(self, tmp_path, monkeypatch):
        server = make_server(tmp_path)

        r = await server._grade_one_test_plan(
            tmp_path / "app", "prds/notes/tests/mvp/missing.txt", tmp_path / "work", None
        )

        assert r.normalized_score == 0.0
        assert r.error

    @pytest.mark.asyncio
    async def test_one_plan_raising_does_not_lose_the_others(self, tmp_path, monkeypatch):
        """gather(return_exceptions=True): a raised plan is zeroed, siblings keep their scores."""
        server = make_server(tmp_path)
        _stub_extraction(server, monkeypatch)
        calls = {"n": 0}

        async def flaky(app_dir, test_plan_rel, work_dir, test_assets_dir):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("grading exploded")
            return PlanResult(
                test_plan=Path(test_plan_rel).stem,
                score=10,
                full_points=10,
                normalized_score=1.0,
                steps_total=1,
                steps_passed=1,
                seeding_failed=False,
                duration_s=1.0,
            )

        monkeypatch.setattr(server, "_grade_one_test_plan", flaky)

        response = await server.verify(_FakeRequest(), make_verify_request())

        assert response.test_plans_total == 2
        # One exploded, one scored 1.0 -> the surviving plan is not lost.
        assert response.reward == pytest.approx(0.5)
        assert any("grading exploded" in (r.error or "") for r in response.results)


class TestBuildContract:
    """The grading stack invokes ViBench's two scripts; package.json is not the contract."""

    def test_an_app_with_both_scripts_is_buildable(self, tmp_path):
        app = tmp_path / "app"
        app.mkdir()
        (app / "setup-environment.sh").write_text("#!/bin/bash\n")
        (app / "start-server.sh").write_text("#!/bin/bash\n")

        assert make_server(tmp_path)._looks_buildable(app) is True

    def test_a_python_app_is_not_penalised_for_having_no_package_json(self, tmp_path):
        app = tmp_path / "app"
        app.mkdir()
        (app / "setup-environment.sh").write_text("#!/bin/bash\n")
        (app / "start-server.sh").write_text("#!/bin/bash\n")
        (app / "main.py").write_text("print('hi')")

        assert make_server(tmp_path)._looks_buildable(app) is True

    def test_missing_start_script_is_a_build_failure(self, tmp_path):
        app = tmp_path / "app"
        app.mkdir()
        (app / "setup-environment.sh").write_text("#!/bin/bash\n")

        assert make_server(tmp_path)._looks_buildable(app) is False

    def test_empty_tree_is_a_build_failure(self, tmp_path):
        app = tmp_path / "app"
        app.mkdir()

        assert make_server(tmp_path)._looks_buildable(app) is False

    def test_absent_dir_is_a_build_failure(self, tmp_path):
        assert make_server(tmp_path)._looks_buildable(tmp_path / "nope") is False


class TestGradeOneTestPlan:
    def _plan(self, tmp_path: Path) -> str:
        plan = tmp_path / "prds" / "notes" / "tests" / "mvp" / "test1.txt"
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text("<step>x<skippable>n</skippable></step>")
        return "prds/notes/tests/mvp/test1.txt"

    @pytest.mark.asyncio
    async def test_seeding_failure_is_reported_as_such(self, tmp_path, monkeypatch):
        server = make_server(tmp_path)
        rel = self._plan(tmp_path)

        async def fake_run(cmd, timeout_s=None):
            return 1, "seeding blew up"

        monkeypatch.setattr(server, "_run_vibench_script", fake_run)

        r = await server._grade_one_test_plan(tmp_path / "app", rel, tmp_path / "work", None)

        assert r.seeding_failed is True
        assert r.normalized_score == 0.0

    @pytest.mark.asyncio
    async def test_missing_report_after_seeding_is_an_evaluation_failure(self, tmp_path, monkeypatch):
        """Blaming seeding here sent debugging to the wrong stage once already."""
        server = make_server(tmp_path)
        rel = self._plan(tmp_path)
        work = tmp_path / "work"

        async def fake_run(cmd, timeout_s=None):
            # Seeding succeeds (creates its output dir); evaluation writes no report.
            if "run-seed.py" in " ".join(cmd):
                (work / "test1" / "seed" / "seeding").mkdir(parents=True, exist_ok=True)
            return 0, "no report produced"

        monkeypatch.setattr(server, "_run_vibench_script", fake_run)

        r = await server._grade_one_test_plan(tmp_path / "app", rel, work, None)

        assert r.seeding_failed is False
        assert r.normalized_score == 0.0

    @pytest.mark.asyncio
    async def test_parses_a_scorecard_and_counts_passed_steps(self, tmp_path, monkeypatch):
        server = make_server(tmp_path, keep_evaluation_artifacts=True)
        rel = self._plan(tmp_path)
        work = tmp_path / "work"

        async def fake_run(cmd, timeout_s=None):
            out = work / "test1"
            if "run-seed.py" in " ".join(cmd):
                (out / "seed" / "seeding").mkdir(parents=True, exist_ok=True)
            else:
                (out / "evaluation-finished.json").write_text(
                    json.dumps(
                        {"score": 30, "full_points": 50, "steps": [{"points": 10}, {"points": 20}, {"points": 0}]}
                    )
                )
            return 0, ""

        monkeypatch.setattr(server, "_run_vibench_script", fake_run)

        r = await server._grade_one_test_plan(tmp_path / "app", rel, work, None)

        assert (r.score, r.full_points) == (30.0, 50.0)
        assert r.normalized_score == pytest.approx(0.6)
        assert (r.steps_total, r.steps_passed) == (3, 2)

    @pytest.mark.asyncio
    async def test_test_assets_are_passed_only_to_evaluation(self, tmp_path, monkeypatch):
        """The builder must never see them; the evaluation agent needs them."""
        server = make_server(tmp_path)
        rel = self._plan(tmp_path)
        work = tmp_path / "work"
        assets = tmp_path / "prds" / "notes" / "test_assets"
        assets.mkdir(parents=True)
        seen = []

        async def fake_run(cmd, timeout_s=None):
            seen.append(" ".join(cmd))
            if "run-seed.py" in " ".join(cmd):
                (work / "test1" / "seed" / "seeding").mkdir(parents=True, exist_ok=True)
            return 0, ""

        monkeypatch.setattr(server, "_run_vibench_script", fake_run)

        await server._grade_one_test_plan(tmp_path / "app", rel, work, "prds/notes/test_assets")

        assert "--test-assets" not in seen[0]
        assert "--test-assets" in seen[1]

    @pytest.mark.asyncio
    async def test_zero_full_points_does_not_divide_by_zero(self, tmp_path, monkeypatch):
        server = make_server(tmp_path)
        rel = self._plan(tmp_path)
        work = tmp_path / "work"

        async def fake_run(cmd, timeout_s=None):
            out = work / "test1"
            if "run-seed.py" in " ".join(cmd):
                (out / "seed" / "seeding").mkdir(parents=True, exist_ok=True)
            else:
                (out / "evaluation-finished.json").write_text(json.dumps({"score": 0, "full_points": 0, "steps": []}))
            return 0, ""

        monkeypatch.setattr(server, "_run_vibench_script", fake_run)

        r = await server._grade_one_test_plan(tmp_path / "app", rel, work, None)

        assert r.normalized_score == 0.0


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
        (dest / "setup-environment.sh").write_text("#!/bin/bash\n")
        (dest / "start-server.sh").write_text("#!/bin/bash\n")

    monkeypatch.setattr(server, "_unpack_artifact", fake_unpack)
    monkeypatch.setattr(server, "_resolve_artifact", lambda p: Path(p))
