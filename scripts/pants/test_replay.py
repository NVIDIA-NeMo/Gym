# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import subprocess
import zipfile

import pytest
import replay
from replay import BRANCH, Driver, driver_lock


REAL_READINESS = Driver.readiness


class FakeGitHub:
    def __init__(self, output):
        self.output = output
        self.posts = []
        self.runs = []
        self.visible = True
        self.uncertain = False
        self.downloads = []
        self.fail_download = None
        self.blobs = {}
        self.jobs = []
        self.artifacts = []

    def api(self, path, *, paginate=False, payload=None):
        if payload is not None:
            state = json.loads((self.output / "state.json").read_text())
            entry = state["entries"][state["current"]]
            assert entry["status"] == "dispatch-intent"
            assert entry["replay_token"] == payload["inputs"]["replay_token"]
            self.posts.append(payload)
            self.runs.append(
                {
                    "id": len(self.posts),
                    "display_title": f"Pants CPU {entry['replay_token']} source {entry['source_sha']}",
                    "event": "workflow_dispatch",
                    "head_branch": BRANCH,
                    "head_sha": "a" * 40,
                    "run_attempt": 1,
                    "status": "queued",
                    "conclusion": None,
                }
            )
            if self.uncertain:
                raise OSError("Connection lost after GitHub accepted POST")
            return None
        if path.startswith("git/ref/"):
            return {"object": {"sha": "a" * 40}}
        if "/workflows/" in path:
            return [{"workflow_runs": self.runs if self.visible else []}]
        if path.endswith("/artifacts?per_page=100"):
            return [{"artifacts": self.artifacts}]
        if "/jobs?" in path:
            return [{"jobs": self.jobs}]
        return next(run for run in self.runs if run["id"] == int(path.split("/")[-1]))

    def download(self, path, destination):
        self.downloads.append(path)
        if path == self.fail_download:
            self.fail_download = None
            destination.write_bytes(b"partial")
            raise OSError("Interrupted download")
        destination.write_bytes(self.blobs[path])


@pytest.fixture
def setup(tmp_path, monkeypatch):
    repo, output = tmp_path / "controller", tmp_path / "output"
    repo.mkdir()
    plan = {
        "schema_version": 1,
        "main_anchor": "c" * 40,
        "snapshots": [{"source_sha": sha * 40, "dependency_inputs": {}, "uv_version": "0.13.4"} for sha in ("b", "c")],
    }
    plan["plan_id"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    github = FakeGitHub(output)
    monkeypatch.setattr(replay, "_now", lambda: "2026-11-05T12:00:00Z")
    monkeypatch.setattr(Driver, "guard", lambda self: "d" * 64)
    monkeypatch.setattr(Driver, "readiness", lambda self: {"ready": True, "state": "recent-arc-assignment"})
    monkeypatch.setattr(Driver, "preflight", lambda self, entry: {"source_sha": entry["source_sha"]})
    monkeypatch.setattr(
        Driver, "caches", lambda self, directory, contract: directory.mkdir(parents=True, exist_ok=True)
    )
    driver = Driver(repo=repo, plan=path, output=output, controller_sha="a" * 40, github=github)
    return driver, github, path


def resume(driver, github, plan):
    return Driver(repo=driver.repo, plan=plan, output=driver.output, controller_sha="a" * 40, github=github)


def archive(files):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as output:
        for name, value in files.items():
            output.writestr(name, json.dumps(value))
    return data.getvalue()


def terminal_evidence(driver, github, *, wrong_source=False, conclusion="success"):
    run = github.runs[-1]
    run.update(status="completed", conclusion=conclusion)
    entry = driver.state["entries"][driver.state["current"]]
    route = {
        "source_sha": "f" * 40 if wrong_source else entry["source_sha"],
        "experiment_sha": "a" * 40,
        "replay_token": entry["replay_token"],
        "cache_namespace": driver.state["cache_namespace"],
        "execution_mode": "incremental",
        "force_full": True,
    }
    github.jobs = [
        {
            "id": 100,
            "name": "unassigned",
            "runner_id": 0,
            "runner_name": "unassigned-placeholder",
            "started_at": "2026-11-05T00:00:00Z",
            "completed_at": "2026-11-05T03:00:00Z",
            "conclusion": "cancelled",
        },
        {
            "id": 101,
            "name": "assigned",
            "runner_id": 1,
            "started_at": "2026-11-05T00:00:00Z",
            "completed_at": "2026-11-05T00:01:00Z",
            "conclusion": "success",
        },
    ]
    github.blobs["actions/jobs/101/logs"] = b"Cache restored successfully\nactual test log\n"
    github.artifacts = [{"id": 10, "name": "full-ci-plan"}, {"id": 11, "name": "full-ci-comparison"}]
    github.blobs["actions/artifacts/10/zip"] = archive({"routing.json": route})
    github.blobs["actions/artifacts/11/zip"] = archive(
        {"comparison/results.json": {"valid": conclusion == "success", "routing": route}}
    )


def test_persisted_intent_recovers_uncertain_post_without_duplicate_dispatch(setup):
    driver, github, plan = setup
    github.uncertain = True
    with pytest.raises(OSError, match="accepted POST"):
        driver.transition()
    driver = resume(driver, github, plan)
    github.visible = False
    assert driver.transition()  # Eventual visibility does not authorize another POST.
    assert len(github.posts) == 1
    github.visible = True
    assert driver.transition()
    assert driver.state["entries"][0]["run_id"] == 1
    assert len(github.posts) == 1


def test_intent_persisted_before_post_is_never_blindly_reissued(setup):
    driver, github, plan = setup
    entry = driver.state["entries"][0]
    entry.update(status="dispatch-intent", replay_token="crash-before-post", intent_at="2026-11-05T12:00:00Z")
    driver.save()
    assert resume(driver, github, plan).transition()
    assert not github.posts


def test_success_advances_to_explicit_same_source_warm_control_then_next_snapshot(setup):
    driver, github, _ = setup
    assert driver.transition()
    terminal_evidence(driver, github)
    assert driver.transition()
    assert driver.state["current"] == 1
    assert driver.state["entries"][1]["kind"] == "warm-control"
    assert driver.transition()
    assert github.posts[0]["inputs"]["source_sha"] == github.posts[1]["inputs"]["source_sha"]
    assert github.posts[0]["inputs"]["replay_token"] != github.posts[1]["inputs"]["replay_token"]
    terminal_evidence(driver, github)
    assert driver.transition()
    assert driver.transition()
    assert github.posts[-1]["inputs"]["source_sha"] == "c" * 40


def test_mismatched_controller_and_duplicate_token_runs_are_rejected(setup):
    driver, github, _ = setup
    driver.transition()
    github.runs[0]["head_sha"] = "f" * 40
    with pytest.raises(ValueError, match="differs"):
        driver.transition()
    github.runs[0]["head_sha"] = "a" * 40
    github.runs.append(dict(github.runs[0], id=2))
    with pytest.raises(ValueError, match="Duplicate"):
        driver.transition()
    assert len(github.posts) == 1


@pytest.mark.parametrize("wrong_source,conclusion", [(True, "success"), (False, "failure")])
def test_invalid_source_or_unsuccessful_full_run_blocks_until_explicit_matching_acceptance(
    setup, wrong_source, conclusion
):
    driver, github, plan = setup
    driver.transition()
    terminal_evidence(driver, github, wrong_source=wrong_source, conclusion=conclusion)
    assert not driver.transition()
    driver = resume(driver, github, plan)
    assert not driver.transition()
    assert len(github.posts) == 1
    with pytest.raises(ValueError, match="does not match"):
        driver.accept_failure(123, "Investigated")
    with pytest.raises(ValueError, match="reason"):
        driver.accept_failure(1, "  ")
    driver.accept_failure(1, "Retained source/coverage failure; proceed explicitly")
    assert driver.state["entries"][0]["status"] == "accepted-failure"
    assert not driver.state["entries"][0]["outcome"]["valid"]
    assert driver.state["current"] == 1


def test_partial_artifact_download_resumes_without_redispatch_or_unassigned_logs(setup):
    driver, github, plan = setup
    driver.transition()
    terminal_evidence(driver, github)
    github.fail_download = "actions/artifacts/11/zip"
    with pytest.raises(OSError, match="Interrupted"):
        driver.transition()
    driver = resume(driver, github, plan)
    assert driver.transition()
    assert len(github.posts) == 1
    assert github.downloads.count("actions/artifacts/10/zip") == 1
    assert github.downloads.count("actions/artifacts/11/zip") == 2
    assert "actions/jobs/100/logs" not in github.downloads
    execution = json.loads((driver.output / "hosted/1/execution.json").read_text())
    assert [row["execution_seconds"] for row in execution] == [0, 60]


def test_cutoff_stops_new_dispatch_but_keeps_existing_run_evidence(setup, monkeypatch):
    driver, github, _ = setup
    driver.transition()
    terminal_evidence(driver, github)
    monkeypatch.setattr(replay, "_now", lambda: "2026-11-09T17:00:01Z")
    assert driver.transition()
    assert not driver.transition()
    assert len(github.posts) == 1


def test_plan_digest_controller_identity_and_local_lock_fail_closed(setup):
    driver, github, plan = setup
    with pytest.raises(ValueError, match="frozen controller"):
        Driver(repo=driver.repo, plan=plan, output=driver.output, controller_sha="f" * 40, github=github)
    with driver_lock(driver.output), pytest.raises(BlockingIOError), driver_lock(driver.output):
        pass
    value = json.loads(plan.read_text())
    value["snapshots"][0]["source_sha"] = "f" * 40
    plan.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="digest"):
        resume(driver, github, plan)


def test_real_guard_requires_clean_local_and_published_controller(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    git("init", "-q", "-b", BRANCH)
    for key, value in (("user.name", "Test"), ("user.email", "test@example.org"), ("commit.gpgsign", "false")):
        git("config", key, value)
    target = repo / "scripts/pants/full_ci.py"
    target.parent.mkdir(parents=True)
    target.write_text("source\n")
    git("add", ".")
    git("commit", "-qm", "fixture")
    head = git("rev-parse", "HEAD")
    plan = {"snapshots": [{"source_sha": "b" * 40}]}
    plan["plan_id"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan))

    class Remote:
        sha = head

        def api(self, path):
            assert path == f"git/ref/heads/{BRANCH}"
            return {"object": {"sha": self.sha}}

    remote = Remote()
    driver = Driver(repo=repo, plan=plan_file, output=tmp_path / "output", controller_sha=head, github=remote)
    expected = hashlib.sha256(
        (hashlib.sha256(b"source\n").hexdigest() + "  scripts/pants/full_ci.py\n").encode()
    ).hexdigest()
    assert driver.guard() == expected
    target.write_text("changed\n")
    with pytest.raises(ValueError, match="not pristine"):
        driver.guard()
    target.write_text("source\n")
    remote.sha = "f" * 40
    with pytest.raises(ValueError, match="Published"):
        driver.guard()
    remote.sha = head
    git("checkout", "-qb", "other")
    with pytest.raises(ValueError, match="branch/HEAD"):
        driver.guard()


def test_missing_lock_assets_aborts_before_persisting_dispatch_intent(setup, monkeypatch):
    driver, github, _ = setup

    def unsupported(self, entry):
        raise ValueError("No validated dependency bundle")

    monkeypatch.setattr(Driver, "preflight", unsupported)
    with pytest.raises(ValueError, match="dependency bundle"):
        driver.transition()
    assert not github.posts
    assert driver.state["entries"][0]["status"] == "pending"


@pytest.mark.parametrize("workflow", ["pants-full-ci.yml", "pants-core0-diagnostic.yml", "pants-pilot.yml"])
def test_other_active_pilot_workflows_block_new_dispatch(setup, monkeypatch, workflow):
    driver, github, _ = setup

    def active(path, **kwargs):
        assert path.startswith("actions/runs?branch=")
        return [{"workflow_runs": [{"id": 40, "path": f".github/workflows/{workflow}", "status": "queued"}]}]

    monkeypatch.setattr(github, "api", active)
    value = REAL_READINESS(driver)
    assert not value["ready"] and value["run_ids"] == [40]


def test_arc_stalled_queue_and_unknown_once_are_distinguished_from_fresh_assignment(setup, monkeypatch):
    driver, github, _ = setup
    jobs = [{"id": 40, "labels": ["nemo-ci-aws-cpu-4-16"], "status": "queued", "runner_id": 0}]

    def main_jobs(path, **kwargs):
        if path.startswith("actions/runs?branch="):
            return [{"workflow_runs": []}]
        if path == "actions/workflows/cicd-main.yml/runs?branch=main&per_page=2":
            return {"workflow_runs": [{"id": 20, "created_at": "2026-11-05T10:00:00Z"}]}
        return [{"jobs": jobs}]

    monkeypatch.setattr(github, "api", main_jobs)
    assert REAL_READINESS(driver)["state"] == "stalled-unassigned-arc"
    assert not REAL_READINESS(driver)["ready"]
    jobs[0].update(
        status="completed", runner_id=5, started_at="2026-11-05T08:00:00Z", completed_at="2026-11-05T11:59:00Z"
    )
    assert REAL_READINESS(driver)["state"] == "recent-arc-assignment"
    assert REAL_READINESS(driver)["ready"]
    jobs.clear()
    assert REAL_READINESS(driver) == {
        "ready": True,
        "state": "unknown",
        "reason": "No recent ARC assignment; permit at most one bounded attempt",
    }
    driver.state["unknown_readiness_dispatches"] = 1
    assert not REAL_READINESS(driver)["ready"]


def test_artifact_path_escape_fails_without_writing_outside_evidence(setup):
    driver, github, _ = setup
    driver.transition()
    terminal_evidence(driver, github)
    github.blobs["actions/artifacts/10/zip"] = archive({"../../escaped.json": {"unexpected": True}})
    with pytest.raises(ValueError, match="Unsafe artifact member"):
        driver.transition()
    assert not list(driver.output.rglob("escaped.json"))
    assert len(github.posts) == 1


def test_unresolved_dispatch_deadline_is_persisted_and_late_run_recovers_after_cutoff(setup, monkeypatch):
    driver, github, plan = setup
    driver.transition()
    github.visible = False
    assert driver.state["entries"][0]["reconcile_until"] == "2026-11-05T12:15:00Z"
    monkeypatch.setattr(replay, "_now", lambda: "2026-11-05T12:15:00Z")
    driver = resume(driver, github, plan)
    assert not driver.transition()
    assert driver.state["entries"][0]["status"] == "dispatch-unresolved"
    assert "never be resent" in driver.state["entries"][0]["reconciliation_error"]
    assert len(github.posts) == 1
    monkeypatch.setattr(replay, "_now", lambda: "2026-11-09T17:00:01Z")
    driver = resume(driver, github, plan)
    assert not driver.transition()
    github.visible = True
    terminal_evidence(driver, github)
    assert driver.transition()
    assert driver.state["entries"][0]["status"] == "passed"
    assert "reconciliation_error" not in driver.state["entries"][0]
    assert not driver.transition()  # Recovery remains possible, but no new post after cutoff.
    assert len(github.posts) == 1


def test_existing_intent_without_deadline_does_not_restart_reconciliation_clock(setup, monkeypatch):
    driver, github, plan = setup
    driver.transition()
    driver.state["entries"][0].pop("reconcile_until")
    driver.save()
    github.visible = False
    monkeypatch.setattr(replay, "_now", lambda: "2026-11-05T14:00:00Z")
    driver = resume(driver, github, plan)
    assert not driver.transition()
    assert driver.state["entries"][0]["reconcile_until"] == "2026-11-05T12:15:00Z"
    assert len(github.posts) == 1


def test_new_controller_generation_has_isolated_caches(setup, tmp_path):
    driver, github, plan = setup
    replacement = Driver(
        repo=driver.repo, plan=plan, output=tmp_path / "replacement", controller_sha="e" * 40, github=github
    )
    assert replacement.state["plan_id"] == driver.state["plan_id"]
    assert replacement.state["cache_namespace"] != driver.state["cache_namespace"]
    assert resume(driver, github, plan).state["cache_namespace"] == driver.state["cache_namespace"]
