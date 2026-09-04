# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from resources_servers.swebench_pro.verification import (
    APPLY_PATCH_FUNCTION,
    DEFAULT_ENVIRONMENT_REPAIRS,
    ENVIRONMENT_REPAIRS,
    SCRUB_HARNESS_ENV,
    VerificationInputs,
    assemble_workspace_files,
    build_environment_repairs,
    build_seed_normalization,
    create_entryscript,
    drop_patch_sections,
    grade_output,
    inconclusive_reason,
    parse_string_list,
    patch_section_path,
    run_verification,
    strip_binary_hunks,
)


DEFAULT_REPAIR_SCRIPT = build_environment_repairs(DEFAULT_ENVIRONMENT_REPAIRS)


def make_inputs(**overrides) -> VerificationInputs:
    values = {
        "instance_id": "instance_test",
        "base_commit": "abc123",
        "patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n",
        "run_script": "#!/bin/bash\nexit 0\n",
        "parser_script": "import json\n",
        "selected_test_files_to_run": '["a.py", "b.py"]',
        "fail_to_pass": '["test_new"]',
        "pass_to_pass": '["test_old"]',
    }
    values.update(overrides)
    return VerificationInputs(**values)


def test_parse_string_list_accepts_json_python_and_lists() -> None:
    assert parse_string_list('["a", "b"]') == ["a", "b"]
    assert parse_string_list("['a', 'b']") == ["a", "b"]
    assert parse_string_list(["a"]) == ["a"]


def test_parse_string_list_rejects_non_string_items() -> None:
    with pytest.raises(ValueError):
        parse_string_list("[1]")


def test_strip_binary_hunks_preserves_text_diffs() -> None:
    patch = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
        "diff --git a/image.png b/image.png\nGIT binary patch\nliteral 1\nA\n"
    )
    assert strip_binary_hunks(patch) == "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"


def test_create_entryscript_matches_upstream_contract() -> None:
    inputs = make_inputs(
        before_repo_set_cmd="ignored setup line\nnpm install",
        base_dockerfile="ENV FOO=bar\n",
    )
    script = create_entryscript(asdict(inputs))
    assert "git reset --hard abc123" in script
    assert "git apply -v --whitespace=nowarn /workspace/patch.diff" in script
    assert "apply_patch\n" in script
    assert "npm install" in script
    assert "bash /workspace/run_script.sh a.py,b.py" in script
    assert "export FOO=bar" in script
    assert "go mod download" not in script


def test_create_entryscript_prefetches_go_modules_before_tests() -> None:
    inputs = make_inputs(
        run_script="#!/bin/bash\nset -e\ngo test ./...\n",
        repo_language="Go",
        prefetch_go_modules=True,
    )

    script = create_entryscript(asdict(inputs))

    assert "go mod download" in script
    assert script.index("git apply") < script.index("go mod download") < script.index("run_script.sh")


def test_create_entryscript_does_not_prefetch_non_go_tasks() -> None:
    inputs = make_inputs(repo_language="Python", prefetch_go_modules=True)

    assert "go mod download" not in create_entryscript(asdict(inputs))


def test_assemble_workspace_files_embeds_prepared_assets() -> None:
    inputs = make_inputs(patch="diff --git a/image.png b/image.png\nGIT binary patch\nliteral 1\nA\n")
    files, entryscript = assemble_workspace_files(inputs.instance_id, None, inputs.patch, asdict(inputs))

    assert files["patch.diff"] == ""
    assert files["run_script.sh"] == inputs.run_script
    assert files["parser.py"] == inputs.parser_script
    assert files["entryscript.sh"] == entryscript


def test_grade_output_requires_all_named_tests() -> None:
    output = {
        "tests": [
            {"name": "test_new", "status": "PASSED"},
            {"name": "test_old", "status": "PASSED"},
        ]
    }
    assert grade_output(output, asdict(make_inputs()))
    assert not grade_output(output, asdict(make_inputs(fail_to_pass='["missing"]')))
    assert grade_output(output, asdict(make_inputs(fail_to_pass="[]", pass_to_pass="[]")))


@pytest.mark.asyncio
async def test_run_verification_returns_resolved_result(tmp_path) -> None:
    output = {
        "tests": [
            {"name": "test_new", "status": "PASSED"},
            {"name": "test_old", "status": "PASSED"},
        ]
    }
    sandbox = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout="", stderr=""),
                SimpleNamespace(return_code=0, stdout="run", stderr=""),
                SimpleNamespace(return_code=0, stdout="test stdout", stderr=""),
                SimpleNamespace(return_code=0, stdout="test stderr", stderr=""),
                SimpleNamespace(return_code=0, stdout="0\n", stderr=""),
                SimpleNamespace(return_code=0, stdout=json.dumps(output), stderr=""),
            ]
        )
    )

    result = await run_verification(sandbox, make_inputs(), tmp_path, timeout_s=30)

    assert result.completed
    assert result.resolved
    assert result.patch_applied
    assert result.test_output == "STDOUT:\ntest stdout\n\nSTDERR:\ntest stderr"
    assert json.loads((tmp_path / "output.json").read_text()) == output


@pytest.mark.asyncio
async def test_run_verification_rejects_malformed_parser_output(tmp_path) -> None:
    sandbox = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout="", stderr=""),
                SimpleNamespace(return_code=1, stdout="", stderr="failed"),
                SimpleNamespace(return_code=0, stdout="", stderr=""),
                SimpleNamespace(return_code=0, stdout="failed", stderr=""),
                SimpleNamespace(return_code=0, stdout="1\n", stderr=""),
                SimpleNamespace(return_code=0, stdout="not-json", stderr=""),
            ]
        )
    )

    result = await run_verification(sandbox, make_inputs(), tmp_path, timeout_s=30)

    assert not result.completed
    assert not result.resolved
    assert result.test_output == "STDOUT:\n\n\nSTDERR:\nfailed"
    assert "invalid JSON" in result.error


def test_patch_section_path_reads_adds_deletes_and_binaries() -> None:
    added = (
        "diff --git a/dump.rdb b/dump.rdb\nnew file mode 100644\n--- /dev/null\n+++ b/dump.rdb\n@@ -0,0 +1 @@\n+x\n"
    )
    deleted = (
        "diff --git a/gone.py b/gone.py\ndeleted file mode 100644\n--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
    )
    binary = "diff --git a/logo.png b/logo.png\nnew file mode 100644\nBinary files /dev/null and b/logo.png differ\n"
    spaced = "diff --git a/my file.txt b/my file.txt\n--- a/my file.txt\n+++ b/my file.txt\n@@ -1 +1 @@\n-a\n+b\n"

    assert patch_section_path(added) == "dump.rdb"
    assert patch_section_path(deleted) == "gone.py"
    assert patch_section_path(binary) == "logo.png"
    assert patch_section_path(spaced) == "my file.txt"


def test_drop_patch_sections_keeps_the_fix_and_removes_named_artifacts() -> None:
    artifact = (
        "diff --git a/dump.rdb b/dump.rdb\nnew file mode 100644\n--- /dev/null\n+++ b/dump.rdb\n@@ -0,0 +1 @@\n+x\n"
    )
    fix = "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-old\n+new\n"

    assert drop_patch_sections(artifact + fix, {"dump.rdb"}) == fix
    assert drop_patch_sections(artifact + fix, set()) == artifact + fix
    assert drop_patch_sections(artifact + fix, {"src/app.py"}) == artifact
    assert drop_patch_sections("", {"dump.rdb"}) == ""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


def _make_repo(path: Path) -> str:
    """Build a task image: one tracked source file plus an untracked runtime artifact."""
    (path / "src").mkdir(parents=True)
    (path / "src" / "sorted.js").write_text("module.exports = OLD;\n")
    _git(path, "init", "-q", ".")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    _git(path, "add", "-A")
    _git(path, "-c", "commit.gpgsign=false", "commit", "-qm", "base")
    (path / "appendonlydir").mkdir()
    (path / "appendonlydir" / "appendonly.aof.manifest").write_text("file base seq 1 type b\n")
    return _git(path, "rev-parse", "HEAD").stdout.strip()


def _run_apply(eval_repo: Path, workspace: Path, patch: str) -> tuple[int, str]:
    (workspace / "patch.diff").write_text(patch)
    script = APPLY_PATCH_FUNCTION.replace("/workspace", str(workspace)) + "\napply_patch\n"
    done = subprocess.run(["bash", "-c", script], cwd=eval_repo, capture_output=True, text=True, check=False)
    return done.returncode, done.stdout + done.stderr


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_apply_patch_recovers_the_fix_when_a_container_artifact_already_exists(tmp_path) -> None:
    """An untracked artifact the image ships must not reject the agent's real fix.

    `git add -N . && git diff` reports it as a new file; the evaluation container is a
    fresh copy of the same image, so `git apply` fails "already exists in working
    directory" and, being all-or-nothing, drops the source fix too.
    """
    agent = tmp_path / "agent"
    base = _make_repo(agent)
    eval_repo = tmp_path / "eval"
    shutil.copytree(agent, eval_repo)
    pristine_artifact = (eval_repo / "appendonlydir" / "appendonly.aof.manifest").read_text()

    (agent / "src" / "sorted.js").write_text("module.exports = FIXED;\n")
    (agent / "appendonlydir" / "appendonly.aof.manifest").write_text("file base seq 2 type b\n")
    _git(agent, "add", "-N", ".")
    patch = _git(agent, "--no-pager", "diff", base).stdout
    assert "appendonlydir/appendonly.aof.manifest" in patch, "fixture must reproduce the sweep-in"

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    status, log = _run_apply(eval_repo, workspace, patch)

    assert status == 0, log
    assert (eval_repo / "src" / "sorted.js").read_text() == "module.exports = FIXED;\n"
    # The artifact belongs to the container, not the agent: it must be left alone.
    assert (eval_repo / "appendonlydir" / "appendonly.aof.manifest").read_text() == pristine_artifact


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_apply_patch_reports_empty_and_unappliable_patches(tmp_path) -> None:
    agent = tmp_path / "agent"
    base = _make_repo(agent)
    eval_repo = tmp_path / "eval"
    shutil.copytree(agent, eval_repo)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # An agent that changed nothing did not fail to apply anything.
    empty_status, empty_log = _run_apply(eval_repo, workspace, "")
    assert empty_status == 0, empty_log
    assert (eval_repo / "src" / "sorted.js").read_text() == "module.exports = OLD;\n"

    unappliable = (
        "diff --git a/src/sorted.js b/src/sorted.js\n"
        "index 1111111..2222222 100644\n"
        "--- a/src/sorted.js\n"
        "+++ b/src/sorted.js\n"
        "@@ -1,2 +1,2 @@\n"
        " context that is not in the file\n"
        "-neither is this\n"
        "+nor this\n"
    )
    broken_status, _ = _run_apply(eval_repo, workspace, unappliable)
    assert broken_status == 1
    assert (eval_repo / "src" / "sorted.js").read_text() == "module.exports = OLD;\n"
    assert base


def test_entryscript_scrubs_harness_env_before_the_image_env() -> None:
    """A Dockerfile ENV must still win over the scrub, so the scrub has to come first."""
    script = create_entryscript(asdict(make_inputs(base_dockerfile="ENV OTEL_SERVICE_NAME=flipt\n")))

    assert script.index(SCRUB_HARNESS_ENV) < script.index("export OTEL_SERVICE_NAME=flipt")


def test_scrub_harness_env_unsets_only_telemetry_variables(tmp_path) -> None:
    probe = tmp_path / "probe.sh"
    probe.write_text(
        SCRUB_HARNESS_ENV + '\necho "svc=${OTEL_SERVICE_NAME-unset} '
        'endpoint=${OTEL_EXPORTER_OTLP_ENDPOINT-unset} keep=${KEEP_ME-unset}"\n'
    )
    done = subprocess.run(
        ["bash", str(probe)],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": "/usr/bin:/bin",
            "OTEL_SERVICE_NAME": "execd",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://10.0.0.1:4318",
            "KEEP_ME": "yes",
        },
    )

    assert done.stdout.strip() == "svc=unset endpoint=unset keep=yes", done.stderr


def test_entryscript_warms_the_test_environment_before_the_tests_run() -> None:
    script = create_entryscript(asdict(make_inputs()))

    assert script.index(DEFAULT_REPAIR_SCRIPT) < script.index("bash /workspace/run_script.sh")
    assert script.index("apply_patch\n") < script.index(DEFAULT_REPAIR_SCRIPT)


def test_warm_test_environment_leaves_no_x_server_behind(tmp_path) -> None:
    """A left-running warm-up server makes Xvfb reject any task that wants that display."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    started = tmp_path / "starts"
    # Stand in for Xvfb: create the lock the block waits on, then idle until killed.
    xvfb_stub = stub_dir / "Xvfb"
    xvfb_stub.write_text(
        f'#!/bin/sh\ndisplay="${{1#:}}"\n: > "/tmp/.X${{display}}-lock"\necho "$display" >> "{started}"\nsleep 60\n'
    )
    xvfb_stub.chmod(0o755)
    bash = shutil.which("bash") or "/bin/bash"
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    done = subprocess.run(
        [bash, "-c", DEFAULT_REPAIR_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
        # An empty cwd, or the dependency warm-up pages in the test runner's own tree.
        cwd=scratch,
        env={"PATH": f"{stub_dir}:/usr/bin:/bin"},
    )

    assert done.returncode == 0, done.stderr
    assert started.exists(), "the warm-up never started Xvfb"
    display = started.read_text().strip()
    # It must not squat on :99, which the qutebrowser tasks' own run_script asks for...
    assert display != "99"
    # ...and it must clean up the display it did take.
    assert not Path(f"/tmp/.X{display}-lock").exists(), f"warm-up left :{display} behind"


def _result(**overrides) -> VerificationInputs:
    from resources_servers.swebench_pro.verification import VerificationResult

    values = {
        "completed": True,
        "resolved": False,
        "patch_applied": True,
        "test_results": {"tests": []},
    }
    values.update(overrides)
    return VerificationResult(**values)


def test_inconclusive_reason_treats_a_failing_test_as_a_verdict() -> None:
    """The retry hangs off this: re-running FAILED tests would be re-rolling for a nicer answer."""
    sample = asdict(make_inputs(fail_to_pass='["test_new"]', pass_to_pass='["test_old"]'))
    ran_and_failed = {"tests": [{"name": "test_new", "status": "FAILED"}, {"name": "test_old", "status": "PASSED"}]}
    all_passed = {"tests": [{"name": "test_new", "status": "PASSED"}, {"name": "test_old", "status": "PASSED"}]}

    assert inconclusive_reason(_result(test_results=ran_and_failed), sample) is None
    assert inconclusive_reason(_result(test_results=all_passed), sample) is None


def test_inconclusive_reason_flags_runs_that_produced_no_verdict() -> None:
    sample = asdict(make_inputs(fail_to_pass='["test_new"]', pass_to_pass='["test_old"]'))
    only_one = {"tests": [{"name": "test_old", "status": "PASSED"}]}

    assert "never reported an outcome" in inconclusive_reason(_result(test_results=only_one), sample)
    assert "no tests at all" in inconclusive_reason(_result(), sample)
    assert "no usable output" in inconclusive_reason(_result(test_results=None), sample)
    assert "did not complete" in inconclusive_reason(_result(completed=False, error="OOM"), sample)


def test_inconclusive_reason_ignores_tests_the_task_does_not_grade_on() -> None:
    """Suites report far more than the graded set; only the graded set decides the reward."""
    sample = asdict(make_inputs(fail_to_pass='["test_new"]', pass_to_pass="[]"))
    extra = {"tests": [{"name": "test_new", "status": "PASSED"}, {"name": "unrelated", "status": "FAILED"}]}

    assert inconclusive_reason(_result(test_results=extra), sample) is None


def test_environment_repairs_are_individually_selectable() -> None:
    """Each repair must be droppable on its own, so it can be justified by measurement."""
    markers = {
        "stale_database_state": "ng_stale",
        "core_dump_limit": "ulimit -c 0",
        "dependency_cache": "ng_warm",
        "xvfb_cache": "Xvfb :121",
    }
    assert set(markers) == set(ENVIRONMENT_REPAIRS), "a repair was added without a leave-one-out marker"

    for dropped in DEFAULT_ENVIRONMENT_REPAIRS:
        kept = tuple(r for r in DEFAULT_ENVIRONMENT_REPAIRS if r != dropped)
        script = create_entryscript(asdict(make_inputs(environment_repairs=kept)))
        assert markers[dropped] not in script, f"{dropped} survived being dropped"
        for other in kept:
            assert markers[other] in script, f"dropping {dropped} also removed {other}"


def test_environment_repairs_default_and_disable_are_distinguishable() -> None:
    default = create_entryscript(asdict(make_inputs()))
    none_at_all = create_entryscript(asdict(make_inputs(environment_repairs=())))

    assert "Xvfb :121" in default, "None must mean the default set, not 'no repairs'"
    assert "all environment repairs disabled" in none_at_all
    assert "Xvfb :121" not in none_at_all


def test_build_environment_repairs_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown environment repairs"):
        build_environment_repairs(["xvfb_cache", "not_a_repair"])


def test_seed_normalization_only_carries_repairs_that_outlive_one_command() -> None:
    """`ulimit` and exported vars die with the shell, so they cannot help the agent here."""
    script = build_seed_normalization(DEFAULT_ENVIRONMENT_REPAIRS)

    assert "Xvfb :121" in script
    assert "ng_warm" in script
    # Nothing that dies with the shell that set it belongs here.
    assert "ulimit -c 0" not in script
    assert "dns-result-order" not in script
