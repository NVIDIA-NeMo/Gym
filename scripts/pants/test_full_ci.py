# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import os
import subprocess
from pathlib import Path

import full_ci
import pytest


ROOT = Path(__file__).resolve().parents[2]


def repository(tmp_path, monkeypatch, *, command="exit 0\n"):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    Path("scripts/ci").mkdir(parents=True)
    script = Path("scripts/ci/core_unit_tests.sh")
    script.write_text("#!/bin/bash\nset -eu\n" + command)
    script.chmod(0o755)
    Path("tracked.yaml").write_text("fixture: original\n")
    Path("pyproject.toml").write_text("# fixture\n")
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return tmp_path


def invoke(monkeypatch, *, output="results/full-ci", engine="baseline", job="core"):
    monkeypatch.setattr(
        "sys.argv",
        [
            "full_ci.py",
            "--engine",
            engine,
            "--job",
            job,
            "--shard",
            "2",
            "--shards",
            "4",
            "--output",
            output,
        ],
    )
    return full_ci.main()


def test_server_discovery_matches_sampled_constants_and_partitions_every_module(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = Path("nemo_gym/config/global_config.py")
    config.parent.mkdir(parents=True)
    config.write_text((ROOT / config).read_text())
    types = full_ci.server_types()
    assert "environment_servers" in types
    expected = []
    for server_type in types:
        for name in ("aaa", "zzz"):
            path = Path(server_type) / name
            path.mkdir(parents=True)
            (path / "README.md").write_text("testable module")
            expected.append(str(path))
        # Match test_all's exclusions for caches, packaging metadata and missing README.
        for name in ("__pycache__", "no-readme", "example.egg-info", "example.dist-info"):
            path = Path(server_type) / name
            path.mkdir()
            if name != "no-readme":
                (path / "README.md").write_text("excluded")
    shards = [full_ci.server_scope("servers", shard=i, shards=12, changed_files="") for i in range(12)]
    assert sorted(server for shard in shards for server in shard) == sorted(expected)
    assert len({server for shard in shards for server in shard}) == len(expected)
    assert shards[-1] == []


@pytest.mark.parametrize("changed", ["", "README.md", "resources_servers/../x", "unknown/server/a.py"])
def test_invalid_changed_scope_fails_closed(tmp_path, monkeypatch, changed):
    monkeypatch.chdir(tmp_path)
    config = Path("nemo_gym/config/global_config.py")
    config.parent.mkdir(parents=True)
    config.write_text((ROOT / config).read_text())
    with pytest.raises(ValueError, match="valid changed"):
        full_ci.server_scope("changed-servers", shard=0, shards=1, changed_files=changed)


def test_added_sampled_server_type_is_not_silently_omitted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = Path("nemo_gym/config/global_config.py")
    config.parent.mkdir(parents=True)
    config.write_text('EXTRA = "future_servers"\nSERVER_TYPE_KEYS: tuple[str, ...] = (EXTRA,)\n')
    server = Path("future_servers/example")
    server.mkdir(parents=True)
    (server / "README.md").write_text("example")
    assert full_ci.server_scope("servers", shard=0, shards=1, changed_files="") == [str(server)]


def test_original_command_failure_and_report_paths_are_preserved(tmp_path, monkeypatch):
    repository(
        tmp_path,
        monkeypatch,
        command="""
[[ "$1" == 2 && "$2" == 4 ]]
[[ "$UV_PYTHON_INSTALL_DIR" == /fixture/python ]]
[[ "$LD_LIBRARY_PATH" == /fixture/lib ]]
[[ "$JAVA_HOME_17_X64" == /fixture/jdk ]]
[[ "$GITHUB_SHA" == "$(git rev-parse HEAD)" ]]
[[ -z "${GITHUB_TOKEN:-}" ]]
printf 'NLTK_DATA=/fixture/nltk\\n' >> "$GITHUB_ENV"
mkdir -p "$GYM_CI_JUNIT_DIR"
printf coverage > "$COVERAGE_FILE"
printf '<testsuite />' > "$GYM_CI_JUNIT_DIR/core.xml"
echo failing command diagnostic
exit 7
""",
    )
    env_file = tmp_path / "github-env"
    monkeypatch.setenv("GITHUB_ENV", str(env_file))
    monkeypatch.setenv("UV_PYTHON_INSTALL_DIR", "/fixture/python")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/fixture/lib")
    monkeypatch.setenv("JAVA_HOME_17_X64", "/fixture/jdk")
    monkeypatch.setenv("GITHUB_SHA", "controller-revision")
    monkeypatch.setenv("GITHUB_TOKEN", "not-forwarded")
    assert invoke(monkeypatch, output="results/with spaces") == 7
    output = Path("results/with spaces")
    assert (output / ".coverage.core-2").read_text() == "coverage"
    assert (output / "junit/core.xml").is_file()
    assert "failing command diagnostic" in (output / "runner.log").read_text()
    assert env_file.read_text() == "NLTK_DATA=/fixture/nltk\n"
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["command_exit_code"] == metadata["exit_code"] == 7
    assert not metadata["source_changed"]


def test_successful_command_that_mutates_tracked_source_invalidates_measurement(tmp_path, monkeypatch):
    repository(tmp_path, monkeypatch, command="echo changed >> tracked.yaml\n")
    assert invoke(monkeypatch) == 1
    metadata = json.loads(Path("results/full-ci/metadata.json").read_text())
    assert metadata["command_exit_code"] == 0
    assert metadata["exit_code"] == 1
    assert metadata["source_changed"]


def test_dirty_checkout_is_rejected_before_execution(tmp_path, monkeypatch):
    repository(tmp_path, monkeypatch)
    Path("tracked.yaml").write_text("local edits")
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch)
    assert error.value.code == 2
    assert not Path("results/full-ci").exists()


def test_manifest_covers_configuration_and_rejects_missing_tracked_input(tmp_path, monkeypatch):
    repository(tmp_path, monkeypatch)
    files, initial = full_ci.source_snapshot()
    assert "tracked.yaml" in files
    Path("tracked.yaml").write_text("fixture: modified\n")
    assert full_ci.source_snapshot()[1] != initial
    Path("tracked.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="tracked.yaml"):
        full_ci.source_snapshot()


def test_git_directory_file_and_dangling_symlinks_are_fingerprinted(tmp_path, monkeypatch):
    repository(tmp_path, monkeypatch)
    target = Path(".agents/skills/example")
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("tracked skill contents")
    links = Path(".claude/skills")
    links.mkdir(parents=True)
    (links / "example").symlink_to("../../.agents/skills/example", target_is_directory=True)
    Path("file-link").symlink_to("tracked.yaml")
    Path("broken-link").symlink_to("missing-target")
    subprocess.run(["git", "add", "."], check=True)
    sources, digest = full_ci.source_snapshot()
    assert str(links / "example") in sources
    assert "broken-link" in sources
    assert "file-link" in sources
    assert str(target / "SKILL.md") in sources
    (target / "SKILL.md").write_text("updated tracked referent")
    assert full_ci.source_snapshot()[1] != digest
    _, updated = full_ci.source_snapshot()
    Path("broken-link").unlink()
    Path("broken-link").symlink_to("different-missing-target")
    assert full_ci.source_snapshot()[1] != updated


@pytest.mark.parametrize("existing_config", [False, True])
def test_pants_failure_preserves_status_environment_and_cleans_only_owned_files(
    tmp_path, monkeypatch, existing_config
):
    repository(tmp_path, monkeypatch)
    output = Path("results/probe")
    output.mkdir(parents=True)
    script = output / "command.sh"
    script.write_text("echo expected failure\nexit 9\n")
    if existing_config:
        Path("pants.toml").write_text("# existing configuration\n")
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.setenv("PANTS_LOCAL_STORE_DIR", str(tmp_path / "store"))
    actual_run = full_ci.run_logged

    def emulate_pants(command, *, log, env):
        assert env["PANTS_LOCAL_STORE_DIR"] == str(tmp_path / "store")
        parsed = ast.parse(Path("BUILD.pants-full-ci").read_text())
        target = parsed.body[1].value
        fields = {kw.arg: ast.literal_eval(kw.value) for kw in target.keywords}
        assert fields["cache_scope"] == "session"
        assert fields["workspace_invalidation_sources"] == []
        assert "GITHUB_ENV" in fields["extra_env_vars"]
        assert "UV_PYTHON_INSTALL_DIR" in fields["extra_env_vars"]
        assert "GITHUB_TOKEN" not in fields["extra_env_vars"]
        wrapped = fields["command"].replace("{chroot}", str(sandbox))
        return actual_run(["bash", "-c", wrapped], log=log, env=env)

    monkeypatch.setattr(full_ci, "run_logged", emulate_pants)
    env = {"PATH": os.environ["PATH"], "GITHUB_ENV": str(tmp_path / "env"), "UV_PYTHON_INSTALL_DIR": "/python"}
    assert full_ci.run_pants(script=script, output=output, env=env) == 9
    assert not Path("BUILD.pants-full-ci").exists()
    assert Path("pants.toml").exists() == existing_config
    if existing_config:
        assert Path("pants.toml").read_text() == "# existing configuration\n"
    assert "expected failure" in (output / "runner.log").read_text()


@pytest.mark.parametrize("change_content", [False, True])
def test_workspace_rewrite_keeps_input_digest_and_rejects_content_changes(tmp_path, monkeypatch, change_content):
    repository(tmp_path, monkeypatch)
    # The real-Pants probe separately checks watcher scheduling. Here, assert the
    # generated process contract and exercise the actual post-execution guard.
    Path("scripts/ci/lint.sh").write_text(
        "echo execution >> results/executions\n"
        + ("echo changed >> tracked.yaml\n" if change_content else "touch tracked.yaml\n")
    )
    Path("scripts/ci/lint.sh").chmod(0o755)
    _, original_digest = full_ci.source_snapshot()
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    actual_run = full_ci.run_logged

    def execute_generated_command(command, *, log, env):
        parsed = ast.parse(Path("BUILD.pants-full-ci").read_text())
        fields = {kw.arg: ast.literal_eval(kw.value) for kw in parsed.body[1].value.keywords}
        assert fields["cache_scope"] == "session"
        assert fields["workspace_invalidation_sources"] == []
        assert "GYM_PANTS_INPUT_DIGEST" in fields["extra_env_vars"]
        assert env["GYM_PANTS_INPUT_DIGEST"] == original_digest
        return actual_run(
            ["bash", "-c", fields["command"].replace("{chroot}", str(sandbox))],
            log=log,
            env=env,
        )

    monkeypatch.setattr(full_ci, "run_logged", execute_generated_command)
    assert invoke(monkeypatch, engine="pants", job="lint") == int(change_content)
    metadata = json.loads(Path("results/full-ci/metadata.json").read_text())
    assert metadata["command_exit_code"] == 0
    assert metadata["source_changed"] is change_content
    assert Path("results/executions").read_text().splitlines() == ["execution"]


def test_pants_launcher_exception_records_failure_and_cleans_bootstrap(tmp_path, monkeypatch):
    repository(tmp_path, monkeypatch)
    monkeypatch.setenv("PANTS_BIN", str(tmp_path / "does-not-exist"))
    with pytest.raises(FileNotFoundError):
        invoke(monkeypatch, engine="pants", job="lint")
    metadata = json.loads(Path("results/full-ci/metadata.json").read_text())
    assert metadata["exit_code"] != 0
    assert not Path("BUILD.pants-full-ci").exists()
    assert not Path("pants.toml").exists()


@pytest.mark.parametrize("job", ["core", "sandbox"])
def test_native_groups_preserve_entire_requested_shard_and_never_fallback(tmp_path, monkeypatch, job):
    repository(tmp_path, monkeypatch)
    seen = []

    def native(**kwargs):
        seen.append(kwargs)
        return 7

    def forbidden(**kwargs):
        pytest.fail("Native failure must not silently fall back to workspace execution")

    monkeypatch.setattr(full_ci, "run_native", native)
    monkeypatch.setattr(full_ci, "run_pants", forbidden)
    assert invoke(monkeypatch, engine="pants", job=job) == 7
    assert len(seen) == 1
    assert seen[0]["job"] == job
    assert seen[0]["execution_mode"] == "incremental"
    assert (seen[0]["shard"], seen[0]["shards"]) == (2, 4)
    metadata = json.loads(Path("results/full-ci/metadata.json").read_text())
    assert metadata["implementation"] == "native Python test targets"
    assert metadata["execution_provenance"] == "native/metadata.json"
    assert metadata["command_exit_code"] == metadata["exit_code"] == 7


@pytest.mark.parametrize("execution_mode", ["incremental", "forced"])
@pytest.mark.parametrize("missing_coverage", [False, True])
def test_native_artifacts_preserve_coverage_and_junit_and_require_complete_success(
    tmp_path, monkeypatch, missing_coverage, execution_mode
):
    repository(tmp_path, monkeypatch)
    output = Path("results/full-ci")
    (output / "junit").mkdir(parents=True)
    monkeypatch.setenv("PANTS_BIN", "/test/pants")
    monkeypatch.setattr(full_ci.shutil, "which", lambda *args, **kwargs: "/test/uv")

    def native_process(command, *, log, env):
        assert command[command.index("--execution-mode") + 1] == execution_mode
        assert command[command.index("--source") + 1] == str(tmp_path)
        assert command[command.index("--group") + 1] == "core"
        assert command[command.index("--shard") + 1] == "2"
        assert command[command.index("--shards") + 1] == "4"
        assert "GITHUB_TOKEN" not in env
        native = Path(command[command.index("--output") + 1])
        (native / "junit").mkdir(parents=True)
        (native / "junit/core.xml").write_text("all native testcase occurrences")
        (native / "metadata.json").write_text(
            json.dumps({"exit_code": 0, "source_sha": "sample", "execution_mode": execution_mode})
        )
        if not missing_coverage:
            (native / ".coverage.core-2").write_bytes(b"raw coverage database")
        return 0

    monkeypatch.setattr(full_ci, "run_logged", native_process)
    if missing_coverage:
        with pytest.raises(ValueError, match="without required"):
            full_ci.run_native(
                job="core",
                shard=2,
                shards=4,
                output=output,
                env={"GITHUB_SHA": "sample"},
                execution_mode=execution_mode,
            )
    else:
        assert (
            full_ci.run_native(
                job="core",
                shard=2,
                shards=4,
                output=output,
                env={"GITHUB_SHA": "sample"},
                execution_mode=execution_mode,
            )
            == 0
        )
        assert (output / ".coverage.core-2").read_bytes() == b"raw coverage database"
        assert (output / "junit/core.xml").read_text() == "all native testcase occurrences"
