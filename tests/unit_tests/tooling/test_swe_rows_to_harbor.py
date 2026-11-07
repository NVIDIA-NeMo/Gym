# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""scripts/swe_rows_to_harbor: converted tasks load as Harbor tasks, reuse the servers' commands, and grade alike."""

import asyncio
import base64
import importlib.util
import json
import re
import shlex
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from scripts.swe_rows_to_harbor import convert

from nemo_gym.data.tasks.harbor.dockerfile import RECORD_VERSION
from nemo_gym.data.tasks.harbor.task import load_task
from resources_servers.swebench.anti_cheat import ANTI_CHEAT_SCRIPT_FPATH, apply_anti_cheat_setup
from resources_servers.swebench.patch_capture import capture_model_patch, prepare_git_for_commits


REPO_ROOT = Path(__file__).resolve().parents[3]

GOLDEN_PATCH = (
    "diff --git a/widget/core.py b/widget/core.py\n--- a/widget/core.py\n+++ b/widget/core.py\n"
    "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
)
TEST_FILE_PATCH = (
    "diff --git a/tests/test_core.py b/tests/test_core.py\n--- a/tests/test_core.py\n+++ b/tests/test_core.py\n"
    "@@ -1 +1 @@\n-old = True\n+old = False\n"
)
PRISTINE_FILE_PATCH = "diff --git a/notes.txt b/notes.txt\nnew file mode 100644\n--- /dev/null\n+++ b/notes.txt\n@@ -0,0 +1 @@\n+scratch\n"
SCALE_ROW = {
    "instance_id": "octo_widget_pr7",
    "workdir": "/workspace/widget",
    "image_url": "aweaiteam/scaleswe:octo_widget_pr7",
    "patch": GOLDEN_PATCH,
    "pre_commands": "git checkout 0123abc -f && git reset --hard HEAD\\n",
    "f2p_patch": TEST_FILE_PATCH,
    "f2p_script": "def test_new():\n    assert True\n",
    "FAIL_TO_PASS": json.dumps(["test_fail_to_pass.py::test_new"]),
    "PASS_TO_PASS": json.dumps(["tests/test_core.py::test_old"]),
    "responses_create_params": {"input": [{"role": "user", "content": "Fix the widget.\r\nIt should return 2."}]},
    "source_server": "scale_swe",
}
REBENCH_ROW = {
    "instance_id": "octo__gadget-9",
    "repo": "octo/gadget",
    "base_commit": "f00d" * 10,
    "patch": "diff --git a/gadget.go b/gadget.go\n--- a/gadget.go\n+++ b/gadget.go\n@@ -1 +1 @@\n-slow\n+fast\n",
    "test_patch": "diff --git a/gadget_test.go b/gadget_test.go\n--- a/gadget_test.go\n+++ b/gadget_test.go\n@@ -1 +1 @@\n-a\n+b\n",
    "image_name": "docker.io/swerebenchv2/octo-gadget:9-f00dfee",
    "install_config": {"install": ["make build"], "test_cmd": "make test", "log_parser": "parse_log_fake"},
    "FAIL_TO_PASS": ["TestNew"],
    "PASS_TO_PASS": ["TestOld"],
    "responses_create_params": {"input": [{"role": "user", "content": "Make the gadget fast."}]},
    "source_server": "swe_rebench",
}
UNSUPPORTED_ROW = {"instance_id": "delivery__iceberg-1", "source_server": "swemer_v1"}

FAKE_PARSERS = """from lib.agent.swe_constants import TestStatus


def parse_log_fake(log):
    statuses = {}
    for line in log.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in (TestStatus.PASSED, "FAILED"):
            statuses[parts[1]] = parts[0]
    return statuses


NAME_TO_PARSER = {"parse_log_fake": parse_log_fake}
"""


def _git(*args: str) -> str:
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.com"]
    return subprocess.run(["git", *identity, *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def parser_cache(tmp_path: Path) -> Path:
    """A stand-in for the server's clone of SWE-rebench's parsers, so no test reaches the network."""
    repo = tmp_path / "parser-cache" / "SWE-rebench-V2"
    (repo / "lib" / "agent").mkdir(parents=True)
    (repo / "LICENSE").write_text("MIT License\n")
    (repo / "lib" / "__init__.py").write_text("")
    (repo / "lib" / "agent" / "__init__.py").write_text("")
    (repo / "lib" / "agent" / "swe_constants.py").write_text("class TestStatus:\n    PASSED = 'PASSED'\n")
    (repo / "lib" / "agent" / "log_parsers.py").write_text(FAKE_PARSERS)
    _git("init", "-q", str(repo))
    _git("-C", str(repo), "add", "-A")
    _git("-C", str(repo), "commit", "-q", "-m", "parsers")
    yield repo.parent
    # The upstream module imports `lib.agent...` by absolute name; keep that package out of later tests.
    for name in [module for module in sys.modules if module == "lib" or module.startswith("lib.")]:
        del sys.modules[name]


@pytest.fixture
def converted(tmp_path: Path, parser_cache: Path) -> tuple[Path, dict]:
    rows = tmp_path / "rows.jsonl"
    typed_scale = {"task_id": {"taskset": "swe", "task_id": "octo_widget_pr7"}, "task_input": SCALE_ROW}
    rows.write_text("\n".join(json.dumps(row) for row in (typed_scale, REBENCH_ROW, UNSUPPORTED_ROW)) + "\n")
    out = tmp_path / "dataset"
    manifest = convert.convert(
        rows, out, dataset_name="nemo-gym/test-swe", skip_unsupported=True, parser_cache_dir=parser_cache
    )
    return out, manifest


def _load_grader(task_dir: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"grade_{task_dir.name}", task_dir / "tests/nemo_gym/grade.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canned_exec(begin: str, end: str, body: str):
    async def fake_exec(self, command: str, timeout_s: float | None = None, **_: object):
        assert command == "bash /tmp/nemo_gym_eval.sh"
        return SimpleNamespace(
            stdout=f"setup noise\n{begin}\n{body}\n{end}\n", stderr="", return_code=0, error_type=None
        )

    return fake_exec


class _Recorder:
    """A sandbox that records what a server helper would run, so the tests compare commands, not intent."""

    def __init__(self, return_codes: dict[str, int] | None = None) -> None:
        self.commands: list[str] = []
        self.uploads: list[tuple[Path, str]] = []
        self.return_codes = return_codes or {}

    async def exec(self, command: str, **_: object) -> SimpleNamespace:
        self.commands.append(command)
        code = next((rc for key, rc in self.return_codes.items() if key in command), 0)
        return SimpleNamespace(return_code=code, stdout="", stderr="", error_type=None)

    async def upload(self, local_path: Path, remote_path: str) -> None:
        self.uploads.append((Path(local_path), remote_path))


def test_converts_supported_rows_and_lists_the_rest(converted: tuple[Path, dict], parser_cache: Path) -> None:
    out, manifest = converted

    assert [entry["task"] for entry in manifest["converted"]] == [
        "scale_swe__octo_widget_pr7",
        "swe_rebench__octo__gadget-9",
    ]
    assert manifest["skipped"] == [{"line": 3, "source": "swemer_v1", "instance_id": "delivery__iceberg-1"}]
    assert "CR line endings" in manifest["converted"][0]["note"]
    assert "note" not in manifest["converted"][1]
    assert manifest["swe_rebench_log_parsers_commit"] == _git(
        "-C", str(parser_cache / "SWE-rebench-V2"), "rev-parse", "HEAD"
    )
    dataset = tomllib.loads((out / "dataset.toml").read_text())
    assert dataset["dataset"] == {"name": "nemo-gym/test-swe", "version": "0.1.0"}
    assert dataset["gym"] == {"type": "train", "license": "TBD", "domain": "coding"}


@pytest.mark.parametrize(
    ("task", "row", "source"),
    [
        ("scale_swe__octo_widget_pr7", SCALE_ROW, "scale_swe"),
        ("swe_rebench__octo__gadget-9", REBENCH_ROW, "swe_rebench"),
    ],
)
def test_tasks_load_with_gyms_harbor_loader(
    converted: tuple[Path, dict], parser_cache: Path, task: str, row: dict, source: str
) -> None:
    out, _ = converted
    adapter = convert.ScaleSWEAdapter() if source == "scale_swe" else convert.SWERebenchAdapter(parser_cache)
    task_dir = out / task
    image = row.get("image_url") or row.get("image_name")
    workdir = tomllib.loads((task_dir / "task.toml").read_text())["environment"]["workdir"]
    record = {
        "record_version": RECORD_VERSION,
        "image": f"{image}@sha256:{'0' * 64}",
        "config": {"Env": ["PATH=/usr/bin:/bin"], "WorkingDir": workdir, "User": ""},
    }

    loaded = load_task(task_dir, root=out, image_configs={image: record})

    server = convert.server_config(source)
    resources = server["sandbox_config"]["resources"]
    prompt = row["responses_create_params"]["input"][0]["content"]
    # Gym reads instruction.md as text: the file holds the prompt byte for byte, the agent gets LF line endings.
    assert (task_dir / "instruction.md").read_bytes() == prompt.encode()
    assert loaded.instruction == prompt.replace("\r\n", "\n")
    assert len(loaded.overlay) == 1
    encoded = re.search(r"printf '%s' '([A-Za-z0-9+/=]+)'", loaded.overlay[0].command).group(1)
    assert base64.b64decode(encoded).decode() == (task_dir / "environment/setup.sh").read_text()
    assert loaded.config.agent.timeout_sec == convert.DEFAULT_AGENT_TIMEOUT_SEC
    assert loaded.config.verifier.environment_mode == "separate"
    assert loaded.config.verifier.timeout_sec == server["evaluation_timeout"] + convert.VERIFIER_TIMEOUT_MARGIN_SEC
    assert loaded.verifier_image == image
    assert [hook.command for hook in loaded.config.verifier.collect] == [adapter.plan(row).capture_command]
    assert loaded.config.environment.cpus == resources["cpu"]
    assert loaded.config.environment.memory_mb == resources["memory_mib"]
    assert loaded.config.environment.storage_mb == resources["disk_gib"] * 1024
    vendored = task_dir / "tests/nemo_gym/resources_servers" / source / "verification.py"
    assert vendored.read_text() == (REPO_ROOT / "resources_servers" / source / "verification.py").read_text()
    if source == "swe_rebench":
        assert loaded.config.environment.env == {"GRADLE_USER_HOME": "/root/.gradle"}
        upstream = task_dir / "tests/nemo_gym/swe_rebench_parsers/SWE-rebench-V2/lib/agent/log_parsers.py"
        assert upstream.read_text() == FAKE_PARSERS


def test_evaluation_timeout_override_bounds_the_test_run_and_the_verifier(tmp_path: Path, parser_cache: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(REBENCH_ROW) + "\n")
    out = tmp_path / "dataset"

    manifest = convert.convert(
        rows, out, dataset_name="nemo-gym/test-swe", parser_cache_dir=parser_cache, evaluation_timeout_sec=1200
    )

    # The override replaces swe_rebench's own default, both where grade.py bounds the test run and in task.toml.
    assert convert.server_config("swe_rebench")["evaluation_timeout"] != 1200
    task_dir = out / "swe_rebench__octo__gadget-9"
    verifier = tomllib.loads((task_dir / "task.toml").read_text())["verifier"]
    assert verifier["timeout_sec"] == 1200 + convert.VERIFIER_TIMEOUT_MARGIN_SEC
    assert json.loads((task_dir / "tests/nemo_gym/task.json").read_text())["evaluation_timeout"] == 1200
    assert manifest["evaluation_timeout_sec"] == 1200


def test_setup_and_capture_run_the_servers_own_commands() -> None:
    plan = convert.ScaleSWEAdapter().plan(SCALE_ROW)
    workdir = SCALE_ROW["workdir"]

    anti_cheat = _Recorder()
    asyncio.run(apply_anti_cheat_setup(anti_cheat, workdir, SCALE_ROW["instance_id"], "scale_swe"))
    [(uploaded, remote)] = anti_cheat.uploads
    assert uploaded == ANTI_CHEAT_SCRIPT_FPATH
    assert f"cat > {shlex.quote(remote)}" in plan.setup_script
    assert ANTI_CHEAT_SCRIPT_FPATH.read_text().rstrip("\n") in plan.setup_script
    assert shlex.quote(anti_cheat.commands[0]) in plan.setup_script

    identity = _Recorder()
    asyncio.run(prepare_git_for_commits(identity, workdir, "scale_swe"))
    assert shlex.quote(identity.commands[0]) in plan.setup_script

    capture = _Recorder()
    asyncio.run(capture_model_patch(capture, workdir, "BASE", mode="worktree"))
    diff_command = next(
        command for command in capture.commands if "--no-pager diff" in command and "add -N" in command
    )
    assert diff_command.replace("BASE", '"$base"') in plan.capture_command

    # The pre_commands run as the server cleans them: the dataset's trailing literal backslash-n is gone.
    assert "git checkout 0123abc -f && git reset --hard HEAD\n)" in plan.setup_script


def test_setup_matches_the_scale_swe_servers_seed_helpers() -> None:
    app = pytest.importorskip("resources_servers.scale_swe.app")
    plan = convert.ScaleSWEAdapter().plan(SCALE_ROW)
    workdir = SCALE_ROW["workdir"]

    missing_repo = _Recorder({"rev-parse --git-dir": 1})
    asyncio.run(app.ScaleSWEResourcesServer._ensure_git_repo(None, missing_repo, workdir))
    precheck, init = missing_repo.commands
    assert f"if ! {precheck} >/dev/null 2>&1" in plan.setup_script
    assert init in plan.setup_script

    pristine = _Recorder()
    asyncio.run(app.ScaleSWEResourcesServer._pristine_untracked_files(None, pristine, workdir))
    assert f"{pristine.commands[0]} > " in plan.setup_script


@pytest.mark.parametrize(("status", "reward"), [("PASSED", 1.0), ("FAILED", 0.0)])
def test_scale_swe_grader_scores_like_the_server(
    converted: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, reward: float
) -> None:
    out, _ = converted
    task_dir = out / "scale_swe__octo_widget_pr7"
    grader = _load_grader(task_dir)
    verification = grader._load("scale_swe_verification_markers", "resources_servers/scale_swe/verification.py")
    artifacts, logs, root = tmp_path / "artifacts", tmp_path / "logs", tmp_path / "root"
    artifacts.mkdir()
    (artifacts / "model_patch.diff").write_text(GOLDEN_PATCH + TEST_FILE_PATCH + PRISTINE_FILE_PATCH)
    (artifacts / "pristine_untracked.txt").write_text("notes.txt\n")
    body = f"PASSED test_fail_to_pass.py::test_new\n{status} tests/test_core.py::test_old"
    monkeypatch.setattr(
        grader.LocalSandbox, "exec", _canned_exec(verification.TEST_OUTPUT_BEGIN, verification.TEST_OUTPUT_END, body)
    )

    assert grader.main(["--artifacts", str(artifacts), "--logs", str(logs), "--root", str(root)]) == 0

    verdict = json.loads((logs / "reward.json").read_text())
    assert verdict["reward"] == reward
    assert verdict["evaluation_completed"] == 1.0
    assert verdict["patch_capture_failed"] == 0.0
    # The patch the tests ran with: the agent's source change only, as the server builds it.
    assert (root / "tmp/nemo_gym_patch.diff").read_text() == GOLDEN_PATCH
    assert "git checkout 0123abc -f" in (root / "tmp/nemo_gym_eval.sh").read_text()
    assert json.loads((logs / "grading.json").read_text())["test_results"]["resolved"] is (reward == 1.0)


def test_grader_grades_an_empty_patch_when_the_capture_failed(
    converted: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, _ = converted
    grader = _load_grader(out / "scale_swe__octo_widget_pr7")
    verification = grader._load("scale_swe_verification_markers", "resources_servers/scale_swe/verification.py")
    logs, root = tmp_path / "logs", tmp_path / "root"
    body = "FAILED test_fail_to_pass.py::test_new\nPASSED tests/test_core.py::test_old"
    monkeypatch.setattr(
        grader.LocalSandbox, "exec", _canned_exec(verification.TEST_OUTPUT_BEGIN, verification.TEST_OUTPUT_END, body)
    )

    grader.main(["--artifacts", str(tmp_path / "missing"), "--logs", str(logs), "--root", str(root)])

    verdict = json.loads((logs / "reward.json").read_text())
    assert verdict["reward"] == 0.0
    assert verdict["patch_capture_failed"] == 1.0
    assert not (root / "tmp/nemo_gym_patch.diff").exists()


def test_swe_rebench_grader_uses_the_servers_parser_lookup(
    converted: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, _ = converted
    task_dir = out / "swe_rebench__octo__gadget-9"
    grader = _load_grader(task_dir)
    verification = grader._load("swe_rebench_verification_markers", "resources_servers/swe_rebench/verification.py")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "model_patch.diff").write_text(REBENCH_ROW["patch"])
    monkeypatch.setattr(
        grader.LocalSandbox,
        "exec",
        _canned_exec(verification.TEST_OUTPUT_BEGIN, verification.TEST_OUTPUT_END, "PASSED TestNew\nPASSED TestOld"),
    )
    argv = ["--artifacts", str(artifacts), "--root", str(tmp_path / "root")]

    grader.main([*argv, "--logs", str(tmp_path / "logs")])
    assert json.loads((tmp_path / "logs/reward.json").read_text())["reward"] == 1.0

    task_json = task_dir / "tests/nemo_gym/task.json"
    task = json.loads(task_json.read_text())
    task["row"]["install_config"]["log_parser"] = "parse_log_unknown"
    task_json.write_text(json.dumps(task))
    grader.main([*argv, "--logs", str(tmp_path / "logs-unknown")])
    verdict = json.loads((tmp_path / "logs-unknown/reward.json").read_text())
    assert (verdict["reward"], verdict["evaluation_completed"]) == (0.0, 0.0)
    assert "parse_log_unknown" in json.loads((tmp_path / "logs-unknown/grading.json").read_text())["error"]


def test_local_sandbox_timeout_stops_the_whole_process_group() -> None:
    spec = importlib.util.spec_from_file_location("swe_rows_grade", REPO_ROOT / "scripts/swe_rows_to_harbor/grade.py")
    grader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grader)
    started = time.monotonic()

    # The background sleep holds the output pipe open; only killing the group lets the read finish.
    result = asyncio.run(grader.LocalSandbox().exec("sleep 30 & sleep 30; echo late", timeout_s=1))

    assert (result.return_code, result.error_type) == (124, "timeout")
    assert "late" not in result.stdout
    assert time.monotonic() - started < 15


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "# canary GUID 1234\nFix it."}],
        [{"role": "user", "content": "Fix it."}, {"role": "user", "content": "Again."}],
        [{"role": "user", "content": [{"type": "input_text", "text": "Fix it."}]}],
    ],
)
def test_prompts_gym_would_change_or_cannot_carry_are_refused(messages: list[dict]) -> None:
    with pytest.raises(ValueError):
        convert._prompt({"responses_create_params": {"input": messages}})


def test_unsupported_rows_fail_without_skip_and_output_is_never_reused(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(UNSUPPORTED_ROW) + "\n")
    with pytest.raises(ValueError, match="swemer_v1"):
        convert.convert(rows, tmp_path / "out", dataset_name="nemo-gym/test-swe")

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("existing")
    with pytest.raises(FileExistsError):
        convert.convert(rows, occupied, dataset_name="nemo-gym/test-swe", skip_unsupported=True)
