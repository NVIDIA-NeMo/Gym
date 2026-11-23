# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Scale-SWE Harbor adapter (``resources_servers/scale_swe/harbor/adapter.py``).

They convert tiny rows in ``tmp_path`` and need no network or sandbox; running a converted task is the oracle run
(``gym dataset validate``), not these tests. The grader, the capture hook and the seed script run here against
temporary git repositories, with the steps that need root (ownership, the privilege drop) stubbed out.
"""

import ast
import asyncio
import base64
import dataclasses
import hashlib
import importlib.util
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
ADAPTER_PATH = REPO_ROOT / "resources_servers/scale_swe/harbor/adapter.py"
VERIFICATION = REPO_ROOT / "resources_servers/scale_swe/verification.py"
PATCH_CAPTURE = REPO_ROOT / "resources_servers/swebench/patch_capture.py"
SERVER_CONFIG = REPO_ROOT / "resources_servers/scale_swe/configs/scale_swe.yaml"
PAIRING_CONFIG = REPO_ROOT / "resources_servers/scale_swe/configs/scale_swe_opencode.yaml"
TEMPLATE = REPO_ROOT / "resources_servers/scale_swe/harbor/task-template"
needs_git = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("timeout") is None, reason="needs git and timeout"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


adapter = _load("scale_swe_harbor_adapter_under_test", ADAPTER_PATH)


def _row(instance_id: str) -> dict:
    return {
        "instance_id": instance_id,
        "repo": f"example/{instance_id}",
        "workdir": "/testbed",
        "image_url": f"example.org/scaleswe:{instance_id}",
        "responses_create_params": {"input": [{"role": "user", "content": f"Fix the bug in {instance_id}.\n"}]},
        "pre_commands": "git checkout -q HEAD",
        "patch": "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n",
        "f2p_patch": "diff --git a/t.py b/t.py\n",
        "f2p_script": "pytest t.py",
        "FAIL_TO_PASS": ["t.py::test_fixed"],
        "PASS_TO_PASS": ["t.py::test_kept"],
    }


@pytest.fixture
def rows(tmp_path: Path) -> Path:
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(_row(name)) + "\n" for name in ("repo_a_pr1", "repo_b_pr2", "repo_c_pr3")))
    return path


def _convert(rows: Path, out: Path, *extra: str) -> int:
    return adapter.main(["--rows", str(rows), "--output-dir", str(out), *extra])


def _tree(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mode)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _native_config(tmp_path: Path, **changes) -> Path:
    """configs/scale_swe.yaml with keys of the server block set (dotted paths; None writes null)."""
    document = yaml.safe_load(SERVER_CONFIG.read_text())
    server = document["scale_swe_resources_server"]["resources_servers"]["scale_swe"]
    for dotted, value in changes.items():
        *parents, leaf = dotted.split(".")
        node = server
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    path = tmp_path / "native.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def _bundle(dockerfile: str) -> dict[str, bytes]:
    encoded = re.search(r"printf '%s' '([A-Za-z0-9+/=]+)'", dockerfile).group(1)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(encoded)), mode="r:gz") as archive:
        return {info.name: archive.extractfile(info).read() for info in archive.getmembers()}


def test_two_conversions_are_byte_identical(tmp_path: Path, rows: Path) -> None:
    for name in ("a", "b"):
        extra = ["--native-rows-out", str(tmp_path / f"{name}.jsonl"), "--id-map-out", str(tmp_path / f"{name}.json")]
        assert _convert(rows, tmp_path / name / "ds", *extra) == 0
    assert _tree(tmp_path / "a" / "ds") == _tree(tmp_path / "b" / "ds")
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    # The bundle in each Dockerfile is a gzip with mtime 0 holding tar members with mtime 0, so it cannot carry the
    # time of the conversion (two conversions in the same second would not show a clock-based mtime).
    dockerfile = (tmp_path / "a" / "ds" / "repo_a_pr1" / "environment" / "Dockerfile").read_text()
    encoded = re.search(r"printf '%s' '([A-Za-z0-9+/=]+)'", dockerfile).group(1)
    compressed = base64.b64decode(encoded)
    assert compressed[4:8] == b"\0\0\0\0"
    with tarfile.open(fileobj=io.BytesIO(compressed), mode="r:gz") as archive:
        members = archive.getmembers()
    assert members and all(info.mtime == 0 for info in members)


def test_task_folder_contents(tmp_path: Path, rows: Path) -> None:
    assert _convert(rows, tmp_path / "ds") == 0
    task = tmp_path / "ds" / "repo_b_pr2"
    assert (task / "instruction.md").read_text() == "Fix the bug in repo_b_pr2.\n"
    toml = (task / "task.toml").read_text()
    assert 'environment_mode = "separate"' in toml
    assert 'docker_image = "example.org/scaleswe:repo_b_pr2"' in toml
    # Hidden data stays out of what the agent's sandbox receives (instruction.md and environment/).
    visible = (task / "instruction.md").read_text() + (task / "environment" / "setup.sh").read_text()
    visible += "".join(data.decode() for data in _bundle((task / "environment" / "Dockerfile").read_text()).values())
    for hidden in ("+x = 2", "pytest t.py", "t.py::test_fixed", "t.py::test_kept"):
        assert hidden not in visible
    assert json.loads((task / "tests/nemo_gym/task.json").read_text())["row"]["FAIL_TO_PASS"] == ["t.py::test_fixed"]
    assert (task / "solution" / "golden.patch").read_text() == _row("repo_b_pr2")["patch"]


def test_ids_file_order_native_rows_and_id_map(tmp_path: Path, rows: Path) -> None:
    ids = tmp_path / "ids.txt"
    ids.write_text("# subset\nrepo_c_pr3\nrepo_a_pr1\n")
    native, id_map = tmp_path / "native.jsonl", tmp_path / "map.json"
    extra = ["--ids-file", str(ids), "--native-rows-out", str(native), "--id-map-out", str(id_map)]
    assert _convert(rows, tmp_path / "ds", *extra) == 0
    lines = rows.read_text().splitlines(keepends=True)
    assert native.read_text() == lines[2] + lines[0]
    mapping = json.loads(id_map.read_text())
    assert mapping == {"0": "repo_c_pr3", "1": "repo_a_pr1"}
    for index, line in enumerate(native.read_text().splitlines()):
        assert mapping[str(index)] == json.loads(line)["instance_id"]
    assert sorted(path.name for path in (tmp_path / "ds").iterdir() if path.is_dir()) == ["repo_a_pr1", "repo_c_pr3"]


def test_non_empty_output_is_refused(tmp_path: Path, rows: Path, capsys: pytest.CaptureFixture) -> None:
    out = tmp_path / "ds"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    assert _convert(rows, out) == 2
    assert "not empty" in capsys.readouterr().err
    assert [path.name for path in out.iterdir()] == ["keep.txt"]


def test_indented_comment_in_ids_file_is_a_comment(tmp_path: Path, rows: Path) -> None:
    ids = tmp_path / "ids.txt"
    ids.write_text("  # subset\nrepo_b_pr2\n")
    assert _convert(rows, tmp_path / "ds", "--ids-file", str(ids)) == 0
    assert sorted(path.name for path in (tmp_path / "ds").iterdir() if path.is_dir()) == ["repo_b_pr2"]


@pytest.mark.parametrize(
    "ids_text, message",
    [("repo_a_pr1\nrepo_a_pr1\n", "duplicate id"), ("repo_a_pr1\nnot_a_row\n", "not in rows.jsonl")],
)
def test_bad_ids_file_is_refused(tmp_path: Path, rows: Path, capsys, ids_text: str, message: str) -> None:
    ids = tmp_path / "ids.txt"
    ids.write_text(ids_text)
    assert _convert(rows, tmp_path / "ds", "--ids-file", str(ids)) == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "ds").exists()


def test_ids_file_that_is_not_utf8_is_refused(tmp_path: Path, rows: Path, capsys: pytest.CaptureFixture) -> None:
    ids = tmp_path / "ids.txt"
    ids.write_bytes(b"repo_a_pr1\nrepo_b_\xff\n")
    assert _convert(rows, tmp_path / "ds", "--ids-file", str(ids)) == 2
    assert "ids.txt is not valid UTF-8 (invalid start byte at byte 18)" in capsys.readouterr().err
    assert not (tmp_path / "ds").exists()


@pytest.mark.parametrize(
    "bad_line, message",
    [
        ({**_row("repo_d_pr4"), "workdir": None}, "repo_d_pr4: workdir is missing"),
        ({k: v for k, v in _row("repo_d_pr4").items() if k != "workdir"}, "repo_d_pr4: workdir is missing"),
        ({k: v for k, v in _row("repo_d_pr4").items() if k != "image_url"}, "repo_d_pr4: image_url is missing"),
        ({**_row("repo_d_pr4"), "image_url": " "}, "repo_d_pr4: image_url is missing"),
        (["not", "an", "object"], "is not a JSON object"),
        ({**_row("repo_d_pr4"), "image_url": "img:c\nRUN touch /pwned"}, "repo_d_pr4: image_url 'img:c\\nRUN"),
        ({**_row("repo_d_pr4"), "workdir": "/testbed/$(id)"}, "repo_d_pr4: workdir '/testbed/$(id)' is not"),
        ({**_row("repo_d_pr4"), "workdir": "testbed"}, "repo_d_pr4: workdir 'testbed' is not"),
        (
            {**_row("repo_d_pr4"), "responses_create_params": {"input": [{"role": "user", "content": "a \ud800"}]}},
            "repo_d_pr4: the row holds text that is not valid UTF-8",
        ),
        ({**_row("repo_d_pr4"), "patch": "+x = \udfff\n"}, "repo_d_pr4: the row holds text that is not valid UTF-8"),
        (
            {**_row("repo_d_pr4"), "responses_create_params": {"input": ["just a string"]}},
            "repo_d_pr4: expected one user message",
        ),
        (
            {**_row("repo_d_pr4"), "responses_create_params": [1]},
            "repo_d_pr4: expected responses_create_params.input to hold one message",
        ),
    ],
)
def test_malformed_row_is_refused_before_writing(
    tmp_path: Path, rows: Path, capsys: pytest.CaptureFixture, bad_line: object, message: str
) -> None:
    # The bad row comes after good ones, so a check made while writing would leave their folders behind.
    with rows.open("a") as handle:
        handle.write(json.dumps(bad_line) + "\n")
    assert _convert(rows, tmp_path / "ds") == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "ds").exists()


@pytest.mark.parametrize("name", [".", "..", ".hidden"])
def test_dot_prefixed_instance_id_is_refused_before_writing(
    tmp_path: Path, rows: Path, capsys: pytest.CaptureFixture, name: str
) -> None:
    # "." is the output folder itself and ".." its parent, where a sibling instruction.md would be overwritten.
    out = tmp_path / "ds"
    out.mkdir()
    (tmp_path / "instruction.md").write_text("sibling\n")
    with rows.open("a") as handle:
        handle.write(json.dumps(_row(name)) + "\n")
    before = _tree(tmp_path)
    assert _convert(rows, out) == 2
    assert f"instance_id {name!r} is not a usable folder name" in capsys.readouterr().err
    assert _tree(tmp_path) == before
    assert list(out.iterdir()) == []


def test_instance_id_with_an_inner_dot_converts(tmp_path: Path, rows: Path) -> None:
    with rows.open("a") as handle:
        handle.write(json.dumps(_row("repo.d_pr4")) + "\n")
    assert _convert(rows, tmp_path / "ds") == 0
    assert (tmp_path / "ds" / "repo.d_pr4" / "instruction.md").read_text() == "Fix the bug in repo.d_pr4.\n"


def test_non_string_instance_id_is_refused_with_ids_file(
    tmp_path: Path, rows: Path, capsys: pytest.CaptureFixture
) -> None:
    with rows.open("a") as handle:
        handle.write(json.dumps({**_row("repo_d_pr4"), "instance_id": ["x"]}) + "\n")
    ids = tmp_path / "ids.txt"
    ids.write_text("repo_a_pr1\n")
    assert _convert(rows, tmp_path / "ds", "--ids-file", str(ids)) == 2
    assert "instance_id ['x'] is not a usable folder name" in capsys.readouterr().err
    assert not (tmp_path / "ds").exists()


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--limit", "-1"], "--limit must be 0 (all) or positive"),
        (["--native-rows-out", "{tmp}/no_such_dir/native.jsonl"], "does not exist"),
        (["--id-map-out", "{tmp}/no_such_dir/map.json"], "does not exist"),
    ],
)
def test_bad_output_arguments_are_refused_before_writing(
    tmp_path: Path, rows: Path, capsys: pytest.CaptureFixture, extra: list[str], message: str
) -> None:
    assert _convert(rows, tmp_path / "ds", *[arg.format(tmp=tmp_path) for arg in extra]) == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "ds").exists()


@pytest.mark.parametrize(
    "changes, refused",
    [
        ({"is_verifying_golden_patch": True}, "is_verifying_golden_patch"),
        ({"enforce_agent_no_network": True}, "enforce_agent_no_network"),
        ({"sandbox_config.provider_options.network_policy": {"egress": []}}, "network_policy"),
        ({"sandbox_config.derive_cpu_env": False}, "derive_cpu_env"),
        ({"evaluation_timeout": None}, "evaluation_timeout: null"),
        ({"sandbox_config.resources.gpu": 1}, "sandbox_config.resources.gpu"),
        ({"sandbox_config.resources.cpu": 1.5}, "sandbox_config.resources.cpu: 1.5 (not a whole number)"),
        ({"patch_capture_mode": "staged"}, "unknown patch_capture_mode"),
    ],
)
def test_unreproduced_native_settings_are_refused(
    tmp_path: Path, rows: Path, capsys: pytest.CaptureFixture, changes: dict, refused: str
) -> None:
    config = _native_config(tmp_path, **changes)
    assert _convert(rows, tmp_path / "ds", "--native-config", str(config)) == 2
    assert refused in capsys.readouterr().err
    assert not (tmp_path / "ds").exists()


def test_settings_read_from_native_config(tmp_path: Path, rows: Path) -> None:
    config = _native_config(tmp_path, evaluation_timeout=600, **{"sandbox_config.resources.cpu": 2})
    assert _convert(rows, tmp_path / "ds", "--native-config", str(config)) == 0
    toml = (tmp_path / "ds" / "repo_a_pr1" / "task.toml").read_text()
    assert "cpus = 2" in toml and f"timeout_sec = {600 + adapter.VERIFIER_TIMEOUT_MARGIN_SEC}" in toml
    # The pairing YAML pulls configs/scale_swe.yaml in through config_paths: the same tasks as the default.
    assert _convert(rows, tmp_path / "pairing", "--native-config", str(PAIRING_CONFIG)) == 0
    assert _convert(rows, tmp_path / "default") == 0
    pairing, default = _tree(tmp_path / "pairing"), _tree(tmp_path / "default")
    assert {k: v for k, v in pairing.items() if k != "README.md"} == {
        k: v for k, v in default.items() if k != "README.md"
    }


def test_dataset_readme_records_provenance(tmp_path: Path, rows: Path) -> None:
    ids = tmp_path / "ids.txt"
    ids.write_text("repo_a_pr1\nrepo_b_pr2\n")
    assert _convert(rows, tmp_path / "ds", "--ids-file", str(ids)) == 0
    readme = (tmp_path / "ds" / "README.md").read_text()
    rows_sha = hashlib.sha256(rows.read_bytes()).hexdigest()
    assert f"- Rows: `rows.jsonl`, sha256 `{rows_sha}`, 3 lines" in readme
    assert f"- Selected: 2 task(s) from `ids.txt` (sha256 `{hashlib.sha256(ids.read_bytes()).hexdigest()}`)" in readme
    assert re.search(r"- Gym commit: `([0-9a-f]{40}|unknown)`", readme)
    for path in (VERIFICATION, PATCH_CAPTURE, REPO_ROOT / adapter.ANTI_CHEAT_MODULE):
        relative = path.relative_to(REPO_ROOT)
        assert f"- `{relative}`: `{hashlib.sha256(path.read_bytes()).hexdigest()}`" in readme
    command = readme.split("```bash\n", 1)[1].split("```", 1)[0]
    assert command.split() == [
        "python",
        "resources_servers/scale_swe/harbor/adapter.py",
        "\\",
        "--rows",
        "rows.jsonl",
        "\\",
        "--output-dir",
        "<dir>",
        "\\",
        "--ids-file",
        "ids.txt",
    ]
    assert str(tmp_path) not in readme
    assert not (tmp_path / "ds" / "conversion.json").exists()


def test_vendored_grader_is_the_servers_code(tmp_path: Path, rows: Path) -> None:
    assert _convert(rows, tmp_path / "ds") == 0
    task = tmp_path / "ds" / "repo_a_pr1"
    copies = {
        VERIFICATION: [task / "tests/nemo_gym/resources_servers/scale_swe/verification.py"],
        PATCH_CAPTURE: [],
    }
    bundle = _bundle((task / "environment" / "Dockerfile").read_text())
    for source, task_copies in copies.items():
        relative = str(source.relative_to(REPO_ROOT))
        original = source.read_bytes()
        for copy in [path.read_bytes() for path in task_copies] + [bundle[f"capture/{relative}"]]:
            header, body = copy.split(b"\n", 1)
            assert body == original
            assert re.fullmatch(
                rf"# Vendored unchanged from {re.escape(relative)}"
                rf" at ([0-9a-f]{{40}}|unknown)( \(uncommitted changes\))?;"
                rf" source sha256 {hashlib.sha256(original).hexdigest()}; license Apache-2\.0",
                header.decode(),
            )
    assert bundle["anti_cheat_setup.sh"] == (REPO_ROOT / adapter.ANTI_CHEAT_MODULE).read_bytes()
    # The task's copy grades a fixed case exactly as the server's module does.
    server = _load("scale_swe_verification_server", VERIFICATION)
    vendored = _load("scale_swe_verification_vendored", copies[VERIFICATION][0])
    statuses = {"t.py::test_fixed": "PASSED", "t.py::test_kept": "FAILED"}
    for fail_to_pass, pass_to_pass in ((["t.py::test_fixed"], ["t.py::test_kept"]), (["t.py::test_fixed"], [])):
        assert vendored.grade(statuses, fail_to_pass, pass_to_pass) == server.grade(
            statuses, fail_to_pass, pass_to_pass
        )


# -- the code a converted task runs: grade.py, capture.py, local_sandbox.py, setup.sh, solve.sh, the capture hook -----


def _git_env(tmp_path: Path) -> dict[str, str]:
    """An environment whose git reads and writes only files under tmp_path."""
    return {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CEILING_DIRECTORIES": str(tmp_path),
    }


def _git_repo(tmp_path: Path, *, commit: bool = True) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = _git_env(tmp_path)
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=env)
    if commit:
        (repo / "m.py").write_text("x = 1\n")
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, env=env)
        identity = ["-c", "user.name=t", "-c", "user.email=t@example.org"]
        subprocess.run(["git", "-C", str(repo), *identity, "commit", "-q", "-m", "base"], check=True, env=env)
    return repo


class _Grader:
    """``grade.py`` of a converted task, with its outputs under tmp_path and the root-only steps stubbed.

    The audit hook cannot be removed once installed, so it is never armed here (``_arm_after_tests`` is stubbed);
    the tests that arm it run ``grade.py`` in a subprocess (``_drive``).
    """

    def __init__(self, tmp_path: Path, rows: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _convert(rows, tmp_path / "ds") == 0
        here = tmp_path / "ds" / "repo_a_pr1" / "tests" / "nemo_gym"
        # The workdir check runs on this machine: point the row at a folder that exists.
        task = json.loads((here / "task.json").read_text())
        task["row"]["workdir"] = str(tmp_path)
        (here / "task.json").write_text(json.dumps(task))
        self.module = _load(f"scale_swe_grade_{tmp_path.name}", here / "grade.py")
        self.artifacts, self.logs = tmp_path / "artifacts", tmp_path / "verifier"
        self.artifacts.mkdir()
        self.calls: list = []
        self.result = None
        monkeypatch.setattr(sys, "path", list(sys.path))
        monkeypatch.setattr(self.module, "ARTIFACTS", self.artifacts)
        monkeypatch.setattr(self.module, "LOGS", self.logs)
        monkeypatch.setattr(self.module.os, "seteuid", lambda uid: None)
        monkeypatch.setattr(self.module, "_give_root_ownership", lambda: self.calls.append("ownership") or 0)
        self.protect_reward_path = self.module._protect_reward_path
        monkeypatch.setattr(self.module, "_protect_reward_path", lambda: self.calls.append("protect"))
        record = {"mappings": set(), "kept": {}, "seconds": 0.0}
        monkeypatch.setattr(self.module, "_prepare_after_tests", lambda: self.calls.append("prepare") or record)
        monkeypatch.setattr(self.module, "_arm_after_tests", lambda: self.calls.append("arm"))
        monkeypatch.setattr(self.module, "_check_mappings", lambda record: self.calls.append("check"))

        def drop() -> None:
            pass

        self.drop = drop
        monkeypatch.setattr(self.module, "_privilege_dropper", lambda: drop)
        real_load = self.module._load

        def load(name: str, path: Path):
            verification = real_load(name, path)

            async def run_verification(*, sandbox, inputs, timeout_s, log_dir):
                self.calls.append(("run", sandbox.preexec_fn is drop, timeout_s, inputs.instance_id))
                return self.result(verification)

            # The real files go to /tmp in the verifier sandbox; here they go under tmp_path.
            files = {str(tmp_path / "eval" / "nemo_gym_eval.sh"): "true\n"}
            monkeypatch.setattr(verification, "verification_files", lambda inputs: files)
            monkeypatch.setattr(verification, "run_verification", run_verification)
            return verification

        monkeypatch.setattr(self.module, "_load", load)

    def capture(self, status: str | None, patch: str | None = "diff --git a/m.py b/m.py\n") -> None:
        if status is not None:
            (self.artifacts / "capture.json").write_text(json.dumps({"status": status, "error": "e"}))
        if patch is not None:
            (self.artifacts / "model_patch.diff").write_text(patch)

    def run(self) -> tuple[int, dict | None, dict]:
        code = self.module.main()
        reward = self.logs / "reward.json"
        grading = json.loads((self.logs / "grading.json").read_text())
        return code, json.loads(reward.read_text()) if reward.exists() else None, grading


@pytest.fixture
def grader(tmp_path: Path, rows: Path, monkeypatch: pytest.MonkeyPatch) -> _Grader:
    return _Grader(tmp_path, rows, monkeypatch)


@pytest.mark.parametrize(
    "status, capture_json, unreadable",
    [
        ("missing", None, 0.0),
        ("non-JSON", "{not json", 0.0),
        ("git_error", None, 1.0),
        ("runner_failed", None, 1.0),
        ("infra_error", None, 0.0),
    ],
)
def test_grader_scores_a_failed_capture_zero_without_running_tests(
    grader: _Grader, status: str, capture_json: str | None, unreadable: float
) -> None:
    if status == "non-JSON":
        (grader.artifacts / "capture.json").write_text(capture_json)
    elif status != "missing":
        grader.capture(status)
    code, reward, grading = grader.run()
    assert code == 0 and grading["masked"] is False
    assert reward == {
        "reward": 0.0,
        "evaluation_completed": 1.0,
        "resolved": 0.0,
        "patch_unreadable": unreadable,
        "capture_failed": 1.0 - unreadable,
    }
    assert grader.calls == []


def _nested(depth: int) -> str:
    return "[" * depth + "]" * depth


def _plant_capture(artifacts: Path, case: str, limits: tuple[int, int]) -> None:
    """Write capture artifacts the agent's sandbox could leave (``limits``: the grader's capture and patch caps)."""
    capture_max, patch_max = limits
    capture, patch = artifacts / "capture.json", artifacts / "model_patch.diff"
    ok = json.dumps({"status": "ok"})
    if case == "nested arrays":
        capture.write_text(_nested(10_000))
    elif case == "nested inside an object":
        capture.write_text('{"status": "ok", "error": ' + _nested(10_000) + "}")
    elif case == "a JSON list":
        capture.write_text('["ok"]')
    elif case == "status not a string":
        capture.write_text('{"status": ["ok"]}')
    elif case == "capture.json a symlink":
        (artifacts / "real.json").write_text(ok)
        capture.symlink_to(artifacts / "real.json")
    elif case == "capture.json a FIFO":
        os.mkfifo(capture)
    elif case == "capture.json a directory":
        capture.mkdir()
    elif case == "capture.json over its limit":
        capture.write_text(ok + " " * (capture_max + 1 - len(ok)))
    elif case == "capture.json a symlink to /proc/self/mem":
        capture.symlink_to("/proc/self/mem")
    else:
        capture.write_text(ok)
        if case == "patch a symlink":
            (artifacts / "real.diff").write_text("diff --git a/m.py b/m.py\n")
            patch.symlink_to(artifacts / "real.diff")
        elif case == "patch a FIFO":
            os.mkfifo(patch)
        elif case == "patch a directory":
            patch.mkdir()
        elif case == "patch over its limit":
            with patch.open("wb") as handle:
                handle.truncate(patch_max + 1)
        elif case == "patch a symlink to /proc/self/mem":
            patch.symlink_to("/proc/self/mem")
        else:
            raise AssertionError(case)


CAPTURE_CASES = [
    "nested arrays",
    "nested inside an object",
    "a JSON list",
    "status not a string",
    "capture.json a symlink",
    "capture.json a FIFO",
    "capture.json a directory",
    "capture.json over its limit",
    "capture.json a symlink to /proc/self/mem",
    "patch a symlink",
    "patch a FIFO",
    "patch a directory",
    "patch over its limit",
    "patch a symlink to /proc/self/mem",
]


@pytest.mark.parametrize("case", CAPTURE_CASES)
def test_grader_scores_a_malformed_capture_zero_without_running_tests(grader: _Grader, case: str) -> None:
    # The capture artifacts come from the agent's sandbox: whatever shape they have, the sample is a completed 0
    # (never exit 75), and a FIFO is refused without blocking.
    if "/proc/self/mem" in case and not Path("/proc/self/mem").exists():
        pytest.skip("needs Linux procfs")
    _plant_capture(grader.artifacts, case, (grader.module.CAPTURE_MAX_BYTES, grader.module.PATCH_MAX_BYTES))
    start = time.monotonic()
    code, reward, grading = grader.run()
    assert time.monotonic() - start < 30
    assert code == 0 and grading["masked"] is False
    assert reward["reward"] == 0.0 and reward["capture_failed"] == 1.0 and reward["evaluation_completed"] == 1.0
    assert grader.calls == []
    # Nothing nested from the agent's JSON reaches grading.json.
    assert "[[" not in (grader.logs / "grading.json").read_text()


def test_grader_grades_a_patch_that_is_not_utf8(grader: _Grader) -> None:
    grader.capture("ok", patch=None)
    (grader.artifacts / "model_patch.diff").write_bytes(b"diff --git a/m.py b/m.py\n+x = '\xff'\n")
    grader.result = lambda v: v.VerificationResult(
        completed=True, resolved=False, patch_applied=True, test_results={}, test_output=v.TEST_OUTPUT_BEGIN
    )
    code, reward, grading = grader.run()
    assert code == 0 and reward["reward"] == 0.0 and "capture_failed" not in reward
    assert [call[0] for call in grader.calls if isinstance(call, tuple)] == ["run"]
    assert reward["model_patch_bytes"] == float(len("diff --git a/m.py b/m.py\n+x = '�'\n".encode()))


def test_grader_scores_a_patch_the_server_code_cannot_split_zero(
    grader: _Grader, monkeypatch: pytest.MonkeyPatch
) -> None:
    grader.capture("ok")
    real_load = grader.module._load

    def load(name: str, path: Path):
        module = real_load(name, path)

        def explode(patch: str, test_patch: str) -> str:
            raise RecursionError("simulated")

        monkeypatch.setattr(module, "drop_test_patch_files", explode)
        return module

    monkeypatch.setattr(grader.module, "_load", load)
    code, reward, grading = grader.run()
    assert code == 0 and grading["masked"] is False and reward["reward"] == 0.0 and reward["capture_failed"] == 1.0
    assert grader.calls == []


@pytest.mark.parametrize("resolved", [True, False])
def test_grader_scores_a_verdict(grader: _Grader, resolved: bool) -> None:
    grader.capture("ok")
    grader.result = lambda v: v.VerificationResult(
        completed=True,
        resolved=resolved,
        patch_applied=True,
        test_results={"tests_observed": 2},
        test_output=v.TEST_OUTPUT_BEGIN + "\n",
    )
    code, reward, grading = grader.run()
    assert code == 0 and grading["masked"] is False
    assert reward["reward"] == float(resolved) and reward["resolved"] == float(resolved)
    assert reward["evaluation_completed"] == 1.0 and reward["tests_observed"] == 2.0
    assert reward["no_verdict_after_tests"] == 0.0
    # The reward path and the grader's mapped files are protected and the audit hook armed before the agent's code
    # runs; that code runs with the privileges dropped, bounded by the server's evaluation_timeout; the mappings are
    # checked before the verdict.
    assert grader.calls == ["ownership", "protect", "prepare", "arm", ("run", True, 1200, "repo_a_pr1"), "check"]


def _no_verdict_result(v, *, return_code_97: bool = False, marker: bool = False, begin: bool = True):
    output = ("git apply output\n" if not marker else f"error: {v.TEST_PATCH_FAILED}: No such file\n") + (
        f"{v.TEST_OUTPUT_BEGIN}\nFAILED t.py::test_fixed\n" if begin else ""
    )
    if return_code_97:
        return v.VerificationResult(
            completed=False, resolved=False, patch_applied=False, test_results=None, test_output=output, error="97"
        )
    return v.VerificationResult(
        completed=False,
        resolved=False,
        patch_applied=True,
        test_results={"tests_observed": 1},
        test_output=output + (f"{v.TEST_PATCH_FAILED}\n" if not marker else ""),
        error="held-out test patch did not apply",
        test_patch_failed=True,
    )


@pytest.mark.parametrize(
    "case",
    [
        "exit 97 from the tests",
        "marker printed by the tests",
        "marker printed before TEST_OUTPUT_BEGIN (the agent's patch file names)",
    ],
)
def test_grader_scores_a_run_the_tests_ended_without_a_verdict_zero(grader: _Grader, case: str) -> None:
    # After TEST_OUTPUT_BEGIN the agent's code is running, and git prints the agent's patch paths before it: neither
    # exit 97 nor the test-patch marker is a fact the agent cannot write, so neither masks.
    grader.capture("ok")
    grader.result = lambda v: _no_verdict_result(v, return_code_97=case.startswith("exit 97"), marker="before" in case)
    code, reward, grading = grader.run()
    assert code == 0 and grading["masked"] is False
    assert reward["reward"] == 0.0 and reward["resolved"] == 0.0 and reward["no_verdict_after_tests"] == 1.0


@pytest.mark.parametrize("case", ["test-patch marker", "exit 97"])
def test_grader_masks_a_run_that_stopped_before_the_tests(grader: _Grader, case: str) -> None:
    # No TEST_OUTPUT_BEGIN: the eval script stopped before any agent code ran.
    grader.capture("ok")
    grader.result = lambda v: _no_verdict_result(v, return_code_97=case == "exit 97", begin=False)
    code, reward, grading = grader.run()
    assert code == grader.module.NO_VERDICT == 75
    assert reward is None and grading["masked"] is True


def test_grader_masks_a_run_without_test_output(grader: _Grader) -> None:
    # A normal exit without TEST_OUTPUT_BEGIN is the server's infrastructure_failure.
    grader.capture("ok")
    grader.result = lambda v: v.VerificationResult(
        completed=True, resolved=False, patch_applied=True, test_results={}, test_output="pre_commands output\n"
    )
    code, reward, grading = grader.run()
    assert code == 75 and reward is None and grading["reason"] == "eval script stopped before the tests started"


def test_grader_masks_a_missing_workdir_before_any_test(grader: _Grader, tmp_path: Path) -> None:
    grader.capture("ok")
    task_json = tmp_path / "ds" / "repo_a_pr1" / "tests" / "nemo_gym" / "task.json"
    task = json.loads(task_json.read_text())
    task["row"]["workdir"] = str(tmp_path / "no_such_workdir")
    task_json.write_text(json.dumps(task))
    code, reward, grading = grader.run()
    assert code == 75 and reward is None and grading["reason"].startswith("workdir ")
    assert grader.calls == []


def test_grader_masks_when_the_reward_cannot_be_protected(grader: _Grader, monkeypatch: pytest.MonkeyPatch) -> None:
    grader.capture("ok")

    def fail() -> None:
        raise PermissionError("simulated")

    monkeypatch.setattr(grader.module, "_protect_reward_path", fail)
    code, reward, grading = grader.run()
    assert code == 75 and reward is None
    assert grading["reason"].startswith("could not protect the reward folder")
    assert grader.calls == ["ownership"]


def test_grader_masks_when_its_mapped_files_cannot_be_protected(
    grader: _Grader, monkeypatch: pytest.MonkeyPatch
) -> None:
    grader.capture("ok")

    def fail() -> dict:
        raise PermissionError("simulated")

    monkeypatch.setattr(grader.module, "_prepare_after_tests", fail)
    code, reward, grading = grader.run()
    assert code == 75 and reward is None
    assert grading["reason"].startswith("could not protect the files the grader has mapped")
    assert grader.calls == ["ownership", "protect"]


def test_grader_masks_when_it_raises(grader: _Grader, monkeypatch: pytest.MonkeyPatch) -> None:
    # Before the tests started (no subprocess here), an exception masks, as the server masks a verify that raised.
    grader.capture("ok")
    grader.result = lambda v: 1 / 0
    code, reward, grading = grader.run()
    assert code == 75 and reward is None and grading["reason"] == "the grader raised"


def test_write_output_writes_utf8_bytes_without_a_text_layer(grader: _Grader, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_fdopen(*args, **kwargs):
        raise AssertionError("os.fdopen would look up a codec")

    monkeypatch.setattr(grader.module.os, "fdopen", no_fdopen)
    grader.logs.mkdir()
    grader.module._write_output("grading.json", '{"x": "é中"}')
    assert (grader.logs / "grading.json").read_bytes() == '{"x": "é中"}'.encode()


def test_exit_record_says_1_until_the_wrapper_overwrites_it(grader: _Grader, monkeypatch: pytest.MonkeyPatch) -> None:
    # A test process that kills the wrapper around test.sh leaves this record: the harbor server reads exit 1 without
    # a reward, a measured 0, instead of a run with no record.
    from environment_servers.harbor.verifier import EXIT_RECORD_NAME, parse_exit_record

    module = grader.module
    grader.logs.mkdir()
    (grader.logs / EXIT_RECORD_NAME).write_text("75 0 1\n")
    (grader.logs / "reward.json").write_text('{"reward": 1}')
    monkeypatch.setattr(module, "REWARD_OWNER_UID", os.getuid())
    monkeypatch.setattr(module, "PROTECTED_DIRS", (grader.logs,))
    grader.protect_reward_path()
    assert module.EXIT_RECORD_NAME == EXIT_RECORD_NAME
    assert (grader.logs / EXIT_RECORD_NAME).read_text() == "1\n"
    assert parse_exit_record(grader.logs) == (1, None)
    assert not (grader.logs / "reward.json").exists()


# The functions and constants swe_rebench's grade.py copies from this one (only the docstring and grade() differ).
SHARED_GRADER_NAMES = (
    "HERE ARTIFACTS LOGS NO_VERDICT FAILED_CLOSED REWARD_OWNER_UID PROTECTED_DIRS EXIT_RECORD_NAME KEPT_CAPABILITIES"
    " _PR_CAPBSET_DROP _PR_SET_NO_NEW_PRIVS _PR_CAP_AMBIENT _PR_CAP_AMBIENT_CLEAR_ALL _LINUX_CAPABILITY_VERSION_3"
    " CAPTURE_MAX_BYTES PATCH_MAX_BYTES CAPTURE_TEXT_MAX_CHARS CAPTURE_WARNINGS_MAX PROC_MAPS WARM_IMPORTS"
    " REFUSED_AFTER_TESTS _AFTER_TESTS _FailClosed _Unreadable _CapHeader _CapData _privilege_dropper"
    " _give_root_ownership _protect_reward_path _write_output _load _no_verdict _verdict _capture_failed _fail_closed"
    " _read_agent_file _capture_summary _read_capture _inline_to_thread _warm_event_loop _read_mappings _mapped_paths"
    " _mapping_ids _protected_mode _file_state _protect_mappings _prepare_after_tests _after_tests_hook"
    " _arm_after_tests _check_mappings main"
).split()


def _top_level_sources(path: Path) -> dict[str, str]:
    text = path.read_text()
    sources = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            sources[node.name] = ast.get_source_segment(text, node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            sources[node.targets[0].id] = ast.get_source_segment(text, node)
    return sources


def test_swe_rebench_grader_shares_the_protection_code() -> None:
    ours = _top_level_sources(TEMPLATE / "tests/nemo_gym/grade.py")
    theirs = _top_level_sources(
        REPO_ROOT / "resources_servers/swe_rebench/harbor/task-template/tests/nemo_gym/grade.py"
    )
    assert set(ours) - {"grade"} == set(theirs) - {"grade"} == set(SHARED_GRADER_NAMES)
    for name in SHARED_GRADER_NAMES:
        assert ours[name] == theirs[name], name


# -- grade.py with the audit hook armed and its mapped files protected, in a subprocess ------------------------------

# Runs a converted task's grade.py main() as the verifier would, with only what needs root redirected. Only files
# under the scratch folder are protected (the mapping check still sees every mapping), the eval script and the
# workdir live there, and no ownership is moved. Without root, the reward owner is this uid (chown to oneself needs
# no privilege), seteuid is a no-op and the privilege drop is skipped; as root, they are the real ones. The eval
# script is a real subprocess and the vendored run_verification and parser are real. Prints the outcome as a Python
# literal (json may be what a test attacks).
DRIVER = r"""
import os
import sys

here, scratch, scenario = sys.argv[1], sys.argv[2], sys.argv[3]
if scenario == "json_copy":
    sys.path.insert(0, os.path.join(scratch, "lib"))
    import _json  # the scratch copy, mapped before json uses it
import importlib
import importlib.util
import mmap
from pathlib import Path

S = Path(scratch)
target = S / "lib" / os.environ["NG_TARGET"] if os.environ.get("NG_TARGET") else None
mapping = None
if target is not None:
    with open(target, "rb") as handle:
        mapping = mmap.mmap(handle.fileno(), 0, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ)
sys.path.insert(0, str(S / "lib"))
sys.path.insert(0, here)
import local_sandbox

spec = importlib.util.spec_from_file_location("grade_under_test", os.path.join(here, "grade.py"))
grade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grade)
grade.ARTIFACTS, grade.LOGS = S / "artifacts", S / "verifier"
grade.PROTECTED_DIRS = (S / "verifier",)
grade._give_root_ownership = lambda: 0
if os.geteuid() != 0:
    grade.REWARD_OWNER_UID = os.getuid()
    os.seteuid = lambda uid: None
    grade._privilege_dropper = lambda: None
all_mapped_paths = grade._mapped_paths
grade._mapped_paths = lambda: [path for path in all_mapped_paths() if path.startswith(scratch + os.sep)]
real_load = grade._load


def load(name, path):
    module = real_load(name, path)
    if hasattr(module, "verification_files"):
        module.verification_files = lambda inputs, **kwargs: {}
    if hasattr(module, "repo_directory"):
        module.repo_directory = lambda repo: str(S / "repo")
    return module


grade._load = load
real_exec = local_sandbox.LocalSandbox.exec


async def exec_(self, command, timeout_s=None, **kwargs):
    result = await real_exec(self, command.replace("/tmp/nemo_gym_eval.sh", str(S / "eval.sh")), timeout_s, **kwargs)
    if scenario == "lazy_import":
        importlib.import_module("standin")
    if scenario == "mapped_file_changed":
        os.chmod(target, 0o664)
    if scenario == "raise_after_tests":
        raise RuntimeError("simulated failure after the tests")
    return result


local_sandbox.LocalSandbox.exec = exec_
code = grade.main()
# The hook stays armed in this process: let it open what this driver reads back.
grade._AFTER_TESTS["allowed"] |= {str(S / "verifier" / ".ng-exit")}


def read(name):
    path = S / "verifier" / name
    return open(path, "rb").read().decode() if path.exists() else None


outcome = {
    "grade_file": grade.__file__,
    "json_file": getattr(sys.modules["_json"], "__file__", None),
    "code": code,
    "reward": read("reward.json"),
    "grading": read("grading.json"),
    "exit_record": read(".ng-exit"),
    "test_output": read("test_output.log"),
    "refused": grade._AFTER_TESTS["refused"],
    "mapped_prefix": bytes(mapping[:8]) if mapping is not None else None,
}
print("OUTCOME", repr(outcome))
"""

PASSING = "echo ___NEMO_GYM_SCALE_SWE_TEST_BEGIN___\necho 'PASSED t.py::test_fixed'\necho 'PASSED t.py::test_kept'\n"
FAILING = "echo ___NEMO_GYM_SCALE_SWE_TEST_BEGIN___\necho 'FAILED t.py::test_fixed'\necho 'PASSED t.py::test_kept'\n"


def _interpreters() -> list[str]:
    """This Python and the system one (another version and build) when it can run the grader (3.9+)."""
    found = [sys.executable]
    system = shutil.which("python3", path="/usr/bin:/usr/local/bin")
    if system and os.path.realpath(system) != os.path.realpath(sys.executable):
        version = subprocess.run([system, "-I", "-c", "import sys; print(sys.version_info >= (3, 9))"], **_QUIET)
        if version.stdout.strip() == "True":
            found.append(system)
    return found


def _interpreter_params() -> list:
    return [pytest.param(python, id=f"python{index}") for index, python in enumerate(_interpreters())]


_QUIET = {"capture_output": True, "text": True, "timeout": 60}


# Deeper than the default recursion limit (1000), and shorter than PATH_MAX under tmp_path, as a tree the agent's
# restored /logs/artifacts can hold.
OWNERSHIP_TREE_DEPTH = 1100
OWNERSHIP_DRIVER = r"""
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location("grade_under_test", sys.argv[1])
grade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grade)
chowned = []
os.lchown = lambda path, uid, gid: chowned.append(path)
count = grade._give_root_ownership(sys.argv[2])
print("OUTCOME", repr({"grade_file": grade.__file__, "count": count, "chowned": sorted(chowned)}))
"""


def _ownership_tree(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    """A walked root holding a folder tree OWNERSHIP_TREE_DEPTH deep and a symlink to a folder outside it; returns
    the root, the outside folder and every entry under the root."""
    top, outside = tmp_path / "root", tmp_path / "outside"
    (outside / "inner").mkdir(parents=True)
    folder, entries = top / "artifacts", []
    folder.mkdir(parents=True)
    entries.append(str(folder))
    for _ in range(OWNERSHIP_TREE_DEPTH):
        folder = folder / "d"
        folder.mkdir()
        entries.append(str(folder))
    (folder / "leaf.txt").write_text("x")
    (top / "link").symlink_to(outside, target_is_directory=True)
    return top, outside, sorted([*entries, str(folder / "leaf.txt"), str(top / "link")])


def _expected_chowned(entries: list[str]) -> list[str]:
    # Without root every entry is this uid's, so each one is handed to root; as root none needs to be.
    return entries if os.getuid() != 0 else []


def test_root_ownership_walk_does_not_recurse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The image's Python (3.9 to 3.11) runs the grader, and its os.walk recurses: a tree as deep as the agent's
    # restored artifacts can make raised RecursionError before the tests and masked the sample. The walk keeps its own
    # list of folders, follows no symlink and moves the ownership of every entry it lists.
    module = _load(f"grade_ownership_{tmp_path.name}", TEMPLATE / "tests/nemo_gym/grade.py")
    top, outside, entries = _ownership_tree(tmp_path)
    chowned: list[str] = []

    def recursive_walk(*args, **kwargs):
        raise AssertionError("os.walk recurses before Python 3.12")

    monkeypatch.setattr(module.os, "walk", recursive_walk)
    monkeypatch.setattr(module.os, "lchown", lambda path, uid, gid: chowned.append(path))
    assert module._give_root_ownership(str(top)) == len(_expected_chowned(entries))
    assert sorted(chowned) == _expected_chowned(entries)
    assert not any(path.startswith(str(outside)) for path in chowned)


def _walk_interpreters() -> list:
    """The interpreters of ``_interpreters`` and every Python 3.9 to 3.11 on PATH (their os.walk recurses)."""
    found = {os.path.realpath(python): python for python in _interpreters()}
    for minor in (9, 10, 11):
        python = shutil.which(f"python3.{minor}")
        if python:
            found.setdefault(os.path.realpath(python), python)
    return [pytest.param(python, id=Path(python).name) for python in found.values()]


@pytest.mark.parametrize("python", _walk_interpreters())
def test_root_ownership_walk_handles_a_deep_tree_on_every_python(tmp_path: Path, python: str) -> None:
    top, _, entries = _ownership_tree(tmp_path)
    (tmp_path / "driver.py").write_text(OWNERSHIP_DRIVER)
    grade_file = TEMPLATE / "tests/nemo_gym/grade.py"
    process = subprocess.run([python, "-I", "-B", str(tmp_path / "driver.py"), str(grade_file), str(top)], **_QUIET)
    lines = [line for line in process.stdout.splitlines() if line.startswith("OUTCOME ")]
    assert process.returncode == 0 and len(lines) == 1, process.stderr[-2000:]
    outcome = ast.literal_eval(lines[0][len("OUTCOME ") :])
    assert Path(outcome["grade_file"]) == grade_file
    assert outcome["count"] == len(_expected_chowned(entries))
    assert outcome["chowned"] == _expected_chowned(entries)


def _drive(
    tmp_path: Path, rows: Path, eval_script: str, scenario: str = "plain", *, python: str = sys.executable, **env
) -> tuple[dict, subprocess.CompletedProcess]:
    """Run grade.py's main() of a converted task in a subprocess (see ``DRIVER``) and return its outcome."""
    if not (tmp_path / "ds").exists():
        assert _convert(rows, tmp_path / "ds") == 0
    here = tmp_path / "ds" / "repo_a_pr1" / "tests" / "nemo_gym"
    scratch = tmp_path / "scratch"
    for folder in ("artifacts", "verifier", "repo", "lib"):
        (scratch / folder).mkdir(parents=True, exist_ok=True)
    task = json.loads((here / "task.json").read_text())
    task["row"]["workdir"], task["evaluation_timeout"] = str(scratch / "repo"), 60
    (here / "task.json").write_text(json.dumps(task))
    (scratch / "artifacts" / "capture.json").write_text('{"status": "ok"}')
    (scratch / "artifacts" / "model_patch.diff").write_text("diff --git a/m.py b/m.py\n")
    (scratch / "eval.sh").write_text(eval_script + "echo ___NEMO_GYM_SCALE_SWE_TEST_END___\n")
    (tmp_path / "driver.py").write_text(DRIVER)
    process = subprocess.run(
        [python, "-I", "-B", str(tmp_path / "driver.py"), str(here), str(scratch), scenario],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **env},
    )
    lines = [line for line in process.stdout.splitlines() if line.startswith("OUTCOME ")]
    assert len(lines) == 1, process.stdout + process.stderr
    outcome = ast.literal_eval(lines[0][len("OUTCOME ") :])
    # The code under test is the converted task's grade.py.
    assert Path(outcome["grade_file"]) == here / "grade.py"
    return outcome, process


@pytest.mark.parametrize("python", _interpreter_params())
@pytest.mark.parametrize("resolved", [True, False])
def test_armed_grader_scores_a_real_run(tmp_path: Path, rows: Path, python: str, resolved: bool) -> None:
    # The whole path after the tests (the vendored run_verification and parser, the inline to_thread, the outputs)
    # runs under the audit hook with one real subprocess, and needs no import, no new mapping and no refused open.
    outcome, process = _drive(tmp_path, rows, PASSING if resolved else FAILING, python=python)
    assert outcome["code"] == 0, process.stderr
    assert outcome["refused"] is None and "failing closed" not in process.stderr
    assert json.loads(outcome["reward"])["reward"] == float(resolved)
    assert json.loads(outcome["grading"])["masked"] is False
    assert outcome["exit_record"] == "1\n"
    assert "PASSED t.py::test_kept" in outcome["test_output"]


@pytest.mark.parametrize("python", _interpreter_params())
def test_in_place_write_to_a_mapped_file_is_refused(tmp_path: Path, rows: Path, python: str) -> None:
    # The mutation control: the tests (the eval script, same uid) try to rewrite a file the grader has mapped. The
    # protection takes its write bits and its owner, so the write fails, the mapped bytes stay, the check passes and
    # the failing run still scores 0.
    (tmp_path / "scratch" / "lib").mkdir(parents=True)
    target = tmp_path / "scratch" / "lib" / "mapped.bin"
    target.write_bytes(b"A" * 4096)
    attack = f"printf 'XXXXXXXX' 1<> {shlex.quote(str(target))} && echo WRITE_LANDED || echo WRITE_REFUSED\n"
    outcome, process = _drive(tmp_path, rows, attack + FAILING, python=python, NG_TARGET="mapped.bin")
    assert "WRITE_REFUSED" in outcome["test_output"] and "WRITE_LANDED" not in outcome["test_output"]
    assert target.read_bytes() == b"A" * 4096 and outcome["mapped_prefix"] == b"A" * 8
    assert outcome["code"] == 0 and outcome["refused"] is None, process.stderr
    assert json.loads(outcome["reward"])["reward"] == 0.0


def test_changed_mapped_file_fails_closed(tmp_path: Path, rows: Path) -> None:
    # The check after the tests: a mapped file whose mode (or owner, size, mtime) changed while the tests ran means
    # the protection did not hold, so the grader exits 1 without a reward, even for a passing run. The change is made
    # in the grader process just after the tests, since the tests themselves cannot make it (the test above).
    (tmp_path / "scratch" / "lib").mkdir(parents=True)
    (tmp_path / "scratch" / "lib" / "mapped.bin").write_bytes(b"A" * 4096)
    outcome, process = _drive(tmp_path, rows, PASSING, "mapped_file_changed", NG_TARGET="mapped.bin")
    assert outcome["code"] == 1 and outcome["reward"] is None
    assert "a file the grader has mapped changed during the tests" in process.stderr


def _file_backed_json() -> tuple[str, Path] | None:
    for python in _interpreters():
        found = subprocess.run([python, "-I", "-c", "import _json; print(getattr(_json, '__file__', ''))"], **_QUIET)
        if found.stdout.strip():
            return python, Path(found.stdout.strip())
    return None


def test_in_place_write_to_the_mapped_json_extension_is_refused(tmp_path: Path, rows: Path) -> None:
    # The review's forgery rewrote the loaded _json extension in place. Here the grader's json uses a scratch copy
    # of it, and the tests' in-place write to that copy is refused before any byte changes.
    found = _file_backed_json()
    if found is None:
        pytest.skip("no Python here has a file-backed _json extension")
    python, json_lib = found
    lib = tmp_path / "scratch" / "lib"
    lib.mkdir(parents=True)
    shutil.copyfile(json_lib, lib / json_lib.name)
    original = (lib / json_lib.name).read_bytes()
    attack = (
        f"printf 'XXXXXXXX' 1<> {shlex.quote(str(lib / json_lib.name))} && echo WRITE_LANDED || echo WRITE_REFUSED\n"
    )
    outcome, process = _drive(tmp_path, rows, attack + FAILING, "json_copy", python=python, NG_TARGET=json_lib.name)
    assert outcome["json_file"] == str(lib / json_lib.name)
    assert "WRITE_REFUSED" in outcome["test_output"]
    assert (lib / json_lib.name).read_bytes() == original
    assert outcome["code"] == 0, process.stderr
    assert json.loads(outcome["reward"])["reward"] == 0.0


def test_import_after_the_tests_fails_closed(tmp_path: Path, rows: Path) -> None:
    # A module the grader would import after the tests is one the tests could have rewritten: here they make it
    # forge json.dumps. The import is refused, so the grader exits 1 without a reward instead of writing a 1.
    lib = tmp_path / "scratch" / "lib"
    lib.mkdir(parents=True)
    (lib / "standin.py").write_text("")
    forged = "import json\njson.dumps = lambda *args, **kwargs: '{\"reward\": 1.0}'\n"
    attack = f"printf '%s' {shlex.quote(forged)} > {shlex.quote(str(lib / 'standin.py'))}\n"
    outcome, process = _drive(tmp_path, rows, attack + FAILING, "lazy_import")
    assert "json.dumps" in (lib / "standin.py").read_text()
    assert outcome["code"] == 1 and outcome["reward"] is None
    # Refused at its first file access: the open of standin's bytecode or source (an extension module would be
    # refused at the import event its loader raises).
    assert "after the tests started" in outcome["refused"] and "standin" in outcome["refused"]
    assert "failing closed after the tests started" in process.stderr


def test_exception_after_the_tests_fails_closed_never_masks(tmp_path: Path, rows: Path) -> None:
    outcome, process = _drive(tmp_path, rows, PASSING, "raise_after_tests")
    assert outcome["code"] == 1 and outcome["code"] != 75 and outcome["reward"] is None
    assert "RuntimeError: simulated failure after the tests" in process.stderr
    # The pre-created exit record stays at 1 (the wrapper writes the real code after test.sh).
    assert outcome["exit_record"] == "1\n"


@needs_git
def test_capture_hook_reads_the_patch_and_splits_errors(
    tmp_path: Path, rows: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _convert(rows, tmp_path / "ds") == 0
    state = tmp_path / "state"
    for name, data in _bundle((tmp_path / "ds" / "repo_a_pr1" / "environment" / "Dockerfile").read_text()).items():
        (state / name).parent.mkdir(parents=True, exist_ok=True)
        (state / name).write_bytes(data)
    repo = _git_repo(tmp_path)
    for key, value in _git_env(tmp_path).items():
        monkeypatch.setenv(key, value)
    base = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    (state / "config.json").write_text(json.dumps({"workdir": str(repo), "patch_capture_mode": "worktree"}))
    (state / "base_commit").write_text(base)
    (state / "pristine_untracked.txt").write_text("")
    monkeypatch.setattr(sys, "path", list(sys.path))
    capture = _load(f"scale_swe_capture_{tmp_path.name}", state / "capture" / "capture.py")
    out = tmp_path / "artifacts"
    monkeypatch.setattr(capture, "OUT_DIR", out)

    def status() -> str:
        return json.loads((out / "capture.json").read_text())["status"]

    (repo / "m.py").write_text("x = 2\n")
    assert capture.main() == 0
    assert status() == "ok" and "+x = 2" in (out / "model_patch.diff").read_text()

    real_load = capture._load
    for error_type, expected in ((None, "git_error"), ("timeout", "infra_error"), ("sandbox", "infra_error")):

        def load(name: str, path: Path, error_type: str | None = error_type):
            module = real_load(name, path)
            if hasattr(module, "CaptureCommandError"):

                async def fail(*args, **kwargs):
                    raise module.CaptureCommandError("failed", return_code=128, error_type=error_type)

                monkeypatch.setattr(module, "capture_model_patch", fail)
            return module

        monkeypatch.setattr(capture, "_load", load)
        assert capture.main() == 0
        assert status() == expected

    monkeypatch.setattr(capture, "_load", real_load)
    (state / "base_commit").write_text("not a commit\n")
    assert capture.main() == 0
    assert status() == "git_error"
    assert json.loads((out / "capture.json").read_text())["error"].startswith("seed base commit was changed")


def test_local_sandbox_exec_returns_output_and_times_out(tmp_path: Path) -> None:
    # A copy, so importing it leaves no bytecode cache in the template.
    shutil.copy(TEMPLATE / "tests/nemo_gym/local_sandbox.py", tmp_path / "local_sandbox.py")
    local_sandbox = _load("scale_swe_local_sandbox_under_test", tmp_path / "local_sandbox.py")
    result = asyncio.run(local_sandbox.LocalSandbox().exec("echo out; echo err >&2; exit 3", timeout_s=30))
    assert (result.stdout, result.stderr, result.return_code, result.error_type) == ("out\n", "err\n", 3, None)
    start = time.monotonic()
    result = asyncio.run(local_sandbox.LocalSandbox().exec("echo started; sleep 30", timeout_s=1))
    assert (result.stdout, result.return_code, result.error_type) == ("started\n", None, "timeout")
    assert time.monotonic() - start < 10


def _setup_script(workdir: str, *, apply_anti_cheating: bool = True) -> str:
    settings = dataclasses.replace(adapter.ServerSettings.read(), apply_anti_cheating=apply_anti_cheating)
    return adapter.ScaleSWEHarborAdapter(settings=settings).setup_script({**_row("repo_a_pr1"), "workdir": workdir})


def _run_setup(tmp_path: Path, repo: Path) -> tuple[subprocess.CompletedProcess, Path]:
    state = tmp_path / "state"
    state.mkdir()
    shutil.copy(REPO_ROOT / adapter.ANTI_CHEAT_MODULE, state / "anti_cheat_setup.sh")
    script = _setup_script(str(repo))
    assert script.count(f"\nS={adapter.STATE_DIR}\n") == 1
    script = script.replace(f"\nS={adapter.STATE_DIR}\n", f"\nS={state}\n")
    process = subprocess.run(
        ["bash", "-c", script], env=_git_env(tmp_path), capture_output=True, text=True, timeout=120
    )
    return process, state


@needs_git
def test_setup_script_seeds_a_repository(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    (repo / "notes.txt").write_text("untracked before the agent\n")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    process, state = _run_setup(tmp_path, repo)
    assert process.returncode == 0, process.stderr
    assert (state / "base_commit").read_text() == head
    assert (state / "pristine_untracked.txt").read_text() == "notes.txt\n"
    # The anti-cheat scrub ran and removed its own copy from the workdir.
    branch = subprocess.run(["git", "-C", str(repo), "symbolic-ref", "HEAD"], capture_output=True, text=True).stdout
    assert branch == "refs/heads/_nel_work\n" and not (repo / "anti_cheat_setup.sh").exists()


@needs_git
def test_setup_script_fails_without_a_base_commit(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path, commit=False)
    process, state = _run_setup(tmp_path, repo)
    assert process.returncode == 1
    assert "did not name a commit" in process.stderr
    assert not (state / "base_commit").exists()


@needs_git
def test_setup_script_creates_a_repository_when_the_image_has_none(tmp_path: Path) -> None:
    workdir = tmp_path / "plain"
    workdir.mkdir()
    (workdir / "m.py").write_text("x = 1\n")
    process, state = _run_setup(tmp_path, workdir)
    assert process.returncode == 0, process.stderr
    assert re.fullmatch(r"[0-9a-f]{40}\n", (state / "base_commit").read_text())


def test_anti_cheat_step_follows_the_server_setting() -> None:
    assert "bash anti_cheat_setup.sh" in _setup_script("/testbed", apply_anti_cheating=True)
    assert "anti_cheat_setup.sh" not in _setup_script("/testbed", apply_anti_cheating=False)


def test_solve_script_and_capture_hook(tmp_path: Path, rows: Path) -> None:
    assert _convert(rows, tmp_path / "ds") == 0
    task = tmp_path / "ds" / "repo_a_pr1"
    assert (task / "solution" / "solve.sh").read_text() == (
        "#!/bin/bash\n"
        "# Applies the row's golden patch with the flags the server's eval script applies a patch with\n"
        "# (verification.build_eval_script), for `gym dataset validate`.\n"
        "cd /testbed || exit 1\n"
        'git apply --reject --recount --ignore-space-change --whitespace=nowarn "$(dirname "$0")/golden.patch"'
        " || true\n"
    )
    collect = tomllib.loads((task / "task.toml").read_text())["verifier"]["collect"]
    assert collect == [
        {
            "service": "main",
            "timeout_sec": adapter.CAPTURE_TIMEOUT_SEC,
            "command": "rm -rf /logs/artifacts/nemo_gym; python3 -I /var/lib/nemo_gym/capture/capture.py;"
            " test -f /logs/artifacts/nemo_gym/capture.json || { mkdir -p /logs/artifacts/nemo_gym && echo"
            ' \'{"status": "runner_failed", "error": "python3 could not run capture.py"}\''
            " > /logs/artifacts/nemo_gym/capture.json; }",
        }
    ]
