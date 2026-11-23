# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Scale-SWE Harbor adapter (``resources_servers/scale_swe/harbor/adapter.py``).

They convert tiny rows in ``tmp_path`` and need no network or sandbox; running a converted task is the oracle run
(``gym dataset validate``), not these tests. The grader, the capture hook and the seed script run here against
temporary git repositories, with the steps that need root (ownership, the privilege drop) stubbed out.
"""

import asyncio
import base64
import dataclasses
import hashlib
import importlib.util
import io
import json
import os
import re
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
    """``grade.py`` of a converted task, with its outputs under tmp_path and the root-only steps stubbed."""

    def __init__(self, tmp_path: Path, rows: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _convert(rows, tmp_path / "ds") == 0
        here = tmp_path / "ds" / "repo_a_pr1" / "tests" / "nemo_gym"
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
        monkeypatch.setattr(self.module, "_protect_reward_path", lambda: self.calls.append("protect"))

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
    # The reward path is protected before the agent's code runs, and that code runs with the privileges dropped,
    # bounded by the server's evaluation_timeout.
    assert grader.calls == ["ownership", "protect", ("run", True, 1200, "repo_a_pr1")]


def test_grader_masks_a_run_without_a_verdict(grader: _Grader) -> None:
    grader.capture("ok")
    grader.result = lambda v: v.VerificationResult(
        completed=False,
        resolved=False,
        patch_applied=True,
        test_results=None,
        test_output=v.TEST_PATCH_FAILED,
        error="held-out test patch did not apply",
        test_patch_failed=True,
    )
    code, reward, grading = grader.run()
    assert code == grader.module.NO_VERDICT == 75
    assert reward is None and grading["masked"] is True
    assert grading["reason"] == "held-out test patch did not apply"


def test_grader_masks_when_the_reward_cannot_be_protected(grader: _Grader, monkeypatch: pytest.MonkeyPatch) -> None:
    grader.capture("ok")

    def fail() -> None:
        raise PermissionError("simulated")

    monkeypatch.setattr(grader.module, "_protect_reward_path", fail)
    code, reward, grading = grader.run()
    assert code == 75 and reward is None
    assert grading["reason"].startswith("could not protect the reward folder")
    assert grader.calls == ["ownership"]


def test_grader_masks_when_it_raises(grader: _Grader, monkeypatch: pytest.MonkeyPatch) -> None:
    grader.capture("ok")
    grader.result = lambda v: 1 / 0
    code, reward, grading = grader.run()
    assert code == 75 and reward is None and grading["reason"] == "the grader raised"


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
