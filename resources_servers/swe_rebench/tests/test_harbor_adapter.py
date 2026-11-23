# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the SWE-rebench-V2 Harbor adapter (``resources_servers/swe_rebench/harbor/adapter.py``).

They run without the network: the upstream parser tree is a stand-in folder, and the adapter's check that it sits at
the server's pinned commit is pointed at that folder. Running a converted task needs a sandbox; that is the oracle
run (``gym dataset validate``), not these tests. The grader, the capture hook, the seed and ``local_sandbox.py`` run
here from copies under ``tmp_path``, against temporary git repositories, with the steps that need root (ownership,
the privilege drop, the uid switch) stubbed out.
"""

import asyncio
import base64
import gzip
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
ADAPTER_PATH = REPO_ROOT / "resources_servers/swe_rebench/harbor/adapter.py"
EXAMPLE_ROWS = REPO_ROOT / "resources_servers/swe_rebench/data/example.jsonl"
PAIRING_CONFIG = REPO_ROOT / "resources_servers/swe_rebench/configs/swe_rebench.yaml"
TEMPLATE = REPO_ROOT / "resources_servers/swe_rebench/harbor/task-template"
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_adapter():
    return _load("swe_rebench_harbor_adapter_under_test", ADAPTER_PATH)


adapter = _load_adapter()


@pytest.fixture
def parser_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in upstream tree whose HEAD the adapter reads as the server's pinned commit.

    Its ``lib/agent/log_parsers.py`` defines ``NAME_TO_PARSER`` with every parser the example rows name, so the
    vendored server ``log_parsers.resolve_parser`` can load it from a converted task's ``parser_cache``.
    """
    repo = tmp_path / "upstream"
    for relative in adapter.UPSTREAM_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# stand-in for {relative}\n")
    names = sorted(
        {json.loads(line)["install_config"]["log_parser"] for line in EXAMPLE_ROWS.read_text().splitlines()}
    )
    (repo / "lib/agent/log_parsers.py").write_text(
        "# stand-in for lib/agent/log_parsers.py\n"
        "def parse_log_stand_in(log):\n"
        "    return {}\n\n\n"
        f"NAME_TO_PARSER = {{name: parse_log_stand_in for name in {names!r}}}\n"
    )
    pinned = adapter._load_repo_module(adapter.LOG_PARSERS_MODULE).UPSTREAM_COMMIT
    real_git = adapter._git

    def fake_git(*args: str, cwd: Path = adapter.REPO_ROOT) -> str | None:
        if Path(cwd) == repo and args == ("rev-parse", "HEAD"):
            return pinned
        return real_git(*args, cwd=cwd)

    monkeypatch.setattr(adapter, "_git", fake_git)
    return repo


@pytest.fixture
def rows(tmp_path: Path) -> Path:
    path = tmp_path / "rows.jsonl"
    path.write_bytes(EXAMPLE_ROWS.read_bytes())
    return path


def _ids(rows: Path) -> list[str]:
    return [json.loads(line)["instance_id"] for line in rows.read_text().splitlines() if line.strip()]


def _first_row(rows: Path) -> dict:
    return json.loads(rows.read_text().splitlines()[0])


def _convert(rows: Path, out: Path, parser_repo: Path, *extra: str) -> int:
    return adapter.main(["--rows", str(rows), "--output-dir", str(out), "--parser-repo", str(parser_repo), *extra])


def _tree(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_two_conversions_are_byte_identical(tmp_path: Path, rows: Path, parser_repo: Path) -> None:
    assert _convert(rows, tmp_path / "a" / "ds", parser_repo) == 0
    assert _convert(rows, tmp_path / "b" / "ds", parser_repo) == 0
    first, second = _tree(tmp_path / "a" / "ds"), _tree(tmp_path / "b" / "ds")
    assert first == second
    assert len([name for name in first if name.endswith("task.toml")]) == len(_ids(rows))


def test_non_empty_output_is_refused(tmp_path: Path, rows: Path, parser_repo: Path) -> None:
    out = tmp_path / "ds"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    assert _convert(rows, out, parser_repo) == 2


def test_parser_repo_at_another_commit_is_refused(tmp_path: Path, rows: Path) -> None:
    other = tmp_path / "not_a_checkout"
    other.mkdir()
    assert _convert(rows, tmp_path / "ds", other) == 2


@pytest.mark.parametrize(
    "server_update",
    [
        {"is_verifying_golden_patch": True},
        {"enforce_agent_no_network": True},
        {"evaluation_timeout": None},
        {"patch_capture_mode": "staged"},
        {"sandbox_config": {"derive_cpu_env": False}},
        {"sandbox_config": {"legacy_container_setup": True}},
        {"sandbox_config": {"use_maven_mirror": False}},
        {"sandbox_config": {"provider_options": {"network_policy": {"egress": []}}}},
        {"sandbox_config": {"resources": {"cpu": 8, "memory_mib": 30720, "disk_gib": 30, "gpu": 1}}},
        {"sandbox_config": {"resources": {"cpu": 1.5, "memory_mib": 30720, "disk_gib": 30}}},
    ],
)
def test_native_settings_without_a_task_form_are_refused(
    tmp_path: Path, rows: Path, parser_repo: Path, server_update: dict
) -> None:
    document = yaml.safe_load(adapter.SERVER_CONFIG.read_text())
    server = document[adapter.SERVER_INSTANCE]["resources_servers"]["swe_rebench"]
    for key, value in server_update.items():
        if isinstance(value, dict):
            server[key] = {**(server.get(key) or {}), **value}
        else:
            server[key] = value
    config = tmp_path / "native.yaml"
    config.write_text(yaml.safe_dump(document))
    assert _convert(rows, tmp_path / "ds", parser_repo, "--native-config", str(config)) == 2
    assert not (tmp_path / "ds").exists()


def test_pairing_yaml_settings_match_the_default(tmp_path: Path, rows: Path, parser_repo: Path) -> None:
    # configs/swe_rebench.yaml pulls the server settings in through config_paths.
    assert _convert(rows, tmp_path / "pairing", parser_repo, "--native-config", str(PAIRING_CONFIG)) == 0
    assert _convert(rows, tmp_path / "default", parser_repo) == 0
    task = tmp_path / "pairing" / _first_row(rows)["instance_id"]
    config = json.loads(_bundle_files((task / "environment/Dockerfile").read_text())["config.json"])
    assert config["anti_cheat_required"] is True
    assert tomllib.loads((task / "task.toml").read_text())["environment"]["cpus"] == 8
    pairing, default = _tree(tmp_path / "pairing"), _tree(tmp_path / "default")
    assert {k: v for k, v in pairing.items() if k != "README.md"} == {
        k: v for k, v in default.items() if k != "README.md"
    }


def test_row_with_tools_is_refused(tmp_path: Path, rows: Path, parser_repo: Path) -> None:
    row = _first_row(rows)
    row["responses_create_params"]["tools"] = [{"type": "function", "name": "x"}]
    rows.write_text(json.dumps(row) + "\n")
    assert _convert(rows, tmp_path / "ds", parser_repo) == 2


def _bad_row(change: dict) -> dict:
    row = json.loads(EXAMPLE_ROWS.read_text().splitlines()[0])
    return {**row, "instance_id": "bad__row-1", **change}


@pytest.mark.parametrize(
    "bad_row, message",
    [
        (_bad_row({"image_name": "docker.io/x:1\nRUN echo injected"}), "image_name 'docker.io/x:1\\nRUN"),
        (_bad_row({"image_name": ""}), "image_name '' is empty"),
        (_bad_row({"responses_create_params": [1]}), "expected responses_create_params.input to hold one message"),
        (_bad_row({"responses_create_params": {"input": ["just a string"]}}), "expected one user message"),
        (
            _bad_row({"responses_create_params": {"input": [{"role": "user", "content": "x\ud800y"}]}}),
            "the row holds text that is not valid UTF-8",
        ),
        (_bad_row({"instance_id": ".."}), "instance_id '..' is not a usable folder name"),
        (_bad_row({"instance_id": ".hidden"}), "instance_id '.hidden' is not a usable folder name"),
    ],
)
def test_malformed_row_is_refused_before_writing(
    tmp_path: Path, rows: Path, parser_repo: Path, capsys: pytest.CaptureFixture, bad_row: dict, message: str
) -> None:
    # The bad row comes after good ones, so a check made while writing would leave their folders behind.
    with rows.open("a") as handle:
        handle.write(json.dumps(bad_row) + "\n")
    assert _convert(rows, tmp_path / "ds", parser_repo) == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "ds").exists()


def test_repeated_instance_id_without_ids_file_is_refused(
    tmp_path: Path, rows: Path, parser_repo: Path, capsys: pytest.CaptureFixture
) -> None:
    line = rows.read_text().splitlines()[0]
    rows.write_text(f"{line}\n{line}\n")
    assert _convert(rows, tmp_path / "ds", parser_repo) == 2
    assert "appear more than once" in capsys.readouterr().err
    assert not (tmp_path / "ds").exists()


def test_malformed_image_digest_is_refused_before_writing(
    tmp_path: Path, rows: Path, parser_repo: Path, capsys: pytest.CaptureFixture
) -> None:
    images = [json.loads(line)["image_name"] for line in rows.read_text().splitlines()]
    # Only the last row's digest is malformed, so a check made while writing would leave the earlier folders behind.
    digests = {name: "sha256:" + "a" * 64 for name in images}
    digests[images[-1]] = "sha256:" + "A" * 64
    path = tmp_path / "digests.json"
    path.write_text(json.dumps(digests))
    assert _convert(rows, tmp_path / "ds", parser_repo, "--image-digests", str(path)) == 2
    assert "is not sha256:<64 hex>" in capsys.readouterr().err
    assert not (tmp_path / "ds").exists()


def test_standalone_python_download_fits_its_budget() -> None:
    script = (TEMPLATE / "tests/nemo_gym/python.sh").read_text()
    curl = re.search(r"^\s*curl (.*)$", script, re.MULTILINE).group(1).split()
    flags = {flag: int(curl[curl.index(flag) + 1]) for flag in ("--retry", "--retry-delay", "--max-time")}
    curl_worst = (flags["--retry"] + 1) * flags["--max-time"] + flags["--retry"] * flags["--retry-delay"]
    wget_worst = int(re.search(r"^\s*timeout (\d+) wget ", script, re.MULTILINE).group(1))
    for worst in (curl_worst, wget_worst):
        assert worst < adapter.PYTHON_DOWNLOAD_BUDGET_SEC
    assert adapter.PYTHON_DOWNLOAD_BUDGET_SEC < adapter.BUILD_TIMEOUT_SEC
    assert adapter.PYTHON_DOWNLOAD_BUDGET_SEC < adapter.VERIFIER_TIMEOUT_MARGIN_SEC


def test_text_outside_the_basic_multilingual_plane_loads_as_toml(
    tmp_path: Path, rows: Path, parser_repo: Path
) -> None:
    row = _first_row(rows)
    row["language"] = "dart \U0001f600 \\ud83d"
    rows.write_text(json.dumps(row) + "\n")
    assert _convert(rows, tmp_path / "ds", parser_repo) == 0
    toml = tomllib.loads((tmp_path / "ds" / row["instance_id"] / "task.toml").read_text())
    assert toml["metadata"]["language"] == row["language"]


def test_ids_order_native_rows_and_id_map(tmp_path: Path, rows: Path, parser_repo: Path) -> None:
    ids = list(reversed(_ids(rows)))[:3]
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("# subset header\n" + "\n".join(ids) + "\n")
    native, id_map = tmp_path / "native.jsonl", tmp_path / "map.json"
    extra = ["--ids-file", str(ids_file), "--native-rows-out", str(native), "--id-map-out", str(id_map)]
    assert _convert(rows, tmp_path / "ds", parser_repo, *extra) == 0
    raw = {json.loads(line)["instance_id"]: line for line in rows.read_text().splitlines()}
    assert native.read_text().splitlines() == [raw[item] for item in ids]
    assert json.loads(id_map.read_text()) == {str(index): item for index, item in enumerate(ids)}
    assert sorted(path.name for path in (tmp_path / "ds").iterdir() if path.is_dir()) == sorted(ids)


def test_indented_comment_in_ids_file_is_a_comment(tmp_path: Path, rows: Path, parser_repo: Path) -> None:
    first = _ids(rows)[0]
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text(f"  # subset header\n{first}\n")
    assert _convert(rows, tmp_path / "ds", parser_repo, "--ids-file", str(ids_file)) == 0
    assert sorted(path.name for path in (tmp_path / "ds").iterdir() if path.is_dir()) == [first]


def test_missing_or_duplicate_id_is_refused(tmp_path: Path, rows: Path, parser_repo: Path) -> None:
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("no-such-row\n")
    assert _convert(rows, tmp_path / "a", parser_repo, "--ids-file", str(ids_file)) == 2
    first = _ids(rows)[0]
    ids_file.write_text(f"{first}\n{first}\n")
    assert _convert(rows, tmp_path / "b", parser_repo, "--ids-file", str(ids_file)) == 2


def _bundle_files(dockerfile: str) -> dict[str, bytes]:
    encoded = dockerfile.split("printf '%s' '", 1)[1].split("'", 1)[0]
    archive = tarfile.open(fileobj=io.BytesIO(gzip.decompress(base64.b64decode(encoded))))
    return {member.name: archive.extractfile(member).read() for member in archive.getmembers()}


def test_task_contents(tmp_path: Path, rows: Path, parser_repo: Path) -> None:
    digests = tmp_path / "digests.json"
    row = _first_row(rows)
    digest = "sha256:" + "a" * 64
    digests.write_text(json.dumps({json.loads(line)["image_name"]: digest for line in rows.read_text().splitlines()}))
    assert _convert(rows, tmp_path / "ds", parser_repo, "--image-digests", str(digests)) == 0
    task = tmp_path / "ds" / row["instance_id"]

    # The instruction is the row's user message, byte for byte.
    assert (task / "instruction.md").read_bytes() == row["responses_create_params"]["input"][0]["content"].encode()
    # The image is pinned, and the separate verifier runs the plain row image.
    dockerfile = (task / "environment/Dockerfile").read_text()
    assert f"FROM {row['image_name']}@{digest}\n" in dockerfile
    toml = (task / "task.toml").read_text()
    assert 'environment_mode = "separate"' in toml
    assert f'docker_image = "{row["image_name"]}@{digest}"' in toml
    assert '"GRADLE_USER_HOME" = "/root/.gradle"' in toml or 'GRADLE_USER_HOME = "/root/.gradle"' in toml
    # The server's sandbox resources on both sandboxes, and its evaluation_timeout plus the download margin.
    parsed = tomllib.loads(toml)
    for table in (parsed["environment"], parsed["verifier"]["environment"]):
        assert (table["cpus"], table["memory_mb"], table["storage_mb"]) == (8, 30720, 30720)
    assert parsed["verifier"]["timeout_sec"] == 3600 + 900
    # The base commit reaches the capture hook on its command line only, never the agent's sandbox.
    assert row["base_commit"] in toml
    bundle = _bundle_files(dockerfile)
    assert all(row["base_commit"].encode() not in data for data in bundle.values())
    # The seed follows the server's anti_cheat_required and patch_capture_mode.
    config = json.loads(bundle["config.json"])
    assert config["anti_cheat_required"] is True and config["apply_anti_cheating"] is True
    assert config["patch_capture_mode"] == "worktree"
    # Hidden data stays out of everything the agent sees.
    visible = (task / "instruction.md").read_bytes() + dockerfile.encode() + b"".join(bundle.values())
    hidden = [row["patch"], row["test_patch"], *row["FAIL_TO_PASS"][:3], json.dumps(row["install_config"])]
    assert all(value.encode() not in visible for value in hidden if len(value) >= 20)
    # The verifier and the oracle get what they need.
    task_json = json.loads((task / "tests/nemo_gym/task.json").read_text())
    assert task_json["row"]["FAIL_TO_PASS"] == row["FAIL_TO_PASS"]
    assert task_json["row"]["install_config"] == row["install_config"]
    assert task_json["row"]["test_patch"] == row["test_patch"]
    assert task_json["overwrite_maven_settings"] is True
    assert (task / "solution/golden.patch").read_text() == row["patch"]


def test_vendored_modules_are_the_servers_byte_for_byte(tmp_path: Path, rows: Path, parser_repo: Path) -> None:
    assert _convert(rows, tmp_path / "ds", parser_repo) == 0
    task = next(path for path in (tmp_path / "ds").iterdir() if path.is_dir())
    bundle = _bundle_files((task / "environment/Dockerfile").read_text())
    copies = {
        adapter.VERIFICATION_MODULE: (task / "tests/nemo_gym" / adapter.VERIFICATION_MODULE).read_bytes(),
        adapter.LOG_PARSERS_MODULE: (task / "tests/nemo_gym" / adapter.LOG_PARSERS_MODULE).read_bytes(),
        adapter.PATCH_CAPTURE_MODULE: bundle[adapter.PATCH_CAPTURE_MODULE],
        adapter.ANTI_CHEAT_MODULE: bundle[adapter.ANTI_CHEAT_MODULE],
    }
    for relative, copy in copies.items():
        header, body = copy.split(b"\n", 1)
        assert header.startswith(b"# Vendored unchanged from " + relative.encode())
        assert body == (REPO_ROOT / relative).read_bytes()
    assert bundle[adapter.ANTI_CHEAT_SCRIPT] == (REPO_ROOT / adapter.ANTI_CHEAT_SCRIPT).read_bytes()
    for relative in adapter.MIRROR_FILES:
        assert (task / "tests/nemo_gym" / relative).read_bytes() == (REPO_ROOT / relative).read_bytes()
    # The upstream tree sits where the server's log_parsers looks for its clone, so resolve_parser never clones.
    log_parsers = adapter._load_repo_module(adapter.LOG_PARSERS_MODULE)
    upstream = task / "tests/nemo_gym/parser_cache" / log_parsers._repo_dir_name()
    for relative in adapter.UPSTREAM_FILES:
        copy = (upstream / relative).read_bytes()
        if not relative.endswith(".py"):
            assert copy == (parser_repo / relative).read_bytes()
            continue
        header, body = copy.split(b"\n", 1)
        assert body == (parser_repo / relative).read_bytes()
        assert f" at {log_parsers.UPSTREAM_COMMIT};".encode() in header and header.endswith(b"; license MIT")


def test_solve_script_and_capture_hook(tmp_path: Path, rows: Path, parser_repo: Path) -> None:
    assert _convert(rows, tmp_path / "ds", parser_repo) == 0
    row = _first_row(rows)
    task = tmp_path / "ds" / row["instance_id"]
    assert (task / "solution" / "solve.sh").read_text() == (
        "#!/bin/bash\n"
        "# Applies the row's golden patch with the flags the server's eval script applies a patch with\n"
        "# (verification.build_eval_script), for `gym dataset validate`.\n"
        "cd /rohd || exit 1\n"
        'git apply --reject --recount --ignore-space-change --whitespace=nowarn "$(dirname "$0")/golden.patch"'
        " || true\n"
    )
    collect = tomllib.loads((task / "task.toml").read_text())["verifier"]["collect"]
    assert collect == [
        {
            "service": "main",
            "timeout_sec": adapter.CAPTURE_TIMEOUT_SEC,
            "command": 'rm -rf /logs/artifacts/nemo_gym; "$(cat /var/lib/nemo_gym/python_bin)" -I'
            f" /var/lib/nemo_gym/capture.py {shlex.quote(row['base_commit'])};"
            " test -f /logs/artifacts/nemo_gym/capture.json || { mkdir -p /logs/artifacts/nemo_gym && echo"
            ' \'{"status": "runner_failed", "error": "the capture hook could not run capture.py"}\''
            " > /logs/artifacts/nemo_gym/capture.json; }",
        }
    ]


# -- the code a converted task runs: grade.py, capture.py, seed.py, setup.sh, local_sandbox.py ---------------------


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


def _state_dir(tmp_path: Path, rows: Path, parser_repo: Path) -> Path:
    """The seed bundle of the first example row, unpacked into tmp_path (``/var/lib/nemo_gym`` in a task)."""
    assert _convert(rows, tmp_path / "ds", parser_repo) == 0
    dockerfile = (tmp_path / "ds" / _first_row(rows)["instance_id"] / "environment" / "Dockerfile").read_text()
    state = tmp_path / "state"
    for name, data in _bundle_files(dockerfile).items():
        (state / name).parent.mkdir(parents=True, exist_ok=True)
        (state / name).write_bytes(data)
    return state


class _Grader:
    """``grade.py`` of a converted task, with its outputs under tmp_path and the root-only steps stubbed.

    The server's ``log_parsers.py`` runs as vendored, so ``resolve_parser`` loads the stand-in upstream tree from the
    task's ``parser_cache``; only ``verification_files`` and ``run_verification`` are replaced.
    """

    def __init__(self, tmp_path: Path, rows: Path, parser_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _convert(rows, tmp_path / "ds", parser_repo) == 0
        self.row = _first_row(rows)
        self.here = tmp_path / "ds" / self.row["instance_id"] / "tests" / "nemo_gym"
        self.module = _load(f"swe_rebench_grade_{tmp_path.name}", self.here / "grade.py")
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

        monkeypatch.setattr(self.module, "_privilege_dropper", lambda: drop)
        real_load = self.module._load
        here = self.here

        def load(name: str, path: Path):
            module = real_load(name, path)
            if not hasattr(module, "run_verification"):
                return module

            async def run_verification(*, sandbox, inputs, parser, timeout_s, log_dir):
                parser_file = Path(parser.__code__.co_filename)
                self.calls.append(
                    (
                        "run",
                        sandbox.preexec_fn is drop,
                        timeout_s,
                        inputs.instance_id,
                        parser_file.is_relative_to(here),
                    )
                )
                return self.result(module)

            def verification_files(inputs, *, overwrite_maven_settings):
                self.calls.append(("files", overwrite_maven_settings))
                # The real files go to /tmp and /root in the verifier sandbox; here they go under tmp_path.
                return {str(tmp_path / "eval" / "nemo_gym_eval.sh"): "true\n"}

            monkeypatch.setattr(module, "verification_files", verification_files)
            monkeypatch.setattr(module, "run_verification", run_verification)
            return module

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
def grader(tmp_path: Path, rows: Path, parser_repo: Path, monkeypatch: pytest.MonkeyPatch) -> _Grader:
    return _Grader(tmp_path, rows, parser_repo, monkeypatch)


@pytest.mark.parametrize(
    "status, unreadable",
    [("missing", 0.0), ("non-JSON", 0.0), ("git_error", 1.0), ("runner_failed", 1.0), ("infra_error", 0.0)],
)
def test_grader_scores_a_failed_capture_zero_without_running_tests(
    grader: _Grader, status: str, unreadable: float
) -> None:
    if status == "non-JSON":
        (grader.artifacts / "capture.json").write_text("{not json")
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
    # The parser comes from the task's vendored parser_cache; the server's Maven setting reaches verification_files;
    # the reward path is protected before the agent's code runs, and that code runs with the privileges dropped,
    # bounded by the server's evaluation_timeout.
    instance_id = grader.row["instance_id"]
    assert grader.calls == [("files", True), "ownership", "protect", ("run", True, 3600, instance_id, True)]


def test_grader_masks_an_unknown_parser(grader: _Grader) -> None:
    task_json = grader.here / "task.json"
    task = json.loads(task_json.read_text())
    task["row"]["install_config"]["log_parser"] = "parse_log_no_such_parser"
    task_json.write_text(json.dumps(task))
    grader.capture("ok")
    code, reward, grading = grader.run()
    assert code == grader.module.NO_VERDICT == 75
    assert reward is None and grading["masked"] is True
    assert grading["reason"].startswith("log parser 'parse_log_no_such_parser' could not be resolved")
    assert grader.calls == []


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
    assert code == 75
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
    assert grader.calls == [("files", True), "ownership"]


def test_grader_masks_when_it_raises(grader: _Grader) -> None:
    grader.capture("ok")
    grader.result = lambda v: 1 / 0
    code, reward, grading = grader.run()
    assert code == 75 and reward is None and grading["reason"] == "the grader raised"


@needs_git
def test_capture_hook_reads_the_patch_and_splits_errors(
    tmp_path: Path, rows: Path, parser_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state_dir(tmp_path, rows, parser_repo)
    repo = _git_repo(tmp_path)
    for key, value in _git_env(tmp_path).items():
        monkeypatch.setenv(key, value)
    base = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (state / "config.json").write_text(json.dumps({"workdir": str(repo), "patch_capture_mode": "worktree"}))
    (state / "pristine_untracked.txt").write_text("")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["capture.py", base])
    capture = _load(f"swe_rebench_capture_{tmp_path.name}", state / "capture.py")
    out = tmp_path / "artifacts"
    monkeypatch.setattr(capture, "OUT_DIR", out)

    def status() -> str:
        return json.loads((out / "capture.json").read_text())["status"]

    (repo / "m.py").write_text("x = 2\n")
    assert capture.main() == 0
    assert status() == "ok" and "+x = 2" in (out / "model_patch.diff").read_text()
    assert json.loads((out / "capture.json").read_text())["base_commit"] == base

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
    (state / "pristine_untracked.txt").unlink()
    assert capture.main() == 0
    assert status() == "git_error"
    assert json.loads((out / "capture.json").read_text())["error"].startswith("seed record missing")


def _local_sandbox(tmp_path: Path):
    # A copy, so importing it leaves no bytecode cache in the template.
    shutil.copy(TEMPLATE / "tests/nemo_gym/local_sandbox.py", tmp_path / "local_sandbox.py")
    return _load(f"swe_rebench_local_sandbox_{tmp_path.name}", tmp_path / "local_sandbox.py")


def test_local_sandbox_exec_returns_output_and_times_out(tmp_path: Path) -> None:
    local_sandbox = _local_sandbox(tmp_path)
    result = asyncio.run(local_sandbox.LocalSandbox().exec("echo out; echo err >&2; exit 3", timeout_s=30))
    assert (result.stdout, result.stderr, result.return_code, result.error_type) == ("out\n", "err\n", 3, None)
    start = time.monotonic()
    result = asyncio.run(local_sandbox.LocalSandbox().exec("echo started; sleep 30", timeout_s=1))
    assert (result.stdout, result.return_code, result.error_type) == ("started\n", None, "timeout")
    assert time.monotonic() - start < 10


def test_local_sandbox_exec_honours_cwd_and_upload_copies(tmp_path: Path) -> None:
    local_sandbox = _local_sandbox(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    result = asyncio.run(local_sandbox.LocalSandbox().exec("pwd", timeout_s=30, cwd=str(workdir)))
    assert result.stdout == f"{workdir}\n"
    (tmp_path / "source.sh").write_text("echo uploaded\n")
    asyncio.run(local_sandbox.LocalSandbox().upload(tmp_path / "source.sh", str(workdir / "copy.sh")))
    assert (workdir / "copy.sh").read_text() == "echo uploaded\n"


def _run_setup(tmp_path: Path, state: Path, workdir: Path, **config: object) -> subprocess.CompletedProcess:
    """Run the bundle's setup.sh (python.sh, then seed.py) with its state folder under tmp_path."""
    settings = json.loads((state / "config.json").read_text())
    (state / "config.json").write_text(json.dumps({**settings, "workdir": str(workdir), **config}))
    script = (state / "setup.sh").read_text()
    assert script.count(f"\nS={adapter.STATE_DIR}\n") == 1
    (state / "setup.sh").write_text(script.replace(f"\nS={adapter.STATE_DIR}\n", f"\nS={state}\n"))
    # This test's Python comes first on PATH, so python.sh finds it and never downloads one.
    env = {**_git_env(tmp_path), "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    return subprocess.run(["bash", str(state / "setup.sh")], env=env, capture_output=True, text=True, timeout=300)


@needs_git
def test_setup_script_seeds_a_repository(tmp_path: Path, rows: Path, parser_repo: Path) -> None:
    state = _state_dir(tmp_path, rows, parser_repo)
    repo = _git_repo(tmp_path)
    (repo / "notes.txt").write_text("untracked before the agent\n")
    process = _run_setup(tmp_path, state, repo)
    assert process.returncode == 0, process.stderr
    assert Path((state / "python_bin").read_text().strip()).is_file()
    assert (state / "pristine_untracked.txt").read_text() == "notes.txt\n"
    # The server's anti-cheat scrub ran (required) and removed its own copy from the workdir.
    branch = subprocess.run(["git", "-C", str(repo), "symbolic-ref", "HEAD"], capture_output=True, text=True).stdout
    assert branch == "refs/heads/_nel_work\n" and not (repo / "anti_cheat_setup.sh").exists()


@needs_git
def test_seed_fails_when_a_required_scrub_fails(tmp_path: Path, rows: Path, parser_repo: Path) -> None:
    state = _state_dir(tmp_path, rows, parser_repo)
    (state / adapter.ANTI_CHEAT_SCRIPT).write_text("echo scrub refused >&2\nexit 1\n")
    process = _run_setup(tmp_path, state, _git_repo(tmp_path))
    assert process.returncode == 1
    assert "anti-cheat setup failed" in process.stderr
    assert not (state / "pristine_untracked.txt").exists()


@needs_git
def test_seed_fails_when_untracked_files_cannot_be_listed(tmp_path: Path, rows: Path, parser_repo: Path) -> None:
    state = _state_dir(tmp_path, rows, parser_repo)
    plain = tmp_path / "plain"
    plain.mkdir()
    process = _run_setup(tmp_path, state, plain, apply_anti_cheating=False)
    assert process.returncode == 1
    assert "could not list the pristine untracked files" in process.stderr
    assert not (state / "pristine_untracked.txt").exists()
