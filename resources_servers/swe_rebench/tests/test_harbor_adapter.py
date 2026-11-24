# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the SWE-rebench-V2 Harbor adapter (``resources_servers/swe_rebench/harbor/adapter.py``).

They run without the network: the upstream parser tree is a stand-in folder, and the adapter's check that it sits at
the server's pinned commit is pointed at that folder. Running a converted task needs a sandbox; that is the oracle
run (``gym dataset validate``), not these tests. The grader, the capture hook, the seed and ``local_sandbox.py`` run
here from copies under ``tmp_path``, against temporary git repositories, with the steps that need root (ownership,
the privilege drop, the uid switch) stubbed out.
"""

import ast
import asyncio
import base64
import errno
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
import tempfile
import time
import tomllib
from collections.abc import Iterator
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


@pytest.mark.parametrize(
    "control", [{"temperature": 0}, {"top_p": 0.4}, {"parallel_tool_calls": False}, {"max_output_tokens": 64}]
)
def test_request_control_is_refused_before_writing(
    tmp_path: Path, rows: Path, parser_repo: Path, capsys: pytest.CaptureFixture, control: dict
) -> None:
    # The task cannot carry a sampling or request control (the agent's run configuration sets the request), so a
    # row that sets one, even to a zero or false value, is refused instead of silently dropped.
    row = _bad_row({})
    row["responses_create_params"] = {**row["responses_create_params"], **control}
    with rows.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    assert _convert(rows, tmp_path / "ds", parser_repo) == 2
    assert (
        f"bad__row-1: responses_create_params.{next(iter(control))} has no task-folder form" in capsys.readouterr().err
    )
    assert not (tmp_path / "ds").exists()


def test_null_request_controls_convert_unchanged(tmp_path: Path, rows: Path, parser_repo: Path) -> None:
    assert _convert(rows, tmp_path / "plain", parser_repo) == 0
    changed = []
    for line in rows.read_text().splitlines():
        row = json.loads(line)
        row["responses_create_params"].update(temperature=None, top_p=None, tools=[])
        changed.append(json.dumps(row) + "\n")
    rows.write_text("".join(changed))
    assert _convert(rows, tmp_path / "nulls", parser_repo) == 0
    # Every task file is the same; only the dataset README differs, since it records the rows file's digest.
    nulls, plain = _tree(tmp_path / "nulls"), _tree(tmp_path / "plain")
    assert nulls.pop("README.md") != plain.pop("README.md")
    assert nulls == plain


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


def test_native_env_name_outside_the_basic_multilingual_plane_loads_as_toml(
    tmp_path: Path, rows: Path, parser_repo: Path
) -> None:
    env = {'LABEL \U0001f600 "q" \\ud83d': "Hello \U0001f600", "CONTROL": "tab\there\x7f"}
    document = yaml.safe_load(adapter.SERVER_CONFIG.read_text())
    server = document[adapter.SERVER_INSTANCE]["resources_servers"]["swe_rebench"]
    server["sandbox_config"] = {**(server.get("sandbox_config") or {}), "env": env}
    config = tmp_path / "native.yaml"
    config.write_text(yaml.safe_dump(document))
    assert _convert(rows, tmp_path / "ds", parser_repo, "--native-config", str(config)) == 0
    toml = tomllib.loads((tmp_path / "ds" / _first_row(rows)["instance_id"] / "task.toml").read_text())
    assert {key: toml["environment"]["env"][key] for key in env} == env


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
    task's ``parser_cache``; only ``verification_files`` and ``run_verification`` are replaced, and the repository
    directory is a folder under tmp_path. The audit hook cannot be removed once installed, so it is never armed here
    (``_arm_after_tests`` is stubbed); the tests that arm it run ``grade.py`` in a subprocess (``_drive``).
    """

    def __init__(self, tmp_path: Path, rows: Path, parser_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _convert(rows, tmp_path / "ds", parser_repo) == 0
        self.row = _first_row(rows)
        self.here = tmp_path / "ds" / self.row["instance_id"] / "tests" / "nemo_gym"
        self.module = _load(f"swe_rebench_grade_{tmp_path.name}", self.here / "grade.py")
        self.artifacts, self.logs = tmp_path / "artifacts", tmp_path / "verifier"
        self.artifacts.mkdir()
        self.calls: list = []
        self.repo_directory = tmp_path
        self.result = None
        monkeypatch.setattr(sys, "path", list(sys.path))
        monkeypatch.setattr(self.module, "ARTIFACTS", self.artifacts)
        monkeypatch.setattr(self.module, "LOGS", self.logs)
        monkeypatch.setattr(self.module.os, "seteuid", lambda uid: None)
        monkeypatch.setattr(self.module, "_give_root_ownership", lambda: self.calls.append("ownership") or 0)
        self.protect_reward_path = self.module._protect_reward_path
        monkeypatch.setattr(self.module, "_protect_reward_path", lambda: self.calls.append("protect"))
        record = {"mappings": set(), "kept": {}, "ancestors": [], "unreadable_ancestors": [], "seconds": 0.0}
        monkeypatch.setattr(self.module, "_prepare_after_tests", lambda: self.calls.append("prepare") or record)
        monkeypatch.setattr(self.module, "_arm_after_tests", lambda: self.calls.append("arm"))
        self.check_mappings = self.module._check_mappings
        monkeypatch.setattr(self.module, "_check_mappings", lambda record: self.calls.append("check"))

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
            # The workdir check runs on this machine: point the repository directory at a folder that exists.
            monkeypatch.setattr(module, "repo_directory", lambda repo: str(self.repo_directory))
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
    assert [call[0] for call in grader.calls if isinstance(call, tuple)] == ["files", "run"]
    assert reward["model_patch_bytes"] == float(len("diff --git a/m.py b/m.py\n+x = '\ufffd'\n".encode()))


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
    # the reward path and the grader's mapped files are protected and the audit hook armed before the agent's code
    # runs, that code runs with the privileges dropped, bounded by the server's evaluation_timeout, and the mappings
    # are checked before the verdict.
    instance_id = grader.row["instance_id"]
    assert reward["no_verdict_after_tests"] == 0.0
    assert grader.calls == [
        ("files", True),
        "ownership",
        "protect",
        "prepare",
        "arm",
        ("run", True, 3600, instance_id, True),
        "check",
    ]


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


@pytest.mark.parametrize(
    "error, test_patch_failed",
    [
        ("repo directory /x not present in the image", False),
        ("held-out test patch did not apply", True),
        ("Test execution ended before output was complete (return code -9)", False),
        ("log parser 'parse_log_pytest' raised: simulated", False),
    ],
)
def test_grader_scores_a_run_without_a_verdict_zero(grader: _Grader, error: str, test_patch_failed: bool) -> None:
    # The server masks these (exit 97, the test-patch marker, a killed test run, a parser that raised). The install
    # block already runs the agent's code before TEST_OUTPUT_BEGIN, so each could be the agent's doing: a completed 0.
    grader.capture("ok")
    grader.result = lambda v: v.VerificationResult(
        completed=False,
        resolved=False,
        patch_applied=True,
        test_results=None,
        test_output=v.TEST_PATCH_FAILED if test_patch_failed else "",
        error=error,
        test_patch_failed=test_patch_failed,
    )
    code, reward, grading = grader.run()
    assert code == 0 and grading["masked"] is False and grading["error"] == error
    assert reward["reward"] == 0.0 and reward["resolved"] == 0.0 and reward["no_verdict_after_tests"] == 1.0


def test_grader_masks_a_missing_repository_before_any_test(grader: _Grader, tmp_path: Path) -> None:
    grader.capture("ok")
    grader.repo_directory = tmp_path / "no_such_repository"
    code, reward, grading = grader.run()
    assert code == 75 and reward is None and grading["reason"].startswith("repo directory ")
    assert grader.calls == []


def test_grader_masks_when_the_reward_cannot_be_protected(grader: _Grader, monkeypatch: pytest.MonkeyPatch) -> None:
    grader.capture("ok")

    def fail() -> None:
        raise PermissionError("simulated")

    monkeypatch.setattr(grader.module, "_protect_reward_path", fail)
    code, reward, grading = grader.run()
    assert code == 75 and reward is None
    assert grading["reason"].startswith("could not protect the reward folder")
    assert grader.calls == [("files", True), "ownership"]


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
    assert grader.calls == [("files", True), "ownership", "protect"]


def test_grader_masks_when_it_raises(grader: _Grader) -> None:
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


def test_reward_path_protection_drops_old_rewards_and_writes_no_exit_record(
    grader: _Grader, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The harbor server takes test.sh's exit status from its run exec and never reads an exit record, so the grader
    # leaves none (a pre-created one was collected as a stale log saying exit 1).
    module = grader.module
    grader.logs.mkdir()
    (grader.logs / "reward.json").write_text('{"reward": 1}')
    (grader.logs / "reward.txt").write_text("1")
    monkeypatch.setattr(module, "REWARD_OWNER_UID", os.getuid())
    monkeypatch.setattr(module, "PROTECTED_DIRS", (grader.logs,))
    grader.protect_reward_path()
    assert sorted(path.name for path in grader.logs.iterdir()) == []
    assert not hasattr(module, "EXIT_RECORD_NAME")


def _stub_ancestor_maps(module, monkeypatch: pytest.MonkeyPatch, maps: dict[str, object]) -> None:
    """Ancestors 101, 102 and 1, nearest first; ``maps`` gives each one's mapped paths or the error reading them."""

    def mapped_paths(path: str = module.PROC_MAPS) -> list[str]:
        if path == module.PROC_MAPS:
            return []
        found = maps[path]
        if isinstance(found, BaseException):
            raise found
        return found

    monkeypatch.setattr(module, "_mapped_paths", mapped_paths)
    monkeypatch.setattr(module, "_ancestors", lambda: [(101, "gone"), (102, "sh"), (1, "init")])
    monkeypatch.setattr(module, "REWARD_OWNER_UID", os.getuid())


@pytest.mark.parametrize("refusal", [errno.EACCES, errno.EPERM])
def test_mapping_protection_covers_ancestors_and_skips_one_it_may_not_inspect(
    grader: _Grader, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, refusal: int
) -> None:
    # The files an ancestor has mapped are protected like the grader's own; a gone ancestor is skipped; an ancestor
    # whose maps the grader may not read (on a real sandbox, pid 1: the provider's service or the sandbox's init) is
    # skipped and listed, and the readable relay below it is still protected and checked after the tests.
    module = grader.module
    mapped = tmp_path / "relay.bin"
    mapped.write_bytes(b"A" * 64)
    maps: dict[str, object] = {
        "/proc/101/maps": FileNotFoundError(),
        "/proc/102/maps": [str(mapped)],
        "/proc/1/maps": OSError(refusal, os.strerror(refusal), "/proc/1/maps"),
    }
    _stub_ancestor_maps(module, monkeypatch, maps)
    record = module._protect_mappings()
    assert list(record["kept"]) == [str(mapped)] and record["ancestors"] == ["gone", "sh", "init"]
    assert record["unreadable_ancestors"] == [{"pid": 1, "name": "init"}]
    assert mapped.stat().st_mode & 0o222 == 0
    grader.check_mappings(record)  # the skipped ancestor recorded nothing, so nothing of it is reported changed
    os.chmod(mapped, 0o644)
    with pytest.raises(module._FailClosed, match="relay.bin"):
        grader.check_mappings(record)
    # Any other error reading a readable ancestor's maps still raises before the tests, which grade() masks
    # (test_grader_masks_when_its_mapped_files_cannot_be_protected).
    maps["/proc/102/maps"] = OSError(errno.EIO, "Input/output error")
    with pytest.raises(OSError, match="Input/output error"):
        module._protect_mappings()


def test_ancestor_walk_ends_at_a_stat_it_may_not_read(grader: _Grader, monkeypatch: pytest.MonkeyPatch) -> None:
    # A parent whose /proc/<pid>/stat the grader may not read is listed without a name, and the walk ends there (its
    # parent is unknown); its maps are tried like any other ancestor's.
    module = grader.module
    stats = {"/proc/102/stat": b"102 (s) h) S 1 102 102 0\n", "/proc/1/stat": PermissionError(errno.EACCES, "denied")}
    real_open = open

    def fake_open(path, *args, **kwargs):
        if str(path) not in stats:
            return real_open(path, *args, **kwargs)
        found = stats[str(path)]
        if isinstance(found, BaseException):
            raise found
        return io.BytesIO(found)

    monkeypatch.setattr(module.os, "getppid", lambda: 102)
    monkeypatch.setattr(module, "open", fake_open, raising=False)
    assert module._ancestors() == [(102, "s) h"), (1, None)]

    def mapped_paths(path: str = module.PROC_MAPS) -> list[str]:
        if path == "/proc/1/maps":
            raise PermissionError(errno.EACCES, "denied", path)
        return []

    monkeypatch.setattr(module, "_mapped_paths", mapped_paths)
    record = module._protect_mappings()
    assert record["ancestors"] == ["s) h", None] and record["unreadable_ancestors"] == [{"pid": 1, "name": None}]


def test_grader_does_not_mask_for_an_ancestor_it_may_not_inspect(
    grader: _Grader, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # On a real sandbox pid 1's maps are refused to the grader: the run is graded, and grading.json lists pid 1.
    module = grader.module
    mapped = tmp_path / "relay.bin"
    mapped.write_bytes(b"A" * 64)
    maps: dict[str, object] = {
        "/proc/101/maps": FileNotFoundError(),
        "/proc/102/maps": [str(mapped)],
        "/proc/1/maps": PermissionError(errno.EACCES, "Permission denied", "/proc/1/maps"),
    }
    _stub_ancestor_maps(module, monkeypatch, maps)
    monkeypatch.setattr(
        module, "_prepare_after_tests", lambda: grader.calls.append("prepare") or module._protect_mappings()
    )
    grader.capture("ok")
    grader.result = lambda v: v.VerificationResult(
        completed=True,
        resolved=True,
        patch_applied=True,
        test_results={"tests_observed": 2},
        test_output=v.TEST_OUTPUT_BEGIN + "\n",
    )
    code, reward, grading = grader.run()
    assert code == 0 and grading["masked"] is False and reward["reward"] == 1.0
    protected = grading["protected_mappings"]
    assert protected["files"] == 1 and protected["ancestors"] == ["gone", "sh", "init"]
    assert protected["unreadable_ancestors"] == [{"pid": 1, "name": "init"}]


# Loads grade.py and runs _protect_mappings over this process's real ancestors, reading each one's real stat and
# maps, but changes no file (every mapped path is dropped after it is read). Prints what the walk found.
REAL_WALK = r"""
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("grade_real_walk", sys.argv[1])
grade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grade)
read_mappings = grade._mapped_paths


def mapped_paths(*maps):
    read_mappings(*maps)
    return []


grade._mapped_paths = mapped_paths
record = grade._protect_mappings()
walk = [[pid, name] for pid, name in grade._ancestors()]
print(json.dumps({"file": grade.__file__, "walk": walk, "unreadable": record["unreadable_ancestors"]}))
"""


def test_mapping_protection_walks_the_real_process_tree() -> None:
    # No /proc stub: the real ancestors of a child process, up to pid 1, whatever this machine lets it read (pid 1's
    # maps are refused to a user that does not own it, and to root in a user namespace). The walk must finish and
    # return a record; an ancestor it may not inspect is listed, not raised.
    grade_path = TEMPLATE / "tests/nemo_gym/grade.py"
    done = subprocess.run(
        [sys.executable, "-I", "-c", REAL_WALK, str(grade_path)], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    found = json.loads(done.stdout)
    assert found["file"] == str(grade_path)
    pids = [pid for pid, _ in found["walk"]]
    assert pids[0] == os.getpid() and (pids[-1] == 1 or found["walk"][-1][1] is None)
    assert {entry["pid"] for entry in found["unreadable"]} <= set(pids)


# The functions and constants this grade.py copies from Scale-SWE's (only the docstring and grade() differ).
SHARED_GRADER_NAMES = (
    "HERE ARTIFACTS LOGS NO_VERDICT FAILED_CLOSED REWARD_OWNER_UID PROTECTED_DIRS KEPT_CAPABILITIES"
    " _PR_CAPBSET_DROP _PR_SET_NO_NEW_PRIVS _PR_CAP_AMBIENT _PR_CAP_AMBIENT_CLEAR_ALL _LINUX_CAPABILITY_VERSION_3"
    " CAPTURE_MAX_BYTES PATCH_MAX_BYTES CAPTURE_TEXT_MAX_CHARS CAPTURE_WARNINGS_MAX PROC_MAPS WARM_IMPORTS"
    " REFUSED_AFTER_TESTS _AFTER_TESTS _FailClosed _Unreadable _CapHeader _CapData _privilege_dropper"
    " _give_root_ownership _protect_reward_path _write_output _load _no_verdict _verdict _capture_failed _fail_closed"
    " _read_agent_file _capture_summary _read_capture _inline_to_thread _warm_event_loop _read_mappings _mapped_paths"
    " _ancestors"
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


def _top_level_sources(path: Path) -> dict[str, str]:
    text = path.read_text()
    sources = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            sources[node.name] = ast.get_source_segment(text, node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            sources[node.targets[0].id] = ast.get_source_segment(text, node)
    return sources


def test_grader_shares_the_scale_swe_protection_code() -> None:
    ours = _top_level_sources(TEMPLATE / "tests/nemo_gym/grade.py")
    theirs = _top_level_sources(REPO_ROOT / "resources_servers/scale_swe/harbor/task-template/tests/nemo_gym/grade.py")
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
grade._mapped_paths = lambda *maps: [path for path in all_mapped_paths(*maps) if path.startswith(scratch + os.sep)]
# The grader's ancestors here are the ones the test started, below pytest (whose own are not the test's to protect).
all_ancestors = grade._ancestors
below = int(os.environ["NG_ANCESTORS_BELOW"])


def ancestors():
    found = all_ancestors()
    return found[: [pid for pid, _ in found].index(below)]


grade._ancestors = ancestors
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


# A stand-in for the processes above the grader that outlive the tests (the provider's daemon): it maps a file, runs
# its command (the harbor server's shell, which runs timeout, which runs the grader) and reports the mapped bytes.
ANCESTOR = r"""
import mmap
import subprocess
import sys

with open(sys.argv[1], "rb") as handle:
    mapping = mmap.mmap(handle.fileno(), 0, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ)
code = subprocess.run(sys.argv[2:]).returncode
print("ANCESTOR_PREFIX", repr(bytes(mapping[:8])), flush=True)
sys.exit(code)
"""
# The shell around test.sh: the harbor server's (only exit after timeout) and an older one's (more after it).
RELAY_SHELLS = {
    "harbor_server": 'timeout -s KILL 120 "$@"; exit $?',
    "older_harbor_server": 'start=$(date +%s); timeout -s KILL 120 "$@"; rc=$?; echo "$rc $start" >/dev/null; exit $rc',
}

BEGIN, END = "___NEMO_GYM_SWE_REBENCH_TEST_BEGIN___", "___NEMO_GYM_SWE_REBENCH_TEST_END___"
# Stand-ins for two upstream parser kinds: a line-oriented one and one that reads a JUnit XML report (pyexpat).
STAND_IN_PARSERS = """
import re
import xml.etree.ElementTree as ET


def parse_log_stand_in(log):
    return {}


def parse_log_lines_stand_in(log):
    return {name: status for status, name in re.findall(r"^(PASSED|FAILED) (\\S+)", log, re.MULTILINE)}


def parse_log_junit_stand_in(log):
    root = ET.fromstring(log[log.index("<") : log.rindex(">") + 1])
    failed = {case.get("name") for case in root.iter("testcase") if case.find("failure") is not None}
    return {case.get("name"): "FAILED" if case.get("name") in failed else "PASSED" for case in root.iter("testcase")}
"""
OUTPUTS = {
    ("lines", True): f"echo {BEGIN}\necho 'PASSED t1'\necho 'PASSED t2'\n",
    ("lines", False): f"echo {BEGIN}\necho 'FAILED t1'\necho 'PASSED t2'\n",
    ("junit", True): f'echo {BEGIN}\necho \'<testsuite><testcase name="t1"/><testcase name="t2"/></testsuite>\'\n',
    ("junit", False): (
        f"echo {BEGIN}\n"
        'echo \'<testsuite><testcase name="t1"><failure/></testcase><testcase name="t2"/></testsuite>\'\n'
    ),
}
PASSING, FAILING = OUTPUTS[("lines", True)], OUTPUTS[("lines", False)]


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


@pytest.fixture
def scratch(tmp_path: Path) -> Iterator[Path]:
    """The folder ``_drive`` gives the grader as its verifier, artifacts, workdir and library tree.

    As root the grader really drops to the reward uid, which must traverse every ancestor of these folders, and
    pytest's own temporary folders are private to root. A root run therefore uses a folder of its own directly under
    the system temporary folder, open for traversal only, and removes it afterwards; other runs use ``tmp_path``.
    """
    if os.geteuid() != 0:
        yield tmp_path / "scratch"
        return
    root = Path(tempfile.mkdtemp(prefix="ng-grade-"))
    try:
        root.chmod(0o755)
        yield root / "scratch"
    finally:
        shutil.rmtree(root)


def _drive(
    tmp_path: Path,
    scratch: Path,
    rows: Path,
    parser_repo: Path,
    eval_script: str,
    scenario: str = "plain",
    *,
    parser: str = "lines",
    python: str = sys.executable,
    ancestor: Path | None = None,
    relay_shell: str = RELAY_SHELLS["harbor_server"],
    **env,
) -> tuple[dict, subprocess.CompletedProcess]:
    """Run grade.py's main() of a converted task in a subprocess (see ``DRIVER``) and return its outcome.

    With ``ancestor``, the grader runs under the stand-in ``ANCESTOR`` mapping that file, then ``relay_shell``, then
    ``timeout``, as under the harbor server.
    """
    if not (tmp_path / "ds").exists():
        upstream = parser_repo / "lib/agent/log_parsers.py"
        stand_in = upstream.read_text().replace("def parse_log_stand_in(log):\n    return {}\n", STAND_IN_PARSERS)
        names = ", ".join(f"'parse_log_{kind}_stand_in': parse_log_{kind}_stand_in" for kind in ("lines", "junit"))
        upstream.write_text(stand_in + f"NAME_TO_PARSER.update({{{names}}})\n")
        assert _convert(rows, tmp_path / "ds", parser_repo) == 0
    here = tmp_path / "ds" / _first_row(rows)["instance_id"] / "tests" / "nemo_gym"
    for folder in ("artifacts", "verifier", "repo", "lib"):
        (scratch / folder).mkdir(parents=True, exist_ok=True)
    task = json.loads((here / "task.json").read_text())
    task["row"]["install_config"]["log_parser"] = f"parse_log_{parser}_stand_in"
    task["row"]["FAIL_TO_PASS"], task["row"]["PASS_TO_PASS"], task["evaluation_timeout"] = ["t1"], ["t2"], 60
    (here / "task.json").write_text(json.dumps(task))
    (scratch / "artifacts" / "capture.json").write_text('{"status": "ok"}')
    (scratch / "artifacts" / "model_patch.diff").write_text("diff --git a/m.py b/m.py\n")
    (scratch / "eval.sh").write_text(eval_script + f"echo {END}\n")
    (tmp_path / "driver.py").write_text(DRIVER)
    command = [python, "-I", "-B", str(tmp_path / "driver.py"), str(here), str(scratch), scenario]
    if ancestor is not None:
        (tmp_path / "ancestor.py").write_text(ANCESTOR)
        command = [python, "-I", "-B", str(tmp_path / "ancestor.py"), str(ancestor), "sh", "-c", relay_shell, "sh"]
        command += [python, "-I", "-B", str(tmp_path / "driver.py"), str(here), str(scratch), scenario]
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "NG_ANCESTORS_BELOW": str(os.getpid()), **env},
    )
    lines = [line for line in process.stdout.splitlines() if line.startswith("OUTCOME ")]
    assert len(lines) == 1, process.stdout + process.stderr
    outcome = ast.literal_eval(lines[0][len("OUTCOME ") :])
    # The code under test is the converted task's grade.py.
    assert Path(outcome["grade_file"]) == here / "grade.py"
    return outcome, process


@pytest.mark.parametrize("python", _interpreter_params())
@pytest.mark.parametrize("parser", ["lines", "junit"])
@pytest.mark.parametrize("resolved", [True, False])
def test_armed_grader_scores_a_real_run(
    tmp_path: Path, scratch: Path, rows: Path, parser_repo: Path, python: str, parser: str, resolved: bool
) -> None:
    # The whole path after the tests (the vendored run_verification, the upstream-style parser, XML included, the
    # inline to_thread, the outputs) runs under the audit hook with one real subprocess, and needs no import, no new
    # mapping and no refused open.
    eval_script = OUTPUTS[(parser, resolved)]
    outcome, process = _drive(tmp_path, scratch, rows, parser_repo, eval_script, parser=parser, python=python)
    assert outcome["code"] == 0, process.stderr
    assert outcome["refused"] is None and "failing closed" not in process.stderr
    assert json.loads(outcome["reward"])["reward"] == float(resolved)
    assert json.loads(outcome["grading"])["masked"] is False
    assert outcome["exit_record"] is None


@pytest.mark.parametrize("python", _interpreter_params())
def test_in_place_write_to_a_mapped_file_is_refused(
    tmp_path: Path, scratch: Path, rows: Path, parser_repo: Path, python: str
) -> None:
    # The mutation control: the tests (the eval script, same uid) try to rewrite a file the grader has mapped. The
    # protection takes its write bits and its owner, so the write fails, the mapped bytes stay, the check passes and
    # the failing run still scores 0.
    (scratch / "lib").mkdir(parents=True)
    target = scratch / "lib" / "mapped.bin"
    target.write_bytes(b"A" * 4096)
    attack = f"printf 'XXXXXXXX' 1<> {shlex.quote(str(target))} && echo WRITE_LANDED || echo WRITE_REFUSED\n"
    outcome, process = _drive(
        tmp_path, scratch, rows, parser_repo, attack + FAILING, python=python, NG_TARGET="mapped.bin"
    )
    assert "WRITE_REFUSED" in outcome["test_output"] and "WRITE_LANDED" not in outcome["test_output"]
    assert target.read_bytes() == b"A" * 4096 and outcome["mapped_prefix"] == b"A" * 8
    assert outcome["code"] == 0 and outcome["refused"] is None, process.stderr
    assert json.loads(outcome["reward"])["reward"] == 0.0


def test_changed_mapped_file_fails_closed(tmp_path: Path, scratch: Path, rows: Path, parser_repo: Path) -> None:
    # The check after the tests: a mapped file whose mode (or owner, size, mtime) changed while the tests ran means
    # the protection did not hold, so the grader exits 1 without a reward, even for a passing run. The change is made
    # in the grader process just after the tests, since the tests themselves cannot make it (the test above).
    (scratch / "lib").mkdir(parents=True)
    (scratch / "lib" / "mapped.bin").write_bytes(b"A" * 4096)
    outcome, process = _drive(
        tmp_path, scratch, rows, parser_repo, PASSING, "mapped_file_changed", NG_TARGET="mapped.bin"
    )
    assert outcome["code"] == 1 and outcome["reward"] is None
    assert "a file the grader has mapped changed during the tests" in process.stderr


@pytest.mark.skipif(shutil.which("timeout") is None or shutil.which("sh") is None, reason="needs timeout and sh")
@pytest.mark.parametrize("shell", sorted(RELAY_SHELLS))
def test_in_place_write_to_a_file_an_ancestor_has_mapped_is_refused(
    tmp_path: Path, scratch: Path, rows: Path, parser_repo: Path, shell: str
) -> None:
    # The processes that relay test.sh's exit status (timeout, the shell, the daemon above them) outlive the tests:
    # the files they have mapped are protected too, whatever the shape of the tree, so the tests cannot rewrite in
    # place the code that relays the status.
    (scratch / "lib").mkdir(parents=True)
    target = scratch / "lib" / "ancestor.bin"
    target.write_bytes(b"A" * 4096)
    attack = f"printf 'XXXXXXXX' 1<> {shlex.quote(str(target))} && echo WRITE_LANDED || echo WRITE_REFUSED\n"
    outcome, process = _drive(
        tmp_path, scratch, rows, parser_repo, attack + FAILING, ancestor=target, relay_shell=RELAY_SHELLS[shell]
    )
    assert "WRITE_REFUSED" in outcome["test_output"] and "WRITE_LANDED" not in outcome["test_output"]
    assert target.read_bytes() == b"A" * 4096 and "ANCESTOR_PREFIX b'AAAAAAAA'" in process.stdout
    assert outcome["code"] == 0 and outcome["refused"] is None, process.stderr
    assert json.loads(outcome["reward"])["reward"] == 0.0
    protected = json.loads(outcome["grading"])["protected_mappings"]
    assert protected["ancestors"][:2] == ["timeout", "sh"] and len(protected["ancestors"]) == 3
    assert protected["files"] == 1


def _file_backed_json() -> tuple[str, Path] | None:
    for python in _interpreters():
        found = subprocess.run([python, "-I", "-c", "import _json; print(getattr(_json, '__file__', ''))"], **_QUIET)
        if found.stdout.strip():
            return python, Path(found.stdout.strip())
    return None


def test_in_place_write_to_the_mapped_json_extension_is_refused(
    tmp_path: Path, scratch: Path, rows: Path, parser_repo: Path
) -> None:
    # The review's forgery rewrote the loaded _json extension in place. Here the grader's json uses a scratch copy
    # of it, and the tests' in-place write to that copy is refused before any byte changes.
    found = _file_backed_json()
    if found is None:
        pytest.skip("no Python here has a file-backed _json extension")
    python, json_lib = found
    lib = scratch / "lib"
    lib.mkdir(parents=True)
    shutil.copyfile(json_lib, lib / json_lib.name)
    original = (lib / json_lib.name).read_bytes()
    attack = (
        f"printf 'XXXXXXXX' 1<> {shlex.quote(str(lib / json_lib.name))} && echo WRITE_LANDED || echo WRITE_REFUSED\n"
    )
    outcome, process = _drive(
        tmp_path, scratch, rows, parser_repo, attack + FAILING, "json_copy", python=python, NG_TARGET=json_lib.name
    )
    assert outcome["json_file"] == str(lib / json_lib.name)
    assert "WRITE_REFUSED" in outcome["test_output"]
    assert (lib / json_lib.name).read_bytes() == original
    assert outcome["code"] == 0, process.stderr
    assert json.loads(outcome["reward"])["reward"] == 0.0


def test_import_after_the_tests_fails_closed(tmp_path: Path, scratch: Path, rows: Path, parser_repo: Path) -> None:
    # A module the grader would import after the tests is one the tests could have rewritten: here they make it
    # forge json.dumps. The import is refused, so the grader exits 1 without a reward instead of writing a 1.
    lib = scratch / "lib"
    lib.mkdir(parents=True)
    (lib / "standin.py").write_text("")
    forged = "import json\njson.dumps = lambda *args, **kwargs: '{\"reward\": 1.0}'\n"
    attack = f"printf '%s' {shlex.quote(forged)} > {shlex.quote(str(lib / 'standin.py'))}\n"
    outcome, process = _drive(tmp_path, scratch, rows, parser_repo, attack + FAILING, "lazy_import")
    assert "json.dumps" in (lib / "standin.py").read_text()
    assert outcome["code"] == 1 and outcome["reward"] is None
    assert "after the tests started" in outcome["refused"] and "standin" in outcome["refused"]
    assert "failing closed after the tests started" in process.stderr


def test_exception_after_the_tests_fails_closed_never_masks(
    tmp_path: Path, scratch: Path, rows: Path, parser_repo: Path
) -> None:
    outcome, process = _drive(tmp_path, scratch, rows, parser_repo, PASSING, "raise_after_tests")
    assert outcome["code"] == 1 and outcome["reward"] is None
    assert "RuntimeError: simulated failure after the tests" in process.stderr
    assert outcome["exit_record"] is None


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
