# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""scripts/swe_rows_to_harbor: converted tasks load as Harbor tasks, reuse the servers' commands, and grade alike."""

import asyncio
import errno
import importlib.util
import json
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import ModuleType

import pytest
from scripts.swe_rows_to_harbor import convert


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("compatible", [True, False])
def test_grader_uses_task_venv_only_when_it_meets_python_requirement(tmp_path: Path, compatible: bool) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name in ("python", "python3"):
        path = binaries / name
        path.write_text("#!/bin/sh\nexit 1\n")
        path.chmod(0o755)
    (binaries / "mkdir").symlink_to("/bin/mkdir")
    task_python = tmp_path / "task-python"
    if compatible:
        task_python.symlink_to(sys.executable)
    else:
        task_python.write_text("#!/bin/sh\nexit 1\n")
        task_python.chmod(0o755)
    marker = tmp_path / "graded"
    grader = tmp_path / "grade.py"
    grader.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('graded')\n")
    script = convert._test_script()
    script = script.replace("/opt/conda/bin/python3", str(tmp_path / "absent-conda"))
    script = script.replace("/usr/local/bin/python3", str(tmp_path / "absent-local"))
    script = script.replace("/env/.venv/bin/python", shlex.quote(str(task_python)))
    script = script.replace("/tests/nemo_gym/grade.py", shlex.quote(str(grader)))
    script = script.replace("/logs/verifier", shlex.quote(str(tmp_path / "logs")))
    result = subprocess.run(
        ["/bin/bash", "-c", script], env={"PATH": str(binaries)}, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == (0 if compatible else 1)
    assert marker.exists() is compatible
    if not compatible:
        assert "no Python 3.10+" in result.stderr


@pytest.mark.parametrize("grader_exit,compatible", [(0, True), (7, True), (0, False)])
def test_private_grader_runtime_runs_and_cleans_up(tmp_path: Path, grader_exit: int, compatible: bool) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name in ("python", "python3"):
        path = binaries / name
        path.write_text("#!/bin/sh\nexit 1\n")
        path.chmod(0o755)
    for name, source in (
        ("mkdir", "/bin/mkdir"),
        ("mktemp", "/usr/bin/mktemp"),
        ("tar", "/usr/bin/tar"),
        ("rm", "/bin/rm"),
    ):
        (binaries / name).symlink_to(source)
    tests = tmp_path / "tests"
    tests.mkdir()
    executable = tmp_path / "python3"
    executable.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n' if compatible else "#!/bin/sh\nexit 9\n"
    )
    executable.chmod(0o755)
    with tarfile.open(tests / "python.tar.gz", "w:gz") as archive:
        archive.add(executable, arcname="python/bin/python3")
    marker = tmp_path / "graded"
    (tests / "grade.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('graded')\nraise SystemExit({grader_exit})\n"
    )
    script = convert._test_script()
    for candidate in ("/opt/conda/bin/python3", "/usr/local/bin/python3", "/env/.venv/bin/python"):
        script = script.replace(candidate, str(tmp_path / "absent"))
    script = script.replace("/tests/nemo_gym", str(tests)).replace("/logs/verifier", str(tmp_path / "logs"))
    script = script.replace("/tmp/nemo-gym-swe-grader-XXXXXXXX", str(tmp_path / "runtime-XXXXXXXX"))
    result = subprocess.run(
        ["/bin/bash", "-c", script], env={"PATH": str(binaries)}, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == (grader_exit if compatible else 1), result.stderr
    assert marker.exists() is compatible
    assert not list(tmp_path.glob("runtime-*"))
    assert (tests / "python.tar.gz").is_file()


GOLDEN_PATCH = (
    "diff --git a/widget/core.py b/widget/core.py\n--- a/widget/core.py\n+++ b/widget/core.py\n"
    "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
)
TEST_FILE_PATCH = (
    "diff --git a/tests/test_core.py b/tests/test_core.py\n--- a/tests/test_core.py\n+++ b/tests/test_core.py\n"
    "@@ -1 +1 @@\n-old = True\n+old = False\n"
)
PRISTINE_FILE_PATCH = "diff --git a/notes.txt b/notes.txt\nnew file mode 100644\n--- /dev/null\n+++ b/notes.txt\n@@ -0,0 +1 @@\n+scratch\n"
NEXT_ROW = {
    "instance_id": "example-task",
    "workdir": "/testbed",
    "image_ref": "example/image:task",
    "patch": GOLDEN_PATCH,
    "test_patch": TEST_FILE_PATCH,
    "expected_output_json": '{"test_one": "PASSED"}',
    "responses_create_params": {"input": [{"role": "user", "content": "Fix the widget.\r\nIt should return 2."}]},
    "source_server": "swe_next",
}
UNSUPPORTED_ROW = {"instance_id": "delivery__iceberg-1", "source_server": "swemer_v1"}


@pytest.mark.parametrize("cross_device", [False, True])
def test_conversion_delivers_supplied_grader_archive_only_with_hidden_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cross_device: bool
) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text(json.dumps(NEXT_ROW) + "\n")
    archive = tmp_path / "supplied-python.tar.gz"
    archive.write_bytes(b"supplied runtime archive")
    if cross_device:

        def cannot_link(*args, **kwargs):
            raise OSError(errno.EXDEV, "Cross-device link")

        monkeypatch.setattr(convert.os, "link", cannot_link)
    output = tmp_path / "tasks"
    manifest = convert.convert(source, output, dataset_name="test/private-grader", grader_python_archive=archive)
    task = output / manifest["converted"][0]["task"]
    assert (task / "tests/nemo_gym/python.tar.gz").read_bytes() == archive.read_bytes()
    assert ((task / "tests/nemo_gym/python.tar.gz").stat().st_ino == archive.stat().st_ino) is not cross_device
    assert not list((task / "environment").glob("*.tar.gz"))
    assert (task / "instruction.md").read_bytes() == NEXT_ROW["responses_create_params"]["input"][0][
        "content"
    ].encode()
    assert manifest["grader_python_archive"] == str(archive)


def _load_grader(task_dir: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"grade_{task_dir.name}", task_dir / "tests/nemo_gym/grade.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
