# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import configparser
import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

import pytest
import yaml
from scripts.ci.combine_core_coverage import combine
from scripts.ci.core_test_shards import DURATIONS_PATH, partition_files


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/unit-tests.yml"


def test_partition_is_complete_disjoint_balanced_and_order_independent():
    durations = {"migration.py": 8, "benchmarks.py": 7, "other.py": 6, "small.py": 5}
    files = [*durations, "new.py", "migration.py"]
    assignments = partition_files(files, durations, 4)
    assert assignments == partition_files(list(reversed(files)), durations, 4)
    assert set(assignments) == set(files)
    assert set(assignments.values()) == set(range(4))
    assert len({assignments[file] for file in ("migration.py", "benchmarks.py", "other.py", "new.py")}) == 4
    assert partition_files(files, {}, 1) == dict.fromkeys(files, 0)
    with pytest.raises(ValueError, match="positive"):
        partition_files(files, durations, 0)


def test_recorded_weights_spread_the_expensive_files():
    durations = json.loads(DURATIONS_PATH.read_text())["files"]
    assert durations and all(value > 0 for value in durations.values())
    assignments = partition_files(list(durations), durations, 4)
    slowest = sorted(durations, key=durations.get, reverse=True)[:4]
    assert len({assignments[file] for file in slowest}) == 4
    loads = [sum(durations[file] for file, shard in assignments.items() if shard == index) for index in range(4)]
    assert max(loads) - min(loads) < 1


def test_real_collection_keeps_each_file_and_parameter_exactly_once(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    for index in range(8):
        (tmp_path / f"test_{index}.py").write_text(
            "import pytest\n@pytest.mark.parametrize('n', [1, 2])\ndef test_example(n):\n    assert n\n"
        )
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTEST_ADDOPTS="")

    def collect(*options):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "scripts.ci.core_test_shards", "--collect-only", "-q", *options],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return {line for line in result.stdout.splitlines() if line.startswith("test_") and "::" in line}

    complete = collect()
    shards = [collect(f"--core-shard-index={index}", "--core-num-shards=4") for index in range(4)]
    assert len(complete) == 16
    assert Counter(node for shard in shards for node in shard) == Counter(dict.fromkeys(complete, 1))
    for index in range(8):
        assert sum(any(node.startswith(f"test_{index}.py::") for node in shard) for shard in shards) == 1


def test_four_workers_preserve_file_ownership_and_combine_coverage(tmp_path: Path) -> None:
    """Run actual xdist workers with sharding enabled and check coverage of every assigned file."""
    from coverage import CoverageData

    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (tmp_path / "example_source.py").write_text(
        "".join(f"def value_{index}():\n    return {index}\n" for index in range(8))
    )
    for index in range(8):
        (tests / f"test_{index}.py").write_text(
            "import os\nfrom pathlib import Path\nimport pytest\n"
            f"from example_source import value_{index}\n"
            "@pytest.mark.parametrize('n', [0, 1])\ndef test_value(n):\n"
            f"    assert value_{index}() == {index}\n"
            f"    path = Path(__file__).parents[1] / 'evidence' / f'{index}-{{n}}'\n"
            "    with path.open('x') as out:\n        out.write(str(os.getpid()))\n"
        )
    settings = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["coverage"]["run"]
    config = configparser.ConfigParser()
    config["run"] = {
        "relative_files": str(settings["relative_files"]),
        "source": "\n".join(settings["source"]),
        "concurrency": "\n".join(settings["concurrency"]),
        "omit": "tests/*",
    }
    coverage_config = tmp_path / ".coveragerc"
    with coverage_config.open("w") as stream:
        config.write(stream)
    database = tmp_path / ".coverage.core-0"
    env = {name: value for name, value in os.environ.items() if not name.startswith("COV_CORE_")}
    env.update(PYTHONPATH=str(ROOT), PYTEST_ADDOPTS="", COVERAGE_FILE=str(database))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "scripts.ci.core_test_shards",
            "-n",
            "4",
            "--dist=loadfile",
            "--core-shard-index=0",
            "--core-num-shards=1",
            "--cov=.",
            f"--cov-config={coverage_config}",
            "--cov-report=",
            "--cov-fail-under=100",
            "-q",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    records = {path.name: path.read_text() for path in evidence.iterdir()}
    assert len(records) == 16
    assert len(set(records.values())) == 4
    for index in range(8):
        assert records[f"{index}-0"] == records[f"{index}-1"]
    data = CoverageData(basename=str(database))
    data.read()
    assert "example_source.py" in data.measured_files()
    assert set(data.lines("example_source.py")) == set(range(1, 17))


def test_core_launcher_uses_four_workers_and_preserves_shard_and_coverage_options(tmp_path: Path) -> None:
    """Inspect the actual core launcher after replacing only dependency installation and pytest execution."""
    ci_dir = tmp_path / "repo/scripts/ci"
    ci_dir.mkdir(parents=True)
    for name in ("core_unit_tests.sh", "sanitize_env.sh", "test_worker_env.sh"):
        shutil.copy2(ROOT / "scripts/ci" / name, ci_dir / name)
    (ci_dir / "setup_dev.sh").write_text("true\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    command = bin_dir / "ng_dev_test"
    command.write_text(
        "#!/usr/bin/env python3\nimport json, os\nfrom pathlib import Path\n"
        "keys = ('PYTEST_ADDOPTS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', "
        "'NUMEXPR_NUM_THREADS')\n"
        "Path(os.environ['CI_CAPTURE']).write_text(json.dumps({key: os.environ[key] for key in keys}))\n"
    )
    command.chmod(0o755)
    capture = tmp_path / "environment.json"
    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        CI_CAPTURE=str(capture),
        GYM_CI_JUNIT_DIR=str(tmp_path / "junit reports"),
        PYTEST_ADDOPTS="-n 99",
        OMP_NUM_THREADS="64",
    )
    subprocess.run([str(ci_dir / "core_unit_tests.sh"), "2", "4"], env=env, check=True)
    recorded = json.loads(capture.read_text())
    options = shlex.split(recorded["PYTEST_ADDOPTS"])
    assert options[options.index("-n") + 1] == "4"
    assert "--dist=loadfile" in options
    assert "--core-shard-index=2" in options and "--core-num-shards=4" in options
    assert "--cov-fail-under=0" in options
    assert f"--junitxml={tmp_path / 'junit reports/core.xml'}" in options
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert recorded[name] == "1"


@pytest.mark.parametrize("args", [["0"], ["-1", "4"], ["4", "4"], ["0", "0"], ["x", "4"], ["0", "4", "x"]])
def test_script_rejects_invalid_shards_before_setup(args):
    result = subprocess.run([str(ROOT / "scripts/ci/core_unit_tests.sh"), *args], capture_output=True, text=True)
    assert result.returncode == 2
    assert "Usage:" in result.stderr or "Expected 0 <=" in result.stderr


@pytest.mark.parametrize(
    "options", [["--core-shard-index=0"], ["--core-num-shards=4"], ["--core-shard-index=4", "--core-num-shards=4"]]
)
def test_plugin_rejects_invalid_shards(tmp_path, options):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "scripts.ci.core_test_shards", *options],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(ROOT), PYTEST_ADDOPTS=""),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4
    assert "core sharding requires" in result.stderr


@pytest.mark.parametrize("missing", [*range(5)])
def test_combine_rejects_each_missing_input_before_reporting(tmp_path, monkeypatch, missing):
    names = [*(f".coverage.core-{index}" for index in range(4)), ".coverage.sandbox"]
    for index, name in enumerate(names):
        if index != missing:
            (tmp_path / name).write_bytes(b"data")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(FileNotFoundError, match=names[missing]):
        combine(tmp_path)
    assert not calls


def test_combine_enforces_the_configured_gate_across_checkout_roots(tmp_path, monkeypatch, capfd):
    from coverage import Coverage, CoverageData

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COVERAGE_FILE", str(tmp_path / ".coverage"))
    (tmp_path / "target.py").write_text("a = 1\nb = 2\n")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.coverage.run]\nrelative_files = true\nsource = ["target"]\n'
        "[tool.coverage.report]\nfail_under = 95.0\nprecision = 2\n"
    )
    # Report repository sources, not temporary dependencies or synthetic code.
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "target.py"], cwd=tmp_path, check=True)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for index, name in enumerate([*(f".coverage.core-{i}" for i in range(4)), ".coverage.sandbox"]):
        checkout = tmp_path / f"runner-{index}"
        checkout.mkdir()
        (checkout / "target.py").write_text((tmp_path / "target.py").read_text())
        monkeypatch.chdir(checkout)
        data = CoverageData(basename=str(inputs / name))
        data.add_lines({"target.py": [1]})
        data.write()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        combine(inputs)
    data = CoverageData(basename=str(inputs / ".coverage.sandbox"))
    data.add_lines({"target.py": [1, 2]})
    data.write()
    combine(inputs)
    coverage = Coverage()
    coverage.load()
    assert coverage.get_data().measured_files() == {"target.py"}

    # These paths were recorded by real CI shards but don't exist on the
    # reporting runner. They must not hide missing tracked production code.
    data = CoverageData(basename=str(inputs / ".coverage.core-0"))
    data.read()
    data.add_lines({"cache/uv/.temporary/python/__init__.py": [1], "x.py": [1]})
    data.write()
    combine(inputs)
    (tmp_path / "target.py").unlink()
    with pytest.raises(subprocess.CalledProcessError):
        combine(inputs)
    assert "No source for code: 'target.py'" in capfd.readouterr().out
    subprocess.run(["git", "rm", "--cached", "target.py"], check=True)
    with pytest.raises(ValueError, match="No tracked repository sources"):
        combine(inputs)


def test_production_coverage_records_one_relative_source_across_checkout_roots(tmp_path):
    from coverage import CoverageData

    config = ROOT / "pyproject.toml"
    databases = []
    for index in range(2):
        checkout = tmp_path / f"checkout-{index}"
        checkout.mkdir()
        (checkout / "target.py").write_text("answer = 42\n")
        database = tmp_path / f".coverage.producer-{index}"
        # Use the production config, including relative_files and source. The
        # temporary fixture must not inherit the production /tmp/* omission.
        producer = subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy, sys; from coverage import Coverage; "
                "cov = Coverage(config_file=sys.argv[1], data_file=sys.argv[2], omit=[]); "
                "cov.start(); runpy.run_path('target.py', run_name='__main__'); cov.stop(); cov.save(); "
                "print(cov.get_data().data_filename())",
                str(config),
                str(database),
            ],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        )
        # Production multiprocessing coverage adds a process-specific suffix.
        database = producer.stdout.strip()
        data = CoverageData(basename=database)
        data.read()
        assert data.measured_files() == {"target.py"}
        assert data.lines("target.py") == [1]
        databases.append(str(database))

    combined = CoverageData(basename=str(tmp_path / ".coverage.combined"))
    for database in databases:
        data = CoverageData(basename=database)
        data.read()
        combined.update(data)
    assert combined.measured_files() == {"target.py"}
    assert combined.lines("target.py") == [1]


def test_workflow_requires_all_shards_sandbox_and_coverage_and_preserves_routing():
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    core = jobs["test"]
    assert core["strategy"] == {"fail-fast": False, "matrix": {"shard": [0, 1, 2, 3]}}
    assert any('core_unit_tests.sh "${{ matrix.shard }}" 4' in step.get("run", "") for step in core["steps"])
    for job in ("test", "sandbox-tests"):
        upload = next(step for step in jobs[job]["steps"] if "upload-artifact@" in step.get("uses", ""))
        assert upload["with"]["include-hidden-files"] is True
        assert upload["with"]["if-no-files-found"] == "error"
    assert jobs["core-coverage"]["needs"] == ["test", "sandbox-tests"]
    assert set(jobs["test-result"]["needs"]) == {
        "detect",
        "test",
        "sandbox-tests",
        "core-coverage",
        "changed-server-tests",
    }
    assert jobs["test-result"]["name"] == "Test"
    assert jobs["test-result"]["if"] == "always()"
    assert jobs["changed-server-tests"]["if"] == "needs.detect.outputs.run_servers == 'true'"
    assert jobs["server-suite"]["strategy"]["matrix"]["shard"] == list(range(8))


@pytest.mark.parametrize(
    "failed", ["DETECT_RESULT", "CORE_RESULT", "SANDBOX_RESULT", "COVERAGE_RESULT", "SERVERS_RESULT", None]
)
@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped"])
def test_existing_test_gate_fails_closed(failed, result):
    command = yaml.safe_load(WORKFLOW.read_text())["jobs"]["test-result"]["steps"][0]["run"]
    env = {
        name: "success"
        for name in ("DETECT_RESULT", "CORE_RESULT", "SANDBOX_RESULT", "COVERAGE_RESULT", "SERVERS_RESULT")
    }
    env.update(RUN_FULL="true", RUN_SERVERS="true")
    if failed:
        env[failed] = result
    completed = subprocess.run(["bash", "-c", command], env=env)
    assert completed.returncode == (1 if failed else 0)


def test_docs_only_gate_needs_successful_detection_but_no_tests():
    command = yaml.safe_load(WORKFLOW.read_text())["jobs"]["test-result"]["steps"][0]["run"]
    completed = subprocess.run(
        ["bash", "-c", command], env={"DETECT_RESULT": "success", "RUN_FULL": "false", "RUN_SERVERS": "false"}
    )
    assert completed.returncode == 0
