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
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import TestSpec as SWETestSpec

from resources_servers.swebench import swebench_patches
from resources_servers.swebench.swebench_patches import (
    patch_swebench_verified_eval_script,
    patch_swebench_verified_grading_spec,
    run_instance,
)


OLD_ID = "astropy/units/tests/test_units.py::test_compose_roundtrip[]"
NEW_ID = "astropy/units/tests/test_units.py::test_compose_roundtrip[unit0]"
F2P = "astropy/units/tests/test_units.py::test_unknown_unit3"


@pytest.fixture
def spec() -> SWETestSpec:
    return SWETestSpec(
        instance_id="astropy__astropy-7606",
        repo="astropy/astropy",
        version="3.0",
        repo_script_list=[],
        eval_script_list=[],
        env_script_list=[],
        arch="x86_64",
        FAIL_TO_PASS=[F2P],
        PASS_TO_PASS=[OLD_ID],
        language="py",
        docker_specs={},
        namespace="swebench",
    )


def write_log(path: Path, results: str) -> Path:
    path.write_text(f"{START_TEST_OUTPUT}\n{results}\n{END_TEST_OUTPUT}\n")
    return path


def prediction(spec: SWETestSpec) -> dict[str, str]:
    return {"instance_id": spec.instance_id, "model_name_or_path": "test", "model_patch": "patch"}


@pytest.mark.parametrize(
    ("results", "resolved"),
    [
        (f"PASSED {NEW_ID}", True),
        (f"{NEW_ID} PASSED", True),
        (f"FAILED {NEW_ID}", False),
        (f"ERROR {NEW_ID}", False),
        ("", False),
        (f"PASSED {OLD_ID}", True),
        (f"FAILED {OLD_ID}\nPASSED {NEW_ID}", False),
        (f"PASSED {NEW_ID}\nFAILED {F2P}", False),
    ],
)
async def test_rewrite_report_uses_observed_id_without_crediting_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec: SWETestSpec, results: str, resolved: bool
) -> None:
    monkeypatch.setattr(swebench_patches, "RUN_EVALUATION_LOG_DIR", tmp_path)
    log_dir = tmp_path / "run" / "test" / spec.instance_id
    log_dir.mkdir(parents=True)
    path = write_log(log_dir / swebench_patches.LOG_TEST_OUTPUT, f"PASSED {F2P}\n{results}")
    original_log = path.read_text()
    result = await run_instance(spec, prediction(spec), False, False, None, "run", rewrite_reports=True)
    assert result == {"completed": True, "resolved": resolved}
    assert spec.PASS_TO_PASS == [OLD_ID]
    assert path.read_text() == original_log


@pytest.mark.parametrize("invalid", ["no markers", ">>>>> Tests Timed Out", ">>>>> Patch Apply Failed"])
def test_invalid_logs_do_not_gain_credit(tmp_path: Path, spec: SWETestSpec, invalid: str) -> None:
    path = write_log(tmp_path / "test.log", f"PASSED {NEW_ID}\nPASSED {F2P}")
    path.write_text(invalid if invalid == "no markers" else path.read_text() + invalid)
    patched = patch_swebench_verified_grading_spec(spec, path)
    assert patched is spec
    assert not get_eval_report(patched, prediction(spec), str(path), True)[spec.instance_id]["resolved"]


def test_alias_is_instance_specific(tmp_path: Path, spec: SWETestSpec) -> None:
    path = write_log(tmp_path / "test.log", f"PASSED {NEW_ID}\nPASSED {F2P}")
    other = replace(spec, instance_id="astropy__astropy-12345")
    assert patch_swebench_verified_grading_spec(other, path) is other
    assert not get_eval_report(other, prediction(other), str(path), True)[other.instance_id]["resolved"]


def test_sphinx_reporting_flag_does_not_modify_test_patch(spec: SWETestSpec) -> None:
    command = "tox --current-env -epy39 -v -- tests/test_extension.py"
    patch_command = f"git apply - <<'EOF'\n{command}\nEOF"
    sphinx = replace(spec, repo="sphinx-doc/sphinx", eval_script_list=[patch_command, command])
    script = patch_swebench_verified_eval_script(sphinx)
    assert patch_command in script
    assert "tox --current-env -epy39 -v -- -rA tests/test_extension.py\n" in script
    assert sphinx.eval_script_list == [patch_command, command]
    assert patch_swebench_verified_eval_script(spec) == spec.eval_script
    already_verbose = replace(sphinx, eval_script_list=[command.replace("-- tests", "-- -rA tests")])
    assert patch_swebench_verified_eval_script(already_verbose) == already_verbose.eval_script


async def test_live_run_grades_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec: SWETestSpec) -> None:
    monkeypatch.setattr(swebench_patches, "RUN_EVALUATION_LOG_DIR", tmp_path)
    container = AsyncMock()
    container.exec_run.return_value.exit_code = 0
    container.exec_run.return_value.output = b""
    container.exec_run_with_timeout.return_value = (
        f"{START_TEST_OUTPUT}\nPASSED {NEW_ID}\nPASSED {F2P}\n{END_TEST_OUTPUT}",
        False,
        1.0,
    )
    result = await run_instance(spec, prediction(spec), False, False, container, "run")
    assert result == {"completed": True, "resolved": True}
    container.cleanup.assert_awaited_once()


@pytest.mark.parametrize("test_body", ["assert True", "assert False", "raise RuntimeError('collection failed')"])
def test_sphinx_pytest_output_is_gradeable(tmp_path: Path, spec: SWETestSpec, test_body: str) -> None:
    import os
    import subprocess
    import sys

    test_file = tmp_path / "test_extension.py"
    test_file.write_text(
        test_body if test_body.startswith("raise ") else f"def test_needs_extensions():\n    {test_body}\n"
    )
    node_id = "test_extension.py::test_needs_extensions"
    sphinx = replace(
        spec,
        repo="sphinx-doc/sphinx",
        instance_id="sphinx-doc__sphinx-9711",
        version="4.0",
        FAIL_TO_PASS=[node_id],
        PASS_TO_PASS=[],
        eval_script_list=["tox --current-env -epy39 -v -- test_extension.py"],
    )
    # Execute the actual pytest arguments forwarded by the generated tox command.
    command = patch_swebench_verified_eval_script(sphinx).splitlines()[-1]
    args = command.split(" -- ", 1)[1].split()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=tmp_path,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    path = write_log(tmp_path / "test.log", result.stdout + result.stderr)
    report = get_eval_report(sphinx, prediction(sphinx), str(path), True)[sphinx.instance_id]
    assert report["resolved"] is (test_body == "assert True")
    if test_body == "assert True":
        assert f"PASSED {node_id}" in result.stdout
