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
"""`scripts/check_runtime_write_paths.py` keeps server code from resolving paths against its source folder."""

import importlib.util
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "check_runtime_write_paths", REPO_ROOT / "scripts" / "check_runtime_write_paths.py"
)
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))


def _findings(root: Path) -> set[tuple[str, str, str]]:
    return {(file, rule, scope) for file, _, rule, scope in check.scan(root)}


def test_flags_working_directory_calls_and_relative_path_defaults(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "resources_servers/demo/app.py",
        """
        import os
        from pathlib import Path
        from pydantic import BaseModel, Field

        class DemoConfig(BaseModel):
            artifacts_dir: Path = Path("results/demo")
            cache_dir: str = Field(default="cache/demo")
            data_root: str = "/abs/data"
            debug_dir: Path | None = None

        class Demo:
            def output(self):
                return Path.cwd() / "results"

            def crops(self):
                return os.path.join(os.getcwd(), "crops")

        def start(log_dir: str = "cache/logs", *, work_root: str = "outputs/x", home_dir: str = "~/x"):
            pass
        """,
    )

    assert _findings(tmp_path) == {
        ("resources_servers/demo/app.py", "relative-default", "DemoConfig.artifacts_dir"),
        ("resources_servers/demo/app.py", "relative-default", "DemoConfig.cache_dir"),
        ("resources_servers/demo/app.py", "cwd", "Demo.output"),
        ("resources_servers/demo/app.py", "cwd", "Demo.crops"),
        ("resources_servers/demo/app.py", "relative-default", "start.log_dir"),
        ("resources_servers/demo/app.py", "relative-default", "start.work_root"),
    }


def test_skips_tests_scripts_cli_and_marked_reads(tmp_path: Path) -> None:
    cwd_call = "from pathlib import Path\nROOT = Path.cwd()\n"
    _write(tmp_path, "resources_servers/demo/tests/test_app.py", cwd_call)
    _write(tmp_path, "resources_servers/demo/scripts/build.py", cwd_call)
    _write(tmp_path, "resources_servers/demo/prepare_data.py", cwd_call)
    _write(tmp_path, "nemo_gym/cli/run.py", cwd_call)
    _write(
        tmp_path,
        "responses_api_agents/demo/app.py",
        "from pathlib import Path\nP = Path.cwd()  # runtime-path: read\n",
    )

    assert _findings(tmp_path) == set()


def test_allowlisted_sites_pass_and_new_or_stale_entries_fail(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _write(
        tmp_path, "responses_api_agents/demo/app.py", "from pathlib import Path\n\ndef run():\n    return Path.cwd()\n"
    )
    allowlist = tmp_path / "allowlist.txt"

    allowlist.write_text("# header\n\nresponses_api_agents/demo/app.py cwd run\n")
    assert check.main(["--root", str(tmp_path), "--allowlist", str(allowlist)]) == 0

    allowlist.write_text("")
    assert check.main(["--root", str(tmp_path), "--allowlist", str(allowlist)]) == 1
    assert "responses_api_agents/demo/app.py:4: cwd in run" in capsys.readouterr().out

    allowlist.write_text("responses_api_agents/demo/app.py cwd run\nresponses_api_agents/gone/app.py cwd run\n")
    assert check.main(["--root", str(tmp_path), "--allowlist", str(allowlist)]) == 1
    assert "'responses_api_agents/gone/app.py cwd run' no longer matches" in capsys.readouterr().out


def test_repository_has_no_sites_outside_the_allowlist() -> None:
    assert check.main(["--root", str(REPO_ROOT)]) == 0


def _yaml_findings(root: Path) -> set[tuple[str, str, str]]:
    return {(file, rule, scope) for file, _, rule, scope in check.scan_yaml(root)}


def test_flags_relative_folder_settings_in_yaml_configs(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "responses_api_agents/demo/configs/demo.yaml",
        """
        demo_agent:
          responses_api_agents:
            demo:
              artifacts_dir: results/demo
              env_dir: ${oc.env:DEMO_DIR,results/demo}
              nested_dir: ${oc.select:a,${oc.select:b,results/x}/agent}
              unset_dir: ${oc.env:DEMO_DIR,null}
              absolute_dir: /scratch/demo
              under_results_dir: ${results_dir}/demo
              select_results_dir: ${oc.select:results_dir,results}/demo
              read_only_dir: data/prompts  # runtime-path: read
              timeout: 10
        prepare_script_args:
          output_dir: benchmarks/demo/data
        """,
    )
    _write(tmp_path, "benchmarks/demo/config.yaml", "jobs:\n  - jobs_dir: results/jobs\n")

    assert _yaml_findings(tmp_path) == {
        (
            "responses_api_agents/demo/configs/demo.yaml",
            "yaml-relative-default",
            "demo_agent.responses_api_agents.demo.artifacts_dir",
        ),
        (
            "responses_api_agents/demo/configs/demo.yaml",
            "yaml-relative-default",
            "demo_agent.responses_api_agents.demo.env_dir",
        ),
        (
            "responses_api_agents/demo/configs/demo.yaml",
            "yaml-relative-default",
            "demo_agent.responses_api_agents.demo.nested_dir",
        ),
        ("benchmarks/demo/config.yaml", "yaml-relative-default", "jobs.0.jobs_dir"),
    }


def test_yaml_tests_folders_are_skipped(tmp_path: Path) -> None:
    _write(tmp_path, "resources_servers/demo/tests/fixture.yaml", "artifacts_dir: results/demo\n")

    assert _yaml_findings(tmp_path) == set()
