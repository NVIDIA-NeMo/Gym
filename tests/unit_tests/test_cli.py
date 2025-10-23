# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tomllib
from pathlib import Path
from subprocess import run

from nemo_gym import PARENT_DIR
from nemo_gym.cli import RunConfig


# TODO: Eventually we want to add more tests to ensure that the CLI flows do not break
class TestCLI:
    def test_sanity(self) -> None:
        RunConfig(entrypoint="", name="")

    def test_pyproject_scripts(self) -> None:
        pyproject_path = Path(PARENT_DIR) / "pyproject.toml"
        with pyproject_path.open("rb") as f:
            pyproject_data = tomllib.load(f)

        project_scripts = pyproject_data["project"]["scripts"]

        def _test_single_arg(script: str, arg: str):
            result = run(
                f"{script} {arg}",
                capture_output=True,
                text=True,
                check=True,
                shell=True,
            )
            assert "Help for " in result.stdout, f"""Test failed for `{script} {arg}`.
Stdout: {result.stdout}
Stderr: {result.stderr}
"""

        for script in project_scripts:
            _test_single_arg(script, "+h=true")
            _test_single_arg(script, "+help=true")
