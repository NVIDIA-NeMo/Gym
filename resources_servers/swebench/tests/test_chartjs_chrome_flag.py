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
"""Chart.js evaluation scripts start Chrome without /dev/shm."""

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from swebench.harness.run_evaluation import make_test_spec

from resources_servers.swebench.swebench_patches import (
    CHARTJS_CHROME_FLAG,
    chartjs_chrome_flag_edit,
    patch_swebench_chartjs_chrome_flag,
)
from resources_servers.swebench.tests.test_pytest_result_summary import ASTROPY_ROW, row, spec_for


CHARTJS_3_ROW = row("chartjs/Chart.js", "chartjs__Chart.js-7951", "3.0", "test/specs/controller.bar.tests.js", [], [])
CHARTJS_4_ROW = row("chartjs/Chart.js", "chartjs__Chart.js-11352", "4.3", "test/specs/scale.time.tests.js", [], [])

# The Chrome launcher of Chart.js's karma.conf.js and karma.conf.cjs.
KARMA_CONFIG = """module.exports = function(karma) {
  karma.set({
    browsers: (args.browsers || 'chrome,firefox').split(','),
    customLaunchers: {
      chrome: {
        base: 'Chrome',
        flags: [
          '--disable-accelerated-2d-canvas',
          '--disable-background-timer-throttling'
        ]
      },
      firefox: {
        base: 'Firefox',
        prefs: {
          'layers.acceleration.disabled': true
        }
      }
    }
  });
};
"""


def karma_line(script: list[str]) -> int:
    return next(index for index, line in enumerate(script) if "karma start" in line)


class TestChartjsChromeFlag:
    @pytest.mark.parametrize(
        ("data", "karma_config"),
        [(CHARTJS_3_ROW, "karma.conf.js"), (CHARTJS_4_ROW, "karma.conf.cjs")],
        ids=["chartjs-3", "chartjs-4"],
    )
    def test_the_flag_is_added_to_the_karma_config_just_before_karma_starts(
        self, data: dict, karma_config: str
    ) -> None:
        script = spec_for(data).eval_script_list
        edit = chartjs_chrome_flag_edit(karma_config)

        assert script[karma_line(script) - 1] == edit
        # Only this line is added to what SWE-bench builds.
        assert [line for line in script if line != edit] == make_test_spec(data, namespace="swebench").eval_script_list

    def test_other_repositories_keep_their_scripts(self) -> None:
        assert not any(CHARTJS_CHROME_FLAG in line for line in spec_for(ASTROPY_ROW).eval_script_list)

    def test_patching_twice_adds_the_line_once(self) -> None:
        test_spec = spec_for(CHARTJS_3_ROW)

        patch_swebench_chartjs_chrome_flag(test_spec)

        assert test_spec.eval_script_list.count(chartjs_chrome_flag_edit("karma.conf.js")) == 1

    def test_a_chartjs_script_without_karma_is_rejected(self) -> None:
        test_spec = SimpleNamespace(
            repo="chartjs/Chart.js", instance_id="chartjs__Chart.js-1", eval_script_list=["npm test"]
        )

        with pytest.raises(ValueError, match="does not start Karma"):
            patch_swebench_chartjs_chrome_flag(test_spec)

    @pytest.mark.skipif(
        not sys.platform.startswith("linux") or shutil.which("sed") is None,
        reason="the edit is a GNU sed command, as in the Linux evaluation image",
    )
    def test_the_edit_puts_the_flag_first_in_the_chrome_launcher(self, tmp_path: Path) -> None:
        (tmp_path / "karma.conf.js").write_text(KARMA_CONFIG)

        subprocess.run(["bash", "-c", chartjs_chrome_flag_edit("karma.conf.js")], cwd=tmp_path, check=True)

        assert (tmp_path / "karma.conf.js").read_text() == KARMA_CONFIG.replace(
            "        flags: [\n", f"        flags: [\n          '{CHARTJS_CHROME_FLAG}',\n"
        )
