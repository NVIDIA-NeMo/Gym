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

import shutil
import subprocess

import pytest

from nemo_gym.orchestration.api import ResumeConfig
from nemo_gym.orchestration.executors.resume_script import _walltime_to_seconds, render_resume_prologue


# ---------------------------------------------------------------------------
# _walltime_to_seconds
# ---------------------------------------------------------------------------


def test_minutes_only():
    assert _walltime_to_seconds("90") == 90 * 60


def test_minutes_seconds():
    assert _walltime_to_seconds("05:30") == 5 * 60 + 30


def test_hours_minutes_seconds():
    assert _walltime_to_seconds("02:00:00") == 2 * 3600


def test_days_hours():
    assert _walltime_to_seconds("1-06") == 86400 + 6 * 3600


def test_days_hours_minutes_seconds():
    assert _walltime_to_seconds("2-00:00:00") == 2 * 86400


def test_invalid_duration_raises():
    with pytest.raises(ValueError, match="Invalid max_walltime"):
        _walltime_to_seconds("not-a-duration")


def test_invalid_day_component_raises():
    with pytest.raises(ValueError, match="day component must be an integer"):
        _walltime_to_seconds("x-06:00:00")


# ---------------------------------------------------------------------------
# render_resume_prologue
# ---------------------------------------------------------------------------


def test_default_prologue_has_no_walltime_check():
    out = render_resume_prologue(ResumeConfig())
    assert "_gym_accumulated >=" not in out
    assert "-ge 3" in out  # default max_retries


def test_prologue_with_max_walltime_adds_check():
    out = render_resume_prologue(ResumeConfig(max_walltime="48:00:00"))
    assert "_gym_accumulated >= 172800" in out


def test_prologue_uses_custom_max_retries():
    out = render_resume_prologue(ResumeConfig(max_retries=9))
    assert "-ge 9" in out
    assert "Infra retry limit (9)" in out


def test_prologue_queues_own_successor():
    out = render_resume_prologue(ResumeConfig())
    assert "sbatch --dependency=afternotok:$SLURM_JOB_ID" in out


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("resume", [ResumeConfig(), ResumeConfig(max_walltime="12:00:00")])
def test_rendered_prologue_is_valid_bash(resume, tmp_path):
    script = tmp_path / "prologue.sh"
    script.write_text(render_resume_prologue(resume))
    subprocess.run(["bash", "-n", str(script)], check=True)
