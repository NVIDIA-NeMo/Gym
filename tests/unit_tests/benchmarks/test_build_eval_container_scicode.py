# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SciCode data download in build_eval_container.sh fails the build unless the builder opts out."""

import re
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "benchmarks/nemotron_3.5_super/build_eval_container.sh"
STUBS = {
    "fails": "#!/bin/bash\nexit 1\n",
    "succeeds_without_the_file": "#!/bin/bash\nexit 0\n",
    "succeeds": "#!/bin/bash\necho data > benchmarks/scicode/data/test_data.h5\n",
}


def _scicode_block() -> str:
    match = re.search(r"^SCICODE_TEST_DATA=.*?^fi$", SCRIPT.read_text(), re.S | re.M)
    assert match, "the SciCode download block moved or lost its test_data.h5 check"
    return match.group(0)


@pytest.mark.parametrize(
    ("stub", "skip", "continues"),
    [
        ("fails", "0", False),
        ("fails", "1", True),
        ("succeeds_without_the_file", "0", False),
        ("succeeds_without_the_file", "1", True),
        ("succeeds", "0", True),
    ],
)
def test_scicode_data_failure_is_fatal_unless_skipped(tmp_path: Path, stub: str, skip: str, continues: bool) -> None:
    (tmp_path / "bin").mkdir()
    (tmp_path / "benchmarks/scicode/data").mkdir(parents=True)
    gdown = tmp_path / "bin/gdown"
    gdown.write_text(STUBS[stub])
    gdown.chmod(0o755)

    result = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{_scicode_block()}\necho CONTINUED"],
        cwd=tmp_path,
        env={"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "SKIP_SCICODE_DATA": skip},
        capture_output=True,
        text=True,
    )

    assert ("CONTINUED" in result.stdout) is continues, result.stderr
    assert (result.returncode == 0) is continues
    if not continues:
        assert "SKIP_SCICODE_DATA=1" in result.stderr
