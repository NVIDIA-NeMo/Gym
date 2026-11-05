# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared artifact downloads do not reuse partial results or skip other harnesses."""

import subprocess
import sys

from scripts.harness_conformance import artifacts


def test_failed_download_invalidates_partial_result_and_continues(tmp_path, monkeypatch):
    calls = []

    def download(command, **kwargs):
        calls.append(command)
        if "harness-conformance-pi-12-2" in command:
            partial = tmp_path / "pi" / "probes" / "ci-result.json"
            partial.parent.mkdir(parents=True)
            partial.write_text('"partial result"')
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(artifacts.subprocess, "run", download)
    artifacts.download_artifacts(["pi", "hermes"], repo="NVIDIA-NeMo/Gym", run_id="12", attempt="2", output=tmp_path)
    assert len(calls) == 2 and "harness-conformance-hermes-12-2" in calls[1]
    assert not (tmp_path / "pi" / "probes" / "ci-result.json").exists()


def test_p0_reporter_imports_without_gym_or_site_packages():
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; from scripts.harness_conformance import checks; "
            "assert not any(name.startswith('nemo_gym') for name in sys.modules)",
        ],
        check=True,
    )
