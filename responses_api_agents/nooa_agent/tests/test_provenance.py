# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
from pathlib import Path

from responses_api_agents.nooa_agent import provenance


SHA = "94c6b1db495df2cc9d3ddaae542520f6e12fb619"


def test_nooa_pin_reads_the_commit_of_the_nooa_requirement(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "-e nemo-gym[dev,sandbox] @ ../../\n"
        f"nooa @ git+https://github.com/NVIDIA-NeMo/labs-OO-Agents.git@{SHA}\n"
        f"nooa-cli @ https://github.com/NVIDIA-NeMo/labs-OO-Agents/archive/{SHA}.tar.gz#subdirectory=packages/nooa-cli\n"
    )
    assert provenance.nooa_pin(requirements) == SHA
    assert provenance.nooa_pin(tmp_path / "missing.txt") is None
    requirements.write_text("nooa>=0.0.6\n")
    assert provenance.nooa_pin(requirements) is None


def test_the_shipped_requirements_pin_nooa_to_a_commit() -> None:
    assert provenance.nooa_pin() is not None


def test_git_commit_reads_head_or_nothing(tmp_path: Path) -> None:
    assert provenance.git_commit(tmp_path) is None
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "x",
        ],
        check=True,
    )
    commit = provenance.git_commit(tmp_path)
    assert commit is not None and len(commit) == 40


def test_provenance_metadata_omits_unknown_values(monkeypatch) -> None:
    provenance.provenance_metadata.cache_clear()
    monkeypatch.setattr(provenance, "nooa_pin", lambda: SHA)
    monkeypatch.setattr(provenance, "git_commit", lambda: None)
    assert provenance.provenance_metadata() == {"nooa_commit": SHA}
    provenance.provenance_metadata.cache_clear()
