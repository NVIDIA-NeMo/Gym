# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference paths advertised in the GDPVal prompt must match the sandbox layout.

Stirrup copies the *contents* of ``input_files_dir`` into the sandbox working
directory, so the prompt must name files as they appear there.
"""

from responses_api_agents.stirrup_agent.app import _build_gdpval_user_prompt


def _section(prompt: str) -> str:
    return prompt.split("<reference_files>", 1)[1].split("</reference_files>", 1)[0]


def test_reference_paths_are_relative_to_the_staging_dir(tmp_path):
    ref_dir = tmp_path / "gdpval_ref_files_abc123"
    ref_dir.mkdir()
    (ref_dir / "Comp_Plan.docx").write_text("x")
    (ref_dir / "Quota.xlsx").write_text("x")

    section = _section(_build_gdpval_user_prompt("do the thing", str(ref_dir)))

    assert "- Comp_Plan.docx" in section
    assert "- Quota.xlsx" in section
    # The staging dir name does not exist inside the sandbox; advertising it
    # sent the agent chasing a nonexistent directory.
    assert "gdpval_ref_files_abc123" not in section


def test_nested_reference_paths_keep_their_subdirectory(tmp_path):
    ref_dir = tmp_path / "gdpval_ref_files_xyz"
    (ref_dir / "source").mkdir(parents=True)
    (ref_dir / "source" / "Ledger.xlsx").write_text("x")

    section = _section(_build_gdpval_user_prompt("t", str(ref_dir)))

    assert "- source/Ledger.xlsx" in section
    assert "gdpval_ref_files_xyz" not in section


def test_no_reference_files_renders_none(tmp_path):
    assert "None" in _section(_build_gdpval_user_prompt("t", None))
