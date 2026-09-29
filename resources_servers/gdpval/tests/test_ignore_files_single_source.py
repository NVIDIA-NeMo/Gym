# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run-state exclusions have exactly one definition.

Both judging paths -- rubric scoring via ``file_reader`` and pairwise comparison
via ``comparison`` -- must skip the same files. When each kept its own copy they
could drift, and a file dropped from only one list leaks the agent's own
trajectory into that judge's prompt without anything failing.
"""

from pathlib import Path

from resources_servers.gdpval import comparison
from responses_api_agents.stirrup_agent import file_reader


def test_comparison_reuses_the_agents_definition():
    assert comparison._ignore_files() is file_reader.IGNORE_FILES, (
        "comparison.py must import IGNORE_FILES, not define its own copy"
    )


def test_comparison_does_not_pull_the_agent_in_at_import_time():
    """The lazy import is load-bearing, not stylistic.

    ``stirrup_agent/__init__`` imports its app, which imports Ray and FastAPI. A
    module-level import here took comparison.py from 0.69 s to 3.77 s, and
    ``multistage_elo`` imports this module at module scope.
    """
    source = Path(comparison.__file__).read_text()
    module_level = [
        line
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) and "responses_api_agents" in line
    ]
    assert not module_level, f"import responses_api_agents lazily, not at module scope: {module_level}"


def test_the_definition_still_covers_every_stirrup_run_artefact():
    """Pinned as an exact set.

    Asserting membership for a few names cannot catch a *deletion* from the list,
    which is the change that would silently start leaking a run-state file into
    judged submissions.
    """
    assert set(file_reader.IGNORE_FILES) == {
        "finish_params.json",
        "history.json",
        "history.pkl",
        "metadata.json",
        "inprogress_history.json",
        "log.txt",
        "reference_files",
    }
