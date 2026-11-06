# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The repository paths a workspace diff section names: every call's change marks are read from them."""

import pytest

from resources_servers.agentic_if.if_grading.checkers.run import diff_files
from resources_servers.agentic_if.if_grading.git_diff import patch_section_paths


# Sections as `git diff --binary --full-index` writes them, with core.quotepath on (non-ASCII bytes octal-escaped).
OLD, NEW, NONE = "1" * 40, "2" * 40, "0" * 40
INDEX = f"index {OLD}..{NEW} 100644"
HUNK = "@@ -1 +1,2 @@\n k\n+l\n"
BINARY = "GIT binary patch\nliteral 3\nKcmc~}NCE%>uK>{i\n\nliteral 0\nHcmV?d00001\n\n"


def content(old_header, new_header):
    """A text change whose header and content headers name the two sides as `old_header` and `new_header`."""
    return f"diff --git {old_header} {new_header}\n{INDEX}\n--- {old_header}\n+++ {new_header}\n{HUNK}"


@pytest.mark.parametrize(
    "section,paths",
    [
        (content('"a/caf\\303\\251.txt"', '"b/caf\\303\\251.txt"'), ("café.txt", "café.txt")),
        (content('"a/quo\\"te.txt"', '"b/quo\\"te.txt"'), ('quo"te.txt', 'quo"te.txt')),
        (content('"a/back\\\\slash.txt"', '"b/back\\\\slash.txt"'), ("back\\slash.txt", "back\\slash.txt")),
        (content('"a/tab\\there.txt"', '"b/tab\\there.txt"'), ("tab\there.txt", "tab\there.txt")),
        # Git ends an unquoted name that has a space with a tab
        (
            f"diff --git a/with space.txt b/with space.txt\n{INDEX}\n"
            f"--- a/with space.txt\t\n+++ b/with space.txt\t\n{HUNK}",
            ("with space.txt", "with space.txt"),
        ),
        (
            f"diff --git a/added.txt b/added.txt\nnew file mode 100644\nindex {NONE}..{NEW}\n"
            "--- /dev/null\n+++ b/added.txt\n@@ -0,0 +1 @@\n+n\n",
            (None, "added.txt"),
        ),
        # binary sections have no content headers: the header and the file mode lines decide
        (f"diff --git a/new.bin b/new.bin\nnew file mode 100644\nindex {NONE}..{NEW}\n{BINARY}", (None, "new.bin")),
        (
            f"diff --git a/gone.bin b/gone.bin\ndeleted file mode 100644\nindex {OLD}..{NONE}\n{BINARY}",
            ("gone.bin", None),
        ),
        (f"diff --git a/data.bin b/data.bin\n{INDEX}\n{BINARY}", ("data.bin", "data.bin")),
        # a mode change repeats one path, quoted or with " b/" inside it
        (
            'diff --git "a/m\\303\\266de.sh" "b/m\\303\\266de.sh"\nold mode 100644\nnew mode 100755\n',
            ("möde.sh", "möde.sh"),
        ),
        ("diff --git a/x b/y.sh b/x b/y.sh\nold mode 100644\nnew mode 100755\n", ("x b/y.sh", "x b/y.sh")),
        # rename and copy records name both paths
        (
            'diff --git "a/caf\\303\\251.bin" b/moved.bin\nsimilarity index 100%\nrename from "caf\\303\\251.bin"\n'
            "rename to moved.bin\n",
            ("café.bin", "moved.bin"),
        ),
        (
            "diff --git a/src.txt b/copy.txt\nsimilarity index 100%\ncopy from src.txt\ncopy to copy.txt\n",
            ("src.txt", "copy.txt"),
        ),
        # anything before the first section names no file
        (f"{INDEX}\n{HUNK}", (None, None)),
    ],
    ids=[
        "utf8",
        "quote",
        "backslash",
        "tab",
        "space",
        "added",
        "new_binary",
        "deleted_binary",
        "changed_binary",
        "mode_quoted",
        "mode_b_slash_in_path",
        "rename",
        "copy",
        "preamble",
    ],
)
def test_section_paths_follow_git_quoting_and_records(section, paths):
    assert patch_section_paths(section) == paths


def test_a_diff_is_keyed_by_each_file_after_the_change():
    diff = (
        content("a/kept.py", "b/kept.py")
        + f"diff --git a/gone.bin b/gone.bin\ndeleted file mode 100644\nindex {OLD}..{NONE}\n{BINARY}"
        + "diff --git a/old.py b/new.py\nsimilarity index 100%\nrename from old.py\nrename to new.py\n"
    )
    files = diff_files(diff)
    assert {path: paths for path, (_section, paths) in files.items()} == {
        "kept.py": ("kept.py",),
        "gone.bin": ("gone.bin",),  # a deleted file is keyed by its path before the change
        "new.py": ("old.py", "new.py"),  # a rename holds both paths
    }
    assert files["kept.py"][0].startswith("a/kept.py b/kept.py\n")
    assert diff_files(None) == diff_files("") == {}
