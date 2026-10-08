# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Guardwords check reports locations without printing matched values."""

import sys

from scripts.ci import check_guardwords
from scripts.ci.check_guardwords import find_added_matches


def test_finds_added_match_and_returns_only_its_location() -> None:
    diff = """diff --git a/example.py b/example.py
--- a/example.py
+++ example.py
@@ -1 +1,2 @@
 unchanged
+value = 'sensitive-marker'
"""

    locations = find_added_matches(diff, ["sensitive-marker"], case_sensitive=False)

    assert locations == ["example.py:2"]
    assert all("sensitive-marker" not in location for location in locations)


def test_ignores_deleted_matches() -> None:
    diff = """diff --git a/example.py b/example.py
--- a/example.py
+++ example.py
@@ -1 +0,0 @@
-sensitive-marker
"""

    assert find_added_matches(diff, ["sensitive-marker"], case_sensitive=False) == []


def test_case_insensitive_literal_matching() -> None:
    diff = """diff --git a/example.py b/example.py
--- a/example.py
+++ example.py
@@ -0,0 +1 @@
+VALUE = 'Sensitive-Marker'
"""

    assert find_added_matches(diff, ["sensitive-marker"], case_sensitive=False) == ["example.py:1"]


def test_cli_prints_only_location_for_a_match(tmp_path, monkeypatch, capsys) -> None:
    patterns_file = tmp_path / "guardwords.yaml"
    patterns_file.write_text(
        "version: 1\nmatching:\n  mode: literal\n  case_sensitive: false\n"
        "patterns:\n  - id: synthetic\n    category: test\n    pattern: sensitive-marker\n",
        encoding="utf-8",
    )
    diff = """diff --git example.py example.py
--- example.py
+++ example.py
@@ -1,0 +2 @@
+value = 'sensitive-marker'
"""
    monkeypatch.setattr(check_guardwords, "_git_diff", lambda _base, _head: diff)
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_guardwords.py", "--patterns", str(patterns_file), "--base", "base", "--head", "head"],
    )

    assert check_guardwords.main() == 1
    output = capsys.readouterr()
    assert output.out == "example.py:2\n"
    assert output.err == ""
