# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from resources_servers.swerl_constrained.eval.constraints import (
    check_minimal_editing,
    check_no_hardcoded_secrets,
    run_constraints,
)


SMALL_PATCH = """\
diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,3 @@
-raise RuntimeError("not installed")
+raise ImportError("not installed")
"""

BIG_PATCH = "\n".join(
    ["diff --git a/foo.py b/foo.py", "--- a/foo.py", "+++ b/foo.py", "@@ -1,100 +1,100 @@"]
    + [f"+line {i}" for i in range(150)]
)

GOLDEN_PATCH = """\
diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,3 @@
-raise RuntimeError("not installed")
+raise ImportError("not installed")
"""

SECRET_PATCH = """\
diff --git a/config.py b/config.py
--- a/config.py
+++ b/config.py
@@ -1 +1,2 @@
+API_KEY = "sk-abcdefgh12345678abcdefgh12345678"
"""

ENV_PATCH = """\
diff --git a/config.py b/config.py
--- a/config.py
+++ b/config.py
@@ -1 +1,2 @@
+API_KEY = os.environ["API_KEY"]
"""


class TestMinimalEditing:
    def test_same_size_as_golden(self):
        score, detail = check_minimal_editing(SMALL_PATCH, GOLDEN_PATCH)
        assert score == 1.0
        assert detail["ratio"] <= 2.0

    def test_large_patch_vs_small_golden(self):
        score, detail = check_minimal_editing(BIG_PATCH, GOLDEN_PATCH)
        assert score == 0.0

    def test_no_golden_small_patch(self):
        score, _ = check_minimal_editing(SMALL_PATCH, None)
        assert score == 1.0

    def test_no_golden_huge_patch(self):
        huge = "\n".join([f"+line {i}" for i in range(300)])
        score, _ = check_minimal_editing(huge, None)
        assert score == 0.0

    def test_none_patch(self):
        score, _ = check_minimal_editing(None, None)
        assert score == 1.0

    def test_mid_range_score_between_zero_and_one(self):
        mid_patch = "\n".join([f"+line {i}" for i in range(6)])
        golden = "\n".join([f"+line {i}" for i in range(2)])
        score, detail = check_minimal_editing(mid_patch, golden)
        assert 0.0 < score < 1.0


class TestNoHardcodedSecrets:
    def test_clean_patch(self):
        score, detail = check_no_hardcoded_secrets(SMALL_PATCH, None)
        assert score == 1.0
        assert detail["violations"] == []

    def test_secret_in_added_line(self):
        score, detail = check_no_hardcoded_secrets(SECRET_PATCH, None)
        assert score == 0.0
        assert len(detail["violations"]) > 0

    def test_env_var_read_not_flagged(self):
        score, detail = check_no_hardcoded_secrets(ENV_PATCH, None)
        assert score == 1.0
        assert detail["violations"] == []

    def test_none_patch(self):
        score, _ = check_no_hardcoded_secrets(None, None)
        assert score == 1.0


class TestRunConstraints:
    def test_all_known(self):
        results = run_constraints(["minimal_editing", "no_hardcoded_secrets"], SMALL_PATCH, GOLDEN_PATCH)
        assert set(results.keys()) == {"minimal_editing", "no_hardcoded_secrets"}
        for r in results.values():
            assert r["score"] is not None

    def test_unknown_constraint(self):
        results = run_constraints(["does_not_exist"], SMALL_PATCH, None)
        assert results["does_not_exist"]["score"] is None
        assert "error" in results["does_not_exist"]["detail"]

    def test_empty_constraints(self):
        results = run_constraints([], SMALL_PATCH, None)
        assert results == {}
