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
"""Answer separation: tests can't be gamed via immutable-file edits, and the ground truth is never
staged into the agent VM."""

import pytest
from patch_guard import strip_immutable


SRC = ("diff --git a/src/foo.c b/src/foo.c\n"
       "--- a/src/foo.c\n+++ b/src/foo.c\n@@ -1 +1 @@\n-old\n+new\n")
TESTS = ("diff --git a/tests/t.c b/tests/t.c\n"
         "--- a/tests/t.c\n+++ b/tests/t.c\n@@ -1 +1 @@\n-old\n+new\n")


def test_immutable_section_is_dropped():
    patch, dropped = strip_immutable(SRC + TESTS, ["tests/"])
    assert "src/foo.c" in patch and "tests/t.c" not in patch
    assert any("tests/t.c" in d for d in dropped)


def test_clean_patch_kept_intact():
    patch, dropped = strip_immutable(SRC, ["tests/"])
    assert dropped == [] and "src/foo.c" in patch


def test_path_escape_rejected():
    with pytest.raises(ValueError):
        strip_immutable(SRC.replace("src/foo.c", "../etc/passwd"), [])


def test_unsupported_format_rejected():
    # A context diff would slip past the stripper yet still apply via `patch` -- reject it outright.
    with pytest.raises(ValueError):
        strip_immutable("*** 1,3 ****\n--- 1,3 ----\n", [])


def test_ground_truth_paths_never_enter_agent_vm():
    import template
    for p in ("/data/poc.bin", "/src/poc.bin", "/src/crash.log", "/data/patch.diff"):
        assert p in template.FORBIDDEN


def test_agent_poc_uses_its_own_path_not_ground_truth():
    # score uploads the agent's PoC to /data/agent_poc.bin; /data/poc.bin is the ground truth (stage 4),
    # so the agent's artifact must never be written there (it would make stage 4 re-test the agent PoC).
    import score
    text = open(score.__file__, encoding="utf-8").read()
    assert "/data/agent_poc.bin" in text and 'inputs["/data/poc.bin"]' not in text
