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

"""Tests for the SWE pivot verifier logic.

These tests exercise the verification functions directly without
requiring the NeMo Gym server infrastructure.
"""

import os
import sys
import types

import pytest


# Stub out nemo_gym so we can import app.py without the full dependency
_nemo_stub = types.ModuleType("nemo_gym")
_nemo_base = types.ModuleType("nemo_gym.base_resources_server")
for cls_name in (
    "BaseResourcesServerConfig",
    "BaseRunRequest",
    "BaseVerifyRequest",
    "BaseVerifyResponse",
    "SimpleResourcesServer",
):
    setattr(_nemo_base, cls_name, type(cls_name, (), {}))
_nemo_stub.base_resources_server = _nemo_base
sys.modules["nemo_gym"] = _nemo_stub
sys.modules["nemo_gym.base_resources_server"] = _nemo_base

# Now we can import app.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import (
    _parse_arguments,
    compute_argument_similarity,
    compute_diff_size_penalty,
    extract_command,
    extract_command_verb,
    extract_edit_content,
    extract_editor_command,
    extract_file_path,
    extract_tool_info,
    get_tool_category,
    verify_target_match,
    verify_tool_name_match,
)


# =========================================================================
# Tool name diversity coverage
# =========================================================================


class TestToolCategoryMapping:
    """Every tool name from the actual dataset must map to a category."""

    # All tool names observed in the 10K difficult_patches.jsonl dataset
    EDIT_TOOLS = [
        "str_replace_editor",
        "text_editor",
        "file_editor",
        "code_editor",
        "edit_file",
        "edit",
        "modify",
        "write",
        "write_file",
        "save_file",
        "create_file",
        "apply_patch",
        "patch_file",
        "apply_diff",
        "apply_changes",
    ]
    BASH_TOOLS = [
        "execute_bash",
        "bash",
        "shell",
        "terminal",
        "run_bash",
        "shell_run",
        "bash_exec",
        "shell_command",
        "shell_cmd",
        "run_shell_cmd",
        "exec_command",
        "exec_cmd",
    ]
    SEARCH_TOOLS = [
        "grep",
        "search_text",
        "find_pattern",
        "text_search",
        "glob",
        "find_files",
        "file_glob",
        "match_files",
        "grep_files",
        "search_files",
        "find_in_files",
        "code_search",
        "list_dir",
        "ls",
        "show_dir",
        "ls_dir",
        "list_directory",
    ]
    READ_TOOLS = [
        "read",
        "read_file",
        "view_file",
        "cat_file",
        "read_file_content",
    ]
    PLANNING_TOOLS = [
        "task_tracker",
        "todo_tracker",
        "plan_tracker",
        "todo_read",
        "todo_write",
        "update_plan",
        "list_tasks",
        "update_tasks",
        "read_todos",
        "write_todos",
        "edit_plan",
        "modify_plan",
    ]
    FINISH_TOOLS = ["finish", "submit"]
    THINK_TOOLS = ["think"]

    @pytest.mark.parametrize("name", EDIT_TOOLS)
    def test_edit_tools(self, name):
        assert get_tool_category(name) == "edit", f"{name} should be 'edit'"

    @pytest.mark.parametrize("name", BASH_TOOLS)
    def test_bash_tools(self, name):
        assert get_tool_category(name) == "bash", f"{name} should be 'bash'"

    @pytest.mark.parametrize("name", SEARCH_TOOLS)
    def test_search_tools(self, name):
        assert get_tool_category(name) == "search", f"{name} should be 'search'"

    @pytest.mark.parametrize("name", READ_TOOLS)
    def test_read_tools(self, name):
        assert get_tool_category(name) == "read", f"{name} should be 'read'"

    @pytest.mark.parametrize("name", PLANNING_TOOLS)
    def test_planning_tools(self, name):
        assert get_tool_category(name) == "planning", f"{name} should be 'planning'"

    @pytest.mark.parametrize("name", FINISH_TOOLS)
    def test_finish_tools(self, name):
        assert get_tool_category(name) == "finish", f"{name} should be 'finish'"

    @pytest.mark.parametrize("name", THINK_TOOLS)
    def test_think_tools(self, name):
        assert get_tool_category(name) == "think", f"{name} should be 'think'"

    def test_unknown_tool(self):
        assert get_tool_category("some_random_tool") == "unknown"

    @pytest.mark.parametrize("name", EDIT_TOOLS + BASH_TOOLS + SEARCH_TOOLS)
    def test_camel_case_variants(self, name):
        """CamelCase versions of tool names should also be recognized."""
        camel = "".join(w.capitalize() for w in name.split("_"))
        cat = get_tool_category(name)
        assert get_tool_category(camel) == cat, f"CamelCase '{camel}' should map to '{cat}'"


# =========================================================================
# Argument parsing
# =========================================================================


class TestArgumentParsing:
    def test_parse_dict(self):
        assert _parse_arguments({"path": "/foo"}) == {"path": "/foo"}

    def test_parse_json_string(self):
        assert _parse_arguments('{"path": "/foo"}') == {"path": "/foo"}

    def test_parse_invalid_string(self):
        assert _parse_arguments("not json") == {}

    def test_parse_none(self):
        assert _parse_arguments(None) == {}

    def test_extract_file_path(self):
        assert extract_file_path({"path": "/workspace/foo.py"}) == "/workspace/foo.py"
        assert extract_file_path({"file_path": "/bar.py"}) == "/bar.py"
        assert extract_file_path({"command": "ls"}) == ""

    def test_extract_command(self):
        assert extract_command({"command": "grep -r 'foo' src/"}) == "grep -r 'foo' src/"
        assert extract_command({"cmd": "ls -la"}) == "ls -la"
        assert extract_command({"path": "/foo"}) == ""

    def test_extract_command_verb(self):
        assert extract_command_verb("grep -r 'foo' src/") == "grep"
        assert extract_command_verb("/usr/bin/python3 test.py") == "python3"
        assert extract_command_verb("cd /workspace && ls") == "cd"
        assert extract_command_verb("") == ""

    def test_extract_editor_command(self):
        assert extract_editor_command({"command": "str_replace"}) == "str_replace"
        assert extract_editor_command({"command": "view"}) == "view"
        assert extract_editor_command({}) == ""

    def test_extract_edit_content_str_replace(self):
        args = {"old_str": "x = 1", "new_str": "x = 2"}
        content = extract_edit_content(args)
        assert "x = 1" in content
        assert "x = 2" in content

    def test_extract_edit_content_patch(self):
        args = {"patch": "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new"}
        assert "old" in extract_edit_content(args)

    def test_extract_edit_content_write(self):
        args = {"content": "print('hello')"}
        assert extract_edit_content(args) == "print('hello')"

    def test_extract_edit_content_empty(self):
        assert extract_edit_content({}) == ""


# =========================================================================
# Level 0: Tool name match
# =========================================================================


class TestToolNameMatch:
    def test_exact_match(self):
        assert verify_tool_name_match("execute_bash", "bash", "execute_bash", "bash")

    def test_category_match_different_names(self):
        """str_replace_editor and code_editor are both 'edit'."""
        assert verify_tool_name_match("str_replace_editor", "edit", "code_editor", "edit")

    def test_category_match_bash_variants(self):
        assert verify_tool_name_match("shell_run", "bash", "execute_bash", "bash")

    def test_mismatch_edit_vs_bash(self):
        assert not verify_tool_name_match("str_replace_editor", "edit", "execute_bash", "bash")

    def test_mismatch_search_vs_edit(self):
        assert not verify_tool_name_match("grep", "search", "edit", "edit")

    def test_unknown_categories_dont_match(self):
        """Two unknown tools should NOT match by category."""
        assert not verify_tool_name_match("tool_a", "unknown", "tool_b", "unknown")

    def test_same_unknown_tool_matches(self):
        """Same exact name should match even if unknown category."""
        assert verify_tool_name_match("custom_tool", "unknown", "custom_tool", "unknown")


# =========================================================================
# Level 1: Target match
# =========================================================================


class TestTargetMatch:
    def test_edit_same_file(self):
        r_args = {"path": "/workspace/project/src/foo.py", "command": "str_replace"}
        e_args = {"path": "/workspace/project/src/foo.py", "command": "str_replace"}
        assert verify_target_match("edit", r_args, "edit", e_args)

    def test_edit_same_basename_different_path(self):
        """Basename match — different workspace prefix but same file."""
        r_args = {"path": "/workspace_v2/src/foo.py"}
        e_args = {"path": "/workspace/src/foo.py"}
        assert verify_target_match("edit", r_args, "edit", e_args)

    def test_edit_different_file(self):
        """P1 consumer-vs-producer: editing the wrong file."""
        r_args = {"path": "/workspace/views.py", "command": "str_replace"}
        e_args = {"path": "/workspace/models.py", "command": "str_replace"}
        assert not verify_target_match("edit", r_args, "edit", e_args)

    def test_edit_different_editor_command(self):
        """'view' vs 'str_replace' on same file should fail."""
        r_args = {"path": "/workspace/foo.py", "command": "view"}
        e_args = {"path": "/workspace/foo.py", "command": "str_replace"}
        assert not verify_target_match("edit", r_args, "edit", e_args)

    def test_bash_same_verb_same_path(self):
        r_args = {"command": "grep -r 'pattern' src/utils/"}
        e_args = {"command": "grep -n 'pattern' src/utils/"}
        assert verify_target_match("bash", r_args, "bash", e_args)

    def test_bash_same_verb_different_path(self):
        """grep on different directories should fail."""
        r_args = {"command": "grep -r 'pattern' src/views/"}
        e_args = {"command": "grep -r 'pattern' src/models/"}
        assert not verify_target_match("bash", r_args, "bash", e_args)

    def test_bash_same_verb_no_path(self):
        """If no paths extractable, verb match alone is enough."""
        r_args = {"command": "grep -r 'pattern'"}
        e_args = {"command": "grep -n 'pattern'"}
        assert verify_target_match("bash", r_args, "bash", e_args)

    def test_bash_cd_same_dir(self):
        r_args = {"command": "cd /workspace/project && ls"}
        e_args = {"command": "cd /workspace/project && git status"}
        assert verify_target_match("bash", r_args, "bash", e_args)

    def test_bash_cd_different_dir(self):
        r_args = {"command": "cd /workspace/frontend && ls"}
        e_args = {"command": "cd /workspace/backend && ls"}
        assert not verify_target_match("bash", r_args, "bash", e_args)

    def test_bash_different_verb(self):
        """grep vs pytest are different operations."""
        r_args = {"command": "grep -r 'foo' ."}
        e_args = {"command": "pytest tests/"}
        assert not verify_target_match("bash", r_args, "bash", e_args)

    def test_search_same_target(self):
        r_args = {"path": "/workspace/src"}
        e_args = {"path": "/workspace/src"}
        assert verify_target_match("search", r_args, "search", e_args)

    def test_view_range_full_overlap(self):
        """Rollout covers entire expected range — passes."""
        r_args = {"command": "view", "path": "/foo.py", "view_range": [190, 260]}
        e_args = {"command": "view", "path": "/foo.py", "view_range": [200, 250]}
        assert verify_target_match("edit", r_args, "edit", e_args)

    def test_view_range_exact_match(self):
        r_args = {"command": "view", "path": "/foo.py", "view_range": [200, 250]}
        e_args = {"command": "view", "path": "/foo.py", "view_range": [200, 250]}
        assert verify_target_match("edit", r_args, "edit", e_args)

    def test_view_range_80_percent_overlap(self):
        """Rollout covers exactly 80% of expected range — passes."""
        # Expected: [200, 250] = 50 lines. Need 40 lines overlap.
        # Rollout: [210, 260] -> overlap [210, 250] = 40 lines = 80%
        r_args = {"command": "view", "path": "/foo.py", "view_range": [210, 260]}
        e_args = {"command": "view", "path": "/foo.py", "view_range": [200, 250]}
        assert verify_target_match("edit", r_args, "edit", e_args)

    def test_view_range_below_80_percent(self):
        """Rollout covers less than 80% of expected range — fails."""
        # Expected: [200, 250] = 50 lines. Need 40 lines overlap.
        # Rollout: [240, 290] -> overlap [240, 250] = 10 lines = 20%
        r_args = {"command": "view", "path": "/foo.py", "view_range": [240, 290]}
        e_args = {"command": "view", "path": "/foo.py", "view_range": [200, 250]}
        assert not verify_target_match("edit", r_args, "edit", e_args)

    def test_view_range_no_overlap(self):
        """Completely disjoint ranges — fails."""
        r_args = {"command": "view", "path": "/foo.py", "view_range": [1, 50]}
        e_args = {"command": "view", "path": "/foo.py", "view_range": [200, 250]}
        assert not verify_target_match("edit", r_args, "edit", e_args)

    def test_view_range_missing_rollout(self):
        """Rollout has no view_range (views whole file) — passes."""
        r_args = {"command": "view", "path": "/foo.py"}
        e_args = {"command": "view", "path": "/foo.py", "view_range": [200, 250]}
        assert verify_target_match("edit", r_args, "edit", e_args)

    def test_view_range_missing_expected(self):
        """Expected has no view_range — passes."""
        r_args = {"command": "view", "path": "/foo.py", "view_range": [200, 250]}
        e_args = {"command": "view", "path": "/foo.py"}
        assert verify_target_match("edit", r_args, "edit", e_args)

    def test_view_range_both_missing(self):
        """Neither has view_range — passes."""
        r_args = {"command": "view", "path": "/foo.py"}
        e_args = {"command": "view", "path": "/foo.py"}
        assert verify_target_match("edit", r_args, "edit", e_args)

    def test_view_range_different_file_still_fails(self):
        """Even with good view_range overlap, wrong file fails."""
        r_args = {"command": "view", "path": "/bar.py", "view_range": [200, 250]}
        e_args = {"command": "view", "path": "/foo.py", "view_range": [200, 250]}
        assert not verify_target_match("edit", r_args, "edit", e_args)

    def test_view_range_string_format(self):
        """view_range as JSON string should also work."""
        r_args = {"command": "view", "path": "/foo.py", "view_range": "[200, 250]"}
        e_args = {"command": "view", "path": "/foo.py", "view_range": [200, 250]}
        assert verify_target_match("edit", r_args, "edit", e_args)

    def test_no_file_info_passes(self):
        """If we can't extract file info, pass (don't penalize)."""
        assert verify_target_match("edit", {}, "edit", {})

    def test_cross_category_passes(self):
        """Different categories always pass target match (handled by level 0)."""
        assert verify_target_match("edit", {"path": "a.py"}, "bash", {"command": "ls"})


# =========================================================================
# Level 2: Argument similarity
# =========================================================================


class TestArgumentSimilarity:
    def test_identical_edit(self):
        args = {"old_str": "x = 1", "new_str": "x = 2"}
        sim = compute_argument_similarity("edit", args, "edit", args)
        assert sim == 1.0

    def test_similar_edit(self):
        r_args = {"old_str": "x = 1", "new_str": "x = 2"}
        e_args = {"old_str": "x = 1", "new_str": "x = 3"}
        sim = compute_argument_similarity("edit", r_args, "edit", e_args)
        assert 0.5 < sim < 1.0

    def test_completely_different_edit(self):
        r_args = {"old_str": "import os", "new_str": "import os\nimport sys\nimport json\nfrom pathlib import Path"}
        e_args = {"old_str": "return None", "new_str": "return self.value"}
        sim = compute_argument_similarity("edit", r_args, "edit", e_args)
        assert sim < 0.5

    def test_identical_bash(self):
        args = {"command": "pytest tests/test_foo.py -v"}
        sim = compute_argument_similarity("bash", args, "bash", args)
        assert sim == 1.0

    def test_similar_bash(self):
        r_args = {"command": "pytest tests/test_foo.py -v"}
        e_args = {"command": "pytest tests/test_foo.py -xvs"}
        sim = compute_argument_similarity("bash", r_args, "bash", e_args)
        assert sim > 0.7

    def test_non_edit_bash_returns_1(self):
        """For non-edit/bash categories, similarity is always 1.0."""
        assert compute_argument_similarity("search", {}, "search", {}) == 1.0
        assert compute_argument_similarity("read", {}, "read", {}) == 1.0

    def test_one_empty_edit_content(self):
        """One has content, other doesn't — should be 0."""
        r_args = {"old_str": "foo", "new_str": "bar"}
        e_args = {}
        sim = compute_argument_similarity("edit", r_args, "edit", e_args)
        assert sim == 0.0

    def test_both_empty_edit(self):
        sim = compute_argument_similarity("edit", {}, "edit", {})
        assert sim == 1.0


# =========================================================================
# Level 4: Diff-size shaping
# =========================================================================


class TestDiffSizeShaping:
    def test_non_edit_no_penalty(self):
        assert compute_diff_size_penalty("bash", {"command": "ls"}, 0.1) == 0.0

    def test_empty_edit_no_penalty(self):
        assert compute_diff_size_penalty("edit", {}, 0.1) == 0.0

    def test_small_edit_small_penalty(self):
        args = {"old_str": "x", "new_str": "y"}
        penalty = compute_diff_size_penalty("edit", args, 0.1)
        assert 0 < penalty < 0.01  # very small content

    def test_large_edit_larger_penalty(self):
        args = {"old_str": "x", "new_str": "y" * 600}
        penalty = compute_diff_size_penalty("edit", args, 0.1)
        assert penalty == pytest.approx(0.1, abs=0.01)  # capped at alpha

    def test_alpha_scales_penalty(self):
        args = {"old_str": "x", "new_str": "y" * 250}
        p1 = compute_diff_size_penalty("edit", args, 0.1)
        p2 = compute_diff_size_penalty("edit", args, 0.2)
        assert p2 > p1  # higher alpha = higher penalty


# =========================================================================
# Integration: Full verification flow
# =========================================================================


class TestFullVerificationFlow:
    """End-to-end tests simulating the layered verification."""

    def _verify(
        self,
        rollout_name,
        rollout_args,
        expected_name,
        expected_args,
        enable_target=True,
        enable_similarity=True,
        enable_diff_size=True,
        sim_threshold=0.4,
        sim_full=0.8,
        alpha=0.1,
    ):
        """Simulate the verify flow from app.py without the server."""
        r_name, r_cat, r_args = rollout_name, get_tool_category(rollout_name), _parse_arguments(rollout_args)
        e_name, e_cat, e_args = expected_name, get_tool_category(expected_name), _parse_arguments(expected_args)

        # Level 0
        if not verify_tool_name_match(r_name, r_cat, e_name, e_cat):
            return 0.0, "tool_name_mismatch"

        reward = 1.0

        # Level 1
        if enable_target:
            if not verify_target_match(r_cat, r_args, e_cat, e_args):
                return 0.0, "target_mismatch"

        # Level 2
        if enable_similarity:
            sim = compute_argument_similarity(r_cat, r_args, e_cat, e_args)
            if sim >= sim_full:
                reward = 1.0
            elif sim >= sim_threshold:
                reward = 0.5 + 0.5 * (sim - sim_threshold) / (sim_full - sim_threshold)
            else:
                return 0.0, "similarity_below_threshold"

        # Level 4
        if enable_diff_size:
            penalty = compute_diff_size_penalty(r_cat, r_args, alpha)
            reward *= 1.0 - penalty

        return reward, "none"

    def test_perfect_match(self):
        """Identical action should get near-1.0 reward."""
        args = {"command": "str_replace", "path": "/foo.py", "old_str": "x=1", "new_str": "x=2"}
        reward, reason = self._verify("str_replace_editor", args, "str_replace_editor", args)
        assert reward > 0.9
        assert reason == "none"

    def test_wrong_tool_type(self):
        """Edit when expected bash should get 0."""
        reward, reason = self._verify(
            "str_replace_editor",
            {"path": "/foo.py"},
            "execute_bash",
            {"command": "pytest"},
        )
        assert reward == 0.0
        assert reason == "tool_name_mismatch"

    def test_diversified_tool_names_match(self):
        """code_editor rollout vs str_replace_editor expected should match."""
        args = {"command": "str_replace", "path": "/foo.py", "old_str": "a", "new_str": "b"}
        reward, reason = self._verify("code_editor", args, "str_replace_editor", args)
        assert reward > 0.9
        assert reason == "none"

    def test_p1_wrong_file(self):
        """Consumer-vs-producer: editing views.py instead of models.py."""
        r_args = {"command": "str_replace", "path": "/workspace/views.py", "old_str": "a", "new_str": "b"}
        e_args = {"command": "str_replace", "path": "/workspace/models.py", "old_str": "c", "new_str": "d"}
        reward, reason = self._verify("str_replace_editor", r_args, "str_replace_editor", e_args)
        assert reward == 0.0
        assert reason == "target_mismatch"

    def test_p4_overengineered_edit(self):
        """Over-engineering: correct file but bloated content."""
        e_args = {"command": "str_replace", "path": "/foo.py", "old_str": "x", "new_str": "y"}
        r_args = {
            "command": "str_replace",
            "path": "/foo.py",
            "old_str": "x",
            "new_str": "y" * 200 + "\n# lots of extra code",
        }
        reward, reason = self._verify("str_replace_editor", r_args, "str_replace_editor", e_args)
        # Should get low reward: similarity is low AND diff-size penalty
        assert reward < 0.5

    def test_p6_execution_error_similar(self):
        """Execution error: right file, slightly wrong content."""
        e_args = {"command": "str_replace", "path": "/foo.py", "old_str": "if x > 0:", "new_str": "if x >= 0:"}
        r_args = {"command": "str_replace", "path": "/foo.py", "old_str": "if x > 0:", "new_str": "if x > 0:  # fixed"}
        reward, reason = self._verify("str_replace_editor", r_args, "str_replace_editor", e_args)
        # Partial credit — similar but not identical
        assert 0.3 < reward < 1.0

    def test_bash_grep_same_path(self):
        """Same grep command on same path should get high reward."""
        r_args = {"command": "grep -rn 'pattern' src/"}
        e_args = {"command": "grep -r 'pattern' src/"}
        reward, reason = self._verify("execute_bash", r_args, "execute_bash", e_args)
        assert reward > 0.8

    def test_bash_grep_different_path(self):
        """grep on different directory should fail at target match."""
        r_args = {"command": "grep -r 'foo' src/views/"}
        e_args = {"command": "grep -r 'foo' src/models/"}
        reward, reason = self._verify("execute_bash", r_args, "execute_bash", e_args)
        assert reward == 0.0
        assert reason == "target_mismatch"

    def test_bash_grep_vs_pytest(self):
        """grep vs pytest should fail at target match (different verb)."""
        r_args = {"command": "grep -r 'foo' ."}
        e_args = {"command": "pytest tests/ -v"}
        reward, reason = self._verify("execute_bash", r_args, "execute_bash", e_args)
        assert reward == 0.0
        assert reason == "target_mismatch"

    def test_no_target_match_still_checks_similarity(self):
        """With target match disabled, wrong file can still get partial credit from similarity."""
        e_args = {"command": "str_replace", "path": "/a.py", "old_str": "x=1", "new_str": "x=2"}
        r_args = {"command": "str_replace", "path": "/b.py", "old_str": "x=1", "new_str": "x=2"}
        reward, reason = self._verify(
            "str_replace_editor",
            r_args,
            "str_replace_editor",
            e_args,
            enable_target=False,
        )
        # Same content, different file — similarity is high
        assert reward > 0.8

    def test_diff_size_disabled(self):
        """With diff-size disabled, large edits still get full credit if similar."""
        args = {"command": "str_replace", "path": "/foo.py", "old_str": "x", "new_str": "x" * 400}
        reward_with, _ = self._verify("str_replace_editor", args, "str_replace_editor", args, enable_diff_size=True)
        reward_without, _ = self._verify(
            "str_replace_editor", args, "str_replace_editor", args, enable_diff_size=False
        )
        assert reward_without >= reward_with
        assert reward_without == 1.0


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    def test_empty_arguments(self):
        """Empty args on both sides should not crash."""
        assert verify_target_match("edit", {}, "edit", {})
        assert compute_argument_similarity("edit", {}, "edit", {}) == 1.0
        assert compute_diff_size_penalty("edit", {}, 0.1) == 0.0

    def test_arguments_as_json_string(self):
        """Tool call arguments may arrive as a JSON string."""
        tc = {"name": "str_replace_editor", "arguments": '{"command": "view", "path": "/foo.py"}'}
        name, cat, args = extract_tool_info(tc)
        assert name == "str_replace_editor"
        assert cat == "edit"
        assert args["path"] == "/foo.py"

    def test_arguments_as_dict(self):
        tc = {"name": "execute_bash", "arguments": {"command": "ls -la"}}
        name, cat, args = extract_tool_info(tc)
        assert cat == "bash"
        assert args["command"] == "ls -la"

    def test_file_path_with_spaces(self):
        args = {"path": "  /workspace/foo.py  "}
        assert extract_file_path(args) == "/workspace/foo.py"

    def test_apply_patch_content(self):
        args = {"patch": "--- a/f.py\n+++ b/f.py\n-old\n+new"}
        content = extract_edit_content(args)
        assert "old" in content and "new" in content

    def test_similarity_capped_on_large_content(self):
        """Large edits should not cause O(n*m) blowup."""
        import time

        r_args = {"old_str": "x" * 50000, "new_str": "y" * 50000}
        e_args = {"old_str": "x" * 50000, "new_str": "z" * 50000}
        start = time.time()
        sim = compute_argument_similarity("edit", r_args, "edit", e_args)
        elapsed = time.time() - start
        assert elapsed < 1.0, f"Took {elapsed:.1f}s — cap not working"
        assert 0.0 <= sim <= 1.0

    def test_path_extraction_from_bash_command(self):
        from app import _extract_paths_from_command

        paths = _extract_paths_from_command("grep -r 'foo' /workspace/src/models/")
        assert any("models" in p for p in paths)

    def test_path_extraction_relative(self):
        from app import _extract_paths_from_command

        paths = _extract_paths_from_command("cat ./src/utils/helpers.py")
        assert any("helpers.py" in p for p in paths)

    def test_path_extraction_no_path(self):
        from app import _extract_paths_from_command

        paths = _extract_paths_from_command("echo hello world")
        assert paths == []
