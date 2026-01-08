# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import base64
import difflib
import json
import re
from typing import Dict, List, Optional, Tuple, TypedDict

SEARCH_REPLACE_REGEX = (
    r"```.*?\n### (.*)\s*<<<<<<< SEARCH\n([\s\S]*?)\n=======\n([\s\S]*?)\n>>>>>>> REPLACE\n*```"
)
# Regular expression pattern to match ```python\n{text}\n```
PYTHON_BLOCK_PATTERN = r"```python\n(.*?)\n```"
THINK_START = "<think>"
THINK_END = "</think>"
ANSWER_START = "<solution>"
ANSWER_END = "</solution>"

class FormatError(Exception):
    """Raised when the search/replace format is invalid."""


class FormatSolutionError(Exception):
    """Raised when the <solution>...</solution> block is missing or malformed."""
    
class ChangeSimilarity(TypedDict):
    path: str
    pred_change: str
    oracle_change: str
    similarity: float


def create_patch_from_code(python_code: str, test_id: int = 0) -> str:
    """Wrap a Python snippet into a git-style patch for ``reproduce_bug_{id}.py``."""
    patch_header = f"""diff --git a/reproduce_bug_{test_id}.py b/reproduce_bug_{test_id}.py
new file mode 100644
index 0000000..e69de29
"""
    patch_body: list[str] = []
    patch_body.append("--- /dev/null")
    patch_body.append(f"+++ b/reproduce_bug_{test_id}.py")

    code_lines = python_code.split("\n")
    patch_body.append(f"@@ -0,0 +1,{len(code_lines)} @@")

    for line in code_lines:
        patch_body.append(f"+{line}")

    return patch_header + "\n".join(patch_body) + "\n"



def extract_python_blocks(text: str) -> list[str]:
    """Extract Python code blocks from the given text."""
    python_blocks = re.findall(PYTHON_BLOCK_PATTERN, text, re.DOTALL)
    if python_blocks:
        return python_blocks

    # Fallback pattern for shebang-style scripts wrapped between dashed lines.
    pattern = re.compile(
        r"""
        ^-+\s*                      # Line with only dashes
        \n                          # Newline
        (                           # Start capture group for the code block
            \#\!/usr/bin/env\ python.*?   # Shebang line and rest of the code (non-greedy)
            (?=                    # Lookahead to find where to stop:
                \n^-+\s*$          # Either another line of dashes
                |                  # OR
                \Z                # End of string
            )
        )
        """,
        re.MULTILINE | re.DOTALL | re.VERBOSE,
    )

    match = pattern.search(text)
    return [match.group(1).strip().replace("#!/usr/bin/env python", "")] if match else []


def parse_search_replace(text: str) -> dict[str, list[tuple[str, str]]]:
    """Parse SEARCH/REPLACE blocks into a mapping of path -> list[(search, replace)]."""
    path_search_replaces: list[tuple[str, str, str]] = re.findall(SEARCH_REPLACE_REGEX, text)
    path_search_replace_dict: dict[str, list[tuple[str, str]]] = {}
    for path, search, replace in path_search_replaces:
        search_replace_pair = (search, replace)
        path_list = path_search_replace_dict.setdefault(path, [])
        if search_replace_pair not in path_list:
            path_list.append(search_replace_pair)
    return path_search_replace_dict

def parse_git_patch(patch_text: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse an oracle patch in diff format and convert it to search/replace format.

    Args:
        patch_text: The diff patch text in standard git diff format

    Returns:
        A dictionary mapping file paths to lists of (search, replace) pairs
        Compatible with the apply_code_change function
    """
    result = {}

    # Split the patch into file sections
    file_sections = re.split(r'^diff --git', patch_text, flags=re.MULTILINE)

    for section in file_sections:
        if not section.strip():
            continue

        # Extract file path from the +++ line
        file_path_match = re.search(r'^\+\+\+ (?:b/)?(.+)$', section, re.MULTILINE)
        if not file_path_match:
            continue

        file_path = file_path_match.group(1)

        # Find all hunks in this file - FIXED: Use a more robust pattern
        hunk_pattern = r'^(@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@[^\n]*\n)((?:(?!^@@|^diff --git).*\n?)*)'
        hunk_matches = re.findall(hunk_pattern, section, re.MULTILINE)

        hunks = []
        for match in hunk_matches:
            header, old_start, old_count, new_start, new_count, content = match
            hunks.append((old_start, old_count, new_start, new_count, content))

        search_replace_pairs = []

        for hunk in hunks:
            old_start, old_count, new_start, new_count, hunk_content = hunk

            # Parse the hunk content into lines with their prefixes
            lines = hunk_content.rstrip('\n').split('\n') if hunk_content.strip() else []

            # Group consecutive changes together
            change_groups = []
            current_group = []

            for line in lines:
                if line.startswith(('-', '+')):
                    current_group.append(line)
                else:
                    if current_group:
                        change_groups.append(current_group)
                        current_group = []
                    # This is a context line
                    if change_groups:
                        # Add context to the last group
                        change_groups[-1].append(line)
                    else:
                        # Start a new group with context
                        current_group = [line]

            if current_group:
                change_groups.append(current_group)

            # Process each change group
            for group in change_groups:
                if not any(line.startswith(('-', '+')) for line in group):
                    continue  # Skip groups with only context

                # Find the context before changes
                context_before = []
                change_start_idx = 0
                for i, line in enumerate(group):
                    if line.startswith(('-', '+')):
                        change_start_idx = i
                        break
                    # Remove prefix only if it's a space (context line prefix)
                    context_before.append(line[1:] if line.startswith(' ') else line)

                # Find the context after changes
                context_after = []
                change_end_idx = len(group)
                for i in range(len(group) - 1, -1, -1):
                    if group[i].startswith(('-', '+')):
                        change_end_idx = i + 1
                        break
                    # Remove prefix only if it's a space (context line prefix)
                    context_after.insert(0, group[i][1:] if group[i].startswith(' ') else group[i])

                # Process the actual changes
                deleted_lines = []
                added_lines = []

                for i in range(change_start_idx, change_end_idx):
                    line = group[i]
                    if line.startswith('-'):
                        deleted_lines.append(line[1:])
                    elif line.startswith('+'):
                        added_lines.append(line[1:])
                    else:
                        # Context line in the middle of changes
                        # Remove prefix only if it's a space
                        content = line[1:] if line.startswith(' ') else line
                        deleted_lines.append(content)
                        added_lines.append(content)

                # Build search and replace content
                search_content = context_before + deleted_lines + context_after
                replace_content = context_before + added_lines + context_after

                # Only create a search/replace pair if there are actual changes
                if deleted_lines != added_lines:
                    search_text = '\n'.join(search_content)
                    replace_text = '\n'.join(replace_content)
                    search_replace_pairs.append((search_text, replace_text))

        if search_replace_pairs:
            result[file_path] = search_replace_pairs

    return result

def get_search_replace_pairs(patch):
    search_replace_pairs = parse_git_patch(patch)
    search_replace_diff_list = []
    for file_path, pairs in search_replace_pairs.items():
        for search, replace in pairs:
            search_replace_diff_list.append(f'```python\n### {file_path}\n')
            search_replace_diff_list.append(f'<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n')
            search_replace_diff_list.append(f'```\n')
    search_replace_diff_str = '\n'.join(search_replace_diff_list)
    return search_replace_pairs, search_replace_diff_str

def apply_code_change(
    code_context: dict[str, str],
    search_replace_dict: dict[str, list[tuple[str, str]]],
    silent: bool = False,
) -> dict[str, str]:
    """Apply search/replace edits to the original code context.

    Edits are applied in a stable order to avoid interactions between multiple
    replacements applied to the same file.
    """
    new_content_dict: dict[str, str] = {}
    for path, search_replaces in search_replace_dict.items():
        original_content = "\n" + code_context.get(path, "")
        if not original_content:
            continue

        positioned_operations: list[tuple[int, str, str]] = []

        for search, replace in search_replaces:
            if not silent and len(search) == len(replace) and search == replace:
                raise FormatError("Search and replace blocks are identical")

            search_with_newline = "\n" + search
            if search_with_newline not in original_content:
                if not silent:
                    raise FormatError(f"Search block not found in the code: {search}")
                continue

            start_pos = original_content.find(search_with_newline)
            positioned_operations.append((start_pos, search, replace))

        # Apply from end to beginning so earlier replacements don't shift positions.
        positioned_operations.sort(key=lambda x: x[0], reverse=True)

        for start_pos, search, replace in positioned_operations:
            search_with_newline = "\n" + search
            replace_with_newline = "\n" + replace

            before = original_content[:start_pos]
            after = original_content[start_pos + len(search_with_newline) :]
            original_content = before + replace_with_newline + after

        new_content_dict[path] = original_content[1:]

    return new_content_dict

def generate_git_diff(
    code_context: dict[str, str],
    new_content_dict: dict[str, str],
    remove_repo_name: bool = False,
) -> tuple[str, dict[str, str]]:
    """Generate git-style patches for each modified file.

    Returns:
        A tuple of (full_patch_str, per_file_patch_dict[path] -> patch_without_header).
    """

    def generate_unified_diff(
        old_code: str,
        new_code: str,
        file_path: str,
        n_context: int = 3,
    ) -> str:
        old_file_git = f"a/{file_path}"
        new_file_git = f"b/{file_path}"
        original_lines = old_code.splitlines()
        modified_lines = new_code.splitlines()

        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile="old",
            tofile="new",
            lineterm="",
            n=n_context,
        )
        diff_list = list(diff)
        if not diff_list:
            return ""
        diff_header = f"diff --git {old_file_git} {new_file_git}\n"
        diff_content = "\n".join(diff_list)
        return f"{diff_header}{diff_content}\n"

    diffs: list[str] = []
    diffs_dict: dict[str, str] = {}
    for path, new_content in new_content_dict.items():
        old_content = code_context.get(path, "")
        # For SWE-bench-Verified, paths contain the repo name; images do not.
        if remove_repo_name:
            patch = generate_unified_diff(old_content, new_content, "/".join(path.split("/")[1:]))
        else:
            patch = generate_unified_diff(old_content, new_content, path)
        if not patch:
            continue
        diffs.append(patch)
        diffs_dict[path] = "\n".join(patch.split("\n")[2:])
    return "\n".join(diffs), diffs_dict


def calculate_reward(
    oracle_patch: dict[str, str] | None = None,
    pred_patch: dict[str, str] | None = None,
    scale_factor: float = 1.0,
) -> tuple[float, dict]:
    """Compute a similarity-based reward between oracle and predicted patches."""
    oracle_patch = oracle_patch or {}
    pred_patch = pred_patch or {}

    all_file_paths = set(oracle_patch.keys()).union(set(pred_patch.keys()))
    similarities: list[ChangeSimilarity] = []
    for path in all_file_paths:
        pred_change = pred_patch.get(path, "")
        oracle_change = oracle_patch.get(path, "")
        if oracle_change == "" or pred_change == "":
            change_similarity = 0.0
        else:
            change_similarity = difflib.SequenceMatcher(
                None,
                pred_change,
                oracle_change,
                autojunk=False,
            ).ratio()
        similarities.append(
            ChangeSimilarity(
                path=path,
                pred_change=pred_change,
                oracle_change=oracle_change,
                similarity=change_similarity,
            )
        )

    if not similarities:
        # Both patches empty → identical, maximal reward.
        return 1.0 * scale_factor, dict(similarities=[])

    reward = sum(s["similarity"] for s in similarities) / len(similarities) * scale_factor
    return reward, dict(similarities=similarities)


def extract_pred_patch(
    code_context: dict[str, str],
    text_output: str,
    remove_repo_name: bool = False,
) -> Optional[dict]:
    """
    Extracts the predicted patch and its dict from the model output if possible.
    Returns (pred_patch, pred_patch_dict) or None if extraction fails.
    """
    # Extract the <solution>...</solution> block (ignore any <think>...</think>).
    if ANSWER_START not in text_output or ANSWER_END not in text_output:
        return None
    if THINK_START in text_output and THINK_END in text_output:
        text_output = text_output.split(THINK_END)[-1].strip()
    text_output = text_output.split(ANSWER_START)[1].split(ANSWER_END)[0].strip()

    
    pred_search_replaces = parse_search_replace(text_output)
    pred_new_content = apply_code_change(code_context, pred_search_replaces)
    pred_patch, pred_patch_dict = generate_git_diff(code_context, pred_new_content, remove_repo_name=remove_repo_name)
    if pred_patch == "":
        return None
    return {"model_patch": pred_patch, "model_patch_dict": pred_patch_dict}


def extract_pred_patch_relaxed_formatting(
    code_context: dict[str, str],
    text_output: str,
    remove_repo_name: bool = False,
) -> Optional[dict]:
    """
    Extracts the predicted patch and its dict from the model output if possible.
    Returns (pred_patch, pred_patch_dict) or None if extraction fails.
    """
    # Extract the <solution>...</solution> block (ignore any <think>...</think>).
    if THINK_START in text_output and THINK_END in text_output:
        text_output = text_output.split(THINK_END)[-1].strip()
    if ANSWER_START in text_output and ANSWER_END in text_output:
        text_output = text_output.split(ANSWER_START)[1].split(ANSWER_END)[0].strip()

    pred_search_replaces = parse_search_replace(text_output)
    pred_new_content = apply_code_change(code_context, pred_search_replaces)
    pred_patch, pred_patch_dict = generate_git_diff(code_context, pred_new_content, remove_repo_name=remove_repo_name)
    if pred_patch == "":
        return None
    return {"model_patch": pred_patch, "model_patch_dict": pred_patch_dict}


def extract_repro_test(text_output: str,
                       instance_id: str) -> tuple[str, dict] | None:
    test_script_blocks = extract_python_blocks(text_output)
    if not test_script_blocks:
        return None
    processed_test_script = [create_patch_from_code(test_script_blocks[-1], len(test_script_blocks))]
    reproduction_tests_dict = {"instance_id": instance_id, "test_patch": [processed_test_script[0]]}
    repro_test_info_base64 = base64.b64encode(json.dumps(reproduction_tests_dict).encode()).decode()
    return {
            "repro_test_info_base64": repro_test_info_base64,
            "reproduction_tests_dict": reproduction_tests_dict,
        }