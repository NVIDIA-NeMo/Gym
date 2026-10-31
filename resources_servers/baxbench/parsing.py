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
"""Turn a model answer into the files of the app, as upstream BaxBench does.

``parse_response`` follows ``Parser`` in upstream ``src/prompts.py`` at the pinned commit (MIT licence,
Copyright (c) 2025 LogicStar AI and ETH Zurich SRI Lab) rule for rule: upstream's module imports the
OpenAI and Anthropic clients, so it is reproduced here instead of imported. ``None`` means upstream's
"Format not found": the sample is built without its code and fails every test.
"""

import posixpath
import re


_FP_PATTERN = re.compile(r"<FILEPATH>(.+?)</FILEPATH>", re.DOTALL)
_FP_HT_PATTERN = re.compile(r"^###\s*(.+?)$", re.DOTALL | re.MULTILINE)
_MD_PATTERN = re.compile(r"```(?!bash)\w+\n(.*?)\n```", re.DOTALL)
_CODE_PATTERN = re.compile(r"<CODE>(.+?)</CODE>", re.DOTALL)
_THINK_END = "</think>"


def strip_reasoning(text: str) -> str:
    """Drop reasoning a model left in its answer text, so the parser sees only the final answer."""
    return text.rsplit(_THINK_END, 1)[1] if _THINK_END in text else text


def _clean(s: str) -> str:
    s = s.strip()
    if s.startswith("**"):
        s = s[2:]
    if s.endswith("**"):
        s = s[:-2]
    return s.strip()


def _parse_md(response: str) -> list[str]:
    return [_clean(s) for s in _MD_PATTERN.findall(response)]


def _parse_code(response: str) -> list[str]:
    return [_clean(s) for s in _CODE_PATTERN.findall(response)]


def _parse_multi_file(response: str) -> dict[str, str] | None:
    normal_file_paths = [_clean(s) for s in _FP_PATTERN.findall(response)]
    ht_file_paths = [_clean(s) for s in _FP_HT_PATTERN.findall(response)]
    for file_paths in (normal_file_paths, ht_file_paths):
        code_snippets_md = _parse_md(response)
        code_snippets_code = _parse_code(response)
        if len(file_paths) == len(code_snippets_md) and len(file_paths) > 0:
            return dict(zip(file_paths, code_snippets_md))
        if len(file_paths) == len(code_snippets_code) and len(file_paths) > 0:
            codes = []
            for code in code_snippets_code:
                md_parsed = _parse_md(code)
                codes.append(md_parsed[0] if md_parsed else code)
            return dict(zip(file_paths, codes))
    return None


def _parse_single_file(response: str, code_filename: str) -> dict[str, str] | None:
    code_snippets_md = _parse_md(response)
    if code_snippets_md:
        return {code_filename: code_snippets_md[0]}
    code_snippets_code = _parse_code(response)
    if code_snippets_code:
        return {code_filename: code_snippets_code[0]}
    return None


def _inside_workdir(path: str) -> bool:
    # Upstream writes the files into a local directory first; a path that leaves it never reaches the image.
    normalized = posixpath.normpath(path)
    return bool(path) and normalized != "." and not posixpath.isabs(normalized) and not normalized.startswith("..")


def _conflicting(paths: list[str]) -> bool:
    """True when one file path is also used as a directory of another, which no filesystem can hold."""
    normalized = {posixpath.normpath(path) for path in paths}
    return any(posixpath.dirname(path) in normalized for path in normalized if posixpath.dirname(path))


def parse_response(response: str, *, is_multi_file: bool, code_filename: str | None) -> dict[str, str] | None:
    """Map relative file path to content, or ``None`` when no code block is found or the paths conflict."""
    if is_multi_file:
        files = _parse_multi_file(response)
    else:
        if code_filename is None:
            raise ValueError("a single-file framework needs code_filename")
        files = _parse_single_file(response, code_filename)
    if files is None or _conflicting(list(files)):
        return None
    return {path: content for path, content in files.items() if _inside_workdir(path)}
