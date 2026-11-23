# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
from markdown_it import MarkdownIt


ROOT = Path(__file__).resolve().parents[3]
SKILL = ROOT / ".agents/skills/review-pr-team"


@pytest.mark.parametrize("page", sorted(SKILL.rglob("*.md")), ids=lambda path: str(path.relative_to(SKILL)))
def test_review_skill_references(page: Path) -> None:
    """Resolve repository links, including Python symbols cited as code in the link label."""
    for block in MarkdownIt("commonmark").enable("table").parse(page.read_text()):
        children = block.children or []
        for index, token in enumerate(children):
            if token.type == "code_inline" and re.fullmatch(r"[\w./-]+/[\w.-]+\.(?:py|mdx?|ya?ml)", token.content):
                assert any((base / token.content).is_file() for base in (ROOT, SKILL, page.parent)), (
                    f"{page.name}: missing file {token.content}"
                )
            if token.type != "link_open":
                continue
            url = urlsplit(token.attrs["href"])
            if url.scheme or url.netloc or not url.path:
                continue
            target = (page.parent / unquote(url.path)).resolve()
            assert target.is_relative_to(ROOT) and target.exists(), f"{page.name}: missing target {url.path}"
            label = children[index + 1]
            if target.suffix == ".py" and label.type == "code_inline":
                nodes = ast.parse(target.read_text()).body
                for name in label.content.split("."):
                    definition = next(
                        (
                            node
                            for node in nodes
                            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                            and node.name == name
                        ),
                        None,
                    )
                    assert definition is not None, f"{target.relative_to(ROOT)} has no {label.content}"
                    nodes = definition.body
