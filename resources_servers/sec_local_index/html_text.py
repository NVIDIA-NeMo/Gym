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
"""The HTML-to-text reduction that parse_html_page hands the model.

Whatever the filing was fetched from, the model has to see the same characters,
so this is the one implementation both servers and both modes call. It restates
``finance_agent.tools.ParseHtmlPage``; a conformance test in the
finance_agent_v2 venv compares the two and fails if upstream changes.
"""

from __future__ import annotations

from bs4 import BeautifulSoup


def html_to_text(html: str) -> str:
    """Strip scripts and styles, then collapse the document to non-empty lines."""
    soup = BeautifulSoup(html, "html.parser")
    for script_or_style in soup(["script", "style"]):
        _ = script_or_style.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return "\n".join(chunk for chunk in chunks if chunk)
