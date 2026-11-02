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
"""The rows the question converter writes for a training run."""

from __future__ import annotations

from resources_servers.finance_sec_search.scripts.convert_questions import (
    EDGAR_SEARCH_TOOL,
    PROMPT,
    convert_entry,
)


def test_converter_preserves_prompt_and_exposes_edgar_search() -> None:
    converted = convert_entry(
        {"question": "What was revenue?", "expected_answer": "Example"},
        search_tool="edgar_search",
    )

    params = converted["responses_create_params"]
    assert params["input"][0]["content"] == PROMPT + "What was revenue?"
    assert [tool["name"] for tool in params["tools"]] == [
        "retrieve_information",
        "parse_html_page",
        "edgar_search",
        "submit_final_result",
    ]
    assert params["tools"][2] == EDGAR_SEARCH_TOOL
