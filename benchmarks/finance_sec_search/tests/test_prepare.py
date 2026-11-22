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

# Smoke tests for prepare.py and its variants, converting a small fixture CSV
# instead of the upstream public CSV. These run in the repo-root venv, like
# `gym eval prepare` itself.

import csv
import io
import json
import re
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from benchmarks.finance_sec_search import prepare as prepare_module
from benchmarks.finance_sec_search import (
    prepare_edgar_search,
    prepare_edgar_search_web_search,
    prepare_web_search,
)


BENCHMARK_DIR = Path(prepare_module.__file__).parent
QUESTION = "What was the company's revenue in fiscal 2024?"
EDGAR_CONFIGS = ("config_edgar_search_no_web_search.yaml", "config_edgar_search_web_search.yaml")


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    """Run a variant's prepare() on a one-question CSV and return its row."""
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=["Question", "Answer", "Question Type"])
    writer.writeheader()
    writer.writerow({"Question": QUESTION, "Answer": "$1.0 billion", "Question Type": "Quantitative Retrieval"})
    monkeypatch.setattr(prepare_module.urllib.request, "urlopen", lambda url: io.BytesIO(text.getvalue().encode()))
    monkeypatch.setattr(prepare_module, "DATA_DIR", tmp_path)
    for name in ("OUTPUT_FPATH", "OUTPUT_FPATH_WEB", "OUTPUT_FPATH_EDGAR", "OUTPUT_FPATH_EDGAR_WEB"):
        monkeypatch.setattr(prepare_module, name, tmp_path / getattr(prepare_module, name).name)

    def run(variant) -> dict:
        [row] = [json.loads(line) for line in variant.prepare().read_text(encoding="utf-8").splitlines()]
        return row

    return run


def _tool_names(row: dict) -> list:
    return [tool["name"] for tool in row["responses_create_params"]["tools"]]


def _prompt(row: dict) -> str:
    [message] = row["responses_create_params"]["input"]
    return message["content"]


def test_edgar_search_variants_ask_with_the_published_prompt_and_tools(prepared) -> None:
    sec_tools = ["retrieve_information", "parse_html_page", "edgar_search", "submit_final_result"]
    without_web, with_web = prepared(prepare_edgar_search), prepared(prepare_edgar_search_web_search)

    assert _tool_names(without_web) == sec_tools
    assert _tool_names(with_web) == ["web_search", *sec_tools]
    assert with_web["responses_create_params"]["tools"][0] == prepare_module.PUBLISHED_WEB_TOOL
    for row in (without_web, with_web):
        assert _prompt(row) == prepare_module.PUBLISHED_PROMPT + QUESTION
        assert (row["question"], row["expected_answer"]) == (QUESTION, "$1.0 billion")


def test_the_default_variants_are_unchanged(prepared) -> None:
    sec_tools = ["retrieve_information", "parse_html_page", "sec_filing_search", "submit_final_result"]
    without_web, with_web = prepared(prepare_module), prepared(prepare_web_search)

    assert _tool_names(without_web) == sec_tools
    assert _tool_names(with_web) == ["web_search", *sec_tools]
    assert with_web["responses_create_params"]["tools"][0] == prepare_module.WEB_TOOL
    for row in (without_web, with_web):
        assert _prompt(row) == prepare_module.PROMPT + QUESTION


def test_the_prompt_tools_and_configs_stop_at_one_date() -> None:
    [asked_as_of] = re.findall(r"current date is (\w+ \d{2}, \d{4})\.", prepare_module.PUBLISHED_PROMPT)
    cutoff = prepare_module.PUBLISHED_MAX_END_DATE
    assert datetime.strptime(asked_as_of, "%B %d, %Y").date().isoformat() == cutoff

    edgar_dates = prepare_module.EDGAR_SEARCH_TOOL["parameters"]["properties"]
    web_dates = prepare_module.PUBLISHED_WEB_TOOL["parameters"]["properties"]
    assert edgar_dates["end_date"]["default"] == cutoff
    for parameter in (edgar_dates["start_date"], edgar_dates["end_date"], web_dates["end_date"]):
        assert f"later than {cutoff}, it will be set to {cutoff}." in parameter["description"]

    for name in EDGAR_CONFIGS:
        config = yaml.safe_load((BENCHMARK_DIR / name).read_text(encoding="utf-8"))
        [server] = [value for key, value in config.items() if key.endswith("_benchmark_resources_server")]
        settings = server["resources_servers"]["finance_sec_search"]
        assert (settings["edgar_search_mode"], settings["max_end_date"]) == ("live", cutoff)
        assert "SEC_API_KEY" in settings["sec_api_key"]
