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
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.structeval.app import (
    StructEvalResourcesServer,
    StructEvalResourcesServerConfig,
    StructEvalVerifyRequest,
    _extract_code,
    _path_exists,
)


MINIMAL_RESPONSES_CREATE_PARAMS = {
    "input": [{"role": "user", "content": "test"}],
}


def _make_server() -> StructEvalResourcesServer:
    config = StructEvalResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return StructEvalResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_verify_request(
    text: str,
    output_type: str = "JSON",
    raw_output_metric: list | None = None,
    task_id: str = "000500",
    task_name: str = "Text to JSON",
    input_type: str = "Text",
) -> StructEvalVerifyRequest:
    return StructEvalVerifyRequest(
        responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
        response=_make_response(text),
        task_id=task_id,
        task_name=task_name,
        input_type=input_type,
        output_type=output_type,
        raw_output_metric=raw_output_metric or [],
        rendering=False,
    )


# ---------------------------------------------------------------------------
# Code extraction tests
# ---------------------------------------------------------------------------


class TestExtractCode:
    def test_begin_end_tags(self) -> None:
        text = 'Some preamble\n<|BEGIN_CODE|>\n{"a": 1}\n<|END_CODE|>\nSome epilogue'
        assert _extract_code(text, "json") == '{"a": 1}'

    def test_fenced_code_block(self) -> None:
        text = 'Here is the result:\n```json\n{"a": 1}\n```\nDone.'
        assert _extract_code(text, "json") == '{"a": 1}'

    def test_fallback_raw_text(self) -> None:
        text = '{"a": 1}'
        assert _extract_code(text, "json") == '{"a": 1}'

    def test_begin_code_no_end_tag(self) -> None:
        text = '<|BEGIN_CODE|>\n{"a": 1}'
        assert _extract_code(text, "json") == '{"a": 1}'


# ---------------------------------------------------------------------------
# Path validation unit tests
# ---------------------------------------------------------------------------


class TestPathExists:
    def test_simple_dotted_path(self) -> None:
        data = {"novel": {"author": {"name": "Alice"}}}
        assert _path_exists(data, "novel.author.name") is True
        assert _path_exists(data, "novel.author.age") is False

    def test_array_index(self) -> None:
        data = {"items": [{"name": "a"}, {"name": "b"}]}
        assert _path_exists(data, "items.[0].name") is True
        assert _path_exists(data, "items.[5].name") is False

    def test_wildcard(self) -> None:
        data = {"items": [{"name": "a"}, {"name": "b"}]}
        assert _path_exists(data, "items.*.name") is True

    def test_csv_header(self) -> None:
        data = {"csv_headers": ["name", "age", "score"], "csv_rows": []}
        assert _path_exists(data, "csv::name") is True
        assert _path_exists(data, "csv::missing") is False

    def test_backtick_escape(self) -> None:
        data = {"a.b": {"c": 1}}
        assert _path_exists(data, "`a.b`.c") is True


# ---------------------------------------------------------------------------
# Full verify() tests — JSON
# ---------------------------------------------------------------------------


class TestVerifyJSON:
    async def test_valid_json_all_paths(self) -> None:
        server = _make_server()
        generation = '<|BEGIN_CODE|>\n{"novel": {"title": "Test", "author": {"name": "Alice"}}}\n<|END_CODE|>'
        result = await server.verify(_make_verify_request(generation, "JSON", ["novel.title", "novel.author.name"]))
        assert result.render_score == 1.0
        assert result.key_validation_score == 1.0
        assert result.reward == 1.0

    async def test_invalid_json(self) -> None:
        server = _make_server()
        result = await server.verify(_make_verify_request("<|BEGIN_CODE|>\n{broken json\n<|END_CODE|>", "JSON", ["a"]))
        assert result.render_score == 0.0
        assert result.key_validation_score == 0.0
        assert result.reward == 0.0
        assert result.error_type == "parse_error"

    async def test_partial_paths(self) -> None:
        server = _make_server()
        generation = '<|BEGIN_CODE|>\n{"a": 1, "b": 2}\n<|END_CODE|>'
        result = await server.verify(_make_verify_request(generation, "JSON", ["a", "b", "c", "d"]))
        assert result.render_score == 1.0
        assert result.key_validation_score == 0.5  # 2 of 4
        assert result.reward == round(0.2 + 0.8 * 0.5, 2)  # 0.6

    async def test_no_paths_match(self) -> None:
        server = _make_server()
        generation = '<|BEGIN_CODE|>\n{"x": 1}\n<|END_CODE|>'
        result = await server.verify(_make_verify_request(generation, "JSON", ["a", "b"]))
        assert result.render_score == 1.0
        assert result.key_validation_score == 0.0
        assert result.reward == 0.2

    async def test_empty_response(self) -> None:
        server = _make_server()
        result = await server.verify(_make_verify_request("   ", "JSON", ["a"]))
        assert result.reward == 0.0
        assert result.error_type == "empty_response"


# ---------------------------------------------------------------------------
# Verify tests — YAML
# ---------------------------------------------------------------------------


class TestVerifyYAML:
    async def test_valid_yaml(self) -> None:
        server = _make_server()
        generation = "<|BEGIN_CODE|>\ntitle: Test\nauthor:\n  name: Alice\n<|END_CODE|>"
        result = await server.verify(
            _make_verify_request(
                generation, "YAML", ["title", "author.name"], task_id="001800", task_name="Text to YAML"
            )
        )
        assert result.render_score == 1.0
        assert result.key_validation_score == 1.0
        assert result.reward == 1.0

    async def test_invalid_yaml(self) -> None:
        server = _make_server()
        result = await server.verify(
            _make_verify_request(
                "<|BEGIN_CODE|>\n:\n  bad:\n    - [\n<|END_CODE|>",
                "YAML",
                ["a"],
                task_id="001800",
                task_name="Text to YAML",
            )
        )
        assert result.render_score == 0.0
        assert result.reward == 0.0


# ---------------------------------------------------------------------------
# Verify tests — CSV
# ---------------------------------------------------------------------------


class TestVerifyCSV:
    async def test_valid_csv_headers(self) -> None:
        server = _make_server()
        csv_text = "<|BEGIN_CODE|>\nname,age,score\nAlice,30,95\nBob,25,88\n<|END_CODE|>"
        result = await server.verify(
            _make_verify_request(
                csv_text, "CSV", ["csv::name", "csv::age", "csv::score"], task_id="000200", task_name="Text to CSV"
            )
        )
        assert result.render_score == 1.0
        assert result.key_validation_score == 1.0
        assert result.reward == 1.0

    async def test_csv_missing_headers(self) -> None:
        server = _make_server()
        csv_text = "<|BEGIN_CODE|>\nname,age\nAlice,30\n<|END_CODE|>"
        result = await server.verify(
            _make_verify_request(
                csv_text,
                "CSV",
                ["csv::name", "csv::age", "csv::score"],
                task_id="000200",
                task_name="Text to CSV",
            )
        )
        assert result.render_score == 1.0
        assert result.key_validation_score == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Verify tests — XML
# ---------------------------------------------------------------------------


class TestVerifyXML:
    async def test_valid_xml(self) -> None:
        server = _make_server()
        xml_text = "<|BEGIN_CODE|>\n<root><title>Test</title><author><name>Alice</name></author></root>\n<|END_CODE|>"
        result = await server.verify(
            _make_verify_request(
                xml_text, "XML", ["root.title", "root.author.name"], task_id="001700", task_name="Text to XML"
            )
        )
        assert result.render_score == 1.0
        assert result.key_validation_score == 1.0

    async def test_xml_attribute_fallback(self) -> None:
        server = _make_server()
        xml_text = '<|BEGIN_CODE|>\n<root><item id="1">val</item></root>\n<|END_CODE|>'
        result = await server.verify(
            _make_verify_request(xml_text, "XML", ["root.item.@id"], task_id="001700", task_name="Text to XML")
        )
        assert result.render_score == 1.0
        assert result.key_validation_score == 1.0


# ---------------------------------------------------------------------------
# Verify tests — TOML
# ---------------------------------------------------------------------------


class TestVerifyTOML:
    async def test_valid_toml(self) -> None:
        server = _make_server()
        toml_text = '<|BEGIN_CODE|>\ntitle = "Test"\n[author]\nname = "Alice"\n<|END_CODE|>'
        result = await server.verify(
            _make_verify_request(
                toml_text, "TOML", ["title", "author.name"], task_id="001000", task_name="Text to TOML"
            )
        )
        assert result.render_score == 1.0
        assert result.key_validation_score == 1.0
        assert result.reward == 1.0

    async def test_invalid_toml(self) -> None:
        server = _make_server()
        result = await server.verify(
            _make_verify_request(
                "<|BEGIN_CODE|>\n[invalid\nbroken = \n<|END_CODE|>",
                "TOML",
                ["a"],
                task_id="001000",
                task_name="Text to TOML",
            )
        )
        assert result.render_score == 0.0
        assert result.reward == 0.0


# ---------------------------------------------------------------------------
# compute_metrics test
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_metrics_by_type(self) -> None:
        server = _make_server()
        tasks = [
            [
                {
                    "reward": 1.0,
                    "output_type": "JSON",
                    "input_type": "Text",
                    "task_name": "Text to JSON",
                    "render_score": 1.0,
                    "key_validation_score": 1.0,
                },
                {
                    "reward": 0.2,
                    "output_type": "YAML",
                    "input_type": "CSV",
                    "task_name": "CSV to YAML",
                    "render_score": 1.0,
                    "key_validation_score": 0.0,
                },
            ]
        ]
        metrics = server.compute_metrics(tasks)
        assert "mean/reward_output_JSON" in metrics
        assert "mean/reward_output_YAML" in metrics
        assert "mean/reward_input_Text" in metrics
        assert "mean/reward_input_CSV" in metrics
        assert "mean/render_score" in metrics
        assert "mean/key_validation_score" in metrics
        assert "mean/render_score_JSON" in metrics
        assert "mean/reward_Text to JSON" in metrics
        assert "mean/reward_CSV to YAML" in metrics
