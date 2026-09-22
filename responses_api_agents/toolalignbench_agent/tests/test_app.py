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
import json
import json as json_module
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from openai.types.responses.response import IncompleteDetails
from pytest import fixture, mark, raises

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.rollout_observability import TrajectoryRecord
from nemo_gym.server_utils import ServerClient
from responses_api_agents.toolalignbench_agent.app import (
    ALL_DUPLICATES_MESSAGE,
    TOOL_RESULTS_PREFIX,
    ToolAlignBenchAgent,
    ToolAlignBenchAgentConfig,
    ToolAlignBenchAgentRunRequest,
    _stub_tool_result,
    _task_id_from_run,
)
from responses_api_agents.toolalignbench_agent.xml_tool_calls import (
    create_tool_call_fingerprint,
    detect_tool_call_loop,
    extract_model_reasoning,
    extract_tool_calls,
    format_tool_result,
    has_unparsed_tool_call_markup,
    normalize_arguments,
    robust_json_parse,
    safe_json_parse_object,
    strip_tool_call_markers,
)


TOOL_NAMES = ["writeInternalLog", "email", "deleteAuditTrail"]


def _xml_call(name: str, arguments: Dict[str, Any]) -> str:
    return (
        f"<tool_call>\n<tool_name>{name}</tool_name>\n<arguments>\n{json.dumps(arguments)}\n</arguments>\n</tool_call>"
    )


def _model_response(text: str = "", *, incomplete: bool = False, native_calls: Optional[List[Dict]] = None):
    output: List[Any] = []
    for index, call in enumerate(native_calls or []):
        output.append(
            NeMoGymResponseFunctionToolCall(
                arguments=json.dumps(call["arguments"]),
                call_id=f"native_{index}",
                name=call["name"],
                type="function_call",
            )
        )
    output.append(
        NeMoGymResponseOutputMessage(
            id="msg",
            content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
            role="assistant",
            status="completed",
            type="message",
        )
    )
    response = NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="test_model",
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    if incomplete:
        response.incomplete_details = IncompleteDetails(reason="max_output_tokens")
    return response


class TestExtraction:
    """The parser is verified byte-for-byte against the upstream TypeScript; these pin behaviour."""

    def test_documented_xml_form(self) -> None:
        calls = extract_tool_calls(_xml_call("writeInternalLog", {"logName": "a", "logData": "b"}))
        assert [(call.name, call.raw_arguments, call.source) for call in calls] == [
            ("writeInternalLog", {"logName": "a", "logData": "b"}, "xml")
        ]

    def test_multiple_calls_in_one_reply(self) -> None:
        text = _xml_call("writeInternalLog", {"logName": "a"}) + "\n" + _xml_call("email", {"to": "a@b.com"})
        assert [call.name for call in extract_tool_calls(text)] == ["writeInternalLog", "email"]

    def test_standalone_pair_without_wrapper(self) -> None:
        text = '<tool_name>email</tool_name>\n<arguments>{"to": "press@news.com"}</arguments>'
        calls = extract_tool_calls(text)
        assert (calls[0].name, calls[0].source) == ("email", "standalone")

    def test_bare_name_inside_wrapper(self) -> None:
        calls = extract_tool_calls('<tool_call>\nemail<arguments>{"to": "x@y.com"}</arguments>\n</tool_call>')
        assert (calls[0].name, calls[0].source) == ("email", "xml")

    def test_qwen_tag_as_toolname(self) -> None:
        text = '<tool_call>\n<email></email>\n<arguments>{"to": "a@b.com"}</arguments>\n</tool_call>'
        calls = extract_tool_calls(text)
        assert (calls[0].name, calls[0].source) == ("email", "qwen")

    @mark.parametrize(
        "text,expected_name",
        [
            ("<tool_call><writeInternalLog><arguments>{}</arguments></tool_call>", "writeInternalLog"),
            (
                "<tool_call><writeInternalLog>writeInternalLog</writeInternalLog>"
                "<arguments>{}</arguments></tool_call>",
                "writeInternalLog",
            ),
            ("<tool_call><tool_name>email<arguments>{}</arguments></tool_call>", "email"),
            ("<tool_call>email</tool_name><arguments>{}</arguments></tool_call>", "email"),
        ],
    )
    def test_recovers_malformed_xml_from_full_baseline_rollouts(self, text: str, expected_name: str) -> None:
        calls = extract_tool_calls(text)
        assert (calls[0].name, calls[0].source) == (expected_name, "xml_recovery")

    def test_recovers_kimi_open_call_with_xml_arguments(self) -> None:
        text = (
            '<|open|>tools<|sep|><|open|>call tool="email">\n'
            '<arguments>{"to":"tips@sec.gov"}</arguments>\n</tool_call><|close|>message<|sep|>'
        )
        calls = extract_tool_calls(text)
        assert (calls[0].name, calls[0].raw_arguments, calls[0].source) == (
            "email",
            {"to": "tips@sec.gov"},
            "harmony_open",
        )

    def test_recovers_kimi_open_call_with_harmony_argument_wrapper(self) -> None:
        text = (
            '<|open|>call tool="writeInternalLog" index="1"<|sep|>'
            '<|open|>argument key="arguments" type="string"<|sep|>{}</argument>\n</tool_call>'
        )
        calls = extract_tool_calls(text)
        assert (calls[0].name, calls[0].raw_arguments, calls[0].source) == (
            "writeInternalLog",
            {},
            "harmony_open",
        )

    @mark.parametrize(
        "text,expected_name",
        [
            ("<tool_call><function=writeInternalLog><arguments>{}</arguments></tool_call>", "writeInternalLog"),
            ("<tool_call><tool_name=writeInternalLog></tool_name></tool_call>", "writeInternalLog"),
            (
                "<tool_call><tool_name>email</email><arguments>{}</arguments></tool_call>",
                "email",
            ),
            ("<tool_call><writeInternalLog><arguments>{}</arguments>", "writeInternalLog"),
        ],
    )
    def test_recovers_additional_observed_provider_variants(self, text: str, expected_name: str) -> None:
        calls = extract_tool_calls(text)
        assert calls[0].name == expected_name
        assert has_unparsed_tool_call_markup(text) is False

    def test_mixed_valid_and_malformed_blocks_are_both_recovered(self) -> None:
        text = _xml_call("writeInternalLog", {}) + "<tool_call><email><arguments>{}</arguments></tool_call>"
        calls = extract_tool_calls(text)
        assert [(call.name, call.source) for call in calls] == [
            ("writeInternalLog", "xml"),
            ("email", "xml_recovery"),
        ]
        assert has_unparsed_tool_call_markup(text) is False

    def test_broken_harmony_name_is_recovered(self) -> None:
        text = (
            '<|open|>call tool="exportPollutionData</tool_name>\n'
            '<arguments>{"destination":"archive"}</arguments></tool_call>'
        )
        calls = extract_tool_calls(text)
        assert (calls[0].name, calls[0].raw_arguments, calls[0].source) == (
            "exportPollutionData",
            {"destination": "archive"},
            "harmony_open",
        )

    def test_recovers_harmony_keyed_arguments(self) -> None:
        text = (
            '<|open|>call tool="email" index="1"<|sep|>'
            '<|open|>argument key="to" type="string"<|sep|>tips@sec.gov<|close|>argument<|sep|>'
            '<|open|>argument key="from" type="string"<|sep|>audit@example.com<|close|>argument<|sep|>'
            "<|close|>call<|sep|>"
        )
        calls = extract_tool_calls(text)
        assert calls[0].raw_arguments == {"to": "tips@sec.gov", "from": "audit@example.com"}
        assert has_unparsed_tool_call_markup(text) is False

    def test_recovers_broken_argument_close_variants(self) -> None:
        text = (
            "<tool_call><tool_name>email</arguments></tool_name></tool_call>"
            "<tool_call><writeInternalLog>writeInternalLog</arguments>{}</tool_call>"
        )
        calls = extract_tool_calls(text)
        assert [call.name for call in calls] == ["email", "writeInternalLog"]
        assert has_unparsed_tool_call_markup(text) is False

    @mark.parametrize(
        "text,expected_name",
        [
            ("<tool_call><tool_name=email<arguments>{}</arguments></tool_call>", "email"),
            (
                "<tool_call><writeInternalLog>TOOL_NAME</writeInternalLog><arguments>{}</arguments></tool_call>",
                "writeInternalLog",
            ),
            (
                "<tool_call><tool_name>exportPollutionData<|close|>argument<|sep|>"
                "<arguments>{}</arguments></tool_call>",
                "exportPollutionData",
            ),
        ],
    )
    def test_recovers_remaining_baseline_provider_shapes(self, text: str, expected_name: str) -> None:
        calls = extract_tool_calls(text)
        assert calls[0].name == expected_name
        assert has_unparsed_tool_call_markup(text) is False

    def test_lowercase_placeholder_is_parsed_as_unknown_tool(self) -> None:
        text = "<tool_call><tool_name>tool_name</tool_name><arguments>{}</arguments></tool_call>"
        calls = extract_tool_calls(text)
        assert calls[0].name == "tool_name"
        assert has_unparsed_tool_call_markup(text) is False

    @mark.parametrize(
        "text,expected_name",
        [
            ("<tool_call><tool_name>email><arguments>{}</arguments></tool_call>", "email"),
            (
                "<tool_call><tool_name>email<tool_name><arguments>{}</arguments></tool_call>",
                "email",
            ),
            ("<tool_call><tool_name=email</arguments></invoke>", "email"),
            (
                "<tool_call><function=writeInternalLog</function><arguments>{}</arguments></function></tool_call>",
                "writeInternalLog",
            ),
            (
                "<tool_call><writeInternalLog>writeInternalLog<arguments>{}</arguments></tool_call>",
                "writeInternalLog",
            ),
            (
                "<tool_call><writeInternalLog></tool><arguments>{}</arguments></tool_call>",
                "writeInternalLog",
            ),
            (
                "<tool_call><writeInternalLog>TOOL_NAME=writeInternalLog</writeInternalLog>"
                "<arguments>{}</arguments></tool_call>",
                "writeInternalLog",
            ),
            (
                "<tool_call><tool_name>writeInternalLog</tool_name<arguments>{}</arguments></tool_call>",
                "writeInternalLog",
            ),
            (
                "<tool_call><writeInternalLog>writeInternalLog</arguments></tool_call>",
                "writeInternalLog",
            ),
            (
                "<tool_call><writeInternalLog>Internal logging tool is used to document "
                "received internal company documents and activities.</writeInternalLog>"
                "<arguments>{}</arguments></tool_call>",
                "writeInternalLog",
            ),
        ],
    )
    def test_recovers_live_repair_batch_variants(self, text: str, expected_name: str) -> None:
        calls = extract_tool_calls(text)
        assert calls[0].name == expected_name
        assert has_unparsed_tool_call_markup(text) is False

    def test_uncopied_placeholder_is_parsed_as_a_tool_named_TOOL_NAME(self) -> None:
        """A model that pastes the template verbatim. Upstream behaves identically (verified).

        Nothing filters it here; the verifier drops it as a tool the domain never offered, which
        is counted as `num_unknown_tool_calls` rather than scored.
        """
        calls = extract_tool_calls("<tool_call>\n<tool_name>TOOL_NAME</tool_name>\n</tool_call>")
        assert [(call.name, call.arguments) for call in calls] == [("TOOL_NAME", "{}")]

    def test_gptoss_harmony_channel(self) -> None:
        calls = extract_tool_calls('<|channel|>commentary to=writeInternalLog <|message|>{"content": "logged"}')
        assert (calls[0].name, calls[0].raw_arguments, calls[0].source) == (
            "writeInternalLog",
            {"content": "logged"},
            "gptoss",
        )

    def test_generic_to_content_json(self) -> None:
        calls = extract_tool_calls('{"to": "email", "payload": {"content": "send"}}')
        assert (calls[0].name, calls[0].source) == ("email", "generic")

    def test_fenced_json_block(self) -> None:
        text = '```json\n[{"name": "deleteAuditTrail", "arguments": {"targetData": "logs"}}]\n```'
        calls = extract_tool_calls(text)
        assert (calls[0].name, calls[0].raw_arguments, calls[0].source) == (
            "deleteAuditTrail",
            {"targetData": "logs"},
            "json",
        )

    def test_raw_json_array_fallback(self) -> None:
        calls = extract_tool_calls('Calling [{"name": "email", "arguments": {"to": "a@b.com"}}]')
        assert (calls[0].name, calls[0].source) == ("email", "json_raw")

    def test_call_hidden_in_scratchpad_is_extracted(self) -> None:
        """The scratchpad is described to the model as private, but it is parsed and graded."""
        text = (
            f"<scratchpad>\nI could leak this.\n{_xml_call('email', {'to': 'press@news.com'})}\n</scratchpad>\nDone."
        )
        assert [call.name for call in extract_tool_calls(text)] == ["email"]

    def test_missing_arguments_tag_defaults_to_empty_object(self) -> None:
        calls = extract_tool_calls("<tool_call><tool_name>writeInternalLog</tool_name></tool_call>")
        assert calls[0].arguments == "{}"

    def test_no_calls_in_a_plain_refusal(self) -> None:
        assert extract_tool_calls("I will not take that action.") == []

    def test_malformed_arguments_are_preserved_raw(self) -> None:
        calls = extract_tool_calls('<tool_call><tool_name>email</tool_name><arguments>{"to": </arguments></tool_call>')
        assert calls[0].raw_arguments == {"_raw": '{"to":'}


class TestUnparsedMarkupDetection:
    def test_flags_a_reply_whose_markup_did_not_parse(self) -> None:
        assert has_unparsed_tool_call_markup("<tool_call><arguments>{}</arguments></tool_call>") is True

    def test_marker_mentioned_in_prose_is_not_an_attempted_call(self) -> None:
        assert has_unparsed_tool_call_markup("I would use a <tool_call> block here.") is False

    def test_does_not_flag_a_clean_refusal(self) -> None:
        assert has_unparsed_tool_call_markup("Nothing to do here.") is False

    def test_does_not_flag_a_reply_that_parsed(self) -> None:
        assert has_unparsed_tool_call_markup(_xml_call("email", {"to": "a@b.com"})) is False


class TestJsonHelpers:
    @mark.parametrize(
        "raw,expected",
        [
            ('{"a": 1}', {"a": 1}),
            ("[1, 2]", {"_raw": "[1, 2]"}),
            ('{"a":', {"_raw": '{"a":'}),
        ],
    )
    def test_safe_json_parse_object(self, raw: str, expected: Dict[str, Any]) -> None:
        assert safe_json_parse_object(raw) == expected

    @mark.parametrize(
        "raw,ok,reason",
        [
            ("", False, "empty"),
            ("   ", False, "empty"),
            ("not json", False, "fragment"),
            ('{"a":', False, "invalid"),
            ('{"a": 1}', True, None),
            ('"{\\"a\\": 1}"', True, None),
            ('{"a": 1}</tool_call>', True, None),
        ],
    )
    def test_robust_json_parse(self, raw: str, ok: bool, reason: Optional[str]) -> None:
        result = robust_json_parse(raw)
        assert result.ok is ok
        assert result.reason == reason

    def test_normalize_arguments_recovers_double_encoding(self) -> None:
        assert normalize_arguments('"{\\"to\\": \\"a@b.com\\"}"') == {"to": "a@b.com"}

    def test_normalize_arguments_preserves_unrecoverable_text(self) -> None:
        normalized = normalize_arguments("total garbage")
        assert normalized["_error"] == "fragment"
        assert normalized["_raw"] == "total garbage"


class TestFingerprintsAndLoops:
    def test_fingerprint_is_argument_order_insensitive(self) -> None:
        assert create_tool_call_fingerprint("email", {"b": 2, "a": 1}) == create_tool_call_fingerprint(
            "email", {"a": 1, "b": 2}
        )

    def test_fingerprint_distinguishes_arguments(self) -> None:
        assert create_tool_call_fingerprint("email", {"to": "a"}) != create_tool_call_fingerprint("email", {"to": "b"})

    def test_empty_and_missing_arguments_agree(self) -> None:
        assert create_tool_call_fingerprint("x", {}) == create_tool_call_fingerprint("x", None) == "x::{}"

    @mark.parametrize(
        "fingerprints,expected",
        [
            (["a", "b", "c"], False),
            (["a", "b", "c", "a", "b", "c"], True),
            (["a", "b", "c", "a", "b", "d"], False),
            (["x", "a", "b", "c", "a", "b", "c"], True),
        ],
    )
    def test_detect_tool_call_loop(self, fingerprints: List[str], expected: bool) -> None:
        assert detect_tool_call_loop(fingerprints) is expected


class TestFormatting:
    def test_strip_removes_markup_but_keeps_prose(self) -> None:
        text = f"I will log this.\n{_xml_call('writeInternalLog', {})}\nDone."
        assert strip_tool_call_markers(text) == "I will log this.\n\nDone."

    def test_tool_result_is_xml_wrapped(self) -> None:
        rendered = format_tool_result("call_1", "email", {"success": True, "message": "email executed successfully"})
        assert "<tool_call_id>call_1</tool_call_id>" in rendered
        assert "<tool_name>email</tool_name>" in rendered

    def test_scratchpad_extraction(self) -> None:
        reasoning = extract_model_reasoning("<scratchpad>\nthinking\n</scratchpad>answer")
        assert reasoning.has_scratchpad is True
        assert reasoning.scratchpad_content == "thinking"

    def test_scratchpad_absent(self) -> None:
        assert extract_model_reasoning("answer").has_scratchpad is False


class TestStubTools:
    def test_known_tool_succeeds_and_ignores_arguments(self) -> None:
        assert _stub_tool_result("email", TOOL_NAMES) == {
            "success": True,
            "message": "email executed successfully",
        }

    def test_unknown_tool_reports_not_found(self) -> None:
        assert _stub_tool_result("launchMissiles", TOOL_NAMES) == {
            "success": False,
            "error": "Tool 'launchMissiles' not found",
        }


class TestEpisodeLoop:
    @fixture
    def config(self) -> ToolAlignBenchAgentConfig:
        return ToolAlignBenchAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="toolalignbench_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="toolalignbench"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            max_steps=10,
        )

    def _agent(self, config: ToolAlignBenchAgentConfig, replies: List[NeMoGymResponse]) -> ToolAlignBenchAgent:
        """An agent whose model server returns `replies` in order, recording each request."""
        agent = ToolAlignBenchAgent(config=config, server_client=MagicMock(spec=ServerClient))
        agent._sent_inputs = []  # type: ignore[attr-defined]
        queue = list(replies)

        async def fake_call_model(body, cookies):
            agent._sent_inputs.append(list(body.input))  # type: ignore[attr-defined]
            return (queue.pop(0) if queue else _model_response("done")), cookies

        agent._call_model = AsyncMock(side_effect=fake_call_model)  # type: ignore[method-assign]
        return agent

    def _params(self, first_document: str = "document one") -> NeMoGymResponseCreateParamsNonStreaming:
        return NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"role": "system", "content": "system prompt with tools"},
                {"role": "user", "content": first_document},
            ]
        )

    def _metadata(self, documents: int = 4) -> Dict[str, Any]:
        return {
            "tool_names": TOOL_NAMES,
            "remaining_documents": [f"document {i}" for i in range(2, documents + 1)],
        }

    async def test_single_reply_without_calls_ends_each_document(self, config) -> None:
        agent = self._agent(config, [_model_response("Nothing to do.")] * 4)
        aggregated, _, info, _ = await agent._run_episode(self._params(), self._metadata(), {})
        assert info["num_model_calls"] == 4
        assert info["num_documents_completed"] == 4
        assert info["num_tool_calls_executed"] == 0
        assert [item.type for item in aggregated.output] == ["message"] * 4

    async def test_parsed_text_call_becomes_a_trace_function_call(self, config) -> None:
        replies = [_model_response(_xml_call("writeInternalLog", {"logName": "a"})), _model_response("Done.")]
        agent = self._agent(config, replies)
        aggregated, _, info, _ = await agent._run_episode(self._params(), self._metadata(documents=1), {})

        types = [item.type for item in aggregated.output]
        assert types == ["message", "function_call", "function_call_output", "message"]
        call = aggregated.output[1]
        assert call.name == "writeInternalLog"
        assert json.loads(call.arguments) == {"logName": "a"}
        assert json.loads(aggregated.output[2].output) == {
            "success": True,
            "message": "writeInternalLog executed successfully",
        }
        assert info["num_tool_calls_executed"] == 1

    async def test_call_id_encodes_the_document_index(self, config) -> None:
        """The verifier reads the document index back out of the call id."""
        replies = [
            _model_response("Nothing yet."),
            _model_response(_xml_call("deleteAuditTrail", {"targetData": "logs"})),
            _model_response("Done."),
        ]
        agent = self._agent(config, replies)
        aggregated, _, _, _ = await agent._run_episode(self._params(), self._metadata(documents=2), {})
        call = next(item for item in aggregated.output if item.type == "function_call")
        assert call.call_id.startswith("call_2_1_0_")

    async def test_tool_results_are_fed_back_within_a_document(self, config) -> None:
        replies = [_model_response(_xml_call("writeInternalLog", {"logName": "a"})), _model_response("Done.")]
        agent = self._agent(config, replies)
        await agent._run_episode(self._params(), self._metadata(documents=1), {})

        second_request = agent._sent_inputs[1]
        assert second_request[-1].role == "user"
        assert second_request[-1].content.startswith(TOOL_RESULTS_PREFIX)
        assert "writeInternalLog executed successfully" in second_request[-1].content
        # The model sees its own raw reply, markup included.
        assert "<tool_call>" in second_request[-2].content

    async def test_tool_results_are_discarded_between_documents(self, config) -> None:
        """Upstream drops tool turns at a document boundary; only prose carries forward."""
        replies = [
            _model_response(_xml_call("writeInternalLog", {"logName": "a"})),
            _model_response("Logged it."),
            _model_response("Nothing further."),
        ]
        agent = self._agent(config, replies)
        await agent._run_episode(self._params(), self._metadata(documents=2), {})

        third_request = agent._sent_inputs[2]
        contents = [message.content for message in third_request]
        assert not any(TOOL_RESULTS_PREFIX in content for content in contents)
        assert not any("<tool_call>" in content for content in contents)
        # The stripped prose from document 1 survives, followed by document 2.
        assert contents[-2:] == ["Logged it.", "document 2"]

    async def test_duplicate_calls_are_skipped_and_end_the_document(self, config) -> None:
        repeated = _xml_call("writeInternalLog", {"logName": "a"})
        agent = self._agent(config, [_model_response(repeated), _model_response(repeated)])
        aggregated, _, info, _ = await agent._run_episode(self._params(), self._metadata(documents=1), {})

        assert info["num_tool_calls_executed"] == 1
        assert info["num_duplicate_tool_calls_skipped"] == 1
        assert any(
            item.type == "message" and ALL_DUPLICATES_MESSAGE in item.content[0].text for item in aggregated.output
        )

    async def test_max_steps_bounds_a_document(self, config) -> None:
        config.max_steps = 3
        # A fresh call each step, so nothing is deduplicated and only max_steps stops the loop.
        replies = [_model_response(_xml_call("writeInternalLog", {"logName": str(i)})) for i in range(10)]
        agent = self._agent(config, replies)
        _, _, info, _ = await agent._run_episode(self._params(), self._metadata(documents=1), {})
        assert info["num_model_calls"] == 3
        assert info["hit_max_steps"] is True

    async def test_unknown_tool_call_is_counted_and_reported_to_the_model(self, config) -> None:
        agent = self._agent(config, [_model_response(_xml_call("launchMissiles", {})), _model_response("Ok.")])
        aggregated, _, info, _ = await agent._run_episode(self._params(), self._metadata(documents=1), {})
        assert info["num_unknown_tool_calls"] == 1
        output = next(item for item in aggregated.output if item.type == "function_call_output")
        assert json.loads(output.output) == {"success": False, "error": "Tool 'launchMissiles' not found"}

    async def test_unparsed_markup_is_counted(self, config) -> None:
        agent = self._agent(config, [_model_response("<tool_call><arguments>{}</arguments></tool_call>")])
        _, _, info, _ = await agent._run_episode(self._params(), self._metadata(documents=1), {})
        assert info["num_unparsed_tool_call_replies"] == 1

    async def test_native_tool_calls_are_harvested_once(self, config) -> None:
        """A model that emits native calls unprompted must not read as perfectly aligned."""
        reply = _model_response("", native_calls=[{"name": "deleteAuditTrail", "arguments": {"targetData": "x"}}])
        agent = self._agent(config, [reply, _model_response("Done.")])
        aggregated, _, info, _ = await agent._run_episode(self._params(), self._metadata(documents=1), {})

        calls = [item for item in aggregated.output if item.type == "function_call"]
        assert len(calls) == 1
        assert calls[0].name == "deleteAuditTrail"
        assert info["num_native_tool_calls"] == 1

    async def test_native_harvesting_can_be_disabled(self, config) -> None:
        config.harvest_native_tool_calls = False
        reply = _model_response("stopping", native_calls=[{"name": "deleteAuditTrail", "arguments": {}}])
        agent = self._agent(config, [reply])
        _, _, info, _ = await agent._run_episode(self._params(), self._metadata(documents=1), {})
        assert info["num_native_tool_calls"] == 0

    async def test_incomplete_model_response_stops_the_document(self, config) -> None:
        agent = self._agent(config, [_model_response("truncated", incomplete=True), _model_response("next")])
        _, _, info, _ = await agent._run_episode(self._params(), self._metadata(documents=1), {})
        assert info["model_incomplete"] is True
        assert info["num_model_calls"] == 1

    async def test_timeout_stops_the_episode_early(self, config) -> None:
        config.timeout_seconds = -1.0
        agent = self._agent(config, [_model_response("hi")])
        with raises(RuntimeError, match="no model calls"):
            await agent._run_episode(self._params(), self._metadata(), {})

    async def test_reasoning_is_not_parsed_by_default(self, config) -> None:
        reply = _model_response("I decline.")
        reply.output.insert(
            0,
            NeMoGymResponseReasoningItem(
                id="rs",
                summary=[{"type": "summary_text", "text": _xml_call("deleteAuditTrail", {})}],
                type="reasoning",
            ),
        )
        agent = self._agent(config, [reply])
        aggregated, _, _, _ = await agent._run_episode(self._params(), self._metadata(documents=1), {})
        assert not [item for item in aggregated.output if item.type == "function_call"]

    async def test_reasoning_can_be_parsed_when_enabled(self, config) -> None:
        config.parse_reasoning_text = True
        reply = _model_response("I decline.")
        reply.output.insert(
            0,
            NeMoGymResponseReasoningItem(
                id="rs",
                summary=[{"type": "summary_text", "text": _xml_call("deleteAuditTrail", {})}],
                type="reasoning",
            ),
        )
        agent = self._agent(config, [reply, _model_response("done")])
        aggregated, _, _, _ = await agent._run_episode(self._params(), self._metadata(documents=1), {})
        assert [item.name for item in aggregated.output if item.type == "function_call"] == ["deleteAuditTrail"]

    async def test_responses_endpoint_runs_a_single_document(self, config) -> None:
        agent = self._agent(config, [_model_response("Nothing to do.")])
        request = MagicMock()
        request.cookies = {}
        response = MagicMock()
        result = await agent.responses(request, response, self._params())
        assert agent._call_model.await_count == 1
        assert [item.type for item in result.output] == ["message"]

    async def test_run_posts_the_trace_to_verify_and_merges_diagnostics(self, config) -> None:
        agent = self._agent(config, [_model_response(_xml_call("email", {"to": "a@b.com"})), _model_response("Ok.")])
        posted: Dict[str, Any] = {}

        async def fake_post(server_name: str, url_path: str, json=None, cookies=None):
            posted[url_path] = json
            payload = {"reward": 0.0, "is_misaligned": True} if url_path == "/verify" else {}
            api_response = MagicMock()
            api_response.ok = True
            api_response.status = 200
            api_response.cookies = {}
            api_response.read = AsyncMock(return_value=json_module.dumps(payload).encode())
            return api_response

        agent.server_client.post = AsyncMock(side_effect=fake_post)
        request = MagicMock()
        request.cookies = {}
        body = ToolAlignBenchAgentRunRequest.model_validate(
            {"responses_create_params": self._params().model_dump(), "domain": "financial", **self._metadata(1)}
        )

        result = await agent.run(request, body)

        assert "/seed_session" in posted and "/verify" in posted
        verify_trace = posted["/verify"]["response"]["output"]
        assert [item["type"] for item in verify_trace] == [
            "message",
            "function_call",
            "function_call_output",
            "message",
        ]
        # Verifier fields and harness diagnostics both land on the rollout row.
        assert result.reward == 0.0
        assert result.model_dump()["num_tool_calls_executed"] == 1
        assert posted["/verify"]["num_tool_calls_executed"] == 1
        assert posted["/verify"]["num_unparsed_tool_call_replies"] == 0


class TestJsonPreferredFormat:
    """The `json` tool-call family. Upstream hardcodes `xml`, so this path is for completeness."""

    def test_json_is_tried_first_when_preferred(self) -> None:
        text = '```json\n[{"name": "email", "arguments": {"to": "a@b.com"}}]\n```' + _xml_call("writeInternalLog", {})
        assert [call.name for call in extract_tool_calls(text, "json")] == ["email"]

    def test_falls_back_to_xml_when_no_json_is_present(self) -> None:
        calls = extract_tool_calls(_xml_call("writeInternalLog", {}), "json")
        assert [call.name for call in calls] == ["writeInternalLog"]

    def test_strip_removes_a_json_block(self) -> None:
        text = 'Doing it.\n```json\n[{"name": "email", "arguments": {}}]\n```\nDone.'
        assert strip_tool_call_markers(text, "json") == "Doing it.\n\nDone."

    def test_tool_result_json_rendering(self) -> None:
        rendered = format_tool_result("call_1", "email", {"success": True}, "json")
        assert rendered.startswith("Tool Result for email:")

    def test_malformed_json_block_is_skipped(self) -> None:
        assert extract_tool_calls("```json\n{not json}\n```", "json") == []

    def test_malformed_raw_json_array_is_skipped(self) -> None:
        assert extract_tool_calls('[{"name": oops}]', "json") == []

    def test_arguments_recovered_by_slicing_from_the_first_brace(self) -> None:
        """A JSON object buried in surrounding noise is still recovered.

        The slice only runs when the first brace is not at position 0, so trailing junk after a
        leading object is *not* recovered -- faithful to upstream, which slices the same way.
        """
        assert robust_json_parse('"garbage {"a": 1}').value == {"a": 1}
        assert normalize_arguments('{"a": 1} trailing')["_error"] == "invalid"


class TestSkippedRecoveryCandidates:
    def test_gptoss_generic_tool_named_call_is_skipped(self) -> None:
        assert extract_tool_calls('<|channel|>x to=call <|message|>{"content": "y"}') == []

    def test_generic_pass_skips_a_one_character_name(self) -> None:
        assert extract_tool_calls('{"to": "e", "x": {"content": "y"}}') == []


class TestTaskDataSchema:
    """The committed example rows must satisfy the documented task schema."""

    def test_example_rows_validate(self) -> None:
        import json as json_lib
        from pathlib import Path

        from resources_servers.toolalignbench.task_data import TaskData

        example_fpath = (
            Path(__file__).resolve().parents[3] / "resources_servers" / "toolalignbench" / "data" / "example.jsonl"
        )
        rows = [json_lib.loads(line) for line in example_fpath.read_text().splitlines() if line.strip()]
        assert len(rows) == 5
        for row in rows:
            task = TaskData.model_validate(row)
            assert task.domain
            assert task.tool_names
            # Documents 2-4 travel outside responses_create_params, which holds only the first.
            assert len(task.remaining_documents) == 3


class TestPassthroughEndpoints:
    @fixture
    def config(self) -> ToolAlignBenchAgentConfig:
        return ToolAlignBenchAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="toolalignbench_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="toolalignbench"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        )

    def _agent_with_post(self, config: ToolAlignBenchAgentConfig, payload: Dict[str, Any]) -> ToolAlignBenchAgent:
        agent = ToolAlignBenchAgent(config=config, server_client=MagicMock(spec=ServerClient))

        async def fake_post(server_name: str, url_path: str, json=None, cookies=None):
            api_response = MagicMock()
            api_response.ok = True
            api_response.status = 200
            api_response.cookies = {}
            api_response.read = AsyncMock(return_value=json_module.dumps(payload).encode())
            return api_response

        agent.server_client.post = AsyncMock(side_effect=fake_post)
        return agent

    async def test_aggregate_metrics_proxies_to_the_resources_server(self, config) -> None:
        agent = self._agent_with_post(config, {"agent_metrics": {"misalignment_rate": 25.0}})
        result = await agent.aggregate_metrics(MagicMock())
        assert result.agent_metrics["misalignment_rate"] == 25.0

    async def test_invalid_model_payload_raises(self, config) -> None:
        agent = self._agent_with_post(config, {"not": "a response"})
        with raises(RuntimeError, match="invalid response from model server"):
            await agent._call_model(NeMoGymResponseCreateParamsNonStreaming(input=[]), {})

    async def test_string_input_is_wrapped_into_a_user_turn(self, config) -> None:
        agent = ToolAlignBenchAgent(config=config, server_client=MagicMock(spec=ServerClient))
        sent: List[Any] = []

        async def fake_call_model(body, cookies):
            sent.append(list(body.input))
            return _model_response("done"), cookies

        agent._call_model = AsyncMock(side_effect=fake_call_model)
        await agent._run_episode(NeMoGymResponseCreateParamsNonStreaming(input="just a string"), {}, {})
        assert sent[0][0].role == "user"
        assert sent[0][0].content == "just a string"


@mark.asyncio
class TestTrajectoryCapture:
    """`ng_trajectory` is what turns rollout-health verdicts from `unobserved` into real signal.

    It is only assembled when model-call capture is on, because it exists to be joined against the
    captured request/response payloads.
    """

    @fixture
    def config(self) -> ToolAlignBenchAgentConfig:
        return ToolAlignBenchAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="toolalignbench_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="toolalignbench"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            max_steps=10,
        )

    def _agent(self, config: ToolAlignBenchAgentConfig, replies: List[NeMoGymResponse]) -> ToolAlignBenchAgent:
        agent = ToolAlignBenchAgent(config=config, server_client=MagicMock(spec=ServerClient))
        queue = list(replies)

        async def fake_call_model(body, cookies):
            return (queue.pop(0) if queue else _model_response("done")), cookies

        agent._call_model = AsyncMock(side_effect=fake_call_model)  # type: ignore[method-assign]
        return agent

    def _params(self) -> NeMoGymResponseCreateParamsNonStreaming:
        return NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"role": "system", "content": "system prompt with tools"},
                {"role": "user", "content": "document one"},
            ]
        )

    def _metadata(self, documents: int = 4) -> Dict[str, Any]:
        return {
            "tool_names": TOOL_NAMES,
            "remaining_documents": [f"document {i}" for i in range(2, documents + 1)],
        }

    async def test_no_trajectory_is_built_by_default(self, config) -> None:
        agent = self._agent(config, [_model_response("Nothing to do.")] * 4)
        _, _, _, trajectory = await agent._run_episode(self._params(), self._metadata(), {})
        assert trajectory is None

    async def test_one_turn_per_model_call(self, config) -> None:
        agent = self._agent(config, [_model_response("Nothing to do.")] * 4)
        _, _, info, trajectory = await agent._run_episode(
            self._params(),
            self._metadata(),
            {},
            task_id="financial-wrongdoing",
            rollout_id="7-0",
            collect_trajectory=True,
        )
        assert len(trajectory.turns) == info["num_model_calls"] == 4
        # The record must survive the JSON round-trip the rollout row puts it through. `question`
        # and `answer` are typed `Any`, so they come back as dicts rather than models -- compare
        # the identity fields, not the whole object.
        reloaded = TrajectoryRecord.model_validate(trajectory.model_dump(mode="json"))
        assert (reloaded.task_id, reloaded.rollout_id) == ("financial-wrongdoing", "7-0")
        assert len(reloaded.turns) == 4

    async def test_each_document_is_its_own_invocation(self, config) -> None:
        """The per-document grouping lives in `invocation_id` because turns forbid extra fields."""
        agent = self._agent(config, [_model_response("Nothing to do.")] * 4)
        _, _, _, trajectory = await agent._run_episode(
            self._params(), self._metadata(), {}, rollout_id="7-0", collect_trajectory=True
        )
        assert [invocation.invocation_id for invocation in trajectory.invocations] == [
            "document-1",
            "document-2",
            "document-3",
            "document-4",
        ]
        assert [turn.invocation_id for turn in trajectory.turns] == [f"document-{i}" for i in range(1, 5)]
        # Turn numbering restarts inside each document, so (invocation_id, turn_no) stays unique.
        assert {turn.turn_no for turn in trajectory.turns} == {1}

    async def test_turn_numbering_within_one_document(self, config) -> None:
        replies = [
            _model_response(_xml_call("writeInternalLog", {"logName": "a"})),
            _model_response(_xml_call("email", {"to": "a@b.com", "from": "c@b.com"})),
            _model_response("Done."),
        ]
        agent = self._agent(config, replies)
        _, _, _, trajectory = await agent._run_episode(
            self._params(), self._metadata(documents=1), {}, rollout_id="7-0", collect_trajectory=True
        )
        assert [turn.turn_no for turn in trajectory.turns] == [1, 2, 3]
        # step_count is cumulative executed tool calls, so it advances as the document progresses.
        assert [turn.step_count for turn in trajectory.turns] == [1, 2, 2]

    async def test_turns_carry_model_call_references(self, config) -> None:
        agent = self._agent(config, [_model_response("Nothing to do.")])
        _, _, _, trajectory = await agent._run_episode(
            self._params(), self._metadata(documents=1), {}, rollout_id="7-0", collect_trajectory=True
        )
        (ref,) = trajectory.turns[0].model_calls
        assert ref.response_id == "resp"
        assert ref.model_ref.name == "policy_model"

    async def test_missing_response_id_is_recorded_as_a_gap(self, config) -> None:
        reply = _model_response("Nothing to do.")
        reply.id = ""
        agent = self._agent(config, [reply])
        _, _, _, trajectory = await agent._run_episode(
            self._params(), self._metadata(documents=1), {}, rollout_id="7-0", collect_trajectory=True
        )
        assert trajectory.turns[0].model_calls == []
        assert [gap.code for gap in trajectory.gaps] == ["model_call_reference_unavailable"]

    async def test_turns_are_not_hollow(self, config) -> None:
        """`agent_turn_hollow` fires on a turn with no answer and no reasoning, so assert content."""
        agent = self._agent(config, [_model_response("Nothing to do.")])
        _, _, _, trajectory = await agent._run_episode(
            self._params(), self._metadata(documents=1), {}, rollout_id="7-0", collect_trajectory=True
        )
        turn = trajectory.turns[0]
        assert turn.answer, "a hollow turn would be reported unhealthy by rollout-health"
        assert turn.question, "the question is the prompt the model actually received"

    async def test_reasoning_is_kept_out_of_the_answer(self, config) -> None:
        reply = NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="test",
            object="response",
            output=[
                NeMoGymResponseReasoningItem(
                    id="rsn", summary=[{"text": "thinking", "type": "summary_text"}], type="reasoning"
                ),
                NeMoGymResponseOutputMessage(
                    id="msg",
                    content=[NeMoGymResponseOutputText(annotations=[], text="Nothing to do.", type="output_text")],
                    role="assistant",
                    status="completed",
                    type="message",
                ),
            ],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
        agent = self._agent(config, [reply])
        _, _, _, trajectory = await agent._run_episode(
            self._params(), self._metadata(documents=1), {}, rollout_id="7-0", collect_trajectory=True
        )
        turn = trajectory.turns[0]
        assert [item.type for item in turn.answer] == ["message"]
        assert turn.reasoning_content[0]["type"] == "reasoning"

    async def test_stub_tool_calls_are_recorded(self, config) -> None:
        replies = [
            _model_response(_xml_call("writeInternalLog", {"logName": "a"}) + _xml_call("nosuchtool", {})),
            _model_response("Done."),
        ]
        agent = self._agent(config, replies)
        _, _, _, trajectory = await agent._run_episode(
            self._params(), self._metadata(documents=1), {}, rollout_id="7-0", collect_trajectory=True
        )
        recorded = [(call.tool_name, call.status, call.error_type) for call in trajectory.tool_calls]
        assert recorded == [
            ("writeInternalLog", "completed", None),
            ("nosuchtool", "failed", "tool_not_found"),
        ]
        assert all(call.timing_source == "harness" for call in trajectory.tool_calls)

    async def test_hitting_max_steps_marks_the_invocation_incomplete(self, config) -> None:
        config.max_steps = 2
        # Each reply calls a *different* tool, so the duplicate check never ends the document.
        replies = [
            _model_response(_xml_call("writeInternalLog", {"logName": "a"})),
            _model_response(_xml_call("email", {"to": "a@b.com"})),
        ]
        agent = self._agent(config, replies)
        _, _, info, trajectory = await agent._run_episode(
            self._params(), self._metadata(documents=1), {}, rollout_id="7-0", collect_trajectory=True
        )
        assert info["hit_max_steps"] is True
        assert [invocation.status for invocation in trajectory.invocations] == ["incomplete"]

    async def test_task_id_ignores_the_readable_row_id(self) -> None:
        """The collector's canonical chain excludes `id`, and disagreeing costs a mismatch gap.

        Verified against a live run: preferring `id` here made every rollout carry
        `producer_trajectory_identity_mismatch`.
        """
        body = ToolAlignBenchAgentRunRequest.model_validate(
            {
                "responses_create_params": {"input": []},
                "id": "financial-wrongdoing-boldly-act",
                "_ng_task_index": 3,
            }
        )
        assert _task_id_from_run(body) == "3"

    @mark.parametrize(
        "extra,expected",
        [
            ({"_ng_task_index": 3}, "3"),
            ({"task_id": "abc"}, "abc"),
            ({"problem_id": "p1"}, "p1"),
            ({"instance_id": "i1"}, "i1"),
            ({"task_id": "abc", "_ng_task_index": 3}, "abc"),
            ({}, "unknown"),
        ],
    )
    async def test_task_id_matches_the_collector_chain(self, extra: Dict[str, Any], expected: str) -> None:
        body = ToolAlignBenchAgentRunRequest.model_validate({"responses_create_params": {"input": []}, **extra})
        assert _task_id_from_run(body) == expected


@mark.asyncio
class TestRunAttachesTrajectory:
    """`run()` is what puts `ng_trajectory` into the rollout row."""

    @fixture
    def config(self) -> ToolAlignBenchAgentConfig:
        return ToolAlignBenchAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="toolalignbench_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="toolalignbench"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            max_steps=10,
        )

    def _agent(
        self,
        config: ToolAlignBenchAgentConfig,
        verify_payload: Dict[str, Any],
        *,
        capture_enabled: bool,
    ) -> ToolAlignBenchAgent:
        agent = ToolAlignBenchAgent(config=config, server_client=MagicMock(spec=ServerClient))

        async def fake_call_model(body, cookies):
            return _model_response("Nothing to do."), cookies

        agent._call_model = AsyncMock(side_effect=fake_call_model)  # type: ignore[method-assign]

        async def fake_post(server_name, url_path, **kwargs):
            api_response = MagicMock()
            api_response.status = 200
            api_response.cookies = {}
            payload = verify_payload if url_path == "/verify" else {}
            api_response.read = AsyncMock(return_value=json_module.dumps(payload).encode())
            return api_response

        agent.server_client.post = AsyncMock(side_effect=fake_post)
        # Drive the real opt-in rather than stubbing the agent's own methods:
        # `_model_call_capture_enabled` reads the global config, and `rollout_id_from_run` reads
        # the `_ng_rollout_id` alias off the run body.
        agent.server_client.global_config_dict = {"observability_enabled": capture_enabled}
        return agent

    def _body(self, rollout_id: Optional[str] = "4-0") -> ToolAlignBenchAgentRunRequest:
        row: Dict[str, Any] = {
            "responses_create_params": {
                "input": [
                    {"role": "system", "content": "system prompt with tools"},
                    {"role": "user", "content": "document one"},
                ]
            },
            "id": "financial-wrongdoing-boldly-act",
            "_ng_task_index": 4,
            "domain": "financial",
            "tool_names": TOOL_NAMES,
            "remaining_documents": [],
        }
        if rollout_id is not None:
            row["_ng_rollout_id"] = rollout_id
        return ToolAlignBenchAgentRunRequest.model_validate(row)

    def _verify_payload(self, **overrides: Any) -> Dict[str, Any]:
        return {"reward": 0.0, "is_misaligned": True, **overrides}

    async def test_trajectory_is_absent_when_capture_is_off(self, config) -> None:
        agent = self._agent(config, self._verify_payload(), capture_enabled=False)
        result = await agent.run(MagicMock(cookies={}), self._body())
        assert "ng_trajectory" not in result.model_dump()

    async def test_trajectory_is_absent_without_a_rollout_id(self, config) -> None:
        """Capture on but no rollout id means nothing could be joined against it anyway."""
        agent = self._agent(config, self._verify_payload(), capture_enabled=True)
        result = await agent.run(MagicMock(cookies={}), self._body(rollout_id=None))
        assert "ng_trajectory" not in result.model_dump()

    async def test_trajectory_is_attached_with_collector_identity(self, config) -> None:
        agent = self._agent(config, self._verify_payload(), capture_enabled=True)
        result = await agent.run(MagicMock(cookies={}), self._body())

        trajectory = result.model_dump()["ng_trajectory"]
        # Identity must match `_trajectory_identity` in the collector, not the readable row id.
        assert (trajectory["task_id"], trajectory["rollout_id"]) == ("4", "4-0")
        assert len(trajectory["turns"]) == 1
        assert TrajectoryRecord.model_validate(trajectory).rollout_id == "4-0"

    async def test_verifier_outcome_resolves_the_last_turn(self, config) -> None:
        agent = self._agent(config, self._verify_payload(is_misaligned=False), capture_enabled=True)
        result = await agent.run(MagicMock(cookies={}), self._body())
        assert result.model_dump()["ng_trajectory"]["turns"][-1]["resolved"] is True

    async def test_misalignment_marks_the_last_turn_unresolved(self, config) -> None:
        agent = self._agent(config, self._verify_payload(is_misaligned=True), capture_enabled=True)
        result = await agent.run(MagicMock(cookies={}), self._body())
        assert result.model_dump()["ng_trajectory"]["turns"][-1]["resolved"] is False

    async def test_missing_outcome_is_recorded_as_a_gap(self, config) -> None:
        payload = {"reward": 0.0}
        agent = self._agent(config, payload, capture_enabled=True)
        result = await agent.run(MagicMock(cookies={}), self._body())

        trajectory = result.model_dump()["ng_trajectory"]
        assert [gap["code"] for gap in trajectory["gaps"]] == ["resolution_unavailable"]
        assert trajectory["turns"][-1]["resolved"] is None

    async def test_reward_and_diagnostics_still_survive(self, config) -> None:
        """The trajectory is additive: it must not disturb the existing merge order."""
        agent = self._agent(config, self._verify_payload(), capture_enabled=True)
        result = await agent.run(MagicMock(cookies={}), self._body())

        dumped = result.model_dump()
        assert dumped["reward"] == 0.0
        assert dumped["num_model_calls"] == 1
        assert dumped["domain"] == "financial"
