# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from resources_servers.browser_gym.schemas import BrowserAction, CUAStep, CUATrajectory
from responses_api_agents.browser_agent.adapters.base import CUAAdapterResponse
from responses_api_agents.browser_agent.trajectory_writer import (
    _looks_like_base64,
    _strip_base64_fields,
    append_debug_step,
    finalize_debug_trajectory,
    init_debug_trajectory,
    save_debug_trajectory,
)


class TestCUAAdapterResponse:
    def test_default(self):
        resp = CUAAdapterResponse()
        assert resp.actions == []
        assert resp.message is None
        assert resp.done is False

    def test_with_actions(self):
        actions = [BrowserAction(action_type="click", coordinate=[100, 200])]
        resp = CUAAdapterResponse(actions=actions, done=False)
        assert len(resp.actions) == 1

    def test_done_with_message(self):
        resp = CUAAdapterResponse(done=True, message="Task completed")
        assert resp.done is True
        assert resp.message == "Task completed"


class TestOpenAIAdapter:
    def test_action_parsing_click(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter()
        action = adapter._map_openai_action({"type": "click", "coordinate": [100, 200]})
        assert action.action_type == "click"
        assert action.coordinate == [100, 200]

    def test_action_parsing_type(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter()
        action = adapter._map_openai_action({"type": "type", "text": "hello world"})
        assert action.action_type == "type"
        assert action.text == "hello world"

    def test_action_parsing_scroll(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter()
        action = adapter._map_openai_action(
            {"type": "scroll", "coordinate": [640, 360], "scroll_x": 0, "scroll_y": 300}
        )
        assert action.action_type == "scroll"
        assert action.scroll_y == 300

    def test_action_parsing_keypress(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter()
        action = adapter._map_openai_action({"type": "keypress", "keys": ["Control", "a"]})
        assert action.action_type == "keypress"
        assert action.keys == ["Control", "a"]

    def test_action_parsing_drag(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter()
        action = adapter._map_openai_action(
            {"type": "drag", "start_x": 10, "start_y": 20, "destination_x": 100, "destination_y": 200}
        )
        assert action.action_type == "drag"
        assert action.start_coordinate == [10, 20]
        assert action.end_coordinate == [100, 200]

    def test_action_parsing_goto(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter()
        action = adapter._map_openai_action({"type": "goto", "url": "https://example.com"})
        assert action.action_type == "goto"
        assert action.url == "https://example.com"

    def test_action_parsing_tab_actions(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter()
        assert adapter._map_openai_action({"type": "new_tab", "url": "https://example.com"}).action_type == "new_tab"
        assert adapter._map_openai_action({"type": "close_tab"}).action_type == "close_tab"
        assert adapter._map_openai_action({"type": "switch_tab", "tab_index": 1}).action_type == "switch_tab"

    def test_action_parsing_unknown(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter()
        action = adapter._map_openai_action({"type": "unknown_action"})
        assert action is None

    def test_tools_config(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter(viewport_width=1024, viewport_height=768)
        tools = adapter._get_tools()
        assert tools[0]["type"] == "computer_use_preview"
        assert tools[0]["display_width"] == 1024
        assert tools[0]["display_height"] == 768

    def test_parse_actions_computer_call(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter()
        response = {
            "id": "resp_123",
            "output": [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "action": {"type": "click", "coordinate": [100, 200]},
                }
            ],
        }
        result = adapter._parse_actions(response)
        assert len(result.actions) == 1
        assert result.actions[0].action_type == "click"
        assert result.done is False

    def test_parse_actions_final_message(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter()
        response = {
            "id": "resp_123",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done!"}],
                }
            ],
        }
        result = adapter._parse_actions(response)
        assert len(result.actions) == 0
        assert result.done is True
        assert result.message == "Done!"

    def test_parse_actions_extracts_usage(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter()
        response = {
            "id": "resp_123",
            "output": [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "action": {"type": "click", "coordinate": [100, 200]},
                }
            ],
            "usage": {
                "input_tokens": 500,
                "output_tokens": 120,
                "total_tokens": 620,
            },
        }
        result = adapter._parse_actions(response)
        assert result.usage is not None
        assert result.usage.input_tokens == 500
        assert result.usage.output_tokens == 120
        assert result.usage.total_tokens == 620

    def test_parse_actions_no_usage(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter()
        response = {
            "id": "resp_123",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done!"}],
                }
            ],
        }
        result = adapter._parse_actions(response)
        assert result.usage is None

    def test_reset(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter()
        adapter._last_response_id = "resp_123"
        adapter._pending_call_ids = ["call_1"]
        adapter.reset()
        assert adapter._last_response_id is None
        assert adapter._pending_call_ids == []

    def test_response_id_tracking(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter()
        response = {
            "id": "resp_abc123",
            "output": [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "action": {"type": "screenshot"},
                }
            ],
        }
        adapter._parse_actions(response)
        assert adapter._pending_call_ids == ["call_1"]


class TestAnthropicAdapterParsing:
    def test_action_mapping_left_click(self):
        from responses_api_agents.browser_agent.adapters.anthropic_adapter import AnthropicCUAAdapter

        adapter = AnthropicCUAAdapter.__new__(AnthropicCUAAdapter)
        action = adapter._map_anthropic_action({"action": "left_click", "coordinate": [100, 200]})
        assert action.action_type == "click"
        assert action.button == "left"

    def test_action_mapping_right_click(self):
        from responses_api_agents.browser_agent.adapters.anthropic_adapter import AnthropicCUAAdapter

        adapter = AnthropicCUAAdapter.__new__(AnthropicCUAAdapter)
        action = adapter._map_anthropic_action({"action": "right_click", "coordinate": [100, 200]})
        assert action.action_type == "right_click"

    def test_action_mapping_key(self):
        from responses_api_agents.browser_agent.adapters.anthropic_adapter import AnthropicCUAAdapter

        adapter = AnthropicCUAAdapter.__new__(AnthropicCUAAdapter)
        action = adapter._map_anthropic_action({"action": "key", "text": "Enter"})
        assert action.action_type == "keypress"
        assert action.key == "Enter"

    def test_action_mapping_mouse_move(self):
        from responses_api_agents.browser_agent.adapters.anthropic_adapter import AnthropicCUAAdapter

        adapter = AnthropicCUAAdapter.__new__(AnthropicCUAAdapter)
        action = adapter._map_anthropic_action({"action": "mouse_move", "coordinate": [300, 400]})
        assert action.action_type == "hover"
        assert action.coordinate == [300, 400]

    def test_action_mapping_scroll(self):
        from responses_api_agents.browser_agent.adapters.anthropic_adapter import AnthropicCUAAdapter

        adapter = AnthropicCUAAdapter.__new__(AnthropicCUAAdapter)
        action = adapter._map_anthropic_action(
            {"action": "scroll", "coordinate": [640, 360], "direction": "down", "amount": 3}
        )
        assert action.action_type == "scroll"
        assert action.scroll_direction == "down"
        assert action.scroll_amount == 3

    def test_action_mapping_drag(self):
        from responses_api_agents.browser_agent.adapters.anthropic_adapter import AnthropicCUAAdapter

        adapter = AnthropicCUAAdapter.__new__(AnthropicCUAAdapter)
        action = adapter._map_anthropic_action(
            {"action": "left_click_drag", "start_coordinate": [10, 20], "coordinate": [100, 200]}
        )
        assert action.action_type == "drag"

    def test_action_mapping_unknown(self):
        from responses_api_agents.browser_agent.adapters.anthropic_adapter import AnthropicCUAAdapter

        adapter = AnthropicCUAAdapter.__new__(AnthropicCUAAdapter)
        action = adapter._map_anthropic_action({"action": "unknown_action"})
        assert action is None

    def test_beta_flags_sonnet(self):
        from responses_api_agents.browser_agent.adapters.anthropic_adapter import AnthropicCUAAdapter

        adapter = AnthropicCUAAdapter.__new__(AnthropicCUAAdapter)
        adapter._is_opus = False
        adapter._effort_level = "high"
        flags = adapter._get_beta_flags()
        assert "computer-use-2025-01-24" in flags
        assert "token-efficient-tools-2025-02-19" in flags
        assert "context-management-2025-06-27" in flags

    def test_beta_flags_opus(self):
        from responses_api_agents.browser_agent.adapters.anthropic_adapter import AnthropicCUAAdapter

        adapter = AnthropicCUAAdapter.__new__(AnthropicCUAAdapter)
        adapter._is_opus = True
        adapter._effort_level = "medium"
        flags = adapter._get_beta_flags()
        assert "computer-use-2025-11-24" in flags
        assert "effort-2025-11-24" in flags

    def test_trim_conversation_history_preserves_first(self):
        from responses_api_agents.browser_agent.adapters.anthropic_adapter import AnthropicCUAAdapter

        adapter = AnthropicCUAAdapter.__new__(AnthropicCUAAdapter)
        adapter._turns_to_keep = 2
        adapter._screenshot_turn_limit = 10

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "initial task"}]},
        ]
        for i in range(5):
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": f"tu_{i}", "name": "computer", "input": {}}],
                }
            )
            messages.append(
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"tu_{i}", "content": "ok"}]}
            )

        result = adapter._trim_conversation_history(messages)
        assert result[0]["content"][0]["text"] == "initial task"
        assert len(result) < len(messages)

    def test_strip_leading_orphaned_tool_results(self):
        from responses_api_agents.browser_agent.adapters.anthropic_adapter import AnthropicCUAAdapter

        adapter = AnthropicCUAAdapter.__new__(AnthropicCUAAdapter)
        messages = [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "orphan"}]},
            {"role": "user", "content": [{"type": "text", "text": "real message"}]},
        ]
        result = adapter._strip_leading_orphaned_tool_results(messages)
        assert result[0]["content"][0]["text"] == "real message"


def _make_gemini_adapter(viewport_width=1280, viewport_height=720):
    """Create a GeminiCUAAdapter with viewport attrs set, without importing google.genai."""
    from responses_api_agents.browser_agent.adapters.gemini_adapter import GeminiCUAAdapter

    adapter = GeminiCUAAdapter.__new__(GeminiCUAAdapter)
    adapter._viewport_width = viewport_width
    adapter._viewport_height = viewport_height
    adapter._current_url = "about:blank"
    adapter._contents = []
    adapter._pending_action_names = []
    adapter._model = "gemini-2.5-computer-use-preview-10-2025"
    adapter._is_gemini_3 = False
    adapter._max_conversation_turns = 8
    adapter._api_caller = None
    return adapter


class TestGeminiAdapterParsing:
    def test_action_mapping_click_at(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("click_at", {"x": 500, "y": 500})
        assert action.action_type == "click"
        assert action.coordinate == [640, 360]

    def test_action_mapping_hover_at(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("hover_at", {"x": 0, "y": 0})
        assert action.action_type == "hover"
        assert action.coordinate == [0, 0]

    def test_action_mapping_hover_at_max(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("hover_at", {"x": 999, "y": 999})
        assert action.action_type == "hover"
        assert action.coordinate == [int(999 / 1000 * 1280), int(999 / 1000 * 720)]

    def test_action_mapping_type_text_at(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("type_text_at", {"x": 500, "y": 500, "text": "hello", "press_enter": True})
        assert action.action_type == "type"
        assert action.text == "hello"
        assert action.press_enter is True
        assert action.coordinate == [640, 360]

    def test_action_mapping_type_text_at_clear_default(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("type_text_at", {"x": 0, "y": 0, "text": "a"})
        assert action.clear_before_typing is True

    def test_action_mapping_scroll_at(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("scroll_at", {"x": 500, "y": 500, "direction": "up", "magnitude": 800})
        assert action.action_type == "scroll"
        assert action.scroll_direction == "up"
        assert action.coordinate == [640, 360]
        assert action.scroll_y < 0

    def test_action_mapping_scroll_document_down(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("scroll_document", {"direction": "down"})
        assert action.action_type == "scroll"
        assert action.scroll_direction == "down"
        assert action.scroll_y == int(720 * 0.8)
        assert action.scroll_x == 0

    def test_action_mapping_scroll_document_left(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("scroll_document", {"direction": "left"})
        assert action.scroll_x == -(1280 // 2)
        assert action.scroll_y == 0

    def test_action_mapping_navigate(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("navigate", {"url": "https://example.com"})
        assert action.action_type == "goto"
        assert action.url == "https://example.com"

    def test_action_mapping_navigate_bare_url(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("navigate", {"url": "example.com"})
        assert action.url == "https://example.com"

    def test_action_mapping_search(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("search", {"query": "test query"})
        assert action.action_type == "goto"
        assert action.url == "https://www.google.com"

    def test_action_mapping_drag_and_drop(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action(
            "drag_and_drop", {"x": 500, "y": 500, "destination_x": 750, "destination_y": 250}
        )
        assert action.action_type == "drag"
        assert action.start_coordinate == [640, 360]
        assert action.end_coordinate == [int(750 / 1000 * 1280), int(250 / 1000 * 720)]

    def test_action_mapping_keypress(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("keypress", {"key": "Enter"})
        assert action.action_type == "keypress"

    def test_action_mapping_keypress_combo_string(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("keypress", {"keys": "Control+a"})
        assert action.keys == ["Control", "a"]

    def test_action_mapping_key_combination(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("key_combination", {"keys": ["Control", "v"]})
        assert action.action_type == "keypress"
        assert action.keys == ["Control", "v"]

    def test_action_mapping_go_back(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("go_back", {})
        assert action.action_type == "go_back"

    def test_action_mapping_go_forward(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("go_forward", {})
        assert action.action_type == "go_forward"

    def test_action_mapping_wait(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("wait_5_seconds", {})
        assert action.action_type == "wait"
        assert action.duration == 5000

    def test_action_mapping_new_tab(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("new_tab", {"url": "https://test.com"})
        assert action.action_type == "new_tab"
        assert action.url == "https://test.com"

    def test_action_mapping_switch_tab(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("switch_tab", {"index": 2})
        assert action.action_type == "switch_tab"
        assert action.tab_index == 2

    def test_action_mapping_close_tab(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("close_tab", {})
        assert action.action_type == "close_tab"

    def test_action_mapping_list_tabs(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("list_tabs", {})
        assert action.action_type == "screenshot"

    def test_action_mapping_open_web_browser(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("open_web_browser", {})
        assert action.action_type == "screenshot"

    def test_action_mapping_unknown(self):
        adapter = _make_gemini_adapter()
        action = adapter._map_gemini_action("unknown_action", {})
        assert action is None


class TestGeminiDenormalization:
    def test_denorm_x(self):
        adapter = _make_gemini_adapter(viewport_width=1280)
        assert adapter._denorm_x(0) == 0
        assert adapter._denorm_x(500) == 640
        assert adapter._denorm_x(1000) == 1280

    def test_denorm_y(self):
        adapter = _make_gemini_adapter(viewport_height=720)
        assert adapter._denorm_y(0) == 0
        assert adapter._denorm_y(500) == 360
        assert adapter._denorm_y(1000) == 720

    def test_denorm_custom_viewport(self):
        adapter = _make_gemini_adapter(viewport_width=1920, viewport_height=1080)
        assert adapter._denorm_x(500) == 960
        assert adapter._denorm_y(500) == 540

    def test_click_at_denormalizes(self):
        adapter = _make_gemini_adapter(viewport_width=1000, viewport_height=1000)
        action = adapter._map_gemini_action("click_at", {"x": 500, "y": 250})
        assert action.coordinate == [500, 250]

    def test_drag_denormalizes_all_coords(self):
        adapter = _make_gemini_adapter(viewport_width=1000, viewport_height=1000)
        action = adapter._map_gemini_action(
            "drag_and_drop", {"x": 100, "y": 200, "destination_x": 800, "destination_y": 900}
        )
        assert action.start_coordinate == [100, 200]
        assert action.end_coordinate == [800, 900]


class TestGeminiUrlTracking:
    def test_initial_url(self):
        adapter = _make_gemini_adapter()
        assert adapter._current_url == "about:blank"

    def test_reset_clears_url(self):
        adapter = _make_gemini_adapter()
        adapter._current_url = "https://example.com"
        adapter.reset()
        assert adapter._current_url == "about:blank"


class TestAdapterFactory:
    def test_create_openai(self):
        from responses_api_agents.browser_agent.adapters import AdapterFactory
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = AdapterFactory.create("openai")
        assert isinstance(adapter, OpenAICUAAdapter)

    def test_create_unknown(self):
        from responses_api_agents.browser_agent.adapters import AdapterFactory

        with pytest.raises(ValueError, match="Unknown adapter type"):
            AdapterFactory.create("nonexistent")

    def test_available_adapters(self):
        from responses_api_agents.browser_agent.adapters import AdapterFactory

        adapters = AdapterFactory.available_adapters()
        assert "openai" in adapters


class TestBuildNemoResponse:
    def test_builds_valid_response(self):
        from responses_api_agents.browser_agent.app import _build_nemo_response

        traj = CUATrajectory(
            steps=[],
            task_prompt="test task",
            initial_screenshot="base64data",
            final_message="All done!",
        )
        resp = _build_nemo_response(traj, "env-123", '{"key": "val"}', "computer-use-preview")

        assert resp.id.startswith("cua_")
        assert resp.model == "computer-use-preview"
        assert resp.object == "response"
        assert resp.env_id == "env-123"
        assert resp.trajectory == traj
        assert resp.local_storage_dump == '{"key": "val"}'
        assert resp.parallel_tool_calls is False
        assert len(resp.output) == 1
        assert resp.output[0].role == "assistant"
        assert resp.output[0].content[0].text == "All done!"

    def test_builds_response_without_message(self):
        from responses_api_agents.browser_agent.app import _build_nemo_response

        traj = CUATrajectory(steps=[], task_prompt="test", initial_screenshot="")
        resp = _build_nemo_response(traj, "env-1", None, "test-model")
        assert resp.output[0].content[0].text == "Task completed"

    def test_builds_response_with_usage(self):
        from nemo_gym.openai_utils import (
            NeMoGymResponseInputTokensDetails,
            NeMoGymResponseOutputTokensDetails,
            NeMoGymResponseUsage,
        )
        from responses_api_agents.browser_agent.app import _build_nemo_response

        traj = CUATrajectory(steps=[], task_prompt="test", initial_screenshot="")
        usage = NeMoGymResponseUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
        )
        resp = _build_nemo_response(traj, "env-1", None, "test-model", usage=usage)
        assert resp.usage is not None
        assert resp.usage.input_tokens == 100
        assert resp.usage.output_tokens == 50
        assert resp.usage.total_tokens == 150

    def test_builds_response_without_usage(self):
        from responses_api_agents.browser_agent.app import _build_nemo_response

        traj = CUATrajectory(steps=[], task_prompt="test", initial_screenshot="")
        resp = _build_nemo_response(traj, "env-1", None, "test-model")
        assert resp.usage is None


class TestCUAAdapterUsage:
    def test_default_values(self):
        from responses_api_agents.browser_agent.adapters.base import CUAAdapterUsage

        usage = CUAAdapterUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_custom_values(self):
        from responses_api_agents.browser_agent.adapters.base import CUAAdapterUsage

        usage = CUAAdapterUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_adapter_response_with_usage(self):
        from responses_api_agents.browser_agent.adapters.base import CUAAdapterUsage

        usage = CUAAdapterUsage(input_tokens=10, output_tokens=20, total_tokens=30)
        resp = CUAAdapterResponse(usage=usage)
        assert resp.usage is not None
        assert resp.usage.input_tokens == 10

    def test_adapter_response_default_no_usage(self):
        resp = CUAAdapterResponse()
        assert resp.usage is None


class TestExtractTokenIds:
    def test_extracts_from_output_items(self):
        from responses_api_agents.browser_agent.adapters.base import extract_token_ids_from_response

        response = {
            "output": [
                {"type": "reasoning", "text": "thinking..."},
                {
                    "type": "message",
                    "prompt_token_ids": [1, 2, 3],
                    "generation_token_ids": [4, 5, 6],
                    "generation_log_probs": [-0.1, -0.2, -0.3],
                },
            ]
        }
        result = extract_token_ids_from_response(response)
        assert result["prompt_token_ids"] == [1, 2, 3]
        assert result["generation_token_ids"] == [4, 5, 6]
        assert result["generation_log_probs"] == [-0.1, -0.2, -0.3]

    def test_extracts_from_provider_specific_fields(self):
        from responses_api_agents.browser_agent.adapters.base import extract_token_ids_from_response

        response = {
            "provider_specific_fields": {
                "prompt_token_ids": [10, 20],
                "generation_token_ids": [30, 40],
                "generation_log_probs": [-1.0, -2.0],
            }
        }
        result = extract_token_ids_from_response(response)
        assert result["prompt_token_ids"] == [10, 20]
        assert result["generation_token_ids"] == [30, 40]

    def test_extracts_from_top_level(self):
        from responses_api_agents.browser_agent.adapters.base import extract_token_ids_from_response

        response = {
            "prompt_token_ids": [7, 8],
            "generation_token_ids": [9, 10],
            "generation_log_probs": [-0.5, -0.6],
        }
        result = extract_token_ids_from_response(response)
        assert result["prompt_token_ids"] == [7, 8]
        assert result["generation_token_ids"] == [9, 10]

    def test_returns_empty_when_missing(self):
        from responses_api_agents.browser_agent.adapters.base import extract_token_ids_from_response

        result = extract_token_ids_from_response({"output": [{"type": "message"}]})
        assert result["prompt_token_ids"] == []
        assert result["generation_token_ids"] == []
        assert result["generation_log_probs"] == []

    def test_returns_empty_for_empty_response(self):
        from responses_api_agents.browser_agent.adapters.base import extract_token_ids_from_response

        result = extract_token_ids_from_response({})
        assert result["prompt_token_ids"] == []
        assert result["generation_token_ids"] == []
        assert result["generation_log_probs"] == []

    def test_output_items_take_priority_over_top_level(self):
        from responses_api_agents.browser_agent.adapters.base import extract_token_ids_from_response

        response = {
            "prompt_token_ids": [99],
            "generation_token_ids": [99],
            "generation_log_probs": [-99.0],
            "output": [
                {
                    "type": "message",
                    "prompt_token_ids": [1],
                    "generation_token_ids": [2],
                    "generation_log_probs": [-1.0],
                }
            ],
        }
        result = extract_token_ids_from_response(response)
        assert result["prompt_token_ids"] == [1]
        assert result["generation_token_ids"] == [2]


class TestOpenAITokenIdExtraction:
    def test_parse_actions_extracts_token_ids(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter()
        response = {
            "id": "resp_123",
            "output": [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "action": {"type": "click", "coordinate": [100, 200]},
                    "prompt_token_ids": [1, 2, 3],
                    "generation_token_ids": [4, 5],
                    "generation_log_probs": [-0.1, -0.2],
                }
            ],
        }
        result = adapter._parse_actions(response)
        assert result.prompt_token_ids == [1, 2, 3]
        assert result.generation_token_ids == [4, 5]
        assert result.generation_log_probs == [-0.1, -0.2]

    def test_parse_actions_empty_when_no_token_ids(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter()
        response = {
            "id": "resp_123",
            "output": [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "action": {"type": "click", "coordinate": [100, 200]},
                }
            ],
        }
        result = adapter._parse_actions(response)
        assert result.prompt_token_ids == []
        assert result.generation_token_ids == []
        assert result.generation_log_probs == []


class TestGeminiTokenIdExtraction:
    def test_parse_response_extracts_token_ids(self):
        adapter = _make_gemini_adapter()
        data = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Done"}],
                    }
                }
            ],
            "prompt_token_ids": [10, 20, 30],
            "generation_token_ids": [40, 50],
            "generation_log_probs": [-0.5, -0.6],
        }
        result = adapter._parse_response(data)
        assert result.prompt_token_ids == [10, 20, 30]
        assert result.generation_token_ids == [40, 50]
        assert result.generation_log_probs == [-0.5, -0.6]

    def test_parse_response_empty_when_no_token_ids(self):
        adapter = _make_gemini_adapter()
        data = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Done"}],
                    }
                }
            ],
        }
        result = adapter._parse_response(data)
        assert result.prompt_token_ids == []
        assert result.generation_token_ids == []


class TestBuildNemoResponseTokenIds:
    def test_uses_training_variant_with_token_ids(self):
        from nemo_gym.openai_utils import NeMoGymResponseOutputMessageForTraining
        from resources_servers.browser_gym.schemas import BrowserAction, CUAStep, CUATrajectory
        from responses_api_agents.browser_agent.app import _build_nemo_response

        steps = [
            CUAStep(
                action=BrowserAction(action_type="click", coordinate=[100, 200]),
                screenshot_after="img1",
                current_url="https://example.com",
                prompt_token_ids=[1, 2, 3],
                generation_token_ids=[4, 5],
                generation_log_probs=[-0.1, -0.2],
            ),
            CUAStep(
                action=BrowserAction(action_type="type", text="hello"),
                screenshot_after="img2",
                current_url="https://example.com",
                prompt_token_ids=[10, 20],
                generation_token_ids=[30, 40],
                generation_log_probs=[-0.3, -0.4],
            ),
        ]
        traj = CUATrajectory(steps=steps, task_prompt="test", initial_screenshot="init", final_message="Done")
        resp = _build_nemo_response(traj, "env-1", None, "test-model")

        output_msg = resp.output[0]
        assert isinstance(output_msg, NeMoGymResponseOutputMessageForTraining)
        assert output_msg.prompt_token_ids == [1, 2, 3, 10, 20]
        assert output_msg.generation_token_ids == [4, 5, 30, 40]
        assert output_msg.generation_log_probs == [-0.1, -0.2, -0.3, -0.4]

    def test_uses_base_variant_without_token_ids(self):
        from nemo_gym.openai_utils import NeMoGymResponseOutputMessage, NeMoGymResponseOutputMessageForTraining
        from responses_api_agents.browser_agent.app import _build_nemo_response

        traj = CUATrajectory(steps=[], task_prompt="test", initial_screenshot="")
        resp = _build_nemo_response(traj, "env-1", None, "test-model")

        output_msg = resp.output[0]
        assert isinstance(output_msg, NeMoGymResponseOutputMessage)
        assert not isinstance(output_msg, NeMoGymResponseOutputMessageForTraining)


class TestCUAStepTokenIds:
    def test_step_stores_token_ids(self):
        step = CUAStep(
            action=BrowserAction(action_type="click"),
            screenshot_after="img",
            current_url="https://example.com",
            prompt_token_ids=[1, 2],
            generation_token_ids=[3, 4],
            generation_log_probs=[-0.1, -0.2],
        )
        assert step.prompt_token_ids == [1, 2]
        assert step.generation_token_ids == [3, 4]
        assert step.generation_log_probs == [-0.1, -0.2]

    def test_step_defaults_empty(self):
        step = CUAStep(
            action=BrowserAction(action_type="click"),
            screenshot_after="img",
            current_url="https://example.com",
        )
        assert step.prompt_token_ids == []
        assert step.generation_token_ids == []
        assert step.generation_log_probs == []


class TestCUAAdapterResponseTokenIds:
    def test_default_empty(self):
        resp = CUAAdapterResponse()
        assert resp.prompt_token_ids == []
        assert resp.generation_token_ids == []
        assert resp.generation_log_probs == []

    def test_with_token_ids(self):
        resp = CUAAdapterResponse(
            prompt_token_ids=[1, 2],
            generation_token_ids=[3, 4],
            generation_log_probs=[-0.5, -0.6],
        )
        assert resp.prompt_token_ids == [1, 2]
        assert resp.generation_token_ids == [3, 4]
        assert resp.generation_log_probs == [-0.5, -0.6]


class TestAnthropicUsageExtraction:
    def test_parse_response_extracts_usage(self):
        pytest.importorskip("anthropic", reason="anthropic SDK not installed")
        from responses_api_agents.browser_agent.adapters.anthropic_adapter import AnthropicCUAAdapter

        adapter = AnthropicCUAAdapter.__new__(AnthropicCUAAdapter)
        adapter._pending_tool_use_ids = []

        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Done")]
        mock_response.stop_reason = "end_turn"
        mock_response.id = "msg_123"
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=200)

        result = adapter._parse_response(mock_response)
        assert result.usage is not None
        assert result.usage.input_tokens == 500
        assert result.usage.output_tokens == 200
        assert result.usage.total_tokens == 700

    def test_parse_response_no_usage(self):
        pytest.importorskip("anthropic", reason="anthropic SDK not installed")
        from responses_api_agents.browser_agent.adapters.anthropic_adapter import AnthropicCUAAdapter

        adapter = AnthropicCUAAdapter.__new__(AnthropicCUAAdapter)
        adapter._pending_tool_use_ids = []

        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Done")]
        mock_response.stop_reason = "end_turn"
        mock_response.id = "msg_456"
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = None

        result = adapter._parse_response(mock_response)
        assert result.usage is None


class TestGeminiUsageExtraction:
    def test_parse_response_with_usage(self):
        adapter = _make_gemini_adapter()

        data = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Done"}],
                    }
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 1000,
                "candidates_token_count": 400,
            },
        }

        result = adapter._parse_response(data)
        assert result.usage is not None
        assert result.usage.input_tokens == 1000
        assert result.usage.output_tokens == 400
        assert result.usage.total_tokens == 1400

    def test_parse_response_no_usage(self):
        adapter = _make_gemini_adapter()

        data = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Done"}],
                    }
                }
            ],
        }

        result = adapter._parse_response(data)
        assert result.usage is None


# ── Helpers ──────────────────────────────────────────────────────

# 1x1 red PNG, base64 encoded (valid minimal image)
_TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8BQDwAEgAF/pooBPQAAAABJRU5ErkJggg=="


def _make_trajectory(num_steps: int = 2, initial_screenshot: str = _TINY_PNG_B64) -> CUATrajectory:
    steps = []
    for i in range(num_steps):
        steps.append(
            CUAStep(
                action=BrowserAction(action_type="click", coordinate=[10 * i, 20 * i]),
                screenshot_after=_TINY_PNG_B64,
                current_url=f"https://example.com/page{i}",
                raw_provider_response={"step": i, "image": f"data:image/png;base64,{_TINY_PNG_B64}"},
            )
        )
    return CUATrajectory(
        steps=steps,
        task_prompt="Do the thing",
        initial_screenshot=initial_screenshot,
        final_message="All done",
    )


# ── TrajectoryWriter tests ───────────────────────────────────────


class TestStripBase64Fields:
    def test_replaces_data_uri(self):
        obj = {"img": "data:image/png;base64,abc123"}
        result = _strip_base64_fields(obj)
        assert result["img"] == "<base64_image>"

    def test_replaces_long_base64_string(self):
        long_b64 = "A" * 600
        result = _strip_base64_fields({"data": long_b64})
        assert result["data"] == "<base64_image>"

    def test_preserves_short_strings(self):
        result = _strip_base64_fields({"text": "hello"})
        assert result["text"] == "hello"

    def test_recurses_into_lists(self):
        result = _strip_base64_fields([{"img": "data:image/png;base64,x"}, "short"])
        assert result[0]["img"] == "<base64_image>"
        assert result[1] == "short"

    def test_preserves_non_string_values(self):
        result = _strip_base64_fields({"count": 42, "flag": True, "empty": None})
        assert result == {"count": 42, "flag": True, "empty": None}


class TestLooksLikeBase64:
    def test_valid_base64(self):
        assert _looks_like_base64("A" * 200) is True

    def test_short_string(self):
        assert _looks_like_base64("abc") is False

    def test_non_base64_chars(self):
        assert _looks_like_base64("!" * 200) is False


class TestInitDebugTrajectory:
    def test_creates_directory_and_files(self, tmp_path):
        rollout_dir = init_debug_trajectory(
            output_dir=str(tmp_path),
            env_id="test-env-001",
            initial_screenshot=_TINY_PNG_B64,
            task_prompt="Click the button",
            adapter_type="openai",
            model_name="computer-use-preview",
        )

        assert rollout_dir == tmp_path / "test-env-001"
        assert (rollout_dir / "screenshots" / "00_initial.png").exists()
        assert (rollout_dir / "screenshots" / "00_initial.png").stat().st_size > 0

        jsonl_path = rollout_dir / "conversation.jsonl"
        assert jsonl_path.exists()
        header = json.loads(jsonl_path.read_text().strip())
        assert header["type"] == "header"
        assert header["env_id"] == "test-env-001"
        assert header["task_prompt"] == "Click the button"
        assert header["adapter_type"] == "openai"
        assert header["model"] == "computer-use-preview"


class TestAppendDebugStep:
    def test_appends_step_and_screenshot(self, tmp_path):
        rollout_dir = tmp_path / "env-append"
        screenshots_dir = rollout_dir / "screenshots"
        screenshots_dir.mkdir(parents=True)
        (rollout_dir / "conversation.jsonl").write_text("")

        action = BrowserAction(action_type="click", coordinate=[100, 200])
        append_debug_step(
            rollout_dir=rollout_dir,
            step_idx=1,
            action=action,
            screenshot_after=_TINY_PNG_B64,
            current_url="https://example.com",
            raw_provider_response={"id": "resp_1"},
        )

        assert (screenshots_dir / "01_after.png").exists()

        lines = (rollout_dir / "conversation.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        step = json.loads(lines[0])
        assert step["type"] == "step"
        assert step["step"] == 1
        assert step["action"]["action_type"] == "click"
        assert step["current_url"] == "https://example.com"

    def test_strips_base64_from_raw_response(self, tmp_path):
        rollout_dir = tmp_path / "env-strip"
        (rollout_dir / "screenshots").mkdir(parents=True)
        (rollout_dir / "conversation.jsonl").write_text("")

        action = BrowserAction(action_type="screenshot")
        append_debug_step(
            rollout_dir=rollout_dir,
            step_idx=1,
            action=action,
            screenshot_after=_TINY_PNG_B64,
            current_url="https://example.com",
            raw_provider_response={"image": f"data:image/png;base64,{_TINY_PNG_B64}"},
        )

        step = json.loads((rollout_dir / "conversation.jsonl").read_text().strip())
        assert step["raw_provider_response"]["image"] == "<base64_image>"


class TestFinalizeDebugTrajectory:
    def test_writes_conversation_and_verification(self, tmp_path):
        rollout_dir = tmp_path / "env-finalize"
        rollout_dir.mkdir(parents=True)
        trajectory = _make_trajectory(num_steps=1)

        result = finalize_debug_trajectory(
            rollout_dir=rollout_dir,
            env_id="env-finalize",
            trajectory=trajectory,
            reward=1.0,
            local_storage_dump='{"key": "value"}',
            adapter_type="anthropic",
            model_name="claude-sonnet",
            verifier_metadata={"task_id": "TASK-001"},
            verification_result={"assertions": [{"result": "pass"}]},
        )

        assert result == str(rollout_dir)

        conv = json.loads((rollout_dir / "conversation.json").read_text())
        assert conv["env_id"] == "env-finalize"
        assert conv["task_prompt"] == "Do the thing"
        assert conv["final_message"] == "All done"
        assert conv["num_steps"] == 1
        assert len(conv["steps"]) == 1
        assert conv["steps"][0]["action"]["action_type"] == "click"

        verif = json.loads((rollout_dir / "verification.json").read_text())
        assert verif["reward"] == 1.0
        assert verif["local_storage_dump"] == {"key": "value"}
        assert verif["verifier_metadata"]["task_id"] == "TASK-001"
        assert verif["get_actual_state_response"]["assertions"][0]["result"] == "pass"

    def test_handles_invalid_local_storage_json(self, tmp_path):
        rollout_dir = tmp_path / "env-bad-ls"
        rollout_dir.mkdir(parents=True)
        trajectory = _make_trajectory(num_steps=0)

        finalize_debug_trajectory(
            rollout_dir=rollout_dir,
            env_id="env-bad-ls",
            trajectory=trajectory,
            local_storage_dump="not-valid-json{",
        )

        verif = json.loads((rollout_dir / "verification.json").read_text())
        assert verif["local_storage_dump"] == "not-valid-json{"


class TestSaveDebugTrajectory:
    def test_normal_write(self, tmp_path):
        trajectory = _make_trajectory(num_steps=2)

        result = save_debug_trajectory(
            output_dir=str(tmp_path),
            env_id="env-full",
            trajectory=trajectory,
            reward=1.0,
            local_storage_dump='{"done": true}',
            adapter_type="gemini",
            model_name="gemini-2.5-cu",
            verifier_metadata={"task_id": "T-001"},
        )

        rollout_dir = Path(result)
        assert rollout_dir.exists()
        assert (rollout_dir / "screenshots" / "00_initial.png").exists()
        assert (rollout_dir / "screenshots" / "01_after.png").exists()
        assert (rollout_dir / "screenshots" / "02_after.png").exists()
        assert (rollout_dir / "conversation.json").exists()
        assert (rollout_dir / "verification.json").exists()

        conv = json.loads((rollout_dir / "conversation.json").read_text())
        assert conv["num_steps"] == 2
        assert conv["adapter_type"] == "gemini"

        verif = json.loads((rollout_dir / "verification.json").read_text())
        assert verif["reward"] == 1.0

    def test_empty_trajectory(self, tmp_path):
        trajectory = _make_trajectory(num_steps=0, initial_screenshot="")

        result = save_debug_trajectory(
            output_dir=str(tmp_path),
            env_id="env-empty",
            trajectory=trajectory,
        )

        rollout_dir = Path(result)
        assert rollout_dir.exists()
        assert not (rollout_dir / "screenshots" / "00_initial.png").exists()

        conv = json.loads((rollout_dir / "conversation.json").read_text())
        assert conv["num_steps"] == 0
        assert conv["steps"] == []

        verif = json.loads((rollout_dir / "verification.json").read_text())
        assert verif["reward"] is None

    def test_permission_error(self, tmp_path):
        trajectory = _make_trajectory(num_steps=1)

        with patch(
            "responses_api_agents.browser_agent.trajectory_writer.Path.mkdir", side_effect=PermissionError("denied")
        ):
            with pytest.raises(PermissionError, match="denied"):
                save_debug_trajectory(
                    output_dir=str(tmp_path),
                    env_id="env-permerr",
                    trajectory=trajectory,
                )
