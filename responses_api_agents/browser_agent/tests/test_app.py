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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from resources_servers.browser_gym.schemas import BrowserAction, CUATrajectory
from responses_api_agents.browser_agent.adapters.base import BaseCUAAdapter, CUAAdapterResponse


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

        adapter = OpenAICUAAdapter(api_key="test-key")
        action = adapter._map_openai_action({"type": "click", "coordinate": [100, 200]})
        assert action.action_type == "click"
        assert action.coordinate == [100, 200]

    def test_action_parsing_type(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter(api_key="test-key")
        action = adapter._map_openai_action({"type": "type", "text": "hello world"})
        assert action.action_type == "type"
        assert action.text == "hello world"

    def test_action_parsing_scroll(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter(api_key="test-key")
        action = adapter._map_openai_action(
            {"type": "scroll", "coordinate": [640, 360], "scroll_x": 0, "scroll_y": 300}
        )
        assert action.action_type == "scroll"
        assert action.scroll_y == 300

    def test_action_parsing_keypress(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter(api_key="test-key")
        action = adapter._map_openai_action({"type": "keypress", "keys": ["Control", "a"]})
        assert action.action_type == "keypress"
        assert action.keys == ["Control", "a"]

    def test_action_parsing_drag(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter(api_key="test-key")
        action = adapter._map_openai_action(
            {"type": "drag", "start_x": 10, "start_y": 20, "destination_x": 100, "destination_y": 200}
        )
        assert action.action_type == "drag"
        assert action.start_coordinate == [10, 20]
        assert action.end_coordinate == [100, 200]

    def test_action_parsing_goto(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter(api_key="test-key")
        action = adapter._map_openai_action({"type": "goto", "url": "https://example.com"})
        assert action.action_type == "goto"
        assert action.url == "https://example.com"

    def test_action_parsing_tab_actions(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter(api_key="test-key")
        assert adapter._map_openai_action({"type": "new_tab", "url": "https://example.com"}).action_type == "new_tab"
        assert adapter._map_openai_action({"type": "close_tab"}).action_type == "close_tab"
        assert adapter._map_openai_action({"type": "switch_tab", "tab_index": 1}).action_type == "switch_tab"

    def test_action_parsing_unknown(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter(api_key="test-key")
        action = adapter._map_openai_action({"type": "unknown_action"})
        assert action is None

    def test_headers_with_org(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter(api_key="sk-test", organization="org-123")
        headers = adapter._get_headers()
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["Openai-Organization"] == "org-123"

    def test_headers_without_org(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter(api_key="sk-test")
        headers = adapter._get_headers()
        assert "Openai-Organization" not in headers

    def test_tools_config(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter(api_key="test", viewport_width=1024, viewport_height=768)
        tools = adapter._get_tools()
        assert tools[0]["type"] == "computer_use_preview"
        assert tools[0]["display_width"] == 1024
        assert tools[0]["display_height"] == 768

    def test_parse_actions_computer_call(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter(api_key="test")
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

        adapter = OpenAICUAAdapter(api_key="test")
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

    def test_reset(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter(api_key="test")
        adapter._last_response_id = "resp_123"
        adapter._pending_call_ids = ["call_1"]
        adapter.reset()
        assert adapter._last_response_id is None
        assert adapter._pending_call_ids == []

    def test_response_id_tracking(self):
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = OpenAICUAAdapter(api_key="test")
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
                {"role": "assistant", "content": [{"type": "tool_use", "id": f"tu_{i}", "name": "computer", "input": {}}]}
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
    adapter._client = None
    adapter._timeout = 300.0
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

        adapter = AdapterFactory.create("openai", api_key="test")
        assert isinstance(adapter, OpenAICUAAdapter)

    def test_create_unknown(self):
        from responses_api_agents.browser_agent.adapters import AdapterFactory

        with pytest.raises(ValueError, match="Unknown adapter type"):
            AdapterFactory.create("nonexistent", api_key="test")

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
