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
            {"type": "drag", "start_coordinate": [10, 20], "coordinate": [100, 200]}
        )
        assert action.action_type == "drag"
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


class TestGeminiAdapterParsing:
    def test_action_mapping_click_at(self):
        from responses_api_agents.browser_agent.adapters.gemini_adapter import GeminiCUAAdapter

        adapter = GeminiCUAAdapter.__new__(GeminiCUAAdapter)
        action = adapter._map_gemini_action("click_at", {"x": 100, "y": 200})
        assert action.action_type == "click"
        assert action.coordinate == [100, 200]

    def test_action_mapping_type_text_at(self):
        from responses_api_agents.browser_agent.adapters.gemini_adapter import GeminiCUAAdapter

        adapter = GeminiCUAAdapter.__new__(GeminiCUAAdapter)
        action = adapter._map_gemini_action("type_text_at", {"x": 50, "y": 50, "text": "hello", "press_enter": True})
        assert action.action_type == "type"
        assert action.text == "hello"
        assert action.press_enter is True

    def test_action_mapping_scroll_at(self):
        from responses_api_agents.browser_agent.adapters.gemini_adapter import GeminiCUAAdapter

        adapter = GeminiCUAAdapter.__new__(GeminiCUAAdapter)
        action = adapter._map_gemini_action("scroll_at", {"x": 640, "y": 360, "direction": "up", "amount": 5})
        assert action.action_type == "scroll"
        assert action.scroll_direction == "up"
        assert action.scroll_amount == 5

    def test_action_mapping_navigate(self):
        from responses_api_agents.browser_agent.adapters.gemini_adapter import GeminiCUAAdapter

        adapter = GeminiCUAAdapter.__new__(GeminiCUAAdapter)
        action = adapter._map_gemini_action("navigate", {"url": "https://example.com"})
        assert action.action_type == "goto"
        assert action.url == "https://example.com"

    def test_action_mapping_drag_and_drop(self):
        from responses_api_agents.browser_agent.adapters.gemini_adapter import GeminiCUAAdapter

        adapter = GeminiCUAAdapter.__new__(GeminiCUAAdapter)
        action = adapter._map_gemini_action(
            "drag_and_drop", {"start_x": 10, "start_y": 20, "end_x": 100, "end_y": 200}
        )
        assert action.action_type == "drag"
        assert action.start_coordinate == [10, 20]
        assert action.end_coordinate == [100, 200]

    def test_action_mapping_keypress(self):
        from responses_api_agents.browser_agent.adapters.gemini_adapter import GeminiCUAAdapter

        adapter = GeminiCUAAdapter.__new__(GeminiCUAAdapter)
        action = adapter._map_gemini_action("keypress", {"key": "Enter"})
        assert action.action_type == "keypress"
        assert action.key == "Enter"

    def test_action_mapping_key_combination(self):
        from responses_api_agents.browser_agent.adapters.gemini_adapter import GeminiCUAAdapter

        adapter = GeminiCUAAdapter.__new__(GeminiCUAAdapter)
        action = adapter._map_gemini_action("key_combination", {"keys": ["Control", "v"]})
        assert action.action_type == "keypress"
        assert action.keys == ["Control", "v"]

    def test_action_mapping_scroll_document(self):
        from responses_api_agents.browser_agent.adapters.gemini_adapter import GeminiCUAAdapter

        adapter = GeminiCUAAdapter.__new__(GeminiCUAAdapter)
        action = adapter._map_gemini_action("scroll_document", {"direction": "down"})
        assert action.action_type == "scroll"
        assert action.scroll_direction == "down"

    def test_action_mapping_go_back(self):
        from responses_api_agents.browser_agent.adapters.gemini_adapter import GeminiCUAAdapter

        adapter = GeminiCUAAdapter.__new__(GeminiCUAAdapter)
        action = adapter._map_gemini_action("go_back", {})
        assert action.action_type == "go_back"

    def test_action_mapping_unknown(self):
        from responses_api_agents.browser_agent.adapters.gemini_adapter import GeminiCUAAdapter

        adapter = GeminiCUAAdapter.__new__(GeminiCUAAdapter)
        action = adapter._map_gemini_action("unknown_action", {})
        assert action is None


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
