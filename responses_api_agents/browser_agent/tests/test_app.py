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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from resources_servers.browser_gym.schemas import BrowserAction, CUAStep, CUATrajectory
from responses_api_agents.browser_agent.adapters.base import CUAAdapterResponse, CUAAdapterUsage
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


class TestGenericAdapterAnthropicActions:
    """Test GenericCUAAdapter with Anthropic-style action dicts (keyed by 'action')."""

    def _make_adapter(self):
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        return GenericCUAAdapter(denormalize_coords=False)

    def test_action_mapping_left_click(self):
        adapter = self._make_adapter()
        action = adapter._map_action({"action": "left_click", "coordinate": [100, 200]})
        assert action.action_type == "click"
        assert action.button == "left"

    def test_action_mapping_right_click(self):
        adapter = self._make_adapter()
        action = adapter._map_action({"action": "right_click", "coordinate": [100, 200]})
        assert action.action_type == "right_click"

    def test_action_mapping_key(self):
        adapter = self._make_adapter()
        action = adapter._map_action({"action": "key", "text": "Enter"})
        assert action.action_type == "keypress"
        assert action.key == "Enter"

    def test_action_mapping_mouse_move(self):
        adapter = self._make_adapter()
        action = adapter._map_action({"action": "mouse_move", "coordinate": [300, 400]})
        assert action.action_type == "hover"
        assert action.coordinate == [300, 400]

    def test_action_mapping_scroll(self):
        adapter = self._make_adapter()
        action = adapter._map_action({"action": "scroll", "coordinate": [640, 360], "direction": "down", "amount": 3})
        assert action.action_type == "scroll"
        assert action.scroll_direction == "down"
        assert action.scroll_amount == 3

    def test_action_mapping_drag(self):
        adapter = self._make_adapter()
        action = adapter._map_action(
            {"action": "left_click_drag", "start_coordinate": [10, 20], "coordinate": [100, 200]}
        )
        assert action.action_type == "drag"

    def test_action_mapping_unknown(self):
        adapter = self._make_adapter()
        action = adapter._map_action({"action": "unknown_action"})
        assert action is None


def _make_generic_gemini_adapter(viewport_width=1280, viewport_height=720):
    """Create a GenericCUAAdapter configured for Gemini (denormalize_coords=True)."""
    from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

    return GenericCUAAdapter(
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        denormalize_coords=True,
    )


class TestGenericAdapterGeminiActions:
    """Test GenericCUAAdapter with Gemini-style action dicts (keyed by 'type')."""

    def test_action_mapping_click_at(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "click_at", "x": 500, "y": 500})
        assert action.action_type == "click"
        assert action.coordinate == [640, 360]

    def test_action_mapping_hover_at(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "hover_at", "x": 0, "y": 0})
        assert action.action_type == "hover"
        assert action.coordinate == [0, 0]

    def test_action_mapping_hover_at_max(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "hover_at", "x": 999, "y": 999})
        assert action.action_type == "hover"
        assert action.coordinate == [int(999 / 1000 * 1280), int(999 / 1000 * 720)]

    def test_action_mapping_type_text_at(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action(
            {"type": "type_text_at", "x": 500, "y": 500, "text": "hello", "press_enter": True}
        )
        assert action.action_type == "type"
        assert action.text == "hello"
        assert action.press_enter is True
        assert action.coordinate == [640, 360]

    def test_action_mapping_type_text_at_clear_default(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "type_text_at", "x": 0, "y": 0, "text": "a"})
        assert action.clear_before_typing is True

    def test_action_mapping_scroll_at(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "scroll_at", "x": 500, "y": 500, "direction": "up", "magnitude": 800})
        assert action.action_type == "scroll"
        assert action.scroll_direction == "up"
        assert action.coordinate == [640, 360]
        assert action.scroll_y < 0

    def test_action_mapping_scroll_document_down(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "scroll_document", "direction": "down"})
        assert action.action_type == "scroll"
        assert action.scroll_direction == "down"
        assert action.scroll_y == int(720 * 0.8)
        assert action.scroll_x == 0

    def test_action_mapping_scroll_document_left(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "scroll_document", "direction": "left"})
        assert action.scroll_x == -(1280 // 2)
        assert action.scroll_y == 0

    def test_action_mapping_navigate(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "navigate", "url": "https://example.com"})
        assert action.action_type == "goto"
        assert action.url == "https://example.com"

    def test_action_mapping_navigate_bare_url(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "navigate", "url": "example.com"})
        assert action.url == "https://example.com"

    def test_action_mapping_search(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "search", "query": "test query"})
        assert action.action_type == "goto"
        assert action.url == "https://www.google.com"

    def test_action_mapping_drag_and_drop(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action(
            {"type": "drag_and_drop", "x": 500, "y": 500, "destination_x": 750, "destination_y": 250}
        )
        assert action.action_type == "drag"
        assert action.start_coordinate == [640, 360]
        assert action.end_coordinate == [int(750 / 1000 * 1280), int(250 / 1000 * 720)]

    def test_action_mapping_keypress(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "keypress", "key": "Enter"})
        assert action.action_type == "keypress"

    def test_action_mapping_keypress_combo_string(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "keypress", "keys": "Control+a"})
        assert action.keys == ["Control", "a"]

    def test_action_mapping_key_combination(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "key_combination", "keys": ["Control", "v"]})
        assert action.action_type == "keypress"
        assert action.keys == ["Control", "v"]

    def test_action_mapping_go_back(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "go_back"})
        assert action.action_type == "go_back"

    def test_action_mapping_go_forward(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "go_forward"})
        assert action.action_type == "go_forward"

    def test_action_mapping_wait(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "wait_5_seconds"})
        assert action.action_type == "wait"
        assert action.duration == 5000

    def test_action_mapping_new_tab(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "new_tab", "url": "https://test.com"})
        assert action.action_type == "new_tab"
        assert action.url == "https://test.com"

    def test_action_mapping_switch_tab(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "switch_tab", "index": 2})
        assert action.action_type == "switch_tab"
        assert action.tab_index == 2

    def test_action_mapping_close_tab(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "close_tab"})
        assert action.action_type == "close_tab"

    def test_action_mapping_list_tabs(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "list_tabs"})
        assert action.action_type == "screenshot"

    def test_action_mapping_open_web_browser(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "open_web_browser"})
        assert action.action_type == "screenshot"

    def test_action_mapping_unknown(self):
        adapter = _make_generic_gemini_adapter()
        action = adapter._map_action({"type": "unknown_action"})
        assert action is None


class TestGenericAdapterDenormalization:
    def test_denorm_x(self):
        adapter = _make_generic_gemini_adapter(viewport_width=1280)
        assert adapter._denorm_x(0) == 0
        assert adapter._denorm_x(500) == 640
        assert adapter._denorm_x(1000) == 1280

    def test_denorm_y(self):
        adapter = _make_generic_gemini_adapter(viewport_height=720)
        assert adapter._denorm_y(0) == 0
        assert adapter._denorm_y(500) == 360
        assert adapter._denorm_y(1000) == 720

    def test_denorm_custom_viewport(self):
        adapter = _make_generic_gemini_adapter(viewport_width=1920, viewport_height=1080)
        assert adapter._denorm_x(500) == 960
        assert adapter._denorm_y(500) == 540

    def test_click_at_denormalizes(self):
        adapter = _make_generic_gemini_adapter(viewport_width=1000, viewport_height=1000)
        action = adapter._map_action({"type": "click_at", "x": 500, "y": 250})
        assert action.coordinate == [500, 250]

    def test_drag_denormalizes_all_coords(self):
        adapter = _make_generic_gemini_adapter(viewport_width=1000, viewport_height=1000)
        action = adapter._map_action(
            {"type": "drag_and_drop", "x": 100, "y": 200, "destination_x": 800, "destination_y": 900}
        )
        assert action.start_coordinate == [100, 200]
        assert action.end_coordinate == [800, 900]


class TestGenericAdapterReset:
    def test_reset_clears_state(self):
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        adapter = GenericCUAAdapter()
        adapter._input_items = [{"type": "message"}]
        adapter._pending_call_ids = ["call_1"]
        adapter._pending_safety_checks = [{"id": "sc_1"}]
        adapter.reset()
        assert adapter._input_items == []
        assert adapter._pending_call_ids == []
        assert adapter._pending_safety_checks == []


class TestAdapterFactory:
    def test_create_openai(self):
        from responses_api_agents.browser_agent.adapters import AdapterFactory
        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        adapter = AdapterFactory.create("openai")
        assert isinstance(adapter, OpenAICUAAdapter)

    def test_create_anthropic_sonnet(self):
        from responses_api_agents.browser_agent.adapters import AdapterFactory
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        adapter = AdapterFactory.create("anthropic_sonnet", api_caller=None)
        assert isinstance(adapter, GenericCUAAdapter)

    def test_create_anthropic_opus(self):
        from responses_api_agents.browser_agent.adapters import AdapterFactory
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        adapter = AdapterFactory.create("anthropic_opus", api_caller=None)
        assert isinstance(adapter, GenericCUAAdapter)

    def test_create_gemini(self):
        from responses_api_agents.browser_agent.adapters import AdapterFactory
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        adapter = AdapterFactory.create("gemini", api_caller=None)
        assert isinstance(adapter, GenericCUAAdapter)

    def test_create_unknown(self):
        from responses_api_agents.browser_agent.adapters import AdapterFactory

        with pytest.raises(ValueError, match="Unknown adapter type"):
            AdapterFactory.create("nonexistent")

    def test_available_adapters(self):
        from responses_api_agents.browser_agent.adapters import AdapterFactory

        adapters = AdapterFactory.available_adapters()
        assert "openai" in adapters
        assert "anthropic_sonnet" in adapters
        assert "gemini" in adapters


class TestBuildNemoResponse:
    def test_builds_valid_response(self):
        from responses_api_agents.browser_agent.app import _build_nemo_response

        traj = CUATrajectory(
            steps=[],
            task_prompt="test task",
            initial_screenshot="base64data",
            final_message="All done!",
        )
        resp = _build_nemo_response(traj, "env-123", '{"key": "val"}', '{"init": "ls"}', "computer-use-preview")

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
        resp = _build_nemo_response(traj, "env-1", None, None, "test-model")
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
        resp = _build_nemo_response(traj, "env-1", None, None, "test-model", usage=usage)
        assert resp.usage is not None
        assert resp.usage.input_tokens == 100
        assert resp.usage.output_tokens == 50
        assert resp.usage.total_tokens == 150

    def test_builds_response_without_usage(self):
        from responses_api_agents.browser_agent.app import _build_nemo_response

        traj = CUATrajectory(steps=[], task_prompt="test", initial_screenshot="")
        resp = _build_nemo_response(traj, "env-1", None, None, "test-model")
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


class TestGenericAdapterTokenIdExtraction:
    def test_parse_response_extracts_token_ids(self):
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        adapter = GenericCUAAdapter()
        data = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done"}],
                    "prompt_token_ids": [10, 20, 30],
                    "generation_token_ids": [40, 50],
                    "generation_log_probs": [-0.5, -0.6],
                }
            ],
        }
        result = adapter._parse_response(data)
        assert result.prompt_token_ids == [10, 20, 30]
        assert result.generation_token_ids == [40, 50]
        assert result.generation_log_probs == [-0.5, -0.6]

    def test_parse_response_empty_when_no_token_ids(self):
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        adapter = GenericCUAAdapter()
        data = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done"}],
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
        resp = _build_nemo_response(traj, "env-1", None, None, "test-model")

        output_msg = resp.output[0]
        assert isinstance(output_msg, NeMoGymResponseOutputMessageForTraining)
        assert output_msg.prompt_token_ids == [1, 2, 3, 10, 20]
        assert output_msg.generation_token_ids == [4, 5, 30, 40]
        assert output_msg.generation_log_probs == [-0.1, -0.2, -0.3, -0.4]

    def test_uses_base_variant_without_token_ids(self):
        from nemo_gym.openai_utils import NeMoGymResponseOutputMessage, NeMoGymResponseOutputMessageForTraining
        from responses_api_agents.browser_agent.app import _build_nemo_response

        traj = CUATrajectory(steps=[], task_prompt="test", initial_screenshot="")
        resp = _build_nemo_response(traj, "env-1", None, None, "test-model")

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


class TestGenericAdapterUsageExtraction:
    def test_parse_response_extracts_usage(self):
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        adapter = GenericCUAAdapter()
        data = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done"}],
                }
            ],
            "usage": {
                "input_tokens": 500,
                "output_tokens": 200,
                "total_tokens": 700,
            },
        }
        result = adapter._parse_response(data)
        assert result.usage is not None
        assert result.usage.input_tokens == 500
        assert result.usage.output_tokens == 200
        assert result.usage.total_tokens == 700

    def test_parse_response_no_usage(self):
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        adapter = GenericCUAAdapter()
        data = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done"}],
                }
            ],
        }
        result = adapter._parse_response(data)
        assert result.usage is None


class TestGenericAdapterResponseParsing:
    """Test GenericCUAAdapter._parse_response with NeMoGymResponse format."""

    def test_parse_computer_call(self):
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        adapter = GenericCUAAdapter()
        response = {
            "id": "resp_123",
            "output": [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "action": {"action": "left_click", "coordinate": [100, 200]},
                }
            ],
        }
        result = adapter._parse_response(response)
        assert len(result.actions) == 1
        assert result.actions[0].action_type == "click"
        assert result.done is False

    def test_parse_final_message_marks_done(self):
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        adapter = GenericCUAAdapter()
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
        result = adapter._parse_response(response)
        assert len(result.actions) == 0
        assert result.done is True
        assert result.message == "Done!"

    def test_parse_message_and_computer_call_not_done(self):
        """When response has both a message and a computer_call, done should be False."""
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        adapter = GenericCUAAdapter()
        response = {
            "id": "resp_mixed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "I will click the button."}],
                },
                {
                    "type": "computer_call",
                    "id": "cu_1",
                    "call_id": "call_1",
                    "action": {"type": "click", "x": 100, "y": 200},
                    "pending_safety_checks": [],
                    "status": "completed",
                },
            ],
        }
        result = adapter._parse_response(response)
        assert len(result.actions) == 1
        assert result.done is False
        assert result.message == "I will click the button."

    def test_parse_tracks_call_ids(self):
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        adapter = GenericCUAAdapter()
        response = {
            "id": "resp_123",
            "output": [
                {
                    "type": "computer_call",
                    "call_id": "call_abc",
                    "action": {"type": "click", "x": 100, "y": 200},
                }
            ],
        }
        adapter._parse_response(response)
        assert adapter._pending_call_ids == ["call_abc"]

    def test_parse_appends_computer_call_to_input_items(self):
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        adapter = GenericCUAAdapter()
        adapter._input_items = [{"type": "message", "role": "user", "content": "test"}]
        response = {
            "output": [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "action": {"type": "click", "x": 50, "y": 50},
                }
            ],
        }
        adapter._parse_response(response)
        assert len(adapter._input_items) == 2
        assert adapter._input_items[1]["type"] == "computer_call"


class TestGenericAdapterContextManagement:
    """Test trimming and GC in GenericCUAAdapter."""

    def test_trim_preserves_first_item(self):
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        adapter = GenericCUAAdapter(turns_to_keep=2)
        adapter._input_items = [
            {"type": "message", "role": "user", "content": "initial task"},
        ]
        for i in range(5):
            adapter._input_items.append({"type": "computer_call", "call_id": f"c{i}", "action": {}})
            adapter._input_items.append({"type": "computer_call_output", "call_id": f"c{i}", "output": {}})

        adapter._trim_input_items()
        assert adapter._input_items[0]["content"] == "initial task"
        assert len(adapter._input_items) <= 1 + 2 * 2 + 1

    def test_gc_replaces_old_screenshots(self):
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        adapter = GenericCUAAdapter(turns_to_keep=1)
        adapter._input_items = [
            {"type": "message", "role": "user", "content": "task"},
            {"type": "computer_call", "call_id": "c0", "action": {}},
            {"type": "computer_call_output", "call_id": "c0", "output": {"image_url": "data:image/png;base64,OLD"}},
            {"type": "computer_call", "call_id": "c1", "action": {}},
            {"type": "computer_call_output", "call_id": "c1", "output": {"image_url": "data:image/png;base64,NEW"}},
        ]
        adapter._gc_old_screenshots()
        assert adapter._input_items[2]["output"]["image_url"] == "[screenshot-trimmed]"
        assert adapter._input_items[4]["output"]["image_url"] == "data:image/png;base64,NEW"


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


def _make_agent_config(**overrides):
    from responses_api_agents.browser_agent.app import BrowserAgentConfig

    defaults = {
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "",
        "name": "",
        "resources_server": {"type": "resources_servers", "name": "browser_gym"},
        "model_server": {"type": "responses_api_models", "name": "openai_model"},
    }
    defaults.update(overrides)
    return BrowserAgentConfig(**defaults)


class TestBrowserAgentConfig:
    def test_default_max_steps(self):
        config = _make_agent_config()
        assert config.max_steps == 250

    def test_default_run_timeout(self):
        config = _make_agent_config()
        assert config.run_timeout_seconds == 7200.0

    def test_custom_run_timeout(self):
        config = _make_agent_config(run_timeout_seconds=3600.0)
        assert config.run_timeout_seconds == 3600.0

    def test_relative_debug_output_dir_resolved_to_absolute(self):
        config = _make_agent_config(cua_debug_output_dir="debug_output")
        assert Path(config.cua_debug_output_dir).is_absolute()


class TestBrowserAgentServer:
    def test_server_instantiation(self):
        from nemo_gym.server_utils import ServerClient
        from responses_api_agents.browser_agent.app import BrowserAgent

        config = _make_agent_config()
        server = BrowserAgent(config=config, server_client=MagicMock(spec=ServerClient))
        assert server.config.cua_adapter_type == "openai"
        assert server.config.max_steps == 250

    def test_server_instantiation_anthropic(self):
        from nemo_gym.server_utils import ServerClient
        from responses_api_agents.browser_agent.app import BrowserAgent

        config = _make_agent_config(cua_adapter_type="anthropic_sonnet", cua_model="claude-sonnet-4-20250514")
        server = BrowserAgent(config=config, server_client=MagicMock(spec=ServerClient))
        assert server.config.cua_adapter_type == "anthropic_sonnet"

    def test_server_instantiation_gemini(self):
        from nemo_gym.server_utils import ServerClient
        from responses_api_agents.browser_agent.app import BrowserAgent

        config = _make_agent_config(cua_adapter_type="gemini", cua_model="gemini-2.5-cu")
        server = BrowserAgent(config=config, server_client=MagicMock(spec=ServerClient))
        assert server.config.cua_adapter_type == "gemini"


class TestBrowserAgentRunFlow:
    """Tests for the CUA loop orchestration in _responses_via_adapter."""

    @pytest.fixture
    def agent(self):
        from nemo_gym.server_utils import ServerClient
        from responses_api_agents.browser_agent.app import BrowserAgent

        config = _make_agent_config(cua_debug_trajectories=False)
        return BrowserAgent(config=config, server_client=MagicMock(spec=ServerClient))

    def _make_mock_adapter(self, init_response=None, step_responses=None):
        adapter = AsyncMock()
        if init_response is None:
            init_response = CUAAdapterResponse(
                actions=[BrowserAction(action_type="click", coordinate=[100, 200])],
                done=False,
            )
        adapter.initialize.return_value = init_response

        if step_responses is None:
            step_responses = [CUAAdapterResponse(done=True, message="Task completed")]
        adapter.step.side_effect = step_responses

        adapter.reset = MagicMock()
        return adapter

    def _make_step_response(self, screenshot="c2NyZWVuc2hvdA==", url="https://example.com", error=None):
        return {
            "screenshot": screenshot,
            "current_url": url,
            "error": error,
        }

    @pytest.mark.asyncio
    async def test_single_action_then_done(self, agent):
        """Adapter returns one action on initialize, then done=True on step."""
        mock_adapter = self._make_mock_adapter()

        step_http_resp = AsyncMock()
        step_http_resp.ok = True
        step_http_resp.cookies = {}
        step_http_resp.json = AsyncMock(return_value=self._make_step_response())

        with (
            patch.object(agent, "_create_adapter", return_value=mock_adapter),
            patch("responses_api_agents.browser_agent.app.raise_for_status", new_callable=AsyncMock),
            patch("responses_api_agents.browser_agent.app.get_response_json", return_value=self._make_step_response()),
        ):
            agent.server_client.post = AsyncMock(return_value=step_http_resp)

            cookie_jar = {"cookies": {}}
            trajectory, ls_dump, initial_ls, usage, debug_dir = await agent._responses_via_adapter(
                task_prompt="Click the button",
                env_id="test-env-1",
                screenshot_b64="aW5pdGlhbA==",
                cookie_jar=cookie_jar,
            )

        assert len(trajectory.steps) == 1
        assert trajectory.steps[0].action.action_type == "click"
        mock_adapter.initialize.assert_awaited_once()
        mock_adapter.step.assert_awaited_once()
        mock_adapter.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_adapter_initialize_failure_returns_empty_trajectory(self, agent):
        """When adapter.initialize() raises, return an empty trajectory gracefully."""
        mock_adapter = AsyncMock()
        mock_adapter.initialize.side_effect = RuntimeError("Model API unreachable")
        mock_adapter.reset = MagicMock()

        with patch.object(agent, "_create_adapter", return_value=mock_adapter):
            cookie_jar = {"cookies": {}}
            trajectory, ls_dump, initial_ls, usage, debug_dir = await agent._responses_via_adapter(
                task_prompt="Click the button",
                env_id="test-env-fail",
                screenshot_b64="aW5pdGlhbA==",
                cookie_jar=cookie_jar,
            )

        assert len(trajectory.steps) == 0
        assert ls_dump == ""
        assert initial_ls == ""
        assert usage is None
        mock_adapter.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_adapter_step_failure_ends_loop(self, agent):
        """When adapter.step() raises, the loop ends and trajectory is returned."""
        init_resp = CUAAdapterResponse(
            actions=[BrowserAction(action_type="click", coordinate=[50, 50])],
            done=False,
        )
        mock_adapter = self._make_mock_adapter(init_response=init_resp)
        mock_adapter.step.side_effect = RuntimeError("API timeout")

        step_http_resp = AsyncMock()
        step_http_resp.ok = True
        step_http_resp.cookies = {}

        with (
            patch.object(agent, "_create_adapter", return_value=mock_adapter),
            patch("responses_api_agents.browser_agent.app.raise_for_status", new_callable=AsyncMock),
            patch("responses_api_agents.browser_agent.app.get_response_json", return_value=self._make_step_response()),
        ):
            agent.server_client.post = AsyncMock(return_value=step_http_resp)

            cookie_jar = {"cookies": {}}
            trajectory, ls_dump, initial_ls, usage, debug_dir = await agent._responses_via_adapter(
                task_prompt="Do something",
                env_id="test-env-step-fail",
                screenshot_b64="aW5pdGlhbA==",
                cookie_jar=cookie_jar,
            )

        assert len(trajectory.steps) == 1
        mock_adapter.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_browser_stuck_breaks_loop(self, agent):
        """When browser returns error:browser_stuck, the loop ends."""
        init_resp = CUAAdapterResponse(
            actions=[BrowserAction(action_type="click", coordinate=[10, 20])],
            done=False,
        )
        mock_adapter = self._make_mock_adapter(init_response=init_resp)

        stuck_response = self._make_step_response(screenshot="", url="error:browser_stuck")

        step_http_resp = AsyncMock()
        step_http_resp.ok = True
        step_http_resp.cookies = {}

        with (
            patch.object(agent, "_create_adapter", return_value=mock_adapter),
            patch("responses_api_agents.browser_agent.app.raise_for_status", new_callable=AsyncMock),
            patch("responses_api_agents.browser_agent.app.get_response_json", return_value=stuck_response),
        ):
            agent.server_client.post = AsyncMock(return_value=step_http_resp)

            cookie_jar = {"cookies": {}}
            trajectory, ls_dump, initial_ls, usage, debug_dir = await agent._responses_via_adapter(
                task_prompt="Navigate",
                env_id="test-env-stuck",
                screenshot_b64="aW5pdGlhbA==",
                cookie_jar=cookie_jar,
            )

        assert len(trajectory.steps) == 0
        mock_adapter.step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_action_error_propagated_to_adapter(self, agent):
        """When a step returns an error field, it's passed to adapter.step() as action_error."""
        init_resp = CUAAdapterResponse(
            actions=[BrowserAction(action_type="keypress", key="InvalidKey")],
            done=False,
        )
        mock_adapter = self._make_mock_adapter(
            init_response=init_resp,
            step_responses=[CUAAdapterResponse(done=True, message="Adjusted action")],
        )

        error_step_response = self._make_step_response(error="keypress failed: Unknown key: InvalidKey")

        step_http_resp = AsyncMock()
        step_http_resp.ok = True
        step_http_resp.cookies = {}

        with (
            patch.object(agent, "_create_adapter", return_value=mock_adapter),
            patch("responses_api_agents.browser_agent.app.raise_for_status", new_callable=AsyncMock),
            patch("responses_api_agents.browser_agent.app.get_response_json", return_value=error_step_response),
        ):
            agent.server_client.post = AsyncMock(return_value=step_http_resp)

            cookie_jar = {"cookies": {}}
            await agent._responses_via_adapter(
                task_prompt="Press key",
                env_id="test-env-err",
                screenshot_b64="aW5pdGlhbA==",
                cookie_jar=cookie_jar,
            )

        call_kwargs = mock_adapter.step.call_args
        assert call_kwargs[1]["action_error"] == "keypress failed: Unknown key: InvalidKey"

    @pytest.mark.asyncio
    async def test_multi_step_loop(self, agent):
        """Adapter returns actions across multiple steps before done."""
        init_resp = CUAAdapterResponse(
            actions=[BrowserAction(action_type="click", coordinate=[100, 100])],
            done=False,
        )
        step_resps = [
            CUAAdapterResponse(
                actions=[BrowserAction(action_type="type", text="hello")],
                done=False,
            ),
            CUAAdapterResponse(done=True, message="All done"),
        ]
        mock_adapter = self._make_mock_adapter(init_response=init_resp, step_responses=step_resps)

        step_http_resp = AsyncMock()
        step_http_resp.ok = True
        step_http_resp.cookies = {}

        with (
            patch.object(agent, "_create_adapter", return_value=mock_adapter),
            patch("responses_api_agents.browser_agent.app.raise_for_status", new_callable=AsyncMock),
            patch("responses_api_agents.browser_agent.app.get_response_json", return_value=self._make_step_response()),
        ):
            agent.server_client.post = AsyncMock(return_value=step_http_resp)

            cookie_jar = {"cookies": {}}
            trajectory, ls_dump, initial_ls, usage, debug_dir = await agent._responses_via_adapter(
                task_prompt="Fill form",
                env_id="test-env-multi",
                screenshot_b64="aW5pdGlhbA==",
                cookie_jar=cookie_jar,
            )

        assert len(trajectory.steps) == 2
        assert trajectory.steps[0].action.action_type == "click"
        assert trajectory.steps[1].action.action_type == "type"
        assert trajectory.final_message == "All done"
        assert mock_adapter.step.await_count == 2

    @pytest.mark.asyncio
    async def test_usage_accumulation(self, agent):
        """Token usage is accumulated across adapter calls."""
        from responses_api_agents.browser_agent.adapters.base import CUAAdapterUsage

        init_resp = CUAAdapterResponse(
            actions=[BrowserAction(action_type="click", coordinate=[10, 10])],
            done=False,
            usage=CUAAdapterUsage(input_tokens=100, output_tokens=50, total_tokens=150),
        )
        step_resp = CUAAdapterResponse(
            done=True,
            message="Done",
            usage=CUAAdapterUsage(input_tokens=200, output_tokens=80, total_tokens=280),
        )
        mock_adapter = self._make_mock_adapter(init_response=init_resp, step_responses=[step_resp])

        step_http_resp = AsyncMock()
        step_http_resp.ok = True
        step_http_resp.cookies = {}

        with (
            patch.object(agent, "_create_adapter", return_value=mock_adapter),
            patch("responses_api_agents.browser_agent.app.raise_for_status", new_callable=AsyncMock),
            patch("responses_api_agents.browser_agent.app.get_response_json", return_value=self._make_step_response()),
        ):
            agent.server_client.post = AsyncMock(return_value=step_http_resp)

            cookie_jar = {"cookies": {}}
            trajectory, ls_dump, initial_ls, usage, debug_dir = await agent._responses_via_adapter(
                task_prompt="Test usage",
                env_id="test-env-usage",
                screenshot_b64="aW5pdGlhbA==",
                cookie_jar=cookie_jar,
            )

        assert usage is not None
        assert usage.input_tokens == 300
        assert usage.output_tokens == 130
        assert usage.total_tokens == 430


class TestGenericAdapterLifecycle:
    """Tests for GenericCUAAdapter initialize/step/reset lifecycle."""

    def _make_adapter(self, **kwargs):
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        defaults = {
            "model": "test-model",
            "viewport_width": 1280,
            "viewport_height": 720,
            "turns_to_keep": 3,
            "denormalize_coords": False,
            "system_prompt": "Be helpful",
        }
        defaults.update(kwargs)
        return GenericCUAAdapter(**defaults)

    @pytest.mark.asyncio
    async def test_initialize_builds_input_and_calls_api(self):
        mock_response = {
            "output": [
                {
                    "type": "computer_call",
                    "id": "cu_1",
                    "call_id": "call_1",
                    "action": {"action": "left_click", "coordinate": [100, 200]},
                    "pending_safety_checks": [],
                    "status": "completed",
                }
            ],
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        }
        api_caller = AsyncMock(return_value=mock_response)
        adapter = self._make_adapter(api_caller=api_caller)

        result = await adapter.initialize("Click the button", "c2NyZWVuc2hvdA==")

        api_caller.assert_called_once()
        assert len(result.actions) == 1
        assert not result.done

        payload = api_caller.call_args[0][0]
        assert "tools" in payload
        assert payload["tools"][0]["type"] == "computer_use_preview"
        assert payload["tools"][0]["display_width"] == 1280
        assert payload["tools"][0]["display_height"] == 720

    @pytest.mark.asyncio
    async def test_step_appends_output_and_calls_api(self):
        init_response = {
            "output": [
                {
                    "type": "computer_call",
                    "id": "cu_1",
                    "call_id": "call_1",
                    "action": {"action": "left_click", "coordinate": [100, 200]},
                    "pending_safety_checks": [],
                    "status": "completed",
                }
            ],
        }
        step_response = {
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Done"}],
                }
            ],
        }
        call_count = 0

        async def mock_caller(payload):
            nonlocal call_count
            call_count += 1
            return init_response if call_count == 1 else step_response

        adapter = self._make_adapter(api_caller=mock_caller)
        await adapter.initialize("Click", "c2NyZWVuc2hvdA==")
        result = await adapter.step("bmV3X3NjcmVlbnNob3Q=", action_result="https://example.com")

        assert result.done
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_step_with_action_error(self):
        init_response = {
            "output": [
                {
                    "type": "computer_call",
                    "id": "cu_1",
                    "call_id": "call_1",
                    "action": {"action": "left_click", "coordinate": [100, 200]},
                    "pending_safety_checks": [],
                    "status": "completed",
                }
            ],
        }
        step_response = {
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": ""}],
                }
            ]
        }

        async def mock_caller(payload):
            return init_response if not hasattr(mock_caller, "called") else step_response

        mock_caller_fn = AsyncMock(side_effect=[init_response, step_response])
        adapter = self._make_adapter(api_caller=mock_caller_fn)
        await adapter.initialize("Click", "c2NyZWVuc2hvdA==")
        await adapter.step("bmV3", action_error="element not found")

        last_call_payload = mock_caller_fn.call_args_list[-1][0][0]
        has_error = any(
            "error:" in str(item.get("output", {}).get("current_url", ""))
            for item in last_call_payload.get("input", [])
            if isinstance(item, dict) and item.get("type") == "computer_call_output"
        )
        assert has_error

    @pytest.mark.asyncio
    async def test_initialize_without_api_caller_raises(self):
        adapter = self._make_adapter(api_caller=None)
        with pytest.raises(RuntimeError, match="api_caller"):
            await adapter.initialize("Click", "c2NyZWVuc2hvdA==")

    def test_reset_clears_state(self):
        adapter = self._make_adapter(api_caller=AsyncMock())
        adapter._input_items = [{"type": "message"}]
        adapter._pending_call_ids = ["call_1"]
        adapter._pending_safety_checks = [{"id": "sc_1"}]

        adapter.reset()

        assert adapter._input_items == []
        assert adapter._pending_call_ids == []
        assert adapter._pending_safety_checks == []


class TestGenericAdapterMoreActions:
    """Cover more action mapping branches in GenericCUAAdapter."""

    def _make_adapter(self, denormalize=False):
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        return GenericCUAAdapter(
            model="test",
            viewport_width=1280,
            viewport_height=720,
            turns_to_keep=8,
            denormalize_coords=denormalize,
        )

    def test_anthropic_double_click(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"action": "double_click", "coordinate": [100, 200]})
        assert result.action_type == "double_click"

    def test_anthropic_triple_click(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"action": "triple_click", "coordinate": [50, 50]})
        assert result.action_type == "triple_click"

    def test_anthropic_type(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"action": "type", "text": "hello"})
        assert result.action_type == "type"
        assert result.text == "hello"

    def test_anthropic_scroll(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"action": "scroll", "coordinate": [640, 360], "direction": "down", "amount": 3})
        assert result.action_type == "scroll"

    def test_anthropic_screenshot(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"action": "screenshot"})
        assert result.action_type == "screenshot"

    def test_anthropic_wait(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"action": "wait", "duration": 2000})
        assert result.action_type == "wait"

    def test_anthropic_hold_key(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"action": "hold_key", "key": "Shift"})
        assert result.action_type == "keypress"

    def test_anthropic_zoom(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"action": "zoom", "region": [0, 0, 500, 500]})
        assert result.action_type == "zoom"

    def test_gemini_double_click(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"type": "double_click", "x": 100, "y": 200})
        assert result.action_type == "double_click"

    def test_gemini_keypress(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"type": "keypress", "keys": "Enter"})
        assert result.action_type == "keypress"

    def test_gemini_wait(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"type": "wait", "ms": 500})
        assert result.action_type == "wait"
        assert result.duration == 500

    def test_gemini_wait_5_seconds(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"type": "wait_5_seconds"})
        assert result.action_type == "wait"
        assert result.duration == 5000

    def test_gemini_goto(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"type": "goto", "url": "https://google.com"})
        assert result.action_type == "goto"
        assert result.url == "https://google.com"

    def test_gemini_goto_bare_url_gets_https(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"type": "navigate", "url": "google.com"})
        assert result.url == "https://google.com"

    def test_gemini_go_back(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"type": "go_back"})
        assert result.action_type == "go_back"

    def test_gemini_go_forward(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"type": "go_forward"})
        assert result.action_type == "go_forward"

    def test_gemini_new_tab(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"type": "new_tab", "url": "https://test.com"})
        assert result.action_type == "new_tab"

    def test_gemini_switch_tab(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"type": "switch_tab", "tab_index": 2})
        assert result.action_type == "switch_tab"
        assert result.tab_index == 2

    def test_gemini_close_tab(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"type": "close_tab"})
        assert result.action_type == "close_tab"

    def test_gemini_drag(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"type": "drag", "x": 100, "y": 200, "destination_x": 300, "destination_y": 400})
        assert result.action_type == "drag"

    def test_gemini_scroll_document_up(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"type": "scroll_document", "direction": "up"})
        assert result.action_type == "scroll"
        assert result.scroll_y < 0

    def test_gemini_scroll_document_right(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"type": "scroll_document", "direction": "right"})
        assert result.action_type == "scroll"
        assert result.scroll_x > 0

    def test_gemini_hover(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"type": "hover", "x": 100, "y": 200})
        assert result.action_type == "hover"

    def test_gemini_search(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"type": "search"})
        assert result.action_type == "goto"

    def test_gemini_list_tabs_screenshot(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"type": "list_tabs"})
        assert result.action_type == "screenshot"

    def test_gemini_drag_and_drop_with_denorm(self):
        adapter = self._make_adapter(denormalize=True)
        result = adapter._map_action(
            {"type": "drag_and_drop", "x": 500, "y": 500, "destination_x": 700, "destination_y": 700}
        )
        assert result.action_type == "drag"
        assert result.start_coordinate[0] == int((500 / 1000.0) * 1280)

    def test_gemini_scroll_at(self):
        adapter = self._make_adapter(denormalize=True)
        result = adapter._map_action({"type": "scroll_at", "x": 500, "y": 500, "direction": "down", "magnitude": 400})
        assert result.action_type == "scroll"
        assert result.scroll_y > 0

    def test_unknown_anthropic_action_returns_none(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"action": "unknown_action_xyz"})
        assert result is None

    def test_unknown_gemini_action_returns_none(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"type": "unknown_action_xyz"})
        assert result is None

    def test_anthropic_right_click(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"action": "right_click", "coordinate": [100, 200]})
        assert result.action_type == "right_click"

    def test_anthropic_middle_click(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"action": "middle_click", "coordinate": [100, 200]})
        assert result.action_type == "middle_click"

    def test_anthropic_key(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"action": "key", "text": "Enter"})
        assert result.action_type == "keypress"

    def test_anthropic_mouse_move(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"action": "mouse_move", "coordinate": [100, 200]})
        assert result.action_type == "hover"

    def test_anthropic_left_click_drag(self):
        adapter = self._make_adapter()
        result = adapter._map_action(
            {"action": "left_click_drag", "start_coordinate": [100, 200], "coordinate": [300, 400]}
        )
        assert result.action_type == "drag"

    def test_gemini_click_at_with_denorm(self):
        adapter = self._make_adapter(denormalize=True)
        result = adapter._map_action({"type": "click_at", "x": 500, "y": 500})
        assert result.action_type == "click"
        assert result.coordinate[0] == int(500 / 1000 * 1280)

    def test_gemini_hover_at_with_denorm(self):
        adapter = self._make_adapter(denormalize=True)
        result = adapter._map_action({"type": "hover_at", "x": 500, "y": 500})
        assert result.action_type == "hover"

    def test_gemini_type_text_at(self):
        adapter = self._make_adapter(denormalize=True)
        result = adapter._map_action({"type": "type_text_at", "x": 500, "y": 500, "text": "hello"})
        assert result.action_type == "type"
        assert result.text == "hello"

    def test_gemini_scroll_document_left(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"type": "scroll_document", "direction": "left"})
        assert result.action_type == "scroll"
        assert result.scroll_x < 0

    def test_gemini_scroll_at_left(self):
        adapter = self._make_adapter(denormalize=True)
        result = adapter._map_action({"type": "scroll_at", "x": 500, "y": 500, "direction": "left", "magnitude": 300})
        assert result.action_type == "scroll"
        assert result.scroll_x < 0

    def test_gemini_scroll_at_up(self):
        adapter = self._make_adapter(denormalize=True)
        result = adapter._map_action({"type": "scroll_at", "x": 500, "y": 500, "direction": "up", "magnitude": 300})
        assert result.action_type == "scroll"
        assert result.scroll_y < 0

    def test_gemini_key_combination(self):
        adapter = self._make_adapter()
        result = adapter._map_action({"type": "key_combination", "keys": "Ctrl+C"})
        assert result.action_type == "keypress"
        assert result.keys == ["Ctrl", "C"]

    def test_get_coord_with_coordinate_key(self):
        adapter = self._make_adapter()
        result = adapter._get_coord({"coordinate": [100, 200]})
        assert result == [100, 200]

    def test_get_coord_with_no_coords(self):
        adapter = self._make_adapter()
        result = adapter._get_coord({})
        assert result is None

    def test_get_gemini_coord_fallback_to_coordinate(self):
        adapter = self._make_adapter()
        result = adapter._get_gemini_coord({"coordinate": [100, 200]})
        assert result == [100, 200]


class TestGenericAdapterGcAndTrim:
    """Tests for _gc_old_screenshots and _trim_input_items edge cases."""

    def _make_adapter(self, turns=2):
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        return GenericCUAAdapter(
            model="test",
            viewport_width=1280,
            viewport_height=720,
            turns_to_keep=turns,
            denormalize_coords=False,
        )

    def test_trim_no_op_on_small_list(self):
        adapter = self._make_adapter()
        adapter._input_items = [{"type": "message"}, {"type": "computer_call"}]
        adapter._trim_input_items()
        assert len(adapter._input_items) == 2

    def test_gc_handles_message_items_with_images(self):
        from responses_api_agents.browser_agent.adapters.generic_adapter import SCREENSHOT_PLACEHOLDER

        adapter = self._make_adapter(turns=1)
        adapter._input_items = [
            {"type": "message", "content": "first"},
            {
                "type": "message",
                "content": [{"type": "input_image", "detail": "auto", "image_url": "data:image/png;base64,OLD"}],
            },
            {"type": "computer_call_output", "output": {"image_url": "data:image/png;base64,OLD2"}},
            {"type": "computer_call_output", "output": {"image_url": "data:image/png;base64,OLD3"}},
            {"type": "computer_call_output", "output": {"image_url": "data:image/png;base64,KEEP1"}},
            {"type": "computer_call_output", "output": {"image_url": "data:image/png;base64,KEEP2"}},
        ]
        adapter._gc_old_screenshots()
        assert adapter._input_items[1]["content"][0]["image_url"] == SCREENSHOT_PLACEHOLDER
        assert adapter._input_items[2]["output"]["image_url"] == SCREENSHOT_PLACEHOLDER
        assert adapter._input_items[3]["output"]["image_url"] == SCREENSHOT_PLACEHOLDER
        assert adapter._input_items[4]["output"]["image_url"] == "data:image/png;base64,KEEP1"

    def test_gc_skips_non_dict_items(self):
        adapter = self._make_adapter(turns=1)
        adapter._input_items = [
            {"type": "message"},
            "a_string_item",
            {"type": "computer_call_output", "output": {"image_url": "data:image/png;base64,OLD"}},
            {"type": "computer_call_output", "output": {"image_url": "data:image/png;base64,KEEP"}},
        ]
        adapter._gc_old_screenshots()
        assert adapter._input_items[1] == "a_string_item"


class TestGenericAdapterSafetyChecks:
    """Tests for safety check flow in GenericCUAAdapter."""

    def _make_adapter(self):
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        return GenericCUAAdapter(
            model="test",
            viewport_width=1280,
            viewport_height=720,
            turns_to_keep=8,
            denormalize_coords=False,
        )

    @pytest.mark.asyncio
    async def test_safety_checks_propagated_in_step(self):
        init_response = {
            "output": [
                {
                    "type": "computer_call",
                    "id": "cu_1",
                    "call_id": "call_1",
                    "action": {"action": "left_click", "coordinate": [100, 200]},
                    "pending_safety_checks": [{"id": "sc_1", "code": "unsafe_content", "message": "Caution"}],
                    "status": "completed",
                }
            ],
        }
        step_response = {
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ],
        }

        call_payloads = []

        async def mock_caller(payload):
            call_payloads.append(payload)
            return init_response if len(call_payloads) == 1 else step_response

        adapter = self._make_adapter()
        adapter._api_caller = mock_caller

        await adapter.initialize("Click", "c2NyZWVuc2hvdA==")
        await adapter.step("bmV3")

        step_payload = call_payloads[1]
        has_safety = any(
            "acknowledged_safety_checks" in item for item in step_payload.get("input", []) if isinstance(item, dict)
        )
        assert has_safety


class TestCreateAdapter:
    """Tests for BrowserAgent._create_adapter covering all adapter type branches."""

    def _make_agent(self, adapter_type="openai", **overrides):
        from nemo_gym.server_utils import ServerClient
        from responses_api_agents.browser_agent.app import BrowserAgent

        config = _make_agent_config(cua_adapter_type=adapter_type, **overrides)
        return BrowserAgent(config=config, server_client=MagicMock(spec=ServerClient))

    def test_create_openai_adapter(self):
        agent = self._make_agent("openai")
        cookie_jar = {"cookies": {}}
        adapter = agent._create_adapter(cookie_jar)

        from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

        assert isinstance(adapter, OpenAICUAAdapter)

    def test_create_anthropic_sonnet_adapter(self):
        agent = self._make_agent("anthropic_sonnet", cua_model="claude-sonnet-4-20250514")
        cookie_jar = {"cookies": {}}
        adapter = agent._create_adapter(cookie_jar)

        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        assert isinstance(adapter, GenericCUAAdapter)

    def test_create_anthropic_opus_adapter(self):
        agent = self._make_agent("anthropic_opus", cua_model="claude-opus-4-6")
        cookie_jar = {"cookies": {}}
        adapter = agent._create_adapter(cookie_jar)

        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        assert isinstance(adapter, GenericCUAAdapter)

    def test_create_gemini_adapter(self):
        agent = self._make_agent("gemini", cua_model="gemini-2.5-cu")
        cookie_jar = {"cookies": {}}
        adapter = agent._create_adapter(cookie_jar)

        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        assert isinstance(adapter, GenericCUAAdapter)

    def test_unknown_adapter_type_raises(self):
        agent = self._make_agent("unknown_provider")
        cookie_jar = {"cookies": {}}
        with pytest.raises(ValueError, match="Unknown adapter type"):
            agent._create_adapter(cookie_jar)

    def test_no_model_server_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _make_agent_config(cua_adapter_type="openai", model_server=None)

    def test_viewport_override(self):
        agent = self._make_agent("openai", viewport_width=800, viewport_height=600)
        cookie_jar = {"cookies": {}}
        adapter = agent._create_adapter(cookie_jar, viewport_width=1920, viewport_height=1080)

        assert adapter._viewport_width == 1920
        assert adapter._viewport_height == 1080


class TestModelServerCallers:
    """Tests for _make_openai_model_server_caller and _make_generic_model_server_caller."""

    def _make_agent(self, adapter_type="openai"):
        from nemo_gym.server_utils import ServerClient
        from responses_api_agents.browser_agent.app import BrowserAgent

        config = _make_agent_config(cua_adapter_type=adapter_type)
        return BrowserAgent(config=config, server_client=MagicMock(spec=ServerClient))

    @pytest.mark.asyncio
    async def test_openai_caller_success(self):
        agent = self._make_agent("openai")

        mock_resp = AsyncMock()
        mock_resp.ok = True
        mock_resp.cookies = {"session": "abc"}
        agent.server_client.post = AsyncMock(return_value=mock_resp)

        with (
            patch("responses_api_agents.browser_agent.app.raise_for_status", new_callable=AsyncMock),
            patch(
                "responses_api_agents.browser_agent.app.get_response_json",
                new_callable=AsyncMock,
                return_value={"output": []},
            ),
        ):
            cookie_jar = {"cookies": {}}
            caller = agent._make_openai_model_server_caller(cookie_jar)
            result = await caller({"model": "test", "input": []})

        assert result == {"output": []}
        assert cookie_jar["cookies"] == {"session": "abc"}

    @pytest.mark.asyncio
    async def test_openai_caller_error_logs(self):
        agent = self._make_agent("openai")

        mock_resp = AsyncMock()
        mock_resp.ok = False
        mock_resp.status = 500
        mock_resp.cookies = {}
        mock_resp.content.read = AsyncMock(return_value=b"Internal Server Error")
        agent.server_client.post = AsyncMock(return_value=mock_resp)

        with (
            patch(
                "responses_api_agents.browser_agent.app.raise_for_status",
                new_callable=AsyncMock,
                side_effect=RuntimeError("500"),
            ),
        ):
            cookie_jar = {"cookies": {}}
            caller = agent._make_openai_model_server_caller(cookie_jar)
            with pytest.raises(RuntimeError, match="500"):
                await caller({"model": "test", "input": []})

    @pytest.mark.asyncio
    async def test_generic_caller_success(self):
        agent = self._make_agent("anthropic_sonnet")

        mock_resp = AsyncMock()
        mock_resp.ok = True
        mock_resp.cookies = {"sid": "xyz"}
        agent.server_client.post = AsyncMock(return_value=mock_resp)

        with (
            patch("responses_api_agents.browser_agent.app.raise_for_status", new_callable=AsyncMock),
            patch(
                "responses_api_agents.browser_agent.app.get_response_json",
                new_callable=AsyncMock,
                return_value={"output": []},
            ),
        ):
            cookie_jar = {"cookies": {}}
            caller = agent._make_generic_model_server_caller(cookie_jar)
            result = await caller({"model": "test", "input": []})

        assert result == {"output": []}
        assert cookie_jar["cookies"] == {"sid": "xyz"}

    @pytest.mark.asyncio
    async def test_generic_caller_error_logs(self):
        agent = self._make_agent("anthropic_sonnet")

        mock_resp = AsyncMock()
        mock_resp.ok = False
        mock_resp.status = 503
        mock_resp.cookies = {}
        mock_resp.content.read = AsyncMock(return_value=b"Service Unavailable")
        agent.server_client.post = AsyncMock(return_value=mock_resp)

        with (
            patch(
                "responses_api_agents.browser_agent.app.raise_for_status",
                new_callable=AsyncMock,
                side_effect=RuntimeError("503"),
            ),
        ):
            cookie_jar = {"cookies": {}}
            caller = agent._make_generic_model_server_caller(cookie_jar)
            with pytest.raises(RuntimeError, match="503"):
                await caller({"model": "test", "input": []})


class TestDoResponses:
    """Tests for the _do_responses pipeline: seed session -> run loop -> close."""

    def _make_agent(self, adapter_type="openai"):
        from nemo_gym.server_utils import ServerClient
        from responses_api_agents.browser_agent.app import BrowserAgent

        config = _make_agent_config(cua_adapter_type=adapter_type)
        return BrowserAgent(config=config, server_client=MagicMock(spec=ServerClient))

    @pytest.mark.asyncio
    async def test_do_responses_basic(self):
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
        from responses_api_agents.browser_agent.app import BrowserAgentRunRequest

        agent = self._make_agent()

        seed_resp_data = {"env_id": "env-123", "screenshot": "c2NyZWVu"}
        empty_trajectory = CUATrajectory(steps=[], task_prompt="Click it", initial_screenshot="c2NyZWVu")

        with (
            patch("responses_api_agents.browser_agent.app.raise_for_status", new_callable=AsyncMock),
            patch(
                "responses_api_agents.browser_agent.app.get_response_json",
                new_callable=AsyncMock,
                return_value=seed_resp_data,
            ),
            patch.object(
                agent,
                "_responses_via_adapter",
                new_callable=AsyncMock,
                return_value=(empty_trajectory, "", "", None, None),
            ),
        ):
            mock_close_resp = AsyncMock()
            mock_close_resp.cookies = {}

            mock_seed_resp = AsyncMock()
            mock_seed_resp.cookies = {}

            agent.server_client.post = AsyncMock(side_effect=[mock_seed_resp, mock_close_resp])

            body = BrowserAgentRunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="Click it"),
                verifier_metadata={"start_url": "https://example.com"},
            )
            cookie_jar = {"cookies": {}}
            result, debug_dir = await agent._do_responses(body, cookie_jar)

        assert result.env_id == "env-123"
        assert debug_dir is None

    @pytest.mark.asyncio
    async def test_do_responses_extracts_task_prompt_from_list(self):
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
        from responses_api_agents.browser_agent.app import BrowserAgentRunRequest

        agent = self._make_agent()

        seed_resp_data = {"env_id": "env-456", "screenshot": "c2NyZWVu"}
        empty_trajectory = CUATrajectory(steps=[], task_prompt="Search", initial_screenshot="c2NyZWVu")

        with (
            patch("responses_api_agents.browser_agent.app.raise_for_status", new_callable=AsyncMock),
            patch(
                "responses_api_agents.browser_agent.app.get_response_json",
                new_callable=AsyncMock,
                return_value=seed_resp_data,
            ),
            patch.object(
                agent,
                "_responses_via_adapter",
                new_callable=AsyncMock,
                return_value=(empty_trajectory, "", "", None, None),
            ) as mock_via,
        ):
            mock_resp = AsyncMock()
            mock_resp.cookies = {}
            agent.server_client.post = AsyncMock(return_value=mock_resp)

            body = BrowserAgentRunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                    input=[{"type": "message", "role": "user", "content": "Search for cats"}]
                ),
            )
            cookie_jar = {"cookies": {}}
            await agent._do_responses(body, cookie_jar)

        mock_via.assert_called_once()
        call_args = mock_via.call_args
        assert call_args[0][0] == "Search for cats"


class TestBrowserAgentResponses:
    """Tests for the responses() and run() endpoints."""

    def _make_agent(self, **overrides):
        from nemo_gym.server_utils import ServerClient
        from responses_api_agents.browser_agent.app import BrowserAgent

        config = _make_agent_config(**overrides)
        return BrowserAgent(config=config, server_client=MagicMock(spec=ServerClient))

    @pytest.mark.asyncio
    async def test_responses_endpoint(self):
        from nemo_gym.openai_utils import (
            NeMoGymResponseCreateParamsNonStreaming,
            NeMoGymResponseOutputMessage,
            NeMoGymResponseOutputText,
        )
        from responses_api_agents.browser_agent.app import BrowserAgentRunRequest, CUANeMoGymResponse

        agent = self._make_agent()

        mock_nemo_resp = CUANeMoGymResponse(
            id="cua_test",
            created_at=0,
            model="test",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_1",
                    content=[NeMoGymResponseOutputText(text="done", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            env_id="env-1",
            trajectory=CUATrajectory(steps=[], task_prompt="test", initial_screenshot=""),
        )

        with patch.object(agent, "_do_responses", new_callable=AsyncMock, return_value=(mock_nemo_resp, None)):
            request = MagicMock()
            request.cookies = {}
            response = MagicMock()

            body = BrowserAgentRunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="test"),
            )
            result = await agent.responses(request, response, body)

        assert result.id == "cua_test"

    @pytest.mark.asyncio
    async def test_run_endpoint(self):
        from nemo_gym.openai_utils import (
            NeMoGymResponseCreateParamsNonStreaming,
            NeMoGymResponseOutputMessage,
            NeMoGymResponseOutputText,
        )
        from responses_api_agents.browser_agent.app import BrowserAgentRunRequest, CUANeMoGymResponse

        agent = self._make_agent()

        mock_nemo_resp = CUANeMoGymResponse(
            id="cua_test",
            created_at=0,
            model="test",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_1",
                    content=[NeMoGymResponseOutputText(text="done", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            env_id="env-1",
            trajectory=CUATrajectory(steps=[], task_prompt="test", initial_screenshot=""),
        )

        body = BrowserAgentRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="test"),
        )
        mock_verify_resp_data = {
            "reward": 1.0,
            "response": mock_nemo_resp.model_dump(),
            "responses_create_params": body.responses_create_params.model_dump(),
        }
        mock_http_resp = AsyncMock()
        mock_http_resp.cookies = {}

        with (
            patch.object(agent, "_do_responses", new_callable=AsyncMock, return_value=(mock_nemo_resp, None)),
            patch("responses_api_agents.browser_agent.app.raise_for_status", new_callable=AsyncMock),
            patch(
                "responses_api_agents.browser_agent.app.get_response_json",
                new_callable=AsyncMock,
                return_value=mock_verify_resp_data,
            ),
        ):
            agent.server_client.post = AsyncMock(return_value=mock_http_resp)

            request = MagicMock()
            request.cookies = {}

            result = await agent.run(request, body)

        assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_run_endpoint_with_debug(self, tmp_path):
        from nemo_gym.openai_utils import (
            NeMoGymResponseCreateParamsNonStreaming,
            NeMoGymResponseOutputMessage,
            NeMoGymResponseOutputText,
        )
        from responses_api_agents.browser_agent.app import BrowserAgentRunRequest, CUANeMoGymResponse

        agent = self._make_agent(cua_debug_trajectories=True, cua_debug_output_dir=str(tmp_path))

        mock_nemo_resp = CUANeMoGymResponse(
            id="cua_test",
            created_at=0,
            model="test",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_1",
                    content=[NeMoGymResponseOutputText(text="done", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            env_id="env-1",
            trajectory=CUATrajectory(steps=[], task_prompt="test", initial_screenshot=""),
        )

        debug_dir = tmp_path / "debug_run"
        debug_dir.mkdir(parents=True, exist_ok=True)

        body = BrowserAgentRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="test"),
        )
        mock_verify_resp_data = {
            "reward": 0.5,
            "response": mock_nemo_resp.model_dump(),
            "responses_create_params": body.responses_create_params.model_dump(),
        }
        mock_http_resp = AsyncMock()
        mock_http_resp.cookies = {}

        with (
            patch.object(agent, "_do_responses", new_callable=AsyncMock, return_value=(mock_nemo_resp, debug_dir)),
            patch("responses_api_agents.browser_agent.app.raise_for_status", new_callable=AsyncMock),
            patch(
                "responses_api_agents.browser_agent.app.get_response_json",
                new_callable=AsyncMock,
                return_value=mock_verify_resp_data,
            ),
            patch("responses_api_agents.browser_agent.app.finalize_debug_trajectory") as mock_finalize,
        ):
            agent.server_client.post = AsyncMock(return_value=mock_http_resp)

            request = MagicMock()
            request.cookies = {}

            result = await agent.run(request, body)

        assert result.reward == 0.5
        mock_finalize.assert_called_once()

    @pytest.mark.asyncio
    async def test_do_responses_task_prompt_from_content_parts(self):
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
        from responses_api_agents.browser_agent.app import BrowserAgentRunRequest

        agent = self._make_agent()

        seed_resp_data = {"env_id": "env-prompt", "screenshot": "c2NyZWVu"}
        empty_trajectory = CUATrajectory(steps=[], task_prompt="Click the green button", initial_screenshot="c2NyZWVu")

        with (
            patch("responses_api_agents.browser_agent.app.raise_for_status", new_callable=AsyncMock),
            patch(
                "responses_api_agents.browser_agent.app.get_response_json",
                new_callable=AsyncMock,
                return_value=seed_resp_data,
            ),
            patch.object(
                agent,
                "_responses_via_adapter",
                new_callable=AsyncMock,
                return_value=(empty_trajectory, "", "", None, None),
            ) as mock_via,
        ):
            mock_resp = AsyncMock()
            mock_resp.cookies = {}
            agent.server_client.post = AsyncMock(return_value=mock_resp)

            body = BrowserAgentRunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                    input=[
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Click the green button"}],
                        }
                    ]
                ),
            )
            cookie_jar = {"cookies": {}}
            await agent._do_responses(body, cookie_jar)

        mock_via.assert_called_once()
        assert mock_via.call_args[0][0] == "Click the green button"

    @pytest.mark.asyncio
    async def test_do_responses_close_session_error_handled(self):
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
        from responses_api_agents.browser_agent.app import BrowserAgentRunRequest

        agent = self._make_agent()

        seed_resp_data = {"env_id": "env-close-err", "screenshot": "c2NyZWVu"}
        empty_trajectory = CUATrajectory(steps=[], task_prompt="test", initial_screenshot="c2NyZWVu")

        call_count = [0]

        async def mock_post(**kwargs):
            call_count[0] += 1
            if call_count[0] > 1:
                raise ConnectionError("Connection refused during close")
            resp = AsyncMock()
            resp.cookies = {}
            return resp

        with (
            patch("responses_api_agents.browser_agent.app.raise_for_status", new_callable=AsyncMock),
            patch(
                "responses_api_agents.browser_agent.app.get_response_json",
                new_callable=AsyncMock,
                return_value=seed_resp_data,
            ),
            patch.object(
                agent,
                "_responses_via_adapter",
                new_callable=AsyncMock,
                return_value=(empty_trajectory, "", "", None, None),
            ),
        ):
            agent.server_client.post = AsyncMock(side_effect=mock_post)

            body = BrowserAgentRunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="test"),
            )
            cookie_jar = {"cookies": {}}
            result, _ = await agent._do_responses(body, cookie_jar)

        assert result.env_id == "env-close-err"


class TestBrowserAgentTimeoutAndDebug:
    """Tests for timeout, debug trajectory, and consecutive failure branches."""

    @pytest.fixture
    def agent(self):
        from nemo_gym.server_utils import ServerClient
        from responses_api_agents.browser_agent.app import BrowserAgent

        config = _make_agent_config(
            cua_debug_trajectories=True,
            run_timeout_seconds=0.001,
        )
        return BrowserAgent(config=config, server_client=MagicMock(spec=ServerClient))

    @pytest.mark.asyncio
    async def test_timeout_ends_loop(self, agent):
        init_resp = CUAAdapterResponse(
            actions=[BrowserAction(action_type="click", coordinate=[100, 200])],
            done=False,
        )

        step_resp_data = {"screenshot": "c2NyZWVu", "current_url": "https://example.com", "error": None}
        mock_http_resp = AsyncMock()
        mock_http_resp.ok = True
        mock_http_resp.cookies = {}

        adapter = AsyncMock()
        adapter.initialize.return_value = init_resp
        adapter.step.return_value = CUAAdapterResponse(
            actions=[BrowserAction(action_type="click", coordinate=[200, 300])],
            done=False,
        )
        adapter.reset = MagicMock()

        with (
            patch.object(agent, "_create_adapter", return_value=adapter),
            patch("responses_api_agents.browser_agent.app.raise_for_status", new_callable=AsyncMock),
            patch(
                "responses_api_agents.browser_agent.app.get_response_json",
                new_callable=AsyncMock,
                return_value=step_resp_data,
            ),
            patch(
                "responses_api_agents.browser_agent.app.init_debug_trajectory",
                return_value=Path("/tmp/test_debug"),
            ),
        ):
            agent.server_client.post = AsyncMock(return_value=mock_http_resp)

            cookie_jar = {"cookies": {}}
            trajectory, _, _, _, _ = await agent._responses_via_adapter(
                task_prompt="Test timeout",
                env_id="test-env-timeout",
                screenshot_b64="aW5pdGlhbA==",
                cookie_jar=cookie_jar,
            )

        adapter.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_browser_stuck_ends_loop(self):
        from nemo_gym.server_utils import ServerClient
        from responses_api_agents.browser_agent.app import BrowserAgent

        config = _make_agent_config(cua_debug_trajectories=False)
        agent = BrowserAgent(config=config, server_client=MagicMock(spec=ServerClient))

        init_resp = CUAAdapterResponse(
            actions=[BrowserAction(action_type="click", coordinate=[100, 200])],
            done=False,
        )
        adapter = AsyncMock()
        adapter.initialize.return_value = init_resp
        adapter.reset = MagicMock()

        stuck_resp = {"screenshot": "c2NyZWVu", "current_url": "error:browser_stuck", "error": None}
        mock_http_resp = AsyncMock()
        mock_http_resp.ok = True
        mock_http_resp.cookies = {}

        with (
            patch.object(agent, "_create_adapter", return_value=adapter),
            patch("responses_api_agents.browser_agent.app.raise_for_status", new_callable=AsyncMock),
            patch(
                "responses_api_agents.browser_agent.app.get_response_json",
                new_callable=AsyncMock,
                return_value=stuck_resp,
            ),
        ):
            agent.server_client.post = AsyncMock(return_value=mock_http_resp)

            cookie_jar = {"cookies": {}}
            trajectory, _, _, _, _ = await agent._responses_via_adapter(
                task_prompt="Stuck test",
                env_id="test-env-stuck2",
                screenshot_b64="aW5pdGlhbA==",
                cookie_jar=cookie_jar,
            )

        assert len(trajectory.steps) == 0
        adapter.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_consecutive_action_failures_end_loop(self):
        from nemo_gym.server_utils import ServerClient
        from responses_api_agents.browser_agent.app import BrowserAgent

        config = _make_agent_config(cua_debug_trajectories=False, max_steps=1)
        agent = BrowserAgent(config=config, server_client=MagicMock(spec=ServerClient))

        init_resp = CUAAdapterResponse(
            actions=[
                BrowserAction(action_type="click", coordinate=[100, 200]),
                BrowserAction(action_type="click", coordinate=[200, 300]),
                BrowserAction(action_type="click", coordinate=[300, 400]),
            ],
            done=False,
            usage=None,
        )
        adapter = AsyncMock()
        adapter.initialize.return_value = init_resp
        adapter.reset = MagicMock()

        mock_http_resp = AsyncMock()
        mock_http_resp.cookies = {}

        with (
            patch.object(agent, "_create_adapter", return_value=adapter),
            patch(
                "responses_api_agents.browser_agent.app.raise_for_status",
                new_callable=AsyncMock,
                side_effect=RuntimeError("server error"),
            ),
        ):
            agent.server_client.post = AsyncMock(return_value=mock_http_resp)

            cookie_jar = {"cookies": {}}
            trajectory, _, _, _, _ = await agent._responses_via_adapter(
                task_prompt="Fail test",
                env_id="test-env-failconsec",
                screenshot_b64="aW5pdGlhbA==",
                cookie_jar=cookie_jar,
            )

        assert len(trajectory.steps) == 0
        adapter.reset.assert_called_once()


class TestGenericAdapterMoreActionTypes:
    """Cover uncovered action mapping branches in GenericCUAAdapter."""

    def _make_adapter(self, denormalize=False):
        from responses_api_agents.browser_agent.adapters.generic_adapter import GenericCUAAdapter

        return GenericCUAAdapter(
            model="test", viewport_width=1280, viewport_height=720, denormalize_coords=denormalize
        )

    def test_type_text_at_action(self):
        adapter = self._make_adapter(denormalize=True)
        action = {"type": "type_text_at", "x": 500, "y": 500, "text": "hello"}
        result = adapter._map_action(action)
        assert result is not None
        assert result.action_type == "type"
        assert result.text == "hello"
        assert result.coordinate is not None

    def test_scroll_openai_style(self):
        adapter = self._make_adapter()
        action = {"type": "scroll", "x": 100, "y": 200, "scroll_x": 0, "scroll_y": -300}
        result = adapter._map_action(action)
        assert result is not None
        assert result.action_type == "scroll"
        assert result.scroll_y == -300

    def test_scroll_at_right_direction(self):
        adapter = self._make_adapter(denormalize=True)
        action = {"type": "scroll_at", "x": 500, "y": 500, "direction": "right", "magnitude": 500}
        result = adapter._map_action(action)
        assert result is not None
        assert result.action_type == "scroll"
        assert result.scroll_x > 0
        assert result.scroll_y == 0

    def test_screenshot_openai_style(self):
        adapter = self._make_adapter()
        action = {"type": "screenshot"}
        result = adapter._map_action(action)
        assert result is not None
        assert result.action_type == "screenshot"

    def test_get_gemini_coord_no_denormalize(self):
        adapter = self._make_adapter(denormalize=False)
        coord = adapter._get_gemini_coord({"x": 500, "y": 300})
        assert coord == [500, 300]

    @pytest.mark.asyncio
    async def test_step_empty_input_items_returns_done(self):
        adapter = self._make_adapter()
        adapter._input_items = []
        adapter._pending_call_ids = []
        adapter._pending_safety_checks = []
        result = await adapter.step("c2NyZWVu")
        assert result.done is True


class TestBrowserAgentUncoveredBranches:
    """Cover remaining uncovered branches in BrowserAgent."""

    def _make_agent(self, **overrides):
        from nemo_gym.server_utils import ServerClient
        from responses_api_agents.browser_agent.app import BrowserAgent

        config = _make_agent_config(**overrides)
        return BrowserAgent(config=config, server_client=MagicMock(spec=ServerClient))

    @pytest.mark.asyncio
    async def test_debug_trajectory_init_failure_handled(self):
        agent = self._make_agent(cua_debug_trajectories=True, max_steps=1)

        init_resp = CUAAdapterResponse(
            actions=[],
            done=True,
            message="Done",
            usage=CUAAdapterUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        adapter = AsyncMock()
        adapter.initialize.return_value = init_resp
        adapter.reset = MagicMock()

        mock_http_resp = AsyncMock()
        mock_http_resp.cookies = {}

        with (
            patch.object(agent, "_create_adapter", return_value=adapter),
            patch(
                "responses_api_agents.browser_agent.app.init_debug_trajectory",
                side_effect=OSError("permission denied"),
            ),
        ):
            agent.server_client.post = AsyncMock(return_value=mock_http_resp)

            cookie_jar = {"cookies": {}}
            trajectory, _, _, _, debug_dir = await agent._responses_via_adapter(
                task_prompt="test",
                env_id="env-debug-fail",
                screenshot_b64="aW5pdGlhbA==",
                cookie_jar=cookie_jar,
            )

        assert debug_dir is None
        adapter.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_dump_local_storage_success_path(self):
        agent = self._make_agent(cua_debug_trajectories=False, max_steps=1)

        init_resp = CUAAdapterResponse(actions=[], done=True, message="Done")
        adapter = AsyncMock()
        adapter.initialize.return_value = init_resp
        adapter.reset = MagicMock()

        step_resp = AsyncMock()
        step_resp.cookies = {}

        ls_data = {"local_storage_dump": '{"key":"val"}', "initial_local_storage": '{"init":"data"}'}

        with (
            patch.object(agent, "_create_adapter", return_value=adapter),
            patch("responses_api_agents.browser_agent.app.raise_for_status", new_callable=AsyncMock),
            patch(
                "responses_api_agents.browser_agent.app.get_response_json",
                new_callable=AsyncMock,
                return_value=ls_data,
            ),
        ):
            agent.server_client.post = AsyncMock(return_value=step_resp)

            cookie_jar = {"cookies": {}}
            trajectory, ls_dump, init_ls, _, _ = await agent._responses_via_adapter(
                task_prompt="test",
                env_id="env-ls-ok",
                screenshot_b64="aW5pdGlhbA==",
                cookie_jar=cookie_jar,
            )

        assert ls_dump == '{"key":"val"}'
        assert init_ls == '{"init":"data"}'

    @pytest.mark.asyncio
    async def test_task_prompt_from_part_with_text_attr(self):
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
        from responses_api_agents.browser_agent.app import BrowserAgentRunRequest

        agent = self._make_agent()
        empty_trajectory = CUATrajectory(steps=[], task_prompt="test", initial_screenshot="c2NyZWVu")
        seed_resp_data = {"env_id": "env-text-attr", "screenshot": "c2NyZWVu"}

        mock_via = AsyncMock(return_value=(empty_trajectory, "", "", None, None))

        mock_http_resp = AsyncMock()
        mock_http_resp.cookies = {}

        with (
            patch("responses_api_agents.browser_agent.app.raise_for_status", new_callable=AsyncMock),
            patch(
                "responses_api_agents.browser_agent.app.get_response_json",
                new_callable=AsyncMock,
                return_value=seed_resp_data,
            ),
            patch.object(agent, "_responses_via_adapter", mock_via),
        ):
            agent.server_client.post = AsyncMock(return_value=mock_http_resp)

            body = BrowserAgentRunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                    input=[
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Do the thing"}],
                        }
                    ]
                ),
            )
            cookie_jar = {"cookies": {}}
            await agent._do_responses(body, cookie_jar)

        assert mock_via.call_args[0][0] == "Do the thing"

    @pytest.mark.asyncio
    async def test_responses_endpoint_propagates_cookies(self):
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
        from responses_api_agents.browser_agent.app import BrowserAgentRunRequest

        agent = self._make_agent()
        empty_trajectory = CUATrajectory(steps=[], task_prompt="test", initial_screenshot="c2NyZWVu")

        from responses_api_agents.browser_agent.app import _build_nemo_response

        nemo_resp = _build_nemo_response(empty_trajectory, "env-cookie", "", "", "test-model")

        mock_request = MagicMock()
        mock_request.cookies = {}

        mock_response = MagicMock()

        with patch.object(
            agent,
            "_do_responses",
            new_callable=AsyncMock,
            return_value=(nemo_resp, None),
        ):
            body = BrowserAgentRunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="test"),
            )
            result = await agent.responses(mock_request, mock_response, body)

        assert result.env_id == "env-cookie"

    @pytest.mark.asyncio
    async def test_run_exception_propagates(self):
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
        from responses_api_agents.browser_agent.app import BrowserAgentRunRequest

        agent = self._make_agent()
        mock_request = MagicMock()
        mock_request.cookies = {}

        with patch.object(
            agent,
            "_do_responses",
            new_callable=AsyncMock,
            side_effect=RuntimeError("test error"),
        ):
            body = BrowserAgentRunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="test"),
            )
            with pytest.raises(RuntimeError, match="test error"):
                await agent.run(mock_request, body)

    @pytest.mark.asyncio
    async def test_finalize_debug_trajectory_failure_handled(self):
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
        from responses_api_agents.browser_agent.app import BrowserAgentRunRequest, _build_nemo_response

        agent = self._make_agent(cua_debug_trajectories=True)
        empty_trajectory = CUATrajectory(steps=[], task_prompt="test", initial_screenshot="c2NyZWVu")
        nemo_resp = _build_nemo_response(empty_trajectory, "env-finalize-err", "", "", "test-model")

        verify_data = {
            "reward": 1.0,
            "responses_create_params": {"input": "test"},
            "response": nemo_resp.model_dump(),
        }
        mock_http_resp = AsyncMock()
        mock_http_resp.cookies = {}
        mock_request = MagicMock()
        mock_request.cookies = {}

        with (
            patch.object(
                agent,
                "_do_responses",
                new_callable=AsyncMock,
                return_value=(nemo_resp, Path("/tmp/debug-test")),
            ),
            patch("responses_api_agents.browser_agent.app.raise_for_status", new_callable=AsyncMock),
            patch(
                "responses_api_agents.browser_agent.app.get_response_json",
                new_callable=AsyncMock,
                return_value=verify_data,
            ),
            patch(
                "responses_api_agents.browser_agent.app.finalize_debug_trajectory",
                side_effect=OSError("disk full"),
            ),
        ):
            agent.server_client.post = AsyncMock(return_value=mock_http_resp)

            body = BrowserAgentRunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="test"),
            )
            result = await agent.run(mock_request, body)

        assert result.reward == 1.0
