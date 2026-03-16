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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.browser_gym.app import BrowserGymResourcesServer
from resources_servers.browser_gym.browser_pool import BrowserPool, _normalize_key
from resources_servers.browser_gym.schemas import (
    BrowserAction,
    BrowserGymResourcesServerConfig,
    CUACloseRequest,
    CUADumpLocalStorageRequest,
    CUANeMoGymResponse,
    CUASeedSessionRequest,
    CUAStep,
    CUAStepRequest,
    CUATrajectory,
    CUAVerifyRequest,
)


def _make_config():
    return BrowserGymResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        max_concurrent_browsers=2,
        default_viewport_width=1280,
        default_viewport_height=720,
    )


class TestSanity:
    def test_server_instantiation(self) -> None:
        with patch("resources_servers.browser_gym.app.ensure_playwright"):
            config = _make_config()
            server = BrowserGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
            assert server.browser_pool is not None


class TestBrowserPool:
    def test_pool_creation(self):
        pool = BrowserPool(max_concurrent=4, default_viewport_width=1024, default_viewport_height=768)
        assert pool._default_viewport_width == 1024
        assert pool._default_viewport_height == 768

    def test_get_session_not_found(self):
        pool = BrowserPool()
        with pytest.raises(KeyError, match="No browser session found"):
            pool.get_session("nonexistent")

    @pytest.mark.asyncio
    async def test_close_session_idempotent(self):
        pool = BrowserPool()
        result = await pool.close_session("nonexistent")
        assert result is False


class TestBrowserAction:
    def test_click_action(self):
        action = BrowserAction(action_type="click", coordinate=[100, 200])
        assert action.action_type == "click"
        assert action.coordinate == [100, 200]

    def test_type_action(self):
        action = BrowserAction(action_type="type", text="hello", coordinate=[50, 50], press_enter=True)
        assert action.text == "hello"
        assert action.press_enter is True

    def test_scroll_action_with_direction(self):
        action = BrowserAction(action_type="scroll", scroll_direction="down", scroll_amount=3)
        assert action.scroll_direction == "down"
        assert action.scroll_amount == 3

    def test_scroll_action_with_pixels(self):
        action = BrowserAction(action_type="scroll", scroll_x=0, scroll_y=100, coordinate=[640, 360])
        assert action.scroll_x == 0
        assert action.scroll_y == 100

    def test_drag_action(self):
        action = BrowserAction(action_type="drag", start_coordinate=[10, 20], end_coordinate=[100, 200])
        assert action.start_coordinate == [10, 20]
        assert action.end_coordinate == [100, 200]

    def test_drag_action_with_path(self):
        action = BrowserAction(action_type="drag", path=[[10, 20], [50, 50], [100, 200]])
        assert len(action.path) == 3

    def test_keypress_single(self):
        action = BrowserAction(action_type="keypress", key="Enter")
        assert action.key == "Enter"

    def test_keypress_combo(self):
        action = BrowserAction(action_type="keypress", keys=["Control", "c"])
        assert action.keys == ["Control", "c"]

    def test_goto_action(self):
        action = BrowserAction(action_type="goto", url="https://example.com")
        assert action.url == "https://example.com"

    def test_tab_actions(self):
        action = BrowserAction(action_type="switch_tab", tab_index=2)
        assert action.tab_index == 2

    def test_wait_action(self):
        action = BrowserAction(action_type="wait", duration=2000)
        assert action.duration == 2000

    def test_zoom_action(self):
        action = BrowserAction(action_type="screenshot", region=[0, 0, 500, 500])
        assert action.region is not None
        assert action.region == [0, 0, 500, 500]


class TestSchemas:
    def test_seed_session_request(self):
        req = CUASeedSessionRequest(start_url="http://localhost:3000", viewport_width=1024, viewport_height=768)
        assert req.start_url == "http://localhost:3000"

    def test_step_request(self):
        action = BrowserAction(action_type="click", coordinate=[100, 200])
        req = CUAStepRequest(env_id="test-env", action=action)
        assert req.env_id == "test-env"

    def test_dump_local_storage_request(self):
        req = CUADumpLocalStorageRequest(env_id="test-env")
        assert req.env_id == "test-env"

    def test_close_request(self):
        req = CUACloseRequest(env_id="test-env")
        assert req.env_id == "test-env"

    def test_trajectory(self):
        step = CUAStep(
            action=BrowserAction(action_type="click", coordinate=[100, 200]),
            screenshot_after="base64data",
            current_url="http://example.com",
        )
        traj = CUATrajectory(steps=[step], task_prompt="Do something", initial_screenshot="init_b64")
        assert len(traj.steps) == 1

    def test_cua_nemo_gym_response(self):
        traj = CUATrajectory(steps=[], task_prompt="test", initial_screenshot="")
        resp = CUANeMoGymResponse(
            id="cua_test",
            created_at=1000,
            model="computer-use-preview",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            env_id="env-123",
            trajectory=traj,
            local_storage_dump='{"key": "value"}',
        )
        assert resp.env_id == "env-123"
        assert resp.local_storage_dump == '{"key": "value"}'

    def test_verify_request_mro(self):
        traj = CUATrajectory(steps=[], task_prompt="test", initial_screenshot="")
        response = CUANeMoGymResponse(
            id="cua_test",
            created_at=1000,
            model="test-model",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            env_id="env-123",
            trajectory=traj,
        )
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        rcp = NeMoGymResponseCreateParamsNonStreaming(input="test")
        verify_req = CUAVerifyRequest(responses_create_params=rcp, response=response)
        assert isinstance(verify_req.response, CUANeMoGymResponse)
        assert verify_req.response.env_id == "env-123"


class TestVerifyEndpoint:
    @pytest.mark.asyncio
    async def test_verify_missing_metadata_returns_zero(self):
        with patch("resources_servers.browser_gym.app.ensure_playwright"):
            config = _make_config()
            server = BrowserGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

            traj = CUATrajectory(steps=[], task_prompt="test", initial_screenshot="")
            response = CUANeMoGymResponse(
                id="cua_test",
                created_at=1000,
                model="test",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
                env_id="env-1",
                trajectory=traj,
                local_storage_dump="{}",
            )
            from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

            rcp = NeMoGymResponseCreateParamsNonStreaming(input="test")
            body = CUAVerifyRequest(responses_create_params=rcp, response=response, verifier_metadata=None)
            result = await server.verify(body)
            assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_verify_success(self):
        with patch("resources_servers.browser_gym.app.ensure_playwright"):
            config = _make_config()
            server = BrowserGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

            traj = CUATrajectory(steps=[], task_prompt="test", initial_screenshot="")
            response = CUANeMoGymResponse(
                id="cua_test",
                created_at=1000,
                model="test",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
                env_id="env-1",
                trajectory=traj,
                local_storage_dump='{"key": "value"}',
            )
            from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

            rcp = NeMoGymResponseCreateParamsNonStreaming(input="test")
            body = CUAVerifyRequest(
                responses_create_params=rcp,
                response=response,
                verifier_metadata={"gym_url": "http://localhost:3000", "task_id": "TEST-001"},
            )

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"assertions": [{"result": "pass"}, {"result": "pass"}]})
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            with patch("aiohttp.ClientSession", return_value=mock_session):
                result = await server.verify(body)
                assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_verify_failure(self):
        with patch("resources_servers.browser_gym.app.ensure_playwright"):
            config = _make_config()
            server = BrowserGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

            traj = CUATrajectory(steps=[], task_prompt="test", initial_screenshot="")
            response = CUANeMoGymResponse(
                id="cua_test",
                created_at=1000,
                model="test",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
                env_id="env-1",
                trajectory=traj,
                local_storage_dump='{"key": "value"}',
            )
            from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

            rcp = NeMoGymResponseCreateParamsNonStreaming(input="test")
            body = CUAVerifyRequest(
                responses_create_params=rcp,
                response=response,
                verifier_metadata={"gym_url": "http://localhost:3000", "task_id": "TEST-001"},
            )

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"assertions": [{"result": "pass"}, {"result": "fail"}]})
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            with patch("aiohttp.ClientSession", return_value=mock_session):
                result = await server.verify(body)
                assert result.reward == 0.0


class TestNormalizeKey:
    def test_basic_mappings(self):
        assert _normalize_key("ctrl") == "Control"
        assert _normalize_key("cmd") == "Meta"
        assert _normalize_key("alt") == "Alt"
        assert _normalize_key("shift") == "Shift"
        assert _normalize_key("enter") == "Enter"
        assert _normalize_key("return") == "Enter"
        assert _normalize_key("tab") == "Tab"
        assert _normalize_key("escape") == "Escape"
        assert _normalize_key("esc") == "Escape"

    def test_case_insensitive(self):
        assert _normalize_key("CTRL") == "Control"
        assert _normalize_key("Shift") == "Shift"
        assert _normalize_key("ENTER") == "Enter"
        assert _normalize_key("Tab") == "Tab"

    def test_space_variants(self):
        assert _normalize_key("space") == " "
        assert _normalize_key("spacebar") == " "
        assert _normalize_key("SPACE") == " "

    def test_arrow_keys(self):
        assert _normalize_key("arrowup") == "ArrowUp"
        assert _normalize_key("up") == "ArrowUp"
        assert _normalize_key("arrowdown") == "ArrowDown"
        assert _normalize_key("down") == "ArrowDown"
        assert _normalize_key("arrowleft") == "ArrowLeft"
        assert _normalize_key("left") == "ArrowLeft"
        assert _normalize_key("arrowright") == "ArrowRight"
        assert _normalize_key("right") == "ArrowRight"

    def test_modifier_variants(self):
        assert _normalize_key("control_l") == "Control"
        assert _normalize_key("control_r") == "Control"
        assert _normalize_key("shift_l") == "Shift"
        assert _normalize_key("shift_r") == "Shift"
        assert _normalize_key("alt_l") == "Alt"
        assert _normalize_key("alt_r") == "Alt"
        assert _normalize_key("super_l") == "Meta"
        assert _normalize_key("super_r") == "Meta"

    def test_function_keys(self):
        for i in range(1, 13):
            assert _normalize_key(f"f{i}") == f"F{i}"

    def test_page_keys(self):
        assert _normalize_key("pageup") == "PageUp"
        assert _normalize_key("pagedown") == "PageDown"
        assert _normalize_key("page_up") == "PageUp"
        assert _normalize_key("page_down") == "PageDown"
        assert _normalize_key("prior") == "PageUp"
        assert _normalize_key("next") == "PageDown"

    def test_passthrough_for_unknown(self):
        assert _normalize_key("a") == "a"
        assert _normalize_key("z") == "z"
        assert _normalize_key("1") == "1"
        assert _normalize_key("F13") == "F13"

    def test_delete_variants(self):
        assert _normalize_key("backspace") == "Backspace"
        assert _normalize_key("delete") == "Delete"
        assert _normalize_key("del") == "Delete"

    def test_whitespace_stripping(self):
        assert _normalize_key("  enter  ") == "Enter"
        assert _normalize_key("  ctrl  ") == "Control"


class TestBrowserActionClearBeforeTyping:
    def test_default_none(self):
        action = BrowserAction(action_type="type", text="hello")
        assert action.clear_before_typing is None

    def test_set_true(self):
        action = BrowserAction(action_type="type", text="hello", clear_before_typing=True)
        assert action.clear_before_typing is True

    def test_set_false(self):
        action = BrowserAction(action_type="type", text="hello", clear_before_typing=False)
        assert action.clear_before_typing is False
