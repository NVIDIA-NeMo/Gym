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
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.browser_gym.app import BrowserGymResourcesServer
from resources_servers.browser_gym.browser_pool import BrowserPool, BrowserSession, _normalize_key
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

    def test_pool_size_default(self):
        pool = BrowserPool()
        assert len(pool._slots) == 4

    def test_pool_size_custom(self):
        pool = BrowserPool(pool_size=8)
        assert len(pool._slots) == 8

    def test_pool_size_minimum_one(self):
        pool = BrowserPool(pool_size=0)
        assert len(pool._slots) == 1

    def test_slot_initial_state(self):
        pool = BrowserPool(pool_size=2)
        for slot in pool._slots:
            assert slot.browser is None
            assert slot.playwright is None
            assert slot.session_count == 0
            assert not slot.is_alive()

    def test_get_session_not_found(self):
        pool = BrowserPool()
        with pytest.raises(KeyError, match="No browser session found"):
            pool.get_session("nonexistent")

    @pytest.mark.asyncio
    async def test_close_session_idempotent(self):
        pool = BrowserPool()
        result = await pool.close_session("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_create_session_failure_during_navigation_rolls_back_resources(self):
        pool = BrowserPool(max_concurrent=1, pool_size=1)
        mock_browser = MagicMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_page.goto = AsyncMock(side_effect=RuntimeError("goto failed"))

        with patch.object(pool, "_acquire_slot", new=AsyncMock(return_value=(mock_browser, 0))):
            with pytest.raises(RuntimeError, match="goto failed"):
                await pool.create_session(env_id="env-fail-goto", start_url="https://example.com")

        assert "env-fail-goto" not in pool._sessions
        assert pool._slots[0].session_count == 0
        assert pool._semaphore._value == 1
        mock_page.close.assert_awaited_once()
        mock_context.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_session_failure_after_registration_rolls_back_state(self):
        pool = BrowserPool(max_concurrent=1, pool_size=1)
        mock_browser = MagicMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_page.goto = AsyncMock(return_value=None)
        mock_page.screenshot = AsyncMock(side_effect=RuntimeError("screenshot failed"))

        with patch.object(pool, "_acquire_slot", new=AsyncMock(return_value=(mock_browser, 0))):
            with pytest.raises(RuntimeError, match="screenshot failed"):
                await pool.create_session(env_id="env-fail-screenshot", start_url="https://example.com")

        assert "env-fail-screenshot" not in pool._sessions
        assert pool._slots[0].session_count == 0
        assert pool._semaphore._value == 1
        mock_page.close.assert_awaited_once()
        mock_context.close.assert_awaited_once()


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

            mock_session = MagicMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.closed = False

            with patch.object(server, "_get_verify_session", return_value=mock_session):
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

            mock_session = MagicMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.closed = False

            with patch.object(server, "_get_verify_session", return_value=mock_session):
                result = await server.verify(body)
                assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_get_verify_session_creates_once(self):
        with patch("resources_servers.browser_gym.app.ensure_playwright"):
            config = _make_config()
            server = BrowserGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

            assert server._verify_session is None

            with (
                patch("resources_servers.browser_gym.app.aiohttp.ClientSession") as mock_cls,
                patch("resources_servers.browser_gym.app.aiohttp.TCPConnector"),
            ):
                mock_instance = MagicMock()
                mock_instance.closed = False
                mock_cls.return_value = mock_instance

                session1 = server._get_verify_session()
                session2 = server._get_verify_session()

                mock_cls.assert_called_once()
                assert session1 is session2

    @pytest.mark.asyncio
    async def test_get_verify_session_recreates_if_closed(self):
        with patch("resources_servers.browser_gym.app.ensure_playwright"):
            config = _make_config()
            server = BrowserGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

            with (
                patch("resources_servers.browser_gym.app.aiohttp.ClientSession") as mock_cls,
                patch("resources_servers.browser_gym.app.aiohttp.TCPConnector"),
            ):
                mock_instance = MagicMock()
                mock_instance.closed = False
                mock_cls.return_value = mock_instance

                server._get_verify_session()
                mock_instance.closed = True

                mock_cls.reset_mock()
                mock_new = MagicMock()
                mock_new.closed = False
                mock_cls.return_value = mock_new

                session2 = server._get_verify_session()

                mock_cls.assert_called_once()
                assert session2 is mock_new

    @pytest.mark.asyncio
    async def test_shutdown_closes_verify_session(self):
        with patch("resources_servers.browser_gym.app.ensure_playwright"):
            config = _make_config()
            server = BrowserGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
            app = server.setup_webserver()

            mock_session = AsyncMock()
            mock_session.closed = False
            server._verify_session = mock_session

            mock_pool = AsyncMock()
            mock_pool.start_reaper = MagicMock()
            server.browser_pool = mock_pool

            lifespan = app.router.lifespan_context

            async with lifespan(app):
                pass

            mock_session.close.assert_awaited_once()


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


class TestShutdownHook:
    def test_lifespan_context_is_set(self):
        """Verify that setup_webserver installs a custom lifespan that will call browser_pool.shutdown."""
        with patch("resources_servers.browser_gym.app.ensure_playwright"):
            config = _make_config()
            server = BrowserGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
            app = server.setup_webserver()
            assert app.router.lifespan_context is not None

    @pytest.mark.asyncio
    async def test_shutdown_calls_browser_pool_shutdown(self):
        """Verify that the lifespan teardown actually invokes browser_pool.shutdown()."""
        with patch("resources_servers.browser_gym.app.ensure_playwright"):
            config = _make_config()
            server = BrowserGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
            server.browser_pool = MagicMock(spec=BrowserPool)
            server.browser_pool.shutdown = AsyncMock()
            server.browser_pool.start_reaper = MagicMock()

            app = server.setup_webserver()
            lifespan = app.router.lifespan_context

            async with lifespan(app):
                pass

            server.browser_pool.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_logs_on_error(self):
        """Verify that shutdown is called even if it raises — the error propagates."""
        with patch("resources_servers.browser_gym.app.ensure_playwright"):
            config = _make_config()
            server = BrowserGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
            server.browser_pool = MagicMock(spec=BrowserPool)
            server.browser_pool.shutdown = AsyncMock(side_effect=RuntimeError("browser crash"))
            server.browser_pool.start_reaper = MagicMock()

            app = server.setup_webserver()
            lifespan = app.router.lifespan_context

            with pytest.raises(RuntimeError, match="browser crash"):
                async with lifespan(app):
                    pass

            server.browser_pool.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lifespan_starts_reaper(self):
        """Verify that the lifespan startup calls browser_pool.start_reaper()."""
        with patch("resources_servers.browser_gym.app.ensure_playwright"):
            config = _make_config()
            server = BrowserGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
            server.browser_pool = MagicMock(spec=BrowserPool)
            server.browser_pool.shutdown = AsyncMock()
            server.browser_pool.start_reaper = MagicMock()

            app = server.setup_webserver()
            lifespan = app.router.lifespan_context

            async with lifespan(app):
                server.browser_pool.start_reaper.assert_called_once()


class TestSessionReaper:
    def test_session_has_last_accessed(self):
        session = BrowserSession(
            context=MagicMock(),
            page=MagicMock(),
            viewport_width=1280,
            viewport_height=720,
        )
        assert isinstance(session.last_accessed, float)
        assert session.last_accessed > 0

    def test_get_session_touches_last_accessed(self):
        pool = BrowserPool(session_ttl_seconds=60.0)
        mock_session = BrowserSession(
            context=MagicMock(),
            page=MagicMock(),
            viewport_width=1280,
            viewport_height=720,
        )
        old_ts = mock_session.last_accessed - 100
        mock_session.last_accessed = old_ts
        pool._sessions["test-env"] = mock_session

        retrieved = pool.get_session("test-env")
        assert retrieved.last_accessed > old_ts

    @pytest.mark.asyncio
    async def test_reap_stale_sessions_closes_expired(self):
        pool = BrowserPool(session_ttl_seconds=60.0)

        stale_session = BrowserSession(
            context=MagicMock(),
            page=MagicMock(),
            viewport_width=1280,
            viewport_height=720,
        )
        stale_session.last_accessed = time.monotonic() - 120

        fresh_session = BrowserSession(
            context=MagicMock(),
            page=MagicMock(),
            viewport_width=1280,
            viewport_height=720,
        )

        pool._sessions["stale-env"] = stale_session
        pool._sessions["fresh-env"] = fresh_session

        with patch.object(pool, "close_session", new_callable=AsyncMock) as mock_close:
            await pool._reap_stale_sessions()

            mock_close.assert_called_once_with("stale-env")

    @pytest.mark.asyncio
    async def test_reap_stale_sessions_skips_fresh(self):
        pool = BrowserPool(session_ttl_seconds=60.0)

        fresh_session = BrowserSession(
            context=MagicMock(),
            page=MagicMock(),
            viewport_width=1280,
            viewport_height=720,
        )
        pool._sessions["fresh-env"] = fresh_session

        with patch.object(pool, "close_session", new_callable=AsyncMock) as mock_close:
            await pool._reap_stale_sessions()
            mock_close.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_and_stop_reaper(self):
        pool = BrowserPool(session_ttl_seconds=60.0, reaper_interval_seconds=0.05)
        assert pool._reaper_task is None

        pool.start_reaper()
        assert pool._reaper_task is not None
        assert not pool._reaper_task.done()

        await pool.stop_reaper()
        assert pool._reaper_task is None

    @pytest.mark.asyncio
    async def test_shutdown_stops_reaper(self):
        pool = BrowserPool(session_ttl_seconds=60.0, reaper_interval_seconds=0.05)
        pool.start_reaper()
        assert pool._reaper_task is not None

        await pool.shutdown()
        assert pool._reaper_task is None

    def test_config_ttl_defaults(self):
        pool = BrowserPool()
        assert pool._session_ttl == 7200.0
        assert pool._reaper_interval == 300.0

    def test_config_ttl_custom(self):
        pool = BrowserPool(session_ttl_seconds=3600.0, reaper_interval_seconds=120.0)
        assert pool._session_ttl == 3600.0
        assert pool._reaper_interval == 120.0


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
