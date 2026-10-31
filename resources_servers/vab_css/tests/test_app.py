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
import ast
import asyncio
import http.server
import json
import shutil
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseOutputMessage, NeMoGymResponseOutputText
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.gymnasium import EnvStepRequest
from resources_servers.vab_css import app as app_module
from resources_servers.vab_css import chromium_host
from resources_servers.vab_css.app import (
    ScreenshotError,
    VabCssConfig,
    VabCssEnv,
    _parse_arguments,
    _request_allowed,
)
from resources_servers.vab_css.css_tools import CSSInteractor, compare_images_ssim_gray, parse_find_all_arguments
from resources_servers.vab_css.prepare_data import build_rows


DATA = Path(__file__).parents[1] / "data"
TASKS = DATA / "example_tasks"


def _make_env(**overrides) -> VabCssEnv:
    config = VabCssConfig(host="", port=0, entrypoint="", name="", install_browser=False, **overrides)
    return VabCssEnv(config=config, server_client=MagicMock(spec=ServerClient))


def _response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="r",
        created_at=0.0,
        model="m",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


@pytest.fixture
def task_copy(tmp_path: Path) -> Path:
    return Path(shutil.copytree(TASKS / "example-cinema", tmp_path / "example-cinema"))


async def _tasks_rendered_here(env: VabCssEnv, root: Path) -> Path:
    """The example tasks with screenshots and distances re-rendered here; the committed PNGs depend on host fonts."""
    tasks = Path(shutil.copytree(TASKS, root / "example_tasks"))
    for task in tasks.iterdir():
        record = ast.literal_eval((task / "corruption" / "record.txt").read_text())
        css = task / "css" / "style.css"
        corrupted = css.read_text()
        prefix = f"{record['selector']} {{ {record['property']}: "
        assert prefix + record["corrupted"] in corrupted
        await env.take_screenshot(task / "index.html", task / "corruption" / "index_corrupted.png")
        css.write_text(corrupted.replace(prefix + record["corrupted"], prefix + record["original"]))
        await env.take_screenshot(task / "index.html", task / "index.png")
        css.write_text(corrupted)
        similarity = compare_images_ssim_gray(task / "index.png", task / "corruption" / "index_corrupted.png")
        (task / "corruption" / "distance.txt").write_text(str(1 - similarity))
    return tasks


class TestCssTools:
    def test_find_all_arguments_accept_literals(self):
        assert parse_find_all_arguments("'div', {'id': 'x'}, class_='card'") == (
            ["div", {"id": "x"}],
            {"class_": "card"},
        )

    @pytest.mark.parametrize(
        "arguments", ["__import__('os').system('true')", "'div', **{'a': 1}", "open('/etc/passwd')"]
    )
    def test_find_all_arguments_reject_code(self, arguments):
        with pytest.raises((ValueError, SyntaxError)):
            parse_find_all_arguments(arguments)

    def test_selectors_by_html_elements(self, task_copy):
        interactor = CSSInteractor(task_copy / "index.html")
        elements, selectors = interactor.get_selectors_by_html_elements("'li'")
        assert len(elements) == 4
        assert selectors == [".menu li"]

    def test_edit_and_revert_write_the_stylesheet(self, task_copy):
        css = task_copy / "css" / "style.css"
        interactor = CSSInteractor(task_copy / "index.html")
        assert interactor.edit_rule("'.card'", "'float'", "'left'")
        assert "float: left" in css.read_text().split(".card")[1]
        assert interactor.revert_last_edit()
        assert "float: none" in css.read_text().split(".card")[1]
        assert not interactor.edit_rule(".missing", "float", "left")

    def test_upstream_argument_parsing(self):
        line = "Action: edit_rule('.menu li', 'float', 'left')"
        assert _parse_arguments("edit_rule", line) == [".menu li", "float", "left"]
        line = "Action: get_selectors_by_html_elements(\"'div', class_='card'\")"
        assert _parse_arguments("get_selectors_by_html_elements", line) == ["'div', class_='card'"]


class TestPrepareData:
    def test_rows_apply_the_upstream_distance_filter(self):
        assert build_rows(TASKS, min_distance=0.2, agent_name="a") == []
        rows = build_rows(TASKS, min_distance=0.0, agent_name="a")
        assert [row["task_id"] for row in rows] == ["example-cinema", "example-menu"]
        assert rows[0]["responses_create_params"]["input"][1]["role"] == "assistant"

    def test_example_jsonl_matches_the_example_tasks(self):
        rows = [json.loads(line) for line in (DATA / "example.jsonl").read_text().splitlines()]
        assert rows == build_rows(TASKS, min_distance=0.0, agent_name="vab_css_gymnasium_agent")


@pytest.mark.usefixtures("chromium_host")
class TestEpisode:
    @pytest.mark.asyncio
    async def test_reset_shows_target_html_and_current_screenshot(self, tmp_path):
        env = _make_env(work_dir=str(tmp_path))
        observation, info = await env.reset({"task_id": "example-cinema"}, session_id="s")
        assert info == {
            "task_id": "example-cinema",
            "supports_step_idempotency": True,
            "supports_explicit_close": True,
        }
        assert [part["type"] for part in observation].count("input_image") == 2
        assert observation[-1]["text"] == "The cards should sit side by side in one row."
        env._cleanup_session("s")
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.asyncio
    async def test_fixing_the_corruption_scores_one(self, tmp_path):
        env = _make_env(work_dir=str(tmp_path))
        env.config.data_dir = str(await _tasks_rendered_here(env, tmp_path))
        await env.reset({"task_id": "example-cinema"}, session_id="s")
        observation, reward, terminated, truncated, _ = await env.step(
            _response("Thought: the cards stack.\nAction: edit_rule('.card', 'float', 'left')"), {}, "s"
        )
        assert (reward, terminated, truncated) == (0.0, False, False)
        assert observation[0]["text"].startswith("You have successfully edited the rule.")
        assert observation[1]["type"] == "input_image"

        observation, reward, terminated, truncated, info = await env.step(
            _response("I have fixed the css style"), {}, "s"
        )
        assert observation is None
        assert (reward, terminated, truncated) == (1.0, True, False)
        assert info["success"] and info["improved"] and info["distance"] < 0.01
        assert info["actions"] == ["edit_rule(.card, float, left)"]

    @pytest.mark.asyncio
    async def test_rounds_run_out_and_the_unfixed_page_scores_zero(self, tmp_path):
        env = _make_env(work_dir=str(tmp_path), max_rounds=2)
        env.config.data_dir = str(await _tasks_rendered_here(env, tmp_path))
        await env.reset({"task_id": "example-menu"}, session_id="s")
        observation, *_ = await env.step(_response("Action: select_rule('.menu li')"), {}, "s")
        assert "float" in observation[0]["text"]
        observation, reward, terminated, truncated, info = await env.step(_response("no action here"), {}, "s")
        assert (observation, reward, terminated, truncated) == (None, 0.0, False, True)
        assert info["distance"] == pytest.approx(info["initial_distance"])
        assert not info["success"]

    @pytest.mark.asyncio
    async def test_unknown_or_unsafe_actions_are_reported_to_the_model(self, tmp_path):
        env = _make_env(work_dir=str(tmp_path))
        await env.reset({"task_id": "example-menu"}, session_id="s")
        observation, *_ = await env.step(_response("Action: finalize()"), {}, "s")
        assert "is not valid due to exception" in observation[0]["text"]
        observation, *_ = await env.step(
            _response("Action: get_selectors_by_html_elements(\"__import__('os').getcwd()\")"), {}, "s"
        )
        assert "is not valid due to exception" in observation[0]["text"]
        observation, *_ = await env.step(_response("Thought: hmm"), {}, "s")
        assert observation.startswith("No executable function found!")

    @pytest.mark.asyncio
    async def test_insert_rule_is_not_an_action(self, tmp_path):
        env = _make_env(work_dir=str(tmp_path))
        await env.reset({"task_id": "example-menu"}, session_id="s")
        css_before = (env.session_state["s"]["task_dir"] / "css" / "style.css").read_text()
        observation, *_ = await env.step(
            _response("Action: insert_rule({'selector': '.menu li', 'properties': {'float': 'left'}})"), {}, "s"
        )
        assert observation[0]["text"].startswith("insert_rule() is not valid due to exception")
        assert (env.session_state["s"]["task_dir"] / "css" / "style.css").read_text() == css_before
        env._cleanup_session("s")


class TestBrowserFailures:
    @pytest.mark.asyncio
    async def test_a_render_failure_fails_the_step_instead_of_blaming_the_model(self, tmp_path, monkeypatch):
        env = _make_env(work_dir=str(tmp_path))
        await env.reset({"task_id": "example-cinema"}, session_id="s")

        async def broken_browser():
            raise RuntimeError("browser crashed")

        monkeypatch.setattr(env, "_get_browser", broken_browser)
        with pytest.raises(ScreenshotError, match="browser crashed"):
            await env.step(_response("Action: edit_rule('.card', 'float', 'left')"), {}, "s")
        env._cleanup_session("s")

    @pytest.mark.asyncio
    async def test_malformed_arguments_still_reach_the_model(self, tmp_path):
        env = _make_env(work_dir=str(tmp_path))
        await env.reset({"task_id": "example-cinema"}, session_id="s")
        observation, *_ = await env.step(_response("Action: edit_rule('.card', 'float)"), {}, "s")
        assert "is not valid due to exception" in observation[0]["text"]
        env._cleanup_session("s")


def _step_request(text: str, request_id: str) -> EnvStepRequest:
    return EnvStepRequest(
        responses_create_params={"input": []}, response=_response(text), **{"_ng_step_request_id": request_id}
    )


class TestStepIdempotency:
    @pytest.mark.asyncio
    async def test_a_resent_step_returns_the_first_result_without_running_again(self, tmp_path, monkeypatch):
        env = _make_env(work_dir=str(tmp_path))
        await env.reset({"task_id": "example-cinema"}, session_id="s")
        state = env.session_state["s"]

        async def fake_screenshot(self, html_path, output_path):
            shutil.copy(html_path.parent / "index.png", output_path)

        monkeypatch.setattr(VabCssEnv, "take_screenshot", fake_screenshot)
        request = MagicMock(session={SESSION_ID_KEY: "s"})
        edit = _step_request("Action: edit_rule('.card', 'float', 'left')", "step-1")
        first = await env._step_endpoint(edit, request)
        again = await env._step_endpoint(edit, request)
        assert again == first
        assert state["rounds"] == 1 and state["actions"] == ["edit_rule(.card, float, left)"]
        assert state["interactor"].last_edit["old_value"] == "none"  # the undo baseline is the original value

    @pytest.mark.asyncio
    async def test_a_step_cut_off_by_a_dropped_connection_finishes_once(self, tmp_path, monkeypatch):
        env = _make_env(work_dir=str(tmp_path))
        await env.reset({"task_id": "example-cinema"}, session_id="s")
        rendering, release = asyncio.Event(), asyncio.Event()

        async def slow_screenshot(self, html_path, output_path):
            rendering.set()
            await release.wait()
            shutil.copy(html_path.parent / "index.png", output_path)

        monkeypatch.setattr(VabCssEnv, "take_screenshot", slow_screenshot)
        request = MagicMock(session={SESSION_ID_KEY: "s"})
        edit = _step_request("Action: edit_rule('.card', 'float', 'left')", "step-1")
        dropped = asyncio.ensure_future(env._step_endpoint(edit, request))
        await rendering.wait()
        dropped.cancel()
        resent = asyncio.ensure_future(env._step_endpoint(edit, request))
        release.set()
        result = await resent
        assert result.observation[0]["text"].startswith("You have successfully edited the rule.")
        assert env.session_state["s"]["rounds"] == 1

    @pytest.mark.asyncio
    async def test_close_during_a_running_step_removes_the_tree_after_the_step(self, tmp_path, monkeypatch):
        env = _make_env(work_dir=str(tmp_path))
        await env.reset({"task_id": "example-cinema"}, session_id="s")
        target = (TASKS / "example-cinema" / "index.png").read_bytes()
        rendering, release = asyncio.Event(), asyncio.Event()

        async def slow_screenshot(self, html_path, output_path):
            rendering.set()
            await release.wait()
            output_path.parent.mkdir(parents=True, exist_ok=True)  # Playwright recreates the folder
            output_path.write_bytes(target)

        monkeypatch.setattr(VabCssEnv, "take_screenshot", slow_screenshot)
        request = MagicMock(session={SESSION_ID_KEY: "s"})
        edit = _step_request("Action: edit_rule('.card', 'float', 'left')", "step-1")
        dropped = asyncio.ensure_future(env._step_endpoint(edit, request))
        await rendering.wait()
        (step,) = env.session_state["s"]["running"]
        dropped.cancel()
        await env._close_endpoint(request)
        assert env.session_state == {}
        release.set()
        await asyncio.wait({step})  # the step has written its screenshot by now
        for _ in range(200):
            if not any(tmp_path.iterdir()):
                break
            await asyncio.sleep(0.01)
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.asyncio
    async def test_the_terminal_response_survives_until_close(self, tmp_path):
        env = _make_env(work_dir=str(tmp_path))
        await env.reset({"task_id": "example-cinema"}, session_id="s")
        env.session_state["s"]["task_dir"].joinpath(app_module.SCREENSHOT_NAME).write_bytes(
            (TASKS / "example-cinema" / "index.png").read_bytes()
        )
        request = MagicMock(session={SESSION_ID_KEY: "s"})
        done = _step_request("I have fixed the css style", "step-1")
        first = await env._step_endpoint(done, request)
        assert (first.reward, first.terminated) == (1.0, True)
        assert await env._step_endpoint(done, request) == first
        with pytest.raises(HTTPException) as late:
            await env._step_endpoint(_step_request("Action: select_rule('.card')", "step-2"), request)
        assert late.value.status_code == 409
        await env._close_endpoint(request)
        with pytest.raises(HTTPException) as closed:
            await env._step_endpoint(done, request)
        assert closed.value.status_code == 404


class TestCleanup:
    @pytest.mark.asyncio
    async def test_close_removes_an_unfinished_episode(self, tmp_path):
        env = _make_env(work_dir=str(tmp_path))
        await env.reset({"task_id": "example-cinema"}, session_id="s")
        request = MagicMock(session={SESSION_ID_KEY: "s"})
        assert await env._close_endpoint(request) == {"ok": True}
        assert await env._close_endpoint(request) == {"ok": True}  # idempotent
        assert env.session_state == {}
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.asyncio
    async def test_close_ends_the_session_for_checkpoints_and_routing(self, tmp_path, monkeypatch):
        ended = MagicMock()
        monkeypatch.setattr(VabCssEnv, "checkpoint_session_ended", ended)
        env = _make_env(work_dir=str(tmp_path))
        await env.reset({"task_id": "example-cinema"}, session_id="s")
        request = MagicMock(session={SESSION_ID_KEY: "s"})
        await env._close_endpoint(request)
        ended.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_cancelled_reset_removes_its_copy_after_the_copy_stops(self, tmp_path, monkeypatch):
        started, release, copied = threading.Event(), threading.Event(), threading.Event()
        copytree = shutil.copytree

        def slow_copytree(src, dst):
            started.set()
            release.wait(10)
            try:
                return copytree(src, dst)
            finally:
                copied.set()

        monkeypatch.setattr(app_module.shutil, "copytree", slow_copytree)
        env = _make_env(work_dir=str(tmp_path))
        reset = asyncio.ensure_future(env.reset({"task_id": "example-cinema"}, session_id="s"))
        await asyncio.to_thread(started.wait, 10)
        reset.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reset
        release.set()
        await asyncio.to_thread(copied.wait, 10)  # the copy thread has written the tree again by now
        for _ in range(200):
            if not any(tmp_path.iterdir()):
                break
            await asyncio.sleep(0.05)
        assert list(tmp_path.iterdir()) == []
        assert env.session_state == {}

    def test_close_route_is_served(self):
        paths = {route.path for route in _make_env().setup_webserver().routes}
        assert {"/reset", "/step", "/close"} <= paths

    @pytest.mark.asyncio
    async def test_repeated_reset_replaces_the_earlier_episode(self, tmp_path):
        env = _make_env(work_dir=str(tmp_path))
        await env.reset({"task_id": "example-cinema"}, session_id="s")
        await env.reset({"task_id": "example-menu"}, session_id="s")
        assert [task.name for root in tmp_path.iterdir() for task in root.iterdir()] == ["example-menu"]
        env._cleanup_session("s")

    @pytest.mark.asyncio
    async def test_failed_reset_leaves_no_copy(self, tmp_path):
        data = tmp_path / "data"
        broken = Path(shutil.copytree(TASKS / "example-cinema", data / "broken"))
        (broken / "corruption" / "distance.txt").unlink()
        work = tmp_path / "work"
        work.mkdir()
        env = _make_env(data_dir=str(data), work_dir=str(work))
        with pytest.raises(FileNotFoundError):
            await env.reset({"task_id": "broken"}, session_id="s")
        assert env.session_state == {}
        assert list(work.iterdir()) == []


class TestPageRequests:
    @pytest.mark.parametrize(
        "url, allowed",
        [
            ("http://127.0.0.1:8000/x.png", False),
            ("http://169.254.169.254/latest/meta-data/", False),
            ("http://10.0.0.1/x.png", False),
            ("https://8.8.8.8/x.png", False),
            ("https://fonts.googleapis.com/css?family=Roboto", False),
            ("ftp://8.8.8.8/x.png", False),
            ("data:image/png;base64,AAAA", True),
        ],
    )
    def test_no_network_request_is_allowed(self, tmp_path, url, allowed):
        assert _request_allowed(url, tmp_path) is allowed

    def test_files_outside_the_task_folder_are_refused(self, tmp_path):
        task = (tmp_path / "task").resolve()
        assert _request_allowed((task / "css" / "style.css").as_uri(), task)
        assert not _request_allowed((tmp_path / "secret.png").as_uri(), task)
        assert not _request_allowed((task / ".." / "secret.png").as_uri(), task)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("chromium_host")
    async def test_page_css_cannot_reach_the_server_host(self, tmp_path):
        hits = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                hits.append(self.path)
                self.send_response(404)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            task = Path(shutil.copytree(TASKS / "example-cinema", tmp_path / "example-cinema"))
            css = task / "css" / "style.css"
            probe = f"http://127.0.0.1:{server.server_port}/probe.png"
            css.write_text(css.read_text() + f"\nbody {{ background-image: url({probe}); }}\n")
            await _make_env().take_screenshot(task / "index.html", tmp_path / "shot.png")
        finally:
            server.shutdown()
        assert (tmp_path / "shot.png").stat().st_size > 0
        assert hits == []


REPORT = "Host system is missing dependencies to run browsers. Missing libraries: libnss3.so"


class TestChromiumHost:
    def test_nothing_is_installed_when_chromium_launches(self, monkeypatch):
        monkeypatch.setattr(chromium_host, "missing_host_dependencies", lambda: None)
        monkeypatch.setattr(chromium_host.subprocess, "run", MagicMock(side_effect=AssertionError("ran a command")))
        chromium_host.ensure_chromium_host_libraries()

    def test_a_non_root_process_is_told_what_to_run(self, monkeypatch):
        monkeypatch.setattr(chromium_host, "missing_host_dependencies", lambda: REPORT)
        monkeypatch.setattr(chromium_host.os, "geteuid", lambda: 1000)
        with pytest.raises(RuntimeError, match="install-deps chromium"):
            chromium_host.ensure_chromium_host_libraries()

    def test_root_installs_the_libraries_and_checks_again(self, monkeypatch):
        answers = iter([REPORT, None])
        monkeypatch.setattr(chromium_host, "missing_host_dependencies", lambda: next(answers))
        monkeypatch.setattr(chromium_host.os, "geteuid", lambda: 0)
        monkeypatch.setattr(chromium_host.shutil, "which", lambda name: f"/usr/bin/{name}")
        run = MagicMock(return_value=MagicMock(returncode=0, stderr=""))
        monkeypatch.setattr(chromium_host.subprocess, "run", run)
        chromium_host.ensure_chromium_host_libraries()
        assert run.call_args.args[0][1:] == ["-m", "playwright", "install-deps", "chromium"]

    def test_a_failed_install_raises(self, monkeypatch):
        monkeypatch.setattr(chromium_host, "missing_host_dependencies", lambda: REPORT)
        monkeypatch.setattr(chromium_host.os, "geteuid", lambda: 0)
        monkeypatch.setattr(chromium_host.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(
            chromium_host.subprocess, "run", MagicMock(return_value=MagicMock(returncode=100, stderr="E: no apt"))
        )
        with pytest.raises(RuntimeError, match="exited 100"):
            chromium_host.ensure_chromium_host_libraries()

    @pytest.mark.parametrize(
        "returncode, stderr, expected",
        [(0, "", None), (1, REPORT, REPORT), (1, "chrome: error while loading shared libraries: libXi.so.6", "libXi")],
    )
    def test_the_launch_probe_reads_playwrights_report(self, monkeypatch, returncode, stderr, expected):
        run = MagicMock(return_value=MagicMock(returncode=returncode, stderr=stderr))
        monkeypatch.setattr(chromium_host.subprocess, "run", run)
        report = chromium_host.missing_host_dependencies()
        assert report is None if expected is None else expected in report

    def test_any_other_launch_failure_raises(self, monkeypatch):
        failure = MagicMock(returncode=1, stderr="TargetClosedError: browser crashed")
        monkeypatch.setattr(chromium_host.subprocess, "run", MagicMock(return_value=failure))
        with pytest.raises(RuntimeError, match="does not launch"):
            chromium_host.missing_host_dependencies()
