# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseOutputMessage, NeMoGymResponseOutputText
from nemo_gym.server_utils import ServerClient
from resources_servers.capx.app import CapxEnv, CapxResourcesServerConfig, extract_code, parse_decision
from resources_servers.capx.setup_capx import start_api_servers


# A stand-in for the CaP-X package, imported by the real worker.py subprocess. Configs are JSON; the fake env
# executes generated code with a `complete()` helper that marks the task as completed.
_FAKE_CAPX = {
    "capx/__init__.py": "",
    "capx/envs/__init__.py": "",
    "capx/envs/configs/__init__.py": "",
    "capx/envs/configs/loader.py": """
        import json

        class DictLoader:
            @staticmethod
            def load(paths):
                with open(paths[0]) as f:
                    return json.load(f)
    """,
    "capx/envs/configs/instantiate.py": """
        import contextlib
        import io
        import os
        import time
        import traceback

        class FakeEnv:
            def __init__(self, cfg):
                self.cfg = cfg
                self.builds = int(os.environ.get("FAKE_BUILDS", "0")) + 1
                os.environ["FAKE_BUILDS"] = str(self.builds)

            def reset(self, seed=None, options=None):
                self.completed = False
                prompt = f"{self.cfg['prompt']} seed={seed} pid={os.getpid()} builds={self.builds}"
                return {"full_prompt": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                ]}, {}

            def step(self, code):
                if code == "simulator_bug":
                    raise RuntimeError("MuJoCo state became invalid")

                def complete():
                    self.completed = True

                out, err, ok = io.StringIO(), io.StringIO(), True
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    try:
                        exec(code, {"complete": complete, "time": time})
                    except BaseException:
                        ok = False
                        traceback.print_exc()
                reward = 1.0 if self.completed else 0.0
                info = {"sandbox_rc": 0 if ok else 1, "stdout": out.getvalue(), "stderr": err.getvalue(),
                        "task_completed": self.completed}
                return {}, reward, reward == 1.0, False, info

        def instantiate(factory):
            return FakeEnv(factory["cfg"])
    """,
    "capx/envs/trial.py": """
        def _patch_libero_goal(env, obs):
            pass
    """,
}

_MULTI_TURN_PROMPT = "Executed:\n{executed_code}\nstdout:\n{console_stdout}\nstderr:\n{console_stderr}"


@pytest.fixture
def capx_root(tmp_path: Path) -> Path:
    for rel_path, source in _FAKE_CAPX.items():
        path = tmp_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source))
    (tmp_path / "single.json").write_text(json.dumps({"env": {"cfg": {"prompt": "Stack the cubes."}}}))
    (tmp_path / "multi.json").write_text(
        json.dumps({"env": {"cfg": {"prompt": "Stack the cubes.", "multi_turn_prompt": _MULTI_TURN_PROMPT}}})
    )
    return tmp_path


def _make_env(capx_root: Path, **overrides) -> CapxEnv:
    config = CapxResourcesServerConfig(
        host="",
        port=0,
        entrypoint="",
        name="",
        capx_root=str(capx_root),
        capx_python=sys.executable,
        worker_env={"PYTHONPATH": str(capx_root)},
        launch_api_servers=False,
        **overrides,
    )
    return CapxEnv(config=config, server_client=MagicMock(spec=ServerClient))


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


async def _close_all(env: CapxEnv) -> None:
    for session_id in list(env.session_state):
        await env.close_session(session_id)
    for worker in env._idle_workers:
        await worker.kill()


class TestParsing:
    def test_extract_code_strips_python_fence(self):
        assert extract_code("Plan:\n```python\nopen_gripper()\n```\nDone.") == "open_gripper()"

    def test_extract_code_keeps_unfenced_code(self):
        assert extract_code("  open_gripper()\n") == "open_gripper()"

    def test_parse_decision_regenerate(self):
        assert parse_decision("REGENERATE\n```python\nclose_gripper()\n```") == ("regenerate", "close_gripper()")

    def test_parse_decision_defaults_to_finish(self):
        assert parse_decision("The cube is stacked. FINISH") == ("finish", None)


class TestSingleTurn:
    @pytest.mark.asyncio
    async def test_completed_task_scores_one(self, capx_root):
        env = _make_env(capx_root)
        try:
            obs, info = await env.reset({"capx_config": "single.json", "seed": 7}, session_id="s")
            assert obs.startswith("Stack the cubes. seed=7")
            assert info["supports_explicit_close"] is True

            obs, reward, terminated, truncated, info = await env.step(
                _response("```python\nprint('stacking')\ncomplete()\n```"), {}, session_id="s"
            )
            assert (obs, reward, terminated, truncated) == (None, 1.0, True, False)
            assert info["task_completed"] is True
            assert info["sandbox_rc"] == 0
            assert info["ended_by"] == "single_turn"
            assert (info["capx_config"], info["seed"]) == ("single.json", 7)
        finally:
            await _close_all(env)

    @pytest.mark.asyncio
    async def test_crashing_program_scores_zero(self, capx_root):
        env = _make_env(capx_root)
        try:
            await env.reset({"capx_config": "single.json"}, session_id="s")
            _, reward, terminated, _, info = await env.step(_response("raise ValueError('no grasp')"), {}, "s")
            assert (reward, terminated) == (0.0, True)
            assert info["sandbox_rc"] == 1
            assert "no grasp" in info["stderr"]
        finally:
            await _close_all(env)


class TestMultiTurn:
    @pytest.mark.asyncio
    async def test_regenerate_then_finish(self, capx_root):
        env = _make_env(capx_root)
        try:
            await env.reset({"capx_config": "multi.json"}, session_id="s")

            obs, reward, terminated, truncated, _ = await env.step(_response("print('attempt 1')"), {}, "s")
            assert (reward, terminated, truncated) == (0.0, False, False)
            assert "Executed:\nprint('attempt 1')" in obs
            assert "stdout:\nattempt 1" in obs
            assert "'REGENERATE'" in obs and "'FINISH'" in obs

            obs, _, terminated, _, _ = await env.step(_response("REGENERATE\n```python\ncomplete()\n```"), {}, "s")
            assert not terminated
            assert "print('attempt 1')\ncomplete()" in obs

            obs, reward, terminated, truncated, info = await env.step(_response("FINISH"), {}, "s")
            assert (obs, reward, terminated, truncated) == (None, 1.0, True, False)
            assert info["num_code_blocks"] == 2
            assert info["num_regenerations"] == 1
            assert info["ended_by"] == "finish"
        finally:
            await _close_all(env)

    @pytest.mark.asyncio
    async def test_code_block_limit_truncates(self, capx_root):
        env = _make_env(capx_root, max_code_blocks=2)
        try:
            await env.reset({"capx_config": "multi.json"}, session_id="s")
            await env.step(_response("pass"), {}, "s")
            obs, reward, terminated, truncated, info = await env.step(_response("REGENERATE\npass"), {}, "s")
            assert (obs, reward, terminated, truncated) == (None, 0.0, False, True)
            assert info["ended_by"] == "max_code_blocks"
        finally:
            await _close_all(env)


class TestWorkerLifecycle:
    @pytest.mark.asyncio
    async def test_worker_and_simulator_are_reused_across_episodes(self, capx_root):
        env = _make_env(capx_root, max_workers=1)
        try:
            first_obs, _ = await env.reset({"capx_config": "single.json"}, session_id="a")
            await env.step(_response("complete()"), {}, "a")
            await env.close_session("a")
            second_obs, _ = await env.reset({"capx_config": "single.json", "seed": 1}, session_id="b")
            first_pid = first_obs.split("pid=")[1].split()[0]
            assert f"pid={first_pid}" in second_obs
            assert "builds=1" in second_obs
        finally:
            await _close_all(env)

    @pytest.mark.asyncio
    async def test_worker_is_replaced_after_its_episode_budget(self, capx_root):
        env = _make_env(capx_root, max_workers=1, max_episodes_per_worker=2)
        try:
            pids = []
            for session in ("a", "b", "c"):
                obs, _ = await env.reset({"capx_config": "single.json"}, session_id=session)
                pids.append(obs.split("pid=")[1].split()[0])
                await env.step(_response("complete()"), {}, session)
                await env.close_session(session)
            assert pids[0] == pids[1] != pids[2]
        finally:
            await _close_all(env)

    @pytest.mark.asyncio
    async def test_step_timeout_kills_worker_and_frees_slot(self, capx_root):
        env = _make_env(capx_root, max_workers=1, step_timeout_s=1.0)
        try:
            first_obs, _ = await env.reset({"capx_config": "single.json"}, session_id="a")
            obs, reward, terminated, truncated, info = await env.step(_response("time.sleep(30)"), {}, "a")
            assert (obs, reward, terminated, truncated) == (None, 0.0, False, True)
            assert "timed out" in info["error"]
            await env.close_session("a")

            second_obs, _ = await env.reset({"capx_config": "single.json"}, session_id="b")
            assert first_obs.split("pid=")[1] != second_obs.split("pid=")[1]
        finally:
            await _close_all(env)

    @pytest.mark.asyncio
    async def test_reset_failure_is_reported(self, capx_root):
        env = _make_env(capx_root, max_workers=1)
        try:
            with pytest.raises(RuntimeError, match="missing.json"):
                await env.reset({"capx_config": "missing.json"}, session_id="a")
            # The failed reset must not leak the only worker slot.
            obs, _ = await env.reset({"capx_config": "single.json"}, session_id="b")
            assert obs.startswith("Stack the cubes.")
        finally:
            await _close_all(env)

    @pytest.mark.asyncio
    async def test_step_without_reset_truncates(self, capx_root):
        env = _make_env(capx_root)
        _, reward, _, truncated, info = await env.step(_response("complete()"), {}, "unknown")
        assert (reward, truncated) == (0.0, True)
        assert "reset" in info["error"]


class TestFailureMasking:
    @pytest.mark.asyncio
    async def test_lost_worker_masks_the_sample(self, capx_root):
        env = _make_env(capx_root, max_workers=1)
        try:
            await env.reset({"capx_config": "single.json"}, session_id="a")
            _, reward, _, truncated, info = await env.step(_response("import os\nos._exit(3)"), {}, "a")
            assert (reward, truncated) == (0.0, True)
            assert (info["mask_sample"], info["failure_kind"]) == (True, "session_lost")
        finally:
            await _close_all(env)

    @pytest.mark.asyncio
    async def test_simulator_error_masks_the_sample(self, capx_root):
        env = _make_env(capx_root)
        try:
            await env.reset({"capx_config": "single.json"}, session_id="a")
            _, _, _, _, info = await env.step(_response("simulator_bug"), {}, "a")
            assert (info["mask_sample"], info["failure_kind"]) == (True, "capx:simulator_error")
            assert "MuJoCo state became invalid" in info["failure_reason"]
        finally:
            await _close_all(env)

    @pytest.mark.asyncio
    async def test_unreachable_api_server_masks_the_sample(self, capx_root):
        env = _make_env(capx_root)
        unreachable = (
            'raise RuntimeError("Failed to communicate with GraspNet service at http://127.0.0.1:8115: '
            "HTTPConnectionPool(host='127.0.0.1', port=8115): Max retries exceeded\")"
        )
        try:
            await env.reset({"capx_config": "single.json"}, session_id="a")
            _, _, _, _, info = await env.step(_response(unreachable), {}, "a")
            assert (info["mask_sample"], info["failure_kind"]) == (True, "transport_unreachable")
        finally:
            await _close_all(env)

    @pytest.mark.asyncio
    async def test_api_server_rejecting_bad_arguments_is_a_valid_measurement(self, capx_root):
        env = _make_env(capx_root)
        rejected = (
            "raise RuntimeError('Request to http://127.0.0.1:8114/segment failed. Last error: 500 Server Error')"
        )
        try:
            await env.reset({"capx_config": "single.json"}, session_id="a")
            _, reward, terminated, _, info = await env.step(_response(rejected), {}, "a")
            assert (reward, terminated) == (0.0, True)
            assert "mask_sample" not in info
        finally:
            await _close_all(env)

    @pytest.mark.asyncio
    async def test_program_timeout_is_a_measurement(self, capx_root):
        env = _make_env(capx_root, step_timeout_s=1.0)
        try:
            await env.reset({"capx_config": "single.json"}, session_id="a")
            _, reward, _, truncated, info = await env.step(_response("time.sleep(30)"), {}, "a")
            assert (reward, truncated) == (0.0, True)
            assert info["failure_kind"] == "capx:program_timeout"
            assert "mask_sample" not in info
        finally:
            await _close_all(env)


class TestApiServerStartup:
    def test_failed_api_server_fails_fast_with_its_log(self, tmp_path):
        # A launcher that, like CaP-X's, keeps running after a server dies and only logs the failure.
        (tmp_path / ".venv" / "bin").mkdir(parents=True)
        (tmp_path / ".venv" / "bin" / "python").symlink_to(sys.executable)
        (tmp_path / "capx" / "serving").mkdir(parents=True)
        (tmp_path / "capx" / "serving" / "launch_servers.py").write_text(
            "import sys, time\n"
            "log_dir = sys.argv[sys.argv.index('--log-dir') + 1]\n"
            "open(log_dir + '/sam3_8114.log', 'w').write('OSError: gated repo, missing HF_TOKEN\\n')\n"
            "print('[ERROR] launch_servers:   [FAILED] sam3 (port 8114) - process exited with code 1', flush=True)\n"
            "time.sleep(600)\n"
        )
        with pytest.raises(RuntimeError, match=r"(?s)\['sam3'\].*missing HF_TOKEN"):
            start_api_servers(tmp_path, gpus=None, log_dir=tmp_path / "logs", timeout_s=120)


class TestConfig:
    def test_single_gpu_index_from_hydra_is_accepted(self, capx_root):
        assert _make_env(capx_root, api_server_gpus=1).config.api_server_gpus == 1


class TestMetrics:
    def test_key_metrics_report_overall_and_per_tier_success(self):
        def rollout(tier, reward):
            return {"reward": reward, "info": {"tier": tier}}

        tasks = [[rollout("privileged", 1.0)], [rollout("privileged", 0.0)], [rollout("multiturn", 1.0)]]
        metrics = CapxEnv.compute_metrics(None, tasks)
        key = CapxEnv.get_key_metrics(None, metrics)
        assert round(key["pass@1[avg-of-1]/accuracy"], 2) == 66.67
        assert key["pass@1[avg-of-1]/accuracy_privileged"] == 50.0
        assert key["pass@1[avg-of-1]/accuracy_multiturn"] == 100.0

    @pytest.mark.asyncio
    async def test_generated_code_cannot_break_the_protocol(self, capx_root):
        env = _make_env(capx_root, max_workers=1)
        try:
            await env.reset({"capx_config": "single.json"}, session_id="a")
            code = "import sys\nsys.stdin.close()\nsys.stdout.write('noise\\n')\ncomplete()"
            _, reward, terminated, _, _ = await env.step(_response(code), {}, "a")
            assert (reward, terminated) == (1.0, True)
            await env.close_session("a")
            obs, _ = await env.reset({"capx_config": "single.json"}, session_id="b")
            assert obs.startswith("Stack the cubes.")
        finally:
            await _close_all(env)

    @pytest.mark.asyncio
    async def test_worker_exit_truncates_instead_of_raising(self, capx_root):
        env = _make_env(capx_root, max_workers=1)
        try:
            await env.reset({"capx_config": "multi.json"}, session_id="a")
            _, reward, _, truncated, info = await env.step(_response("import os\nos._exit(3)"), {}, "a")
            assert (reward, truncated) == (0.0, True)
            assert "exited" in info["failure_reason"]
            await env.close_session("a")
            obs, _ = await env.reset({"capx_config": "single.json"}, session_id="b")
            assert obs.startswith("Stack the cubes.")
        finally:
            await _close_all(env)
