# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from app import (
    LongmtEvalConfig,
    LongmtEvalServer,
    LongmtEvalVerifyRequest,
    _assert_no_reasoning,
)
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_test",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_server(
    compute_segale: bool = False,
    assert_no_reasoning: bool = False,
    comet_num_shards: int = 4,
) -> LongmtEvalServer:
    config = LongmtEvalConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        compute_segale=compute_segale,
        assert_no_reasoning=assert_no_reasoning,
        comet_num_shards=comet_num_shards,
    )
    return LongmtEvalServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_request(
    text: str,
    generation: str,
    target_language: str = "de_DE",
    source_language: str = "en",
) -> LongmtEvalVerifyRequest:
    return LongmtEvalVerifyRequest(
        responses_create_params={
            "input": [{"role": "user", "content": f"Translate: {text}"}],
            "parallel_tool_calls": False,
            "temperature": 0,
        },
        response=_make_response(generation),
        text=text,
        source_language=source_language,
        target_language=target_language,
        doc_id="test-doc-1",
    )


@pytest.mark.parametrize("compute_segale", [False, True])
def test_reverify_mode_is_stateless(compute_segale: bool) -> None:
    with patch.object(LongmtEvalServer, "_ensure_actors"):
        server = _make_server(compute_segale=compute_segale)
    with patch.object(LongmtEvalServer, "_ensure_actors") as ensure_actors:
        with TestClient(server.setup_webserver()) as client:
            response = client.get("/reverify_mode")
        assert response.status_code == 200
        assert response.json() == "stateless"
        ensure_actors.assert_not_called()


class TestAssertNoReasoning:
    def test_passes_clean_text(self) -> None:
        _assert_no_reasoning("Die Sonne geht auf.")

    def test_passes_empty(self) -> None:
        _assert_no_reasoning("")

    def test_raises_on_open_tag(self) -> None:
        with pytest.raises(AssertionError, match="reasoning tags"):
            _assert_no_reasoning("<think>still thinking")

    def test_raises_on_close_tag(self) -> None:
        with pytest.raises(AssertionError, match="reasoning tags"):
            _assert_no_reasoning("Reasoning.\n</think>\nDie Sonne")

    def test_raises_on_both_tags(self) -> None:
        with pytest.raises(AssertionError, match="reasoning tags"):
            _assert_no_reasoning("<think>r</think>Die Sonne")


class TestVerify:
    async def test_empty_generation_scores_zero(self) -> None:
        server = _make_server()
        request = _make_request(text="The sun rose.", generation="", target_language="de_DE")
        result = await server.verify(request)
        assert result.reward == 0.0
        assert result.generation == ""

    async def test_non_empty_generation_zero_reward_without_segale(self) -> None:
        server = _make_server(compute_segale=False)
        request = _make_request(
            text="The sun rose over the hills.",
            generation="Die Sonne ging über die Hügel auf.",
            target_language="de_DE",
        )
        result = await server.verify(request)
        # compute_segale=False always returns 0.0 without touching the actor pool.
        assert result.reward == 0.0
        assert result.generation == "Die Sonne ging über die Hügel auf."

    async def test_assert_no_reasoning_raises_on_think_tags(self) -> None:
        server = _make_server(compute_segale=False, assert_no_reasoning=True)
        request = _make_request(
            text="The sun rose over the hills.",
            generation="<think>let me think</think>\nDie Sonne ging über die Hügel auf.",
            target_language="de_DE",
        )
        with pytest.raises(AssertionError, match="reasoning tags"):
            await server.verify(request)

    async def test_assert_no_reasoning_raises_on_unterminated_open_tag(self) -> None:
        server = _make_server(compute_segale=False, assert_no_reasoning=True)
        request = _make_request(
            text="The sun rose.",
            generation="<think>Still thinking, no close tag.",
            target_language="de_DE",
        )
        with pytest.raises(AssertionError, match="reasoning tags"):
            await server.verify(request)

    async def test_assert_no_reasoning_passes_clean_output(self) -> None:
        server = _make_server(compute_segale=False, assert_no_reasoning=True)
        translation = "Die Sonne ging über die Hügel auf."
        request = _make_request(
            text="The sun rose over the hills.",
            generation=translation,
            target_language="de_DE",
        )
        result = await server.verify(request)
        assert result.generation == translation

    async def test_whitespace_only_generation_scores_zero(self) -> None:
        server = _make_server(compute_segale=False)
        request = _make_request(text="The sun rose.", generation="   \n  ", target_language="de_DE")
        result = await server.verify(request)
        assert result.reward == 0.0
        assert result.generation == ""


class TestComputeMetrics:
    def test_empty_tasks(self) -> None:
        server = _make_server()
        assert server.compute_metrics([]) == {}

    def test_skips_rollouts_with_empty_generation(self) -> None:
        server = _make_server()
        tasks = [
            [{"generation": "", "target_language": "de_DE", "comet_qe": 0.9}],
            [{"generation": None, "target_language": "de_DE", "comet_qe": 0.9}],
        ]
        result = server.compute_metrics(tasks)
        assert result == {}

    def test_per_language_comet_aggregation(self) -> None:
        server = _make_server()
        tasks = [
            [
                {
                    "generation": "Die Sonne.",
                    "target_language": "de_DE",
                    "comet_qe": 0.8,
                    "lang_fidelity": 1.0,
                    "total_seg": 2,
                    "misaligned_seg": 0,
                },
                {
                    "generation": "Le soleil.",
                    "target_language": "fr_FR",
                    "comet_qe": 0.9,
                    "lang_fidelity": 1.0,
                    "total_seg": 1,
                    "misaligned_seg": 0,
                },
            ],
            [
                {
                    "generation": "Der Mond.",
                    "target_language": "de_DE",
                    "comet_qe": 0.6,
                    "lang_fidelity": 0.9,
                    "total_seg": 1,
                    "misaligned_seg": 1,
                },
            ],
        ]
        m = server.compute_metrics(tasks)
        # de_DE: mean(0.8, 0.6) = 0.7
        assert m["de_DE"]["comet_qe"] == pytest.approx(0.7)
        assert m["de_DE"]["n_docs"] == 2
        assert m["de_DE"]["total_seg"] == 3
        assert m["de_DE"]["misaligned_seg"] == 1
        assert m["de_DE"]["misaligned_rate"] == pytest.approx(1 / 3)
        # fr_FR: single value
        assert m["fr_FR"]["comet_qe"] == pytest.approx(0.9)
        assert m["fr_FR"]["n_docs"] == 1
        # overall_comet_qe: mean(0.8, 0.9, 0.6) = 0.7667
        assert m["overall_comet_qe"] == pytest.approx((0.8 + 0.9 + 0.6) / 3)

    def test_missing_comet_qe_excluded_from_mean(self) -> None:
        server = _make_server()
        tasks = [
            [
                {"generation": "Die Sonne.", "target_language": "de_DE", "comet_qe": 0.8},
                {"generation": "Der Mond.", "target_language": "de_DE"},  # no comet_qe
            ],
        ]
        m = server.compute_metrics(tasks)
        # Only the row with comet_qe contributes.
        assert m["de_DE"]["comet_qe"] == pytest.approx(0.8)
        assert m["de_DE"]["n_docs"] == 2

    def test_no_comet_qe_in_any_row(self) -> None:
        server = _make_server()
        tasks = [
            [{"generation": "Die Sonne.", "target_language": "de_DE"}],
        ]
        m = server.compute_metrics(tasks)
        assert m["de_DE"]["comet_qe"] is None
        assert "overall_comet_qe" not in m

    def test_lang_fidelity_aggregation(self) -> None:
        server = _make_server()
        tasks = [
            [
                {"generation": "Die Sonne.", "target_language": "de_DE", "lang_fidelity": 0.8},
                {"generation": "Der Mond.", "target_language": "de_DE", "lang_fidelity": 1.0},
            ]
        ]
        m = server.compute_metrics(tasks)
        assert m["de_DE"]["lang_fidelity"] == pytest.approx(0.9)

    def test_misaligned_rate_zero_when_no_segments(self) -> None:
        server = _make_server()
        tasks = [[{"generation": "Die Sonne.", "target_language": "de_DE"}]]
        m = server.compute_metrics(tasks)
        assert m["de_DE"]["misaligned_rate"] is None

    def test_get_key_metrics_returns_per_language_comet(self) -> None:
        server = _make_server()
        metrics = {
            "de_DE": {"comet_qe": 0.82, "n_docs": 5},
            "fr_FR": {"comet_qe": None, "n_docs": 3},
            "ja_JP": {"comet_qe": 0.75, "n_docs": 2},
            "overall_comet_qe": 0.79,
        }
        key = server.get_key_metrics(metrics)
        assert key == {"de_DE": 0.82, "ja_JP": 0.75}

    def test_get_key_metrics_empty(self) -> None:
        server = _make_server()
        assert server.get_key_metrics({}) == {}


class TestBuildSegaleActorClass:
    """Unit tests for _build_segale_actor_class() in segale_actor.py.

    The inner @ray.remote class requires a live Ray cluster + GPUs, so we mock
    ray.remote to capture decoration kwargs and verify the setup logic.
    """

    @pytest.fixture(autouse=True)
    def scoring_runtime(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import setup_scoring

        monkeypatch.setattr(setup_scoring, "ensure_scoring_python", lambda _: tmp_path / "scoring-python")

    def _stub_ray_remote(self, captured: dict):
        def _ray_remote(**decorator_kwargs):
            captured["decorator_kwargs"] = decorator_kwargs

            def _decorate(cls_or_fn):
                class _Decorated:
                    _wrapped = cls_or_fn

                    @staticmethod
                    def remote(*args, **kwargs):
                        raise AssertionError("actor must not instantiate in unit tests")

                return _Decorated

            return _decorate

        return _ray_remote

    def _fake_venv(self, tmp_path: Path) -> Path:
        uv_root = tmp_path / "uv" / "cpython-3.12.12-linux-x86_64-gnu"
        venv_bin = tmp_path / "venv" / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (uv_root / "bin").mkdir(parents=True)
        real_python = uv_root / "bin" / "python3.12"
        real_python.write_text("")
        fake_python = venv_bin / "python3.12"
        fake_python.symlink_to(real_python)
        (venv_bin.parent / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
        return fake_python

    def test_fractional_gpu_mode_by_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """use_extra_gpu=False: actors claim fractional num_gpus, no extra_gpu resource."""
        import segale_actor as sa_module

        fake_python = self._fake_venv(tmp_path)
        mirror_root = tmp_path / "mirror_cache"

        monkeypatch.setattr(sys, "executable", str(fake_python))
        monkeypatch.setenv("LONGMT_EVAL_PY_CACHE", str(mirror_root))
        monkeypatch.setenv("HF_HOME", "/tmp/hf_home")
        monkeypatch.setenv("PYTHONPATH", "/existing/pp")

        captured = {}
        monkeypatch.setattr(sa_module, "ray", MagicMock(remote=self._stub_ray_remote(captured)))

        from segale_actor import _build_segale_actor_class

        _build_segale_actor_class(actors_per_gpu=2, use_extra_gpu=False)

        kw = captured["decorator_kwargs"]
        assert kw["num_gpus"] == pytest.approx(0.5)  # 1 / actors_per_gpu
        assert "resources" not in kw

    def test_extra_gpu_mode(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """use_extra_gpu=True: actors claim extra_gpu resource with num_gpus=0."""
        import segale_actor as sa_module

        fake_python = self._fake_venv(tmp_path)
        mirror_root = tmp_path / "mirror_cache"

        monkeypatch.setattr(sys, "executable", str(fake_python))
        monkeypatch.setenv("LONGMT_EVAL_PY_CACHE", str(mirror_root))

        captured = {}
        monkeypatch.setattr(sa_module, "ray", MagicMock(remote=self._stub_ray_remote(captured)))

        from segale_actor import _build_segale_actor_class

        _build_segale_actor_class(actors_per_gpu=4, use_extra_gpu=True)

        kw = captured["decorator_kwargs"]
        assert kw["num_gpus"] == 0
        assert kw["resources"] == {"extra_gpu": pytest.approx(0.25)}  # 1/4

    def test_propagates_env_vars_and_pins_py_executable(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """runtime_env must propagate HF_HOME, LASER_HOME, site-packages on PYTHONPATH,
        and pin py_executable to the cross-node-mirrored Python."""
        import segale_actor as sa_module

        fake_python = self._fake_venv(tmp_path)
        mirror_root = tmp_path / "mirror_cache"

        monkeypatch.setattr(sys, "executable", str(fake_python))
        monkeypatch.setenv("LONGMT_EVAL_PY_CACHE", str(mirror_root))
        monkeypatch.setenv("HF_HOME", "/tmp/hf_home")
        monkeypatch.setenv("LASER_HOME", "/tmp/laser_home")
        monkeypatch.setenv("PYTHONPATH", "/existing/pp")

        captured = {}
        monkeypatch.setattr(sa_module, "ray", MagicMock(remote=self._stub_ray_remote(captured)))

        from segale_actor import _build_segale_actor_class

        _build_segale_actor_class()

        env = captured["decorator_kwargs"]["runtime_env"]["env_vars"]
        assert "site-packages" in env["PYTHONPATH"]
        assert "/existing/pp" in env["PYTHONPATH"]
        assert env["HF_HOME"] == "/tmp/hf_home"
        assert env["LASER_HOME"] == "/tmp/laser_home"

        py_exec = captured["decorator_kwargs"]["runtime_env"]["py_executable"]
        assert py_exec.startswith(str(mirror_root))
        assert py_exec.endswith("bin/python3.12")
        assert (mirror_root / "cpython-3.12.12-linux-x86_64-gnu" / "bin" / "python3.12").exists()

    def test_default_paths_use_configured_gym_cache_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import segale_actor as sa_module

        fake_python = self._fake_venv(tmp_path)
        cache_dir = tmp_path / "gym_cache"

        monkeypatch.setattr(sys, "executable", str(fake_python))
        monkeypatch.setattr(sa_module, "maybe_get_global_config_dict", lambda: {"cache_dir": str(cache_dir)})
        monkeypatch.delenv("LONGMT_EVAL_PY_CACHE", raising=False)
        monkeypatch.delenv("LASER_HOME", raising=False)
        monkeypatch.delenv("ERSATZ", raising=False)
        monkeypatch.delenv("LONGMT_COMET_CACHE", raising=False)

        captured = {}
        monkeypatch.setattr(sa_module, "ray", MagicMock(remote=self._stub_ray_remote(captured)))

        sa_module._build_segale_actor_class()

        runtime_env = captured["decorator_kwargs"]["runtime_env"]
        assert runtime_env["py_executable"].startswith(str(cache_dir / "longmt-python"))
        assert runtime_env["env_vars"]["LASER_HOME"] == str(cache_dir / "longmt-laser")
        assert runtime_env["env_vars"]["ERSATZ"] == str(cache_dir / "longmt-ersatz")
        assert runtime_env["env_vars"]["LONGMT_COMET_CACHE"] == str(cache_dir / "longmt-comet")

    def test_extra_gpu_sets_noset_cuda_env_var(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """use_extra_gpu=True must set RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES."""
        import segale_actor as sa_module

        fake_python = self._fake_venv(tmp_path)
        monkeypatch.setattr(sys, "executable", str(fake_python))
        monkeypatch.setenv("LONGMT_EVAL_PY_CACHE", str(tmp_path / "mirror"))

        captured = {}
        monkeypatch.setattr(sa_module, "ray", MagicMock(remote=self._stub_ray_remote(captured)))

        from segale_actor import _build_segale_actor_class

        _build_segale_actor_class(use_extra_gpu=True)

        env = captured["decorator_kwargs"]["runtime_env"]["env_vars"]
        assert env["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "1"

    def test_fractional_gpu_does_not_set_noset_cuda(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """use_extra_gpu=False must NOT set RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES."""
        import segale_actor as sa_module

        fake_python = self._fake_venv(tmp_path)
        monkeypatch.setattr(sys, "executable", str(fake_python))
        monkeypatch.setenv("LONGMT_EVAL_PY_CACHE", str(tmp_path / "mirror"))

        captured = {}
        monkeypatch.setattr(sa_module, "ray", MagicMock(remote=self._stub_ray_remote(captured)))

        from segale_actor import _build_segale_actor_class

        _build_segale_actor_class(use_extra_gpu=False)

        env = captured["decorator_kwargs"]["runtime_env"]["env_vars"]
        assert "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES" not in env

    def test_reuses_existing_mirror_without_recopy(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Second call skips copytree when the mirror already exists."""
        import segale_actor as sa_module

        fake_python = self._fake_venv(tmp_path)
        mirror_root = tmp_path / "mirror_cache"
        (mirror_root / "cpython-3.12.12-linux-x86_64-gnu" / "bin").mkdir(parents=True)
        (mirror_root / "cpython-3.12.12-linux-x86_64-gnu" / "bin" / "python3.12").write_text("")

        monkeypatch.setattr(sys, "executable", str(fake_python))
        monkeypatch.setenv("LONGMT_EVAL_PY_CACHE", str(mirror_root))
        monkeypatch.setattr(sa_module, "ray", MagicMock(remote=self._stub_ray_remote({})))

        import shutil as shutil_mod

        from segale_actor import _build_segale_actor_class

        with patch.object(shutil_mod, "copytree") as mock_copy:
            _build_segale_actor_class()
            mock_copy.assert_not_called()

    def test_cleans_stale_tmp_before_copy(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A leftover .tmp from an interrupted mirror run must be cleared first."""
        import segale_actor as sa_module

        fake_python = self._fake_venv(tmp_path)
        mirror_root = tmp_path / "mirror_cache"
        stale_tmp = mirror_root / "cpython-3.12.tmp"
        stale_tmp.mkdir(parents=True)
        (stale_tmp / "leftover.txt").write_text("from prior run")

        monkeypatch.setattr(sys, "executable", str(fake_python))
        monkeypatch.setenv("LONGMT_EVAL_PY_CACHE", str(mirror_root))
        monkeypatch.setattr(sa_module, "ray", MagicMock(remote=self._stub_ray_remote({})))

        from segale_actor import _build_segale_actor_class

        _build_segale_actor_class()

        assert not stale_tmp.exists()
        assert (mirror_root / "cpython-3.12.12-linux-x86_64-gnu" / "bin" / "python3.12").exists()

    def test_raises_if_sys_executable_missing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """If sys.executable points at a nonexistent path, raise RuntimeError immediately."""
        import segale_actor as sa_module

        monkeypatch.setattr(sys, "executable", str(tmp_path / "does_not_exist"))
        monkeypatch.setattr(sa_module, "ray", MagicMock(remote=self._stub_ray_remote({})))

        from segale_actor import _build_segale_actor_class

        with pytest.raises(RuntimeError, match="not found"):
            _build_segale_actor_class()


class TestActorStartup:
    def test_provisions_before_serving(self) -> None:
        with patch.object(LongmtEvalServer, "_ensure_actors") as initialize:
            _make_server(compute_segale=True)
            initialize.assert_called_once()
        with patch.object(LongmtEvalServer, "_ensure_actors") as initialize:
            _make_server(compute_segale=False)
            initialize.assert_not_called()

    def test_startup_failure_prevents_server_creation(self) -> None:
        with patch.object(LongmtEvalServer, "_ensure_actors", side_effect=RuntimeError("setup failed")):
            with pytest.raises(RuntimeError, match="setup failed"):
                _make_server(compute_segale=True)

    def test_failed_startup_releases_actors_without_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import app
        import segale_actor

        actor = MagicMock()
        ping = object()
        actor.ping.remote.return_value = ping
        factory = MagicMock()
        factory.remote.return_value = actor
        build = MagicMock(return_value=factory)
        monkeypatch.setattr(segale_actor, "_build_segale_actor_class", build)
        monkeypatch.setattr(app.ray, "wait", lambda *a, **kw: ([], [ping]))
        killed = []
        monkeypatch.setattr(app.ray, "kill", killed.append)
        server = _make_server(comet_num_shards=1)
        server.config.actors_per_gpu = 1
        with pytest.raises(RuntimeError, match="0/1"):
            server._ensure_actors()
        assert killed == [actor]
        with pytest.raises(RuntimeError, match="restart the server"):
            server._ensure_actors()
        build.assert_called_once()

    def test_partial_creation_failure_releases_existing_actors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import app
        import segale_actor

        actor = MagicMock()
        factory = MagicMock()
        factory.remote.side_effect = [actor, RuntimeError("creation failed")]
        monkeypatch.setattr(segale_actor, "_build_segale_actor_class", lambda **kw: factory)
        killed = []
        monkeypatch.setattr(app.ray, "kill", killed.append)
        with pytest.raises(RuntimeError, match="creation failed"):
            _make_server(compute_segale=True)
        assert killed == [actor]


@pytest.mark.parametrize("failure", ["unavailable", "exception", "error_result"])
async def test_scoring_failures_are_masked_without_reprovisioning(failure: str) -> None:
    server = _make_server()
    server.config.compute_segale = True

    async def score() -> dict:
        if failure == "exception":
            raise TimeoutError("worker timeout")
        return {"error": "non-finite score"}

    with patch.object(LongmtEvalServer, "_ensure_actors") as initialize:
        with patch.object(
            LongmtEvalServer, "_dispatch", side_effect=lambda *a: None if failure == "unavailable" else score()
        ):
            result = await server.verify(_make_request("hello", "bonjour"))
    initialize.assert_not_called()
    assert result.mask_sample is True
    assert result.evaluation_completed is False
    assert result.failure_kind == "verifier_error"
    assert result.segale_error == result.failure_reason
    assert server.compute_metrics([[result.model_dump()]]) == {}


async def test_successful_scoring_remains_unmasked() -> None:
    server = _make_server()
    server.config.compute_segale = True

    async def score() -> dict:
        return {"comet_qe": 0.8, "total_seg": 2, "misaligned_seg": 0}

    with patch.object(LongmtEvalServer, "_dispatch", side_effect=lambda *args: score()):
        result = await server.verify(_make_request("hello", "bonjour"))
    assert result.reward == 0.8
    assert result.evaluation_completed is True
    assert result.mask_sample is False
    assert result.failure_kind is None
    assert result.segale_error is None


@pytest.mark.parametrize("partial", [False, True])
def test_actor_pool_reuse_and_dispatch_order(monkeypatch: pytest.MonkeyPatch, partial: bool) -> None:
    import app
    import segale_actor

    actors = [MagicMock(name=f"actor-{i}") for i in range(4)]
    pings = [actor.ping.remote.return_value for actor in actors]
    factory = MagicMock()
    factory.remote.side_effect = actors
    build = MagicMock(return_value=factory)
    monkeypatch.setattr(segale_actor, "_build_segale_actor_class", build)
    monkeypatch.setattr(app.ray, "wait", lambda *a, **kw: (pings[:3] if partial else pings, []))

    def get(ping: object) -> None:
        if partial and ping is pings[1]:
            raise RuntimeError("actor initialization failed")

    monkeypatch.setattr(app.ray, "get", get)
    killed = []
    monkeypatch.setattr(app.ray, "kill", killed.append)
    server = _make_server(comet_num_shards=2)
    server.config.actors_per_gpu = 2
    server.config.use_extra_gpu = False
    server._ensure_actors()
    server._ensure_actors()
    build.assert_called_once()
    assert factory.remote.call_count == 4
    expected = [actors[0], actors[2]] if partial else [actors[0], actors[2], actors[1], actors[3]]
    assert killed == ([actors[1], actors[3]] if partial else [])
    for actor in expected * 2:
        assert server._dispatch("source", "translation", "fr_FR") is actor.score.remote.return_value
