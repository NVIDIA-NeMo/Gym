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

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


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
