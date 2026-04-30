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
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from app import (
    WmtTranslationResourcesServer,
    WmtTranslationResourcesServerConfig,
    WmtTranslationVerifyRequest,
    _build_comet_actor_class,
    _strip_reasoning_preamble,
    _tokenizer_for,
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
    compute_comet: bool = False, strip_reasoning: bool = False, comet_num_shards: int = 8
) -> WmtTranslationResourcesServer:
    # Tests default strip_reasoning=False so plain-text generations score
    # against the reference directly. Production default is True (drops the
    # <think>...</think> preamble) for reasoning-model outputs.
    config = WmtTranslationResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        compute_comet=compute_comet,
        strip_reasoning=strip_reasoning,
        comet_num_shards=comet_num_shards,
    )
    return WmtTranslationResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_request(text: str, translation: str, generation: str, target_language: str) -> WmtTranslationVerifyRequest:
    return WmtTranslationVerifyRequest(
        responses_create_params={
            "input": [{"role": "user", "content": f"Translate: {text}"}],
            "parallel_tool_calls": False,
            "temperature": 0,
        },
        response=_make_response(generation),
        text=text,
        translation=translation,
        source_language="en",
        target_language=target_language,
        source_lang_name="English",
        target_lang_name="German",
    )


class TestStripReasoningPreamble:
    def test_takes_text_after_close_tag(self) -> None:
        text = "We need to translate the segment.\n</think>\nProlog"
        assert _strip_reasoning_preamble(text) == "Prolog"

    def test_strips_trailing_newlines_before_answer(self) -> None:
        text = "Reasoning.\n</think>\n\nHallo Welt."
        # Only leading newlines get lstripped; embedded newlines in the answer stay.
        assert _strip_reasoning_preamble(text) == "Hallo Welt."

    def test_uses_last_close_tag(self) -> None:
        # Edge case: model emits </think> inside the reasoning (rare, but defensive).
        text = "Step 1: </think> Step 2 thinking. </think>\nFinal."
        assert _strip_reasoning_preamble(text) == "Final."

    def test_empty_when_truncated_mid_reasoning(self) -> None:
        # <think> opened but never closed → truncated reasoning, count as no-answer.
        assert _strip_reasoning_preamble("<think>unfinished reasoning...") == ""

    def test_returns_text_unchanged_when_no_reasoning_tags(self) -> None:
        # Endpoints that return reasoning as a structured output[i].type="reasoning"
        # block (OpenAI Responses API style) leave output_text clean of <think> /
        # </think>. The text is the answer — don't blank it.
        assert _strip_reasoning_preamble("Hallo Welt.") == "Hallo Welt."

    def test_empty_input(self) -> None:
        assert _strip_reasoning_preamble("") == ""


class TestTokenizer:
    def test_default_tokenizer(self) -> None:
        assert _tokenizer_for("de_DE") == "13a"
        assert _tokenizer_for("fr_FR") == "13a"
        assert _tokenizer_for("en") == "13a"

    def test_japanese_tokenizer(self) -> None:
        assert _tokenizer_for("ja_JP") == "ja-mecab"

    def test_korean_tokenizer(self) -> None:
        assert _tokenizer_for("ko_KR") == "ko-mecab"

    def test_chinese_tokenizer(self) -> None:
        assert _tokenizer_for("zh_CN") == "zh"


class TestVerify:
    async def test_empty_generation_scores_zero(self) -> None:
        server = _make_server()
        request = _make_request(
            text="Hello world.",
            translation="Hallo Welt.",
            generation="",
            target_language="de_DE",
        )
        result = await server.verify(request)
        assert result.reward == 0.0
        assert result.sentence_bleu == 0.0
        assert result.generation == ""

    async def test_perfect_generation_high_reward(self) -> None:
        server = _make_server()
        # Long enough for 4-gram precisions to be non-zero.
        ref = "Der schnelle braune Fuchs springt \u00fcber den faulen Hund."
        request = _make_request(
            text="The quick brown fox jumps over the lazy dog.",
            translation=ref,
            generation=ref,
            target_language="de_DE",
        )
        result = await server.verify(request)
        assert result.sentence_bleu > 50.0
        assert result.reward == result.sentence_bleu / 100.0
        assert result.generation == ref

    async def test_bad_generation_low_reward(self) -> None:
        server = _make_server()
        request = _make_request(
            text="The quick brown fox jumps over the lazy dog.",
            translation="Der schnelle braune Fuchs springt \u00fcber den faulen Hund.",
            generation="Something entirely unrelated in English about cats.",
            target_language="de_DE",
        )
        result = await server.verify(request)
        assert result.reward < 0.1

    async def test_strip_reasoning_recovers_score(self) -> None:
        """With strip_reasoning=True, a reasoning preamble must not
        contaminate the scored generation."""
        server = _make_server(strip_reasoning=True)
        ref = "Der schnelle braune Fuchs springt \u00fcber den faulen Hund."
        reasoning_preamble = (
            "We need to translate to German, without additional explanation. "
            "Output just the translated sentence.\n</think>\n"
        )
        request = _make_request(
            text="The quick brown fox jumps over the lazy dog.",
            translation=ref,
            generation=reasoning_preamble + ref,
            target_language="de_DE",
        )
        result = await server.verify(request)
        # Exactly the reference should land in generation, giving near-perfect BLEU.
        assert result.generation == ref
        assert result.sentence_bleu > 50.0

    async def test_strip_reasoning_empty_when_truncated_mid_reasoning(self) -> None:
        """If <think> opens but never closes, verify() emits no generation."""
        server = _make_server(strip_reasoning=True)
        request = _make_request(
            text="Hello.",
            translation="Hallo.",
            generation="<think>We are still thinking about the answer, no close tag.",
            target_language="de_DE",
        )
        result = await server.verify(request)
        assert result.generation == ""
        assert result.reward == 0.0

    async def test_strip_reasoning_keeps_text_without_reasoning_tags(self) -> None:
        """Clean output_text (no <think>/</think>) passes through strip_reasoning.

        Endpoints returning reasoning as a structured output[i].type='reasoning'
        block (OpenAI Responses API style) leave output_text with just the
        answer — blanking it on strip_reasoning=True zeros rewards everywhere.
        """
        server = _make_server(strip_reasoning=True)
        ref = "Der schnelle braune Fuchs springt über den faulen Hund."
        request = _make_request(
            text="The quick brown fox jumps over the lazy dog.",
            translation=ref,
            generation=ref,  # clean, no <think> tags
            target_language="de_DE",
        )
        result = await server.verify(request)
        assert result.generation == ref
        assert result.sentence_bleu > 50.0


class TestComputeMetrics:
    def test_empty_tasks(self) -> None:
        server = _make_server()
        assert server.compute_metrics([]) == {}

    def test_bleu_per_pair_and_aggregations(self) -> None:
        """Feed two language pairs x two rollouts each; expect per-pair BLEU,
        cross-pair aggregates, and std-dev keys without COMET fields."""
        server = _make_server(compute_comet=False)
        # Sentences long enough that 4-gram matches exist (sacrebleu's
        # corpus_bleu uses the default 4-gram geometric mean, which is 0
        # when any n-gram precision is 0). Two "tasks", 2 rollouts each:
        # rollout 0 is perfect, rollout 1 is a slight variant, so std_dev
        # across runs is non-zero but BLEU > 0 on both.
        de_ref = "Der schnelle braune Fuchs springt \u00fcber den faulen Hund in dem sch\u00f6nen Garten."
        de_perfect = de_ref
        de_variant = "Der schnelle braune Fuchs springt \u00fcber den faulen Hund im sch\u00f6nen Garten."
        fr_ref = "Le renard brun rapide saute par dessus le chien paresseux dans le beau jardin."
        fr_perfect = fr_ref
        fr_variant = "Le renard brun rapide saute au dessus du chien paresseux dans le beau jardin."
        tasks = [
            # Task 1: en->de_DE
            [
                {
                    "text": "The quick brown fox jumps over the lazy dog in the beautiful garden.",
                    "translation": de_ref,
                    "generation": de_perfect,
                    "source_language": "en",
                    "target_language": "de_DE",
                },
                {
                    "text": "The quick brown fox jumps over the lazy dog in the beautiful garden.",
                    "translation": de_ref,
                    "generation": de_variant,
                    "source_language": "en",
                    "target_language": "de_DE",
                },
            ],
            # Task 2: en->fr_FR
            [
                {
                    "text": "The quick brown fox jumps over the lazy dog in the beautiful garden.",
                    "translation": fr_ref,
                    "generation": fr_perfect,
                    "source_language": "en",
                    "target_language": "fr_FR",
                },
                {
                    "text": "The quick brown fox jumps over the lazy dog in the beautiful garden.",
                    "translation": fr_ref,
                    "generation": fr_variant,
                    "source_language": "en",
                    "target_language": "fr_FR",
                },
            ],
        ]
        m = server.compute_metrics(tasks)

        # Per-pair BLEU
        assert "en->de_DE/bleu" in m
        assert "en->fr_FR/bleu" in m
        assert m["en->de_DE/bleu"] > 0
        assert m["en->fr_FR/bleu"] > 0
        # Std-dev keys exist for per-pair
        assert "en->de_DE/bleu_std_dev_across_runs" in m
        assert "en->fr_FR/bleu_std_dev_across_runs" in m

        # Cross-pair aggregations
        assert "xx->xx/bleu" in m
        assert "en->xx/bleu" in m
        assert "xx->de_DE/bleu" in m
        assert "xx->fr_FR/bleu" in m

        # No COMET when disabled
        assert not any(k.endswith("/comet") for k in m)

    def test_comet_disabled_does_not_call_ray(self) -> None:
        """With compute_comet=False, compute_metrics must not call Ray's
        COMET path or add /comet keys even when triples would otherwise exist."""
        server = _make_server(compute_comet=False)
        tasks = [
            [
                {
                    "text": "The quick brown fox jumps over the lazy dog in the beautiful garden.",
                    "translation": "Der schnelle braune Fuchs springt \u00fcber den faulen Hund im sch\u00f6nen Garten.",
                    "generation": "Der schnelle braune Fuchs springt \u00fcber den faulen Hund im sch\u00f6nen Garten.",
                    "source_language": "en",
                    "target_language": "de_DE",
                }
            ]
        ]
        m = server.compute_metrics(tasks)
        # BLEU is emitted; /comet keys are not.
        assert "en->de_DE/bleu" in m
        for k in m:
            assert "/comet" not in k

    def test_get_key_metrics_filters(self) -> None:
        server = _make_server()
        agent = {
            "xx->xx/bleu": 35.0,
            "xx->xx/comet": 78.0,
            "en->xx/bleu": 32.0,
            "en->xx/comet": 77.0,
            "en->de_DE/bleu": 30.0,  # not in key metrics
            "mean/reward": 0.45,  # not in key metrics
        }
        key = server.get_key_metrics(agent)
        assert set(key.keys()) == {"xx->xx/bleu", "xx->xx/comet", "en->xx/bleu", "en->xx/comet"}

    def test_per_row_comet_scores_emit_aggregate_metrics(self) -> None:
        """When rollouts carry comet_score (from verify()'s actor pool await),
        compute_metrics buckets those values directly into per-pair and
        cross-pair aggregates."""
        server = _make_server(compute_comet=True)
        de_ref = "Der schnelle braune Fuchs springt über den faulen Hund im schönen Garten."
        fr_ref = "Le renard brun rapide saute par dessus le chien paresseux dans le beau jardin."
        tasks = [
            [
                {
                    "text": "T1",
                    "translation": de_ref,
                    "generation": de_ref,
                    "source_language": "en",
                    "target_language": "de_DE",
                    "comet_score": 0.85,
                },
            ],
            [
                {
                    "text": "T2",
                    "translation": de_ref,
                    "generation": de_ref,
                    "source_language": "en",
                    "target_language": "de_DE",
                    "comet_score": 0.95,
                },
            ],
            [
                {
                    "text": "T3",
                    "translation": fr_ref,
                    "generation": fr_ref,
                    "source_language": "en",
                    "target_language": "fr_FR",
                    "comet_score": 0.90,
                },
            ],
        ]
        m = server.compute_metrics(tasks)
        # Per-pair: en->de_DE = mean(0.85, 0.95) × 100 = 90.0
        # Per-pair: en->fr_FR = 0.90 × 100 = 90.0
        assert m["en->de_DE/comet"] == pytest.approx(90.0)
        assert m["en->fr_FR/comet"] == pytest.approx(90.0)
        # Cross-pair aggregations.
        assert m["xx->xx/comet"] == pytest.approx(90.0)
        assert m["en->xx/comet"] == pytest.approx(90.0)
        assert m["xx->de_DE/comet"] == pytest.approx(90.0)
        assert m["xx->fr_FR/comet"] == pytest.approx(90.0)

    def test_no_comet_rows_emits_bleu_only(self) -> None:
        """Rollouts without comet_score (compute_comet disabled mid-run, or
        actor pool unavailable) yield BLEU metrics with no /comet keys."""
        server = _make_server(compute_comet=True)
        tasks = [
            [
                {
                    "text": "The quick brown fox jumps over the lazy dog in the beautiful garden.",
                    "translation": "Der schnelle braune Fuchs springt über den faulen Hund im schönen Garten.",
                    "generation": "Der schnelle braune Fuchs springt über den faulen Hund im schönen Garten.",
                    "source_language": "en",
                    "target_language": "de_DE",
                    # No comet_score field → simulates pool unavailable.
                }
            ]
        ]
        m = server.compute_metrics(tasks)
        assert "en->de_DE/bleu" in m
        assert not any(k.endswith("/comet") for k in m)


class TestBuildCometActorClass:
    """Unit tests for _build_comet_actor_class() — the pre-dispatch setup logic.

    The inner @ray.remote actor class requires a live Ray cluster + GPUs +
    unbabel-comet (a gated ~10B checkpoint), so we can't construct the actor
    itself in unit tests. What we *can* cover is the code that builds it:
    venv Python resolution, cross-node Python mirror, env_vars propagation,
    and ray.remote decoration args.
    """

    def _stub_ray_remote(self, captured: dict):
        """Return a ray.remote replacement that captures the decoration kwargs."""

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
        """Build a fake uv-style Python install and return its venv python path."""
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

    def test_propagates_runtime_env_and_pins_py_executable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """_build_comet_actor_class must request ray.remote with an env_vars
        dict that preserves CUDA_VISIBLE_DEVICES, threads site-packages onto
        PYTHONPATH, propagates HF_HOME so actors find the prepared cache,
        and pins py_executable to the cross-node-mirrored Python. HF
        offline/online flags are intentionally NOT overridden — the
        benchmark prepare step pre-fetches the COMET model + tokenizer
        into HF_HOME so runtime is fully offline.
        """
        import app as app_module

        fake_python = self._fake_venv(tmp_path)
        mirror_root = tmp_path / "mirror_cache"

        monkeypatch.setattr(sys, "executable", str(fake_python))
        monkeypatch.setenv("WMT_TRANSLATION_COMET_PY_CACHE", str(mirror_root))
        monkeypatch.setenv("HF_HOME", "/tmp/hf_home")
        monkeypatch.setenv("PYTHONPATH", "/existing/pp")

        captured = {}
        monkeypatch.setattr(app_module, "ray", MagicMock(remote=self._stub_ray_remote(captured)))

        _build_comet_actor_class()

        kw = captured["decorator_kwargs"]
        assert kw["num_gpus"] == 0
        assert kw["resources"] == {"extra_gpu": 1}

        env = kw["runtime_env"]["env_vars"]
        assert env["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "1"
        assert "site-packages" in env["PYTHONPATH"]
        assert "/existing/pp" in env["PYTHONPATH"]
        assert env["HF_HOME"] == "/tmp/hf_home"
        # No HF online/offline overrides — actors inherit from parent.
        assert "HF_HUB_OFFLINE" not in env
        assert "TRANSFORMERS_OFFLINE" not in env
        # No token propagation — runtime is offline post-prepare.
        assert "HF_TOKEN" not in env
        assert "HUGGING_FACE_HUB_TOKEN" not in env

        py_exec = kw["runtime_env"]["py_executable"]
        assert py_exec.startswith(str(mirror_root))
        assert py_exec.endswith("bin/python3.12")
        # Mirror was performed on first invocation.
        assert (mirror_root / "cpython-3.12.12-linux-x86_64-gnu" / "bin" / "python3.12").exists()

    def test_reuses_existing_mirror_without_recopy(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Second invocation must skip copytree when the mirror already exists."""
        import app as app_module

        fake_python = self._fake_venv(tmp_path)
        mirror_root = tmp_path / "mirror_cache"
        (mirror_root / "cpython-3.12.12-linux-x86_64-gnu" / "bin").mkdir(parents=True)
        (mirror_root / "cpython-3.12.12-linux-x86_64-gnu" / "bin" / "python3.12").write_text("")

        monkeypatch.setattr(sys, "executable", str(fake_python))
        monkeypatch.setenv("WMT_TRANSLATION_COMET_PY_CACHE", str(mirror_root))
        monkeypatch.setattr(app_module, "ray", MagicMock(remote=self._stub_ray_remote({})))

        import shutil as shutil_mod

        with patch.object(shutil_mod, "copytree") as mock_copy:
            _build_comet_actor_class()
            mock_copy.assert_not_called()

    def test_cleans_stale_tmp_before_copy(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A leftover .tmp from an interrupted prior mirror must be cleared.

        ``Path("cpython-3.12.12-linux-x86_64-gnu").with_suffix(".tmp")``
        produces ``cpython-3.12.tmp`` (pathlib replaces from the last dot).
        Seed the stale path there to exercise the rmtree branch.
        """
        import app as app_module

        fake_python = self._fake_venv(tmp_path)
        mirror_root = tmp_path / "mirror_cache"
        stale_tmp = mirror_root / "cpython-3.12.tmp"
        stale_tmp.mkdir(parents=True)
        (stale_tmp / "leftover.txt").write_text("from prior run")

        monkeypatch.setattr(sys, "executable", str(fake_python))
        monkeypatch.setenv("WMT_TRANSLATION_COMET_PY_CACHE", str(mirror_root))
        monkeypatch.setattr(app_module, "ray", MagicMock(remote=self._stub_ray_remote({})))

        _build_comet_actor_class()

        # After the run, the .tmp dir is gone (rmtree'd + copytree'd + renamed).
        assert not stale_tmp.exists()
        assert (mirror_root / "cpython-3.12.12-linux-x86_64-gnu" / "bin" / "python3.12").exists()

    def test_raises_if_sys_executable_missing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Defensive: if sys.executable points at a vanished path, fail loudly."""
        import app as app_module

        monkeypatch.setattr(sys, "executable", str(tmp_path / "does_not_exist"))
        monkeypatch.setattr(app_module, "ray", MagicMock(remote=self._stub_ray_remote({})))

        with pytest.raises(RuntimeError, match="sys.executable doesn't exist"):
            _build_comet_actor_class()
