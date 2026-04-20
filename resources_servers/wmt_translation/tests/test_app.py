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
from unittest.mock import MagicMock

import pytest
from app import (
    WmtTranslationResourcesServer,
    WmtTranslationResourcesServerConfig,
    WmtTranslationVerifyRequest,
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


def _make_server(compute_comet: bool = False) -> WmtTranslationResourcesServer:
    config = WmtTranslationResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        compute_comet=compute_comet,
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

    def test_comet_failure_falls_back_to_bleu_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the Ray COMET path raises, we still get BLEU metrics and no /comet keys."""
        import app as app_module

        def _broken_remote_builder(num_gpus):
            # Return a stub whose .remote() raises, so ray.get never gets called.
            class _Broken:
                def remote(self, *args, **kwargs):
                    raise RuntimeError("simulated ray failure")

            return _Broken()

        monkeypatch.setattr(app_module, "_build_comet_remote", _broken_remote_builder)

        server = _make_server(compute_comet=True)
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
        assert "en->de_DE/bleu" in m
        assert not any(k.endswith("/comet") for k in m)
