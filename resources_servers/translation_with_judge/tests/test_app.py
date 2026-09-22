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
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pytest import approx, fixture

from nemo_gym.config_types import ModelServerRef
from nemo_gym.failure_kinds import JUDGE_UNPARSEABLE
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.translation_with_judge.app import (
    TranslationWithJudgeResourcesServer,
    TranslationWithJudgeResourcesServerConfig,
    TranslationWithJudgeVerifyRequest,
    _strip_reasoning_preamble,
    _tokenizer_for,
)


def _response_with_text(response_id: str, text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id=response_id,
        created_at=1234.5,
        model="response_model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id=f"{response_id}_message",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


class TestStripReasoningPreamble:
    def test_no_tags_returned_unchanged(self) -> None:
        assert _strip_reasoning_preamble("Das Wetter ist heute schön.") == "Das Wetter ist heute schön."

    def test_closed_think_tag_keeps_only_the_answer(self) -> None:
        text = "We need to translate this sentence.</think>\nDas Wetter ist heute schön."
        assert _strip_reasoning_preamble(text) == "Das Wetter ist heute schön."

    def test_unclosed_think_tag_returns_empty(self) -> None:
        assert _strip_reasoning_preamble("<think>still reasoning about the translation") == ""


class TestTokenizerFor:
    def test_default_is_13a(self) -> None:
        assert _tokenizer_for("deu_Latn") == "13a"
        assert _tokenizer_for("fra_Latn") == "13a"
        assert _tokenizer_for("eng_Latn") == "13a"

    def test_cjk_languages(self) -> None:
        assert _tokenizer_for("jpn_Jpan") == "ja-mecab"
        assert _tokenizer_for("kor_Hang") == "ko-mecab"
        assert _tokenizer_for("zho_Hans") == "zh"

    def test_indic_languages_use_flores200(self) -> None:
        for lang in (
            "hin_Deva",
            "ben_Beng",
            "tam_Taml",
            "tel_Telu",
            "mar_Deva",
            "guj_Gujr",
            "kan_Knda",
            "mal_Mlym",
            "pan_Guru",
            "ory_Orya",
            "asm_Beng",
            "urd_Arab",
        ):
            assert _tokenizer_for(lang) == "flores200"

    def test_matches_on_subtag_not_script(self) -> None:
        # Regression: an earlier version keyed off `lang[:2]`, which broke for
        # any FLORES code whose 2-letter prefix isn't its ISO 639-1 code (e.g.
        # "kan_Knda"[:2] == "ka", not the "kn" it needs to match on).
        assert _tokenizer_for("ben_Beng") == "flores200"
        assert _tokenizer_for("kan_Knda") == "flores200"
        assert _tokenizer_for("mal_Mlym") == "flores200"
        assert _tokenizer_for("mar_Deva") == "flores200"


class TestApp:
    @fixture
    def config(self) -> TranslationWithJudgeResourcesServerConfig:
        return TranslationWithJudgeResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="translation_judge_model"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )

    def test_sanity(self, config: TranslationWithJudgeResourcesServerConfig) -> None:
        TranslationWithJudgeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _make_verify_request(self, generation_text: str) -> TranslationWithJudgeVerifyRequest:
        prompt = [{"role": "user", "content": "Translate into German: The weather is lovely today."}]
        return TranslationWithJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=prompt),
            response=_response_with_text("policy_response", generation_text),
            prompt=prompt,
            solution="Das Wetter ist heute schön.",
            src_lang="eng_Latn",
            tgt_lang="deu_Latn",
        )

    async def test_verify_empty_generation_skips_judge_call(
        self, config: TranslationWithJudgeResourcesServerConfig
    ) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server_mock.post = AsyncMock()
        resources_server = TranslationWithJudgeResourcesServer(config=config, server_client=server_mock)

        response = await resources_server.verify(self._make_verify_request("<think>never finished reasoning"))

        assert response.reward == approx(0.0)
        assert response.generation == ""
        assert response.judge_score is None
        assert response.sentence_bleu == approx(0.0)
        assert response.sentence_chrf == approx(0.0)
        assert response.judge_evaluation is None
        assert response.mask_sample is False
        assert response.failure_kind is None
        server_mock.post.assert_not_called()

    async def test_verify_scores_from_judge_response(self, config: TranslationWithJudgeResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        judge_response = _response_with_text("judge_response", "The translation is accurate and fluent.\n\nScore: 85")
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=judge_response.model_dump_json())
        server_mock.post = AsyncMock(return_value=post_mock)
        resources_server = TranslationWithJudgeResourcesServer(config=config, server_client=server_mock)

        response = await resources_server.verify(self._make_verify_request("Das Wetter ist heute schön."))

        assert response.generation == "Das Wetter ist heute schön."
        assert response.judge_score == approx(85.0)
        assert response.reward == approx(0.85)
        assert response.sentence_bleu == approx(100.0)
        assert response.judge_evaluation is not None
        assert response.judge_evaluation.response == judge_response
        assert response.mask_sample is False
        assert response.failure_kind is None

        server_mock.post.assert_awaited_once()
        _, kwargs = server_mock.post.await_args
        assert kwargs["server_name"] == "translation_judge_model"
        assert kwargs["url_path"] == "/v1/responses"

    async def test_verify_unparseable_judge_response_masks_sample(
        self, config: TranslationWithJudgeResourcesServerConfig
    ) -> None:
        server_mock = MagicMock(spec=ServerClient)
        judge_response = _response_with_text("judge_response", "This translation looks fine overall.")
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=judge_response.model_dump_json())
        server_mock.post = AsyncMock(return_value=post_mock)
        resources_server = TranslationWithJudgeResourcesServer(config=config, server_client=server_mock)

        response = await resources_server.verify(self._make_verify_request("Das Wetter ist heute schön."))

        assert response.judge_score is None
        assert response.reward == approx(0.0)
        assert response.mask_sample is True
        assert response.failure_kind == JUDGE_UNPARSEABLE


class TestParseJudgeScore:
    def test_takes_last_score_occurrence(self) -> None:
        text = "A draft score might be Score: 40, but on reflection...\nScore: 92"
        assert TranslationWithJudgeResourcesServer._parse_judge_score(text) == approx(92.0)

    def test_clips_to_0_100(self) -> None:
        assert TranslationWithJudgeResourcesServer._parse_judge_score("Score: 150") == approx(100.0)
        assert TranslationWithJudgeResourcesServer._parse_judge_score("Score: -10") == approx(0.0)

    def test_returns_none_when_absent(self) -> None:
        assert TranslationWithJudgeResourcesServer._parse_judge_score("No score here.") is None


class TestComputeMetrics:
    def test_aggregates_per_pair_and_cross_pair(self) -> None:
        config = TranslationWithJudgeResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="translation_judge_model"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        resources_server = TranslationWithJudgeResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )

        def _row(src: str, tgt: str, judge_score: float, bleu: float, chrf: float) -> dict[str, Any]:
            return {
                "src_lang": src,
                "tgt_lang": tgt,
                "judge_score": judge_score,
                "sentence_bleu": bleu,
                "sentence_chrf": chrf,
            }

        tasks = [
            [_row("eng_Latn", "deu_Latn", 90.0, 60.0, 70.0)],
            [_row("eng_Latn", "fra_Latn", 70.0, 40.0, 50.0)],
        ]

        metrics = resources_server.compute_metrics(tasks)

        assert metrics["eng_Latn->deu_Latn/judge_score"] == approx(90.0)
        assert metrics["eng_Latn->fra_Latn/judge_score"] == approx(70.0)
        assert metrics["xx->xx/judge_score"] == approx(80.0)
        assert metrics["eng_Latn->xx/judge_score"] == approx(80.0)

        key_metrics = resources_server.get_key_metrics(metrics)
        assert key_metrics["xx->xx/judge_score"] == approx(80.0)
        assert key_metrics["eng_Latn->xx/judge_score"] == approx(80.0)

    def test_unparseable_judge_score_counts_as_zero(self) -> None:
        # A None judge_score (unparseable judge response) must count as 0 in the mean, matching
        # verify()'s own reward=0.0 treatment -- dropping it instead silently biases the mean
        # upward. [None, 100.0] must average to 50.0, not 100.0.
        config = TranslationWithJudgeResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="translation_judge_model"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        resources_server = TranslationWithJudgeResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )

        def _row(judge_score: float | None) -> dict[str, Any]:
            return {
                "src_lang": "eng_Latn",
                "tgt_lang": "deu_Latn",
                "judge_score": judge_score,
                "sentence_bleu": 0.0,
                "sentence_chrf": 0.0,
            }

        tasks = [
            [_row(None)],
            [_row(100.0)],
        ]

        metrics = resources_server.compute_metrics(tasks)

        assert metrics["eng_Latn->deu_Latn/judge_score"] == approx(50.0)
        assert metrics["xx->xx/judge_score"] == approx(50.0)
