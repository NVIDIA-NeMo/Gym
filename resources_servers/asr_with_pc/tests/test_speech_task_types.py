# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify dispatch and aggregation of the OpenASR and Multilingual-ASR task types."""

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse
from resources_servers.asr_with_pc.app import ASRWithPCConfig, ASRWithPCResourcesServer, ASRWithPCVerifyRequest


def _server(task_type: str = "OpenASR", **config: Any) -> ASRWithPCResourcesServer:
    config = ASRWithPCConfig(host="0.0.0.0", port=8080, entrypoint="", name="", task_type=task_type, **config)
    return ASRWithPCResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _response(text: str, *, as_function_call: bool = False) -> NeMoGymResponse:
    if as_function_call:
        output = [
            {
                "type": "function_call",
                "id": "fc",
                "call_id": "c",
                "name": "f",
                "arguments": "{}",
                "status": "completed",
            },
            {
                "id": "msg_1",
                "role": "assistant",
                "type": "message",
                "status": "completed",
                "content": [
                    {"type": "refusal", "refusal": "no"},
                    {"type": "output_text", "text": text, "annotations": []},
                ],
            },
        ]
    else:
        output = [
            {
                "id": "msg_1",
                "role": "assistant",
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ]
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=output,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


async def _verify(task_type: str, reference: str, hypothesis: str, **extra: Any) -> Dict[str, Any]:
    body = ASRWithPCVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "test"}], "parallel_tool_calls": True},
        response=_response(hypothesis),
        expected_answer=reference,
        task_type=task_type,
        **extra,
    )
    return (await _server(task_type).verify(body)).model_dump()


class TestOpenASR:
    async def test_scores_with_compound_merging_and_additive_counts(self) -> None:
        result = await _verify("OpenASR", "Everything is all right, Dr. Smith.", "everything is alright doctor smith")
        assert result["wer"] == 0.0
        assert (result["wer_errors"], result["wer_ref_words"]) == (0, 5)
        assert result["is_correct"] is True and result["reward"] == 1.0
        assert result["text"] == "everything is alright doctor smith"
        assert result["raw_pred_text"] == "everything is alright doctor smith"
        # PC variants do not apply.
        assert (result["wer_c"], result["wer_pc"], result["per"]) == (None, None, None)

    async def test_empty_reference_is_excluded(self) -> None:
        result = await _verify("OpenASR", "", "hello")
        assert (result["wer"], result["is_correct"], result["reward"]) == (None, None, 0.0)
        assert result["wer_ref_words"] is None


class TestMultilingualASR:
    async def test_language_is_required(self) -> None:
        with pytest.raises(ValueError, match="require a language"):
            await _verify("Multilingual-ASR", "hallo", "hallo")

    async def test_cer_uses_characters(self) -> None:
        result = await _verify("Multilingual-ASR", "你好世界", "你好世", language="cmn_hans_cn", use_cer=True)
        assert (result["wer_errors"], result["wer_ref_words"]) == (1, 4)
        assert result["cer"] == pytest.approx(0.25)
        assert result["language"] == "cmn_hans_cn"

    async def test_han_adjacent_whitespace_mode_is_read_from_the_row(self) -> None:
        kwargs = dict(language="cmn_hans_cn", use_cer=True, normalize_compound=False)
        spaced = await _verify("Multilingual-ASR", "你 好 世 界", "你好世界", **kwargs)
        ignored = await _verify(
            "Multilingual-ASR", "你 好 世 界", "你好世界", cer_whitespace_mode="han_adjacent", **kwargs
        )
        assert spaced["wer_errors"] == 3
        assert ignored["wer_errors"] == 0

    async def test_english_defaults_do_not_normalize_compounds(self) -> None:
        # Normalizing compounds would join "any way" and "anyway" into one token and score a match.
        row = dict(language="en_us")
        assert (await _verify("Multilingual-ASR", "any way", "anyway", **row))["wer_errors"] == 2
        assert (await _verify("Multilingual-ASR", "any way", "anyway", normalize_compound=True, **row))[
            "wer_errors"
        ] == 0

    async def test_other_languages_normalize_compounds_by_default(self) -> None:
        row = dict(language="de_de")
        assert (await _verify("Multilingual-ASR", "fuß ball", "fußball", **row))["wer_errors"] == 0
        assert (await _verify("Multilingual-ASR", "fuß ball", "fußball", normalize_compound=False, **row))[
            "wer_errors"
        ] == 2

    async def test_diacritics_are_removed_unless_disabled(self) -> None:
        row = dict(language="de")
        assert (await _verify("Multilingual-ASR", "Fußball", "fussball", **row))["wer_errors"] == 0
        assert (await _verify("Multilingual-ASR", "Fußball", "fussball", remove_diacritics=False, **row))[
            "wer_errors"
        ] == 1

    async def test_merge_compounds_is_forwarded(self) -> None:
        row = dict(language="de", normalize_compound=False)
        assert (await _verify("Multilingual-ASR", "fuß ball", "fußball", **row))["wer_errors"] == 2
        assert (await _verify("Multilingual-ASR", "fuß ball", "fußball", merge_compounds=True, **row))[
            "wer_errors"
        ] == 0

    async def test_empty_reference_is_excluded(self) -> None:
        result = await _verify("Multilingual-ASR", "", "x", language="de")
        assert (result["wer"], result["is_correct"]) == (None, None)


class TestVerifyRequestFields:
    async def test_only_output_text_of_message_items_is_scored(self) -> None:
        body = ASRWithPCVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "x"}], "parallel_tool_calls": True},
            response=_response("hello world", as_function_call=True),
            expected_answer="hello world",
            task_type="OpenASR",
        )
        result = await _server().verify(body)
        assert result.raw_pred_text == "hello world" and result.wer == 0.0

    async def test_guardrail_applies_to_every_task_type(self) -> None:
        looped = "hello world " + "ab" * 40
        body = ASRWithPCVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "x"}], "parallel_tool_calls": True},
            response=_response(looped),
            expected_answer="hello world ab",
            task_type="OpenASR",
        )
        assert (await _server(cleanup_generation_loops=True).verify(body)).raw_pred_text == "hello world ab"
        assert (await _server().verify(body)).raw_pred_text == looped


class TestAggregation:
    @staticmethod
    def _chunk(index: int, hypothesis: str, *, parent: str = "call", reference: str = "good morning everyone") -> dict:
        return {
            "parent_id": parent,
            "chunk_index": index,
            "subset_for_metrics": "earnings",
            "expected_answer": reference,
            "raw_pred_text": hypothesis,
            "pred_text": hypothesis,
        }

    def test_chunks_are_merged_into_one_session_in_chunk_order(self) -> None:
        tasks = [[self._chunk(1, "morning everyone")], [self._chunk(0, "good")]]
        metrics = _server().compute_metrics(tasks)
        assert metrics["corpus_wer@k=1"] == 0.0
        assert metrics["wer::earnings@k=1"] == 0.0
        assert metrics["pass@1[avg-of-1]/accuracy"] == 100.0

    def test_chunk_repeats_are_merged_per_attempt(self) -> None:
        tasks = [
            [self._chunk(0, "good"), self._chunk(0, "bad")],
            [self._chunk(1, "morning everyone"), self._chunk(1, "morning everyone")],
        ]
        metrics = _server().compute_metrics(tasks)
        assert metrics["corpus_wer@k=1"] == 0.0
        assert metrics["corpus_wer@k=2"] == pytest.approx(100.0 / 6)

    def test_chunks_need_consistent_repeats_references_and_ids(self) -> None:
        server = _server()
        with pytest.raises(ValueError, match="share parent_id"):
            server.compute_metrics([[self._chunk(0, "a"), self._chunk(1, "b")]])
        with pytest.raises(ValueError, match="Inconsistent repeat counts"):
            server.compute_metrics([[self._chunk(0, "a"), self._chunk(0, "a")], [self._chunk(1, "b")]])
        with pytest.raises(ValueError, match="Inconsistent parent references"):
            server.compute_metrics([[self._chunk(0, "a")], [self._chunk(1, "b", reference="other")]])

    def test_rows_without_a_parent_are_left_alone(self) -> None:
        row = {"is_correct": True, "pred_text": "hello", "text": "hello", "wer_errors": 0, "wer_ref_words": 1}
        assert _server().compute_metrics([[row]])["corpus_wer@k=1"] == 0.0

    def test_corpus_rates_pool_edit_counts_and_break_down_by_group(self) -> None:
        def row(subset: str, errors: int, words: int, **extra: Any) -> dict:
            return {
                "benchmark_profile": "mixed",
                "subset_for_metrics": subset,
                "language": subset,
                "is_correct": errors == 0,
                "pred_text": "x",
                "wer_errors": errors,
                "wer_ref_words": words,
                **extra,
            }

        tasks = [
            [row("a", 1, 10, cer_errors=2, cer_ref_words=20)],
            [row("b", 3, 30)],
        ]
        metrics = _server().compute_metrics(tasks)
        assert metrics["corpus_wer@k=1"] == pytest.approx(10.0)
        assert metrics["wer::mixed@k=1"] == pytest.approx(10.0)
        assert metrics["wer::a@k=1"] == pytest.approx(10.0)
        assert metrics["cer@k=1"] == pytest.approx(10.0)
        assert metrics["cer::a@k=1"] == pytest.approx(10.0)
        assert "cer::b@k=1" not in metrics
        assert "multilingual_macro_wer@k=1" not in metrics

    def test_key_metrics_expose_the_headline_names(self) -> None:
        agent_metrics = {
            "corpus_wer@k=2": 7.0,
            "cer@k=2": 9.0,
            "multilingual_macro_wer@k=2": 6.0,
            "wer::de@k=2": 3.0,
            "wer::de@k=1": 99.0,
            "bad@k=x": 1.0,
        }
        key = _server().get_key_metrics(agent_metrics)
        assert (key["wer"], key["cer"], key["multilingual_macro_wer"], key["wer::de"]) == (7.0, 9.0, 6.0, 3.0)
