# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import logging
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.vlm_eval_kit.app import (
    MCORE_VLMEVALKIT_COMMIT,
    MCORE_VLMEVALKIT_URL,
    UPSTREAM_VLMEVALKIT_URL,
    VlmEvalKitResourcesServer,
    VlmEvalKitResourcesServerConfig,
    VLMEvalKitVerifyRequest,
    build_vlmevalkit_setup_command,
    vlmevalkit_clone_dir,
)


def make_server(**config_overrides) -> VlmEvalKitResourcesServer:
    config = VlmEvalKitResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="", **config_overrides)
    return VlmEvalKitResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def make_response(text: str) -> NeMoGymResponse:
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


class TestApp:
    def test_sanity(self) -> None:
        config = VlmEvalKitResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        VlmEvalKitResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_clone_dirs_are_distinct_per_source(self) -> None:
        # Two sources must not fight over one checkout.
        assert vlmevalkit_clone_dir(UPSTREAM_VLMEVALKIT_URL) == "VLMEvalKit"
        assert vlmevalkit_clone_dir(MCORE_VLMEVALKIT_URL) == "VLMEvalKitMcore"

    def test_setup_command_pins_requested_source(self) -> None:
        cmd = build_vlmevalkit_setup_command(MCORE_VLMEVALKIT_URL, MCORE_VLMEVALKIT_COMMIT, Path("/srv"))
        assert MCORE_VLMEVALKIT_URL in cmd
        assert MCORE_VLMEVALKIT_COMMIT in cmd
        assert "VLMEvalKitMcore" in cmd
        # dual-source contract: never pip-installed (sys.path selection instead)
        assert "pip install" not in cmd


def make_ocrbench_v2_request(
    text: str,
    item_type: str = "cognition VQA en",
    answers=None,
    question: str = "What does the sign say?",
    **extra,
) -> VLMEvalKitVerifyRequest:
    # Field shape mirrors prepare_OCRBench_v2 / the reference OCRBench_v2.evaluate item
    # dict (image_vqa.py:3546-3558): answer/bbox/content literal-parsed, sentinels kept.
    return VLMEvalKitVerifyRequest(
        responses_create_params={
            "input": [{"role": "user", "content": question}],
        },
        response=make_response(text),
        benchmark_name="OCRBench_v2",
        category=item_type,
        answer=answers if answers is not None else ["hello world"],
        index=1,
        question=question,
        bbox="without bbox",
        content="without content",
        **extra,
    )


class TestOCRBenchV2:
    async def test_empty_output_scores_zero(self) -> None:
        # Empty model output must never crash and must score 0.0 (no vlmeval import needed).
        server = make_server()
        result = await server.verify(make_ocrbench_v2_request(""))
        dumped = result.model_dump()
        assert dumped["reward"] == 0.0
        assert dumped["OCRBench_v2"] == 0.0
        assert dumped["OCRBench_v2/cognition VQA en"] == 0.0

    async def test_vqa_containment_hit(self) -> None:
        # Short answers (<5 words) score by case-insensitive containment
        # (vqa_evaluation, Ocrbench_v2/vqa_metric.py).
        pytest.importorskip("vlmeval")
        server = make_server()
        result = await server.verify(make_ocrbench_v2_request("The sign says Hello World."))
        dumped = result.model_dump()
        assert dumped["reward"] == 1.0
        assert dumped["OCRBench_v2"] == 1.0
        assert dumped["OCRBench_v2/cognition VQA en"] == 1.0

    async def test_vqa_containment_miss(self) -> None:
        pytest.importorskip("vlmeval")
        server = make_server()
        result = await server.verify(make_ocrbench_v2_request("The sign says goodbye."))
        assert result.reward == 0.0

    async def test_multiple_choice_eval_exact_letter(self) -> None:
        # eval='multiple choice' strips non-alpha characters and exact-matches
        # (ocrbrnch_v2_eval.py:55-68) — 'B.' scores, a verbose sentence does not.
        pytest.importorskip("vlmeval")
        server = make_server()
        result = await server.verify(make_ocrbench_v2_request("B.", answers=["B"], eval="multiple choice"))
        assert result.reward == 1.0

    async def test_multiple_choice_eval_wrong_letter(self) -> None:
        pytest.importorskip("vlmeval")
        server = make_server()
        result = await server.verify(make_ocrbench_v2_request("A.", answers=["B"], eval="multiple choice"))
        assert result.reward == 0.0

    async def test_kie_f1_partial_is_continuous(self) -> None:
        # KIE scores are the mean per-key F1 (compute_f1_score, TEDS_metric.py:391):
        # one of two keys correct -> reward 0.5. This is the documented continuous-reward
        # deviation: the raw reference score is the reward.
        pytest.importorskip("vlmeval")
        server = make_server()
        result = await server.verify(
            make_ocrbench_v2_request(
                '{"name": "John Smith", "date": "2025"}',
                item_type="key information extraction en",
                answers=[{"name": ["john smith"], "date": ["2024"]}],
            )
        )
        dumped = result.model_dump()
        assert dumped["reward"] == pytest.approx(0.5)
        assert dumped["OCRBench_v2/key information extraction en"] == pytest.approx(0.5)

    async def test_kie_f1_full_match(self) -> None:
        pytest.importorskip("vlmeval")
        server = make_server()
        result = await server.verify(
            make_ocrbench_v2_request(
                '{"name": "John Smith", "date": "2024"}',
                item_type="key information extraction en",
                answers=[{"name": ["john smith"], "date": ["2024"]}],
            )
        )
        assert result.reward == pytest.approx(1.0)

    async def test_think_block_is_stripped(self) -> None:
        pytest.importorskip("vlmeval")
        server = make_server()
        text = "<think>Maybe it says goodbye?</think>\nIt reads: hello world"
        result = await server.verify(make_ocrbench_v2_request(text))
        assert result.reward == 1.0

    async def test_meteor_family_has_wordnet_corpus(self) -> None:
        # The BLEU/METEOR family (full-page OCR / text translation / fine-grained text
        # recognition -> cal_per_metrics -> nltk meteor_score) needs the nltk `wordnet`
        # corpus. Without it every row raises LookupError, which the blanket except in
        # _score_OCRBench_v2 converts to reward 0.0 — a silent wipeout of five task
        # types. setup_VLMEvalKit now installs the corpus; this test forces the wordnet
        # synonym stage with a partially-matching prediction (exact-match rows never
        # consult wordnet, so identical texts would pass even in a broken env) and must
        # score strictly above zero.
        pytest.importorskip("vlmeval")
        server = make_server()
        result = await server.verify(
            make_ocrbench_v2_request(
                "a small feline sat on the mat",
                item_type="full-page OCR en",
                answers=["the small cat sat on the mat"],
            )
        )
        assert result.reward > 0.0

    async def test_spotting_rows_are_serialized(self, monkeypatch) -> None:
        # spotting_evaluation writes FIXED cwd-relative scratch dirs + a shared
        # submit.zip/gt.zip (Ocrbench_v2/spotting_metric.py:123-137), so concurrent
        # calls clobber each other and rows silently score 0. Stand in a deliberately
        # concurrency-detecting fake: any overlapping entry scores 0.0 — with the
        # _SPOTTING_LOCK serialization, every concurrent spotting verify must score 1.0.
        pytest.importorskip("vlmeval")
        import vlmeval.dataset.utils.ocrbrnch_v2_eval as eval_mod

        in_flight = 0
        gate = threading.Lock()

        def racy_process_predictions(items):
            nonlocal in_flight
            with gate:
                in_flight += 1
                overlapped = in_flight > 1
            time.sleep(0.02)  # widen the race window, as the real zip/rmtree dance does
            with gate:
                overlapped = overlapped or in_flight > 1
                in_flight -= 1
            for item in items:
                item["score"] = 0.0 if overlapped else 1.0
            return items

        monkeypatch.setattr(eval_mod, "process_predictions", racy_process_predictions)
        server = make_server()
        requests = [
            make_ocrbench_v2_request(f"[10,10,100,100] word{i}", item_type="text spotting en") for i in range(8)
        ]
        results = await asyncio.gather(*(server.verify(request) for request in requests))
        assert [result.reward for result in results] == [1.0] * 8

    async def test_non_spotting_rows_do_not_take_spotting_lock(self) -> None:
        # Only spotting rows pay the serialization cost: a VQA row must complete even
        # while the spotting lock is held by someone else.
        pytest.importorskip("vlmeval")
        server = make_server()
        async with server._spotting_lock:
            result = await asyncio.wait_for(
                server.verify(make_ocrbench_v2_request("The sign says Hello World.")), timeout=10
            )
        assert result.reward == 1.0

    async def test_scoring_exception_scores_zero_and_logs(self, monkeypatch, caplog) -> None:
        # The blanket except must not be silent: a raising scorer still yields reward 0.0
        # but leaves a warning naming the row, so systematic wipeouts can't hide.
        pytest.importorskip("vlmeval")
        import vlmeval.dataset.utils.ocrbrnch_v2_eval as eval_mod

        def boom(items):
            raise RuntimeError("scratch dir vanished")

        monkeypatch.setattr(eval_mod, "process_predictions", boom)
        server = make_server()
        with caplog.at_level(logging.WARNING):
            result = await server.verify(make_ocrbench_v2_request("hello world"))
        assert result.reward == 0.0
        assert "OCRBench_v2 scoring raised" in caplog.text
        assert "cognition VQA en" in caplog.text

    def test_compute_metrics_buckets_en_zh(self) -> None:
        # EN/ZH bucketing + headline means, mirroring OCRBench_v2.evaluate
        # (image_vqa.py:3560-3565) via the REUSED mcore ocrbench_v2_aggregate_accuracy:
        #   text recognition en -> en_text_recognition (0.5)
        #   cognition VQA en    -> en_visual_text_understanding (1.0)
        #   EN = (0.5 + 1.0) / 2 = 0.75 (unweighted mean over present buckets)
        #   full-page OCR cn    -> cn_text_recognition (0.25) -> ZH = 0.25
        # The reference-ignored translation row must be skipped (otherwise a
        # cn_knowledge_reasoning bucket would appear and drag ZH to 0.125).
        pytest.importorskip("vlmeval")
        server = make_server()
        tasks = [
            [
                {"benchmark_name": "OCRBench_v2", "category": "cognition VQA en", "reward": 1.0},
                {"benchmark_name": "OCRBench_v2", "category": "text recognition en", "reward": 0.5},
            ],
            [
                {"benchmark_name": "OCRBench_v2", "category": "full-page OCR cn", "reward": 0.25},
                {
                    "benchmark_name": "OCRBench_v2",
                    "category": "text translation cn",
                    "reward": 0.0,
                    "ignore": "True",
                },
                # Rows from other benchmarks must be ignored by this aggregate.
                {"benchmark_name": "OtherBenchmark", "answer": "B", "reward": 1.0},
            ],
        ]
        metrics = server.compute_metrics(tasks)
        assert metrics == {
            "OCRBench_v2/en_text_recognition": pytest.approx(0.5),
            "OCRBench_v2/en_visual_text_understanding": pytest.approx(1.0),
            "OCRBench_v2/cn_text_recognition": pytest.approx(0.25),
            "OCRBench_v2_EN": pytest.approx(0.75),
            "OCRBench_v2_ZH": pytest.approx(0.25),
        }

    def test_compute_metrics_en_only(self) -> None:
        # No ZH samples -> no OCRBench_v2_ZH key (calculate_average drops empty buckets).
        pytest.importorskip("vlmeval")
        server = make_server()
        tasks = [[{"benchmark_name": "OCRBench_v2", "category": "math QA en", "reward": 1.0}]]
        metrics = server.compute_metrics(tasks)
        assert metrics == {
            "OCRBench_v2/en_mathematical_calculation": pytest.approx(1.0),
            "OCRBench_v2_EN": pytest.approx(1.0),
        }

    def test_compute_metrics_empty_without_rows(self) -> None:
        server = make_server()
        assert server.compute_metrics([[{"benchmark_name": "OtherBenchmark", "reward": 1.0}]]) == {}

    def test_key_metrics_include_ocrbench_v2(self) -> None:
        server = make_server()
        metrics = server.get_key_metrics(
            {
                "OCRBench_v2_EN": 0.67,
                "OCRBench_v2_ZH": 0.527,
                "mean/OCRBench_v2": 0.6,
                "other": 1.0,
            }
        )
        assert metrics == {"OCRBench_v2_EN": 0.67, "OCRBench_v2_ZH": 0.527}
