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
"""Tests for the gdp_pdf resources server and its data preparation helpers."""

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import approx, fixture, raises

from benchmarks.gdp_pdf.prepare import (
    EXAMPLE_PDF_SIZE_WARN_BYTES,
    _prepare_rows,
    _resolve_hf_token,
    build_gym_row,
    extract_criteria,
    fetch_example_media,
    write_example,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionMessage,
    NeMoGymChoice,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.gdp_pdf.app import (
    GdpPdfConfig,
    GdpPdfResourcesServer,
    GdpPdfVerifyRequest,
    ParseError,
    _criterion_passes,
    _extract_generated_answer,
    _parse_criterion_result,
)


SERVER_DIR = Path(__file__).resolve().parents[1]
JUDGE_PROMPT_FPATH = str(SERVER_DIR / "prompt_templates/judge.txt")
DATA_DIR = SERVER_DIR / "data"

PROMPT = "According to the attached filing, what is the effective indemnity cap?"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@fixture
def config() -> GdpPdfConfig:
    return GdpPdfConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_chat_create_params=NeMoGymChatCompletionCreateParamsNonStreaming(messages=[]),
        judge_prompt_template_fpath=JUDGE_PROMPT_FPATH,
    )


@fixture
def server(config: GdpPdfConfig) -> GdpPdfResourcesServer:
    return GdpPdfResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _assistant_msg(text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id="msg_id",
        content=[NeMoGymResponseOutputText(annotations=[], text=text)],
        role="assistant",
        status="completed",
        type="message",
    )


def _response(output: list, response_id: str = "resp") -> NeMoGymResponse:
    return NeMoGymResponse(
        id=response_id,
        created_at=0.0,
        model="test",
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _model_response(text: str) -> NeMoGymResponse:
    return _response([_assistant_msg(text)], response_id="model_resp")


def _chat_completion(content: str, completion_id: str = "judge_resp") -> NeMoGymChatCompletion:
    """A Chat Completions envelope carrying ``content`` as the assistant message."""
    return NeMoGymChatCompletion(
        id=completion_id,
        created=0,
        model="test-judge",
        object="chat.completion",
        choices=[
            NeMoGymChoice(
                index=0,
                finish_reason="stop",
                message=NeMoGymChatCompletionMessage(role="assistant", content=content),
            )
        ],
    )


def _judge_payload(content: str) -> str:
    """Serialised NeMoGymChatCompletion with ``content`` as the message body."""
    return _chat_completion(content).model_dump_json()


def _scored_content(score: str, rationale: str = "because") -> str:
    """The judge's message content: the JSON object our own parser decodes."""
    return json.dumps({"score": score, "rationale": rationale})


def _criterion(index: int, text: str) -> dict:
    return {
        "index": index,
        "criterion": text,
        "type": "Primary Intent",
        "severity": "Certain dealbreaker",
        "implicitness": "Explicit",
        "subjectiveness": "Objective",
        "failure_mode": "Binary",
    }


def _multimodal_params(prompt: str = PROMPT) -> NeMoGymResponseCreateParamsNonStreaming:
    """Params as the agent rewrites them: prompt text, document text, then page images."""
    return NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_text", "text": "<document>\nfull filing text\n</document>"},
                    {"type": "input_image", "image_url": "data:image/png;base64,page1", "detail": "high"},
                ],
            }
        ]
    )


def _verify_request(
    answer: str,
    criteria: Optional[list] = None,
    domain: str = "Legal",
) -> GdpPdfVerifyRequest:
    meta: dict[str, Any] = {
        "task_id": "task-1",
        "domain": domain,
        "prompt": PROMPT,
        "document_manifest": "documents/task-1/manifest.json",
        "criteria": [_criterion(1, "States the cap is $5M.")] if criteria is None else criteria,
    }
    return GdpPdfVerifyRequest(
        responses_create_params=_multimodal_params(),
        response=_model_response(answer),
        verifier_metadata=meta,
    )


def _mock_judge(server: GdpPdfResourcesServer, *verdicts: str) -> AsyncMock:
    """Return the given verdicts in order, cycling on the last one."""
    payloads = [_judge_payload(v) for v in verdicts]

    async def _post(*args: Any, **kwargs: Any) -> MagicMock:
        payload = payloads[min(_post.calls, len(payloads) - 1)]
        _post.calls += 1
        mock_resp = MagicMock()
        mock_resp.read = AsyncMock(return_value=payload)
        return mock_resp

    _post.calls = 0
    mock = AsyncMock(side_effect=_post)
    server.server_client.post = mock
    return mock


# ---------------------------------------------------------------------------
# Rubric flattening
# ---------------------------------------------------------------------------


class TestExtractCriteria:
    def test_collapses_populated_slots_and_drops_blanks(self) -> None:
        row = {
            "rubric - 1. criterion": "First criterion.",
            "rubric - 1. criterion_type": "Primary Intent",
            "rubric - 1. criterion_severity": "Certain dealbreaker",
            "rubric - 1. criterion_implicitness": "Explicit",
            "rubric - 1. criterion_subjectiveness": "Objective",
            "rubric - 1. criterion_failure_mode": "Binary",
            "rubric - 2. criterion": "",
            "rubric - 3. criterion": "  Third criterion.  ",
            "rubric - 3. criterion_type": "Dodged Bullet",
            "rubric - 3. criterion_severity": None,
            "rubric - 3. criterion_implicitness": None,
            "rubric - 3. criterion_subjectiveness": "Subjective",
            "rubric - 3. criterion_failure_mode": "Scalar",
        }
        criteria = extract_criteria(row)

        assert [c["index"] for c in criteria] == [1, 3]
        assert criteria[0] == {
            "index": 1,
            "criterion": "First criterion.",
            "type": "Primary Intent",
            "severity": "Certain dealbreaker",
            "implicitness": "Explicit",
            "subjectiveness": "Objective",
            "failure_mode": "Binary",
        }
        # Whitespace is stripped and absent metadata becomes None, not "".
        assert criteria[1]["criterion"] == "Third criterion."
        assert criteria[1]["severity"] is None
        assert criteria[1]["type"] == "Dodged Bullet"

    def test_reads_all_thirty_slots(self) -> None:
        row = {f"rubric - {i}. criterion": f"criterion {i}" for i in range(1, 31)}
        assert len(extract_criteria(row)) == 30

    def test_row_with_no_rubric_returns_empty(self) -> None:
        assert extract_criteria({"prompt": "hi"}) == []

    def test_non_string_criterion_is_ignored(self) -> None:
        assert extract_criteria({"rubric - 1. criterion": 12345}) == []


class TestBuildGymRow:
    def test_row_shape(self) -> None:
        row = build_gym_row(
            {
                "prompt": PROMPT,
                "task_id": "t1",
                "task_response_id": "r1",
                "domain": "Legal",
                "rubric - 1. criterion": "Says $5M.",
            },
            "documents/t1/manifest.json",
            "media/pdfs/abc.pdf",
        )
        content = row["responses_create_params"]["input"][0]["content"]
        assert content == [{"type": "input_text", "text": PROMPT}]

        meta = row["verifier_metadata"]
        assert meta["task_id"] == "t1"
        assert meta["domain"] == "Legal"
        assert meta["document_manifest"] == "documents/t1/manifest.json"
        assert meta["source_pdf"] == "media/pdfs/abc.pdf"
        # The prompt is duplicated into metadata so verify() does not depend on
        # the content-block layout the agent rewrites at rollout time.
        assert meta["prompt"] == PROMPT
        assert len(meta["criteria"]) == 1


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


class TestExtractGeneratedAnswer:
    def test_returns_assistant_text(self) -> None:
        assert _extract_generated_answer(_model_response("The cap is $5M.")) == "The cap is $5M."

    def test_joins_multiple_output_text_blocks(self) -> None:
        msg = NeMoGymResponseOutputMessage(
            id="m",
            content=[
                NeMoGymResponseOutputText(annotations=[], text="part one"),
                NeMoGymResponseOutputText(annotations=[], text="part two"),
            ],
            role="assistant",
            status="completed",
            type="message",
        )
        assert _extract_generated_answer(_response([msg])) == "part one\npart two"

    def test_skips_reasoning_items(self) -> None:
        reasoning = NeMoGymResponseReasoningItem(id="r", summary=[], type="reasoning")
        assert _extract_generated_answer(_response([reasoning, _assistant_msg("final")])) == "final"

    def test_no_assistant_message_returns_empty(self) -> None:
        reasoning = NeMoGymResponseReasoningItem(id="r", summary=[], type="reasoning")
        assert _extract_generated_answer(_response([reasoning])) == ""


# ---------------------------------------------------------------------------
# verify()
# ---------------------------------------------------------------------------


class TestVerify:
    @pytest.mark.asyncio
    async def test_all_criteria_pass_gives_reward_one(self, server: GdpPdfResourcesServer) -> None:
        _mock_judge(server, _scored_content("1", "matches"))
        body = _verify_request(
            "The cap is $5M.",
            criteria=[_criterion(1, "States $5M."), _criterion(2, "Cites the clause.")],
        )

        result = await server.verify(body)

        assert result.reward == 1.0
        assert result.num_criteria == 2
        assert result.num_criteria_passed == 2
        assert result.mean_criterion_pass == approx(1.0)
        assert [e.score for e in result.criterion_evaluations] == ["1", "1"]
        assert result.criterion_evaluations[0].rationale == "matches"

    @pytest.mark.asyncio
    async def test_one_failing_criterion_zeroes_reward_but_not_mean(self, server: GdpPdfResourcesServer) -> None:
        _mock_judge(server, _scored_content("1"), _scored_content("0"))
        body = _verify_request(
            "The cap is $5M.",
            criteria=[_criterion(1, "a"), _criterion(2, "b"), _criterion(3, "c")],
        )

        result = await server.verify(body)

        # All-pass is strict; the mean still reflects partial credit.
        assert result.reward == 0.0
        assert result.num_criteria_passed == 1
        assert result.mean_criterion_pass == approx(1 / 3)

    @pytest.mark.asyncio
    async def test_evaluations_preserve_criterion_order(self, server: GdpPdfResourcesServer) -> None:
        _mock_judge(server, _scored_content("1"))
        criteria = [_criterion(2, "second slot"), _criterion(7, "seventh slot")]

        result = await server.verify(_verify_request("answer", criteria=criteria))

        assert [e.index for e in result.criterion_evaluations] == [2, 7]
        assert [e.criterion for e in result.criterion_evaluations] == ["second slot", "seventh slot"]

    @pytest.mark.asyncio
    async def test_empty_model_output_scores_zero_without_judging(self, server: GdpPdfResourcesServer) -> None:
        mock = _mock_judge(server, _scored_content("1"))
        body = _verify_request("", criteria=[_criterion(1, "a"), _criterion(2, "b")])

        result = await server.verify(body)

        assert result.reward == 0.0
        assert result.num_criteria == 2
        assert result.num_criteria_passed == 0
        assert result.mean_criterion_pass == 0.0
        assert all(not e.passed for e in result.criterion_evaluations)
        mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_rubric_scores_zero_and_is_flagged(self, server: GdpPdfResourcesServer) -> None:
        mock = _mock_judge(server, _scored_content("1"))

        result = await server.verify(_verify_request("An answer.", criteria=[]))

        # all([]) is True -- an ungradeable row must not report a vacuous all-pass.
        assert result.reward == 0.0
        assert result.model_extra["missing_rubric"] is True
        mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_verifier_metadata_scores_zero(self, server: GdpPdfResourcesServer) -> None:
        body = GdpPdfVerifyRequest(
            responses_create_params=_multimodal_params(),
            response=_model_response("answer"),
            verifier_metadata=None,
        )
        result = await server.verify(body)
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_judge_prompt_includes_task_prompt_but_omits_criterion_metadata(
        self, server: GdpPdfResourcesServer
    ) -> None:
        """The judge sees the response, the task prompt, and the criterion text -- a deliberate
        deviation from upstream (surge-ai/gdp-pdf scorer.py), which withholds the task prompt --
        but never the criterion's type/severity/etc. metadata."""
        mock = _mock_judge(server, _scored_content("1"))
        body = _verify_request("The cap is $5M.", criteria=[_criterion(1, "States the cap is $5M.")])

        await server.verify(body)

        sent = mock.await_args.kwargs["json"]
        judge_text = sent.messages[0]["content"]
        assert "States the cap is $5M." in judge_text
        assert "The cap is $5M." in judge_text
        assert PROMPT in judge_text
        assert "Primary Intent" not in judge_text
        assert "Certain dealbreaker" not in judge_text
        assert mock.await_args.kwargs["server_name"] == "judge"
        assert sent.response_format["json_schema"]["name"] == "individual_criteria_score"

    @pytest.mark.asyncio
    async def test_judge_transport_error_propagates(self, server: GdpPdfResourcesServer) -> None:
        # A failed judge *call* must reach judge_failsafe, not be scored as a
        # wrong answer.
        server.server_client.post = AsyncMock(side_effect=ConnectionError("judge down"))

        with raises(JudgeError):
            await server.verify(_verify_request("answer"))

    @pytest.mark.asyncio
    async def test_concurrency_semaphore_disabled(self, config: GdpPdfConfig) -> None:
        config.judge_endpoint_max_concurrency = None
        server = GdpPdfResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        _mock_judge(server, _scored_content("1"))

        result = await server.verify(_verify_request("answer"))

        assert result.reward == 1.0


class TestParseCriterionResult:
    """Pure-function tests for the strict JSON parser, mirroring upstream's own
    ``_parse_criterion_result`` test coverage."""

    def test_valid_json(self) -> None:
        score, rationale = _parse_criterion_result(_scored_content("1", "good"))
        assert score == 1
        assert rationale == "good"

    def test_strips_markdown_fences(self) -> None:
        score, _ = _parse_criterion_result(f"```json\n{_scored_content('1')}\n```")
        assert score == 1

    def test_strips_bare_fences_without_language_tag(self) -> None:
        score, _ = _parse_criterion_result(f"```\n{_scored_content('0')}\n```")
        assert score == 0

    def test_invalid_json_raises(self) -> None:
        with raises(ParseError, match="Invalid JSON"):
            _parse_criterion_result("not json at all")

    def test_non_object_raises(self) -> None:
        with raises(ParseError, match="JSON object"):
            _parse_criterion_result("[1, 2, 3]")

    def test_missing_score_key_raises(self) -> None:
        with raises(ParseError, match="score.*rationale"):
            _parse_criterion_result(json.dumps({"rationale": "x"}))

    def test_missing_rationale_key_raises(self) -> None:
        with raises(ParseError, match="score.*rationale"):
            _parse_criterion_result(json.dumps({"score": "1"}))

    def test_boolean_score_rejected(self) -> None:
        with raises(ParseError, match="number or string"):
            _parse_criterion_result(json.dumps({"score": True, "rationale": "x"}))

    def test_non_string_rationale_rejected(self) -> None:
        with raises(ParseError, match="'rationale' must be a string"):
            _parse_criterion_result(json.dumps({"score": "1", "rationale": 5}))

    def test_non_decimal_score_rejected(self) -> None:
        with raises(ParseError, match="not a valid decimal"):
            _parse_criterion_result(json.dumps({"score": "yes", "rationale": "x"}))

    def test_numeric_score_accepted(self) -> None:
        score, _ = _parse_criterion_result(json.dumps({"score": 1, "rationale": "x"}))
        assert score == 1


class TestCriterionPasses:
    def test_score_one_passes(self) -> None:
        assert _criterion_passes(Decimal("1")) is True

    def test_score_zero_fails(self) -> None:
        assert _criterion_passes(Decimal("0")) is False

    def test_rounds_half_up_to_pass(self) -> None:
        assert _criterion_passes(Decimal("0.99996")) is True

    def test_rounds_half_up_to_fail(self) -> None:
        assert _criterion_passes(Decimal("0.99994")) is False

    def test_above_one_still_passes(self) -> None:
        assert _criterion_passes(Decimal("1.5")) is True


class TestJudgeRetry:
    """Integration-level retry/exhaustion behaviour of _judge_criterion."""

    @pytest.mark.asyncio
    async def test_malformed_then_valid_succeeds_on_retry(self, server: GdpPdfResourcesServer) -> None:
        _mock_judge(server, "not json at all", _scored_content("1", "recovered"))

        result = await server.verify(_verify_request("answer"))

        assert result.criterion_evaluations[0].passed is True
        assert result.criterion_evaluations[0].rationale == "recovered"

    @pytest.mark.asyncio
    async def test_exhausting_attempts_raises_judge_error(self, config: GdpPdfConfig) -> None:
        config.judge_max_attempts = 2
        server = GdpPdfResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        mock = _mock_judge(server, "still not json", "still not json")

        with raises(JudgeError, match="unparseable after 2 attempts"):
            await server.verify(_verify_request("answer"))

        assert mock.await_count == 2

    @pytest.mark.asyncio
    async def test_empty_message_content_retries_then_raises(self, config: GdpPdfConfig) -> None:
        config.judge_max_attempts = 1
        server = GdpPdfResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        _mock_judge(server, "")

        with raises(JudgeError):
            await server.verify(_verify_request("answer"))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    @staticmethod
    def _rollout(reward: float, mean_pass: float, domain: str) -> dict:
        return {
            "reward": reward,
            "mean_criterion_pass": mean_pass,
            "verifier_metadata": {"domain": domain},
        }

    def test_score_fn_reports_all_pass_and_mean(self) -> None:
        scores = GdpPdfResourcesServer._score_fn(self._rollout(1.0, 1.0, "Legal"))
        assert scores["all_pass"] == 1.0
        assert scores["mean_criterion_pass"] == 1.0

    def test_compute_metrics_adds_per_domain_breakdown(self, server: GdpPdfResourcesServer) -> None:
        tasks = [
            [self._rollout(1.0, 1.0, "Legal")],
            [self._rollout(0.0, 0.5, "Legal")],
            [self._rollout(1.0, 1.0, "Healthcare")],
        ]
        metrics = server.compute_metrics(tasks)

        assert any(key.startswith("Legal/") for key in metrics)
        assert any(key.startswith("Healthcare/") for key in metrics)
        assert not any(key.startswith("Legal/per_sample_aggregate") for key in metrics)

    def test_compute_metrics_without_domain_has_no_subsets(self, server: GdpPdfResourcesServer) -> None:
        metrics = server.compute_metrics([[{"reward": 1.0, "mean_criterion_pass": 1.0}]])
        assert not any("/" in key and key.split("/")[0] not in ("mean", "std") for key in metrics if "@" not in key)

    def test_get_key_metrics_selects_headline_scores(self, server: GdpPdfResourcesServer) -> None:
        agent_metrics = {
            "mean/input_tokens": 1000.0,
            "mean/output_tokens": 200.0,
            "pass@1[avg-of-5]/all_pass": 0.3,
            "pass@1[avg-of-5]/mean_criterion_pass": 0.72,
            "pass@5/all_pass": 0.5,
        }
        key = server.get_key_metrics(agent_metrics)

        assert key["mean/input_tokens"] == 1000.0
        assert key["mean/output_tokens"] == 200.0
        assert key["pass@1[avg-of-5]/all_pass"] == 0.3
        assert key["pass@1[avg-of-5]/mean_criterion_pass"] == 0.72
        assert key["pass@5/all_pass"] == 0.5

    def test_get_key_metrics_tolerates_missing_token_counts(self, server: GdpPdfResourcesServer) -> None:
        assert server.get_key_metrics({}) == {}


# ---------------------------------------------------------------------------
# Document embedding (exercised against the committed example manifests)
# ---------------------------------------------------------------------------


EXAMPLE_JSONL = DATA_DIR / "example.jsonl"
_example_manifests = (
    sorted((DATA_DIR / "test_media" / "documents").glob("*/manifest.json")) if DATA_DIR.exists() else []
)

requires_example_manifest = pytest.mark.skipif(not _example_manifests, reason="example manifests not present")


def _row(document_manifest: str, source_pdf: str = "media/pdfs/x.pdf") -> dict:
    return {
        "responses_create_params": {"input": [{"role": "user", "content": [{"type": "input_text", "text": PROMPT}]}]},
        "verifier_metadata": {
            "task_id": "t",
            "document_manifest": document_manifest,
            "source_pdf": source_pdf,
            "criteria": [_criterion(1, "a")],
        },
    }


def _write_stub_manifest(document_dir: Path, *, page_count: int = 1, page_size: int = 200) -> Path:
    """A minimal manifest + page PNGs, standing in for a real prepare_document() output."""
    pages_dir = document_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for n in range(1, page_count + 1):
        image_name = f"page_{n:04d}.png"
        (pages_dir / image_name).write_bytes(b"P" * page_size)
        pages.append(
            {"page_number": n, "text": f"page {n} text", "image": f"pages/{image_name}", "width": 10, "height": 10}
        )
    manifest = {
        "source_pdf": "stub.pdf",
        "source_sha256": "stub-sha",
        "source_dpi": 150,
        "page_count": page_count,
        "pages": pages,
    }
    manifest_path = document_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


class TestPrepare:
    """prepare() with HuggingFace and LiteParse rendering stubbed out -- no network, no OCR."""

    @staticmethod
    def _source_row(task_id: str, pdf_name: str, criterion: Optional[str] = "Says the thing.") -> dict:
        row = {
            "prompt": PROMPT,
            "task_id": task_id,
            "task_response_id": f"{task_id}-r",
            "domain": "Legal",
            "pdf_path": f"pdfs/{pdf_name}",
        }
        if criterion:
            row["rubric - 1. criterion"] = criterion
        return row

    def _patch_hf(self, monkeypatch, records: list, downloaded: list, rendered: Optional[list] = None) -> None:
        import benchmarks.gdp_pdf.prepare as prepare_module

        fake_datasets = MagicMock()
        fake_datasets.load_dataset = MagicMock(return_value=records)
        fake_hub = MagicMock()
        fake_hub.hf_hub_download = MagicMock(side_effect=lambda **kw: downloaded.append(kw["filename"]))

        monkeypatch.setitem(__import__("sys").modules, "datasets", fake_datasets)
        monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hub)
        monkeypatch.setattr(prepare_module, "_resolve_hf_token", lambda: "tok")

        def _fake_prepare_document(pdf_path: Path, document_dir: Path, **kwargs: Any) -> Path:
            if rendered is not None:
                rendered.append(pdf_path.name)
            return _write_stub_manifest(document_dir)

        monkeypatch.setattr(prepare_module, "prepare_document", _fake_prepare_document)

    def test_writes_rows_and_downloads_each_pdf(self, monkeypatch, tmp_path: Path) -> None:
        import json

        downloaded: list = []
        rendered: list = []
        self._patch_hf(
            monkeypatch, [self._source_row("t1", "a.pdf"), self._source_row("t2", "b.pdf")], downloaded, rendered
        )

        out = tmp_path / "out.jsonl"
        rows = _prepare_rows(output_path=out, media_dir=tmp_path / "media", documents_dir=tmp_path / "documents")

        assert len(rows) == 2
        assert downloaded == ["pdfs/a.pdf", "pdfs/b.pdf"]
        assert rendered == ["a.pdf", "b.pdf"]
        written = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        assert [r["verifier_metadata"]["task_id"] for r in written] == ["t1", "t2"]
        assert written[0]["verifier_metadata"]["document_manifest"] == "documents/t1/manifest.json"
        assert written[0]["verifier_metadata"]["source_pdf"] == "media/pdfs/a.pdf"

    def test_limit_truncates_the_task_list(self, monkeypatch, tmp_path: Path) -> None:
        downloaded: list = []
        records = [self._source_row(f"t{i}", f"{i}.pdf") for i in range(5)]
        self._patch_hf(monkeypatch, records, downloaded)

        rows = _prepare_rows(
            output_path=tmp_path / "out.jsonl",
            media_dir=tmp_path / "media",
            documents_dir=tmp_path / "documents",
            limit=2,
        )

        assert len(rows) == 2

    def test_rows_missing_a_pdf_or_rubric_are_skipped(self, monkeypatch, tmp_path: Path) -> None:
        downloaded: list = []
        records = [
            self._source_row("good", "a.pdf"),
            self._source_row("no-rubric", "b.pdf", criterion=None),
            {**self._source_row("no-pdf", "c.pdf"), "pdf_path": ""},
        ]
        self._patch_hf(monkeypatch, records, downloaded)

        rows = _prepare_rows(
            output_path=tmp_path / "out.jsonl", media_dir=tmp_path / "media", documents_dir=tmp_path / "documents"
        )

        assert [r["verifier_metadata"]["task_id"] for r in rows] == ["good"]
        assert downloaded == ["pdfs/a.pdf"]

    def test_explicit_token_takes_precedence(self, monkeypatch, tmp_path: Path) -> None:
        import benchmarks.gdp_pdf.prepare as prepare_module

        downloaded: list = []
        self._patch_hf(monkeypatch, [self._source_row("t1", "a.pdf")], downloaded)
        resolved = MagicMock(return_value="from-config")
        monkeypatch.setattr(prepare_module, "_resolve_hf_token", resolved)

        _prepare_rows(
            output_path=tmp_path / "o.jsonl",
            media_dir=tmp_path / "m",
            documents_dir=tmp_path / "documents",
            hf_token="explicit",
        )

        resolved.assert_not_called()

    def test_resolve_hf_token_survives_hydra_system_exit(self, monkeypatch) -> None:
        # get_hf_token() goes through Hydra, which raises SystemExit (a
        # BaseException) when the script is run with its own CLI flags.
        fake_module = MagicMock()
        fake_module.get_hf_token = MagicMock(side_effect=SystemExit(2))
        monkeypatch.setitem(__import__("sys").modules, "nemo_gym.global_config", fake_module)

        assert _resolve_hf_token() is None


class TestFetchExampleMedia:
    """The committed example.jsonl must be usable from a fresh clone, where neither the PDF nor
    the rendered manifest is committed. These run without network or real LiteParse OCR."""

    @staticmethod
    def _write_example_jsonl(tmp_path: Path, names: list) -> None:
        with open(tmp_path / "example.jsonl", "w") as f:
            for n in names:
                task_id = n.removesuffix(".pdf")
                f.write(
                    json.dumps(_row(f"test_media/documents/{task_id}/manifest.json", f"media/pdfs/{n}")) + "\n"
                )

    def _patch(self, monkeypatch, downloaded: list, rendered: list):
        import benchmarks.gdp_pdf.prepare as prepare_module

        fake_hub = MagicMock()

        def _dl(**kw):
            downloaded.append(kw["filename"])
            dest = Path(kw["local_dir"]) / kw["filename"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"%PDF-1.4 stub")
            return str(dest)

        fake_hub.hf_hub_download = MagicMock(side_effect=_dl)
        monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hub)

        def _fake_prepare_document(pdf_path: Path, document_dir: Path, **kwargs: Any) -> Path:
            rendered.append(pdf_path.name)
            return _write_stub_manifest(document_dir)

        monkeypatch.setattr(prepare_module, "prepare_document", _fake_prepare_document)

    def test_downloads_and_renders_only_the_referenced_documents(self, monkeypatch, tmp_path: Path) -> None:
        self._write_example_jsonl(tmp_path, ["a.pdf", "b.pdf"])
        downloaded: list = []
        rendered: list = []
        self._patch(monkeypatch, downloaded, rendered)

        n = fetch_example_media(tmp_path, hf_token="tok")

        assert downloaded == ["pdfs/a.pdf", "pdfs/b.pdf"]
        assert rendered == ["a.pdf", "b.pdf"]
        assert n == 2
        assert (tmp_path / "test_media/documents/a/manifest.json").is_file()

    def test_skips_documents_already_rendered(self, monkeypatch, tmp_path: Path) -> None:
        self._write_example_jsonl(tmp_path, ["a.pdf", "b.pdf"])
        _write_stub_manifest(tmp_path / "test_media/documents/a")
        downloaded: list = []
        rendered: list = []
        self._patch(monkeypatch, downloaded, rendered)

        n = fetch_example_media(tmp_path)

        assert downloaded == ["pdfs/b.pdf"]
        assert rendered == ["b.pdf"]
        assert n == 1

    def test_missing_example_jsonl_is_actionable(self, tmp_path: Path) -> None:
        with raises(FileNotFoundError, match="prepare.py"):
            fetch_example_media(tmp_path)


class TestWriteExample:
    def test_picks_smallest_documents_and_rewrites_paths(self, tmp_path: Path) -> None:
        import json

        # Descending size, so "smallest first" is observable in the output order.
        for i, size in enumerate([5000, 4000, 3000, 2000, 1000, 500]):
            _write_stub_manifest(tmp_path / "documents" / f"t{i}", page_size=size)
        rows = [_row(f"documents/t{i}/manifest.json") for i in range(6)]

        write_example(rows, tmp_path)

        written = [json.loads(line) for line in (tmp_path / "example.jsonl").read_text().splitlines() if line.strip()]
        assert len(written) == 5
        assert [r["verifier_metadata"]["document_manifest"] for r in written] == [
            "test_media/documents/t5/manifest.json",
            "test_media/documents/t4/manifest.json",
            "test_media/documents/t3/manifest.json",
            "test_media/documents/t2/manifest.json",
            "test_media/documents/t1/manifest.json",
        ]
        # Document caches are copied next to the JSONL so example.jsonl is self-contained.
        for row in written:
            assert (tmp_path / row["verifier_metadata"]["document_manifest"]).is_file()

    def test_rows_without_a_local_manifest_are_dropped(self, tmp_path: Path) -> None:
        _write_stub_manifest(tmp_path / "documents" / "present")

        write_example(
            [_row("documents/present/manifest.json"), _row("documents/absent/manifest.json")],
            tmp_path,
        )

        assert len((tmp_path / "example.jsonl").read_text().strip().splitlines()) == 1

    def test_large_example_document_is_flagged(self, tmp_path: Path, capsys) -> None:
        _write_stub_manifest(tmp_path / "documents" / "big", page_size=EXAMPLE_PDF_SIZE_WARN_BYTES + 1)

        write_example([_row("documents/big/manifest.json")], tmp_path)

        assert "large for git" in capsys.readouterr().out


@pytest.mark.skipif(not EXAMPLE_JSONL.is_file(), reason="example.jsonl not present")
class TestExampleData:
    @staticmethod
    def _rows() -> list:
        return [json.loads(line) for line in EXAMPLE_JSONL.read_text().splitlines() if line.strip()]

    def test_example_rows_are_well_formed(self) -> None:
        # Must hold on a fresh clone, where nothing has been downloaded or rendered yet.
        rows = self._rows()
        assert rows
        for row in rows:
            meta = row["verifier_metadata"]
            assert meta["criteria"], "every example row needs a rubric"
            assert meta["prompt"]
            # Documents are rendered on demand, so only the reference lives in git.
            assert meta["document_manifest"].startswith("test_media/")
        # Media stays out of the JSONL; the agent embeds it at rollout time.
        assert "base64" not in EXAMPLE_JSONL.read_text()

    @requires_example_manifest
    def test_referenced_manifests_resolve_once_fetched(self) -> None:
        for row in self._rows():
            assert (DATA_DIR / row["verifier_metadata"]["document_manifest"]).is_file()
