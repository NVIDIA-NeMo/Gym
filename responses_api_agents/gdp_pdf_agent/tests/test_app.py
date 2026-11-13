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
"""Tests for the gdp_pdf agent's manifest-based document embedding, reactive DPI backoff, and
page compositing."""

import base64
import io
import json
import random
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiohttp import ClientPayloadError, ClientResponseError
from fastapi.testclient import TestClient
from PIL import Image

from nemo_gym.agents import AGENT_OUTCOME_METADATA_KEY, AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.agents.utils import verify_call
from nemo_gym.client import MODEL_RESPONSE_TIMEOUT_REASON, ServerClient
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import (
    AGENT_ROUTE_FAILED,
    EpisodeId,
    FailureBoundary,
    FailureEvent,
    FailureHeader,
    FailureReport,
    ModelFailureDetails,
    NeMoGymResponse,
    NgFailureEvent,
    RequestFailureLog,
    SdkErrorDetails,
    TaskId,
)
from nemo_gym.server_runtime import AgentContext
from responses_api_agents.gdp_pdf_agent import app as gdp_pdf_agent_app
from responses_api_agents.gdp_pdf_agent.app import (
    DOCUMENT_DELIVERY_METADATA_KEY,
    DocumentDelivery,
    GdpPdfAgent,
    GdpPdfAgentConfig,
    _build_task_text_block,
    _compose_pages,
    _delivery_notice_blocks,
    _input_limit,
    _load_manifest,
    _manifest_text_blocks,
    _open_page_image,
    _render_manifest_images,
    _strip_image_blocks,
)
from responses_api_agents.simple_agent.app import SimpleAgentRunRequest, SimpleAgentVerifyResponse


SERVER_DIR = Path(__file__).resolve().parents[3] / "resources_servers" / "gdp_pdf"
DATA_DIR = SERVER_DIR / "data"
_example_manifests = (
    sorted((DATA_DIR / "test_media" / "documents").glob("*/manifest.json")) if DATA_DIR.exists() else []
)
requires_example_manifest = pytest.mark.skipif(not _example_manifests, reason="example manifests not present")
_example_manifest_relpath = (
    f"test_media/documents/{_example_manifests[0].parent.name}/manifest.json" if _example_manifests else None
)


def _agent_config(**overrides) -> GdpPdfAgentConfig:
    config_kwargs = {
        "name": "gdp_pdf_simple_agent",
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "app.py",
        "max_steps": 1,
        "resources_server": ResourcesServerRef(type="resources_servers", name="gdp_pdf"),
        "model_server": ModelServerRef(type="responses_api_models", name="policy_model"),
        "documents_base_dir": "resources_servers/gdp_pdf/data",
    }
    config_kwargs.update(overrides)
    return GdpPdfAgentConfig(**config_kwargs)


def _agent(**overrides) -> GdpPdfAgent:
    return GdpPdfAgent(config=_agent_config(**overrides), server_client=AsyncMock(spec=ServerClient))


def _block_type(block) -> str:
    return block.get("type") if isinstance(block, dict) else block.type


def _run_request(document_manifest: str = "test_media/documents/x/manifest.json") -> SimpleAgentRunRequest:
    return SimpleAgentRunRequest.model_validate(
        {
            "responses_create_params": {
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "What is the cap?"}]}]
            },
            "verifier_metadata": {
                "task_id": "t1",
                "document_manifest": document_manifest,
                "criteria": [{"index": 1}],
            },
        }
    )


def _response(text: str = "answer") -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="policy_model",
        object="response",
        tools=[],
        parallel_tool_calls=False,
        tool_choice="none",
        output=[
            {
                "type": "message",
                "id": "m1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    )


def _incomplete_response(text: str = "cut off mid-") -> NeMoGymResponse:
    """A response vLLM accepted but couldn't finish -- ran out of output budget, not rejected."""
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="policy_model",
        object="response",
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
        tools=[],
        parallel_tool_calls=False,
        tool_choice="none",
        output=[
            {
                "type": "message",
                "id": "m1",
                "role": "assistant",
                "status": "incomplete",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    )


def _error(message: str, status: int = 500) -> ClientResponseError:
    error = ClientResponseError(None, (), status=status, message="upstream failure")
    error.response_content = message.encode()
    return error


def _write_text_manifest(base_dir: Path) -> str:
    """A one-page, text-only document manifest under ``base_dir``; returns its path relative to it.

    The committed example manifests are not part of every checkout, so tests of the /run flow that
    do not need page images build this one and run the agent with ``include_images=False``.
    """
    document_dir = base_dir / "documents" / "doc"
    document_dir.mkdir(parents=True)
    manifest = {"source_dpi": 150, "pages": [{"page_number": 1, "text": "The cap is 10."}]}
    (document_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return "documents/doc/manifest.json"


def _text_only_agent(tmp_path: Path, **overrides) -> tuple[GdpPdfAgent, str]:
    agent = _agent(documents_base_dir=str(tmp_path), include_images=False, strip_images_from_output=False, **overrides)
    return agent, _write_text_manifest(tmp_path)


def _image_agent(tmp_path: Path, **overrides) -> tuple[GdpPdfAgent, str, list[dict]]:
    """An agent over a two-page manifest with page screenshots, built under ``tmp_path``; also returns the
    list that collects every request body sent to /verify."""
    document_dir = tmp_path / "documents" / "img"
    document_dir.mkdir(parents=True)
    pages = []
    for number in (1, 2):
        Image.new("RGB", (60, 80), "white").save(document_dir / f"page_{number}.png")
        pages.append({"page_number": number, "text": f"Page {number} text.", "image": f"page_{number}.png"})
    (document_dir / "manifest.json").write_text(json.dumps({"source_dpi": 150, "pages": pages}), encoding="utf-8")
    agent = _agent(documents_base_dir=str(tmp_path), **({"strip_images_from_output": False} | overrides))
    verified: list[dict] = []

    def post(**kw):
        if kw["url_path"] == "/verify":
            verified.append(kw["json"])
            return SimpleNamespace(cookies={}, data=kw["json"] | {"reward": 0.25})
        return SimpleNamespace(cookies={}, data={})

    agent.server_client.post = AsyncMock(side_effect=post)
    return agent, "documents/img/manifest.json", verified


async def _run_with_episodes(agent: GdpPdfAgent, manifest: str, episode) -> SimpleAgentVerifyResponse:
    with (
        patch.object(GdpPdfAgent, "_create_episode", side_effect=episode),
        patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
        patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
        patch.object(gdp_pdf_agent_app, "get_response_json", new=AsyncMock(side_effect=lambda r: r.data)),
    ):
        return await agent.run(SimpleNamespace(cookies={}), _run_request(manifest))


def _model_reply(text: str = "The cap is 10.") -> MagicMock:
    reply = MagicMock(status=200, ok=True, cookies={})
    reply.read = AsyncMock(return_value=_response(text).model_copy(update={"status": "completed"}).model_dump_json())
    return reply


def _model_error(status: int, body: str) -> MagicMock:
    """A model server's error reply, as the HTTP client raises it."""
    request_info = SimpleNamespace(url="http://model", method="POST", headers={}, real_url="http://model")
    reply = MagicMock(status=status, ok=False, cookies={})
    reply.content.read = AsyncMock(return_value=body.encode())
    reply.raise_for_status = MagicMock(
        side_effect=ClientResponseError(request_info, (), status=status, message=HTTPStatus(status).phrase)
    )
    return reply


_EPISODE = EpisodeId(rollout_id="rollout", attempt=0)


def _seed(document_manifest: str | None) -> dict:
    return AgentSeedSessionRequest(
        agent_session_id="gdp-pdf-session",
        episode_id=_EPISODE,
        task_id=TaskId(taskset="gdp_pdf", task_id="t1"),
        agent_context=(
            AgentContext(artifacts={"document_manifest": {"path": document_manifest}})
            if document_manifest is not None
            else None
        ),
    ).model_dump(mode="json")


def _session_run(tmp_path: Path, replies, **overrides) -> tuple[dict, dict, list[dict]]:
    """Seed a session for a two-page document, run its one activation and close it.

    Returns the activation's response, the close's observations and the model requests the agent sent.
    """
    agent, manifest, _ = _image_agent(tmp_path, strip_images_from_output=True, **overrides)
    agent.server_client.global_config_dict = {"observability_enabled": True}
    agent.server_client.post = AsyncMock(side_effect=replies)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=_seed(manifest)).status_code == 200
    question = {"input": [{"role": "user", "content": [{"type": "input_text", "text": "What is the cap?"}]}]}
    response = client.post("/ng-rollout/rollout/v1/responses", json=question)
    assert response.status_code == 200
    close = client.post(
        "/v1/agent_sessions/close",
        json=AgentCloseSessionRequest(agent_session_id="gdp-pdf-session", episode_id=_EPISODE).model_dump(mode="json"),
    )
    requests = [call.kwargs["json"].model_dump(mode="json") for call in agent.server_client.post.await_args_list]
    return response.json(), close.json()["agent_observations"], requests


def _blocks(request: dict) -> list[str]:
    return [block["type"] for block in request["input"][0]["content"]]


class TestAgentSessions:
    def test_the_document_reaches_the_model_and_its_images_stay_out_of_the_observations(self, tmp_path: Path) -> None:
        response, observations, [request] = _session_run(tmp_path, [_model_reply()])

        assert _blocks(request) == ["input_image", "input_image", "input_text"]
        task_text = request["input"][0]["content"][-1]["text"]
        assert "Task:\nWhat is the cap?" in task_text and "## Page 2\n\nPage 2 text." in task_text
        assert response["status"] == "completed"
        delivery = json.loads(response["metadata"][DOCUMENT_DELIVERY_METADATA_KEY])
        assert (delivery["image_count"], delivery["image_dpi"], delivery["limit"]) == (2, 150, None)
        assert "input_image" not in json.dumps(observations)
        assert "multimodal_history_redacted" in [gap["code"] for gap in observations["gaps"]]

    def test_an_input_limit_rejection_backs_off_within_the_one_activation(self, tmp_path: Path) -> None:
        response, _, requests = _session_run(
            tmp_path, [_model_error(400, "This model's maximum context length is 8192 tokens"), _model_reply()]
        )

        assert len(requests) == 2
        assert response["status"] == "completed"
        delivery = json.loads(response["metadata"][DOCUMENT_DELIVERY_METADATA_KEY])
        assert delivery["image_dpi"] == 120
        assert delivery["input_limit_rejections"] == 1
        assert [attempt["limit"] for attempt in delivery["rejected_attempts"]] == ["context"]

    def test_a_document_that_never_fits_is_an_unclassified_zero_attempt(self, tmp_path: Path) -> None:
        """As on /run, the Environment Server verifies the empty answer, which scores 0, instead of failing it."""
        response, _, requests = _session_run(tmp_path, lambda **kwargs: _model_error(413, "Payload Too Large"))

        # 150, 120, 96, 76 and 72 DPI, then text only.
        assert len(requests) == 6
        assert _blocks(requests[-1]) == ["input_text"]
        assert (response["status"], response["output"]) == ("failed", [])
        assert AGENT_OUTCOME_METADATA_KEY not in response["metadata"]
        delivery = json.loads(response["metadata"][DOCUMENT_DELIVERY_METADATA_KEY])
        assert (delivery["limit"], delivery["text_only_fallback"]) == ("payload", True)

    def test_another_model_error_stays_a_classified_failure(self, tmp_path: Path) -> None:
        response, _, requests = _session_run(tmp_path, [_model_error(500, "upstream exploded")], max_model_retries=0)

        assert len(requests) == 1
        assert response["status"] == "failed"
        outcome = json.loads(response["metadata"][AGENT_OUTCOME_METADATA_KEY])
        assert (outcome["failure_kind"], outcome["status_code"]) == ("model_request_error", 500)
        assert json.loads(response["metadata"][DOCUMENT_DELIVERY_METADATA_KEY])["limit"] is None

    @pytest.mark.parametrize(
        ("manifest", "overrides", "detail"),
        [
            (None, {}, "agent_context does not satisfy GdpPdfAgentContext: document_manifest: Field required"),
            ("documents/missing/manifest.json", {}, "agent_context artifact 'documents/missing/manifest.json' is not"),
            ("documents/img/manifest.json", {"dpi": 100}, "rendered at 150 DPI, but this agent's starting dpi is 100"),
        ],
    )
    def test_a_session_without_a_usable_document_is_refused(self, tmp_path: Path, manifest, overrides, detail) -> None:
        agent, _, _ = _image_agent(tmp_path, **overrides)
        agent.server_client.global_config_dict = {}

        seeded = TestClient(agent.setup_webserver()).post("/v1/agent_sessions", json=_seed(manifest))

        assert seeded.status_code == 422
        assert detail in seeded.json()["detail"]
        agent.server_client.post.assert_not_awaited()

    @pytest.mark.parametrize(
        ("manifest_text", "detail"),
        [
            ("{not json", "JSONDecodeError"),
            ("[]", "is not a JSON object"),
            ('{"source_dpi": "high", "pages": []}', "invalid literal for int()"),
            ('{"pages": [{"text": "no page number"}]}', "KeyError"),
            ('{"pages": [{"page_number": 1, "image": "../../outside.png"}]}', "names page image '../../outside.png'"),
            ('{"pages": [{"page_number": 1, "image": "/etc/hosts"}]}', "names page image '/etc/hosts'"),
        ],
        ids=["not-json", "not-an-object", "bad-dpi", "no-page-number", "image-above", "image-absolute"],
    )
    def test_a_malformed_manifest_refuses_the_seed(self, tmp_path: Path, manifest_text: str, detail: str) -> None:
        """Every retry would read the same file, so a 422 fails the episode once instead of a 500 retried."""
        agent, manifest, _ = _image_agent(tmp_path)
        agent.server_client.global_config_dict = {}
        (tmp_path / manifest).write_text(manifest_text, encoding="utf-8")

        seeded = TestClient(agent.setup_webserver()).post("/v1/agent_sessions", json=_seed(manifest))

        assert seeded.status_code == 422
        assert seeded.json()["detail"].startswith(f"document manifest {tmp_path.resolve() / manifest} cannot be read")
        assert detail in seeded.json()["detail"]

    def test_an_empty_body_server_error_backs_off_like_a_payload_limit(self, tmp_path: Path) -> None:
        """vLLM answers an oversized multi-image prompt with an empty 500, which Gym's model server relays."""
        response, _, requests = _session_run(tmp_path, [_model_error(500, ""), _model_reply()], max_model_retries=0)

        assert len(requests) == 2
        assert response["status"] == "completed"
        delivery = json.loads(response["metadata"][DOCUMENT_DELIVERY_METADATA_KEY])
        assert (delivery["image_dpi"], delivery["limit"]) == (120, None)
        assert [attempt["limit"] for attempt in delivery["rejected_attempts"]] == ["payload"]

    def test_an_empty_body_server_error_that_never_stops_stays_a_classified_failure(self, tmp_path: Path) -> None:
        """Only empty bodies said the input was too large, which an outage looks like too: the outcome policy
        judges the classified failure instead of the Environment Server scoring an empty answer 0."""
        response, _, requests = _session_run(tmp_path, lambda **kwargs: _model_error(500, ""), max_model_retries=0)

        assert len(requests) == 6
        assert _blocks(requests[-1]) == ["input_text"]
        assert response["status"] == "failed"
        outcome = json.loads(response["metadata"][AGENT_OUTCOME_METADATA_KEY])
        assert (outcome["failure_kind"], outcome["status_code"]) == ("model_request_error", 500)
        delivery = json.loads(response["metadata"][DOCUMENT_DELIVERY_METADATA_KEY])
        # Every attempt, the text-only one included, was rejected; none of them set a terminal input limit.
        assert (delivery["limit"], delivery["text_only_fallback"], delivery["input_limit_rejections"]) == (
            None,
            True,
            6,
        )


def _recording_agent(tmp_path: Path, **overrides) -> tuple[GdpPdfAgent, str, list[dict]]:
    """The two-page document agent, its model requests recorded; the Resources Server accepts every call."""
    agent, manifest, _ = _image_agent(tmp_path, **overrides)
    agent.server_client.global_config_dict = {}
    model_requests: list[dict] = []

    async def post(**kwargs) -> MagicMock:
        body = kwargs["json"]
        body = body.model_dump(mode="json") if hasattr(body, "model_dump") else body
        if kwargs["url_path"].endswith("/v1/responses"):
            model_requests.append(body)
            return _model_reply()
        reply = MagicMock(status=200, ok=True, cookies={})
        reply.read = AsyncMock(return_value=json.dumps(body | {"reward": 0.5}))
        return reply

    agent.server_client.post = AsyncMock(side_effect=post)
    return agent, manifest, model_requests


@pytest.mark.parametrize(
    "overrides", [{}, {"max_pages": 1}, {"include_images": False}], ids=["all", "max-pages", "text"]
)
def test_an_agent_session_sends_the_model_the_same_request_as_run(tmp_path: Path, overrides: dict) -> None:
    run_agent, manifest, run_requests = _recording_agent(tmp_path / "run", **overrides)
    row = _run_request(manifest).model_dump(mode="json", exclude_unset=True)
    run_result = TestClient(run_agent.setup_webserver()).post("/run", json=row)
    assert run_result.status_code == 200

    session_agent, _, session_requests = _recording_agent(tmp_path / "session", **overrides)
    client = TestClient(session_agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=_seed(manifest)).status_code == 200
    activation = client.post("/ng-rollout/rollout/v1/responses", json=row["responses_create_params"])
    assert activation.status_code == 200

    assert session_requests == run_requests
    session_delivery = json.loads(activation.json()["metadata"][DOCUMENT_DELIVERY_METADATA_KEY])
    expected_truncated = 1 if overrides.get("max_pages") == 1 else 0
    assert session_delivery["pages_truncated"] == expected_truncated
    assert ("## Page 2" in run_requests[0]["input"][0]["content"][-1]["text"]) == (not expected_truncated)


async def test_run_refuses_a_manifest_whose_page_image_leaves_its_directory(tmp_path: Path) -> None:
    agent, manifest, _ = _image_agent(tmp_path)
    (tmp_path / "secret.png").write_bytes(b"not the document")
    (tmp_path / manifest).write_text(
        json.dumps({"source_dpi": 150, "pages": [{"page_number": 1, "image": "../../secret.png"}]}), encoding="utf-8"
    )
    episode = AsyncMock()

    with pytest.raises(ValueError, match=r"names page image '\.\./\.\./secret.png', which is outside"):
        await _run_with_episodes(agent, manifest, episode)
    episode.assert_not_awaited()


class TestIncompleteOutputIsTheResult:
    """An incomplete answer means the model spent its output budget: that is the row's result, not a
    delivery problem to negotiate away. Regenerating it is opt-in (``regenerate_on_incomplete_output``)."""

    async def test_incomplete_answer_with_images_is_kept_and_verified(self, tmp_path: Path) -> None:
        agent, manifest, verified = _image_agent(tmp_path)
        calls = []

        async def episode(params, **kwargs):
            calls.append(params)
            return _incomplete_response("cut off mid-"), None, {}, {}

        result = await _run_with_episodes(agent, manifest, episode)
        assert len(calls) == 1
        assert len(verified) == 1
        assert verified[0]["response"]["status"] == "incomplete"
        assert verified[0]["response"]["output"][0]["content"][0]["text"] == "cut off mid-"
        assert result.response.status == "incomplete"
        assert result.reward == 0.25
        assert result.document_delivery["image_dpi"] == 150
        assert result.document_delivery["rejected_attempts"] == []
        assert result.document_delivery["input_limit_rejections"] == 0
        assert result.document_delivery["output_truncation_regenerations"] == 0
        assert result.agent_adaptations == {"input_limit_rejections": 0, "output_truncation_regenerations": 0}

    async def test_flag_restores_regeneration_and_counts_it(self, tmp_path: Path) -> None:
        agent, manifest, verified = _image_agent(tmp_path, regenerate_on_incomplete_output=True)
        calls = []

        async def episode(params, **kwargs):
            calls.append(params)
            return (_incomplete_response() if len(calls) == 1 else _response("full answer")), None, {}, {}

        result = await _run_with_episodes(agent, manifest, episode)
        assert len(calls) == 2
        assert len(verified) == 1
        assert verified[0]["response"]["status"] != "incomplete"
        assert result.document_delivery["image_dpi"] == 120
        assert [a["limit"] for a in result.document_delivery["rejected_attempts"]] == ["output_truncated"]
        assert result.document_delivery["output_truncation_regenerations"] == 1
        assert result.document_delivery["input_limit_rejections"] == 0
        assert result.agent_adaptations == {"input_limit_rejections": 0, "output_truncation_regenerations": 1}

    async def test_an_image_count_rejection_still_adapts_and_is_counted(self, tmp_path: Path) -> None:
        agent, manifest, verified = _image_agent(tmp_path)
        calls = []

        async def episode(params, **kwargs):
            calls.append(params)
            if len(calls) == 1:
                raise _error("At most 1 image(s) may be provided in one prompt.", status=400)
            return _response("full answer"), None, {}, {}

        result = await _run_with_episodes(agent, manifest, episode)
        assert len(calls) == 2
        images = [b for b in calls[1].input[0].content if _block_type(b) == "input_image"]
        assert len(images) == 1  # two pages composited into the one image the endpoint allows
        assert result.document_delivery["pages_per_image"] == 2
        assert [a["limit"] for a in result.document_delivery["rejected_attempts"]] == ["image_count"]
        assert result.document_delivery["input_limit_rejections"] == 1
        assert result.document_delivery["output_truncation_regenerations"] == 0
        assert result.agent_adaptations == {"input_limit_rejections": 1, "output_truncation_regenerations": 0}
        assert len(verified) == 1 and result.reward == 0.25

    async def test_a_complete_answer_is_unchanged(self, tmp_path: Path) -> None:
        agent, manifest, verified = _image_agent(tmp_path)
        calls = []

        async def episode(params, **kwargs):
            calls.append(params)
            return _response("full answer"), None, {}, {}

        result = await _run_with_episodes(agent, manifest, episode)
        assert len(calls) == 1
        assert len(verified) == 1
        assert result.response.output[0].content[0].text == "full answer"
        assert result.document_delivery["image_dpi"] == 150
        assert result.document_delivery["rejected_attempts"] == []
        assert result.document_delivery["input_limit_rejections"] == 0
        assert result.document_delivery["output_truncation_regenerations"] == 0


class TestConfig:
    def test_defaults_are_conservative(self) -> None:
        config = _agent_config()
        assert config.dpi == 150
        assert config.min_dpi == 72
        assert config.max_images is None
        assert config.max_pages is None
        assert config.include_text is True
        assert config.include_images is True
        assert config.strip_images_from_output is True
        assert config.regenerate_on_incomplete_output is False

    def test_overrides_apply(self) -> None:
        config = _agent_config(dpi=200, max_images=8, max_pages=10, include_images=False)
        assert config.dpi == 200
        assert config.max_images == 8
        assert config.max_pages == 10
        assert config.include_images is False


class TestInputLimit:
    @pytest.mark.parametrize(
        "message,status,expected",
        [
            ("maximum context length is 262144 tokens", 500, ("context", None)),
            ("payload too large", 500, ("payload", None)),
            ("", 413, ("payload", None)),
            ("request_too_large", 500, ("payload", None)),
            ("image dimensions exceed the maximum allowed size", 400, ("payload", None)),
            ("At most 10 images may be provided in one request.", 500, ("image_count", 10)),
            ("Too many images", 400, ("image_count", None)),
            ("invalid image data", 400, (None, None)),
            ("out of memory", 500, (None, None)),
            ("RateLimitError: maximum context length quota", 500, (None, None)),
            ("maximum context length", 401, (None, None)),
            ("", 500, ("payload", None)),  # empty body on a 5xx: treat as retryable, not unclassified
            ("", 400, (None, None)),  # empty body on a non-5xx: no evidence of an input-size issue
        ],
    )
    def test_classifies_upstream_errors(self, message, status, expected) -> None:
        assert _input_limit(_error(message, status)) == expected


class TestDocumentDeliveryAdapt:
    def test_reduces_dpi_by_20_percent_toward_floor(self) -> None:
        delivery = DocumentDelivery(150, None)
        assert delivery.adapt("context", None, min_dpi=72) is True
        assert delivery.image_dpi == 120
        assert delivery.adapt("payload", None, min_dpi=72) is True
        assert delivery.image_dpi == 96

    def test_stops_at_the_floor(self) -> None:
        delivery = DocumentDelivery(80, None)
        # DPI 80 -> 72 (floor).
        assert delivery.adapt("context", None, min_dpi=72) is True
        assert delivery.image_dpi == 72
        assert delivery.text_only_fallback is False
        # Floor still doesn't fit -> engage text-only fallback (one more retry).
        assert delivery.adapt("context", None, min_dpi=72) is True
        assert delivery.image_dpi == 72
        assert delivery.text_only_fallback is True
        # Still failing with text-only -> genuinely give up.
        assert delivery.adapt("context", None, min_dpi=72) is False

    def test_image_count_with_explicit_cap(self) -> None:
        delivery = DocumentDelivery(150, None)
        delivery.image_count = 20
        assert delivery.adapt("image_count", 8, min_dpi=72) is True
        assert delivery.max_images == 8

    def test_image_count_without_explicit_cap_decrements_by_one(self) -> None:
        delivery = DocumentDelivery(150, None)
        delivery.image_count = 5
        assert delivery.adapt("image_count", None, min_dpi=72) is True
        assert delivery.max_images == 4

    def test_image_count_cannot_increase_the_cap(self) -> None:
        delivery = DocumentDelivery(150, 4)
        delivery.image_count = 4
        assert delivery.adapt("image_count", 10, min_dpi=72) is False
        assert delivery.max_images == 4

    def test_records_every_attempt(self) -> None:
        delivery = DocumentDelivery(150, None)
        delivery.adapt("context", None, min_dpi=72)
        delivery.adapt("context", None, min_dpi=72)
        assert [a["image_dpi"] for a in delivery.attempts] == [150, 120]


class TestComposePages:
    def test_single_page_is_not_composited(self) -> None:
        image = Image.new("RGB", (100, 140), "red")
        block = _compose_pages([(1, image)], image_dpi=150, image_format="png", jpeg_quality=90)
        encoded = block["image_url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
            assert decoded.size == (100, 140)

    def test_multiple_pages_are_composited_into_a_grid(self) -> None:
        images = [(i, Image.new("RGB", (100, 140), "red")) for i in range(1, 5)]
        block = _compose_pages(images, image_dpi=150, image_format="png", jpeg_quality=90)
        encoded = block["image_url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
            # 2 columns x 2 rows, each cell >= page size plus a label strip.
            assert decoded.width == 200
            assert decoded.height > 280
        assert block["type"] == "input_image"

    def test_jpeg_format_is_declared_and_decodes_as_jpeg(self) -> None:
        image = Image.new("RGB", (100, 140), "red")
        block = _compose_pages([(1, image)], image_dpi=150, image_format="jpeg", jpeg_quality=90)
        assert block["image_url"].startswith("data:image/jpeg;base64,")
        encoded = block["image_url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
            assert decoded.format == "JPEG"
            assert decoded.size == (100, 140)

    def test_jpeg_payload_is_smaller_than_png_for_a_noisy_page(self) -> None:
        # Solid colors favor PNG; real page scans are noisy, so seed pixel noise to make the
        # comparison representative of the actual corpus.
        rng = random.Random(0)
        image = Image.new("RGB", (200, 260))
        image.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(200 * 260)])

        as_png = _compose_pages([(1, image)], image_dpi=150, image_format="png", jpeg_quality=90)
        as_jpeg = _compose_pages([(1, image)], image_dpi=150, image_format="jpeg", jpeg_quality=90)
        assert len(as_jpeg["image_url"]) < len(as_png["image_url"])


class TestOpenPageImage:
    def test_no_resize_when_dpi_matches_source(self, tmp_path: Path) -> None:
        original = Image.new("RGB", (300, 400), "blue")
        path = tmp_path / "page.png"
        original.save(path)

        image = _open_page_image(path, source_dpi=150, image_dpi=150)
        assert image.size == (300, 400)

    def test_resizes_down_when_dpi_reduced(self, tmp_path: Path) -> None:
        original = Image.new("RGB", (300, 400), "blue")
        path = tmp_path / "page.png"
        original.save(path)

        image = _open_page_image(path, source_dpi=150, image_dpi=75)
        assert image.size == (150, 200)


@requires_example_manifest
class TestManifestTextBlocks:
    @property
    def manifest_path(self) -> Path:
        return _example_manifests[0]

    def test_returns_one_text_block(self) -> None:
        manifest = _load_manifest(self.manifest_path)
        blocks, pages_truncated = _manifest_text_blocks(manifest, max_pages=None)
        assert pages_truncated == 0
        assert len(blocks) == 1
        assert _block_type(blocks[0]) == "input_text"
        assert blocks[0]["text"].startswith("## Page 1")

    def test_max_pages_truncates_and_is_reported(self) -> None:
        manifest = _load_manifest(self.manifest_path)
        blocks, pages_truncated = _manifest_text_blocks(manifest, max_pages=1)
        assert pages_truncated >= 0
        assert len(blocks) <= 1


@requires_example_manifest
class TestRenderManifestImages:
    @property
    def manifest_path(self) -> Path:
        return _example_manifests[0]

    def test_renders_images_from_the_cached_screenshots(self) -> None:
        manifest = _load_manifest(self.manifest_path)
        delivery = DocumentDelivery(150, None)
        blocks, pages_truncated = _render_manifest_images(
            manifest, self.manifest_path.parent, delivery=delivery, max_pages=None, source_dpi=150
        )
        assert pages_truncated == 0
        assert blocks
        assert all(_block_type(b) == "input_image" for b in blocks)
        assert delivery.page_count > 0
        assert delivery.image_count > 0

    def test_max_pages_truncates_and_is_reported(self) -> None:
        manifest = _load_manifest(self.manifest_path)
        delivery = DocumentDelivery(150, None)
        _, pages_truncated = _render_manifest_images(
            manifest, self.manifest_path.parent, delivery=delivery, max_pages=1, source_dpi=150
        )
        assert delivery.page_count == 1
        assert pages_truncated >= 0

    def test_max_images_composites_pages(self) -> None:
        manifest = _load_manifest(self.manifest_path)
        delivery = DocumentDelivery(150, 1)
        blocks, _ = _render_manifest_images(
            manifest, self.manifest_path.parent, delivery=delivery, max_pages=None, source_dpi=150
        )
        # Capped at 1 image; all pages composited into it (up to the 4-page/image limit).
        assert len(blocks) == 1
        assert delivery.pages_per_image >= 1

    def test_dpi_backoff_shrinks_rendered_images(self) -> None:
        manifest = _load_manifest(self.manifest_path)
        full_delivery = DocumentDelivery(150, None)
        full_blocks, _ = _render_manifest_images(
            manifest, self.manifest_path.parent, delivery=full_delivery, max_pages=1, source_dpi=150
        )
        reduced_delivery = DocumentDelivery(75, None)
        reduced_blocks, _ = _render_manifest_images(
            manifest, self.manifest_path.parent, delivery=reduced_delivery, max_pages=1, source_dpi=150
        )
        assert len(reduced_blocks[0]["image_url"]) < len(full_blocks[0]["image_url"])


class TestDeliveryMetadataAndNotice:
    def test_image_cap_records_leading_page_coverage(self) -> None:
        manifest = {
            "pages": [
                {"page_number": page_number, "image": f"page_{page_number:04d}.png"} for page_number in range(1, 11)
            ]
        }
        delivery = DocumentDelivery(150, 2)
        with patch.object(gdp_pdf_agent_app, "_open_page_image", return_value=Image.new("RGB", (8, 10), "white")):
            blocks, pages_truncated = _render_manifest_images(
                manifest, Path("unused"), delivery=delivery, max_pages=None, source_dpi=150
            )

        assert pages_truncated == 0
        assert len(blocks) == 2
        assert delivery.pages_per_image == 4
        assert delivery.image_pages_covered == 8
        assert delivery.image_pages_omitted == 2
        assert delivery.image_coverage_end_page == 8

    def test_describes_composites_and_partial_image_coverage(self) -> None:
        delivery = DocumentDelivery(150, 2)
        delivery.pages_per_image = 4
        delivery.image_pages_covered = 8
        delivery.image_pages_omitted = 2
        delivery.image_coverage_end_page = 8

        blocks = _delivery_notice_blocks(delivery)

        assert len(blocks) == 1
        text = blocks[0]["text"]
        assert "up to 4 pages per image" in text
        assert "each cell is labeled with its original page number" in text
        assert "coverage stops after page 8" in text
        assert "2 later page image(s) were omitted" in text
        assert "remain available in the complete extracted document text" in text

    def test_is_empty_for_unmodified_full_image_coverage(self) -> None:
        delivery = DocumentDelivery(150, None)
        delivery.image_pages_covered = 10
        delivery.image_coverage_end_page = 10
        assert _delivery_notice_blocks(delivery) == []


class TestBuildTaskTextBlock:
    def test_reference_shape_with_prompt_and_extracted_text(self) -> None:
        block = _build_task_text_block(
            prompt_blocks=[{"type": "input_text", "text": "What is the cap?"}],
            delivery_notice_blocks=[],
            page_text_blocks=[{"type": "input_text", "text": "## Page 1\n\nfoo"}],
        )
        assert block["type"] == "input_text"
        text = block["text"]
        assert text.startswith("You are answering a task using text extracted from the source PDF.")
        assert "Task:\nWhat is the cap?" in text
        assert "Extracted PDF text:\n## Page 1" in text

    def test_folds_delivery_notice_between_task_and_extracted_text(self) -> None:
        block = _build_task_text_block(
            prompt_blocks=[{"type": "input_text", "text": "Q?"}],
            delivery_notice_blocks=[
                {"type": "input_text", "text": "<document_delivery>\ncomposites\n</document_delivery>"}
            ],
            page_text_blocks=[{"type": "input_text", "text": "## Page 1\n\nbody"}],
        )
        text = block["text"]
        # Delivery notice appears after the task but before the extracted text section.
        task_idx = text.index("Task:")
        notice_idx = text.index("composites")
        extract_idx = text.index("Extracted PDF text:")
        assert task_idx < notice_idx < extract_idx

    def test_omits_extracted_section_when_no_page_text(self) -> None:
        block = _build_task_text_block(
            prompt_blocks=[{"type": "input_text", "text": "Q?"}],
            delivery_notice_blocks=[],
            page_text_blocks=[],
        )
        text = block["text"]
        assert "Task:\nQ?" in text
        assert "Extracted PDF text" not in text


class TestDocumentDeliveryTextOnlyFallback:
    def test_falls_back_to_text_only_after_dpi_floor(self) -> None:
        delivery = DocumentDelivery(72, None)  # already at floor
        assert delivery.text_only_fallback is False
        # First adapt at floor flips to text-only fallback (retryable).
        assert delivery.adapt("context_length", None, min_dpi=72) is True
        assert delivery.text_only_fallback is True
        # A second adapt at floor -- with text-only already engaged -- gives up.
        assert delivery.adapt("context_length", None, min_dpi=72) is False


class TestRun:
    async def test_missing_manifest_raises_actionable_error(self) -> None:
        agent = _agent()
        with pytest.raises(FileNotFoundError, match="prepare --benchmark gdp_pdf"):
            await agent.run(
                SimpleNamespace(cookies={}), _run_request("test_media/documents/does-not-exist/manifest.json")
            )

    @requires_example_manifest
    async def test_source_dpi_mismatch_raises(self) -> None:
        agent = _agent(strip_images_from_output=False, dpi=72)  # manifest is rendered at 150
        with pytest.raises(ValueError, match="rendered at 150 DPI"):
            await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))

    @requires_example_manifest
    async def test_succeeds_on_first_try_and_verifies_once(self) -> None:
        agent = _agent(strip_images_from_output=False)
        agent.server_client.post = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                cookies={},
                data=(kw["json"] | {"reward": 1.0}) if kw["url_path"] == "/verify" else {},
            )
        )
        with (
            patch.object(GdpPdfAgent, "_create_episode", new=AsyncMock(return_value=(_response(), None, {}, {}))),
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            patch.object(gdp_pdf_agent_app, "get_response_json", new=AsyncMock(side_effect=lambda r: r.data)),
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))
        assert result.reward == 1.0
        assert result.document_delivery["image_dpi"] == 150
        assert result.document_delivery["rejected_attempts"] == []
        assert agent.server_client.post.await_count == 2  # seed_session, verify

    @requires_example_manifest
    async def test_adapts_dpi_on_rejection_then_succeeds(self) -> None:
        agent = _agent(strip_images_from_output=False)
        agent.server_client.post = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                cookies={},
                data=(kw["json"] | {"reward": 1.0}) if kw["url_path"] == "/verify" else {},
            )
        )
        attempts = []

        async def episode(params, **kwargs):
            attempts.append(len(attempts))
            if len(attempts) < 3:
                raise _error("payload too large")
            return _response(), None, {}, {}

        with (
            patch.object(GdpPdfAgent, "_create_episode", side_effect=episode),
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            patch.object(gdp_pdf_agent_app, "get_response_json", new=AsyncMock(side_effect=lambda r: r.data)),
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))
        assert len(attempts) == 3
        assert result.document_delivery["image_dpi"] == 96  # 150 -> 120 -> 96
        assert len(result.document_delivery["rejected_attempts"]) == 2

    @requires_example_manifest
    async def test_incomplete_status_triggers_dpi_backoff_when_regeneration_is_enabled(self) -> None:
        # Opt-in only (regenerate_on_incomplete_output): an incomplete answer is regenerated with a
        # smaller image footprint, detected separately from the ClientResponseError path above.
        agent = _agent(strip_images_from_output=False, include_images=True, regenerate_on_incomplete_output=True)
        agent.server_client.post = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                cookies={},
                data=(kw["json"] | {"reward": 1.0}) if kw["url_path"] == "/verify" else {},
            )
        )
        attempts = []

        async def episode(params, **kwargs):
            attempts.append(len(attempts))
            if len(attempts) < 3:
                return _incomplete_response(), None, {}, {}
            return _response(), None, {}, {}

        with (
            patch.object(GdpPdfAgent, "_create_episode", side_effect=episode),
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            patch.object(gdp_pdf_agent_app, "get_response_json", new=AsyncMock(side_effect=lambda r: r.data)),
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))
        assert len(attempts) == 3
        assert result.document_delivery["image_dpi"] == 96  # 150 -> 120 -> 96
        assert result.reward == 1.0  # the eventual complete response, not the truncated ones

    @requires_example_manifest
    async def test_incomplete_status_is_accepted_once_dpi_floor_reached(self) -> None:
        agent = _agent(
            strip_images_from_output=False, include_images=True, min_dpi=140, regenerate_on_incomplete_output=True
        )
        agent.server_client.post = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                cookies={},
                data=(kw["json"] | {"reward": 0.3}) if kw["url_path"] == "/verify" else {},
            )
        )
        with (
            patch.object(
                GdpPdfAgent, "_create_episode", new=AsyncMock(return_value=(_incomplete_response(), None, {}, {}))
            ),
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            patch.object(gdp_pdf_agent_app, "get_response_json", new=AsyncMock(side_effect=lambda r: r.data)),
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))
        # Floor reached on the very first backoff attempt (min_dpi=140 >= 150*0.8=120), so the
        # truncated response is accepted as-is rather than looping forever.
        assert result.reward == 0.3
        assert result.failure_reason is None or "output_truncated" not in (result.failure_reason or "")

    @requires_example_manifest
    async def test_incomplete_status_ignored_when_text_only(self) -> None:
        # No images means no DPI to reduce -- retrying would be pointless, so the truncated
        # response is accepted immediately without ever calling adapt().
        agent = _agent(strip_images_from_output=False, include_images=False)
        agent.server_client.post = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                cookies={},
                data=(kw["json"] | {"reward": 0.5}) if kw["url_path"] == "/verify" else {},
            )
        )
        with (
            patch.object(
                GdpPdfAgent, "_create_episode", new=AsyncMock(return_value=(_incomplete_response(), None, {}, {}))
            ) as call,
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            patch.object(gdp_pdf_agent_app, "get_response_json", new=AsyncMock(side_effect=lambda r: r.data)),
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))
        assert call.await_count == 1
        assert result.reward == 0.5

    @requires_example_manifest
    async def test_terminal_failure_scores_zero_without_raising(self) -> None:
        agent = _agent(strip_images_from_output=False, min_dpi=140)  # floor reached on first backoff
        agent.server_client.post = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                cookies={},
                data=(kw["json"] | {"reward": 0.0}) if kw["url_path"] == "/verify" else {},
            )
        )
        with (
            patch.object(GdpPdfAgent, "_create_episode", side_effect=_error("maximum context length")),
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            patch.object(gdp_pdf_agent_app, "get_response_json", new=AsyncMock(side_effect=lambda r: r.data)),
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))
        assert result.reward == 0.0
        assert result.response.output == []
        assert result.failure_reason == "GDP.pdf terminal input limit: context"
        # A terminal input limit is a scored attempt (the benchmark's own rule), not a masked failure.
        assert result.mask_sample is False
        assert result.failure_kind is None

    async def test_a_failed_policy_call_is_not_a_model_zero(self, tmp_path) -> None:
        """5xx, 429 and 401/403 replies and transport errors leave typed through the agent boundary: nothing is
        scored, so the row is masked and retried by Collection instead of becoming the model's zero."""
        for error in (_error("out of memory", status=500), _error("slow down", status=429), _error("no", status=401)):
            agent, manifest = _text_only_agent(tmp_path / str(error.status))
            agent.server_client.post = AsyncMock(return_value=SimpleNamespace(cookies={}, data={}))
            with pytest.raises(ClientResponseError) as raised:
                await _run_with_episodes(agent, manifest, [error])
            assert raised.value.status == error.status
            assert [c.kwargs["url_path"] for c in agent.server_client.post.await_args_list] == ["/seed_session"]

    async def test_a_confirmed_policy_timeout_is_typed_as_the_policys(self, tmp_path) -> None:
        operation_id = uuid4().hex
        attempts = [
            FailureEvent(
                boundary=FailureBoundary.MODEL,
                failure_kind="model_request_error",
                failure_reason=MODEL_RESPONSE_TIMEOUT_REASON,
                operation="model_request",
                operation_id=operation_id,
                attempt_id=uuid4().hex,
                details=ModelFailureDetails(request_upload_complete=True),
                foreign_error=SdkErrorDetails(exception_type="TimeoutError", message=""),
            )
            for _ in range(3)
        ]
        report = FailureReport(failure_id=attempts[-1].event_id, events={e.event_id: e for e in attempts})
        error = ClientResponseError(
            None, (), status=500, message="upstream failure", headers=FailureHeader.encode(report)
        )
        error.response_content = b"timed out"
        agent, manifest = _text_only_agent(tmp_path)
        agent.server_client.post = AsyncMock(return_value=SimpleNamespace(cookies={}, data={}))

        with pytest.raises(NgFailureEvent) as raised:
            await _run_with_episodes(agent, manifest, [error])

        assert (raised.value.failure_kind, raised.value.operation) == (
            "model_policy_response_timeout",
            "agent.policy_request",
        )

    async def test_a_real_input_limit_still_scores_zero(self, tmp_path) -> None:
        agent, manifest = _text_only_agent(tmp_path, min_dpi=140)
        agent.server_client.post = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                cookies={},
                data=(kw["json"] | {"reward": 0.0}) if kw["url_path"] == "/verify" else {},
            )
        )

        async def episode(*args, **kwargs):
            raise _error("maximum context length")

        result = await _run_with_episodes(agent, manifest, episode)

        assert (result.reward, result.mask_sample) == (0.0, False)
        assert result.failure_reason == "GDP.pdf terminal input limit: context"

    async def test_auth_and_rate_limit_errors_still_raise(self) -> None:
        # These are systemic, not per-document -- every other row would fail identically, so
        # this must surface loudly rather than silently zeroing the whole run one row at a time.
        agent = _agent(strip_images_from_output=False)
        agent.server_client.post = AsyncMock(return_value=SimpleNamespace(cookies={}, data={}))
        with (
            patch.object(GdpPdfAgent, "_create_episode", side_effect=_error("rate limited", status=429)) as call,
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            pytest.raises(ClientResponseError),
        ):
            await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))
        assert call.await_count == 1

    @requires_example_manifest
    async def test_images_are_stripped_from_the_result_when_enabled(self) -> None:
        agent = _agent(strip_images_from_output=True)
        agent.server_client.post = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                cookies={},
                data=(kw["json"] | {"reward": 1.0}) if kw["url_path"] == "/verify" else {},
            )
        )
        with (
            patch.object(GdpPdfAgent, "_create_episode", new=AsyncMock(return_value=(_response(), None, {}, {}))),
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            patch.object(gdp_pdf_agent_app, "get_response_json", new=AsyncMock(side_effect=lambda r: r.data)),
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))
        # `data` from /verify above didn't carry responses_create_params; nothing to assert on
        # content blocks here beyond the call succeeding without image payloads surviving.
        assert result.reward == 1.0

    async def test_terminal_input_limit_stays_a_scored_zero_with_a_text_only_manifest(self, tmp_path: Path) -> None:
        # The genuine limit case, without the optional example manifests: the row is still verified
        # and scored 0 with its input-limit failure_reason, as the benchmark's methodology prescribes.
        agent, manifest_relpath = _text_only_agent(tmp_path, min_dpi=140)
        agent.server_client.post = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                cookies={},
                data=(kw["json"] | {"reward": 0.0}) if kw["url_path"] == "/verify" else {},
            )
        )
        with (
            patch.object(GdpPdfAgent, "_create_episode", side_effect=_error("maximum context length")),
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            patch.object(gdp_pdf_agent_app, "get_response_json", new=AsyncMock(side_effect=lambda r: r.data)),
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(manifest_relpath))
        assert result.reward == 0.0
        assert result.response.output == []
        assert result.failure_reason == "GDP.pdf terminal input limit: context"
        assert result.mask_sample is False
        assert result.failure_kind is None
        assert agent.server_client.post.await_count == 2  # seed_session, verify

    @pytest.mark.parametrize("status", [500, 502, 503])
    async def test_verify_server_error_is_a_typed_failure_not_a_scored_zero(self, tmp_path: Path, status: int) -> None:
        # A /verify 5xx is the verifier's failure, not the model's: there is no verdict to score. It is
        # raised as the typed agent route failure the parent simple_agent raises, caused by the verifier's
        # reply, instead of a reward 0 labeled as an input limit. (A judge outage never gets here:
        # judge_failsafe answers it as a masked row.)
        agent, manifest_relpath = _text_only_agent(tmp_path)

        async def post(**kw):
            if kw["url_path"] == "/verify":
                raise ClientResponseError(None, (), status=status, message="verifier unavailable")
            return SimpleNamespace(cookies={})

        agent.server_client.post = AsyncMock(side_effect=post)
        with (
            patch.object(GdpPdfAgent, "_create_episode", new=AsyncMock(return_value=(_response(), None, {}, {}))),
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            RequestFailureLog.capture() as failure_log,
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(manifest_relpath))
        failure = next(
            event for event in failure_log.report.events.values() if event.failure_kind == AGENT_ROUTE_FAILED
        )
        assert failure.boundary == FailureBoundary.AGENT
        assert failure.failure_kind == AGENT_ROUTE_FAILED
        assert failure.operation == "verify"
        assert failure.stage == "verification"
        assert failure.foreign_error.http_status == status
        assert result.mask_sample is True
        assert result.response == _response()
        assert "terminal input limit" not in result.failure_reason

    async def test_verify_transport_failure_is_a_typed_failure_not_a_scored_zero(self, tmp_path: Path) -> None:
        # The verifier being unreachable is the same kind of failure as it answering 5xx.
        agent, manifest_relpath = _text_only_agent(tmp_path)

        async def post(**kw):
            if kw["url_path"] == "/verify":
                raise ConnectionError("resources server unreachable")
            return SimpleNamespace(cookies={})

        agent.server_client.post = AsyncMock(side_effect=post)
        with (
            patch.object(GdpPdfAgent, "_create_episode", new=AsyncMock(return_value=(_response(), None, {}, {}))),
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            RequestFailureLog.capture() as failure_log,
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(manifest_relpath))
        failure = next(
            event for event in failure_log.report.events.values() if event.failure_kind == AGENT_ROUTE_FAILED
        )
        assert failure.failure_kind == AGENT_ROUTE_FAILED
        assert failure.operation == "verify"
        assert failure.foreign_error.exception_type == "ConnectionError"
        assert result.mask_sample is True
        assert result.response == _response()

    async def test_verify_failure_after_a_terminal_input_limit_is_still_a_typed_failure(self, tmp_path: Path) -> None:
        # The terminal input limit would have scored 0, but without a /verify reply there is no verdict
        # at all, so the row is a failure rather than a zero that happens to be right.
        agent, manifest_relpath = _text_only_agent(tmp_path, min_dpi=140)

        async def post(**kw):
            if kw["url_path"] == "/verify":
                raise ClientResponseError(None, (), status=500, message="verifier unavailable")
            return SimpleNamespace(cookies={})

        agent.server_client.post = AsyncMock(side_effect=post)
        with (
            patch.object(GdpPdfAgent, "_create_episode", side_effect=_error("maximum context length")),
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            RequestFailureLog.capture() as failure_log,
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(manifest_relpath))
        failure = next(
            event for event in failure_log.report.events.values() if event.failure_kind == AGENT_ROUTE_FAILED
        )
        assert result.mask_sample is True
        assert failure.failure_kind == AGENT_ROUTE_FAILED
        assert failure.operation == "verify"


class TestMaskFailedAttempts:
    """mask_failed_attempts keeps attempts without a gradable answer out of training, never out of evaluation."""

    async def _run(
        self, tmp_path: Path, episode: Any, *, verify_raises: bool = False, **config: Any
    ) -> SimpleAgentVerifyResponse:
        agent, manifest = _text_only_agent(tmp_path, **config)

        async def post(**kw: Any) -> SimpleNamespace:
            if kw["url_path"] == "/verify" and verify_raises:
                raise RuntimeError("judge outage")
            return SimpleNamespace(
                cookies={}, data=(kw["json"] | {"reward": 0.0}) if kw["url_path"] == "/verify" else {}
            )

        agent.server_client.post = AsyncMock(side_effect=post)
        return await _run_with_episodes(agent, manifest, episode)

    async def test_off_by_default_so_evaluation_counts_the_zero(self, tmp_path: Path) -> None:
        result = await self._run(tmp_path, _error("maximum context length", status=400))
        assert result.reward == 0.0
        assert result.mask_sample is False
        assert result.failure_kind is None
        assert result.failure_reason == "GDP.pdf terminal input limit: context"
        assert "mask_sample" not in (result.model_extra.get("instance_config") or {})

    @pytest.mark.parametrize(
        ("error", "kind"),
        [
            (_error("maximum context length", status=400), "model_context_limit"),
            (_error("payload too large", status=413), "model_request_error"),
        ],
    )
    async def test_masks_an_attempt_whose_input_never_fit(
        self, tmp_path: Path, error: ClientResponseError, kind: str
    ) -> None:
        result = await self._run(tmp_path, error, mask_failed_attempts=True)
        assert result.reward == 0.0
        assert result.mask_sample is True
        assert result.failure_kind == kind
        assert result.model_extra["instance_config"]["mask_sample"] is True
        assert result.failure_reason.startswith("GDP.pdf terminal input limit: ")

    async def test_a_failed_verification_is_masked_either_way(self, tmp_path: Path) -> None:
        for mask in (False, True):
            result = await self._run(
                tmp_path / str(mask), [(_response(), None, {}, {})], verify_raises=True, mask_failed_attempts=mask
            )
            assert result.reward == 0.0
            assert result.mask_sample is True
            assert result.failure_kind is not None

    async def test_leaves_a_graded_answer_alone(self, tmp_path: Path) -> None:
        result = await self._run(tmp_path, [(_response(), None, {}, {})], mask_failed_attempts=True)
        assert result.mask_sample is False
        assert result.failure_kind is None
        assert result.failure_reason is None


class TestStripImageBlocks:
    def test_removes_image_blocks_and_records_the_gap(self) -> None:
        result = SimpleAgentVerifyResponse.model_validate(
            {
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "q"},
                                {"type": "input_image", "image_url": "data:image/png;base64,abc", "detail": "high"},
                            ],
                        }
                    ]
                },
                "response": {
                    "id": "resp",
                    "created_at": 0.0,
                    "model": "test",
                    "object": "response",
                    "output": [],
                    "parallel_tool_calls": False,
                    "tool_choice": "none",
                    "tools": [],
                },
                "reward": 1.0,
                "ng_trajectory": {"task_id": "t", "rollout_id": "r", "gaps": []},
            }
        )
        stripped = _strip_image_blocks(result)

        content = stripped.responses_create_params.input[0].content
        assert [_block_type(block) for block in content] == ["input_text"]
        gaps = stripped.model_extra["ng_trajectory"]["gaps"]
        assert {"code": "multimodal_history_redacted"} in gaps


class TestVerifyTransportRetry:
    @staticmethod
    def _agent(tmp_path: Path, failures: int):
        agent, manifest_relpath = _text_only_agent(tmp_path)
        verify_calls = []

        async def post(**kw):
            if kw["url_path"] == "/verify":
                verify_calls.append(kw)
                if len(verify_calls) <= failures:
                    raise ClientPayloadError("Response payload is not completed")
                return SimpleNamespace(cookies={}, data=kw["json"] | {"reward": 1.0})
            return SimpleNamespace(cookies={})

        agent.server_client.post = AsyncMock(side_effect=post)
        return agent, manifest_relpath, verify_calls

    async def _run(self, agent, manifest_relpath):
        with (
            patch.object(GdpPdfAgent, "_create_episode", new=AsyncMock(return_value=(_response(), None, {}, {}))),
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            patch.object(gdp_pdf_agent_app, "get_response_json", new=AsyncMock(side_effect=lambda r: r.data)),
            patch.object(verify_call, "_stateless_verifier", new=AsyncMock(return_value=True)),
            patch.object(verify_call, "_retry_wait_s", return_value=0),
            RequestFailureLog.capture() as failure_log,
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(manifest_relpath))
        return result, failure_log

    async def test_a_payload_error_on_verify_is_retried_and_the_row_is_not_masked(self, tmp_path: Path) -> None:
        agent, manifest_relpath, verify_calls = self._agent(tmp_path, failures=2)
        result, failure_log = await self._run(agent, manifest_relpath)
        assert len(verify_calls) == 3
        assert result.reward == 1.0 and not result.mask_sample and result.failure_kind is None
        # Both lost replies stay observations of the scored row.
        assert [e.operation for e in failure_log.report.events.values()].count("verify") == 2

    async def test_a_persistent_payload_error_still_masks_after_the_bound(self, tmp_path: Path) -> None:
        agent, manifest_relpath, verify_calls = self._agent(tmp_path, failures=5)
        result, _ = await self._run(agent, manifest_relpath)
        assert len(verify_calls) == 3
        assert result.mask_sample is True and result.reward == 0.0 and result.response == _response()
        assert result.failure_kind == AGENT_ROUTE_FAILED
        assert result.failure_reason == "Agent operation verify failed: ClientPayloadError"
