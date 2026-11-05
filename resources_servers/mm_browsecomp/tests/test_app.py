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
import base64
import io
import socket
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from aiohttp import web
from PIL import Image
from pytest import approx

from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.mm_browsecomp.app import (
    ImageFetchError,
    MMBrowseCompResourcesServer,
    MMBrowseCompResourcesServerConfig,
    MMBrowseCompVerifyRequest,
    NonPublicAddressError,
    PublicOnlyResolver,
    ViewImageRequest,
    _encode_image,
)
from resources_servers.mm_browsecomp.judge_prompt import build_judge_prompt, parse_judge_response


_EXCLUDE_FILE = str(
    Path(__file__).parents[2] / "browsecomp_advanced_harness" / "tests" / "dummy_exclude_domains_file.json"
)


def _config(**overrides) -> MMBrowseCompResourcesServerConfig:
    return MMBrowseCompResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        search_provider="exa",
        exa_api_key="test_api_key",  # pragma: allowlist secret
        exclude_domains_file_path=_EXCLUDE_FILE,
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        **overrides,
    )


def _request() -> MagicMock:
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "session"}
    return request


def _response(text: str, model: str = "policy") -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model=model,
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
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _server_with_judge(judge_text: str) -> tuple[MMBrowseCompResourcesServer, AsyncMock]:
    server_client = MagicMock(spec=ServerClient)
    reply = MagicMock()
    reply.json = AsyncMock(return_value=_response(judge_text, model="judge").model_dump())
    reply.read = AsyncMock(return_value=orjson.dumps(reply.json.return_value))
    server_client.post = AsyncMock(return_value=reply)
    return MMBrowseCompResourcesServer(config=_config(), server_client=server_client), server_client.post


def _verify_request(answer: str, **fields) -> MMBrowseCompVerifyRequest:
    return MMBrowseCompVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=_response(answer),
        question="Please answer the following question and also provide your problem-solving roadmap. "
        "Question: Which tower is in the photo?",
        ground_truth="Eiffel Tower",
        checklist=["Identify the city", "Identify the tower", "Name its architect firm"],
        **fields,
    )


def _png(width: int, height: int, mode: str = "RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (width, height)).save(buffer, format="PNG")
    return buffer.getvalue()


class TestJudgePrompt:
    def test_judge_sees_only_text_after_question_marker_and_answer_tail(self) -> None:
        prompt = build_judge_prompt("Preamble. Question: Which tower?", "x" * 30000 + "END", "Eiffel", ["a", "b"])
        assert "--- USER QUESTION ---\nWhich tower?" in prompt
        assert "Preamble" not in prompt
        assert "x" * 25001 not in prompt and ("x" * 24997 + "END") in prompt
        assert "1. a\n2. b" in prompt
        assert "CHECKLIST_SCORE: 1/2" in prompt

    def test_parse_full_verdict(self) -> None:
        verdict = parse_judge_response("CHECKLIST_SCORE: 3/3\nCHECKLIST_RESULT: [1,1,1]\nOVERALL_CORRECTNESS: YES")
        assert verdict.correct and verdict.strict_correct
        assert verdict.checklist_score == approx(1.0)

    def test_parse_uses_last_match_and_markdown(self) -> None:
        verdict = parse_judge_response(
            "Example: OVERALL_CORRECTNESS: YES\nCHECKLIST_SCORE: 1/2\n**OVERALL_CORRECTNESS:** NO"
        )
        assert verdict.correct is False
        assert verdict.checklist_score == approx(0.5)

    def test_correct_without_full_checklist_is_not_strict(self) -> None:
        verdict = parse_judge_response("CHECKLIST_SCORE: 2/3\nOVERALL_CORRECTNESS: YES")
        assert verdict.correct and not verdict.strict_correct

    def test_missing_checklist_score_leaves_score_unknown(self) -> None:
        verdict = parse_judge_response("CHECKLIST_SCORE: N/A\nOVERALL_CORRECTNESS: YES")
        assert verdict.checklist_score is None and not verdict.strict_correct

    def test_no_verdict_is_unparseable(self) -> None:
        assert parse_judge_response("CHECKLIST_SCORE: 2/3") is None


class TestVerify:
    async def test_correct_answer_with_full_checklist(self) -> None:
        server, post = _server_with_judge("CHECKLIST_SCORE: 3/3\nCHECKLIST_RESULT: [1,1,1]\nOVERALL_CORRECTNESS: YES")
        result = await server.verify(_request(), _verify_request("Explanation: ...\nExact Answer: Eiffel Tower"))
        assert result.reward == approx(1.0)
        assert result.extracted_final_answer == "Eiffel Tower"
        assert result.checklist_judgement.strict_correct is True
        assert result.checklist_judgement.checklist_score == approx(1.0)
        judge_input = post.call_args.kwargs["json"].input[0].content
        assert "--- REFERENCE CHECKLIST ---\n1. Identify the city" in judge_input
        assert "Exact Answer: Eiffel Tower" in judge_input

    async def test_wrong_answer(self) -> None:
        server, _ = _server_with_judge("CHECKLIST_SCORE: 1/3\nCHECKLIST_RESULT: [1,0,0]\nOVERALL_CORRECTNESS: NO")
        result = await server.verify(_request(), _verify_request("Exact Answer: Tokyo Tower"))
        assert result.reward == approx(0.0)
        assert result.checklist_judgement.checklist_score == approx(1 / 3)

    async def test_think_block_is_not_graded(self) -> None:
        server, post = _server_with_judge("CHECKLIST_SCORE: 0/3\nOVERALL_CORRECTNESS: NO")
        await server.verify(_request(), _verify_request("<think>secret plan</think>Exact Answer: Tokyo Tower"))
        assert "secret plan" not in post.call_args.kwargs["json"].input[0].content

    async def test_unparseable_judge_scores_zero_without_error(self) -> None:
        server, post = _server_with_judge("I cannot decide.")
        with patch("resources_servers.mm_browsecomp.app.asyncio.sleep", new=AsyncMock()):
            result = await server.verify(_request(), _verify_request("Exact Answer: Eiffel Tower"))
        assert result.reward == approx(0.0)
        assert result.checklist_judgement.checklist_score is None
        assert post.await_count == server.JUDGE_MAX_ATTEMPTS

    async def test_judge_that_never_answers_raises_judge_error(self) -> None:
        server, post = _server_with_judge("")
        post.side_effect = JudgeError("judge down")
        with patch("resources_servers.mm_browsecomp.app.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(JudgeError):
                await server.verify(_request(), _verify_request("Exact Answer: Eiffel Tower"))


class TestMetrics:
    def test_timed_out_rows_count_as_wrong_in_every_metric(self) -> None:
        server, _ = _server_with_judge("")
        judged = {"strict_correct": True, "checklist_score": 1.0}
        unparsed = {"strict_correct": False, "checklist_score": None}
        partial = {"strict_correct": False, "checklist_score": 0.5}
        tasks = [
            [{"reward": 1.0, "in_paper_set": True, "needs_video": False, "checklist_judgement": judged}],
            [{"reward": 0.0, "in_paper_set": True, "needs_video": True, "timed_out": 1}],
            [{"reward": 0.0, "in_paper_set": False, "needs_video": False, "checklist_judgement": unparsed}],
            [{"reward": 0.0, "in_paper_set": False, "needs_video": False, "checklist_judgement": partial}],
        ]
        metrics = server.compute_metrics(tasks)
        assert metrics["strict_accuracy"] == approx(1 / 4)
        # Judged without a parseable CHECKLIST_SCORE: left out, as upstream; timed out: 0.
        assert metrics["checklist_score"] == approx((1.0 + 0.0 + 0.5) / 3)
        assert metrics["paper_224/reward"] == approx(0.5)
        assert metrics["paper_224/num_rollouts"] == 2
        assert metrics["no_video/reward"] == approx(1 / 3)
        assert metrics["no_video/strict_accuracy"] == approx(1 / 3)
        assert metrics["no_video/checklist_score"] == approx(0.75)

    def test_key_metrics_include_strict_accuracy(self) -> None:
        server, _ = _server_with_judge("")
        key = server.get_key_metrics({"mean/reward": 0.2, "strict_accuracy": 0.1, "paper_224/reward": 0.3})
        assert key == {"mean/reward": 0.2, "strict_accuracy": 0.1, "paper_224/reward": 0.3}


class TestViewImage:
    """view_image against a real local HTTP server and the real resolver (no mocked fetches)."""

    @pytest.fixture
    async def site(self):
        """A local server: /a.png (64x32), /big (streamed, no Content-Length), /redirect-private."""
        routes = web.RouteTableDef()
        hits = []

        @routes.get("/a.png")
        async def image(request):
            hits.append(request.path)
            return web.Response(body=_png(64, 32), content_type="image/png")

        @routes.get("/big")
        async def big(request):
            hits.append(request.path)
            response = web.StreamResponse()
            response.enable_chunked_encoding()
            await response.prepare(request)
            for _ in range(4):
                await response.write(b"\x89PNG" + b"0" * 4096)
            await response.write_eof()
            return response

        @routes.get("/redirect-private")
        async def redirect(request):
            hits.append(request.path)
            raise web.HTTPFound("http://10.0.0.5/latest")

        app = web.Application()
        app.add_routes(routes)
        runner = web.AppRunner(app)
        await runner.setup()
        tcp_site = web.TCPSite(runner, "127.0.0.1", 0)
        await tcp_site.start()
        port = tcp_site._server.sockets[0].getsockname()[1]
        yield f"http://localhost:{port}", hits
        await runner.cleanup()

    def test_encode_flattens_alpha_and_fits_max_side(self) -> None:
        data_url, size = _encode_image(_png(3000, 1000, mode="RGBA"), max_side=1536)
        assert size == (1536, 512)
        decoded = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))
        assert decoded.format == "JPEG" and decoded.mode == "RGB"

    def test_encode_rejects_non_images(self) -> None:
        with pytest.raises(ImageFetchError):
            _encode_image(b"<html>not an image</html>", max_side=1536)

    @pytest.mark.parametrize(
        "url, reason",
        [
            ("file:///etc/passwd", "only http(s) image URLs are supported"),
            ("ftp://example.com/a.png", "only http(s) image URLs are supported"),
            ("https://sub.blacklisteddomain.com/a.png", "the image could not be retrieved"),
            (
                "https://raw.githubusercontent.com/MMBrowseComp/MM-BrowseComp/main/MMBC_images/2.png",
                "the image could not be retrieved",
            ),
            ("https://huggingface.co/datasets/someone/mirror/resolve/main/a.png", "the image could not be retrieved"),
            ("http://10.49.0.7/a.png", "the host is not a public internet address"),
            ("http://[::1]/a.png", "the host is not a public internet address"),
        ],
    )
    async def test_refused_urls_name_their_reason(self, url: str, reason: str) -> None:
        server, _ = _server_with_judge("")
        result = await server.view_image(_request(), ViewImageRequest(url=url))
        assert result.images == []
        assert result.results_string == f"Failed to load image from {url}: {reason}"

    async def test_a_name_resolving_to_loopback_is_never_connected(self, site) -> None:
        # The connection's own lookup refuses it, so no earlier check can be rebound around.
        base, hits = site
        server, _ = _server_with_judge("")
        result = await server.view_image(_request(), ViewImageRequest(url=f"{base}/a.png"))
        assert "the host is not a public internet address" in result.results_string
        assert hits == []

    async def test_resolver_refuses_any_non_public_answer(self) -> None:
        resolver = PublicOnlyResolver()
        answers = [("93.184.216.34", socket.AF_INET), ("10.0.0.5", socket.AF_INET)]
        infos = [(family, socket.SOCK_STREAM, 6, "", (host, 443)) for host, family in answers]
        with patch.object(resolver._loop, "getaddrinfo", new=AsyncMock(return_value=infos)):
            with pytest.raises(NonPublicAddressError):
                await resolver.resolve("mixed.example.com", 443)

    async def test_public_image_is_returned_as_data_url(self, site) -> None:
        base, _ = site
        server, _ = _server_with_judge("")
        with patch("resources_servers.mm_browsecomp.app._is_public", return_value=True):
            result = await server.view_image(_request(), ViewImageRequest(url=f"{base}/a.png"))
        assert result.images[0].startswith("data:image/jpeg;base64,")
        assert "64x32 pixels" in result.results_string

    async def test_redirect_to_a_private_literal_is_refused(self, site) -> None:
        base, hits = site
        server, _ = _server_with_judge("")
        with patch("resources_servers.mm_browsecomp.app._is_public", side_effect=lambda a: not a.startswith("10.")):
            result = await server.view_image(_request(), ViewImageRequest(url=f"{base}/redirect-private"))
        assert hits == ["/redirect-private"]
        assert "the host is not a public internet address" in result.results_string

    async def test_streamed_image_over_the_size_cap_is_refused(self, site) -> None:
        base, _ = site
        server = MMBrowseCompResourcesServer(
            config=_config(view_image_max_bytes=10_000), server_client=MagicMock(spec=ServerClient)
        )
        with patch("resources_servers.mm_browsecomp.app._is_public", return_value=True):
            result = await server.view_image(_request(), ViewImageRequest(url=f"{base}/big"))
        assert result.images == [] and "larger than" in result.results_string


class TestDecodeLimits:
    def test_formats_outside_the_raster_list_are_refused(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buffer, format="TIFF")
        with pytest.raises(ImageFetchError, match="supported format"):
            _encode_image(buffer.getvalue(), max_side=1536)

    def test_oversized_png_is_refused_before_decoding(self) -> None:
        # A PNG header that claims 6000x6000 pixels; the pixel data is never read.
        buffer = io.BytesIO()
        Image.new("1", (6000, 6000)).save(buffer, format="PNG")
        with pytest.raises(ImageFetchError, match="too large"):
            _encode_image(buffer.getvalue(), max_side=1536)

    def test_large_jpeg_decodes_at_reduced_scale(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (8000, 4000), (10, 200, 10)).save(buffer, format="JPEG")
        _, size = _encode_image(buffer.getvalue(), max_side=1536)
        assert size == (1536, 768)


def test_config_refuses_grading_without_the_judge() -> None:
    with pytest.raises(ValueError, match="use_judge"):
        _config(use_judge=False)
