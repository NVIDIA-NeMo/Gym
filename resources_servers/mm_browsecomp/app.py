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
"""MM-BrowseComp: the BrowseComp search harness, a view_image tool and the upstream checklist judge.

Search, browse, the read-only workspace shell, the contamination guard and the excluded domains come
from browsecomp_advanced_harness unchanged. This server adds what the multimodal questions need: a
`view_image` tool that returns a web image to a vision model, and upstream's judge, which grades the
final answer and a per-question checklist.
"""

import asyncio
import base64
import io
import ipaddress
import re
import socket
from contextlib import asynccontextmanager
from statistics import fmean
from time import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from aiohttp import ClientConnectorError, ClientSession, ClientTimeout, TCPConnector
from aiohttp.abc import ResolveResult
from aiohttp.resolver import ThreadedResolver
from fastapi import FastAPI, Request
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import NeMoGymEasyInputMessage
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.browsecomp_advanced_harness.app import (
    JudgeEvaluation,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
    TavilySearchVerifyRequest,
    TavilySearchVerifyResponse,
    _is_contaminated,
    _is_contaminated_url,
    _judge_backoff_s,
    _last_assistant_text,
    _sum_provider_retry_counts,
)
from resources_servers.mm_browsecomp.judge_prompt import build_judge_prompt, parse_judge_response


_REDIRECT_STATUSES = (301, 302, 303, 307, 308)
_MAX_REDIRECTS = 3
# Decoded size limit (after JPEG draft scaling): 25M pixels is about 100 MB as RGBA.
_MAX_IMAGE_PIXELS = 25_000_000
# Raster formats a web image comes in; Pillow would otherwise also hand EPS to Ghostscript.
_IMAGE_FORMATS = ("JPEG", "PNG", "GIF", "WEBP", "BMP")
_IMAGE_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NeMo-Gym-MM-BrowseComp/1.0)",
    "Accept": "image/*,*/*;q=0.5",
}
_EXACT_ANSWER_RE = re.compile(r"Exact Answer:\s*(.+)", re.IGNORECASE)


class MMBrowseCompResourcesServerConfig(TavilySearchResourcesServerConfig):
    view_image_max_bytes: int = Field(default=20_000_000, gt=0)
    # Long edge of the image the model receives. The NM4 vLLM processor downsizes to 1,036 px by default.
    view_image_max_side: int = Field(default=1536, gt=0)
    view_image_timeout_s: float = Field(default=30.0, gt=0)
    # Decodes run in threads; this many at once bounds their memory and keeps executor threads free.
    view_image_max_concurrent_decodes: int = Field(default=8, gt=0)

    @model_validator(mode="after")
    def _require_judge(self) -> "MMBrowseCompResourcesServerConfig":
        if not self.use_judge:
            raise ValueError("mm_browsecomp grades with the checklist judge only; use_judge must stay true")
        return self


class ViewImageRequest(BaseModel):
    url: Optional[str] = None


class ViewImageResponse(BaseModel):
    results_string: str
    # Data URLs; browsecomp_agent with tool_images on shows them to the model.
    images: List[str] = Field(default_factory=list)


class MMBrowseCompVerifyRequest(TavilySearchVerifyRequest):
    checklist: List[str]
    in_paper_set: bool = False
    needs_video: bool = False


class ChecklistJudgement(BaseModel):
    checklist_correct: Optional[int] = None
    checklist_total: Optional[int] = None
    checklist_score: Optional[float] = None
    strict_correct: bool = False


class MMBrowseCompVerifyResponse(MMBrowseCompVerifyRequest, TavilySearchVerifyResponse):
    # Nested on purpose: a top-level field would become mean/<field> over judged rows only, but a timed-out
    # or failed rollout has no judgement and must count as wrong. compute_metrics averages these instead.
    checklist_judgement: ChecklistJudgement


class ImageFetchError(Exception):
    """An image the model asked for could not be shown; the message is safe to return to the model."""


class NonPublicAddressError(OSError):
    """A host name resolved to an address outside the public internet."""


def _is_public(address: str) -> bool:
    return ipaddress.ip_address(address.split("%")[0]).is_global


class PublicOnlyResolver(ThreadedResolver):
    """aiohttp's default resolver, refusing any answer that is not a public address.

    The check runs in the lookup of the connection that is opened, on every retry and redirect, so a
    name cannot pass a separate check and then rebind to a cluster address (DNS rebinding).
    """

    async def resolve(
        self, host: str, port: int = 0, family: socket.AddressFamily = socket.AF_INET
    ) -> List[ResolveResult]:
        results = await super().resolve(host, port, family)
        if not results or not all(_is_public(result["host"]) for result in results):
            raise NonPublicAddressError(f"{host} does not resolve to a public internet address")
        return results


def _encode_image(data: bytes, max_side: int) -> tuple[str, tuple[int, int]]:
    """Decode ``data`` (the first frame of an animation), flatten alpha onto white, fit it in ``max_side``
    and return it as a JPEG data URL with its new size."""
    try:
        with Image.open(io.BytesIO(data), formats=_IMAGE_FORMATS) as image:
            # A JPEG decodes straight at 1/2, 1/4 or 1/8 scale when that still covers max_side.
            image.draft("RGB", (max_side, max_side))
            if image.width * image.height > _MAX_IMAGE_PIXELS:
                raise ImageFetchError(f"the image is too large to decode ({image.width}x{image.height})")
            image.load()
            if image.mode in ("RGBA", "LA", "P"):
                rgba = image.convert("RGBA")
                rgb = Image.new("RGB", rgba.size, (255, 255, 255))
                rgb.paste(rgba, mask=rgba.getchannel("A"))
            else:
                rgb = image.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError) as e:
        raise ImageFetchError("the URL did not return an image in a supported format") from e
    except OSError as e:
        raise ImageFetchError(f"the image could not be decoded: {e}") from e
    rgb.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(), rgb.size


class MMBrowseCompResourcesServer(TavilySearchResourcesServer):
    config: MMBrowseCompResourcesServerConfig

    # Image downloads use their own session: Gym's shared client resolves names without the
    # public-address check, and must not read proxy settings for these requests either.
    _image_session: Optional[ClientSession] = PrivateAttr(default=None)
    _decode_slots: Optional[asyncio.Semaphore] = PrivateAttr(default=None)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/view_image")(self.view_image)
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app):
            async with parent_lifespan(app) as state:
                yield state
            if self._image_session is not None:
                await self._image_session.close()

        app.router.lifespan_context = lifespan
        return app

    def _get_image_session(self) -> ClientSession:
        if self._image_session is None:
            self._image_session = ClientSession(
                connector=TCPConnector(resolver=PublicOnlyResolver(), use_dns_cache=False),
                timeout=ClientTimeout(total=self.config.view_image_timeout_s),
                headers=_IMAGE_REQUEST_HEADERS,
            )
        return self._image_session

    async def view_image(self, request: Request, body: ViewImageRequest) -> ViewImageResponse:
        metrics = self._session_id_to_metrics[request.session[SESSION_ID_KEY]]
        url = (body.url or "").strip()
        start = time()
        try:
            async with asyncio.timeout(self.config.view_image_timeout_s):
                data = await self._download_image(url)
            if self._decode_slots is None:
                self._decode_slots = asyncio.Semaphore(self.config.view_image_max_concurrent_decodes)
            async with self._decode_slots:
                data_url, (width, height) = await asyncio.get_running_loop().run_in_executor(
                    None, _encode_image, data, self.config.view_image_max_side
                )
        except (ImageFetchError, TimeoutError, ClientConnectorError) as e:
            self._record_call(metrics, "view_image", "http", "error", start)
            if isinstance(e, ImageFetchError):
                reason = str(e)
            elif isinstance(e, TimeoutError):
                reason = "the download timed out"
            elif isinstance(e.os_error, NonPublicAddressError):
                reason = "the host is not a public internet address"
            else:
                reason = "the host could not be reached"
            print(f"[mm_browsecomp][tool_fail][view_image] url={url[:200]!r} reason={reason}", flush=True)
            return ViewImageResponse(results_string=f"Failed to load image from {url}: {reason}")
        except Exception as e:
            # A broken connection or TLS failure is the website's problem, not the run's: tell the model.
            self._record_call(metrics, "view_image", "http", "error", start)
            print(f"[mm_browsecomp][tool_fail][view_image] url={url[:200]!r} error={e!r}", flush=True)
            return ViewImageResponse(results_string=f"Failed to load image from {url}: {type(e).__name__}")
        self._record_call(metrics, "view_image", "http", "success", start)
        return ViewImageResponse(
            results_string=f"[Image]: {url}\n[Size]: {width}x{height} pixels. The image is attached below.",
            images=[data_url],
        )

    def _check_image_url(self, url: str) -> None:
        """Refuse what the harness must not fetch: non-HTTP URLs, excluded or benchmark-mirror sites, and
        non-public IP literals. Host names are checked by PublicOnlyResolver when the connection opens
        (aiohttp does not resolve an IP literal, so it is checked here)."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ImageFetchError("only http(s) image URLs are supported")
        if self._is_url_excluded(url) or _is_contaminated(url) or _is_contaminated_url(url):
            # Same answer as an unreachable site: naming the reason would point at the answer key.
            raise ImageFetchError("the image could not be retrieved")
        try:
            is_public = _is_public(parsed.hostname)
        except ValueError:
            return
        if not is_public:
            raise ImageFetchError("the host is not a public internet address")

    async def _download_image(self, url: str) -> bytes:
        max_bytes = self.config.view_image_max_bytes
        session = self._get_image_session()
        for _ in range(_MAX_REDIRECTS + 1):
            self._check_image_url(url)
            async with session.get(url, allow_redirects=False) as response:
                if response.status in _REDIRECT_STATUSES and response.headers.get("Location"):
                    url = urljoin(url, response.headers["Location"])
                    continue
                if response.status >= 400:
                    raise ImageFetchError(f"the server answered HTTP {response.status}")
                if (response.content_length or 0) > max_bytes:
                    raise ImageFetchError(f"the image is larger than {max_bytes // 1_000_000} MB")
                data = bytearray()
                async for chunk in response.content.iter_chunked(1 << 16):
                    data += chunk
                    if len(data) > max_bytes:
                        raise ImageFetchError(f"the image is larger than {max_bytes // 1_000_000} MB")
                return bytes(data)
        raise ImageFetchError("too many redirects")

    async def verify(self, request: Request, body: MMBrowseCompVerifyRequest) -> MMBrowseCompVerifyResponse:
        answer = re.sub(r"<think>.*?</think>", "", _last_assistant_text(body.response), flags=re.DOTALL).strip()
        evaluation, judgement, judge_error = await self._judge_with_checklist(body, answer)

        num_tool_calls = getattr(body.response, "num_tool_calls", None)
        if num_tool_calls is None:
            num_tool_calls = sum(o.type == "function_call" for o in body.response.output)
        session_metrics = self._session_id_to_metrics[request.session[SESSION_ID_KEY]]
        num_provider_429s, num_provider_other_retries = _sum_provider_retry_counts(session_metrics)
        verify_response = MMBrowseCompVerifyResponse(
            **body.model_dump(),
            **evaluation.model_dump(),
            checklist_judgement=judgement,
            num_tool_calls=num_tool_calls,
            reset_count=getattr(body.response, "reset_count", 0) or 0,
            metrics=session_metrics,
            num_provider_429s=num_provider_429s,
            num_provider_other_retries=num_provider_other_retries,
        )
        if self.config.workspace == "per_session":
            self._cleanup_workspace(request.session[SESSION_ID_KEY])
        if judge_error is not None:
            raise JudgeError(judge_error)
        return verify_response

    async def _judge_with_checklist(
        self, body: MMBrowseCompVerifyRequest, answer: str
    ) -> tuple[JudgeEvaluation, ChecklistJudgement, Optional[str]]:
        """Grade with upstream's prompt; the retry policy is the BrowseComp judge's."""
        params = self.config.judge_responses_create_params.model_copy(deep=True)
        # As the BrowseComp judge: room for a reasoning judge, temperature 1.0. Upstream sets no
        # temperature, so its gpt-4o judge also sampled at the API default of 1.0.
        params.max_output_tokens = 10240
        params.temperature = 1.0
        prompt = build_judge_prompt(body.question, answer, body.ground_truth, body.checklist)
        params.input = [NeMoGymEasyInputMessage(role="user", content=prompt)]
        exact_answers = _EXACT_ANSWER_RE.findall(answer)
        extracted = exact_answers[-1].strip() if exact_answers else ""

        judge_response = None
        last_error: Optional[str] = None
        for attempt in range(self.JUDGE_MAX_ATTEMPTS):
            try:
                judge_response, text = await self._call_judge(params)
            except JudgeError as e:
                last_error = str(e)
                print(f"[mm_browsecomp][judge] attempt={attempt + 1} error={last_error[:200]}", flush=True)
                await asyncio.sleep(_judge_backoff_s(attempt))
                continue
            verdict = parse_judge_response(text)
            if verdict is not None:
                evaluation = JudgeEvaluation(
                    judge_response_create_params=params,
                    reasoning=text,
                    extracted_final_answer=extracted,
                    reward=1.0 if verdict.correct else 0.0,
                    judge_response=judge_response,
                )
                judgement = ChecklistJudgement(
                    checklist_correct=verdict.checklist_correct,
                    checklist_total=verdict.checklist_total,
                    checklist_score=verdict.checklist_score,
                    strict_correct=verdict.strict_correct,
                )
                return evaluation, judgement, None
            last_error = f"unparseable judge verdict: {text[:200]!r}"
            print(f"[mm_browsecomp][judge] attempt={attempt + 1} parse_error raw={text[:200]!r}", flush=True)

        # As in BrowseComp: a judge that answered without a verdict did grade the row (0.0); only a judge
        # that never answered is a judge failure, which the failsafe masks.
        evaluation = JudgeEvaluation(
            judge_response_create_params=params,
            reasoning="",
            extracted_final_answer=extracted,
            reward=0.0,
            judge_response=judge_response,
        )
        error = None if judge_response is not None else (last_error or "judge exhausted")
        return evaluation, ChecklistJudgement(), error

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Upstream's SA and AVG CS for the full set, the paper's 224 questions and the questions with no
        video checklist item. A timed-out or failed rollout has no judgement and counts as 0 in both; as
        upstream, AVG CS leaves out judged rollouts whose CHECKLIST_SCORE did not parse."""
        rows = [row for task in tasks for row in task]
        subsets = {
            "": rows,
            "paper_224/": [r for r in rows if r.get("in_paper_set")],
            "no_video/": [r for r in rows if not r.get("needs_video")],
        }
        metrics: Dict[str, Any] = {}
        for prefix, subset in subsets.items():
            if not subset:
                continue
            judgements = [r.get("checklist_judgement") for r in subset]
            if prefix:
                metrics[f"{prefix}reward"] = fmean(float(r.get("reward") or 0.0) for r in subset)
                metrics[f"{prefix}num_rollouts"] = len(subset)
            metrics[f"{prefix}strict_accuracy"] = fmean(
                float(bool((j or {}).get("strict_correct"))) for j in judgements
            )
            scores = [0.0 if j is None else j["checklist_score"] for j in judgements]
            scores = [float(score) for score in scores if score is not None]
            metrics[f"{prefix}checklist_score"] = fmean(scores) if scores else 0.0
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key = super().get_key_metrics(agent_metrics)
        for name in ("strict_accuracy", "checklist_score", "paper_224/reward", "no_video/reward"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        return key


if __name__ == "__main__":
    MMBrowseCompResourcesServer.run_webserver()
