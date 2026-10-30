# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded retrieval through Gym's shared aiohttp transport; explicit Lens upload opt-in.

The web-search provider is configurable (search_provider):
    serper   google.serper.dev            google_search + google_lens_search (Google results)
    serpapi  serpapi.com (engine=google / google_lens), answers mapped to Serper's shape
    tavily   api.tavily.com/search        google_search only (no reverse image search)
    exa      api.exa.ai/search            google_search only (no reverse image search)
google_lens_search can use its own provider (lens_provider: serper | serpapi | none; default
"auto" = the search provider when it has Lens, else none), e.g. Tavily text search with SerpAPI's
Google Lens: search_provider=tavily, lens_provider=serpapi.

cache_root (optional) puts a shared local cache in front of every live call (search_cache.py:
flat files + FTS5 fuzzy index over earlier queries, Lens results keyed by image pixels, fetched
pages keyed by URL). A hit is served without calling any provider; a miss calls the provider and,
with cache_write, stores the answer for every later rollout. Tool outputs carry "cache":
"exact" | "fuzzy" | "miss".
fetch_webpage reads pages through Jina Reader (fetch_provider=jina, default) or Exa's /contents
endpoint (fetch_provider=exa, so Exa alone serves text search and page reads). Every provider's
search answer is reduced to the same Serper-shaped object ({"organic": [{title, link, snippet}],
...}) so the model sees one format whatever the backend.

Benchmark-contamination guard (as in browsecomp_advanced_harness): a result that quotes the
Agentic-MME benchmark itself (dataset mirror, repo, paper page) hands the model the answer key.
Results matching contamination_patterns (case-insensitive substring over the whole serialized
result) are dropped per item on every read, from the cache or live; the cache keeps the raw
provider answer. A response with every item dropped, or a contaminated page, is withheld.
"""

import asyncio
import io
import ipaddress
import json
from typing import Any, Literal
from urllib.parse import urlsplit

from aiohttp import FormData
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from nemo_gym.server_utils import request
from responses_api_agents.agentic_mme_agent.search_cache import shared_cache
from responses_api_agents.agentic_mme_agent.tools import ImageWorkspace, image_data_url


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=2000)
    gl: str = Field(default="us", pattern=r"^[a-z]{2}$")
    hl: str = Field(default="en", pattern=r"^[a-z-]{2,10}$")


class LensArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    image_index: int = Field(default=0, ge=0)


MAX_FETCH_CHARS = 50000


class FetchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    url: str = Field(min_length=1, max_length=4096)
    max_chars: int = Field(default=12000, ge=1, le=MAX_FETCH_CHARS)


RETRIEVAL_TOOLS: dict[str, tuple[type[BaseModel], str]] = {
    "google_search": (SearchArgs, "Search Google for external evidence."),
    "google_lens_search": (LensArgs, "Reverse image search on an image_index; defaults to Image 0."),
    "fetch_webpage": (FetchArgs, "Read a public HTTP(S) webpage as text."),
}


SEARCH_PROVIDERS = ("serper", "serpapi", "tavily", "exa")
# Spellings of the benchmark, its Hugging Face dataset (Crystal1047/Agentic-MME), and its GitHub
# repo (ChoS3nE11ven/Agentic-MME). "agentic-mme" etc. also cover the org-prefixed forms.
CONTAMINATION_PATTERNS = ("agentic-mme", "agentic_mme", "agentic mme", "agenticmme", "crystal1047", "chos3ne11ven")
CONTAMINATED_MESSAGE = "Results withheld: every result referenced the benchmark itself."
# Providers that can answer google_lens_search (reverse image search through Google Lens).
LENS_PROVIDERS = ("serper", "serpapi")


class RetrievalConfig(BaseModel):
    mode: Literal["disabled", "live", "replay"] = "disabled"
    search_provider: Literal["serper", "serpapi", "tavily", "exa"] = "serper"
    lens_provider: Literal["auto", "serper", "serpapi", "none"] = "auto"
    fetch_provider: Literal["jina", "exa"] = "jina"
    # Case-insensitive substrings; an empty list disables the contamination guard.
    contamination_patterns: list[str] = Field(default_factory=lambda: list(CONTAMINATION_PATTERNS))
    serper_api_key: SecretStr = SecretStr("")
    serpapi_api_key: SecretStr = SecretStr("")
    tavily_api_key: SecretStr = SecretStr("")
    exa_api_key: SecretStr = SecretStr("")
    imgbb_api_key: SecretStr = SecretStr("")
    jina_api_key: SecretStr = SecretStr("")
    # Local retrieval cache (see search_cache.py); "" disables it. Fuzzy threshold 0 = exact only.
    cache_root: str = ""
    cache_fuzzy_threshold: float = Field(default=0.8, ge=0, le=1)
    cache_write: bool = True
    # Consent to send task images to a third party for Google Lens (required in any case).
    allow_image_upload: bool = False
    # Where Lens images are uploaded: "serpapi" (SerpAPI's own Image API, used by lens_provider=serpapi),
    # "imgbb" (a public URL, needed by Serper's Lens); "auto" picks serpapi for SerpAPI Lens, else imgbb.
    image_host: Literal["auto", "serpapi", "imgbb"] = "auto"
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    max_response_bytes: int = Field(default=2_000_000, gt=0, le=10_000_000)

    @model_validator(mode="after")
    def validate_live(self) -> "RetrievalConfig":
        if self.mode == "live" and not self.provider_key():
            raise ValueError(
                f"live retrieval with search_provider={self.search_provider} requires {self.search_provider}_api_key"
            )
        lens = self.lens_provider_resolved()
        if self.mode == "live" and lens != "none" and not self.provider_key(lens):
            raise ValueError(f"google_lens_search with lens_provider={lens} requires {lens}_api_key")
        if self.mode == "live" and self.fetch_provider == "exa" and not self.exa_api_key.get_secret_value():
            raise ValueError("fetch_webpage with fetch_provider=exa requires exa_api_key")
        host = self.image_host_resolved()
        if host == "serpapi" and lens != "serpapi":
            raise ValueError("image_host=serpapi only serves lens_provider=serpapi (Serper Lens needs a public URL)")
        if self.allow_image_upload and host == "imgbb" and not self.imgbb_api_key.get_secret_value():
            raise ValueError("Lens image upload through ImgBB requires imgbb_api_key")
        return self

    def image_host_resolved(self) -> str:
        if self.image_host == "auto":
            return "serpapi" if self.lens_provider_resolved() == "serpapi" else "imgbb"
        return self.image_host

    def provider_key(self, provider: str | None = None) -> str:
        return getattr(self, f"{provider or self.search_provider}_api_key").get_secret_value()

    def lens_provider_resolved(self) -> str:
        if self.lens_provider == "auto":
            return self.search_provider if self.search_provider in LENS_PROVIDERS else "none"
        return self.lens_provider

    def lens_available(self) -> bool:
        return self.lens_provider_resolved() != "none"


# SerpAPI's Image API (POST /image) accepts JPEG/PNG/WebP up to 500 KB; the image_id lasts 10 minutes.
SERPAPI_IMAGE_MAX_BYTES = 500_000


def jpeg_under(image: Image.Image, max_bytes: int = SERPAPI_IMAGE_MAX_BYTES) -> bytes:
    """JPEG bytes no larger than max_bytes: lower the quality first, then shrink the image."""
    image = image.convert("RGB")
    while True:
        for quality in (92, 85, 75, 65):
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            if buffer.tell() <= max_bytes:
                return buffer.getvalue()
        if min(image.size) <= 64:
            raise ValueError("image cannot be compressed under the upload size limit")
        image = image.resize((max(1, int(image.width * 0.75)), max(1, int(image.height * 0.75))))


def public_url(url: str) -> str:
    """Reject local targets; webpage fetching itself is delegated to fixed Jina infrastructure."""
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
        raise ValueError("a public HTTP(S) URL without credentials is required")
    if parsed.port not in (None, 80, 443):
        raise ValueError("only standard HTTP(S) ports are supported")
    if "." not in host or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("local hostnames are not allowed")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # Numeric host aliases can resolve to loopback despite not being dotted IPv4.
        if host.replace(".", "").isdigit() or host.lower().startswith("0x"):
            raise ValueError("numeric host aliases are not allowed") from None
    else:
        if not address.is_global:
            raise ValueError("non-public addresses are not allowed")
    return url


async def http_payload(config: RetrievalConfig, method: str, url: str, **kwargs: Any) -> bytes:
    # The timeout covers Gym's retry loop as well as reading the response.
    async with asyncio.timeout(config.timeout_seconds):
        response = await request(method, url, allow_redirects=False, **kwargs)
        try:
            if not 200 <= response.status < 300:
                # Do not expose provider URLs, request headers, or secrets in tool errors.
                raise ValueError(f"retrieval provider returned HTTP {response.status}")
            payload = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                payload.extend(chunk)
                if len(payload) > config.max_response_bytes:
                    raise ValueError("retrieval response exceeds the configured byte limit")
            return bytes(payload)
        finally:
            response.release()


# Serper accepts a bare "zh"; SerpAPI rejects it (HTTP 400 "Unsupported `zh` interface language").
_SERPAPI_HL = {"zh": "zh-cn", "zh-hans": "zh-cn", "zh-sg": "zh-cn", "zh-hant": "zh-tw", "zh-hk": "zh-tw"}


def serpapi_to_serper(data: dict[str, Any], lens: bool) -> dict[str, Any]:
    """SerpAPI google / google_lens answer -> Serper's shape (organic first, as Serper returns)."""
    if lens:
        return {
            "organic": [
                {
                    "title": m.get("title"),
                    "link": m.get("link"),
                    "source": m.get("source"),
                    "imageUrl": m.get("image"),
                    "thumbnailUrl": m.get("thumbnail"),
                }
                for m in data.get("visual_matches") or []
                if isinstance(m, dict)
            ][:10]
        }
    out: dict[str, Any] = {
        "organic": [
            {"title": r.get("title"), "link": r.get("link"), "snippet": r.get("snippet"), "date": r.get("date")}
            for r in data.get("organic_results") or []
            if isinstance(r, dict)
        ][:10]
    }
    kg = data.get("knowledge_graph")
    if isinstance(kg, dict) and kg.get("title"):
        out["knowledgeGraph"] = {k: kg.get(k) for k in ("title", "type", "description", "website")}
    box = data.get("answer_box")
    if isinstance(box, dict):
        out["answerBox"] = {k: box.get(k) for k in ("title", "answer", "snippet", "link")}
    return out


def snippets_to_serper(results: list[Any], text_key: str, link_key: str = "url") -> dict[str, Any]:
    """Tavily / Exa result list -> Serper's organic list (snippet capped at 600 characters)."""
    organic = []
    for r in results or []:
        if isinstance(r, dict):
            text = r.get(text_key)
            if isinstance(text, list):  # Exa highlights
                text = " ... ".join(str(t) for t in text)
            organic.append(
                {
                    "title": r.get("title"),
                    "link": r.get(link_key),
                    "snippet": (str(text or "")[:600]),
                    "date": r.get("publishedDate") or r.get("published_date"),
                }
            )
    return {"organic": organic}


class Retrieval:
    def __init__(self, config: RetrievalConfig, replay: list[dict[str, Any]] | None = None) -> None:
        self.config = config
        self.replay = list(replay or [])
        self.replay_position = 0

    async def call(self, name: str, arguments: dict[str, Any], workspace: ImageWorkspace) -> dict[str, Any]:
        if self.config.mode == "disabled":
            raise ValueError("retrieval is disabled for this configuration")
        model = RETRIEVAL_TOOLS[name][0]
        args = model.model_validate(arguments)
        if isinstance(args, FetchArgs):
            public_url(args.url)
            if self._contaminated(args.url):  # no provider call for a benchmark URL
                return {"ok": True, "tool": name, "url": args.url, "context": CONTAMINATED_MESSAGE, "contaminated": 1}
        if isinstance(args, LensArgs):
            workspace.get(args.image_index)
        if self.config.mode == "replay":
            if self.replay_position >= len(self.replay):
                raise ValueError("no recorded retrieval result remains")
            event = self.replay[self.replay_position]
            if event["tool_name"] != name or model.model_validate(event["arguments"]) != args:
                raise ValueError("retrieval replay mismatch; no live fallback is permitted")
            self.replay_position += 1
            return event["output"]
        cache = (
            shared_cache(self.config.cache_root, self.config.cache_fuzzy_threshold, self.config.cache_write)
            if self.config.cache_root
            else None
        )
        if cache is not None:
            hit, kind = await asyncio.to_thread(self._cache_lookup, cache, args, workspace)
            if hit is not None:
                if isinstance(args, FetchArgs):
                    out = {
                        "ok": True,
                        "tool": name,
                        "url": args.url,
                        "context": hit[: args.max_chars],
                        "full_text": hit,
                    }
                else:
                    out = {"ok": True, "tool": name, "context": "", "raw": hit}
                out["cache"] = kind
                return self._decontaminate(out)
        out = await self._live(name, args, workspace)
        if cache is not None:
            await asyncio.to_thread(self._cache_store, cache, args, workspace, out)
            out["cache"] = "miss"
        return self._decontaminate(out)

    def _contaminated(self, value: Any) -> bool:
        patterns = self.config.contamination_patterns
        if not patterns:
            return False
        hay = (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)).lower()
        return any(p.lower() in hay for p in patterns)

    def _decontaminate(self, out: dict[str, Any]) -> dict[str, Any]:
        """Drop benchmark-quoting items (filter on read; the cache keeps the raw answer).

        Search / Lens: per organic item, plus knowledgeGraph / answerBox. Page: the whole page.
        "contaminated" counts the dropped items so the rate shows up in rollout traces.
        """
        if "url" in out:  # fetch_webpage
            # full_text is cache-only (without a cache it used to reach the model); scan what the
            # model would see.
            out.pop("full_text", None)
            if self._contaminated(out["context"]):
                out["context"] = CONTAMINATED_MESSAGE
                out["contaminated"] = 1
            return out
        raw = out["raw"]
        dropped = 0
        if isinstance(raw, dict):
            raw = dict(raw)
            organic = raw.get("organic")
            if isinstance(organic, list):
                kept = [item for item in organic if not self._contaminated(item)]
                dropped += len(organic) - len(kept)
                raw["organic"] = kept
            for key in [k for k, v in raw.items() if k != "organic" and self._contaminated(v)]:
                del raw[key]
                dropped += 1
        if dropped:
            out["contaminated"] = dropped
            if not any(raw.values()):
                out["raw"] = {}
                out["context"] = CONTAMINATED_MESSAGE
                return out
        out["raw"] = raw
        out["context"] = json.dumps(raw, ensure_ascii=False)[:50000]
        return out

    @staticmethod
    def _cache_lookup(cache, args, workspace: ImageWorkspace) -> tuple[Any, str]:
        if isinstance(args, SearchArgs):
            return cache.get_search(args.query)
        if isinstance(args, LensArgs):
            hit = cache.get_lens(workspace.get(args.image_index))
        else:
            hit = cache.get_browse(args.url)
        return hit, "exact" if hit is not None else "miss"

    def _cache_store(self, cache, args, workspace: ImageWorkspace, out: dict[str, Any]) -> None:
        if isinstance(args, FetchArgs):
            cache.put_browse(args.url, out.get("full_text") or out["context"], self.config.fetch_provider)
        elif isinstance(args, SearchArgs):
            cache.put_search(args.query, out["raw"], self.config.search_provider, args.gl, args.hl)
        else:
            cache.put_lens(workspace.get(args.image_index), out["raw"], self.config.lens_provider_resolved())
        out.pop("full_text", None)

    async def _live(self, name: str, args: BaseModel, workspace: ImageWorkspace) -> dict[str, Any]:
        if isinstance(args, FetchArgs):
            if self.config.fetch_provider == "exa":
                # As browsecomp_advanced_harness: Exa /contents with a text cap. Fetch up to the
                # tool's max_chars ceiling so the cached page serves any later max_chars.
                raw = await http_payload(
                    self.config,
                    "POST",
                    "https://api.exa.ai/contents",
                    headers={"x-api-key": self.config.exa_api_key.get_secret_value()},
                    json={"urls": [args.url], "text": {"maxCharacters": MAX_FETCH_CHARS}},
                )
                results = json.loads(raw).get("results") or []
                text = (results[0].get("text") if results and isinstance(results[0], dict) else None) or ""
                if not text.strip():
                    raise ValueError("no page content available for this URL")
            else:
                headers = {"Accept": "text/plain"}
                if self.config.jina_api_key.get_secret_value():
                    headers["Authorization"] = "Bearer " + self.config.jina_api_key.get_secret_value()
                raw = await http_payload(self.config, "GET", "https://r.jina.ai/" + args.url, headers=headers)
                text = raw.decode("utf-8", errors="replace")
            # full_text (the whole page) is only for the cache; _cache_store drops it before returning.
            return {"ok": True, "tool": name, "url": args.url, "context": text[: args.max_chars], "full_text": text}
        if isinstance(args, LensArgs) and not self.config.lens_available():
            raise ValueError(f"reverse image search is not available (lens_provider={self.config.lens_provider})")
        provider = self.config.lens_provider_resolved() if isinstance(args, LensArgs) else self.config.search_provider
        key = self.config.provider_key(provider)
        if isinstance(args, SearchArgs) and provider in ("tavily", "exa"):
            if provider == "tavily":
                raw = await http_payload(
                    self.config,
                    "POST",
                    "https://api.tavily.com/search",
                    headers={"Authorization": "Bearer " + key},
                    json={"query": args.query, "max_results": 5, "search_depth": "basic"},
                )
                data = snippets_to_serper(json.loads(raw).get("results"), "content")
            else:
                raw = await http_payload(
                    self.config,
                    "POST",
                    "https://api.exa.ai/search",
                    headers={"x-api-key": key},
                    json={"query": args.query, "numResults": 5, "contents": {"highlights": True}},
                )
                data = snippets_to_serper(json.loads(raw).get("results"), "highlights")
            return {"ok": True, "tool": name, "context": json.dumps(data, ensure_ascii=False)[:50000], "raw": data}
        headers = {"X-API-KEY": key}
        if isinstance(args, SearchArgs):
            payload = {"q": args.query, "gl": args.gl, "hl": args.hl, "num": 5}
            endpoint = "search"
        else:
            if not self.config.allow_image_upload:
                raise ValueError("Lens requires allow_image_upload=true; the image is uploaded for the search")
            image = workspace.get(args.image_index)
            if self.config.image_host_resolved() == "serpapi":
                # SerpAPI's Image API: no third image host, Lens takes the returned image_id.
                form = FormData()
                form.add_field("api_key", key)
                form.add_field(
                    "image",
                    await asyncio.to_thread(jpeg_under, image),
                    filename="image.jpg",
                    content_type="image/jpeg",
                )
                uploaded = json.loads(await http_payload(self.config, "POST", "https://serpapi.com/image", data=form))
                if not isinstance(uploaded, dict) or not uploaded.get("image_id"):
                    raise ValueError("image upload did not return an image_id")
                payload = {"image_id": str(uploaded["image_id"])}
            else:
                data_url = image_data_url(image)
                upload = await http_payload(
                    self.config,
                    "POST",
                    "https://api.imgbb.com/1/upload",
                    data={
                        "key": self.config.imgbb_api_key.get_secret_value(),
                        "expiration": "600",
                        "image": data_url.split(",", 1)[1],
                    },
                )
                uploaded = json.loads(upload)
                if not uploaded.get("success") or not uploaded.get("data", {}).get("url"):
                    raise ValueError("image upload did not return an image URL")
                payload = {"url": public_url(uploaded["data"]["url"]), "num": 5}
            endpoint = "lens"
        if provider == "serpapi":
            params = {"api_key": key, "num": "5"}
            if endpoint == "search":
                params |= {
                    "engine": "google",
                    "q": args.query,
                    "gl": args.gl,
                    "hl": _SERPAPI_HL.get(args.hl.lower(), args.hl),
                }
            else:
                params |= {"engine": "google_lens"} | {k: payload[k] for k in ("image_id", "url") if k in payload}
            raw = await http_payload(self.config, "GET", "https://serpapi.com/search.json", params=params)
            data = serpapi_to_serper(json.loads(raw), lens=endpoint == "lens")
        else:
            raw = await http_payload(
                self.config, "POST", "https://google.serper.dev/" + endpoint, headers=headers, json=payload
            )
            data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("retrieval provider returned an invalid JSON object")
        return {"ok": True, "tool": name, "context": json.dumps(data, ensure_ascii=False)[:50000], "raw": data}
