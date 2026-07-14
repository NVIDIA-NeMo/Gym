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
import json
import os
import re
import shutil
import threading
from asyncio import sleep
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from time import time
from typing import Any, ClassVar, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from httpx import AsyncClient
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from tavily import AsyncTavilyClient
from tavily.errors import BadRequestError

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    RATE_LIMIT_ERROR_CODES,
    RETRY_ERROR_CODES,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import SESSION_ID_KEY, raise_for_status, request
from resources_servers.browsecomp_advanced_harness.judge_prompt import JUDGE_PROMPT_TEMPLATE


class TavilySearchResourcesServerConfig(BaseResourcesServerConfig):
    # Search/browse backend. "tavily" (default) or "exa". The chosen provider's
    # key must be present (validated below). exclude_domains are honored by both.
    search_provider: str = "tavily"
    tavily_api_key: str | List[str] | None = None
    exa_api_key: str | List[str] | None = None
    exclude_domains_file_path: str
    use_judge: bool = True  # If False, use regex matching instead of LLM judge
    judge_model_server: Optional[ModelServerRef] = None
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    debug: bool = False
    dump_session_id_to_metrics_on_exit: bool = False
    # Terminal/disk mode: "none" = inline content (original behavior);
    # "per_session" = search/browse write pages to a per-session disk workspace
    # and a read-only bash_command tool is exposed for the model to grep/read them.
    workspace: str = "none"
    workspace_root: Optional[str] = None  # default: $BROWSECOMP_WS_ROOT or /tmp/browsecomp_ws
    bash_timeout_s: float = 60.0  # match bc_frankie's _BASH_MAX_DURATION_S
    bash_max_concurrency: int = 64
    max_page_bytes: int = 2_000_000  # cap per-page bytes written to disk
    # Results returned per search query (both providers). The bc_frankie Exa
    # reference uses 10; its Tavily path (and this harness historically) uses 5.
    max_results: int = 5

    @model_validator(mode="after")
    def _check_provider_key(self) -> "TavilySearchResourcesServerConfig":
        if self.search_provider == "tavily" and not self.tavily_api_key:
            raise ValueError("tavily_api_key is required when search_provider='tavily'")
        if self.search_provider == "exa" and not self.exa_api_key:
            raise ValueError("exa_api_key is required when search_provider='exa'")
        if self.search_provider not in ("tavily", "exa"):
            raise ValueError(f"search_provider must be 'tavily' or 'exa', got {self.search_provider!r}")
        return self


class TavilySearchRequest(BaseModel):
    queries: Optional[List[str]] = None  # Make optional to handle missing args gracefully
    max_total_length: int = 30000

    @model_validator(mode="before")
    @classmethod
    def coerce_queries(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        queries = data.get("queries")
        if queries is None:
            return data

        # Case 1: JSON-encoded string → parse it
        if isinstance(queries, str):
            try:
                queries = json.loads(queries)
            except (json.JSONDecodeError, ValueError):
                queries = [queries]

        # Case 2: nested list e.g. [["q1", "q2"]] → flatten one level
        if isinstance(queries, list) and queries and isinstance(queries[0], list):
            queries = [q for sublist in queries for q in sublist if isinstance(q, str)]

        data = dict(data)
        data["queries"] = queries
        return data


class TavilySearchResponse(BaseModel):
    results_string: str


class BrowseRequest(BaseModel):
    urls: List[str]
    goal: Optional[str] = None
    max_total_length: int = 30000

    @model_validator(mode="before")
    @classmethod
    def coerce_urls(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        urls = data.get("urls")
        if urls is None:
            return data

        # Case 1: JSON-encoded string → parse it
        if isinstance(urls, str):
            try:
                urls = json.loads(urls)
            except (json.JSONDecodeError, ValueError):
                urls = [urls]

        # Case 2: nested list e.g. [["q1", "q2"]] → flatten one level
        if isinstance(urls, list) and urls and isinstance(urls[0], list):
            urls = [q for sublist in urls for q in sublist if isinstance(q, str)]

        data = dict(data)
        data["urls"] = urls
        return data


class BrowseResponse(BaseModel):
    results_string: str


class BashCommandRequest(BaseModel):
    keystrokes: Optional[str] = None
    duration: float = 10.0


class BashCommandResponse(BaseModel):
    results_string: str


class TavilySearchRunRequest(BaseRunRequest):
    ground_truth: str
    question: str


class TavilySearchVerifyRequest(TavilySearchRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    judge_response_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    reasoning: str
    extracted_final_answer: str
    reward: float
    judge_response: Optional[NeMoGymResponse] = None


class TavilySearchSingleAsyncTavilyMetrics(BaseModel):
    function: str  # "search" | "browse"
    provider: str = "tavily"  # "tavily" | "exa"
    status: str
    start_time: float
    end_time: float
    time_taken: Optional[float] = None

    @model_validator(mode="after")
    def compute_time_taken(self):
        self.time_taken = self.end_time - self.start_time
        return self


class TavilySearchMetrics(BaseModel):
    async_tavily_calls: List[TavilySearchSingleAsyncTavilyMetrics] = Field(default_factory=list)


class TavilySearchVerifyResponse(TavilySearchVerifyRequest, JudgeEvaluation):
    num_tool_calls: int
    reset_count: int = 0
    metrics: TavilySearchMetrics


def _abort_on_invalid_api_key(provider: str, status: int, body: str) -> None:
    """A 401/403 from the search provider means the API key is bad — every search in the
    run would silently degrade, so kill the benchmark instead of producing a garbage score."""
    print(
        f"[browsecomp][fatal][{provider}_invalid_api_key] HTTP {status}: search-provider API key "
        f"rejected — aborting benchmark. body={body[:300]}",
        flush=True,
    )
    os._exit(1)


class TavilySearchAIOHTTPClientResponse(BaseModel):
    status_code: int
    data: Dict[str, Any]

    def json(self) -> Dict[str, Any]:
        return self.data


class TavilySearchAIOHTTPClient(BaseModel):
    headers: Dict[str, str]
    base_url: str

    debug: bool

    async def post(self, endpoint: str, content: str, timeout: float) -> TavilySearchAIOHTTPClientResponse:
        """
        endpoint: str e.g. "/search" or "/extract"
        timeout: float is not used
        """
        request_kwargs = {
            "method": "POST",
            "headers": self.headers,
            "url": f"{self.base_url}{endpoint}",
            "data": content,
        }

        MAX_NUM_TRIES = 3  # Hardcode for now
        max_num_tries = MAX_NUM_TRIES
        tries = 0
        while tries < max_num_tries:
            tries += 1
            response = await request(**request_kwargs)

            if response.status in (401, 403):
                _abort_on_invalid_api_key("tavily", response.status, (await response.content.read()).decode())

            if response.status in RETRY_ERROR_CODES:
                # If we hit a rate limit, we don't want to hit max num tries, so we increment both.
                rate_limited = response.status in RATE_LIMIT_ERROR_CODES
                if rate_limited:
                    max_num_tries += 1

                content = (await response.content.read()).decode()
                tag = "tavily_rate_limit" if rate_limited else "tavily_retry"
                print(
                    f"[browsecomp][tool_fail][{tag}] endpoint={endpoint} status={response.status} "
                    f"try={tries} body={content[:300]}",
                    flush=True,
                )
                await sleep(0.5)
                continue
            else:
                tavily_response = TavilySearchAIOHTTPClientResponse(
                    status_code=response.status,
                    data=await response.json(),
                )
                if self.debug:
                    print(f"Received the following Tavily response: {tavily_response}")

                return tavily_response

        # We've exited the loop
        await raise_for_status(response)

    @classmethod
    def from_httpx_AsyncClient(cls, client: AsyncClient, debug: bool) -> "TavilySearchAIOHTTPClient":
        return cls(
            headers=client.headers,
            base_url=str(client.base_url),
            debug=debug,
        )


class ExaAIOHTTPClient(BaseModel):
    """Async Exa REST client over NeMo Gym's global aiohttp client (no exa-py / httpx dep).

    Mirrors TavilySearchAIOHTTPClient's retry + rate-limit-tagging loop. Exposes high-level
    ``search``/``get_contents`` that return the parsed Exa JSON, so the harness (and tests)
    call them like the AsyncTavilyClient's ``search``/``extract``.
    """

    headers: Dict[str, str]
    base_url: str = "https://api.exa.ai"
    debug: bool = False

    async def _post(self, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        request_kwargs = {
            "method": "POST",
            "headers": self.headers,
            "url": f"{self.base_url}{endpoint}",
            "data": json.dumps(body),
        }

        MAX_NUM_TRIES = 3  # mirror the Tavily client
        max_num_tries = MAX_NUM_TRIES
        tries = 0
        while tries < max_num_tries:
            tries += 1
            response = await request(**request_kwargs)

            if response.status in (401, 403):
                _abort_on_invalid_api_key("exa", response.status, (await response.content.read()).decode())

            if response.status in RETRY_ERROR_CODES:
                rate_limited = response.status in RATE_LIMIT_ERROR_CODES
                if rate_limited:
                    # don't let rate limits burn the retry budget
                    max_num_tries += 1
                content = (await response.content.read()).decode()
                tag = "exa_rate_limit" if rate_limited else "exa_retry"
                print(
                    f"[browsecomp][tool_fail][{tag}] endpoint={endpoint} status={response.status} "
                    f"try={tries} body={content[:300]}",
                    flush=True,
                )
                await sleep(0.5)
                continue

            data = await response.json()
            if self.debug:
                print(f"Received the following Exa response: status={response.status}")
            return data

        await raise_for_status(response)

    async def search(
        self, query: str, num_results: int, exclude_domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "query": query,
            "numResults": num_results,
            "type": "auto",
            "contents": {"highlights": True},
        }
        if exclude_domains:
            body["excludeDomains"] = list(exclude_domains)
        return await self._post("/search", body)

    async def get_contents(self, urls: List[str], max_characters: int) -> Dict[str, Any]:
        body = {"urls": list(urls), "text": {"maxCharacters": max_characters}}
        return await self._post("/contents", body)


# ---------------------------------------------------------------------------
# Terminal/disk mode: per-session page workspace (pages/ + manifest.tsv) and a
# read-only bash tool. Ported from the bc_frankie harness.
# ---------------------------------------------------------------------------
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_URL_RE = re.compile(r"https?://([^/]+)(/.*)?")

_BASH_HEAD_BYTES = 8192
_BASH_TAIL_BYTES = 4096


def _slugify(text: str, max_len: int = 40) -> str:
    s = (text or "").lower()
    s = _SLUG_RE.sub("-", s).strip("-")
    s = s[:max_len].rstrip("-")
    return s or "untitled"


def _slug_from_url(url: str) -> str:
    m = _URL_RE.match(url or "")
    if not m:
        return _slugify(url)
    host = (m.group(1) or "").replace("www.", "")
    tail = (m.group(2) or "").rstrip("/").rsplit("/", 1)[-1]
    return _slugify(f"{host} {tail}")


def _bash_truncate(s: str) -> str:
    if len(s) <= _BASH_HEAD_BYTES + _BASH_TAIL_BYTES + 64:
        return s
    omitted = len(s) - _BASH_HEAD_BYTES - _BASH_TAIL_BYTES
    return (
        f"{s[:_BASH_HEAD_BYTES]}\n"
        f"[...truncated {omitted} bytes; head shown above, tail follows...]\n"
        f"{s[-_BASH_TAIL_BYTES:]}"
    )


# --- bash_command guards (ported byte-for-byte from the bc_frankie_bash_tool
# harness @ ee72d54: _bash_denylisted + _bash_allowlisted). Layered deny-first,
# then default-deny allow-list. NOT a security boundary (the ulimit -f 0 in
# _run_bash_readonly is the kernel backstop); these block destructive/network/
# interpreter commands + command/process substitution + find -exec / sed -i. ---
_BASH_DENY_COMMANDS = {
    # destructive file ops / writes
    "rm",
    "rmdir",
    "unlink",
    "shred",
    "srm",
    "wipe",
    "mv",
    "cp",
    "dd",
    "truncate",
    "install",
    "ln",
    "tee",
    "mkfs",
    "mke2fs",
    "mkswap",
    "fsck",
    # permissions / metadata
    "chmod",
    "chown",
    "chgrp",
    "chattr",
    "touch",
    # privilege / escape / process / system control
    "sudo",
    "su",
    "doas",
    "pkexec",
    "chroot",
    "unshare",
    "nsenter",
    "mount",
    "umount",
    "kill",
    "pkill",
    "killall",
    "reboot",
    "shutdown",
    "halt",
    "poweroff",
    "init",
    "systemctl",
    "service",
    "crontab",
    "at",
    "batch",
    # network / exfiltration
    "curl",
    "wget",
    "nc",
    "ncat",
    "netcat",
    "telnet",
    "ssh",
    "scp",
    "sftp",
    "rsync",
    "ftp",
    "tftp",
    "socat",
    # interpreters / shells / arbitrary-code engines & wrappers
    "bash",
    "sh",
    "zsh",
    "ksh",
    "dash",
    "csh",
    "tcsh",
    "fish",
    "python",
    "python2",
    "python3",
    "perl",
    "ruby",
    "node",
    "nodejs",
    "php",
    "lua",
    "rscript",
    "awk",
    "gawk",
    "mawk",
    "eval",
    "exec",
    "command",
    "builtin",
    "source",
    "xargs",
    "nohup",
    "setsid",
    "timeout",
    "nice",
    "watch",
    "stdbuf",
    "ionice",
    # environment / secret disclosure
    "env",
    "printenv",
    "export",
    "set",
    "declare",
    "typeset",
}

# Patterns blocked anywhere (evaluated on the quote-stripped command).
_BASH_DENY_GLOBAL_PATTERNS = [
    (r"`", "backtick command substitution"),
    (r"\$\((?!\()", "$(...) command substitution"),
    (r"[<>]\(", "process substitution"),
    (r":\s*\(\s*\)\s*\{", "fork bomb"),
    (r"(?:^|[;|&])\s*\.\s+\S", "sourcing a script with '.'"),
]

_FIND_WRITE_RE = re.compile(r"-(delete|exec|execdir|fprintf|fprint|fls|ok|okdir)\b")
_SED_INPLACE_RE = re.compile(r"(?:^|\s)-i\b")


def _bash_denylisted(keystrokes):
    """Return a reason string if the command is denied, else None.

    Pragmatic speed-bump only — NOT a security boundary. See _BASH_DENY_*.
    """
    s = keystrokes or ""
    # Strip quoted strings so search terms (grep ">" / grep "rm") don't trip rules.
    unq = re.sub(r"'[^']*'", " ", s)
    unq = re.sub(r'"[^"]*"', " ", unq)

    for pat, why in _BASH_DENY_GLOBAL_PATTERNS:
        if re.search(pat, unq):
            return why

    # Output redirection to a real file (allow /dev/null and fd dups like 2>&1).
    redir = re.sub(r"(?:\d*|&)>{1,2}\s*(?:/dev/null|&\s*\d+|&\s*-)", " ", unq)
    if re.search(r">{1,2}", redir):
        return "output redirection to a file (writes blocked)"

    # Command-position check on each pipeline / list segment.
    for seg in re.split(r"[;\n|&]+", unq):
        words = seg.split()
        idx = 0
        while idx < len(words) and re.match(r"^[A-Za-z_]\w*=", words[idx]):
            idx += 1  # skip leading VAR=val assignments
        if idx >= len(words):
            continue
        prog = Path(words[idx].lstrip("\\")).name
        if prog in _BASH_DENY_COMMANDS:
            return f"command '{prog}' is blocked"
        if prog == "find" and _FIND_WRITE_RE.search(seg):
            return "find write/exec action (-delete/-exec/...) is blocked"
        if prog == "sed" and _SED_INPLACE_RE.search(seg):
            return "sed -i (in-place file edit) is blocked"
    return None


# Default-deny companion: a command passes only if EVERY command-position program
# is in this read-only set. Closes the denylist's gap for unenumerated binaries.
_BASH_ALLOW_COMMANDS = {
    "grep",
    "cat",
    "head",
    "tail",
    "sed",
    "ls",
    "wc",
    "sort",
    "uniq",
    "cut",
    "tr",
    "nl",
    "strings",
    "file",
    "find",
    "diff",
    "echo",
    "printf",
    "cd",
}

# Shell keywords permitted as structural glue (pipelines, for-loops, conditionals).
_BASH_KEYWORD_SKIP = {"if", "elif", "then", "else", "while", "until", "do", "!", "time", "{", "(", "[["}
_BASH_KEYWORD_STANDALONE = {"done", "fi", "esac", "}", ")"}
_BASH_KEYWORD_HEADER = {"for", "case", "select", "function"}

# fd-dups / &-redirections contain '&' which would otherwise phantom-split a segment.
_BASH_FD_DUP_RE = re.compile(r"\d*[<>]&\s*[-\d]+")
_BASH_AMP_REDIR_RE = re.compile(r"&>>?\s*\S+")


def _bash_allowlisted(keystrokes):
    """Return a reason string if any command-position program is NOT read-only,
    else None. Default-deny companion to _bash_denylisted."""
    s = keystrokes or ""
    # Strip quoted strings so search terms (grep ">" / grep "rm") don't trip rules.
    unq = re.sub(r"'[^']*'", " ", s)
    unq = re.sub(r'"[^"]*"', " ", unq)
    # Neutralize &-bearing redirections so they don't split segments.
    unq = _BASH_FD_DUP_RE.sub(" ", unq)
    unq = _BASH_AMP_REDIR_RE.sub(" ", unq)

    for seg in re.split(r"[;\n|&]+", unq):
        words = seg.split()
        idx = 0
        while idx < len(words):
            w = words[idx]
            if re.match(r"^[A-Za-z_]\w*=", w):
                idx += 1  # leading VAR=val assignment
            elif w in _BASH_KEYWORD_HEADER:
                idx = len(words)  # loop/case header: no command to check
            elif w in _BASH_KEYWORD_SKIP or w in _BASH_KEYWORD_STANDALONE:
                idx += 1  # structural keyword; real command (if any) follows
            else:
                break
        if idx >= len(words):
            continue
        prog = Path(words[idx].lstrip("\\")).name
        if prog not in _BASH_ALLOW_COMMANDS:
            return f"command '{prog}' not in read-only allow-list"
    return None


class _PageWriter:
    """Per-session disk persistence for search/browse content.

    Each retrieved page lands at workspace/pages/<idx>_<kind>_<slug>[_rN].txt
    with a row appended to workspace/manifest.tsv. The workspace is wiped at
    construction so each session starts from a clean slate.
    """

    MANIFEST_HEADER = "page_id\tkind\tsource\ttitle\tbytes\n"

    def __init__(self, workspace_dir):
        self.workspace = Path(workspace_dir)
        if self.workspace.exists():
            shutil.rmtree(self.workspace, ignore_errors=True)
        self.pages_dir = self.workspace / "pages"
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.workspace / "manifest.tsv"
        self.manifest_path.write_text(self.MANIFEST_HEADER)
        self._lock = threading.Lock()
        self._counter = 0

    def _next_idx(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

    def _append_manifest(self, idx, kind, source, title, n_bytes) -> None:
        src = (source or "").replace("\t", " ").replace("\n", " ")[:300]
        ttl = (title or "").replace("\t", " ").replace("\n", " ")[:200]
        line = f"{idx:04d}\t{kind}\t{src}\t{ttl}\t{n_bytes}\n"
        with self._lock:
            with self.manifest_path.open("a", encoding="utf-8") as f:
                f.write(line)

    def write_search_result(self, query, result_idx, title, url, content) -> str:
        idx = self._next_idx()
        fname = f"{idx:04d}_search_{_slugify(query)}_r{result_idx}.txt"
        body = f"[Query]: {query}\n[URL]: {url}\n[Title]: {title}\n\n{content}"
        (self.pages_dir / fname).write_text(body, errors="replace")
        self._append_manifest(idx, "search", query, title, len(content))
        return f"pages/{fname}"

    def write_browse_page(self, url, title, content) -> str:
        idx = self._next_idx()
        fname = f"{idx:04d}_browse_{_slug_from_url(url)}.txt"
        body = f"[URL]: {url}\n[Title]: {title}\n\n{content}"
        (self.pages_dir / fname).write_text(body, errors="replace")
        self._append_manifest(idx, "browse", url, title, len(content))
        return f"pages/{fname}"


def _last_assistant_text(response) -> str:
    """Text of the LAST assistant message item in response.output (the model's final answer).
    Replaces Response.output_text, which CONCATENATES every assistant message item — wrong when
    the model emits text alongside tool calls mid-trajectory (~49% of BrowseComp samples). Also
    aligns with bc_frankie, which grades only the final message."""
    text = ""
    for item in response.output:
        if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
            t = "".join(
                getattr(p, "text", "")
                for p in (getattr(item, "content", None) or [])
                if getattr(p, "type", None) == "output_text"
            )
            if t:
                text = t
    return text


class TavilySearchResourcesServer(SimpleResourcesServer):
    config: TavilySearchResourcesServerConfig

    _async_tavily_clients: Optional[List[AsyncTavilyClient]] = PrivateAttr(default=None)
    _exa_clients: Optional[List[ExaAIOHTTPClient]] = PrivateAttr(default=None)
    _num_requests: int = 0
    _session_id_to_metrics: Optional[Dict[str, TavilySearchMetrics]] = PrivateAttr(default=None)
    _session_workspaces: Dict[str, "_PageWriter"] = PrivateAttr(default_factory=dict)
    _bash_semaphore: Optional[asyncio.Semaphore] = PrivateAttr(default=None)
    _workspace_root: Optional[str] = PrivateAttr(default=None)

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = JUDGE_PROMPT_TEMPLATE
    JUDGE_MAX_ATTEMPTS: int = 10

    def model_post_init(self, __context) -> None:
        # Tavily clients (built only when a Tavily key is configured).
        tavily_api_keys = self.config.tavily_api_key
        if isinstance(tavily_api_keys, str):
            tavily_api_keys = [tavily_api_keys]

        # Optional Tavily client-source identity header (matches bc_frankie's
        # x-client-source). Value comes ONLY from the env (TAVILY_CLIENT_SOURCE);
        # not hardcoded. Empty/unset => header not sent.
        client_source = os.environ.get("TAVILY_CLIENT_SOURCE", "")
        if tavily_api_keys:
            self._async_tavily_clients = [AsyncTavilyClient(api_key=k) for k in tavily_api_keys]
            for async_tavily_client in self._async_tavily_clients:
                async_tavily_client._client = TavilySearchAIOHTTPClient.from_httpx_AsyncClient(
                    async_tavily_client._client, self.config.debug
                )
                if client_source:
                    async_tavily_client._client.headers["x-client-source"] = client_source

        # Exa clients (built only when an Exa key is configured). One client per
        # key; round-robined like Tavily. Native aiohttp REST (no exa-py dep).
        exa_api_keys = self.config.exa_api_key
        if isinstance(exa_api_keys, str):
            exa_api_keys = [exa_api_keys]
        if exa_api_keys:
            self._exa_clients = [
                ExaAIOHTTPClient(
                    headers={"x-api-key": k, "Content-Type": "application/json"},
                    debug=self.config.debug,
                )
                for k in exa_api_keys
            ]
            print(f"Search provider: exa ({len(self._exa_clients)} key(s))")

        self._session_id_to_metrics = defaultdict(TavilySearchMetrics)

        self._exclude_domains = self._parse_exclude_domains()
        self._page_cache: dict[str, str] = {}
        print(f"Excluded domains: {self._exclude_domains}")

        # Terminal/disk mode setup
        self._bash_semaphore = asyncio.Semaphore(self.config.bash_max_concurrency)
        self._workspace_root = self.config.workspace_root or os.environ.get("BROWSECOMP_WS_ROOT", "/tmp/browsecomp_ws")
        if self.config.workspace == "per_session":
            Path(self._workspace_root).mkdir(parents=True, exist_ok=True)
            print(f"Terminal mode ON; per-session workspaces under {self._workspace_root}")

        if self.config.debug:
            print("Debug mode enabled")

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/search")(self.search)
        app.post("/browse")(self.browse)
        app.post("/bash_command")(self.bash_command)

        main_app_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan_wrapper(app):
            async with main_app_lifespan(app) as maybe_state:
                yield maybe_state

            if self.config.dump_session_id_to_metrics_on_exit:
                out_file = Path(__file__).parent / "session_id_metrics.json"
                print(f"Dumping session_id metrics to {out_file}")

                to_dump = {k: v.model_dump(mode="json") for k, v in self._session_id_to_metrics.items()}
                with out_file.open("w") as f:
                    json.dump(to_dump, f)

        app.router.lifespan_context = lifespan_wrapper

        return app

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        if self.config.workspace == "per_session":
            sid = request.session[SESSION_ID_KEY]
            self._session_workspaces[sid] = _PageWriter(Path(self._workspace_root) / sid)
        return BaseSeedSessionResponse()

    def _get_page_writer(self, sid: str) -> Optional["_PageWriter"]:
        if self.config.workspace != "per_session":
            return None
        pw = self._session_workspaces.get(sid)
        if pw is None:  # lazily create if /seed_session was skipped
            pw = _PageWriter(Path(self._workspace_root) / sid)
            self._session_workspaces[sid] = pw
        return pw

    def _cleanup_workspace(self, sid: str) -> None:
        pw = self._session_workspaces.pop(sid, None)
        if pw is not None:
            shutil.rmtree(pw.workspace, ignore_errors=True)

    async def bash_command(self, request: Request, body: BashCommandRequest) -> BashCommandResponse:
        sid = request.session[SESSION_ID_KEY]
        page_writer = self._get_page_writer(sid)
        if page_writer is None:
            return BashCommandResponse(
                results_string="[bash error: terminal workspace is not enabled for this server]"
            )
        duration = max(1.0, min(self.config.bash_timeout_s, float(body.duration or 10)))
        out = await self._run_bash_readonly(body.keystrokes or "", duration, str(page_writer.workspace))
        return BashCommandResponse(results_string=out)

    async def _run_bash_readonly(self, keystrokes: str, duration: float, cwd: str) -> str:
        # Layered guard (ported from bc_frankie @ ee72d54): deny-list first
        # (specific reason), then default-deny read-only allow-list. Either one
        # blocking returns [blocked: ...] and skips execution.
        blocked = _bash_denylisted(keystrokes) or _bash_allowlisted(keystrokes)
        if blocked is not None:
            print(f"[browsecomp][bash][blocked] {blocked}: {(keystrokes or '')[:160]!r}", flush=True)
            return f"[blocked: {blocked}]\n[exit_code=-3]"
        # Read-only enforcement backstop: `ulimit -f 0` sets RLIMIT_FSIZE=0 so the
        # command cannot write/grow ANY file; reads + pipes still work. cwd is
        # pinned to the session workspace; env is minimal (no inherited secrets).
        wrapped = f"ulimit -c 0 2>/dev/null; ulimit -f 0 2>/dev/null; {keystrokes}"
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": cwd, "LC_ALL": "C.UTF-8"}
        async with self._bash_semaphore:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "bash",
                    "-c",
                    wrapped,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
            except Exception as e:
                return f"[bash error: {e}]\n[exit_code=-2]"
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=duration)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
                return f"[command timed out after {duration:.0f}s]\n[exit_code=-1]"

        stdout = _bash_truncate(stdout_b.decode(errors="replace"))
        stderr = _bash_truncate(stderr_b.decode(errors="replace"))
        truncated = len(stdout_b) + len(stderr_b) > (_BASH_HEAD_BYTES + _BASH_TAIL_BYTES) * 2
        parts = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append("--- stderr ---")
            parts.append(stderr)
        parts.append(f"[exit_code={proc.returncode}{' truncated' if truncated else ''}]")
        return "\n".join(parts)

    def _select_tavily_client(self) -> AsyncTavilyClient:
        client = self._async_tavily_clients[self._num_requests % len(self._async_tavily_clients)]
        self._num_requests += 1
        return client

    def _select_exa_client(self) -> ExaAIOHTTPClient:
        client = self._exa_clients[self._num_requests % len(self._exa_clients)]
        self._num_requests += 1
        return client

    def _record_call(
        self, metrics: "TavilySearchMetrics", function: str, provider: str, status: str, start: float
    ) -> None:
        """Append one per-API-call metering record (provider, function, latency).
        One record per provider HTTP request: per query for search, per call for browse."""
        metrics.async_tavily_calls.append(
            TavilySearchSingleAsyncTavilyMetrics(
                function=function, provider=provider, status=status, start_time=start, end_time=time()
            )
        )

    async def _exa_search_one(self, query: str, max_length: int, metrics: "TavilySearchMetrics") -> str:
        """Exa search: highlight snippets returned INLINE (never written to pages/, even in
        terminal mode). Mirrors the bc_frankie Exa harness formatting exactly."""
        if len(query) > 400:
            return "Query is too long"

        client = self._select_exa_client()
        call_start = time()
        try:
            results = await client.search(
                query, num_results=self.config.max_results, exclude_domains=self._exclude_domains
            )
        except Exception as e:
            self._record_call(metrics, "search", "exa", "error", call_start)
            print(f"[browsecomp][tool_fail][exa_search] query={query[:200]!r} error={e}", flush=True)
            return f"Search failed: {e}"
        self._record_call(metrics, "search", "exa", "success", call_start)

        blocks = [f"[Search Query]: {query}"]
        running_len = len(blocks[0])
        for result in results.get("results", []):
            title = result.get("title", "") or ""
            url = result.get("url", "") or ""
            highlights = result.get("highlights") or []
            snippet = " ... ".join(h for h in highlights if h)
            entry = f"[Title]: {title}\n[URL]: {url}\n[Snippet]: {snippet}\n"
            if running_len + len(entry) > max_length:
                break
            blocks.append(entry)
            running_len += len(entry)
        return "\n".join(blocks)

    async def _search_one(self, query: str, max_length: int, metrics: "TavilySearchMetrics") -> str:
        if len(query) > 400:
            return "Query is too long"

        client = self._select_tavily_client()
        print(f"[tavily_call_begin function=search query={query[:80]!r}]", flush=True)
        call_start = time()
        try:
            results = await client.search(
                query,
                max_results=self.config.max_results,
                exclude_domains=self._exclude_domains,
                search_depth="advanced",
                include_raw_content=True,
            )
        except BadRequestError as e:
            self._record_call(metrics, "search", "tavily", "error", call_start)
            print(f"[browsecomp][tool_fail][tavily_search_bad_request] query={query[:200]!r} error={e}", flush=True)
            return f"Search failed: {e}"
        except Exception as e:
            self._record_call(metrics, "search", "tavily", "error", call_start)
            print(
                f"[tavily_call function=search status=error duration_s={time() - call_start:.2f} query={query[:80]!r} error_type={type(e).__name__} error={str(e)[:200]}]",
                flush=True,
            )
            raise

        self._record_call(metrics, "search", "tavily", "success", call_start)
        print(
            f"[tavily_call function=search status=success duration_s={time() - call_start:.2f} query={query[:80]!r} n_results={len(results.get('results', []))}]",
            flush=True,
        )
        postprocessed_results = self._postprocess_search_results(query, results, max_length)
        return postprocessed_results

    async def _search_one_to_disk(
        self, query: str, page_writer: "_PageWriter", max_per_query: int, metrics: "TavilySearchMetrics"
    ) -> str:
        """Terminal mode: run one search, write each result's raw content to disk, return
        title/url/snippet/[Saved to] metadata. Mirrors the bc_frankie harness tavily_search
        (disk mode) formatting exactly."""
        client = self._select_tavily_client()
        call_start = time()
        try:
            results = await client.search(
                query,
                max_results=self.config.max_results,
                exclude_domains=self._exclude_domains,
                search_depth="advanced",
                include_raw_content=True,
            )
        except BadRequestError as e:
            self._record_call(metrics, "search", "tavily", "error", call_start)
            print(f"[browsecomp][tool_fail][tavily_search_bad_request] query={query[:200]!r} error={e}", flush=True)
            return f"Search failed: {e}"
        self._record_call(metrics, "search", "tavily", "success", call_start)

        blocks = [f"[Search Query]: {query}"]
        running_len = len(blocks[0])
        for ri, result in enumerate(results.get("results", []), start=1):
            title = result.get("title", "") or ""
            url = result.get("url", "") or ""
            snippet = (result.get("content") or "")[:500]
            raw = result.get("raw_content") or result.get("content") or ""
            if raw:
                if len(raw) > self.config.max_page_bytes:
                    raw = raw[: self.config.max_page_bytes]
                saved = page_writer.write_search_result(query, ri, title, url, raw)
                saved_line = f"[Saved to]: {saved} ({len(raw)} bytes)\n"
            else:
                saved_line = ""
            entry = f"[Title]: {title}\n[URL]: {url}\n[Snippet]: {snippet}\n{saved_line}"
            if running_len + len(entry) > max_per_query:
                break
            blocks.append(entry)
            running_len += len(entry)
        return "\n".join(blocks)

    async def search(self, request: Request, body: TavilySearchRequest) -> TavilySearchResponse:
        sid = request.session[SESSION_ID_KEY]
        metrics = self._session_id_to_metrics[sid]

        if self.config.debug:
            print("\n\n body.queries: ", body.queries)

        if body.queries is None or len(body.queries) == 0:
            return TavilySearchResponse(results_string="Query is none or empty")

        max_per_query_length = body.max_total_length // len(body.queries)
        if self.config.search_provider == "exa":
            # Exa: highlights-only, always inline (no disk pages, even in terminal mode).
            results = await asyncio.gather(
                *[self._exa_search_one(q, max_per_query_length, metrics) for q in body.queries]
            )
        else:
            page_writer = self._get_page_writer(sid)
            if page_writer is not None:
                # terminal mode: write each result to disk, return metadata + paths
                results = await asyncio.gather(
                    *[self._search_one_to_disk(q, page_writer, max_per_query_length, metrics) for q in body.queries]
                )
            else:
                # inline mode: return content directly
                results = await asyncio.gather(
                    *[self._search_one(q, max_per_query_length, metrics) for q in body.queries]
                )

        return TavilySearchResponse(results_string="\n\n".join(results))

    async def browse(self, request: Request, body: BrowseRequest) -> BrowseResponse:
        metrics = self._session_id_to_metrics[request.session[SESSION_ID_KEY]]

        if self.config.debug:
            print("\n\n browse urls: ", body.urls)
            print(f"goal={body.goal}")

        urls = [u for u in body.urls if not self._is_url_excluded(u)]
        if not urls:
            return BrowseResponse(results_string="Error: no URLs provided.")
        urls = urls[:5]

        # set max length per url
        max_per_url_length = body.max_total_length // len(urls)

        # fetch full page content (provider-specific); normalize to a list of
        # {url, raw_content} so the shared disk/inline formatting below is provider-agnostic.
        start_time = time()
        if self.config.search_provider == "exa":
            exa_client = self._select_exa_client()
            print(f"[exa_call_begin function=browse n_urls={len(urls)} goal={(body.goal or '')[:80]!r}]", flush=True)
            try:
                raw = await exa_client.get_contents(urls=urls, max_characters=max_per_url_length)
            except Exception as e:
                self._record_call(metrics, "browse", "exa", "error", start_time)
                print(f"[browsecomp][tool_fail][exa_extract] urls={urls} error={e}", flush=True)
                return BrowseResponse(results_string=f"Failed to extract content: {e}")
            self._record_call(metrics, "browse", "exa", "success", start_time)
            result_list = [
                {"url": r.get("url", "") or "", "raw_content": r.get("text", "") or ""} for r in raw.get("results", [])
            ]
        else:
            async_tavily_client = self._select_tavily_client()
            print(
                f"[tavily_call_begin function=extract n_urls={len(urls)} goal={(body.goal or '')[:80]!r}]",
                flush=True,
            )
            try:
                # NOTE: do NOT pass query/goal — the harness extracts the full page content.
                # Passing a query triggers relevance-focused extraction that returns far less
                # (e.g. ~2k vs ~15k bytes for the same URL). `goal` stays a tool-schema field
                # for the model to state intent, but is not sent to Tavily (matches the harness).
                results = await async_tavily_client.extract(urls=urls)
            except Exception as e:
                self._record_call(metrics, "browse", "tavily", "error", start_time)
                print(f"[browsecomp][tool_fail][tavily_extract] urls={urls} error={e}", flush=True)
                return BrowseResponse(results_string=f"Failed to extract content: {e}")
            self._record_call(metrics, "browse", "tavily", "success", start_time)
            print(
                f"[tavily_call function=extract status=success duration_s={time() - start_time:.2f} n_urls={len(urls)} n_results={len(results.get('results', []))}]",
                flush=True,
            )
            result_list = results.get("results", [])

        # return if no results
        if not result_list:
            return BrowseResponse(results_string="No content extracted.")

        page_writer = self._get_page_writer(request.session[SESSION_ID_KEY])
        if page_writer is not None:
            # terminal mode: write each page to disk, return metadata + preview.
            # Mirrors the bc_frankie harness tavily_extract (disk mode) formatting exactly.
            blocks = []
            for result in result_list:
                url = result.get("url", "") or ""
                content = result.get("raw_content", "") or ""
                if content:
                    if len(content) > self.config.max_page_bytes:
                        content = content[: self.config.max_page_bytes]
                    saved = page_writer.write_browse_page(url, "", content)
                    preview = content[:500].replace("\n", " ")
                    blocks.append(f"[URL]: {url}\n[Saved to]: {saved} ({len(content)} bytes)\n[Preview]: {preview}\n")
                else:
                    blocks.append(f"[URL]: {url}\n[Empty content]\n")
            return BrowseResponse(results_string="\n\n".join(blocks))

        # inline mode: return content directly
        blocks = []
        for result in result_list:
            url = result.get("url", "")
            content = result.get("raw_content", "")
            if len(content) > max_per_url_length:
                content = content[:max_per_url_length] + "\n... [truncated]"
            blocks.append(f"[URL]: {url}\n[Content]:\n{content}\n")

        results_string = "\n\n".join(blocks)
        return BrowseResponse(results_string=results_string)

    async def verify(self, request: Request, body: TavilySearchVerifyRequest) -> TavilySearchVerifyResponse:
        question = body.question
        ground_truth = body.ground_truth
        last_assistant_response = _last_assistant_text(body.response)

        if self.config.use_judge:
            judge_evaluation = await self._verify_answer_with_judge(question, ground_truth, last_assistant_response)
        else:
            judge_evaluation = self._verify_answer_with_regex(ground_truth, last_assistant_response)

        # num_tool_calls now comes from the agent loop (counts ALL tool calls,
        # including those made before context resets). Fall back to the old
        # final-output count only if the agent didn't surface it.
        agent_num_tool_calls = getattr(body.response, "num_tool_calls", None)
        if agent_num_tool_calls is None:
            agent_num_tool_calls = sum(o.type == "function_call" for o in body.response.output)
        verify_response = TavilySearchVerifyResponse(
            **body.model_dump(),
            **judge_evaluation.model_dump(),
            num_tool_calls=agent_num_tool_calls,
            reset_count=getattr(body.response, "reset_count", 0) or 0,
            metrics=self._session_id_to_metrics[request.session[SESSION_ID_KEY]],
        )

        # terminal mode: clean up this session's disk workspace
        if self.config.workspace == "per_session":
            self._cleanup_workspace(request.session[SESSION_ID_KEY])

        return verify_response

    ###### UTILITY FUNCTIONS ######

    def _is_url_excluded(self, url: str) -> bool:
        """Check if the URL's domain is in the excluded domains list."""
        hostname = urlparse(url).hostname or ""
        return any(hostname == domain or hostname.endswith("." + domain) for domain in self._exclude_domains)

    def _postprocess_search_results(self, query: str, results: dict, max_length: int) -> str:
        blocks = [f"[Search Query]: {query}"]
        running_len = len(blocks[0])

        for result in results["results"]:
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("raw_content") or result.get("content", "")
            if len(content) > 5000:
                content = content[:5000] + "\n... [truncated]"
            entry = f"[Title]: {title}\n[URL]: {url}\n[Content]:\n{content}\n"

            if running_len + len(entry) > max_length:
                break
            blocks.append(entry)
            running_len += len(entry)

        formatted_results = "\n".join(blocks)
        return formatted_results

    def _parse_exclude_domains(self) -> list[str]:
        with open(self.config.exclude_domains_file_path, "r") as f:
            exclude_config = json.load(f)
        exclude_domains = []
        # this is pretty hard-coded so we ensure the file structure is correct
        notices = exclude_config["notices"]
        for notice in notices:
            for prop in notice["properties"]:
                if prop.get("type") == "domain":
                    exclude_domains.append(prop["value"])
        return exclude_domains

    async def _verify_answer_with_judge(self, question: str, ground_truth: str, response: str) -> JudgeEvaluation:
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            question=question, correct_answer=ground_truth, response=response
        )

        judge_create_params = self.config.judge_responses_create_params.model_copy(deep=True)
        judge_create_params.max_output_tokens = 2048
        judge_create_params.input = [
            NeMoGymEasyInputMessage(role="user", content=judge_prompt),
        ]

        judge_response = None
        for attempt in range(self.JUDGE_MAX_ATTEMPTS):
            judge_call_start = time()
            try:
                # Judge samples at temperature 1.0 on every attempt; parse-error retries simply
                # re-sample at 1.0 (which already varies the output — no need to escalate from 0.0).
                temp = 1.0
                judge_create_params.temperature = temp

                print(
                    f"[judge_call_begin attempt={attempt + 1}/{self.JUDGE_MAX_ATTEMPTS} temp={temp}]",
                    flush=True,
                )
                http_response = await self.server_client.post(
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=judge_create_params,
                )
                judge_response = NeMoGymResponse.model_validate(await http_response.json())
                text = judge_response.output[-1].content[-1].text

                is_correct, extracted, parsed_ok = self._parse_judge(text)
                if parsed_ok:
                    print(
                        f"[judge_call attempt={attempt + 1} status=success duration_s={time() - judge_call_start:.2f} is_correct={is_correct}]",
                        flush=True,
                    )
                    return JudgeEvaluation(
                        judge_response_create_params=judge_create_params,
                        reasoning=text,
                        extracted_final_answer=extracted,
                        reward=1.0 if is_correct else 0.0,
                        judge_response=judge_response,
                    )
                print(
                    f"[judge_call attempt={attempt + 1} status=parse_error duration_s={time() - judge_call_start:.2f} raw_output={text[:200]!r}]",
                    flush=True,
                )

            except Exception as e:
                sleep_s = min(2**attempt, 30)
                print(
                    f"[judge_call attempt={attempt + 1} status=error duration_s={time() - judge_call_start:.2f} error_type={type(e).__name__} error={str(e)[:200]} backoff_s={sleep_s}]",
                    flush=True,
                )
                await sleep(sleep_s)

        print(
            f"[judge_exhausted max_attempts={self.JUDGE_MAX_ATTEMPTS} had_response={judge_response is not None}]",
            flush=True,
        )
        return JudgeEvaluation(
            judge_response_create_params=judge_create_params,
            reasoning="",
            extracted_final_answer="",
            reward=0.0,
            judge_response=judge_response,
        )

    def _verify_answer_with_regex(self, ground_truth: str, response: str) -> JudgeEvaluation:
        """Verify answer by checking if ground_truth (as regex) matches in response."""
        matches = re.findall(r"Answer:\s*(.*)\s*Confidence:", response, re.IGNORECASE)

        if matches:
            answer = matches[-1].strip()  # Get the last item in the list
        else:
            answer = ""
        if self.config.debug:
            print(answer)
        reward = 1.0 if answer == ground_truth else 0.0
        return JudgeEvaluation(
            judge_response_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            reasoning=f"Regex match for '{ground_truth}': {'found' if answer == ground_truth else 'not found'}",
            extracted_final_answer=answer,
            reward=reward,
            judge_response=None,
        )

    def _parse_judge(self, text: str) -> tuple[bool, str, bool]:
        """Parse grading output. Returns (is_correct, extracted, parsed_ok).

        Uses the LAST 'correct: yes/no' match to avoid picking up template
        echoes or reasoning inside <think> blocks.
        """
        matches = list(re.finditer(r"correct:\s*(yes|no)\b", text, re.IGNORECASE))
        if not matches:
            return False, "", False

        is_correct = matches[-1].group(1).lower() == "yes"

        ans_matches = list(re.finditer(r"extracted_final_answer:\s*(.+?)(?:\n|$)", text))
        extracted = ans_matches[-1].group(1).strip() if ans_matches else ""

        if extracted and "The final exact answer extracted from the [response]" in extracted:
            return False, "", False

        return is_correct, extracted, True


if __name__ == "__main__":
    TavilySearchResourcesServer.run_webserver()
