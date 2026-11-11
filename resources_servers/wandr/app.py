# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WANDR: web search, page reading and record-writing tools, scored by the pinned upstream WANDR verifier."""

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

from aiohttp import ClientConnectionError, ClientError, ClientPayloadError, ClientResponse
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.core import AGENT_INVALID_OUTPUT, VERIFIER_ERROR, NeMoGymResponse
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import VerifierFixture
from resources_servers.wandr.web import (
    EXA_BASE_URL,
    PageText,
    UnsafeURLError,
    check_http_url,
    extract_page_isolated,
    fetch_public,
    gym_get,
    gym_post_json,
)


logger = logging.getLogger(__name__)

SERVER_DIR = Path(__file__).resolve().parent
RUN_VERIFIER = SERVER_DIR / "run_verifier.py"
# Written by benchmarks/wandr/prepare.py: {"files": {"<required output file>": {"task": ..., "item_fields": [...]}}}
TASK_INDEX_FILE = "index.json"

RewardMetric = Literal[
    "soft_precision_full",
    "soft_recall_full",
    "soft_f1_full",
    "hard_precision_full",
    "hard_recall_full",
    "hard_f1_full",
    "soft_precision_retrieval",
    "soft_recall_retrieval",
    "soft_f1_retrieval",
    "hard_precision_retrieval",
    "hard_recall_retrieval",
    "hard_f1_retrieval",
]
REWARD_METRICS: tuple[str, ...] = RewardMetric.__args__


class WandrResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    tasks_dir: str = "benchmarks/wandr/data/tasks"
    wandr_core_dir: str = "benchmarks/wandr/data/wandr_core"
    # WANDR reports soft and hard precision, recall and F1 for the full and retrieval-only verdicts.
    reward_metric: RewardMetric = "soft_f1_full"

    # The verifier's judge, and the agent's search when search_backend is perplexity, on the NVIDIA inference API.
    inference_api_key: Optional[str] = None
    # The agent's web_search tool: Exa /search with exa_api_key, or Perplexity search at search_url.
    search_backend: Literal["exa", "perplexity"] = "exa"
    search_url: str = "https://inference-api.nvidia.com/v1/search/perplexity-search"
    exa_api_key: Optional[str] = None
    exa_base_url: str = EXA_BASE_URL
    exa_search_type: str = "auto"
    # Cache age Exa may serve, for search and for the verifier's fetch: 0 always fetches live, unset is Exa's default.
    exa_max_age_hours: Optional[int] = Field(default=None, ge=-1, le=720)
    search_max_results: int = Field(default=10, ge=1, le=20)
    search_max_tokens_per_page: int = Field(default=1024, ge=1)
    max_concurrent_searches: int = Field(default=32, ge=1)
    tool_timeout_s: float = Field(default=60, gt=0)
    fetch_page_chars: int = Field(default=12000, ge=1)
    fetch_max_bytes: int = Field(default=20_000_000, ge=1)
    max_cached_pages: int = Field(default=256, ge=0)

    judge_base_url: str = "https://inference-api.nvidia.com/v1"
    # Renames the task configs' judge model (gpt-5.4) for the gateway.
    judge_model_template: str = "openai/openai/{model}"
    # How the verifier fetches cited pages. "perplexity" is WANDR's own fetch through the Perplexity Agent API and
    # needs perplexity_api_key; "exa" uses Exa /contents with exa_api_key; "direct" is a plain HTTP GET.
    verifier_fetch_backend: Literal["exa", "perplexity", "direct"] = "exa"
    perplexity_api_key: Optional[str] = None
    # WANDR_* tuning passed to the verifier, such as concurrency and healing retries.
    verifier_env: dict[str, str] = Field(default_factory=dict)
    verifier_timeout_s: float = Field(default=14400, gt=0)
    # Per-call timeout of the verifier's triage, canon, dedup and judge calls; None keeps upstream's (15-60 s),
    # which a shared gateway under load exceeds, and upstream drops the record of every call that times out.
    llm_timeout_s: Optional[float] = Field(default=300, gt=0)
    max_concurrent_verifications: int = Field(default=8, ge=1)
    inconclusive_verification_retries: int = Field(default=1, ge=0)
    # Keep each verification's reports (reward.json, wandr_metrics.json, report.txt, debug/) under this directory.
    verifier_artifacts_dir: Optional[str] = None


class WandrRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    task_name: str
    required_output_files: list[str]
    # Set by an agent that writes the output files itself (file name -> file text). The verifier then scores
    # these files as written, and write_records calls in the response are ignored.
    output_files: Optional[dict[str, str]] = None


class WandrVerifyRequest(WandrRunRequest, BaseVerifyRequest):
    pass


class WandrVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    # False only when the verifier could not produce a score (the sample is then masked).
    evaluation_completed: bool = True
    submitted_rows: int = 0
    soft_precision_full: float = 0.0
    soft_recall_full: float = 0.0
    soft_f1_full: float = 0.0
    hard_precision_full: float = 0.0
    hard_recall_full: float = 0.0
    hard_f1_full: float = 0.0
    soft_precision_retrieval: float = 0.0
    soft_recall_retrieval: float = 0.0
    soft_f1_retrieval: float = 0.0
    hard_precision_retrieval: float = 0.0
    hard_recall_retrieval: float = 0.0
    hard_f1_retrieval: float = 0.0


class WebSearchRequest(BaseModel):
    query: Optional[str] = None
    max_results: Optional[int] = None


class WebSearchResponse(BaseModel):
    results: list[dict[str, str]] = Field(default_factory=list)
    error: Optional[str] = None


class FetchPageRequest(BaseModel):
    url: Optional[str] = None
    start: int = 0


class FetchPageResponse(BaseModel):
    url: str = ""
    title: str = ""
    content: str = ""
    start: int = 0
    end: int = 0
    total_chars: int = 0
    error: Optional[str] = None


class WriteRecordsRequest(BaseModel):
    file: Optional[str] = None
    records: Any = None


class RejectedRecord(BaseModel):
    index: int
    reason: str


class WriteRecordsResponse(BaseModel):
    file: str = ""
    accepted: int = 0
    rejected: list[RejectedRecord] = Field(default_factory=list)
    error: Optional[str] = None


def record_error(record: object, item_fields: list[str]) -> Optional[str]:
    """The upstream verifier's row checks: one failing row would zero the whole task, so the tool refuses it."""
    if not isinstance(record, dict):
        return "a record must be a JSON object"
    try:
        check_http_url(record.get("url"))
    except UnsafeURLError as error:
        return str(error)
    item = record.get("item")
    if not isinstance(item, dict):
        return "item must be an object"
    missing = [field for field in item_fields if not isinstance(item.get(field), str) or not item[field]]
    if missing:
        return f"item is missing non-empty string fields: {missing}"
    if not isinstance(record.get("answer"), dict):
        return "answer must be an object"
    excerpts = record.get("excerpts")
    if not isinstance(excerpts, list) or not all(isinstance(excerpt, str) for excerpt in excerpts):
        return "excerpts must be a list of strings"
    return None


def submitted_records(response: NeMoGymResponse, required_files: list[str]) -> dict[str, list[dict]]:
    """Rebuild the submission files from the write_records calls whose tool output accepted rows."""
    outputs = {item.call_id: item.output for item in response.output if item.type == "function_call_output"}
    files: dict[str, list[dict]] = {name: [] for name in required_files}
    for item in response.output:
        if item.type != "function_call" or item.name != "write_records" or item.call_id not in outputs:
            continue
        try:
            arguments = json.loads(item.arguments)
            result = WriteRecordsResponse.model_validate_json(outputs[item.call_id])
        except ValueError:
            continue
        if (
            not isinstance(arguments, dict)
            or result.error is not None
            or result.file not in files
            or not isinstance(arguments.get("records"), list)
        ):
            continue
        rejected = {entry.index for entry in result.rejected}
        files[result.file].extend(record for index, record in enumerate(arguments["records"]) if index not in rejected)
    return files


@dataclass
class VerifierRun:
    rewards: Optional[dict[str, float]] = None
    error: Optional[str] = None


def read_rewards(path: Path) -> VerifierRun:
    """Parse the verifier's reward.json; a missing or non-numeric metric makes the run inconclusive, not 0."""
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return VerifierRun(error=f"unreadable reward.json: {error}")
    if not isinstance(document, dict):
        return VerifierRun(error="reward.json is not an object")
    rewards = {name: document.get(name) for name in REWARD_METRICS}
    invalid = [
        name
        for name, value in rewards.items()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0
    ]
    if invalid:
        return VerifierRun(error=f"reward.json lacks valid metrics: {invalid}")
    return VerifierRun(rewards={name: float(value) for name, value in rewards.items()})


class WandrVerifier:
    """Scoring shared by the server and VERIFIER_FIXTURE; run_upstream_verifier is the only external call."""

    config: WandrResourcesServerConfig

    async def run_upstream_verifier(self, task_name: str, files: dict[str, list[dict] | str]) -> VerifierRun:
        raise NotImplementedError

    async def verify(self, body: WandrVerifyRequest) -> WandrVerifyResponse:
        files: dict[str, list[dict] | str]
        if body.output_files is not None:
            files = {name: body.output_files.get(name, "") for name in body.required_output_files}
            submitted_rows = sum(len([line for line in text.splitlines() if line.strip()]) for text in files.values())
            empty_files = [name for name, text in files.items() if not text.strip()]
        else:
            files = submitted_records(body.response, body.required_output_files)
            submitted_rows = sum(len(rows) for rows in files.values())
            empty_files = [name for name, rows in files.items() if not rows]
        # WANDR scores a submission with an empty required file as an invalid submission: 0 on every metric.
        if empty_files:
            return WandrVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                submitted_rows=submitted_rows,
                failure_kind=AGENT_INVALID_OUTPUT,
                failure_reason=f"no accepted records for required files: {empty_files}",
            )

        run = VerifierRun(error="the verifier did not run")
        for _ in range(1 + self.config.inconclusive_verification_retries):
            run = await self.run_upstream_verifier(body.task_name, files)
            if run.rewards is not None:
                break
        if run.rewards is None:
            return WandrVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                submitted_rows=submitted_rows,
                evaluation_completed=False,
                mask_sample=True,
                failure_kind=VERIFIER_ERROR,
                failure_reason=run.error,
            )
        metrics = run.rewards
        return WandrVerifyResponse(
            **body.model_dump(),
            reward=metrics[self.config.reward_metric],
            submitted_rows=submitted_rows,
            **metrics,
        )


class WandrResourcesServer(WandrVerifier, SimpleResourcesServer):
    config: WandrResourcesServerConfig
    _search_semaphore: asyncio.Semaphore = PrivateAttr()
    _verify_semaphore: asyncio.Semaphore = PrivateAttr()
    _pages: OrderedDict[str, PageText] = PrivateAttr(default_factory=OrderedDict)
    # Output file -> {"task": ..., "item_fields": [...]}, from the index prepare writes.
    _index_files: Optional[dict[str, dict]] = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        config = self.config
        if "exa" in (config.search_backend, config.verifier_fetch_backend) and not config.exa_api_key:
            raise ValueError("search_backend or verifier_fetch_backend is exa, so exa_api_key must be set")
        if config.verifier_fetch_backend == "perplexity" and not config.perplexity_api_key:
            raise ValueError("verifier_fetch_backend is perplexity, so perplexity_api_key must be set")
        self._search_semaphore = asyncio.Semaphore(self.config.max_concurrent_searches)
        self._verify_semaphore = asyncio.Semaphore(self.config.max_concurrent_verifications)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/web_search")(self.web_search)
        app.post("/fetch_page")(self.fetch_page)
        app.post("/write_records")(self.write_records)
        return app

    def _index(self) -> dict[str, dict]:
        if self._index_files is None:
            index_path = _resolve_under_cwd_or_install(self.config.tasks_dir) / TASK_INDEX_FILE
            self._index_files = json.loads(index_path.read_text())["files"]
        return self._index_files

    async def web_search(self, body: WebSearchRequest) -> WebSearchResponse:
        if not body.query or not body.query.strip():
            return WebSearchResponse(error="query must be a non-empty string")
        max_results = min(max(body.max_results or self.config.search_max_results, 1), 20)
        if self.config.search_backend == "exa":
            url = f"{self.config.exa_base_url.rstrip('/')}/search"
            headers = {"x-api-key": self.config.exa_api_key or ""}
            # About four characters per token, so snippets match the Perplexity backend's page budget.
            highlights: dict = {"maxCharacters": 4 * self.config.search_max_tokens_per_page}
            contents: dict = {"highlights": highlights}
            if self.config.exa_max_age_hours is not None:
                contents["maxAgeHours"] = self.config.exa_max_age_hours
            payload = {
                "query": body.query,
                "numResults": max_results,
                "type": self.config.exa_search_type,
                "contents": contents,
            }
        else:
            url = self.config.search_url
            headers = {"Authorization": f"Bearer {self.config.inference_api_key}"}
            payload = {
                "query": body.query,
                "max_results": max_results,
                "max_tokens_per_page": self.config.search_max_tokens_per_page,
            }
        async with self._search_semaphore:
            try:
                status, data = await gym_post_json(url, headers, payload, self.config.tool_timeout_s)
            except (ClientConnectionError, ClientPayloadError, asyncio.TimeoutError, ValueError) as error:
                return WebSearchResponse(error=f"search failed: {type(error).__name__}")
        if status != 200 or not isinstance(data, dict):
            # Provider error bodies can echo credentials; report the status only.
            return WebSearchResponse(error=f"search failed with HTTP {status}")
        if self.config.search_backend == "exa":
            results = [
                {
                    "title": str(result.get("title") or ""),
                    "url": str(result.get("url") or ""),
                    "date": str(result.get("publishedDate") or ""),
                    "snippet": "\n".join(str(text) for text in result.get("highlights") or [])
                    or str(result.get("text") or ""),
                }
                for result in data.get("results") or []
                if isinstance(result, dict)
            ]
        else:
            results = [
                {key: str(result.get(key) or "") for key in ("title", "url", "date", "snippet")}
                for result in data.get("results") or []
                if isinstance(result, dict)
            ]
        return WebSearchResponse(results=results)

    async def _get(
        self, url: str, timeout: float, headers: dict[str, str], server_hostname: Optional[str]
    ) -> ClientResponse:
        return await gym_get(url, timeout, headers, server_hostname)

    async def fetch_page(self, body: FetchPageRequest) -> FetchPageResponse:
        url = (body.url or "").strip()
        page = self._pages.get(url)
        if page is None:
            try:
                fetched = await fetch_public(
                    self._get, url, timeout=self.config.tool_timeout_s, max_bytes=self.config.fetch_max_bytes
                )
                page = await extract_page_isolated(fetched.body, fetched.content_type, fetched.final_url)
            except (ClientError, OSError, asyncio.TimeoutError, ValueError) as error:
                return FetchPageResponse(url=url, error=f"fetch failed: {type(error).__name__}: {error}")
            if self.config.max_cached_pages:
                self._pages[url] = page
                while len(self._pages) > self.config.max_cached_pages:
                    self._pages.popitem(last=False)
        start = min(max(body.start, 0), len(page.text))
        end = min(start + self.config.fetch_page_chars, len(page.text))
        return FetchPageResponse(
            url=url,
            title=page.title,
            content=page.text[start:end],
            start=start,
            end=end,
            total_chars=len(page.text),
            error=None if page.text else "the page has no readable text",
        )

    async def write_records(self, body: WriteRecordsRequest) -> WriteRecordsResponse:
        file = body.file or ""
        item_fields = self._index().get(file, {}).get("item_fields")
        if item_fields is None:
            return WriteRecordsResponse(file=file, error="unknown file: use a results_*.jsonl name the task gives")
        if isinstance(body.records, str):
            # The tool-call parser leaves an array argument as a string when its JSON does not parse.
            return WriteRecordsResponse(
                file=file,
                error="records arrived as a string, not a JSON array: check its JSON (escape quotes inside strings)",
            )
        if not isinstance(body.records, list) or not body.records:
            return WriteRecordsResponse(file=file, error="records must be a non-empty list of record objects")
        rejected = [
            RejectedRecord(index=index, reason=reason)
            for index, record in enumerate(body.records)
            if (reason := record_error(record, item_fields)) is not None
        ]
        return WriteRecordsResponse(file=file, accepted=len(body.records) - len(rejected), rejected=rejected)

    async def run_upstream_verifier(self, task_name: str, files: dict[str, list[dict] | str]) -> VerifierRun:
        # Only names from the prepared index reach the filesystem: a task directory and its own output files.
        foreign = [name for name in files if self._index().get(name, {}).get("task") != task_name]
        if foreign:
            raise ValueError(f"WANDR task {task_name!r} has no output files {foreign}")
        task_dir = _resolve_under_cwd_or_install(self.config.tasks_dir) / task_name
        core_dir = _resolve_under_cwd_or_install(self.config.wandr_core_dir)
        async with self._verify_semaphore:
            with tempfile.TemporaryDirectory(prefix="wandr_verify_") as scratch:
                workspace, logs = Path(scratch) / "workspace", Path(scratch) / "logs"
                workspace.mkdir()
                for name, rows in files.items():
                    text = rows if isinstance(rows, str) else "".join(json.dumps(row) + "\n" for row in rows)
                    (workspace / name).write_text(text)
                run = await self._run_subprocess(task_dir, core_dir, workspace, logs)
                if self.config.verifier_artifacts_dir and logs.exists():
                    target = Path(self.config.verifier_artifacts_dir) / task_name / uuid.uuid4().hex
                    shutil.copytree(logs, target)
                return run

    async def _run_subprocess(self, task_dir: Path, core_dir: Path, workspace: Path, logs: Path) -> VerifierRun:
        tests_dir = task_dir / "tests"
        env = {
            **os.environ,
            **self.config.verifier_env,
            "PYTHONPATH": f"{core_dir}{os.pathsep}{tests_dir}",
            "LOGS_DIR": str(logs),
            "OPENAI_API_KEY": self.config.inference_api_key or "",
            "OPENAI_BASE_URL": self.config.judge_base_url,
            "WANDR_GYM_JUDGE_MODEL_TEMPLATE": self.config.judge_model_template,
            "WANDR_GYM_FETCH_BACKEND": self.config.verifier_fetch_backend,
            "WANDR_GYM_FETCH_MAX_BYTES": str(self.config.fetch_max_bytes),
            "WANDR_BROWSER_FALLBACK": "0",
        }
        if self.config.llm_timeout_s is not None:
            env["WANDR_GYM_LLM_TIMEOUT_SEC"] = str(self.config.llm_timeout_s)
        if self.config.perplexity_api_key:
            env["PERPLEXITY_API_KEY"] = self.config.perplexity_api_key
        if self.config.verifier_fetch_backend == "exa":
            env["WANDR_GYM_EXA_API_KEY"] = self.config.exa_api_key or ""
            env["WANDR_GYM_EXA_BASE_URL"] = self.config.exa_base_url
            if self.config.exa_max_age_hours is not None:
                env["WANDR_GYM_EXA_MAX_AGE_HOURS"] = str(self.config.exa_max_age_hours)
        command = [
            sys.executable,
            str(RUN_VERIFIER),
            str(tests_dir / "verify.py"),
            "--workspace-dir",
            str(workspace),
            "--task-dir",
            str(tests_dir / "wandr_task"),
            "--instruction-path",
            str(task_dir / "instruction.md"),
            "--output-dir",
            str(logs / "wandr"),
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=core_dir,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=self.config.verifier_timeout_s)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return VerifierRun(error=f"the verifier timed out after {self.config.verifier_timeout_s:g}s")
        reward_path, error_path = logs / "reward.json", logs / "error.json"
        if process.returncode == 0 and reward_path.exists():
            return read_rewards(reward_path)
        try:
            detail = str(json.loads(error_path.read_text()).get("error", ""))
        except (OSError, ValueError, AttributeError):
            detail = stderr.decode(errors="replace")[-2000:]
        logger.warning("WANDR verifier failed (exit %s): %s", process.returncode, detail)
        return VerifierRun(error=f"verifier exit {process.returncode}: {detail}"[:2000])


class _FixtureVerifier(WandrVerifier):
    """Replaces the upstream verifier with a fixed result, so the fixture checks this server's scoring offline."""

    config = WandrResourcesServerConfig(host="127.0.0.1", port=0, entrypoint="app.py", name="wandr")

    async def run_upstream_verifier(self, task_name: str, files: dict[str, list[dict] | str]) -> VerifierRun:
        return VerifierRun(rewards={name: 1.0 for name in REWARD_METRICS})


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=_FixtureVerifier,
    request_model=WandrVerifyRequest,
    cases_path=SERVER_DIR / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    WandrResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = WandrResourcesServer.run_webserver()  # noqa: F401
