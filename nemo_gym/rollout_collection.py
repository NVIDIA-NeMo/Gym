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
import asyncio
import glob as glob_module
import json
import os
import warnings
from asyncio import Future, Semaphore
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union
from urllib.parse import urlparse

import orjson
from aiohttp import ClientOSError, ClientTimeout, ServerDisconnectedError
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from tqdm.asyncio import tqdm
from wandb import Table

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.config_types import BaseNeMoGymCLIConfig, BaseServerConfig, ConfigError, ConfigPathNotFoundError
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    RESPONSES_CREATE_PARAMS_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    SKILLS_REF_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    canonical_agent_ref,
    get_global_config_dict,
    get_wandb_run,
    is_global_config_dict_set,
    row_agent_key,
)
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config, validate_prompt_compatibility
from nemo_gym.server_utils import (
    GlobalAIOHTTPAsyncClientConfig,
    ServerClient,
    get_global_aiohttp_client,
    get_nemo_gym_fastapi_num_workers,
    get_response_json,
    is_global_aiohttp_client_request_debug_enabled,
    is_global_aiohttp_client_setup,
    raise_for_status,
    set_global_aiohttp_client,
)
from nemo_gym.skills import SkillsConfig, load_skill_directory


# ---------------------------------------------------------------------------
# Failure-routing sentinels (set by agent servers, read by the dispatcher).
#
# Background:
#   The historical contract was "every dispatched task produces one row in
#   the main rollouts jsonl, succeeded or failed." That contract broke
#   resume-after-walltime: synthetic ``-failed`` rows written during a
#   SIGTERM grace window look identical to real successes to the dedup in
#   ``_load_from_cache`` (which keys only on (task_index, rollout_index)),
#   so chain-hop 2 thinks failed tasks are done and never retries them.
#
# New contract:
#   - Successes go to the main jsonl (``output_jsonl_fpath``).
#   - Failures go to a sidecar (``<output_stem>_failures.jsonl``), one row
#     per attempt, with ``_ng_failure_class`` set.
#   - ``kill_shaped`` failures (Slurm SIGTERM, Ray actor died, OOM, ...) go
#     NOWHERE: the absence of a row is the canonical signal. Resume's
#     set-difference re-dispatches them naturally; per-task timeout bounds
#     the chain-hop wallclock.
#   - On resume, ``_load_from_cache`` reads BOTH files: main jsonl tells
#     it what's permanently done, sidecar tells it how many attempts each
#     non-success has consumed (capped at NEMO_GYM_MAX_ROLLOUT_ATTEMPTS,
#     default 3). Rows flagged ``_ng_failure_terminal=True`` are never
#     retried regardless of attempt count.
# ---------------------------------------------------------------------------

NG_FAILURE_CLASS_KEY = "_ng_failure_class"
NG_NO_PERSIST_KEY = "_ng_no_persist"
NG_TERMINAL_KEY = "_ng_failure_terminal"

_DEFAULT_MAX_ROLLOUT_ATTEMPTS = 3


def _get_max_rollout_attempts() -> int:
    """Read ``NEMO_GYM_MAX_ROLLOUT_ATTEMPTS`` (positive int) or default to 3."""
    raw = os.environ.get("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS")
    if raw is None or raw == "":
        return _DEFAULT_MAX_ROLLOUT_ATTEMPTS
    try:
        n = int(raw)
        if n < 1:
            raise ValueError(f"must be >= 1, got {n}")
        return n
    except (TypeError, ValueError) as e:
        print(
            f"WARNING: could not parse NEMO_GYM_MAX_ROLLOUT_ATTEMPTS={raw!r} ({e}); "
            f"falling back to default {_DEFAULT_MAX_ROLLOUT_ATTEMPTS}.",
            flush=True,
        )
        return _DEFAULT_MAX_ROLLOUT_ATTEMPTS


def _failures_path_for(output_fpath: Path) -> Path:
    """Sidecar path used by the dispatcher and ``_load_from_cache``."""
    return output_fpath.with_name(output_fpath.stem + "_failures.jsonl")


# ---------------------------------------------------------------------------
# External agent dispatch (agent_url): rows with ``agent_ref: {"url": ...}``
# POST straight to ``{url}/run``. This deliberately bypasses
# ``server_utils.request()``, whose retry loop never gives up on connection
# errors — a down external endpoint would hang the run. Retries here are
# bounded, and every failure becomes a result row carrying
# ``NG_FAILURE_CLASS_KEY`` (sidecar + retry on resume); raising instead would
# abort the whole collection.
# ---------------------------------------------------------------------------

EXTERNAL_AGENT_FAILURE_CLASS = "external_agent_error"

_EXTERNAL_AGENT_MAX_TRIES = 3
_EXTERNAL_AGENT_RETRY_SLEEP_SECS = 0.5
# Aggregate metrics is a best-effort, post-collection call; keep its bound fixed.
_EXTERNAL_AGGREGATE_TIMEOUT_SECS = 600.0
_DEFAULT_AGENT_RUN_TIMEOUT_SECS = 1800.0
# Last-resort per-host limit, used by _effective_per_host_connection_limit() only when
# neither the live client nor the run's global config can be consulted. Derived from the
# config model so it cannot drift from the default.
_FALLBACK_PER_HOST_CONNECTION_LIMIT = GlobalAIOHTTPAsyncClientConfig().global_aiohttp_connector_limit_per_host

# Log every external /run failure for the first few, then sample: a down agent at high
# concurrency would otherwise print once per pending row and garble the progress bar.
_NUM_EXTERNAL_AGENT_FAILURES = 0
_EXTERNAL_AGENT_FAILURE_PRINT_HEAD = 5
_EXTERNAL_AGENT_FAILURE_PRINT_INTERVAL = 100


def _normalize_agent_url(url: str) -> str:
    """Validate an external agent URL and strip any trailing slash."""
    normalized = url.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"agent_url must be an absolute http:// or https:// URL, got {url!r}")
    # "/run" is appended to this URL by string concatenation; anything at or after a "?" or
    # "#" (even a bare delimiter, which urlparse reports as an empty query/fragment) would
    # swallow "/run" into the query/fragment and the POST would hit the server root instead.
    if "?" in normalized or "#" in normalized or parsed.params:
        raise ValueError(
            f"agent_url must not carry a query string or fragment, got {url!r}. "
            "Pass auth material via your agent's own configuration instead."
        )
    # user:pass@ credentials would be stamped into agent_ref on every artifact row and into
    # wandb metric labels. The error deliberately does not echo the URL (it holds the secret).
    if parsed.username or parsed.password:
        raise ValueError(
            "agent_url must not embed credentials (user:pass@host). "
            "Pass auth material via your agent's own configuration instead."
        )
    return normalized


def _is_external_url_ref(agent_ref: Any) -> bool:
    """True if this ref dispatches by URL. Name-first, matching row_agent_key: a ref carrying
    both keys is a named agent."""
    return isinstance(agent_ref, dict) and bool(agent_ref.get("url")) and not agent_ref.get("name")


def _rows_need_named_dispatch(rows: List[Dict]) -> bool:
    """True if any row resolves to a named agent server (requiring a ServerClient/head server)."""
    return any(not _is_external_url_ref(row.get(AGENT_REF_KEY_NAME)) for row in rows)


def _effective_per_host_connection_limit() -> Optional[int]:
    """The shared HTTP client's per-host connection cap, or None when unlimited.

    Every request to a single agent_url competes for the same per-host connection pool,
    and time spent waiting for a connection counts against the request's timeout. Never
    initializes the client; before it exists (the normal CLI case — this check runs before
    the first request), the limit the client WILL carry is computed from the run's global
    config, mirroring set_global_aiohttp_client's construction.
    """
    if is_global_aiohttp_client_setup():
        try:
            limit = int(get_global_aiohttp_client().connector.limit_per_host)
        except Exception:
            return _FALLBACK_PER_HOST_CONNECTION_LIMIT
        return limit if limit > 0 else None  # aiohttp uses 0 to mean unlimited
    # Only consult the config if it is already loaded — get_global_config_dict() would
    # otherwise trigger a CLI parse (and sys.exit) in non-CLI processes.
    if not is_global_config_dict_set():
        return _FALLBACK_PER_HOST_CONNECTION_LIMIT
    try:
        cfg = GlobalAIOHTTPAsyncClientConfig.model_validate(get_global_config_dict())
        limit = cfg.global_aiohttp_connector_limit_per_host // get_nemo_gym_fastapi_num_workers()
    except Exception:
        return _FALLBACK_PER_HOST_CONNECTION_LIMIT
    return limit if limit > 0 else None


def _external_agent_failure_result(run_url: str, error: str) -> Dict[str, Any]:
    """Shape an external /run failure into a sidecar-routable result row."""
    global _NUM_EXTERNAL_AGENT_FAILURES
    _NUM_EXTERNAL_AGENT_FAILURES += 1
    n = _NUM_EXTERNAL_AGENT_FAILURES
    if n <= _EXTERNAL_AGENT_FAILURE_PRINT_HEAD or n % _EXTERNAL_AGENT_FAILURE_PRINT_INTERVAL == 0:
        tqdm.write(f"[rollout_collection] external agent /run failed (failure #{n}) url={run_url} error={error}")
    return {NG_FAILURE_CLASS_KEY: EXTERNAL_AGENT_FAILURE_CLASS, "error": error}


async def _post_external_agent_run(row: Dict, timeout_secs: float) -> Dict[str, Any]:
    """POST one row to an external agent's /run; failures become sidecar rows, never exceptions."""
    run_url = f"{row[AGENT_REF_KEY_NAME]['url']}/run"
    client = get_global_aiohttp_client()
    data = orjson.dumps(row)
    headers = {"Content-Type": "application/json"}
    timeout = ClientTimeout(total=timeout_secs)

    response = None
    last_connect_error: Optional[BaseException] = None
    for num_try in range(1, _EXTERNAL_AGENT_MAX_TRIES + 1):
        try:
            # allow_redirects=False: aiohttp turns a redirected POST into a body-less GET;
            # better to fail loudly with the 3xx status than silently dispatch nothing.
            response = await client.request(
                "POST", run_url, data=data, headers=headers, timeout=timeout, allow_redirects=False
            )
            break
        except (ClientOSError, ServerDisconnectedError) as e:
            # ClientOSError covers connection-refused (ClientConnectorError subclasses it) and
            # mid-send resets (ECONNRESET) — the repo's documented steady-state noise at high
            # concurrency; ServerDisconnectedError covers keepalive-reuse races.
            last_connect_error = e
            if num_try < _EXTERNAL_AGENT_MAX_TRIES:
                await asyncio.sleep(_EXTERNAL_AGENT_RETRY_SLEEP_SECS)
        except asyncio.TimeoutError:
            return _external_agent_failure_result(
                run_url,
                f"timed out after {timeout_secs}s (agent_run_timeout_secs; raise it if your rollouts "
                "legitimately run longer)",
            )
        except Exception as e:
            return _external_agent_failure_result(run_url, f"{type(e).__name__}: {e}")
    if response is None:
        return _external_agent_failure_result(
            run_url,
            f"could not reach the agent after {_EXTERNAL_AGENT_MAX_TRIES} tries "
            f"({type(last_connect_error).__name__}: {last_connect_error}). Is your agent running at this URL?",
        )

    # client.request() returns once response HEADERS arrive; the body read below can still
    # raise (ClientPayloadError on a mid-body disconnect, TimeoutError if the total deadline
    # expires while streaming) and must honor the same never-raise contract.
    try:
        content = await response.read()
    except Exception as e:
        return _external_agent_failure_result(run_url, f"reading the response body failed: {type(e).__name__}: {e}")
    # aiohttp's response.ok is `status < 400`, so 3xx must be rejected explicitly — redirects
    # are not followed (a followed redirect would silently turn the POST into a body-less GET).
    if not response.ok or response.status >= 300:
        location = response.headers.get("Location", "")
        return _external_agent_failure_result(
            run_url,
            f"HTTP {response.status}"
            + (f" (redirect to {location}; fix agent_url to point at the final address)" if location else "")
            + f": {content[:500].decode(errors='replace')}",
        )
    try:
        result = orjson.loads(content)
    except orjson.JSONDecodeError as e:
        return _external_agent_failure_result(run_url, f"response is not valid JSON: {e}")
    if not isinstance(result, dict):
        return _external_agent_failure_result(
            run_url, f"expected a JSON object from /run, got {type(result).__name__}"
        )
    # A "success" missing reward/response would be written to the main jsonl and crash later
    # readers (profiling reads result["response"]). Failure rows reported via the sentinel
    # keys legitimately carry no reward, so they are exempt.
    if result.get(NG_FAILURE_CLASS_KEY) is None and not result.get(NG_NO_PERSIST_KEY):
        missing_keys = [key for key in ("reward", "response") if key not in result]
        if missing_keys:
            return _external_agent_failure_result(
                run_url,
                f"/run response is missing required key(s) {missing_keys}; external agents must return "
                "a verify-response-shaped object with at least 'reward' (float) and 'response' (the "
                "Responses API response object)",
            )
        # Presence is not enough: "response": null or a non-dict passes `in` but crashes
        # profiling/aggregation later, and a null reward corrupts metrics.
        if not isinstance(result["response"], dict) or not isinstance(result["reward"], (int, float)):
            return _external_agent_failure_result(
                run_url,
                f"/run response has invalid types: 'reward' must be a number "
                f"(got {type(result['reward']).__name__}) and 'response' a JSON object "
                f"(got {type(result['response']).__name__})",
            )
    return result


async def _post_external_aggregate_metrics(
    agent_url: str, agg_request: AggregateMetricsRequest
) -> Optional[AggregateMetrics]:
    """POST /aggregate_metrics to an external agent; returns None (with a warning) on any failure.

    External agents are not required to implement /aggregate_metrics, and by the time this runs
    the rollouts are already safely on disk — so nothing here is allowed to crash the run.
    """
    url = f"{agent_url}/aggregate_metrics"
    client = get_global_aiohttp_client()
    try:
        response = await client.request(
            "POST",
            url,
            data=orjson.dumps(agg_request.model_dump()),
            headers={"Content-Type": "application/json"},
            timeout=ClientTimeout(total=_EXTERNAL_AGGREGATE_TIMEOUT_SECS),
            allow_redirects=False,
        )
        # Reading the body releases the pooled connection — do it before any early return.
        content = await response.read()
        if response.status in (404, 405, 501):
            print(
                f"External agent {agent_url} does not implement /aggregate_metrics "
                f"(HTTP {response.status}); skipping aggregate metrics for it."
            )
            return None
        # response.ok is `status < 400`; 3xx (redirects are not followed) must also skip.
        if not response.ok or response.status >= 300:
            print(
                f"Skipping aggregate metrics for external agent {agent_url}: "
                f"HTTP {response.status}: {content[:500].decode(errors='replace')}"
            )
            return None
        return AggregateMetrics.model_validate(orjson.loads(content))
    except Exception as e:
        print(f"Skipping aggregate metrics for external agent {agent_url}: {type(e).__name__}: {e}")
        return None


def _agent_metric_label(agent_ref: Dict[str, Any]) -> str:
    """Label embedded between '/' separators in wandb metric keys. wandb sections metric
    names on '/', so a raw URL would inject bogus nesting levels; ':' is replaced purely for
    a clean flat label. URL agents reduce to `host_port[_path]` — the path is included so two
    agents behind one gateway keep distinct labels. The untouched ref still identifies them
    in artifacts."""
    url = agent_ref.get("url")
    if url:
        parsed = urlparse(url)
        return (parsed.netloc + parsed.path).replace(":", "_").replace("/", "_")
    return agent_ref["name"]


class SharedRolloutCollectionConfig(BaseNeMoGymCLIConfig):
    output_jsonl_fpath: str = Field(description="The output data jsonl file path.")
    num_samples_in_parallel: Optional[int] = Field(
        default=None, description="Limit the number of concurrent samples running at once."
    )
    responses_create_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Overrides for the responses_create_params e.g. temperature, max_output_tokens, etc.",
    )
    upload_rollouts_to_wandb: bool = Field(
        default=True,
        description="Upload the rollouts to W&B. Sometimes this should be off because the rollouts are massive. Default: True",
    )
    disable_aggregation: bool = Field(
        default=False,
        description=(
            "Skip the post-rollout aggregate-metrics computation and file write. "
            "Used when sharding rollouts across multiple jobs that will be aggregated together "
            "afterward by `gym eval aggregate`."
        ),
    )
    rollout_collection_driver: Optional[str] = Field(
        default=None,
        description=(
            "Optional dotted ``module.path:function`` to run rollout collection instead of the "
            "built-in helper. Lets a benchmark plug in a custom procedure (e.g. an adaptive, "
            "multi-pass run) while still producing the standard rollout + aggregate-metrics "
            "artifacts. The function is awaited with (rollout_collection_config, global_config_dict). "
            "When unset, the standard single-pass collection runs."
        ),
    )


class E2ERolloutCollectionConfig(SharedRolloutCollectionConfig):
    """
    Spin up all necessary servers and perform a batch of rollout collection using each dataset inside the provided configs.

    Examples:

    ```bash
    gym eval run \
        +output_jsonl_fpath=weather_rollouts.jsonl \
        +num_samples_in_parallel=10
    ```
    """

    split: Union[Literal["train"], Literal["validation"], Literal["benchmark"]]
    reuse_existing_data_preparation: bool = False


class RolloutCollectionConfig(SharedRolloutCollectionConfig):
    """
    Perform a batch of rollout collection.

    Examples:

    ```bash
    gym eval run --no-serve \
        +agent_name=example_single_tool_call_simple_agent \
        +input_jsonl_fpath=weather_query.jsonl \
        +output_jsonl_fpath=weather_rollouts.jsonl \
        +limit=100 \
        +num_repeats=4 \
        +num_samples_in_parallel=10
    ```
    """

    # Validation errors must not echo raw input: agent_url may carry credentials the
    # user was told not to embed, and the error itself would print them to the terminal.
    model_config = ConfigDict(hide_input_in_errors=True)

    agent_name: Optional[str] = Field(
        default=None,
        description="The agent to collect rollouts from. If not specified, uses agent_ref from each data row.",
    )
    agent_url: Optional[str] = Field(
        default=None,
        description=(
            "URL of an external agent server to collect rollouts from, e.g. http://localhost:9000. "
            "The collector POSTs each row — including verifier_metadata, i.e. the task's answer key — "
            "directly to {agent_url}/run, so only point this at an agent you trust. The endpoint lives "
            "outside Gym's config-managed process tree, which is why this is a URL rather than a server "
            "ref. Mutually exclusive with agent_name."
        ),
    )
    agent_run_timeout_secs: float = Field(
        default=_DEFAULT_AGENT_RUN_TIMEOUT_SECS,
        description=(
            "Wallclock bound on a single external-agent /run request (only used with agent_url; Gym-managed "
            "agents are unaffected). Requests exceeding it are recorded in the failures sidecar and retried "
            "on resume. Raise it if your agent's rollouts legitimately run longer than 30 minutes."
        ),
    )
    input_jsonl_fpath: str = Field(
        description="The input data source to use to collect rollouts, in the form of a file path to a jsonl file."
    )
    limit: Optional[int] = Field(
        default=None, description="Maximum number of examples to load and take from the input dataset."
    )
    num_repeats: Union[int, Dict[str, int]] = Field(
        default=1,
        description=(
            "How many times to repeat each example. Either an int (applied to every row) or a "
            "dict keyed by agent identity — agent_ref.name for Gym-managed agents, or the agent URL "
            "for external agents (e.g. {simple_agent: 32, 'http://localhost:9000': 1}). In dict form, "
            "every agent that appears in the input rows must have an entry, unless a special "
            '"_default" key is provided as a fallback. Useful for mean@k.'
        ),
    )
    num_repeats_add_seed: bool = Field(
        default=False,
        description='When num_repeats > 1, pass a per-rollout "seed" via metadata.extra_body (honored by vLLM model servers).',
    )
    resume_from_cache: bool = Field(
        default=False,
        description="If the same command is run multiple times, check the materialized inputs and current outputs and remove the inputs that have already been run",
    )
    prompt_config: Optional[str] = Field(
        default=None,
        description="Path to a prompt YAML file. Builds responses_create_params.input from the template at rollout time. Mutually exclusive with pre-populated responses_create_params.input in the JSONL data.",
    )
    skills: Optional[SkillsConfig] = Field(
        default=None,
        description="Run-level skills config (skills.path). Makes a directory of Agent Skills standard skills available to the agent at rollout time and stamps each result with a skills_ref. Applied to a skill-agnostic dataset; not a dataset-row field.",
    )

    @field_validator("num_repeats", mode="before")
    @classmethod
    def _coerce_null_num_repeats(cls, v):
        # default to 1 if num_repeats is None
        # for backwards compatibility
        return 1 if v is None else v

    @field_validator("agent_url")
    @classmethod
    def _normalize_agent_url_field(cls, v: Optional[str]) -> Optional[str]:
        return _normalize_agent_url(v) if v else v

    @model_validator(mode="after")
    def _validate_agent_selection(self) -> "RolloutCollectionConfig":
        if self.agent_name and self.agent_url:
            raise ValueError(
                "agent_name and agent_url are mutually exclusive. Use +agent_name for a Gym-managed agent "
                "server, or +agent_url to dispatch to an external agent endpoint — not both."
            )
        if self.agent_run_timeout_secs <= 0:
            raise ValueError(f"agent_run_timeout_secs must be > 0, got {self.agent_run_timeout_secs}")
        return self

    @model_validator(mode="after")
    def _validate_num_repeats(self) -> "RolloutCollectionConfig":
        nr = self.num_repeats
        if isinstance(nr, int):
            if nr < 1:
                raise ValueError(f"num_repeats must be >= 1, got {nr}")
        else:
            bad = {name: n for name, n in nr.items() if n < 1}
            if bad:
                raise ValueError(f"num_repeats dict values must be >= 1, got {bad}")
        return self

    @property
    def materialized_jsonl_fpath(self) -> Path:
        output_fpath = Path(self.output_jsonl_fpath)
        return output_fpath.with_stem(output_fpath.stem + "_materialized_inputs").with_suffix(".jsonl")


def _rollout_request_debug_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        TASK_INDEX_KEY_NAME: row.get(TASK_INDEX_KEY_NAME),
        ROLLOUT_INDEX_KEY_NAME: row.get(ROLLOUT_INDEX_KEY_NAME),
        "agent_name": row_agent_key(row),
    }
    return {k: v for k, v in summary.items() if v is not None}


class RolloutCollectionHelper(BaseModel):
    def _preprocess_rows_from_config(self, config: RolloutCollectionConfig) -> List[Dict]:
        range_iterator = repeat(0)
        if config.limit:
            range_iterator = range(config.limit)
            print(f"Limiting the number of rows to {config.limit}")

        if config.num_repeats_add_seed:
            print(
                "Adding unique `seed` values to each input via metadata.extra_body (only honored by vLLM model servers)"
            )

        if config.agent_name:
            print(f"Using `{config.agent_name}` for rows that do not already have an agent ref")

        if config.agent_url:
            print(f"Using external agent `{config.agent_url}` for rows that do not already have an agent ref")

        if config.responses_create_params:
            print(f"Overriding responses_create_params fields with {config.responses_create_params}")
            responses_create_params_overrides = OmegaConf.to_container(
                OmegaConf.create(config.responses_create_params), resolve=True
            )
        else:
            responses_create_params_overrides = dict()

        if isinstance(config.num_repeats, int):
            fixed_num_repeats: Optional[int] = config.num_repeats
            per_agent_repeats: Dict[str, int] = {}
            default_repeats: Optional[int] = None
            print(f"Repeating rows {fixed_num_repeats} times (in a pattern of abc to aabbcc)!")
        else:
            fixed_num_repeats = None
            # URL keys get the same strip/rstrip("/") normalization as agent_url and row refs,
            # so the exact string a user passed to +agent_url always matches its dict entry.
            per_agent_repeats = {
                (k.strip().rstrip("/") if isinstance(k, str) and k.startswith(("http://", "https://")) else k): v
                for k, v in config.num_repeats.items()
                if k != "_default"
            }
            default_repeats = config.num_repeats.get("_default")
            print(f"Per-agent num_repeats: {dict(config.num_repeats)}")
        agents_seen: set[str] = set()

        # Load prompt config if specified
        prompt_cfg = None
        if config.prompt_config:
            prompt_cfg = load_prompt_config(config.prompt_config)
            print(f"Using prompt config: {config.prompt_config}")

        # Resolve skills once for the whole run (hash is content-derived, computed at startup).
        skills_ref_dict = None
        if config.skills:
            skills_ref = load_skill_directory(config.skills.path)
            skills_ref_dict = skills_ref.model_dump()
            print(
                f"Using skills from {config.skills.path} "
                f"(hash={skills_ref.hash}, {len(skills_ref.skills)} skill(s): "
                f"{', '.join(s.name for s in skills_ref.skills)})"
            )

        _input_path = Path(config.input_jsonl_fpath)
        if not _input_path.is_absolute():
            _cwd_path = Path.cwd() / _input_path
            _input_path = _cwd_path if _cwd_path.exists() else PARENT_DIR / _input_path
        if not _input_path.exists():
            raise ConfigPathNotFoundError(
                f"Input file not found: '{config.input_jsonl_fpath}' (--input). Check the path is spelled correctly."
            )
        with open(_input_path) as input_file:
            rows_iterator: Iterator[str] = tqdm(input_file, desc="Reading rows")
            rows_iterator: Iterator[tuple[int, str]] = zip(range_iterator, rows_iterator)
            raw_rows = [
                (row_idx, row_str, loads_jsonl_line(row_str, _input_path, line_no))
                for line_no, (row_idx, row_str) in enumerate(rows_iterator, 1)
            ]

        # Validate and apply prompt config before per-row processing
        if prompt_cfg is not None:
            validate_prompt_compatibility([row for _, _, row in raw_rows], prompt_cfg)
            raw_rows = [(idx, s, apply_prompt_to_row(row, prompt_cfg)) for idx, s, row in raw_rows]

        # For gym eval profile to match rollouts to tasks
        row_to_task_idx: Dict[str, int] = dict()
        task_idx_to_rollout_idx: Dict[int, int] = Counter()
        row_idxs_missing_agent_ref: List[int] = []
        agents_missing_from_num_repeats: set[str] = set()
        rows: List[Dict] = []
        # The first tuple element is the limit counter (all zeros when no limit is set) —
        # data_row_idx is the row's real position for error messages.
        for data_row_idx, (_, row_str, row) in enumerate(raw_rows):
            # Resolve the agent identity. Missing agent_ref is a hard error reported in
            # bulk after the loop; skip the row immediately so the rest of the
            # body can assume agent_key is non-None.
            if config.agent_name:
                row.setdefault(AGENT_REF_KEY_NAME, {"name": config.agent_name})
            elif config.agent_url:
                row.setdefault(AGENT_REF_KEY_NAME, {"url": config.agent_url})

            agent_ref = row.get(AGENT_REF_KEY_NAME) or {}
            row_url = agent_ref.get("url") if isinstance(agent_ref, dict) else None
            if row_url is not None:
                if agent_ref.get("name"):
                    raise ValueError(
                        f"Row {data_row_idx} agent_ref carries both 'name' and 'url' ({agent_ref!r}); "
                        "an agent ref must be exactly one of the two."
                    )
                # A dataset must not be able to route rows (and their answer keys) to an
                # arbitrary host: row-level urls are honored only when they match +agent_url.
                normalized_row_url = _normalize_agent_url(str(row_url))
                if normalized_row_url != config.agent_url:
                    raise ValueError(
                        f"Row {data_row_idx} carries agent_ref.url={row_url!r}, which does not match the "
                        f"configured +agent_url ({config.agent_url!r}). Row-level agent URLs are only "
                        "honored when they match the configured agent_url."
                    )
                agent_ref["url"] = normalized_row_url

            agent_key = row_agent_key(row)
            if agent_key is None:
                row_idxs_missing_agent_ref.append(data_row_idx)
                continue
            agents_seen.add(agent_key)

            # Responses create params
            row[RESPONSES_CREATE_PARAMS_KEY_NAME] = (
                row[RESPONSES_CREATE_PARAMS_KEY_NAME] | responses_create_params_overrides
            )

            # Stamp the run-level skills_ref onto the row so it is sent to the agent in the
            # /run request body and propagated to results. The source dataset stays untouched.
            if skills_ref_dict is not None:
                row[SKILLS_REF_KEY_NAME] = skills_ref_dict

            # Resolve task index. Honor a caller-provided value when present (e.g. when an
            # upstream slicer has stamped a globally-stable index across chunks so that
            # subsequent /aggregate_metrics groupby unions chunks correctly); otherwise dedupe
            # identical input rows to the same task index as before.
            if TASK_INDEX_KEY_NAME not in row:
                row[TASK_INDEX_KEY_NAME] = row_to_task_idx.setdefault(row_str, len(row_to_task_idx))

            # Resolve num_repeats for this row, batching dict-form misses for
            # one consolidated raise after the loop.
            if fixed_num_repeats is not None:
                row_num_repeats = fixed_num_repeats
            elif agent_key in per_agent_repeats:
                row_num_repeats = per_agent_repeats[agent_key]
            elif default_repeats is not None:
                row_num_repeats = default_repeats
            else:
                agents_missing_from_num_repeats.add(agent_key)
                continue

            for _ in range(row_num_repeats):
                row = deepcopy(row)

                # Resolve rollout index
                row[ROLLOUT_INDEX_KEY_NAME] = task_idx_to_rollout_idx[row[TASK_INDEX_KEY_NAME]]
                task_idx_to_rollout_idx[row[TASK_INDEX_KEY_NAME]] += 1

                if config.num_repeats_add_seed:
                    metadata = row[RESPONSES_CREATE_PARAMS_KEY_NAME].setdefault("metadata", {})
                    extra_body = json.loads(metadata.get("extra_body", "{}"))
                    extra_body["seed"] = row[ROLLOUT_INDEX_KEY_NAME]
                    metadata["extra_body"] = json.dumps(extra_body)

                rows.append(row)

        if row_idxs_missing_agent_ref:
            raise ValueError(
                f"No agent specified for rows {row_idxs_missing_agent_ref}. Provide +agent_name (Gym-managed "
                "agent) or +agent_url (external agent endpoint), or include agent_ref in data."
            )

        if config.agent_url and config.agent_url not in agents_seen:
            warnings.warn(
                f"agent_url={config.agent_url} was provided, but every input row already carries its own "
                "agent_ref, so nothing will be dispatched to the external agent. Row-level refs always win "
                "over the config default (same semantics as agent_name).",
                stacklevel=2,
            )

        if agents_missing_from_num_repeats:
            raise ValueError(
                f"num_repeats dict has no entry for agents {sorted(agents_missing_from_num_repeats)} "
                f"and no '_default' fallback. Listed agents: {sorted(per_agent_repeats)}"
            )

        unknown_agents = set(per_agent_repeats) - agents_seen
        if unknown_agents:
            warnings.warn(
                f"num_repeats dict contains agent names that never appeared in input rows "
                f"(possible typo?): {sorted(unknown_agents)}",
                stacklevel=2,
            )

        return rows

    def _load_from_cache(
        self, config: RolloutCollectionConfig
    ) -> Tuple[List[Dict], List[Dict], List[Dict], List[List[str]]]:
        with config.materialized_jsonl_fpath.open() as f:
            original_input_rows = list(map(orjson.loads, f))
        with Path(config.output_jsonl_fpath).open("rb") as f:
            result_strs = [[line.strip()] for line in f]
        results = [orjson.loads(p[0]) for p in result_strs]

        get_key = lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME])

        # Successes (and any legacy '-failed' rows written by pre-fix Gym
        # builds) live in the main jsonl. They short-circuit dispatch.
        successes_seen = set(map(get_key, results))

        # Sidecar: one row per non-kill_shaped failure attempt. Count attempts
        # per key + flag terminal rows so chain-hop 2 retries the right ones.
        failures_fpath = _failures_path_for(Path(config.output_jsonl_fpath))
        attempts_by_key: Counter = Counter()
        terminal_keys: set = set()
        if failures_fpath.exists():
            with failures_fpath.open("rb") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    fr = orjson.loads(line)
                    if TASK_INDEX_KEY_NAME not in fr or ROLLOUT_INDEX_KEY_NAME not in fr:
                        continue
                    k = (fr[TASK_INDEX_KEY_NAME], fr[ROLLOUT_INDEX_KEY_NAME])
                    attempts_by_key[k] += 1
                    if fr.get(NG_TERMINAL_KEY):
                        terminal_keys.add(k)

        max_attempts = _get_max_rollout_attempts()
        maxed_out = {k for k, n in attempts_by_key.items() if n >= max_attempts}
        gated = successes_seen | terminal_keys | maxed_out

        input_rows = [row for row in original_input_rows if get_key(row) not in gated]

        key_to_row = dict(zip(map(get_key, original_input_rows), original_input_rows))
        rows = [key_to_row[get_key(result)] for result in results]

        print(
            f"""Resumed from cache. Found:
- {len(original_input_rows)} original input rows
- {len(rows)} rows already done (in main jsonl)
- {sum(attempts_by_key.values())} prior failure attempts ({len(attempts_by_key)} unique tasks) in sidecar
- {len(terminal_keys)} sidecar-terminal (timeout_exceeded / skipped) → not retried
- {len(maxed_out)} hit max_attempts={max_attempts} → not retried
- {len(input_rows)} rows that still need to be run"""
        )

        return input_rows, rows, results, result_strs

    async def run_from_config(self, config: RolloutCollectionConfig) -> Tuple[List[Dict]]:
        output_fpath = Path(config.output_jsonl_fpath)

        if config.resume_from_cache and config.materialized_jsonl_fpath.exists() and output_fpath.exists():
            (
                input_rows,
                rows,
                results,
                result_strs,
            ) = self._load_from_cache(config)

            # Named refs re-resolve to host:port at request time, so they survive server restarts
            # across resume hops. URL refs are frozen into the materialized inputs — re-stamp them
            # from config so a changed +agent_url redirects the remaining rows instead of silently
            # dispatching to the old address.
            if config.agent_url:

                def _restamp_url_rows(url_rows: List[Dict]) -> int:
                    num_changed = 0
                    for row in url_rows:
                        agent_ref = row.get(AGENT_REF_KEY_NAME) or {}
                        if isinstance(agent_ref, dict) and agent_ref.get("url") not in (None, config.agent_url):
                            row[AGENT_REF_KEY_NAME] = {"url": config.agent_url}
                            num_changed += 1
                    return num_changed

                num_restamped = _restamp_url_rows(input_rows)
                # Completed rows keep their historical refs in the on-disk artifacts, but these
                # in-memory copies drive aggregate-metrics grouping — re-key them too so one live
                # agent aggregates all hops instead of POSTing hop-1 metrics to the dead old URL.
                num_completed_rekeyed = _restamp_url_rows(rows)
                if num_restamped or num_completed_rekeyed:
                    print(
                        f"Re-pointed {num_restamped} pending and {num_completed_rekeyed} completed rows "
                        f"to agent_url={config.agent_url}"
                    )
            else:
                # Fresh runs validate every row URL against +agent_url; resume applies the same
                # rule. Completed rows count too: even with nothing left to dispatch,
                # aggregation would POST result payloads to the frozen URL.
                frozen_urls = sorted(
                    {
                        agent_ref["url"]
                        for row in [*input_rows, *rows]
                        if isinstance(agent_ref := (row.get(AGENT_REF_KEY_NAME) or {}), dict) and agent_ref.get("url")
                    }
                )
                if frozen_urls:
                    raise ValueError(
                        f"Resuming with external-agent rows frozen to {frozen_urls}, but +agent_url "
                        "was not provided. Pass +agent_url=<url> to confirm where these rows (including "
                        "their verifier_metadata) should be dispatched."
                    )
        else:
            if config.resume_from_cache:
                if not output_fpath.exists():
                    print(f"Skipping resume_from_cache because output_fpath {output_fpath} doesn't exist!")
                if not config.materialized_jsonl_fpath.exists():
                    print(
                        f"Skipping resume_from_cache because materialized_jsonl_fpath {config.materialized_jsonl_fpath} doesn't exist!"
                    )
            else:
                print("Clearing output fpath since `resume_from_cache=False`!")

            rows: List[Dict] = []
            results: List[Dict] = []
            result_strs: List[List[str]] = []

            input_rows = self._preprocess_rows_from_config(config)
            # Returned rows are sorted by (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME])

            with config.materialized_jsonl_fpath.open("wb") as f:
                for row in input_rows:
                    f.write(orjson.dumps(row) + b"\n")

            output_fpath.unlink(missing_ok=True)
            # A stale sidecar from a previous fresh run would pre-consume resume retry
            # attempts for this run's task/rollout keys.
            _failures_path_for(output_fpath).unlink(missing_ok=True)

        num_concurrent_samples = config.num_samples_in_parallel

        # Every request to a single agent_url shares one per-host connection pool, and time
        # spent waiting for a connection counts against agent_run_timeout_secs — dispatching
        # beyond the pool limit converts queue wait into spurious timeouts.
        # NOTE: the semaphore below gates ALL rows, so in a mixed named+url run the cap also
        # bounds named-agent dispatch — a deliberate, conservative simplification.
        has_url_rows = any(_is_external_url_ref(row.get(AGENT_REF_KEY_NAME)) for row in input_rows)
        if has_url_rows:
            per_host_limit = _effective_per_host_connection_limit()
            if per_host_limit is not None and (num_concurrent_samples or per_host_limit + 1) > per_host_limit:
                print(
                    f"Capping concurrency at {per_host_limit} (the shared HTTP client's per-host connection "
                    f"limit) for external-agent dispatch"
                    + (
                        f"; requested num_samples_in_parallel={num_concurrent_samples}"
                        if num_concurrent_samples
                        else ""
                    )
                    + "."
                )
                num_concurrent_samples = per_host_limit

        semaphore = nullcontext()
        if num_concurrent_samples:
            print(f"Querying with {num_concurrent_samples} concurrent requests")
            semaphore = Semaphore(num_concurrent_samples)

        output_fpath.parent.mkdir(exist_ok=True, parents=True)
        failures_fpath = _failures_path_for(output_fpath)

        pcts_to_print = [20, 40, 60, 80, 90, 95, 98, 99, 100]
        counts_left = Counter(map(row_agent_key, input_rows))
        global _NUM_EXTERNAL_AGENT_FAILURES
        _NUM_EXTERNAL_AGENT_FAILURES = 0
        num_failures_this_run = 0
        num_no_persist_this_run = 0
        results_file = output_fpath.open("ab")
        failures_file = failures_fpath.open("ab")
        for future in self.run_examples(
            input_rows, semaphore=semaphore, agent_run_timeout_secs=config.agent_run_timeout_secs
        ):
            row, result = await future

            result[TASK_INDEX_KEY_NAME] = row[TASK_INDEX_KEY_NAME]
            result[ROLLOUT_INDEX_KEY_NAME] = row[ROLLOUT_INDEX_KEY_NAME]
            result[AGENT_REF_KEY_NAME] = row[AGENT_REF_KEY_NAME]
            if SKILLS_REF_KEY_NAME in row:
                result[SKILLS_REF_KEY_NAME] = row[SKILLS_REF_KEY_NAME]

            no_persist = bool(result.get(NG_NO_PERSIST_KEY))
            failure_class = result.get(NG_FAILURE_CLASS_KEY)

            rows.append(row)
            results.append(result)
            serialized = orjson.dumps(result)
            result_strs.append([serialized])

            if no_persist:
                # kill_shaped: don't write anywhere. Set-difference on resume
                # naturally re-dispatches; per-task timeout bounds wallclock.
                num_no_persist_this_run += 1
            elif failure_class is not None:
                # Non-kill_shaped failure → sidecar. The aggregator only reads
                # the main jsonl, so this keeps win-rate uncontaminated.
                num_failures_this_run += 1
                failures_file.write(serialized + b"\n")
                failures_file.flush()
            else:
                # Success → main jsonl.
                results_file.write(serialized + b"\n")
                results_file.flush()

            agent_key = row_agent_key(row)
            counts_left[agent_key] -= 1
            if counts_left[agent_key] <= 0:
                counts_left.pop(agent_key)

            current_pct = 100 * len(results) / len(input_rows)
            if pcts_to_print and current_pct >= pcts_to_print[0]:
                while pcts_to_print and current_pct >= pcts_to_print[0]:
                    pcts_to_print.pop(0)

                top_left = counts_left.most_common(5)  # Fix to top 3 for now.
                if top_left:
                    top_left_str = "\n".join(f"{i + 1}. {k}: {v}" for i, (k, v) in enumerate(top_left))
                    # Use tqdm.write here so we can print properly with tqdm being used.
                    tqdm.write(f"Examples left:\n{top_left_str}")

        results_file.close()
        failures_file.close()

        if config.upload_rollouts_to_wandb and get_wandb_run():  # pragma: no cover
            print("Uploading rollouts to W&B. This may take a few minutes if your data is large.")
            get_wandb_run().log({"Rollouts": Table(data=result_strs, columns=["Rollout"])})
        del result_strs

        print("Sorting results to ensure consistent ordering")
        rows.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))
        results.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))

        # Compute and write aggregate metrics via /aggregate_metrics on each agent server
        if config.disable_aggregation:
            print(
                "Skipping aggregate-metrics computation because disable_aggregation=True. "
                "Run `gym eval aggregate` after all shards finish to compute the global metrics."
            )
            aggregate_metrics_fpath = None
        else:
            print("Computing aggregate metrics")
            aggregate_metrics_fpath = await self._call_aggregate_metrics(results, rows, output_fpath)

        if num_failures_this_run or num_no_persist_this_run:
            print(
                f"WARNING: {num_failures_this_run} rollout(s) failed this run (full error rows in "
                f"{failures_fpath}) and {num_no_persist_this_run} were dropped without a record; "
                "the outputs below cover fewer rollouts than were dispatched. Re-run with "
                "+resume_from_cache=true to retry."
            )

        print(f"""Finished rollout collection! View results at:
Fully materialized inputs: {config.materialized_jsonl_fpath}
Rollouts: {output_fpath}
Aggregate metrics: {aggregate_metrics_fpath}""")

        return results

    async def _call_aggregate_metrics(
        self,
        results: List[Dict],
        rows: List[Dict],
        output_fpath: Path,
    ) -> Optional[Path]:
        """Call /aggregate_metrics on each agent server after rollouts complete.

        Writes a single _aggregate_metrics.json with one entry per agent (same shape
        as the old _agent_metrics.json). Returns the file path.
        """
        if not results:
            return None

        # Group results by agent identity (server name, or URL for external agents).
        # Failure/no-persist rows live in the sidecar, not the main jsonl — exclude them here so
        # in-run aggregation computes over the same rows `gym eval aggregate` would read back,
        # and so /aggregate_metrics never receives reward-less failure stubs.
        agent_results: Dict[str, List[Dict]] = {}
        agent_refs: Dict[str, Dict] = {}
        for row, result in zip(rows, results):
            if result.get(NG_FAILURE_CLASS_KEY) is not None or result.get(NG_NO_PERSIST_KEY):
                continue
            agent_key = row_agent_key(row)
            if not agent_key:
                continue
            agent_results.setdefault(agent_key, []).append(result)
            agent_refs.setdefault(agent_key, row.get(AGENT_REF_KEY_NAME) or {})

        if not agent_results:
            print("No successful rollouts to aggregate; skipping aggregate metrics.")
            return None

        # A ServerClient requires a live head server; only construct one if some agent needs
        # name resolution (name-first, matching row_agent_key: a ref carrying both keys is a
        # named agent). Pure agent_url runs work with no Gym servers.
        needs_named = any(ref.get("name") or not ref.get("url") for ref in agent_refs.values())
        server_client = self.setup_server_client() if needs_named else None

        async def _fetch_agent_metrics(agent_key: str, agent_result_list: List[Dict]) -> Optional[Dict]:
            # Strip heavyweight fields before sending, but preserve response.usage
            stripped = []
            for r in agent_result_list:
                entry = {k: v for k, v in r.items() if k not in ("response", "responses_create_params")}
                usage = (r.get("response") or {}).get("usage")
                if usage:
                    entry["response"] = {"usage": usage}
                stripped.append(entry)

            agg_request = AggregateMetricsRequest(verify_responses=stripped)
            agent_ref = agent_refs[agent_key]
            if agent_ref.get("url") and not agent_ref.get("name"):
                agg_result = await _post_external_aggregate_metrics(agent_ref["url"], agg_request)
                if agg_result is None:
                    return None
            else:
                agg_response = await server_client.post(
                    server_name=agent_key,
                    url_path="/aggregate_metrics",
                    json=agg_request,
                )
                await raise_for_status(agg_response)
                agg_result = AggregateMetrics.model_validate(await get_response_json(agg_response))

            agent_entry = {
                AGENT_REF_KEY_NAME: canonical_agent_ref(agent_ref, agent_key),
                "agent_metrics": agg_result.agent_metrics,
                "key_metrics": agg_result.key_metrics,
                "group_level_metrics": agg_result.group_level_metrics,
            }
            return agent_entry

        all_agent_metrics: List[Dict] = []
        tasks = [_fetch_agent_metrics(key, results_list) for key, results_list in agent_results.items()]
        for coro in asyncio.as_completed(tasks):
            agent_entry = await coro
            if agent_entry is None:
                continue
            all_agent_metrics.append(agent_entry)

            agent_identity = row_agent_key({AGENT_REF_KEY_NAME: agent_entry[AGENT_REF_KEY_NAME]})
            key_metrics = agent_entry.get("key_metrics", {})
            print(f"\nKey metrics for {agent_identity}:\n" + json.dumps(key_metrics, indent=4))

        primitive_types = (bool, int, float, str, type(None))
        metrics_to_log = dict()
        for agent_entry in all_agent_metrics:
            agent_label = _agent_metric_label(agent_entry[AGENT_REF_KEY_NAME])
            metrics_to_log.update(
                {
                    f"{agent_label}/{k}": v
                    for k, v in agent_entry["agent_metrics"].items()
                    if isinstance(v, primitive_types)
                }
            )
            metrics_to_log.update(
                {
                    f"key_metrics/{agent_label}/{k}": v
                    for k, v in agent_entry["key_metrics"].items()
                    if isinstance(v, primitive_types)
                }
            )

        if get_wandb_run():  # pragma: no cover
            get_wandb_run().log(metrics_to_log)

        # Write single file with all agents
        metrics_fpath = output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")
        metrics_fpath.write_bytes(orjson.dumps(all_agent_metrics, option=orjson.OPT_INDENT_2))

        return metrics_fpath

    def run_examples(
        self,
        examples: List[Dict],
        head_server_config: Optional[BaseServerConfig] = None,
        semaphore: Optional[Semaphore] = None,
        agent_run_timeout_secs: float = _DEFAULT_AGENT_RUN_TIMEOUT_SECS,
    ) -> Iterator[Future]:  # pragma: no cover
        """
        We provide this function as a lower level interface for running rollout collection.
        """
        # Only construct a ServerClient (which needs a live head server) when a named row exists.
        server_client = self.setup_server_client(head_server_config) if _rows_need_named_dispatch(examples) else None
        semaphore = semaphore or nullcontext()

        async def _post_subroutine(row: Dict) -> Tuple[Dict, Dict]:
            async with semaphore:
                if _is_external_url_ref(row.get(AGENT_REF_KEY_NAME)):
                    return row, await _post_external_agent_run(row, agent_run_timeout_secs)
                res = await server_client.post(server_name=row["agent_ref"]["name"], url_path="/run", json=row)
                try:
                    await raise_for_status(res)
                except Exception:
                    if is_global_aiohttp_client_request_debug_enabled():
                        print(
                            "[rollout_collection] /run failed "
                            f"status={getattr(res, 'status', None)} "
                            f"row={json.dumps(_rollout_request_debug_summary(row), sort_keys=True)}",
                            flush=True,
                        )
                    raise
                return row, await get_response_json(res)

        return tqdm.as_completed(
            map(_post_subroutine, examples),
            desc="Collecting rollouts",
            miniters=10,
            total=len(examples),
            maxinterval=60,
        )

    def setup_server_client(
        self, head_server_config: Optional[BaseServerConfig] = None
    ) -> ServerClient:  # pragma: no cover
        server_client = ServerClient.load_from_global_config(head_server_config)

        # We set this rollout global aiohttp client to use the same max connections as the underlying head server global config.
        if not is_global_aiohttp_client_setup():
            set_global_aiohttp_client(
                cfg=GlobalAIOHTTPAsyncClientConfig.model_validate(server_client.global_config_dict)
            )

        return server_client


class RolloutAggregationConfig(BaseNeMoGymCLIConfig):
    """
    Aggregate metrics across rollout shards produced by `gym eval run --no-serve +disable_aggregation=true`.

    Reads every JSONL file matching `input_glob`, computes aggregate metrics by POSTing to each
    agent server's `/aggregate_metrics` endpoint over the global union of records, and writes a
    single `<output_jsonl_fpath stem>_aggregate_metrics.json` next to the rollouts. By default
    also concatenates all shards into `output_jsonl_fpath`.

    Examples:

    ```bash
    gym eval aggregate \
        "+config_paths=[benchmarks/aime24/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
        +input_glob='results/rollouts-rs*-chunk*.jsonl' \
        +output_jsonl_fpath=results/rollouts.jsonl
    ```
    """

    input_glob: str = Field(
        description=(
            "Glob pattern or comma-separated list of glob patterns matching the rollout shards "
            "to aggregate (e.g. 'results/rollouts-rs*-chunk*.jsonl' or "
            "'results/run1/rollouts.jsonl,results/run2/rollouts.jsonl'). Whitespace around "
            "commas is stripped. Duplicate matches across patterns are deduplicated."
        )
    )
    output_jsonl_fpath: str = Field(
        description=(
            "Path used to derive the aggregate-metrics output location "
            "('<stem>_aggregate_metrics.json' next to this path) and, when merge_shards=True, "
            "the merged-rollouts file."
        ),
    )
    merge_shards: bool = Field(
        default=True,
        description="Concatenate the matched shard JSONLs into output_jsonl_fpath alongside the metrics file.",
    )


def loads_jsonl_line(raw, fpath, line_no: int):
    """Parse one JSONL line, raising a clean `ConfigError` (naming file + line) on malformed JSON."""
    try:
        return orjson.loads(raw)
    except orjson.JSONDecodeError as e:
        raise ConfigError(f"Malformed JSON in '{fpath}' at line {line_no}: {e}") from e


def _expand_input_glob(input_glob: str) -> List[str]:
    """Expand a glob-or-comma-separated-globs string into a sorted, deduplicated list of paths.

    Examples:
      'results/rollouts.jsonl' -> ['results/rollouts.jsonl'] (if it exists)
      'a/*.jsonl, b/*.jsonl'   -> matches of both patterns, deduplicated
    """
    patterns = [p.strip() for p in input_glob.split(",") if p.strip()]
    seen: Dict[str, None] = {}  # preserve insertion order while deduping
    for pattern in patterns:
        for path in sorted(glob_module.glob(pattern)):
            seen.setdefault(path, None)
    return list(seen)


class RolloutAggregationHelper(BaseModel):
    async def run_from_config(self, config: RolloutAggregationConfig) -> Optional[Path]:
        input_paths = _expand_input_glob(config.input_glob)
        if not input_paths:
            raise ConfigPathNotFoundError(f"No shards matched input_glob={config.input_glob!r}")
        print(f"Aggregating {len(input_paths)} shard(s):")
        for p in input_paths:
            print(f"  - {p}")

        results: List[Dict] = []
        for shard_path in input_paths:
            with open(shard_path, "rb") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    results.append(loads_jsonl_line(line, shard_path, line_no))
        print(f"Loaded {len(results)} rollout record(s) from {len(input_paths)} shard(s)")

        # Sort for deterministic aggregation ordering (matches run_from_config's post-collection sort)
        results.sort(key=lambda r: (r.get(TASK_INDEX_KEY_NAME), r.get(ROLLOUT_INDEX_KEY_NAME)))

        output_fpath = Path(config.output_jsonl_fpath)
        output_fpath.parent.mkdir(parents=True, exist_ok=True)

        if config.merge_shards:
            print(f"Merging shards into {output_fpath}")
            with output_fpath.open("wb") as out:
                for r in results:
                    out.write(orjson.dumps(r) + b"\n")

        # `_call_aggregate_metrics` only inspects each row's AGENT_REF_KEY_NAME, which results already carry.
        helper = RolloutCollectionHelper()
        aggregate_metrics_fpath = await helper._call_aggregate_metrics(results, results, output_fpath)

        print(f"""Finished rollout aggregation! View results at:
Merged rollouts: {output_fpath if config.merge_shards else "<not merged>"}
Aggregate metrics: {aggregate_metrics_fpath}""")

        return aggregate_metrics_fpath


# Backward-compatibility shims (CLI refactor): these CLI entry points moved to `nemo_gym.cli.eval`.
# Re-exported lazily to avoid a circular import; accessing them emits a DeprecationWarning.
from nemo_gym.cli._compat import moved_attr_getter  # noqa: E402


__getattr__ = moved_attr_getter(
    __name__,
    {
        "collect_rollouts": "nemo_gym.cli.eval",
        "aggregate_rollouts": "nemo_gym.cli.eval",
    },
)
