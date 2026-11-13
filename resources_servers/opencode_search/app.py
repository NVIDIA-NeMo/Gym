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

"""Resources server for judge-graded search / BrowseComp-style QA rows.

Shape of the environment, and how it differs from resources_servers/swebench:

  seed_session  creates ONE general-purpose sandbox from a fixed image (no per-instance
                image derivation, no conda activation, no anti-cheat git scrub) for the
                session to work in.
  verify        has NO eval sandbox and runs no tests. It extracts the agent's final
                answer and grades it with an LLM judge, exactly as the reference
                simple-evals BrowseComp grader does.

Two kinds of seed reach it. An Environment Server's episode seed opens a session keyed by its
resources_session_id and lends the sandbox to a session harness as sandbox_access (only with
sandbox_workdir set); with plugin_log_path set, that episode's verify reads how the run ended
from the plugin log in the sandbox, and with episode_export_copy_dir set it copies the harness's
session exports out of the sandbox to this host. An agent's own /run seeds with the dataset row
and its session cookie and gets the sandbox back as sandbox_handle: that is how the unmodified
responses_api_agents/opencode_sandboxed_agent runs the benchmark, through a legacy_agent relay.
Either way the agent is told only the question, works inside the sandbox, and the reward comes
from the judge rather than from a test harness.
"""

import asyncio
import base64
import hashlib
import json
import logging
import re
import shlex
import shutil
import sys
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePosixPath
from time import time
from typing import Annotated, Any, ClassVar, Dict, List, Literal, Optional, Tuple

import aiohttp
import anyio
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BeforeValidator, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from nemo_gym import PARENT_DIR
from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import (
    AggregateMetrics,
    AggregateMetricsRequest,
    ModelServerRef,
    get_global_config_dict,
)
from nemo_gym.core import (
    AGENT_LIMIT,
    AGENT_TIMEOUT,
    JUDGE_UNPARSEABLE,
    MODEL_CONTEXT_LIMIT,
    PROVIDER_IO_TIMEOUT,
    PROVIDER_REQUEST_FAILED,
    FailureBoundary,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NgFailureEvent,
    observe_failure,
)
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxAccess,
    SandboxConfig,
    SandboxResources,
    SandboxSpec,
    cpu_cap_env,
    resolve_provider_config,
    resolve_provider_metadata,
    sandbox_failure,
)
from nemo_gym.server_runtime import (
    CHECKPOINT_CLOSE_FOLLOWS_HEADER,
    SHUTDOWN_STOP_CONCURRENCY,
    STOP_TIMEOUT_S,
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    LentSandboxes,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesVerification,
    ReverifyMode,
    SimpleResourcesServer,
    lost_session_verification,
    reject_partial_typed_seed,
)
from nemo_gym.utils import bounded_text, is_nemo_gym_fastapi_entrypoint, reachable_host
from nemo_gym.verification import JudgeError, call_judge

# The grader the apptainer comparators were scored with. NOT the longer
# swe_agents/judge_prompt.py on the legacy branch: those runs set
# use_search_judge=False and graded OFFLINE afterwards, so the in-run judge code never
# executed. Proof: every comparator row carries use_search_judge=False, reward=0.0 and no
# judge* keys at all. Presence in the legacy tree is not evidence of execution; check the
# run's instance_config before trusting a source file.
#
# CORRECTED. This comment used to claim the offline grader used by our comparison pipeline is
# BYTE-IDENTICAL to this template. That stopped being true once the stock simple-evals text
# here was replaced with the ordered-rules R1-R5 cascade; the two now differ in both length
# and content, and the comment simply went stale. Both paths call the same judge.
#
# What the drift is worth, measured by re-grading BOTH arms' answers under BOTH prompts against
# the same judge on the paired BrowseComp questions: the prompt choice moves each arm by well
# under a point, in opposite directions, and does NOT explain the gap between the two arms,
# which survives under either prompt at high significance. Quote that re-grade, not an identity
# claim, whenever these numbers are compared.
from resources_servers.browsecomp_advanced_harness.judge_prompt import JUDGE_PROMPT_TEMPLATE
from resources_servers.opencode_search.exa_cache import ExaCache
from resources_servers.opencode_search.task_data import TaskData


LOGGER = logging.getLogger(__name__)

# A judge that echoes this instead of grading has repeated the template back at us.
_JUDGE_TEMPLATE_ECHO = "The final exact answer extracted from the [response]"

# Every line the OpenCode plugins report starts with this marker.
_PLUGIN_MARKER = "[nemo-gym-plugin] "
# The line nemo_gym_sentinel.js prints once the plugin layer has loaded. Matched beyond "loaded": the receipts plugin's
# own load line starts the same way, and it proves only that the plugin log works.
_PLUGIN_SENTINEL = "[nemo-gym-plugin] loaded sentinel"
# The variable the receipts plugin (opencode_bundle/nemo_gym_receipts.js) takes the plugin log's path from.
_PLUGIN_LOG_ENV = "NEMO_GYM_PLUGIN_LOG"
# The failure kind of an episode's verify refused because its plugin log lacks a line it must hold.
PLUGINS_NOT_LOADED = "opencode_search:plugins_not_loaded"
# How many characters of each end of the plugin log a refusal shows: an Environment Server keeps 2,000 characters of
# a refused reply in the row's failure_reason, so both ends fit with the rest of the detail.
_PLUGIN_LOG_SLICE_CHARS = 600
# The most of the plugin log a verify response carries. A run within its turn budget writes a few short lines per
# model call, far less than this; a longer log keeps its two ends.
_PLUGIN_LOG_ROW_CHARS = 1 << 20
# opencode_agent keeps its runtime and sessions in a sandbox under paths that start with this, and refuses a workdir
# there.
_OPENCODE_STORAGE_PREFIX = "/tmp/nemo-gym-opencode"
# The longest an episode's export copy may keep its sandbox running at verify, from the listing to the last download.
_EXPORT_COPY_TIMEOUT_S = 300
# One file of the export listing: its size in bytes and its name. Only names of ASCII letters, digits, '.', '_' and '-'
# that do not start with '.' are listed, as opencode_agent names every file it exports.
_EXPORT_LISTING_ENTRY = re.compile(r"([0-9]+):([A-Za-z0-9_-][A-Za-z0-9._-]*)")


def _last_assistant_text(response: NeMoGymResponse) -> str:
    """Text of the LAST assistant message item in response.output (the model's final answer).

    Not Response.output_text, which concatenates every assistant message item — wrong when the
    model emits text alongside tool calls mid-trajectory. Skipping non-message items also keeps
    a trailing reasoning item (routine with uses_reasoning_parser) from masking the answer.
    """
    text = ""
    for item in response.output or []:
        if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
            t = "".join(
                getattr(p, "text", "")
                for p in (getattr(item, "content", None) or [])
                if getattr(p, "type", None) == "output_text"
            )
            if t:
                text = t
    return text


def parse_judge_verdict(text: str) -> Tuple[bool, str, bool]:
    """Parse a judge reply into (is_correct, extracted_answer, parsed_ok).

    The LAST 'correct: yes|no' wins: reasoning judges echo the template inside their think
    block, so a first-match parse grades the echo instead of the verdict.
    """
    matches = list(re.finditer(r"correct:\s*(yes|no)\b", text or "", re.IGNORECASE))
    if not matches:
        return False, "", False

    is_correct = matches[-1].group(1).lower() == "yes"

    ans_matches = list(re.finditer(r"extracted_final_answer:\s*(.+?)(?:\n|$)", text or ""))
    extracted = ans_matches[-1].group(1).strip() if ans_matches else ""

    if extracted and _JUDGE_TEMPLATE_ECHO in extracted:
        return False, "", False

    return is_correct, extracted, True


def terminal_reason_from_plugin_log(plugin_log: str) -> str:
    """How a run ended, from the lines nemo_gym_turn_budget.js reported: "context_window", "max_iteration" or "".

    The reading opencode_sandboxed_agent applies to its run output before it forwards the ending on the relay path.
    OpenCode reports an overflow that compaction recovered from exactly like one that ended the session, so the plugin
    follows a recovered overflow with a recovery line, and only an overflow left unmatched ended the run. That outranks
    max_iteration: the run was cut off by a limit of the setup rather than by spending its turn budget.
    """
    unrecovered_overflows = 0
    max_iteration = False
    for line in plugin_log.splitlines():
        if _PLUGIN_MARKER not in line:
            continue
        if "context_overflow_recovered" in line:
            unrecovered_overflows = max(0, unrecovered_overflows - 1)
        elif "context_overflow " in line or line.rstrip().endswith("context_overflow"):
            unrecovered_overflows += 1
        elif "terminal=" in line:
            max_iteration = max_iteration or line.split("terminal=", 1)[1].startswith("max_iteration")
    if unrecovered_overflows:
        return "context_window"
    return "max_iteration" if max_iteration else ""


def _is_sandbox_path(path: str) -> bool:
    """Whether path is an absolute path inside the sandbox that stays where it names."""
    return PurePosixPath(path).is_absolute() and ".." not in PurePosixPath(path).parts and "\x00" not in path


def _in_opencode_storage(path: str) -> bool:
    """Whether a sandbox path lies where opencode_agent keeps its runtime and sessions."""
    return str(PurePosixPath(path)).startswith(_OPENCODE_STORAGE_PREFIX)


def _host_directory_name(text: str) -> str:
    """text as the name of one directory on this host: '_' in place of each character other than an ASCII letter, a
    digit, '.', '_' and '-', and of each dot of a name made only of dots."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", text)
    return name if name.strip(".") else name.replace(".", "_")


def _remove_directory(path: Path) -> None:
    """Remove the directory at path, with everything in it, if there is one."""
    with suppress(FileNotFoundError):
        shutil.rmtree(path)


def _file_sizes(directory: Path, names: List[str]) -> Dict[str, int]:
    """The size in bytes of each named file in directory."""
    return {name: (directory / name).stat().st_size for name in names}


def _episode_file(source: str) -> bytes:
    """The contents of an episode file's source: a regular file named relative to the Gym root, never to the working
    directory of the process."""
    path = PARENT_DIR / source
    if not path.is_file():
        raise FileNotFoundError(f"episode_sandbox_files source {source!r} is not a file in the Gym root ({path})")
    return path.read_bytes()


def _plugin_log_on_the_row(plugin_log: str) -> str:
    """The plugin log as a verify response carries it: whole up to _PLUGIN_LOG_ROW_CHARS characters, else its two ends
    around a line that gives its length, in _PLUGIN_LOG_ROW_CHARS characters in all."""
    if len(plugin_log) <= _PLUGIN_LOG_ROW_CHARS:
        return plugin_log
    gap = f"\n[... the middle of the plugin log is left out; the whole log is {len(plugin_log)} characters ...]\n"
    end = (_PLUGIN_LOG_ROW_CHARS - len(gap)) // 2
    return plugin_log[:end] + gap + plugin_log[-end:]


class SearchResourcesServerConfig(BaseResourcesServerConfig):
    # ---- Exa response cache -------------------------------------------------------------------
    # Shared by every shard and every sandbox: the store is a directory on Lustre, so it outlives
    # any job, and this server fronts it over HTTP because our agents are remote pods and cannot
    # reach a localhost sidecar the way the legacy per-node design did.
    #
    # Empty root disables it entirely and the proxy endpoint is not registered, so the cache is
    # opt-in per benchmark and an un-cached arm costs one config line.
    exa_cache_root: str = ""
    # The EXISTING ~2M-entry store, read-only, searched by exact key only. Not walked at startup:
    # building a fuzzy index over 2M Lustre files would hang the server while rollouts wait on
    # seed_session, whereas an exact lookup is a direct path from the hash and costs one stat().
    exa_cache_legacy_root: str = ""
    # Legacy entries are ~90% Tavily, holding structured results rather than rendered Exa text.
    # Serving them means translating (render_tavily). On by default: the model still gets real
    # search results and the point of the cache is cost.
    exa_cache_use_tavily: bool = True
    # 0.8 matches the legacy threshold. Measured by replaying real queries through this store,
    # fuzzy matching returns hits where exact-only returns none at all -- so this knob IS the
    # feature, not a tuning detail.
    exa_cache_fuzzy_threshold: float = 0.8
    # Upstream, overridable so a run can be pinned to a frozen mirror.
    exa_upstream_url: str = "https://mcp.exa.ai/mcp"

    # STATELESS: a verify without the close-follows header that an Environment Server sends with an episode's verify,
    # such as gym eval reverify's, grades only its body (question, ground truth, saved response), so a saved run can be
    # re-graded under another judge or prompt without running the model again. Such a re-grade cannot reproduce a
    # budget mask: the saved body carries no ending, and an episode's ending was read from the plugin log in its
    # sandbox, which stopped with the episode (see README).
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    # Sandbox the agent works in. One fixed image for every row: these tasks carry no
    # per-instance repo, so there is nothing to derive an image from.
    sandbox_provider: str
    sandbox_config: Dict[str, Any]
    sandbox_image: str
    # Working directory of the sandbox an episode seed lends to a session harness; the seed creates it. None turns
    # lending off, so an episode seed is refused (422). An agent's own /run gets the image's working directory.
    # opencode_agent's rules apply: an absolute directory other than / and /tmp, outside its storage.
    sandbox_workdir: Optional[str] = None

    # What an episode seed gives the harness in the lent sandbox, besides the workdir. Each file maps an absolute sandbox
    # path, outside sandbox_workdir and opencode_agent's storage, to its source: a file in the Gym root, named relative
    # to it and read once at startup. Its directory must already exist in the image, since a provider need not create
    # it: stage under /tmp.
    episode_sandbox_files: Dict[str, str] = Field(default_factory=dict)
    # Environment of the lent sandbox, over sandbox_config.env. An empty value sets nothing, so a config can name a
    # setting it leaves off. Secrets come from an overlay, never from a config file. The search key is one EXA_API_KEY:
    # an EXA_API_KEYS pool is refused at startup. With exa_cache_root set, NEMO_GYM_EXA_BASE_URL defaults to this
    # server's /mcp proxy, and this server must then have a sandbox-reachable host; set the variable here to choose
    # another address.
    episode_sandbox_env: Dict[str, str] = Field(default_factory=dict)
    # Where the receipts plugin (opencode_bundle/nemo_gym_receipts.js) copies the OpenCode plugins' lines in the lent
    # sandbox; it also sets NEMO_GYM_PLUGIN_LOG there. An episode's verify reads it before the sandbox stops, masks a run
    # that ended on its budget as the relay does, and refuses one whose plugins did not load. It lies outside
    # sandbox_workdir, where the model's tools could change it. None reads nothing and refuses nothing, for a harness
    # that brings no such plugins.
    plugin_log_path: Optional[str] = None
    # Lines the plugin log must also hold for a run to be graded, such as the tool surface the run must have had.
    required_plugin_markers: List[str] = Field(default_factory=list)
    # Where the harness writes its session exports in the lent sandbox: opencode_agent's export_dir. opencode_agent's
    # rules apply: an absolute directory other than /, outside its storage, and neither sandbox_workdir, inside it nor
    # above it. Nothing is copied from it without episode_export_copy_dir.
    episode_export_dir: Optional[str] = None
    # An absolute directory on this host, one per run, which a run's overlay sets. An episode's verify then copies the
    # regular files directly in episode_export_dir to <capture key>/<task id>/attempt-<attempt>/ under it, after the
    # plugin log read and before the sandbox stops, and reports the copy in the row as opencode_export_copy. A copy
    # that fails or is refused is logged and reported there, and never changes the reward, mask_sample or
    # evaluation_completed. None copies nothing.
    episode_export_copy_dir: Optional[str] = None
    # The most that an episode's exports may hold in all for them to be copied, in bytes: 256 MiB.
    episode_export_max_bytes: int = Field(default=256 << 20, gt=0)

    @field_validator("sandbox_workdir")
    @classmethod
    def _harness_workdir(cls, workdir: Optional[str]) -> Optional[str]:
        # A harness works in this directory as its task root, so it is never the filesystem root or the shared /tmp,
        # and opencode_agent refuses one in its own storage.
        if workdir is not None and (
            not _is_sandbox_path(workdir)
            or PurePosixPath(workdir) in (PurePosixPath("/"), PurePosixPath("/tmp"))
            or _in_opencode_storage(workdir)
        ):
            raise ValueError(
                f"sandbox_workdir must be an absolute directory other than / and /tmp, outside "
                f"{_OPENCODE_STORAGE_PREFIX}*: {workdir!r}"
            )
        return workdir

    @field_validator("episode_sandbox_files")
    @classmethod
    def _episode_files_in_sandbox(cls, files: Dict[str, str]) -> Dict[str, str]:
        outside = [path for path in files if not _is_sandbox_path(path)]
        if outside:
            raise ValueError(f"episode_sandbox_files must map absolute sandbox paths: {outside!r}")
        stored = [path for path in files if _in_opencode_storage(path)]
        if stored:
            raise ValueError(
                f"episode_sandbox_files must stay out of {_OPENCODE_STORAGE_PREFIX}*, where opencode_agent keeps its "
                f"runtime and sessions: {stored!r}"
            )
        # Relative to the Gym root, so a source is the same file wherever the server starts, and never outside it.
        sources = [source for source in files.values() if Path(source).is_absolute() or ".." in Path(source).parts]
        if sources:
            raise ValueError(
                f"episode_sandbox_files sources must be relative to the Gym root, without '..': {sources!r}"
            )
        return files

    @field_validator("plugin_log_path")
    @classmethod
    def _plugin_log_in_sandbox(cls, path: Optional[str]) -> Optional[str]:
        if path is not None and not _is_sandbox_path(path):
            raise ValueError(f"plugin_log_path must be an absolute sandbox path: {path!r}")
        return path

    @field_validator("episode_export_dir")
    @classmethod
    def _export_dir_in_sandbox(cls, path: Optional[str]) -> Optional[str]:
        # opencode_agent refuses these for its export_dir.
        if path is not None and (
            not _is_sandbox_path(path) or PurePosixPath(path) == PurePosixPath("/") or _in_opencode_storage(path)
        ):
            raise ValueError(
                f"episode_export_dir must be an absolute sandbox directory other than /, outside "
                f"{_OPENCODE_STORAGE_PREFIX}*: {path!r}"
            )
        return path

    @field_validator("episode_export_copy_dir")
    @classmethod
    def _export_copy_dir_on_this_host(cls, path: Optional[str]) -> Optional[str]:
        # Never relative to the working directory of the process.
        if path is not None and not Path(path).is_absolute():
            raise ValueError(f"episode_export_copy_dir must be an absolute directory on this host: {path!r}")
        return path

    @model_validator(mode="after")
    def _export_copy_has_a_source(self) -> "SearchResourcesServerConfig":
        if self.episode_export_copy_dir is not None and self.episode_export_dir is None:
            raise ValueError("episode_export_copy_dir needs episode_export_dir, the sandbox directory it copies from")
        return self

    @model_validator(mode="after")
    def _plugin_log_settings_agree(self) -> "SearchResourcesServerConfig":
        # The plugin writes where NEMO_GYM_PLUGIN_LOG points and verify reads plugin_log_path: were they to differ,
        # every episode would read a log that no plugin wrote and be refused.
        named = self.episode_sandbox_env.get(_PLUGIN_LOG_ENV)
        if named and named != self.plugin_log_path:
            raise ValueError(
                f"episode_sandbox_env.{_PLUGIN_LOG_ENV} ({named!r}) must equal plugin_log_path "
                f"({self.plugin_log_path!r}); plugin_log_path alone sets it"
            )
        if self.required_plugin_markers and self.plugin_log_path is None:
            raise ValueError("required_plugin_markers are read from the plugin log: set plugin_log_path")
        return self

    @model_validator(mode="after")
    def _episode_paths_outside_the_workdir(self) -> "SearchResourcesServerConfig":
        # The model works in sandbox_workdir, where its tools list and change files, so what this server stages and
        # reads in the lent sandbox stays out of it.
        if self.sandbox_workdir is not None:
            workdir = PurePosixPath(self.sandbox_workdir)
            inside = [path for path in self.episode_sandbox_files if PurePosixPath(path).is_relative_to(workdir)]
            if inside:
                raise ValueError(f"episode_sandbox_files must lie outside sandbox_workdir: {inside!r}")
            if self.plugin_log_path is not None and PurePosixPath(self.plugin_log_path).is_relative_to(workdir):
                raise ValueError(f"plugin_log_path must lie outside sandbox_workdir: {self.plugin_log_path!r}")
            # As opencode_agent refuses for its export_dir.
            export_dir = None if self.episode_export_dir is None else PurePosixPath(self.episode_export_dir)
            if export_dir is not None and (export_dir.is_relative_to(workdir) or workdir.is_relative_to(export_dir)):
                raise ValueError(
                    f"episode_export_dir must be neither sandbox_workdir, inside it nor above it: "
                    f"{self.episode_export_dir!r}"
                )
        return self

    # Grading. The judge is the only grader; there is no exact-match fallback, because a
    # BrowseComp answer is free text and string equality scores correct answers as wrong.
    judge_model_server: ModelServerRef
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    # The retry ladder: judge_max_attempts attempts, the first bounded by judge_timeout_seconds and each retry by the
    # shorter judge_retry_timeout_seconds, a fixed judge_retry_delay_seconds apart rather than an exponential backoff,
    # so a dead endpoint holds a rollout for a bounded time. Each attempt is one call_judge, which sends the request up
    # to JUDGE_CALL_MAX_ATTEMPTS (3) times on a timeout, a lost connection or a 408, 429 or 5xx reply, with at most 2 s
    # and then 4 s of backoff. With the defaults, a judge that never answers holds a rollout for
    # (3*300 + 6) + 3*(3*120 + 6) + 3*30 = 2,094 s when each request is sent once. Gym's client also sends a timed-out
    # or refused request again until global_aiohttp_internal_retry_budget_seconds (600 s by default) is spent, which
    # stretches that to about two hours.
    judge_max_attempts: int = 4
    judge_retry_delay_seconds: float = 30.0
    judge_retry_timeout_seconds: float = 120.0
    # Abort the run after this many rollouts in a row fail to grade: the judge never answered, or never gave a
    # parseable verdict (verify masks that row). Guards the silent-starvation mode where a dead judge masks every
    # sample and the run burns its wall producing nothing.
    # Counted per ROLLOUT, not per attempt, so one transient burst cannot trip it. 0 disables.
    # With num_workers > 1, each worker counts the rollouts it graded, so each worker aborts on its own.
    judge_max_consecutive_failures: int = 25
    # Budget covers reasoning + final message combined. 2048 makes reasoning judges burn the
    # whole budget thinking, emitting no final message, which reads as an unparseable verdict.
    judge_max_output_tokens: int = 10240
    # Bounds ONE attempt. Gym's shared aiohttp client has no timeout of its own, so without this
    # a judge that accepts the connection and never answers holds its rollout until the job's
    # wall clock and the retry loop below never runs.
    judge_timeout_seconds: float = 300.0


class SearchRunRequest(BaseRunRequest):
    # The agent forwards the whole dataset row, so tolerate columns we do not model.
    model_config = ConfigDict(extra="allow")

    question: str
    ground_truth: str


class SearchSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")


class SearchTaskSeed(TaskData, BaseSeedSessionRequest):
    """The task_data an episode seed must carry: verify grades with its question and ground_truth."""


class SearchSeedSessionResponse(BaseSeedSessionResponse):
    # The only field the agent reads. Without it opencode_sandboxed_agent silently creates its
    # own sandbox from a hardcoded SWE-bench image instead of connecting to this one.
    sandbox_handle: str


class SearchVerifyRequest(SearchRunRequest, BaseVerifyRequest):
    pass


# How a rollout ended, for endings that are a BUDGET failure rather than a wrong answer. Same two
# agent_error_kinds the legacy stack flagged (the legacy stack's
# swe_agents/app.py:5742-5751, alongside eval/agent timeouts, OOM and judge_masked -- in this stack
# those three already land in the failures sidecar instead).
#
# SCOPE, and this is the part that is easy to get wrong: legacy's mask_sample gates the RL
# GRADIENT, not the eval metric. Its published comparator scored the capped rows: its denominator
# is the full row count including max_iteration endings. So these rows keep their judged reward,
# and a number on that denominator is computed from the rows; Gym's own aggregate metrics leave
# masked rows out (see aggregate_metrics).
# agent_timed_out joins these: legacy masked it in the same branch (app.py:5742-5751, the
# eval_timed_out/agent_timed_out arm), and it is masked the same way.
MASKED_TERMINAL_REASONS = frozenset({"max_iteration", "context_window", "agent_timed_out"})

# The diagnostics each masked terminal reason reports on the verify response. They describe the
# ending only; they do not change the reward or the mask.
MASKED_TERMINAL_FAILURES = {
    "max_iteration": (AGENT_LIMIT, "The agent reached its maximum number of iterations"),
    "context_window": (MODEL_CONTEXT_LIMIT, "The conversation exceeded the model's context window"),
    "agent_timed_out": (AGENT_TIMEOUT, "The agent ran past its time limit"),
}


class SearchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    extracted_final_answer: str
    final_text: str
    judge_parsed: bool
    judge_attempts: int
    judge_time_taken: float
    # True when the reward is not a measurement of the answer: the session was lost before verify,
    # or the judge gave no verdict (reward 0). Also true when the rollout ran out of budget rather
    # than answering wrongly; that row keeps its judged reward. Gym's aggregate metrics leave masked
    # rows out of the quality scores.
    mask_sample: bool = False
    # Whether grading reached a result: the judge's verdict, or the empty answer that needs none.
    # False for a lost session and a judge that gave no verdict.
    evaluation_completed: bool


@dataclass(frozen=True)
class _PluginLog:
    """The plugin log an episode's verify read from its sandbox."""

    text: str
    # Bytes in the file; 0 when it did not exist.
    size: int
    exists: bool


@dataclass(frozen=True)
class _ExportCopy:
    """How an episode's verify copied the session exports out of its sandbox, as its row carries it
    (opencode_export_copy)."""

    # "copied"; "absent" when the sandbox has no episode_export_dir; "oversize" when the files hold more than
    # episode_export_max_bytes, and none is copied; "failed" when the listing or a download failed or ran out of time,
    # and nothing is kept.
    status: Literal["copied", "absent", "oversize", "failed"]
    # The directory on this host the files are copied to. It exists only when they were.
    dir: str
    # The regular files listed in episode_export_dir, by name, with their sizes in bytes, and their total. A failed copy
    # reports none.
    files: Dict[str, int] = field(default_factory=dict)
    bytes: int = 0
    # Why the files were not copied; empty when they were.
    detail: str = ""


@dataclass(frozen=True)
class _VerifyEvidence:
    """What a session's verify keeps for its retries: a digest of the body it graded, the plugin log it read from the
    sandbox and how it copied the session exports, both before stopping the sandbox, and, once grading reached a
    result, the grading fields it answered with.

    A retry must send the same body, so the grading fields and that body rebuild the first reply; a retry with
    another body gets 409 rather than a grade of a different answer.
    """

    body_digest: str
    # None when no plugin log was read (see _read_plugin_log).
    plugin_log: Optional[_PluginLog] = None
    # None when no copy was made (see _copy_exports).
    export_copy: Optional[_ExportCopy] = None
    grade: Optional[Dict[str, Any]] = None


def _body_digest(body: SearchVerifyRequest) -> str:
    return hashlib.sha256(json.dumps(body.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()


# The grading fields of a verify response whose answer no judge graded.
_NOT_GRADED: Dict[str, Any] = {
    "extracted_final_answer": "",
    "final_text": "",
    "judge_parsed": False,
    "judge_attempts": 0,
    "judge_time_taken": 0.0,
    "evaluation_completed": False,
}


async def _stop(sandbox: AsyncSandbox) -> None:
    """Stop an agent run's sandbox within STOP_TIMEOUT_S, the bound LentSandboxes applies to the sandboxes it lends,
    even while the request that stops it is being cancelled.

    A client that disconnects cancels its request handler and every later await in it, so an unshielded stop would
    never run and the sandbox would outlive every reference to it.
    """
    with anyio.CancelScope(shield=True), anyio.fail_after(STOP_TIMEOUT_S):
        await sandbox.stop()


class SearchResourcesServer(SimpleResourcesServer):
    config: SearchResourcesServerConfig

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = JUDGE_PROMPT_TEMPLATE

    seed_request_model: ClassVar[type[BaseSeedSessionRequest]] = SearchTaskSeed

    # The sandbox of each agent's own /run, by its cookie session, and the access lent to each episode session.
    _agent_run_sandboxes: Dict[str, AsyncSandbox] = PrivateAttr(default_factory=dict)
    _sandbox_access: Dict[str, SandboxAccess] = PrivateAttr(default_factory=dict)
    _lent_sandboxes: Optional[LentSandboxes] = PrivateAttr(default=None)
    # episode_sandbox_files with their contents, by sandbox path.
    _episode_files: Dict[str, bytes] = PrivateAttr(default_factory=dict)
    # Rollouts in a row whose judge never produced a verdict. Reset by any success.
    _judge_consecutive_failures: int = PrivateAttr(default=0)
    _exa_cache: Any = PrivateAttr(default=None)

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        if self.config.sandbox_workdir is not None:
            self._episode_files = {
                path: _episode_file(source) for path, source in self.config.episode_sandbox_files.items()
            }
            self._lent_sandboxes = LentSandboxes(
                self.config.sandbox_provider,
                self._lent_sandbox_config(),
                image=self.config.sandbox_image,
                workdir=self.config.sandbox_workdir,
            )

    def _lent_sandbox_config(self) -> SandboxConfig:
        """sandbox_config as LentSandboxes takes it, so a lent sandbox matches an agent run's but for its workdir and
        the episode settings.

        An environment with a pool of search keys (EXA_API_KEYS) is refused: the OpenCode plugins read one key,
        EXA_API_KEY, and this environment is fixed at startup, so no key can be picked from the pool for each row as
        the relay agent does. The pool would reach the sandbox whole, and every search would go out without a key.
        """
        settings = dict(self.config.sandbox_config)
        derive_cpu_env = settings.pop("derive_cpu_env", True)
        config = SandboxConfig.model_validate(settings)
        cpu_env = cpu_cap_env(SandboxResources.from_mapping(config.resources).cpu) if derive_cpu_env else {}
        env = cpu_env | config.env | self._episode_env()
        if env.get("EXA_API_KEYS"):
            raise ValueError(
                "The lent sandbox's environment sets EXA_API_KEYS, a pool of search keys the OpenCode plugins do not "
                "read: set one key as EXA_API_KEY in episode_sandbox_env"
            )
        return config.model_copy(
            update={"env": env, "metadata": config.metadata | {"nemo_gym_agent": self.config.name}}
        )

    def _episode_env(self) -> Dict[str, str]:
        """episode_sandbox_env without its empty values, plus what the plugins need from this server's settings."""
        env = {name: value for name, value in self.config.episode_sandbox_env.items() if value != ""}
        if self.config.plugin_log_path is not None:
            env.setdefault(_PLUGIN_LOG_ENV, self.config.plugin_log_path)
        if self.config.exa_cache_root and not env.get("NEMO_GYM_EXA_BASE_URL"):
            env["NEMO_GYM_EXA_BASE_URL"] = self._exa_proxy_url()
        return env

    def _exa_proxy_url(self) -> str:
        """This server's /mcp proxy at the address other servers reach it at, as opencode_sandboxed_agent derives it.

        A sandbox runs on another host, where a loopback address would reach nothing and every search would fail, so
        a loopback or unspecified host is refused at startup.
        """
        url = f"http://{self.config.host}:{self.config.port}"
        try:
            reachable_host(url)
        except ValueError as error:
            raise ValueError(f"A lent sandbox cannot reach the search cache proxy at {url}: {error}") from error
        return f"{url}/mcp"

    def multiple_workers_unsupported_reason(self) -> Optional[str]:
        # As in chartography and visual_coding_opencode: several workers are not tested with lent sandboxes.
        if self._lent_sandboxes is None:
            return None
        return "lent sandboxes live in one worker's memory"

    # ---- Sessions (the base class serves seed, close and the session lock) ------------------------
    def has_session_state(self, session_id: str) -> bool:
        return session_id in self._sandbox_access or session_id in self._agent_run_sandboxes

    async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
        identity = self.session_identity(session_id)
        if identity is None:
            # An agent's own /run, which attaches to the sandbox by the handle the seed returns.
            await self._start_agent_run_sandbox(session_id)
            return
        if self._lent_sandboxes is None:
            raise ResourcesSeedRejectedError(
                "This server lends a sandbox to an episode seed only with sandbox_workdir set"
            )
        # On failure lend() stops the sandbox it started; after it returns, the base releases a failed seed.
        access = await self._lent_sandboxes.lend(session_id, identity[1], files=self._episode_files)
        await self._create_workdir(self._lent_sandboxes.get(session_id))
        self._sandbox_access[session_id] = access

    async def release_session_state(self, session_id: str) -> None:
        # Each stop raises on failure and keeps its sandbox, so a later verify, close or shutdown tries again.
        if self._lent_sandboxes is not None:
            await self._lent_sandboxes.release(session_id)
        self._sandbox_access.pop(session_id, None)
        sandbox = self._agent_run_sandboxes.get(session_id)
        if sandbox is not None:
            await _stop(sandbox)
            self._agent_run_sandboxes.pop(session_id, None)

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest
        | Annotated[SearchSeedSessionRequest, BeforeValidator(reject_partial_typed_seed)],
    ) -> SearchSeedSessionResponse | ResourcesSeedSessionResponse:
        """The base seed, replying with the sandbox: its access to an episode seed, its handle to an agent's /run.

        FastAPI serializes the reply through the return annotation, so it must name SearchSeedSessionResponse: with
        only the base reply models, sandbox_handle is dropped and the agent silently starts a sandbox of its own.
        """
        reply = await super().seed_session(body, request=request)
        if isinstance(reply, ResourcesSeedSessionResponse):
            reply.sandbox_access = self._sandbox_access[reply.resources_session_id]
            return reply
        return SearchSeedSessionResponse(
            sandbox_handle=self._agent_run_sandboxes[request.session[SESSION_ID_KEY]].sandbox_id
        )

    async def _start_agent_run_sandbox(self, session_id: str) -> None:
        global_config_dict = get_global_config_dict()
        spec = self._agent_run_spec(global_config_dict)
        sandbox = AsyncSandbox(resolve_provider_config(self.config.sandbox_provider, global_config_dict))
        # Registered before it starts, so release_session_state stops a start that failed or was cancelled.
        self._agent_run_sandboxes[session_id] = sandbox
        await sandbox.start(spec)

    def _agent_run_spec(self, global_config_dict: Any) -> SandboxSpec:
        provider_default_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)

        sandbox_resources = SandboxResources.from_mapping(dict(self.config.sandbox_config.get("resources", {})))
        env = dict(self.config.sandbox_config.get("env", {}))
        if self.config.sandbox_config.get("derive_cpu_env", True):
            env = cpu_cap_env(sandbox_resources.cpu) | env

        return SandboxSpec(
            image=self.config.sandbox_image,
            ttl_s=self.config.sandbox_config.get("ttl_s", None),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s", None),
            workdir=None,  # Default to the image's WORKDIR
            env=env,
            files=dict(),
            metadata=provider_default_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {"nemo_gym_agent": self.config.name},
            resources=sandbox_resources,
            entrypoint=None,
            provider_options=self.config.sandbox_config.get("provider_options", {}),
        )

    async def _create_workdir(self, sandbox: AsyncSandbox) -> None:
        """Create sandbox_workdir in a lent sandbox: providers need not create a spec's workdir, and a harness
        never prepares a borrowed one.

        A command after the start, since LentSandboxes has no setup hook. It runs from / because a command's default
        directory is the missing workdir.

        A mkdir that exits non-zero refuses the seed with 422, which an Environment Server does not retry: the image
        caused it (permission denied, a read-only root, a file in the way), so every episode would fail the same way.
        A command that did not complete (an error type, such as a timeout) answers 500, which is retried, since a
        fresh sandbox may succeed. Either way the base stops the sandbox the seed started.
        """
        workdir = self.config.sandbox_workdir
        result = await sandbox.exec(f"mkdir -p -- {shlex.quote(workdir)}", cwd="/", timeout_s=60)
        output = (result.stderr or result.stdout or "").strip()[-500:]
        if result.error_type is not None:
            raise RuntimeError(
                f"Cannot create sandbox_workdir {workdir} in the lent sandbox "
                f"(exit {result.return_code}, error {result.error_type}): {output}"
            )
        if result.return_code != 0:
            raise ResourcesSeedRejectedError(
                f"Cannot create sandbox_workdir {workdir} in the lent sandbox (exit {result.return_code}): {output}"
            )

    async def shutdown(self) -> None:
        """Stop every sandbox still running, as when the server exits with sessions open or agent runs unverified."""
        if self._lent_sandboxes is not None:
            await self._lent_sandboxes.release_all()
        self._sandbox_access.clear()
        stopping = asyncio.Semaphore(SHUTDOWN_STOP_CONCURRENCY)

        async def stop(session_id: str, sandbox: AsyncSandbox) -> None:
            async with stopping:
                try:
                    await _stop(sandbox)
                except Exception:
                    LOGGER.exception("Failed to stop the sandbox of agent run %s at shutdown", session_id)

        sandboxes, self._agent_run_sandboxes = self._agent_run_sandboxes, {}
        await asyncio.gather(*(stop(session_id, sandbox) for session_id, sandbox in sandboxes.items()))

    async def _call_judge(
        self,
        judge_create_params: NeMoGymResponseCreateParamsNonStreaming,
        timeout_seconds: Optional[float] = None,
    ) -> Tuple[Any, str]:
        judge_response = await call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=judge_create_params,
            response_model=NeMoGymResponse,
            timeout=aiohttp.ClientTimeout(
                total=self.config.judge_timeout_seconds if timeout_seconds is None else timeout_seconds
            ),
        )
        if not judge_response.output:
            raise JudgeError("empty judge response")
        return judge_response, _last_assistant_text(judge_response)

    async def _judge_answer(self, question: str, ground_truth: str, final_text: str) -> Dict[str, Any]:
        """Grade final_text against ground_truth. Returns a dict of result fields.

        Two failure modes are deliberately kept distinct: a judge that answered but could not be
        parsed returns judge_parsed False, which verify masks as inconclusive, while a judge that
        never answered is infrastructure noise and raises JudgeError so judge_failsafe routes the
        row to the failures sidecar. Both count toward judge_max_consecutive_failures.
        """
        answer = re.sub(r"<think>.*?</think>", "", final_text or "", flags=re.DOTALL).strip()

        judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            question=question, correct_answer=ground_truth, response=answer
        )

        judge_create_params = (
            self.config.judge_responses_create_params.model_copy(deep=True)
            if self.config.judge_responses_create_params is not None
            else NeMoGymResponseCreateParamsNonStreaming(input=[])
        )
        judge_create_params.max_output_tokens = self.config.judge_max_output_tokens
        # Legacy graded at temperature 0.0; overriding to 1.0 here would silently undo the config
        # and make the grader itself a source of run-to-run variance.
        if self.config.judge_responses_create_params is None:
            judge_create_params.temperature = 0.0
        judge_create_params.input = [NeMoGymEasyInputMessage(role="user", content=judge_prompt)]

        started = time()
        last_error: Optional[str] = None
        # Did the judge ever actually reply? A judge that answers but cannot be parsed leaves a
        # masked row in the rollouts (see verify); only a judge we could never reach is
        # infrastructure noise worth routing to the failures sidecar.
        had_response = False
        for attempt in range(self.config.judge_max_attempts):
            try:
                # Attempt 1 gets the full timeout; retries get the shorter one, so the worst
                # case stays bounded instead of multiplying the full cap by the attempt count.
                timeout_s = (
                    self.config.judge_timeout_seconds if attempt == 0 else self.config.judge_retry_timeout_seconds
                )
                _judge_response, judge_text = await self._call_judge(judge_create_params, timeout_s)
                is_correct, extracted, parsed_ok = parse_judge_verdict(judge_text)
                if parsed_ok:
                    self._judge_consecutive_failures = 0
                    return {
                        "reward": 1.0 if is_correct else 0.0,
                        "extracted_final_answer": extracted,
                        "judge_parsed": True,
                        "judge_attempts": attempt + 1,
                        "judge_time_taken": time() - started,
                    }
                had_response = True
                last_error = f"unparseable verdict (tail={judge_text[-200:]!r})"
            except JudgeError as e:
                last_error = str(e)
            except Exception as e:  # noqa: BLE001 - any transport failure is a judge failure
                last_error = f"{type(e).__name__}: {e}"

            print(
                f"[judge_retry attempt={attempt + 1}/{self.config.judge_max_attempts} error={last_error}]",
                flush=True,
            )
            if attempt + 1 < self.config.judge_max_attempts:
                await asyncio.sleep(max(0.0, self.config.judge_retry_delay_seconds))

        # The rollout went ungraded either way. A verdict that never parsed leaves the quality metrics as quietly as a
        # judge that never answered, since verify masks it, so both count toward the breaker.
        self._judge_consecutive_failures += 1
        outcome = "gave no parseable verdict in" if had_response else "unreachable after"
        print(
            f"[search] judge {outcome} {self.config.judge_max_attempts} attempts; "
            f"last: {last_error} [{self._judge_consecutive_failures} consecutive]",
            flush=True,
        )
        self._maybe_abort_on_judge_failures()

        if had_response:
            # The judge talked to us every time and we could never read a verdict out of it. There
            # is no grade to report: verify masks the row as inconclusive, so a regrade can revisit it.
            return {
                "reward": 0.0,
                "extracted_final_answer": "",
                "judge_parsed": False,
                "judge_attempts": self.config.judge_max_attempts,
                "judge_time_taken": time() - started,
            }

        # Never got a reply at all: the endpoint is down or unreachable. Raising sends the row to
        # the failures sidecar rather than recording a fabricated zero.
        raise JudgeError(
            f"judge unreachable after {self.config.judge_max_attempts} attempts; last error: {last_error}"
        )

    def _maybe_abort_on_judge_failures(self) -> None:
        """Stop the run once the judge looks dead, rather than masking every remaining row.

        Ported from the legacy stack (the legacy stack's swe_agents/app.py).
        Without it a dead grader is indistinguishable from a model that answers nothing: every row
        is masked or routes to the failures sidecar, the headline metric is computed over whatever survived,
        and the run burns its whole wall clock producing a number from a handful of samples.
        """
        limit = self.config.judge_max_consecutive_failures
        if limit and self._judge_consecutive_failures >= limit:
            raise RuntimeError(
                f"judge appears dead: {self._judge_consecutive_failures} consecutive rollouts "
                f"failed to grade (judge_max_consecutive_failures={limit}). Aborting rather than "
                f"masking the rest of the run."
            )

    # ---- Exa cache proxy ----------------------------------------------------------------------
    def setup_webserver(self) -> FastAPI:
        """Add the Exa proxy route alongside the standard resources-server endpoints.

        The plugin inside the sandbox POSTs its MCP JSON-RPC here instead of to mcp.exa.ai, so the
        ONLY plugin change is which base URL it uses. Everything else -- lookup, fuzzy matching,
        storage, provider rules, accounting -- lives here, on the Gym side, where it can be changed
        without touching a file that ships into the sandbox.

        The app's lifespan also stops every sandbox still running when the server exits.
        """
        app = super().setup_webserver()
        if self.config.exa_cache_root:
            app.post("/mcp")(self._exa_proxy)
            app.get("/exa_cache_stats")(self._exa_cache_stats)
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as maybe_state:
                    yield maybe_state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        return app

    @property
    def _cache(self) -> ExaCache:
        if self._exa_cache is None:
            self._exa_cache = ExaCache(
                self.config.exa_cache_root,
                self.config.exa_cache_fuzzy_threshold,
                legacy_root=self.config.exa_cache_legacy_root,
                use_tavily=self.config.exa_cache_use_tavily,
            )
        return self._exa_cache

    async def _exa_cache_stats(self) -> Dict[str, Any]:
        # With num_workers > 1, each worker keeps its own index and stats, so this reports the worker that answers.
        # A worker's index also misses entries other workers stored since it started, which costs an Exa call.
        return dict(self._cache.stats)

    async def _exa_proxy(self, request: Request) -> Response:
        """Serve an Exa MCP call from cache, or forward it upstream and remember the answer.

        Speaks the endpoint's own wire format in both directions -- a single `data: ` SSE frame
        carrying the JSON-RPC envelope -- so the plugin cannot tell a hit from a miss apart from
        latency, and the contamination guard and TDM filter still run on the way out exactly as
        they do today (they are applied by the plugin AFTER this returns).

        Fails OPEN on every error path: a cache that is confused must cost money, never
        correctness. Any exception here forwards upstream unchanged.
        """
        body = await request.body()
        try:
            payload = json.loads(body)
            params = payload.get("params") or {}
            tool = params.get("name") or ""
            args = params.get("arguments") or {}
        except (json.JSONDecodeError, AttributeError):
            return await self._exa_forward(request, body)

        # search keys on the query, fetch keys on the url; they are separate sections of one store.
        if tool == "web_search_exa":
            kind, keytext = "search", str(args.get("query") or "")
        elif tool == "web_fetch_exa":
            urls = args.get("urls") or []
            kind, keytext = "browse", str(urls[0] if urls else "")
        else:
            return await self._exa_forward(request, body)

        # ExaCache.get/put are SYNCHRONOUS and touch Lustre. Calling them directly from this
        # async handler blocks the whole event loop -- not just this request -- because asyncio
        # is single-threaded. The first get() is the worst: it lazily builds the fuzzy index by
        # globbing "*/*/*.json" under the cache root and read_text()-ing every entry (2M entries
        # on the shared store), and every other in-flight search sits behind it. Measured on the
        # cached arm, a large share of searches died on the plugin's deadline, against virtually
        # none on the same harness with the cache off.
        #
        # asyncio.to_thread moves the blocking work to the default executor so the loop stays
        # free. It does NOT make the cache faster -- the first lookup still pays the index walk --
        # it stops that cost being charged to every concurrent rollout.
        if keytext:
            try:
                cached, how = await asyncio.to_thread(self._cache.get, kind, keytext)
            except Exception as e:  # noqa: BLE001 - a broken cache must not break a rollout
                print(f"[exa-cache] lookup failed ({type(e).__name__}): {e}", flush=True)
                cached, how = None, "error"
            if cached is not None:
                print(f"[exa-cache] HIT kind={kind} how={how} key={keytext[:70]!r}", flush=True)
                return Response(content=cached, media_type="text/event-stream")

        upstream = await self._exa_forward(request, body)
        if keytext and upstream.status_code == 200:
            try:
                # Also offloaded: put() writes a temp file and renames it on Lustre, where a
                # create has been measured at ~1% EIO under load and is never fast.
                await asyncio.to_thread(self._cache.put, kind, keytext, upstream.body.decode("utf-8", "replace"))
                print(f"[exa-cache] MISS kind={kind} stored key={keytext[:70]!r}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[exa-cache] store failed ({type(e).__name__}): {e}", flush=True)
        return upstream

    async def _exa_forward(self, request: Request, body: bytes) -> Response:
        """Forward verbatim to Exa, preserving the query string so the per-row API key rides along."""
        qs = request.url.query
        url = self.config.exa_upstream_url + (f"?{qs}" if qs else "")
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    url,
                    data=body,
                    headers={"content-type": "application/json", "accept": "application/json, text/event-stream"},
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    raw = await resp.read()
                    return Response(
                        content=raw,
                        status_code=resp.status,
                        media_type=resp.headers.get("content-type", "text/event-stream"),
                    )
        except Exception as e:  # noqa: BLE001
            print(f"[exa-cache] upstream failed ({type(e).__name__}): {e}", flush=True)
            return Response(content=b"", status_code=502)

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Gym's default aggregate metrics, which leave mask_sample rows out of the quality scores.

        A masked row is either not a measurement (a lost session, a judge that gave no verdict) or
        a rollout that ran out of budget, which keeps its judged reward. The flag stays on every
        row, so a training consumer can mask on it and an analysis can split on it or recompute a
        number over every row.
        """
        masked = sum(1 for vr in body.verify_responses if vr.get("mask_sample"))
        if masked:
            print(
                f"[search] {masked}/{len(body.verify_responses)} row(s) carry mask_sample and are "
                f"left out of the quality metrics",
                file=sys.stderr,
            )
        return await super().aggregate_metrics(body)

    async def verify(self, request: Request, body: SearchVerifyRequest) -> SearchVerifyResponse:
        """Grade the final answer, once per session.

        The grade reads only the request, so a verify without a seed is graded too (allow_missing):
        gym eval reverify, or an agent run whose cookie was lost. An identical retry replays the
        verdict, or is graded again when the previous attempt reached none; a retry with another
        body gets 409. A verify after the session's close is masked as lost.

        With plugin_log_path set, an episode session's verify first reads the plugin log from its sandbox, and the
        ending and plugin receipts in it stand in for what the relay agent sends (_take_plugin_receipts). With
        episode_export_copy_dir set, it then copies the harness's session exports to this host (_copy_exports), which
        never changes the grade. An episode's verify for a sandbox this server no longer holds is masked as lost too
        (_lost_session_reason).
        """
        session_id = request.session[SESSION_ID_KEY]
        digest = _body_digest(body)
        async with self.verifying(session_id, allow_missing=True) as verification:
            lost = self._lost_session_reason(request, verification)
            if lost is not None:
                return SearchVerifyResponse(**(body.model_dump() | _NOT_GRADED | lost_session_verification(lost)))
            evidence = verification.retained
            if not isinstance(evidence, _VerifyEvidence):
                # Read and copied while the sandbox still runs: it is stopped before the judge. A read that fails
                # retains nothing, so a retry of this verify is masked as lost rather than graded without the log. The
                # copy never fails the verify, and a retry replays it.
                plugin_log = await self._read_plugin_log(session_id)
                evidence = _VerifyEvidence(
                    digest, plugin_log=plugin_log, export_copy=await self._copy_exports(session_id)
                )
                verification.retain(evidence)
            elif evidence.body_digest != digest:
                raise ResourcesSessionConflictError("This session was already verified with another body")
            if evidence.export_copy is not None:
                # On the body, where the plugin receipts go, so every reply built from it carries the copy.
                body.opencode_export_copy = asdict(evidence.export_copy)
            if evidence.plugin_log is not None:
                self._take_plugin_receipts(body, evidence.plugin_log)
            if evidence.grade is not None:
                return SearchVerifyResponse(**(body.model_dump() | evidence.grade))
            if self.has_session_state(session_id):
                # Grading reads nothing more from the sandbox, so stop it before waiting on the judge. If the stop
                # fails, verifying() tries again on exit and records the failure without replacing the grade.
                try:
                    await self.release_session_state(session_id)
                except Exception:
                    LOGGER.warning(
                        "Failed to stop the sandbox of session %s before grading", session_id, exc_info=True
                    )
            grade = await self._grade(body)
            if grade["evaluation_completed"]:
                verification.retain(replace(evidence, grade=grade))
            return SearchVerifyResponse(**(body.model_dump() | grade))

    def _lost_session_reason(self, request: Request, verification: ResourcesVerification) -> Optional[str]:
        """Why this verify cannot grade a session this server no longer holds, or None.

        Besides a closed session: with plugin_log_path set, an episode's grade depends on the plugin log in its sandbox
        (the plugin gate and the budget masks), so an episode's verify for a session this server holds neither the
        sandbox nor a first attempt's evidence for, as after a restart, which stops every sandbox, cannot be graded
        either. Only an Environment Server's episode verify sends CHECKPOINT_CLOSE_FOLLOWS_HEADER; a verify without it,
        from the relay agent or gym eval reverify, reads only its body and is graded.
        """
        if verification.is_lost:
            return "the session was closed, or its seed never finished"
        if (
            self.config.plugin_log_path is not None
            and request.headers.get(CHECKPOINT_CLOSE_FOLLOWS_HEADER) == "1"
            and not verification.is_live
            and verification.retained is None
        ):
            return "this server no longer holds the episode's sandbox, so its plugin log cannot be read"
        return None

    async def _read_plugin_log(self, session_id: str) -> Optional[_PluginLog]:
        """The plugin log in an episode session's sandbox, empty when no plugin wrote one; None when there is none to
        read.

        There is none without plugin_log_path, so a harness that brings no such plugins is graded as before, and none
        for an agent's own /run, whose agent sends the ending in the verify body.

        The read prints the file's size in bytes and then its bytes in base64, or that the file is absent. Base64,
        because a provider may return a command's output as messages joined by newlines, and a message boundary inside
        a plugin line would split it; decoding comes after every whitespace character is dropped. The size, because a
        provider that caps a command's output could cut the base64 at the end of a group, which decodes to a log
        without its last lines, its ending among them: a read whose bytes differ from the size has failed.

        A read that fails raises a sandbox failure with a provider failure kind, and the server answers 500, which an
        Environment Server treats as retryable: the episode fails ungraded and runs again on resume, as a relay run
        whose output could not be captured does. A row masked instead would stay in the results with a reward that
        measured nothing, and lower a score that counts masked rows.
        """
        sandbox = None if self._lent_sandboxes is None else self._lent_sandboxes.get(session_id)
        if self.config.plugin_log_path is None or sandbox is None:
            return None
        path = shlex.quote(self.config.plugin_log_path)
        # From /, which exists whatever the harness did to its workdir. head copies exactly the bytes wc counted, so a
        # plugin that is still writing cannot make the two disagree.
        result = await sandbox.exec(
            f"if test -e {path}; then size=$(wc -c < {path}) && printf '%s:' $size "
            f"&& head -c $size -- {path} | base64; else printf absent:; fi",
            cwd="/",
            timeout_s=60,
        )
        if result.return_code != 0 or result.error_type is not None:
            raise self._plugin_log_read_failure(
                sandbox,
                f"exit {result.return_code}, error {result.error_type}: {(result.stderr or '').strip()[-500:]}",
                failure_kind=PROVIDER_IO_TIMEOUT if result.error_type == "timeout" else PROVIDER_REQUEST_FAILED,
                provider_status_reason=result.error_type,
                # With an error type the provider reported no exit of the command.
                exit_code=result.return_code if result.error_type is None else None,
            )
        output = "".join((result.stdout or "").split())
        if output == "absent:":
            return _PluginLog(text="", size=0, exists=False)
        size, _, encoded = output.partition(":")
        try:
            expected, data = int(size), base64.b64decode(encoded, validate=True)
        except ValueError:  # Also binascii.Error, for base64 cut inside a group.
            expected, data = None, b""
        if len(data) != expected:
            raise self._plugin_log_read_failure(
                sandbox,
                f"it returned {len(data)} of the file's {size[:20]!r} bytes: its output was cut short or malformed",
            )
        return _PluginLog(text=data.decode("utf-8", errors="replace"), size=len(data), exists=True)

    def _plugin_log_read_failure(
        self, sandbox: AsyncSandbox, what: str, *, failure_kind: str = PROVIDER_REQUEST_FAILED, **facts: Any
    ) -> NgFailureEvent:
        """The failure a plugin log read that failed raises, with what went wrong and the sandbox it ran in."""
        return sandbox_failure(
            operation="exec",
            failure_kind=failure_kind,
            reason=f"Cannot read the plugin log {self.config.plugin_log_path} in the lent sandbox ({what})",
            sandbox_id=sandbox.sandbox_id,
            provider=sandbox.provider_name,
            **facts,
        )

    def _take_plugin_receipts(self, body: SearchVerifyRequest, plugin_log: _PluginLog) -> None:
        """Put the ending and plugin receipts from an episode's plugin log on its verify body, where the relay agent puts
        them, so every reply built from the body carries them, judge_failsafe's included. The ending comes from the
        whole log, of which the body carries at most _PLUGIN_LOG_ROW_CHARS characters.

        A log without the sentinel or a required marker means the run did not have the plugins it was configured with,
        and its answer is not graded. The refusal is terminal (422): the log was read, and it would read the same again.
        Its detail names the missing lines and gives whether the file existed, its size in bytes and both ends of it,
        with line breaks escaped; its failure report names PLUGINS_NOT_LOADED.
        """
        text = plugin_log.text
        missing = [marker for marker in (_PLUGIN_SENTINEL, *self.config.required_plugin_markers) if marker not in text]
        if missing:
            reason = f"The harness's OpenCode plugins did not load as configured: the plugin log lacks {missing}"
            observe_failure(
                NgFailureEvent(
                    boundary=FailureBoundary.RESOURCES_SERVER,
                    failure_kind=PLUGINS_NOT_LOADED,
                    failure_reason=bounded_text(reason, 2000),
                    operation="verify.plugin_log",
                    stage="verification",
                )
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{reason}. Refusing to grade a run that ran without them. exists={plugin_log.exists} "
                    f"log_bytes={plugin_log.size} head={text[:_PLUGIN_LOG_SLICE_CHARS]!r} "
                    f"tail={text[-_PLUGIN_LOG_SLICE_CHARS:]!r}"
                ),
            )
        body.opencode_terminal_reason = terminal_reason_from_plugin_log(text)
        body.opencode_plugins_loaded = True
        body.opencode_plugin_log = _plugin_log_on_the_row(text)

    async def _copy_exports(self, session_id: str) -> Optional[_ExportCopy]:
        """Copy the session exports out of an episode's sandbox to this host, and say how that went; None when there is
        nothing to copy to: without episode_export_copy_dir, and for an agent's own /run, which has no lent sandbox.

        The files go to <episode_export_copy_dir>/<capture key>/<task id>/attempt-<attempt>/, emptied first, so a
        retried episode never finds an earlier attempt's files there. The copy is evidence about the run, not part of
        its grade: one that fails, runs past _EXPORT_COPY_TIMEOUT_S or is refused is logged and reported, never raised,
        and a copy that did not finish keeps no file.
        """
        sandbox = None if self._lent_sandboxes is None else self._lent_sandboxes.get(session_id)
        identity = self.session_identity(session_id)
        if self.config.episode_export_copy_dir is None or sandbox is None or identity is None:
            return None
        episode, task = identity
        target = Path(
            self.config.episode_export_copy_dir,
            episode.capture_key,
            _host_directory_name(task.task_id),
            f"attempt-{episode.attempt}",
        )
        deadline = asyncio.timeout(_EXPORT_COPY_TIMEOUT_S)
        try:
            async with deadline:
                copy = await self._copy_exports_to(sandbox, target)
        # Deliberate boundary: a failed copy is reported in the row and leaves the grade as it is.
        except Exception as error:
            await asyncio.to_thread(shutil.rmtree, target, ignore_errors=True)
            reason = (
                f"The copy ran past {_EXPORT_COPY_TIMEOUT_S} s"
                if deadline.expired()
                else bounded_text(f"{type(error).__name__}: {error}", 2000)
            )
            copy = _ExportCopy("failed", str(target), detail=reason)
        if copy.status != "copied":
            LOGGER.warning(
                "The session exports of %s were not copied from %s to %s (%s): %s",
                session_id,
                self.config.episode_export_dir,
                target,
                copy.status,
                copy.detail,
            )
        return copy

    async def _copy_exports_to(self, sandbox: AsyncSandbox, target: Path) -> _ExportCopy:
        """Copy the regular files directly in episode_export_dir to target, unless they hold more than
        episode_export_max_bytes. Raises when the listing or a download fails, or a copy's size differs from the
        listed one."""
        # The host's file operations run off the event loop: the directory may be on a shared filesystem.
        await asyncio.to_thread(_remove_directory, target)
        files = await self._list_exports(sandbox)
        if files is None:
            return _ExportCopy(
                "absent", str(target), detail=f"The lent sandbox has no {self.config.episode_export_dir}"
            )
        total = sum(files.values())
        if total > self.config.episode_export_max_bytes:
            return _ExportCopy(
                "oversize",
                str(target),
                files,
                total,
                f"{total} bytes exceed episode_export_max_bytes ({self.config.episode_export_max_bytes})",
            )
        await asyncio.to_thread(target.mkdir, parents=True, exist_ok=True)
        source = PurePosixPath(self.config.episode_export_dir)
        for name in files:
            await sandbox.download(str(source / name), target / name)
        copied = await asyncio.to_thread(_file_sizes, target, list(files))
        if copied != files:
            short = {name: f"{copied[name]} of {size}" for name, size in files.items() if copied[name] != size}
            raise RuntimeError(f"Copies whose size differs from the listing, in bytes: {short}")
        return _ExportCopy("copied", str(target), files, total)

    async def _list_exports(self, sandbox: AsyncSandbox) -> Optional[Dict[str, int]]:
        """The regular files directly in episode_export_dir, by name, with their sizes in bytes; None when the directory
        does not exist. Raises when the listing fails.

        One command, from / as the plugin log read. It lists no link, no directory and no name outside the plain names
        of _EXPORT_LISTING_ENTRY, read in the C locale, so a file the model's tools left there with another name is not
        copied and cannot break the listing. Its output ends with a marker, so a listing cut short is refused, and its
        whitespace is dropped before it is parsed, as a provider may split a command's output with newlines (see
        _read_plugin_log).
        """
        directory = shlex.quote(self.config.episode_export_dir)
        result = await sandbox.exec(
            f"export LC_ALL=C; test -e {directory} || {{ printf absent; exit 0; }}; cd {directory} || exit 1; "
            "for name in *; do case $name in *[!A-Za-z0-9._-]*) continue ;; esac; "
            'if test -f "$name" && ! test -L "$name"; then size=$(wc -c < "$name") || exit 1; '
            'printf "%s:%s/" $size "$name"; fi; done; printf end',
            cwd="/",
            timeout_s=60,
        )
        if result.return_code != 0 or result.error_type is not None:
            raise RuntimeError(
                f"Cannot list {self.config.episode_export_dir} in the lent sandbox (exit {result.return_code}, error "
                f"{result.error_type}): {(result.stderr or '').strip()[-500:]}"
            )
        output = "".join((result.stdout or "").split())
        if output == "absent":
            return None
        entries = output.removesuffix("end").split("/")
        listed = [_EXPORT_LISTING_ENTRY.fullmatch(entry) for entry in entries[:-1]]
        if not output.endswith("end") or entries[-1] or not all(listed):
            raise RuntimeError(
                f"The listing of {self.config.episode_export_dir} was cut short or malformed: {output[-200:]!r}"
            )
        return {match[2]: int(match[1]) for match in listed}

    async def _grade(self, body: SearchVerifyRequest) -> Dict[str, Any]:
        """The grading fields of the verify response: reward, mask and failure fields, and the judge's diagnostics."""
        # The turn-budget plugin's ending, which the relay agent forwards and an episode's plugin log gives; "" on a
        # normal ending.
        terminal_reason = getattr(body, "opencode_terminal_reason", "")
        masked = terminal_reason in MASKED_TERMINAL_REASONS
        failure_kind, failure_reason = MASKED_TERMINAL_FAILURES.get(terminal_reason, (None, None))

        final_text = _last_assistant_text(body.response)
        if not final_text.strip():
            # The agent produced no answer at all. That is a real zero, not judge noise, and
            # spending a judge call on an empty string would only add latency.
            return _NOT_GRADED | {
                "reward": 0.0,
                "mask_sample": masked,
                "failure_kind": failure_kind,
                "failure_reason": failure_reason,
                "evaluation_completed": True,
            }

        try:
            result = await self._judge_answer(
                question=body.question,
                ground_truth=body.ground_truth,
                final_text=final_text,
            )
        except JudgeError:
            # judge_failsafe answers this with a masked row built from the request body, so the row is marked
            # inconclusive there.
            body.evaluation_completed = False
            raise
        if not result["judge_parsed"] and not masked:
            # Without a verdict the reward measures nothing. A budget ending keeps its own kind, and judge_parsed and
            # evaluation_completed record the missing verdict: recovery re-judges a judge_unparseable row from its
            # saved answer, with no ending to mask it again.
            masked, failure_kind = True, JUDGE_UNPARSEABLE
            failure_reason = f"The judge gave no parseable verdict in {result['judge_attempts']} attempt(s)"

        return {
            "reward": result["reward"],
            "extracted_final_answer": result["extracted_final_answer"],
            "final_text": final_text,
            "judge_parsed": result["judge_parsed"],
            "judge_attempts": result["judge_attempts"],
            "judge_time_taken": result["judge_time_taken"],
            "mask_sample": masked,
            "failure_kind": failure_kind,
            "failure_reason": failure_reason,
            "evaluation_completed": result["judge_parsed"],
        }


if __name__ == "__main__":
    SearchResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = SearchResourcesServer.run_webserver()  # noqa: F401
