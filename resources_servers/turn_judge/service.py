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
"""Turn-judge service: the offline judge (offline.py, variant D, one judgment per pair) behind HTTP, for GRPO training.

    python -m resources_servers.turn_judge.service --labeling LABELING_annotator.md --endpoint-file <qwen env file>
        --service-env-out <file> --work-root /tmp/tj-service/<run id> --archive-dir <lustre dir>
        [--port 18300] [--max-sessions 240] [--run-id tj-train-<run>] [--host-ip <ip>]
(launched by turn_judge/judge_sandbox/train/judge_service_step.sh inside the training image).

API (turn_judge iterations/PHASE3_IMPL.md, "Judge service"): POST /judge_group (idempotent by request_id),
GET /judge_group/<request_id>, DELETE /judge_group/<request_id> (cancel: sandboxes stopped, the request is forgotten),
GET /health.

Per request (one sealed GRPO group): build the group workspace on node-local disk from the payload (task row + each
rollout's response.output, read from the generation.json beside ``opencode_results_fpath`` or given inline; R<k> =
generation_index order), then the offline judge: no outcome contrast -> ``no_pairs``; stage 1 = one group-analyst
session (no retry; failure or no usable decisions.json -> ``stage1_failed``; ill-posed -> ``ill_posed``); one session
per pair (``make_pairs``), each capped at ``pair_timeout_s``; at ``deadline_unix`` unfinished sessions are cancelled
(sandboxes stopped) and the pairs that finished are merged (``partial``). A pair without a verdict (capped, failed,
cancelled) is a missing judgment: it is left out of the merge (user, 2026-11-14: "a capped pair = a missing judgment").

Judge endpoint: the Qwen serving chain replaces the endpoint file atomically when a new job is ready and removes it
when its job exits. Each session takes the endpoint current when it starts (running sessions keep theirs); while the
file is missing, sessions wait for it (polling every --endpoint-poll s) until the request's deadline, then the request
returns what it has (stage1_failed / partial); waits are reported in info.endpoint_wait_s.

Storage (mars-shared-storage): workspaces, the answer-key cache, logs (pairs.jsonl, results.jsonl, service.log) and
the OpenCode session exports on node-local disk under --work-root; reads per request: the rollouts' generation.json and,
for Terminal-Bench, the task folder (once per task, cached). At exit (SIGTERM / SIGINT / POST /shutdown) every session
is cancelled (sandboxes stopped), the run's sandboxes are reaped, and the logs are archived ONCE to
<archive-dir>/service-<SLURM_JOB_ID>.tar.gz. Sandboxes carry the run label (--run-id, shared by the jobs of one training
run, made Kubernetes-label safe): the startup reap deletes that label's orphans before any session starts."""

import argparse
import asyncio
import contextlib
import dataclasses
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from resources_servers.turn_judge import runner, workspace
from resources_servers.turn_judge.call_alignment import call_alignment, step_output_tokens
from resources_servers.turn_judge.merge import call_of, call_starts, merge_group, parse_json
from resources_servers.turn_judge.offline import (
    STAGE1_INSTRUCTIONS,
    group_files,
    group_hedges,
    load_endpoint,
    load_group,
    pair_files,
    sandbox_provider,
    task_image,
)
from resources_servers.turn_judge.pairing import Pair, make_pairs
from resources_servers.turn_judge.prompt import build_instructions
from resources_servers.turn_judge.runner import PairRunSettings

LOG = logging.getLogger("turn_judge.service")
DECISION = ("good", "harmful")
# Merge settings of the variant-D evaluation (PHASE3_IMPL.md); a request may override them (``overrides``).
DEFAULT_OVERRIDES = {
    "seed": "0",
    "variant": "D",
    "temperature": 0.3,
    "vote_frac": 0.5,
    "ill_frac": 0.0,
    "ill_min_votes": 2,
    "good_earliest": False,
}
STAGE1_VARIANTS = ("C", "D", "E")  # two-stage variants (stage-1 decision list); the service runs only these
DEFAULT_JUDGE_ENDPOINT = os.environ.get("TURN_JUDGE_MODEL_ENDPOINT_FILE")


# ------------------------------------------------------------------------------------------------ request schema


class RolloutIn(BaseModel):
    model_config = ConfigDict(extra="allow")  # e.g. call_starts: accepted, unused

    generation_index: int
    reward: float
    mask_sample: bool = False
    context_limited: bool = False
    cap_killed: bool = False
    evaluation_completed: Optional[bool] = None
    resolved: Optional[bool] = None
    test_results: Any = None
    test_output: Optional[str] = None
    test_return_code: Optional[int] = None
    model_patch: Optional[str] = None
    patch_applied: Optional[bool] = None
    opencode_results_fpath: Optional[str] = None
    n_calls: Optional[int] = None
    response_output: Optional[list] = None


class JudgeRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    request_id: str
    group_id: str
    agent_name: str = ""
    deadline_unix: Optional[float] = None
    stage1_timeout_s: float = 900.0
    pair_timeout_s: float = 1200.0
    overrides: dict[str, Any] = Field(default_factory=dict)
    task: dict[str, Any]
    rollouts: list[RolloutIn]


class BadRequest(ValueError):
    pass


def check_request(req: JudgeRequest) -> dict:
    """Validate what pydantic cannot; returns the effective overrides."""
    if not req.request_id or not req.group_id:
        raise BadRequest("request_id and group_id must be non-empty")
    if not req.rollouts:
        raise BadRequest("no rollouts")
    gens = [r.generation_index for r in req.rollouts]
    if len(set(gens)) != len(gens):
        raise BadRequest(f"duplicate generation_index in {sorted(gens)}")
    for r in req.rollouts:
        if r.response_output is None and not r.opencode_results_fpath:
            raise BadRequest(f"rollout {r.generation_index}: neither opencode_results_fpath nor response_output")
    unknown = sorted(set(req.overrides) - set(DEFAULT_OVERRIDES))
    if unknown:
        raise BadRequest(f"unknown overrides {unknown}; known: {sorted(DEFAULT_OVERRIDES)}")
    ov = {**DEFAULT_OVERRIDES, **req.overrides}
    ov["seed"] = str(ov["seed"])
    ov["variant"] = str(ov["variant"]).upper()
    if ov["variant"] not in STAGE1_VARIANTS:
        raise BadRequest(f"variant {ov['variant']!r}: the service runs the two-stage variants {STAGE1_VARIANTS}")
    if req.stage1_timeout_s <= 0 or req.pair_timeout_s <= 0:
        raise BadRequest("stage1_timeout_s and pair_timeout_s must be > 0")
    try:
        task_image(req.task)
    except KeyError as e:
        raise BadRequest(str(e)) from None
    return ov


def generation_json_path(fpath: str) -> Path:
    """The rollout's generation.json: Gym's opencode_results_fpath names the OpenCode export (export.json) in the
    rollout's artifact dir; generation.json ({"response": {"output": [...]}, "execution": {...}}) is its sibling."""
    p = Path(fpath)
    return p if p.name == "generation.json" else p.with_name("generation.json")


def read_trajectory(r: RolloutIn) -> list:
    if r.response_output is not None:
        return r.response_output
    with open(generation_json_path(r.opencode_results_fpath)) as f:
        g = json.load(f)
    return (g.get("response") or {}).get("output") or []  # = group_workspace.train_source + build


def rollout_alignment(r: RolloutIn, output: list) -> dict:
    """Whether the judge's calls are OpenCode's model calls one to one, and how many empty model calls end the session
    (call_alignment.py). Not proven (no export, unreadable, any mismatch) -> aligned False: the trainer then applies
    no label to this rollout. ``step_output_tokens``: each OpenCode model call's generated-token count (all steps,
    trailing empty ones included), for the trainer's run-by-run identity check; None when not attributable."""
    tokens = None
    if not r.opencode_results_fpath:
        aligned, k, why = False, 0, "no_export"
    else:
        p = Path(r.opencode_results_fpath)
        p = p if p.name == "export.json" else p.with_name("export.json")
        try:
            with open(p) as f:
                export = json.load(f)
            aligned, k, why = call_alignment(export, output)
            tokens = step_output_tokens(export)
        except Exception as error:  # unreadable export: not proven
            aligned, k, why = False, 0, f"export_unreadable: {type(error).__name__}"
    return {"aligned": aligned, "trailing_empty_calls": k, "alignment_reason": why, "step_output_tokens": tokens}


def workspace_record(r: RolloutIn, output: list) -> dict:
    """The rollout record group_workspace.build reads (verify.json fields + response), rebuilt from the payload."""
    return {
        "reward": r.reward,
        "resolved": r.resolved,
        "mask_sample": r.mask_sample,
        "opencode_context_limited": r.context_limited,
        # build() only tests `"signal: killed" in opencode_run_stderr` (the trainer stashes the boolean)
        "opencode_run_stderr": "signal: killed" if r.cap_killed else "",
        "evaluation_completed": r.evaluation_completed,
        "test_results": r.test_results,
        "test_output": r.test_output,
        "test_return_code": r.test_return_code,
        "model_patch": r.model_patch,
        "patch_applied": r.patch_applied,
        "response": {"output": output},
    }


def build_group_workspace(ws: Path, req: JudgeRequest, answer_key: Optional[Callable] = None) -> dict:
    """Write the group workspace (as tools/group_workspace.py train + build) into ``ws``; returns the R<k> order
    (generation indices), the outputs and the per-rollout call counts. ``answer_key(task, row, ws)`` (the service's
    cache) copies TASK.md + answer_key/ into ws and returns write_answer_key's tuple; None = write them here."""
    rows = sorted(req.rollouts, key=lambda r: r.generation_index)
    outputs = [read_trajectory(r) for r in rows]
    task = req.task.get("instance_id") or req.task.get("task_name") or req.group_id
    ws.mkdir(parents=True, exist_ok=True)
    ak = answer_key(task, req.task, ws) if answer_key else None
    locator = ws / "rollouts.jsonl"
    rollouts = []
    with open(locator, "w") as f:
        for k, (r, out) in enumerate(zip(rows, outputs)):
            f.write(json.dumps({"generation_index": r.generation_index, "response": {"output": out}}) + "\n")
            loc = {"file": str(locator), "line": k}
            if r.opencode_results_fpath:
                loc["artifact"] = os.path.dirname(r.opencode_results_fpath)
            rollouts.append((loc, workspace_record(r, out)))
    workspace.build(str(ws), task, req.task, rollouts, answer_key=ak)
    return {
        "task": task,
        "order": [r.generation_index for r in rows],
        "outputs": outputs,
        "n_calls": [len(call_starts(o)) for o in outputs],
        "n_calls_trainer": [r.n_calls for r in rows],
        "alignment": [rollout_alignment(r, o) for r, o in zip(rows, outputs)],
    }


# ------------------------------------------------------------------------------------------------ service


@dataclasses.dataclass
class ServiceConfig:
    labeling: bytes
    endpoint_env: str
    work_root: Path
    run_id: str  # sandbox run label (label-safe; see sandbox_run_label)
    max_sessions: int = 240
    keep_s: float = 24 * 3600.0
    keep_workspaces: bool = False
    build_threads: int = 8
    sandbox_ttl_s: int = 5400
    archive_dir: Optional[str] = None
    archive_name: str = ""  # default: <run_id>.tar.gz
    keep_exports: bool = False  # archive the OpenCode session exports too (GBs per 4 h job); else node-local only
    clean_local: bool = True  # at exit, after the archive: remove the node-local exports, workspaces, caches
    gym_sha: str = ""
    ak_cache_max: int = 1024
    endpoint_poll_s: float = 15.0  # while the judge endpoint file is missing, sessions re-check it this often
    endpoint_max_age_s: float = 1.0  # a session start re-reads the file if the last check is older than this
    shutdown_wait_s: float = 30.0  # SIGTERM -> SIGKILL is 120 s on this cluster: cancel 30 + stops 15 + reap 30
    reap_timeout_s: float = 300.0


@dataclasses.dataclass
class Job:
    request_id: str
    group_id: str
    agent_name: str
    created: float
    deadline_unix: Optional[float]
    state: str = "queued"
    result: Optional[dict] = None
    done_at: Optional[float] = None
    task: Optional[asyncio.Task] = None
    sessions: set = dataclasses.field(default_factory=set)
    cancel_reason: Optional[str] = None
    dir: Optional[Path] = None
    build: Optional[Future] = None  # the workspace build (worker thread; done only when the thread is)
    sessions_started: int = 0
    endpoint_wait_s: float = 0.0  # summed over the request's sessions: time spent waiting for the endpoint file
    endpoint_wait_max_s: float = 0.0
    endpoint_waiting: int = 0  # sessions waiting for the endpoint file right now
    startup_wait_s: float = 0.0  # longest wait of a session for the startup reap
    summary: dict = dataclasses.field(default_factory=dict)  # payload summary (no test logs / patches / outputs)

    def body(self) -> dict:
        return {"request_id": self.request_id, "state": self.state, "result": self.result}


class JudgeService:
    """The judge logic per request plus the bookkeeping (jobs, session semaphore, logs, sandbox reaping).
    ``run_pair`` and ``reaper`` are injectable (tests)."""

    def __init__(
        self,
        cfg: ServiceConfig,
        provider: Optional[dict] = None,
        default_metadata: Optional[dict] = None,
        run_pair: Callable[..., Awaitable[dict]] = runner.run_pair,
        reaper: Optional[Callable[..., Awaitable[dict]]] = None,
    ):
        self.cfg = cfg
        self.provider = provider
        self.metadata = {**(default_metadata or {}), "role": "turn-judge"}
        self.run_pair = run_pair
        self.reaper = reaper
        self.labeling_sha = hashlib.sha1(cfg.labeling).hexdigest()
        self.jobs: dict[str, Job] = {}
        self.sem = asyncio.Semaphore(cfg.max_sessions)
        self.active_sessions = 0
        self.waiting_sessions = 0
        self.sessions_total = 0
        self.closing = False
        self.started = time.time()
        self.logs = cfg.work_root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        (cfg.work_root / "jobs").mkdir(parents=True, exist_ok=True)
        self.ak_root = cfg.work_root / "answer_keys"
        self.ak_root.mkdir(parents=True, exist_ok=True)
        self._ak: dict[str, tuple] = {}  # key hash -> write_answer_key tuple (insertion order = age)
        self._ak_lock = threading.Lock()
        self._ak_key_locks: dict[str, threading.Lock] = {}
        self.build_pool = ThreadPoolExecutor(cfg.build_threads, thread_name_prefix="tj-build")
        self.log_pool = ThreadPoolExecutor(1, thread_name_prefix="tj-log")  # one writer: ordered appends
        # Archived at exit: logs/ = pairs.jsonl (one row per session: status, timings, decisions.json / pair.json text,
        # notes), results.jsonl (one row per request: payload summary + result), service.log. The OpenCode session
        # exports stay node-local (work_root/exports.jsonl.gz) unless keep_exports.
        self._plog = open(self.logs / "pairs.jsonl", "a")
        self.exports_path = (self.logs if cfg.keep_exports else cfg.work_root) / "exports.jsonl.gz"
        self._xlog = gzip.open(self.exports_path, "at")
        self._rlog = open(self.logs / "results.jsonl", "a")
        self._ep: Optional[tuple[str, str]] = None
        self._ep_mtime: Optional[tuple] = None
        self._ep_checked = 0.0
        self._ep_error = ""
        self._maint: Optional[asyncio.Task] = None
        self.on_closing: Optional[Callable[[], None]] = None  # called first in shutdown (e.g. withdraw the URL file)
        # Sessions start only after the startup reap: the run label is shared by the run's successive jobs, so the
        # reap of the previous job's orphans must not race this job's new sandboxes.
        self._ready = asyncio.Event()
        self._ready.set()
        self._startup: Optional[asyncio.Task] = None
        self._endpoint()

    # ---------------------------------------------------------------- judge endpoint file
    # The serving chain replaces the file atomically when a new job is ready and deletes it when its job exits (if it
    # still names that job): a session takes the endpoint current when it starts; a missing file means no live judge,
    # so sessions wait for it (polling every endpoint_poll_s) until their request's deadline.
    def _endpoint(self, max_age_s: Optional[float] = None) -> Optional[tuple[str, str]]:
        max_age_s = self.cfg.endpoint_max_age_s if max_age_s is None else max_age_s
        now = time.time()
        if now - self._ep_checked < max_age_s:
            return self._ep
        self._ep_checked = now
        try:
            st = os.stat(self.cfg.endpoint_env)
            mtime = (st.st_ino, st.st_mtime_ns, st.st_size)  # an atomic replace changes the inode
            if mtime != self._ep_mtime or self._ep is None:
                ep = load_endpoint(self.cfg.endpoint_env)
                if ep != self._ep:
                    LOG.info("judge endpoint %s (%s) from %s", ep[0], ep[1], self.cfg.endpoint_env)
                self._ep, self._ep_mtime, self._ep_error = ep, mtime, ""
        except Exception as e:  # missing (between serving jobs) or unparseable: no endpoint until it reappears
            msg = f"{type(e).__name__}: {e}"
            if self._ep is not None or msg != self._ep_error:
                LOG.warning(
                    "judge endpoint file %s unavailable (%s): new sessions wait for it", self.cfg.endpoint_env, msg
                )
            self._ep, self._ep_mtime, self._ep_error = None, None, msg
        return self._ep

    async def _wait_endpoint(self, job: Job) -> tuple[str, str]:
        """The current endpoint; while the file is missing, poll every endpoint_poll_s (the request's deadline cancels
        the wait)."""
        ep = self._endpoint()
        if ep is not None:
            return ep
        t0 = time.time()
        job.endpoint_waiting += 1
        try:
            while ep is None:
                await asyncio.sleep(self.cfg.endpoint_poll_s)
                ep = self._endpoint(max_age_s=min(self.cfg.endpoint_max_age_s, self.cfg.endpoint_poll_s))
            return ep
        finally:
            job.endpoint_waiting -= 1
            waited = time.time() - t0
            job.endpoint_wait_s += waited
            job.endpoint_wait_max_s = max(job.endpoint_wait_max_s, waited)

    # ---------------------------------------------------------------- answer-key cache (node-local, once per task)
    def _answer_key(self, task: str, row: dict, ws: Path) -> tuple:
        h = hashlib.sha256(json.dumps([task, row], sort_keys=True, default=str).encode()).hexdigest()[:24]
        with self._ak_lock:
            lock = self._ak_key_locks.setdefault(h, threading.Lock())
        with lock:
            with self._ak_lock:
                ak = self._ak.get(h)
            src = self.ak_root / h
            if ak is None:
                tmp = self.ak_root / f".{h}.tmp"
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir(parents=True)
                ak = workspace.write_answer_key(str(tmp), task, row)
                shutil.rmtree(src, ignore_errors=True)
                tmp.rename(src)
                with self._ak_lock:
                    self._ak[h] = ak
                    old = list(self._ak)[: max(0, len(self._ak) - self.cfg.ak_cache_max)]
                    for o in old:
                        self._ak.pop(o, None)
                for o in old:
                    shutil.rmtree(self.ak_root / o, ignore_errors=True)
            shutil.copytree(src, ws, dirs_exist_ok=True)
        return ak

    # ---------------------------------------------------------------- logs (one writer thread, node-local)
    def _write(self, fh, text: str) -> None:
        fh.write(text)
        fh.flush()

    def _log_row(self, fh, row: dict) -> None:
        self._submit_write(fh, json.dumps(row, default=str) + "\n")

    def _submit_write(self, fh, text: str) -> None:
        try:
            self.log_pool.submit(self._write, fh, text)
        except RuntimeError:  # after shutdown closed the logs (a straggler past shutdown_wait_s)
            pass

    # ---------------------------------------------------------------- HTTP-facing operations
    def submit(self, req: JudgeRequest) -> dict:
        job = self.jobs.get(req.request_id)
        if job is not None:
            return job.body()
        if self.closing:
            raise RuntimeError("shutting down")
        ov = check_request(req)
        job = Job(
            request_id=req.request_id,
            group_id=req.group_id,
            agent_name=req.agent_name,
            created=time.time(),
            deadline_unix=req.deadline_unix,
        )
        job.summary = {
            "task": req.task.get("instance_id") or req.task.get("task_name"),
            "deadline_unix": req.deadline_unix,
            "stage1_timeout_s": req.stage1_timeout_s,
            "pair_timeout_s": req.pair_timeout_s,
            "overrides": req.overrides,
            "rollouts": [
                {
                    k: getattr(r, k)
                    for k in (
                        "generation_index",
                        "reward",
                        "resolved",
                        "mask_sample",
                        "context_limited",
                        "cap_killed",
                        "evaluation_completed",
                        "patch_applied",
                        "n_calls",
                        "opencode_results_fpath",
                    )
                }
                | {"inline_output": r.response_output is not None}
                for r in req.rollouts
            ],
        }
        self.jobs[req.request_id] = job
        job.task = asyncio.get_running_loop().create_task(self._run(job, req, ov))
        return job.body()

    def get(self, request_id: str) -> Optional[dict]:
        job = self.jobs.get(request_id)
        return job.body() if job else None

    async def delete(self, request_id: str) -> Optional[dict]:
        """Cancel (sandboxes stopped) and forget: a later POST with the same id starts a new judgment."""
        job = self.jobs.pop(request_id, None)
        if job is None:
            return None
        if job.task and not job.task.done():
            job.cancel_reason = "DELETE"
            job.task.cancel()
            await asyncio.gather(job.task, return_exceptions=True)
        return job.body()

    def health(self) -> dict:
        ep = self._endpoint(max_age_s=self.cfg.endpoint_poll_s)
        return {
            "ok": not self.closing,
            "running": sum(1 for j in self.jobs.values() if j.state != "done"),
            "sessions": self.active_sessions,
            "sessions_waiting": self.waiting_sessions,
            "sessions_total": self.sessions_total,
            "max_sessions": self.cfg.max_sessions,
            "judge_base_url": ep[0] if ep else None,
            "judge_model": ep[1] if ep else None,
            "jobs": len(self.jobs),
            "run_id": self.cfg.run_id,
            "gym_sha": self.cfg.gym_sha,
            "labeling_sha": self.labeling_sha,
            "uptime_s": round(time.time() - self.started, 1),
        }

    # ---------------------------------------------------------------- one request
    async def _run(self, job: Job, req: JudgeRequest, ov: dict) -> None:
        t0 = time.time()
        try:
            result = await self._judge(job, req, ov, t0)
        except asyncio.CancelledError:
            self._finish(job, self._result(job, "error", t0, error=f"cancelled ({job.cancel_reason or 'cancel'})"))
            raise
        except Exception as e:
            LOG.error("request %s failed: %s", job.request_id, traceback.format_exc())
            result = self._result(job, "error", t0, error=f"{type(e).__name__}: {e}"[:2000])
        self._finish(job, result)

    def _result(
        self,
        job: Job,
        status: str,
        t0: float,
        error: Optional[str] = None,
        labels: Optional[dict] = None,
        ill_posed_evidence: Optional[str] = None,
        **info: Any,
    ) -> dict:
        base = {
            "wall_s": round(time.time() - t0, 1),
            "stage1_s": None,
            "stage1_status": None,
            "pairs_total": 0,
            "pairs_ok": 0,
            "pairs_capped": 0,
            "pairs_failed": 0,
            "pairs_cancelled": 0,
            "pairs_salvaged": 0,
            "sessions": 0,
            "judge_model": None,
            "judge_base_url": None,
            "gym_sha": self.cfg.gym_sha,
            "run_id": self.cfg.run_id,
            "labeling_sha": self.labeling_sha,
            "endpoint_wait_s": round(job.endpoint_wait_s, 1),
            "endpoint_wait_max_s": round(job.endpoint_wait_max_s, 1),
            "startup_reap_wait_s": round(job.startup_wait_s, 1),
        }
        return {
            "status": status,
            "error": error,
            "ill_posed_evidence": ill_posed_evidence,
            "labels": labels or {},
            "info": {**base, **info},
        }

    def _finish(self, job: Job, result: dict) -> None:
        job.result, job.state, job.done_at = result, "done", time.time()
        self._log_row(
            self._rlog,
            {
                "request_id": job.request_id,
                "group_id": job.group_id,
                "agent": job.agent_name,
                "created": job.created,
                "done": job.done_at,
                "payload": job.summary,
                "result": result,
            },
        )
        LOG.info(
            "request %s (%s): %s %s wall %.0fs pairs %s/%s",
            job.request_id,
            job.group_id,
            result["status"],
            (result["error"] or "")[:200],
            result["info"]["wall_s"],
            result["info"]["pairs_ok"],
            result["info"]["pairs_total"],
        )
        fut, job.build = job.build, None  # a finished build holds the whole group: never keep it with the job
        if job.dir is not None and not self.cfg.keep_workspaces:
            d = job.dir
            if fut is not None and not fut.done():  # cancelled mid-build: remove once the thread is done
                fut.add_done_callback(lambda _: shutil.rmtree(d, ignore_errors=True))
            else:
                shutil.rmtree(d, ignore_errors=True)

    async def _judge(self, job: Job, req: JudgeRequest, ov: dict, t0: float) -> dict:
        try:
            return await self._judge_inner(job, req, ov, t0)
        finally:  # every way out (done, deadline, DELETE, shutdown, error): no session outlives its request
            pending = [t for t in job.sessions if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            job.sessions.clear()  # finished session tasks hold their results (exports: MBs) for the job's lifetime

    async def _judge_inner(self, job: Job, req: JudgeRequest, ov: dict, t0: float) -> dict:
        if req.deadline_unix is not None and req.deadline_unix <= time.time():
            return self._result(job, "error", t0, error="deadline_unix already passed at submission")
        job.dir = (
            self.cfg.work_root
            / "jobs"
            / (
                re.sub(r"[^A-Za-z0-9_.-]", "_", req.request_id)[:80]
                + "-"
                + hashlib.sha1(req.request_id.encode()).hexdigest()[:10]
            )
        )
        ws = job.dir / "ws"
        job.build = self.build_pool.submit(_build_and_load, ws, req, self._answer_key)
        built, group = await asyncio.wrap_future(job.build)
        job.build = None
        meta, order = group["meta"], built["order"]
        mism = {
            str(g): [t, s]
            for g, t, s in zip(order, built["n_calls_trainer"], built["n_calls"])
            if t is not None and t != s
        }
        common = {"rollout_order": order, "n_calls_mismatch": mism, "variant": ov["variant"], "seed": ov["seed"]}
        align = {str(g): a for g, a in zip(order, built["alignment"])}
        empty = {str(g): {"n_calls": n, "labels": [], **align[str(g)]} for g, n in zip(order, built["n_calls"])}
        common["alignment_not_proven"] = sum(1 for a in built["alignment"] if not a["aligned"])
        common["trailing_empty_rollouts"] = sum(1 for a in built["alignment"] if a["trailing_empty_calls"])
        pairs = make_pairs([r["reward"] for r in meta["rollouts"]], f"{ov['seed']}:{req.group_id}")
        if not pairs:
            return self._result(job, "no_pairs", t0, labels=empty, **common)
        image = task_image(req.task)
        row_log = {"request_id": req.request_id, "group": req.group_id, "task": built["task"]}

        # stage 1: one group-analyst session, no retry
        s1_files = group_files(group, self.cfg.labeling, ov["variant"])
        t1 = time.time()
        s1 = self._spawn(
            job,
            self._session(
                job,
                "stage1",
                image,
                s1_files,
                STAGE1_INSTRUCTIONS[ov["variant"]],
                req.stage1_timeout_s,
                ov,
                {**row_log, "fail": -1, "passing": -1, "sample": 0, "larger_side": "group-stage"},
                required="out/decisions.json",
                extra_reads=(),
            ),
        )
        done, _ = await asyncio.wait({s1}, timeout=_remaining(req.deadline_unix))
        waiting_ep = job.endpoint_waiting > 0
        if not done:
            s1.cancel()
            await asyncio.gather(s1, return_exceptions=True)
        gres = (
            s1.result()
            if done and not s1.cancelled()
            else {
                "status": "cancelled",
                "timings": {},
                "error": "deadline" + (" (waiting for the judge endpoint file)" if waiting_ep else ""),
            }
        )
        st = {
            "stage1_s": round(time.time() - t1, 1),
            "stage1_status": gres["status"],
            "sessions": job.sessions_started,
            "judge_model": gres.get("judge_model"),
            "judge_base_url": gres.get("judge_base_url"),
            **common,
        }
        text = gres.get("pair_json") if gres["status"] == "ok" else None
        d = parse_json(text)
        if d is None:
            err = (
                f"stage 1 {gres['status']}" + (f": {gres.get('error')}" if gres.get("error") else "")
                if not text
                else "stage 1 decisions.json is not parseable JSON"
            )
            return self._result(job, "stage1_failed", t0, error=err[:2000], labels=empty, **st)
        if d.get("group_ill_posed") is True:
            ev = d.get("ill_posed_evidence")
            return self._result(
                job, "ill_posed", t0, labels=empty, ill_posed_by="stage1", ill_posed_evidence=_text(ev), **st
            )
        if not d.get("decisions"):
            return self._result(
                job, "stage1_failed", t0, error="stage 1 decisions.json has no decision list", labels=empty, **st
            )

        # pairs: one judgment each, capped per session; cut at the deadline
        tasks: list[tuple[Pair, asyncio.Task]] = []
        for pr in pairs:
            instr = build_instructions(
                pr.fail,
                meta["rollouts"][pr.fail]["line"],
                pr.passing,
                meta["rollouts"][pr.passing]["line"],
                ov["variant"],
            )
            files = pair_files(group, pr.fail, pr.passing, self.cfg.labeling, ov["variant"], text)
            tasks.append(
                (
                    pr,
                    self._spawn(
                        job,
                        self._session(
                            job,
                            "pair",
                            image,
                            files,
                            instr,
                            req.pair_timeout_s,
                            ov,
                            {
                                **row_log,
                                "fail": pr.fail,
                                "passing": pr.passing,
                                "sample": 0,
                                "larger_side": pr.larger_side,
                            },
                        ),
                    ),
                )
            )
        done, pending = await asyncio.wait([t for _, t in tasks], timeout=_remaining(req.deadline_unix))
        cut_waiting_ep = job.endpoint_waiting
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        results = [(pr, t.result() if t in done and not t.cancelled() else {"status": "cancelled"}) for pr, t in tasks]
        ok = [(pr, r) for pr, r in results if r["status"] == "ok"]
        counts = {
            "pairs_total": len(results),
            "pairs_ok": len(ok),
            "pairs_capped": sum(r["status"] == "timeout" for _, r in results),
            "pairs_failed": sum(r["status"] in ("error", "no_output") for _, r in results),
            "pairs_cancelled": sum(r["status"] == "cancelled" for _, r in results),
            "pairs_salvaged": sum(r.get("opencode_finished") is False for _, r in ok),
            "pairs_cut_waiting_endpoint": cut_waiting_ep,
            "sessions": job.sessions_started,
        }
        merged = merge_group(
            meta,
            [
                {
                    "fail": pr.fail,
                    "passing": pr.passing,
                    "larger_side": pr.larger_side,
                    "verdict": parse_json(r.get("pair_json")),
                }
                for pr, r in ok
            ],
            group["outputs"],
            ov["ill_frac"],
            ov["ill_min_votes"],
            ov["vote_frac"],
            False,
            decision_ids=True,
            good_earliest=bool(ov["good_earliest"]),
        )
        info = {**st, **counts, "ill_votes": merged["ill_votes"]}
        if any(e["group_ill_posed"] for e in merged["entries"]):
            ev = merged["entries"][0].get("ill_posed_evidence")
            return self._result(
                job, "ill_posed", t0, labels=empty, ill_posed_by="pairs", ill_posed_evidence=_text(ev), **info
            )
        labels = {}
        for ent, r in zip(merged["entries"], meta["rollouts"]):
            out = group["outputs"][r["line"]]
            starts = call_starts(out)
            labels[str(order[r["R"]])] = {
                "n_calls": len(starts),
                "labels": [
                    {
                        "idx": lab["idx"],
                        "call": call_of(starts, lab["idx"]),
                        "label": lab["label"],
                        "decision_id": str(lab.get("decision") or ""),
                    }
                    for lab in ent["labels"]
                    if lab.get("label") in DECISION
                ],
                **align[str(order[r["R"]])],
            }
        status = "ok" if len(ok) == len(results) else "partial"
        return self._result(job, status, t0, labels=labels, **info)

    def _spawn(self, job: Job, coro) -> asyncio.Task:
        t = asyncio.get_running_loop().create_task(coro)
        job.sessions.add(t)
        return t

    async def _session(
        self,
        job: Job,
        kind: str,
        image: str,
        files: dict,
        instructions: str,
        timeout_s: float,
        ov: dict,
        row: dict,
        required: str = "out/pair.json",
        extra_reads: tuple = ("out/decisions.json",),
    ) -> dict:
        """One judge session under the global session cap; logs its row + export. Never raises except when cancelled
        (then the runner's finally has stopped the sandbox, or stops it once its creation returns). The judge endpoint
        is the one current when the session takes its slot; while the endpoint file is missing the session waits for
        it without holding a slot."""
        row = {**row, "kind": kind}
        t0 = time.time()
        started = False
        try:
            if not self._ready.is_set():
                await self._ready.wait()
                job.startup_wait_s = max(job.startup_wait_s, time.time() - t0)
            while True:
                await self._wait_endpoint(job)
                self.waiting_sessions += 1
                try:
                    await self.sem.acquire()
                finally:
                    self.waiting_sessions -= 1
                ep = self._endpoint()  # current at session start (the file may have changed while queued)
                if ep is not None:
                    break
                self.sem.release()  # the file vanished while we were queued: wait for it again
            try:
                started = True
                waited = round(time.time() - t0, 1)
                self.active_sessions += 1
                self.sessions_total += 1
                job.sessions_started += 1
                if job.state == "queued":
                    job.state = "running"
                s = PairRunSettings(
                    judge_base_url=ep[0],
                    judge_model=ep[1],
                    session_timeout_s=timeout_s,
                    sandbox_ttl_s=self.cfg.sandbox_ttl_s,
                    metadata=dict(self.metadata),
                )
                s.opencode.temperature = ov["temperature"]
                res = await self.run_pair(
                    self.provider, image, files, instructions, s, required=required, extra_reads=extra_reads
                )
                res["judge_base_url"], res["judge_model"] = ep
                res.setdefault("timings", {})["queue_s"] = waited
            finally:
                self.active_sessions -= 1
                self.sem.release()
        except asyncio.CancelledError:
            self._log_session(
                row,
                {
                    "status": "cancelled",
                    "error": job.cancel_reason or "deadline",
                    "session_started": started,
                    "timings": {"total_s": round(time.time() - t0, 1)},
                },
            )
            raise
        except Exception as e:  # run_pair never raises; this guards the bookkeeping around it
            res = {"status": "error", "error": f"{type(e).__name__}: {e}"[:2000], "timings": {}}
        self._log_session(row, res)
        return res

    def _log_session(self, row: dict, res: dict) -> None:
        self._log_row(
            self._plog, {**row, **{k: v for k, v in res.items() if k != "export"}, "time": time.strftime("%H:%M:%S")}
        )
        if res.get("export") is not None:
            self._submit_write(
                self._xlog,
                json.dumps(
                    {k: row.get(k) for k in ("request_id", "group", "fail", "passing", "sample")}
                    | {"export": res.get("export")}
                )
                + "\n",
            )
        LOG.info(
            "%s %s R%s/R%s: %s %s%s",
            row.get("request_id"),
            row.get("kind"),
            row.get("fail"),
            row.get("passing"),
            res.get("status"),
            res.get("timings"),
            " " + str(res.get("error"))[:160] if res.get("error") else "",
        )

    # ---------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        """Startup (listening already): reap the orphans of this run label (a previous job of the same training run
        that died without cleanup) in the background; sessions wait for it."""
        self._ready.clear()

        async def startup_reap():
            try:
                await asyncio.wait_for(self.reap("startup"), self.cfg.reap_timeout_s)
            except Exception as e:
                LOG.warning("startup reap gave up: %s: %s", type(e).__name__, e)
            finally:
                self._ready.set()

        self._startup = asyncio.get_running_loop().create_task(startup_reap())
        self._maint = asyncio.get_running_loop().create_task(self._maintenance())

    async def _maintenance(self) -> None:
        while True:
            await asyncio.sleep(300)
            cutoff = time.time() - self.cfg.keep_s
            for rid in [r for r, j in self.jobs.items() if j.state == "done" and (j.done_at or 0) < cutoff]:
                self.jobs.pop(rid, None)
            LOG.info("status: %s", json.dumps(self.health()))

    async def reap(self, when: str) -> dict:
        """Delete the sandboxes labeled with this run (role turn-judge, this user): at startup the orphans of an
        earlier job of the same run, at exit anything a cancelled session left behind."""
        if self.reaper is None:
            return {}
        try:
            out = await self.reaper(self.cfg.run_id)
            LOG.info("reap (%s) of run %s: %s", when, self.cfg.run_id, out)
            return out
        except Exception as e:
            LOG.warning("reap (%s) of run %s failed: %s: %s", when, self.cfg.run_id, type(e).__name__, e)
            return {"error": str(e)}

    async def shutdown(self) -> None:
        """Cancel everything (sandboxes stopped), reap the run's sandboxes, archive the logs once."""
        if self.closing:
            return
        self.closing = True
        if self.on_closing:
            try:
                self.on_closing()
            except Exception as e:
                LOG.warning("on_closing failed: %s: %s", type(e).__name__, e)
        LOG.info("shutdown: cancelling %d running requests", sum(1 for j in self.jobs.values() if j.state != "done"))
        for t in (self._maint, self._startup):
            if t and not t.done():
                t.cancel()
        self._ready.set()
        running = [j.task for j in self.jobs.values() if j.task and not j.task.done()]
        for j in self.jobs.values():
            if j.task and not j.task.done():
                j.cancel_reason = "service shutdown"
                j.task.cancel()
        if running:
            await asyncio.wait(running, timeout=self.cfg.shutdown_wait_s)
        bg = runner.background_stops()
        if bg:
            await asyncio.wait(bg, timeout=self.cfg.shutdown_wait_s / 2)
        try:
            await asyncio.wait_for(self.reap("exit"), self.cfg.shutdown_wait_s)
        except asyncio.TimeoutError:
            LOG.warning("exit reap timed out; leftover sandboxes expire with their TTL (%d s)", self.cfg.sandbox_ttl_s)
        await asyncio.get_running_loop().run_in_executor(None, self._close_and_archive)

    def _close_and_archive(self) -> None:
        self.log_pool.shutdown(wait=True)
        self.build_pool.shutdown(wait=False, cancel_futures=True)
        for fh in (self._plog, self._xlog, self._rlog):
            try:
                fh.close()
            except Exception:
                pass
        if not self.cfg.archive_dir:
            return
        name = self.cfg.archive_name or f"{self.cfg.run_id}.tar.gz"
        dest = Path(self.cfg.archive_dir) / name
        tmp = Path(self.cfg.archive_dir) / f".{name}.tmp"
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            for h in logging.getLogger().handlers:
                h.flush()
            with tarfile.open(tmp, "w:gz", compresslevel=1) as tar:
                tar.add(self.logs, arcname=name[: -len(".tar.gz")] if name.endswith(".tar.gz") else name)
            os.replace(tmp, dest)
            size = dest.stat().st_size
            msg = (
                f"[turn-judge service] logs archived to {dest} ({size / 2**20:.1f} MiB"
                f"{'' if self.cfg.keep_exports else '; session exports not archived (--keep-exports)'})"
            )
            print(msg, flush=True)
            with open(self.logs / "service.log", "a") as f:  # the archived copy is already closed: node-local note
                f.write(msg + "\n")
        except Exception as e:
            print(f"[turn-judge service] archive to {dest} FAILED: {type(e).__name__}: {e}", flush=True)
            return  # keep everything node-local for a manual copy
        if self.cfg.clean_local:  # the bulk (exports: GBs per job) is not needed once the job ends; logs/ stays
            for p in (
                self.cfg.work_root / "exports.jsonl.gz",
                self.ak_root,
                self.cfg.work_root / "tmp",
                *(() if self.cfg.keep_workspaces else (self.cfg.work_root / "jobs",)),
            ):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.exists():
                    p.unlink()


def _build_and_load(ws: Path, req: JudgeRequest, answer_key: Callable) -> tuple[dict, dict]:
    """Worker thread: write the workspace, then read it back exactly as offline.py does (load_group) + hedges."""
    built = build_group_workspace(ws, req, answer_key)
    group = load_group(ws)
    group_hedges(group)  # computed once here, off the event loop
    return built, group


def _text(v: Any) -> Optional[str]:
    return v if v is None or isinstance(v, str) else json.dumps(v)


def _remaining(deadline: Optional[float]) -> Optional[float]:
    return None if deadline is None else max(0.0, deadline - time.time())


# ------------------------------------------------------------------------------------------------ sandbox reaper


def sandbox_run_label(run_id: str) -> str:
    """The sandbox `run` label for a run id, as OpenSandbox will store it (Kubernetes label value: sanitized, <= 63
    chars). Long ids (tj-train-<run dir basename>) keep a readable head plus a hash of the full id, so two runs whose
    names share 63 characters (arms of one experiment) never reap each other's sandboxes."""
    from nemo_gym.sandbox.providers.opensandbox.provider import _metadata_value

    v = _metadata_value(run_id)
    if v == run_id.strip("._-") and len(v) <= 63:
        return v
    return _metadata_value(f"{v[:54]}-{hashlib.sha1(run_id.encode()).hexdigest()[:8]}")


def opensandbox_reaper(user: Optional[str]) -> Callable[..., Awaitable[dict]]:
    """List / delete turn-judge sandboxes on the OpenSandbox cell in OPENSANDBOX_DOMAIN (as judge_sandbox/sandboxes.py):
    metadata role turn-judge, this user, run == the run label."""
    import aiohttp

    async def reap(run_label: str) -> dict:
        dom = os.environ["OPENSANDBOX_DOMAIN"].strip().strip('"').rstrip("/")
        base = dom if "://" in dom else f"http://{dom}"
        headers = {"OPEN-SANDBOX-API-KEY": os.environ["OPENSANDBOX_API_KEY"].strip().strip('"')}
        mine, total, page = [], 0, 1
        async with aiohttp.ClientSession(
            headers=headers,
            connector=aiohttp.TCPConnector(ssl=False, limit=32),
            timeout=aiohttp.ClientTimeout(total=60),
        ) as s:
            while True:
                async with s.get(f"{base}/v1/sandboxes", params={"page": page, "pageSize": 100}) as r:
                    p = await r.json(content_type=None)
                for it in p.get("items", []):
                    total += 1
                    md = it.get("metadata") or {}
                    if (
                        md.get("role") == "turn-judge"
                        and md.get("nemo-gym.nvidia.com/run") == run_label
                        and (user is None or md.get("nemo-gym.nvidia.com/user") == user)
                    ):
                        mine.append(it)
                if not p.get("pagination", {}).get("hasNextPage"):
                    break
                page += 1

            async def rm(it):
                async with s.delete(f"{base}/v1/sandboxes/{it['id']}") as r:
                    await r.read()
                    return r.status

            codes = await asyncio.gather(*(rm(it) for it in mine))
        return {"listed": total, "matched": len(mine), "deleted": {c: codes.count(c) for c in set(codes)}}

    return reap


# ------------------------------------------------------------------------------------------------ HTTP app


def create_app(
    svc: JudgeService,
    on_started: Optional[Callable[[], Awaitable[None]]] = None,
    request_exit: Optional[Callable[[], None]] = None,
):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException

    @asynccontextmanager
    async def lifespan(app):
        await svc.start()
        if on_started:
            asyncio.get_running_loop().create_task(on_started())
        try:
            yield
        finally:
            await svc.shutdown()

    app = FastAPI(title="turn-judge service", lifespan=lifespan)

    @app.post("/judge_group")
    async def post_judge_group(req: JudgeRequest):
        try:
            return svc.submit(req)
        except BadRequest as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    @app.get("/judge_group/{request_id:path}")
    async def get_judge_group(request_id: str):
        body = svc.get(request_id)
        if body is None:
            raise HTTPException(status_code=404, detail=f"unknown request_id {request_id}")
        return body

    @app.delete("/judge_group/{request_id:path}")
    async def delete_judge_group(request_id: str):
        body = await svc.delete(request_id)
        if body is None:
            raise HTTPException(status_code=404, detail=f"unknown request_id {request_id}")
        return body

    @app.get("/health")
    async def health():
        return svc.health()

    @app.post("/shutdown")
    async def shutdown():
        """Graceful exit (the job wrapper calls it before the walltime and when training ends): stop accepting work,
        cancel every session (sandboxes stopped), reap the run's sandboxes, archive the logs once, exit."""
        LOG.info("POST /shutdown")
        if request_exit is None:
            await svc.shutdown()
        else:
            request_exit()  # the server stops; its lifespan shutdown runs svc.shutdown()
        return {"ok": True, "shutting_down": True, "running": sum(1 for j in svc.jobs.values() if j.state != "done")}

    return app


def _advertise_host() -> str:
    out = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
    return out[0] if out else "127.0.0.1"


def warm_up_sandbox_client() -> float:
    """Do the sandbox client's one-time work now, before the service listens; returns its seconds.

    The first sandbox start otherwise does it on the event loop: lazy imports of the OpenSandbox SDK, tenacity and
    Gym's HTTP transport (server_utils: hydra, rich, openai, from a venv on Lustre), then get_global_config_dict(),
    a full Hydra app (cached afterwards). After a resume that blocked the loop 13-136 s while the trainer's polls of
    the requests it had just submitted waited; past the client's 60 s request timeout they failed and opened its
    breaker (LOG 096). The trainer waits for the service address file, which is written only once listening.
    """
    t0 = time.time()
    from nemo_gym.global_config import get_global_config_dict
    from nemo_gym.sandbox.providers import _http_transport  # noqa: F401  (imports nemo_gym.server_utils)
    from nemo_gym.sandbox.providers.opensandbox.provider import _require_opensandbox_sdk, _require_tenacity

    _require_opensandbox_sdk()
    _require_tenacity()
    get_global_config_dict()
    return time.time() - t0


def main() -> None:
    sys.argv, argv = sys.argv[:1], sys.argv[1:]  # Gym parses sys.argv as Hydra overrides on first use
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--labeling",
        default=str(Path(__file__).with_name("LABELING_annotator.md")),
        help="the labeling rules uploaded to every session (LABELING_annotator.md)",
    )
    ap.add_argument(
        "--endpoint-file",
        "--endpoint-env",
        "--judge-endpoint-file",
        dest="endpoint_env",
        default=DEFAULT_JUDGE_ENDPOINT,
        required=DEFAULT_JUDGE_ENDPOINT is None,
        help="judge (Qwen serving) endpoint file: export JUDGE_BASE_URL=... JUDGE_MODEL_NAME=...; read at "
        "every session start; while missing, sessions wait for it until their request's deadline",
    )
    ap.add_argument("--endpoint-poll", type=float, default=15.0, help="re-check a missing endpoint file every N s")
    ap.add_argument(
        "--service-env-out",
        "--endpoint-out",
        dest="endpoint_out",
        default="",
        help="write `export TURN_JUDGE_URL=http://<host>:<port>` here (tmp + rename) once listening; "
        "removed at exit if it still names this service",
    )
    ap.add_argument("--work-root", default="", help="node-local work dir (default /tmp/tj-service/<run label>)")
    ap.add_argument("--archive-dir", default="", help="Lustre dir for the one log archive at exit")
    ap.add_argument(
        "--archive-name", default="", help="default service-<SLURM_JOB_ID>.tar.gz (else <run label>.tar.gz)"
    )
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=18300, help="0 = any free port")
    ap.add_argument(
        "--host-ip",
        "--advertise-host",
        dest="advertise_host",
        default="",
        help="host written to --service-env-out (default: first address of hostname -I)",
    )
    ap.add_argument(
        "--run-id",
        default="",
        help="sandbox run tag, shared by the jobs of one training run (startup "
        "reaps its orphans); default $NEMO_GYM_RUN_ID, else tjsvc-<date>-<pid>",
    )
    ap.add_argument("--max-sessions", type=int, default=240)
    ap.add_argument("--build-threads", type=int, default=8)
    ap.add_argument("--sandbox-ttl", type=int, default=5400, help="sandbox TTL (s): backstop for leaked sandboxes")
    ap.add_argument("--keep-hours", type=float, default=24.0, help="results kept in memory after completion")
    ap.add_argument("--keep-workspaces", action="store_true", help="keep each request's workspace (debugging)")
    ap.add_argument(
        "--keep-exports",
        action="store_true",
        help="also archive the OpenCode session exports (GBs per 4 h job); default: node-local only",
    )
    ap.add_argument(
        "--keep-local",
        action="store_true",
        help="keep the node-local exports / workspaces / caches after the exit archive",
    )
    ap.add_argument("--tb-task-root", default=workspace.TB_TASK_ROOT, help="host dir of /terminal-bench-tasks")
    ap.add_argument("--gym-sha", default=os.environ.get("TJ_GYM_SHA", ""))
    ap.add_argument("--no-reap", action="store_true", help="do not list / delete sandboxes at startup and exit")
    a = ap.parse_args(argv)

    run_id = a.run_id or os.environ.get("NEMO_GYM_RUN_ID") or f"tjsvc-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    label = sandbox_run_label(run_id)
    os.environ["NEMO_GYM_RUN_ID"] = label  # sandbox attribution (nemo_gym.sandbox.attribution.resolve_run_id)
    os.environ.setdefault("NEMO_GYM_WORKLOAD", "turn-judge-service")
    job = os.environ.get("SLURM_JOB_ID", "")
    work_root = Path(a.work_root or f"/tmp/tj-service/{label}")
    (work_root / "logs").mkdir(parents=True, exist_ok=True)
    (work_root / "tmp").mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(work_root / "tmp")  # runner's tarball staging
    tempfile_reset()
    warm_s = warm_up_sandbox_client()  # before the root logger is configured: Hydra reconfigures it
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
        handlers=[logging.FileHandler(work_root / "logs" / "service.log"), logging.StreamHandler(sys.stdout)],
    )  # force: an import configured the root logger
    logging.getLogger("nemo_gym").setLevel(logging.WARNING)
    workspace.TB_TASK_ROOT = a.tb_task_root

    import uvicorn
    from nemo_gym.sandbox.attribution import resolve_attribution

    provider, default_metadata = sandbox_provider()
    cfg = ServiceConfig(
        labeling=Path(a.labeling).read_bytes(),
        endpoint_env=a.endpoint_env,
        work_root=work_root,
        run_id=label,
        max_sessions=a.max_sessions,
        keep_s=a.keep_hours * 3600,
        keep_workspaces=a.keep_workspaces,
        keep_exports=a.keep_exports,
        clean_local=not a.keep_local,
        build_threads=a.build_threads,
        sandbox_ttl_s=a.sandbox_ttl,
        archive_dir=a.archive_dir or None,
        archive_name=a.archive_name or (f"service-{job}.tar.gz" if job else ""),
        gym_sha=a.gym_sha,
        endpoint_poll_s=a.endpoint_poll,
    )
    user = resolve_attribution().get("user")
    svc = JudgeService(cfg, provider, default_metadata, reaper=None if a.no_reap else opensandbox_reaper(user))
    LOG.info(
        "turn-judge service: run %s (sandbox label %s), user %s, slurm job %s, gym %s, labeling %s, max sessions "
        "%d, work %s, judge endpoint file %s, sandbox client warm-up %.1f s",
        run_id,
        label,
        user,
        job or "-",
        a.gym_sha,
        svc.labeling_sha[:12],
        a.max_sessions,
        work_root,
        a.endpoint_env,
        warm_s,
    )
    server: Optional[uvicorn.Server] = None
    tag = f"# turn-judge service {label} job {job or '-'} pid {os.getpid()}"

    def withdraw_url() -> None:  # as the serving chain does: remove the file only if it still names this service
        if a.endpoint_out and os.path.exists(a.endpoint_out) and tag in Path(a.endpoint_out).read_text():
            os.unlink(a.endpoint_out)
            LOG.info("removed %s", a.endpoint_out)

    svc.on_closing = withdraw_url

    async def on_started():
        while not server.started:
            await asyncio.sleep(0.2)
        port = server.servers[0].sockets[0].getsockname()[1] if a.port == 0 else a.port
        url = f"http://{a.advertise_host or _advertise_host()}:{port}"
        LOG.info("listening: %s", url)
        if a.endpoint_out:
            out = Path(a.endpoint_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
            tmp.write_text(
                f"export TURN_JUDGE_URL={url}\n{tag} started {time.strftime('%F %T')} on {os.uname().nodename}\n"
            )
            os.replace(tmp, out)
            LOG.info("wrote %s", out)

    def request_exit() -> None:
        server.should_exit = True

    app = create_app(svc, on_started, request_exit)

    class Server(uvicorn.Server):
        """Our own SIGTERM / SIGINT handling: always a graceful exit (uvicorn's second-signal force exit skips the
        lifespan shutdown, i.e. the sandbox cleanup and the log archive, and it re-raises the signal on return)."""

        def install_signal_handlers(self) -> None:  # uvicorn < 0.29
            pass

        @contextlib.contextmanager
        def capture_signals(self):  # uvicorn >= 0.29
            yield

    server = Server(
        uvicorn.Config(
            app,
            host=a.host,
            port=a.port,
            log_level="warning",
            access_log=False,
            loop="asyncio",
            lifespan="on",
            timeout_graceful_shutdown=10,
        )
    )

    async def serve() -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: (LOG.info("signal %s: shutting down", s.name), request_exit()))
        await server.serve()
        await svc.shutdown()  # no-op when the lifespan shutdown already ran
        LOG.info("exit")

    asyncio.run(serve())


def tempfile_reset() -> None:
    import tempfile

    tempfile.tempdir = None  # re-read TMPDIR (a module imported earlier may have cached /tmp)


if __name__ == "__main__":
    main()
