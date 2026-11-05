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
"""Whether a chain continues: the admission a terminal allocation runs in its installed runtime, and the
accounting of a chain's allocations, including requeued ones.
"""

import os
from pathlib import Path

from nemo_gym.eval.recovery.decision import (
    ALLOCATION_ENDED_STATES,
    IdentityState,
    RecoveryAction,
)
from nemo_gym.eval.recovery.evidence import (
    COLLECT_SCRIPT,
    STORE_SKIP_KEYS,
    CollectedEvidence,
    _managed_source_digest,
    select_managed_evidence,
)
from nemo_gym.eval.recovery.markers import (
    ACCOUNTED_WALLTIME_PREFIX,
    CHAIN_ACCOUNTING_ENV,
    CONTINUATION_REASON_PREFIX,
    PROVISIONAL_MARKER_PREFIX,
    REQUEUE_RECONCILE_STATE,
    REQUEUE_UNKNOWN_REASON,
)
from nemo_gym.eval.recovery.planner import (
    decide_from_collected,
)
from nemo_gym.eval.recovery.policy import (
    effective_outcome_policy,
)


def continuation_admission(
    job_dir: str,
    max_retries: int,
    max_walltime_s: float | None,
    exit_code: str,
    previous_job_id: str,
    previous_state: str = "FAILED",
    outcome_policy: str | None = None,
) -> str:
    """Read saved state in the driver's installed runtime before a terminal allocation continues.

    The shell has already validated and persisted terminal elapsed accounting. This reader never changes
    artifacts, starts servers, resets budgets, or treats an exit code as sufficient retry evidence.
    ``outcome_policy`` is the run's ``recovery_outcome_policy`` section as JSON, rendered from its saved config, so
    an automatic continuation decides with the policy a recovery would apply.
    """
    import json
    import subprocess
    import sys

    job = Path(job_dir)
    if previous_state == REQUEUE_RECONCILE_STATE:
        return _requeue_continuation(job, max_retries, max_walltime_s)
    if exit_code == "76:0":
        print(
            f"{CONTINUATION_REASON_PREFIX}the previous driver exited 76: none of its saved failures is eligible for "
            "another attempt; `gym eval resubmit --recover --inspect` names each identity's state"
        )
        return "stop"
    try:
        raw = subprocess.run(
            [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(job)],
            capture_output=True,
            text=True,
            check=True,
        )
        collected = CollectedEvidence.model_validate_json(raw.stdout)
        if not collected.managed_store:
            print(f"{CONTINUATION_REASON_PREFIX}the saved artifacts have no rollout store manifest to continue from")
            return "stop"
        collected = select_managed_evidence(collected, job)
        # The marker holds this allocation's own provisional line until it queues its follow-up. It says nothing about
        # the run: quoted into a stop, it would read as this allocation dying before its guard, so its sentinel would
        # run the same stop again.
        own = f"{PROVISIONAL_MARKER_PREFIX}{os.environ.get('SLURM_JOB_ID', '')} started"
        if os.environ.get("SLURM_JOB_ID") and (collected.chain_continuation_note or "").startswith(own):
            collected.chain_continuation_note = None
        if collected.walltime_state != "ok":
            print(f"{CONTINUATION_REASON_PREFIX}the chain's accumulated walltime is {collected.walltime_state}")
            return "stop"
        collected.accounted_walltime_s = collected.walltime_used_s
        # A completed outcome plus untouched pending inputs establishes planned segment work. Actual failed
        # attempts still consume infra retries, even if the driver happened to exit 75. A missing output,
        # reservation, judge answer or unowned selected cause is rejected by the ordinary decision below.
        # A scheduler allocation timeout continues untouched work, but never a failed agent attempt.
        segment = previous_state == "TIMEOUT"
        if previous_state == "FAILED" and exit_code == "75:0" and collected.managed_selection is not None:
            from nemo_gym.rollouts import RolloutStore

            report_path = job / "artifacts" / "rollouts_coverage.json"
            if collected.source_digests and collected.source_digests.get(
                "rollouts_coverage.json"
            ) == _managed_source_digest(report_path):
                report = json.loads(report_path.read_text())
                store = RolloutStore.read(
                    job / "artifacts" / "rollouts.jsonl", import_legacy=False, skip_keys=STORE_SKIP_KEYS
                )
                progress = report.get("progress_outcomes_recorded")
                segment = (
                    store is not None
                    and report.get("allocation_id") == previous_job_id
                    and bool(previous_job_id)
                    and type(progress) is int
                    and progress > 0
                    and type(report.get("outcomes_recorded")) is int
                    and progress <= report["outcomes_recorded"]
                    and all(report.get(key) == value for key, value in store.coverage().items())
                )
        # The selection above verified the snapshot and read the Store once; a second read only hashes it again.
        if collected.managed_selection is None and collected.managed_reader_unavailable:
            print(
                f"{CONTINUATION_REASON_PREFIX}the Store reader could not run, so nothing was decided: "
                f"{collected.managed_reader_error or 'no cause reported'}; the saved artifacts were not judged: "
                "`gym eval resubmit --recover --inspect` reads them again"
            )
            return "unknown"
        if collected.managed_selection is None:
            print(
                f"{CONTINUATION_REASON_PREFIX}the Store could not verify the saved artifacts: "
                f"{collected.managed_reader_error or 'no authoritative selection'}"
            )
            return "stop"
        policy, source = effective_outcome_policy(None, json.loads(outcome_policy) if outcome_policy else None)
        decision = decide_from_collected(
            collected,
            max_infra_retries=max_retries,
            max_walltime_s=max_walltime_s,
            live_job=False,
            outcome_policy=policy,
            outcome_policy_source=source,
            planned_missing_continuation=segment,
            allocation_cut=_store_writer_ended(collected, previous_job_id),
        )
        print(CONTINUATION_REASON_PREFIX + decision.reason)
        if decision.action != RecoveryAction.RESUBMIT:
            return "stop"
        return "segment" if segment and not decision.counts.get(IdentityState.RETRYABLE) else "retry"
    except (OSError, ValueError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"{CONTINUATION_REASON_PREFIX}saved recovery evidence is unreadable: {type(error).__name__}: {error}")
        return "stop"


def _requeue_continuation(job: Path, max_retries: int, max_walltime_s: float | None) -> str:
    """In a sentinel: reconcile a requeued chain from the terminal records in ``CHAIN_ACCOUNTING_ENV`` and decide.

    The same reconcile and decision as ``gym eval resubmit --recover``: nothing else of the chain runs or waits (the
    sentinel checks the queue), and every allocation ended. Prints the reconciled walltime when the run continues.
    """
    import subprocess
    import sys

    try:
        rows = chain_accounting(os.environ.get(CHAIN_ACCOUNTING_ENV, ""))
        raw = subprocess.run(
            [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(job)], capture_output=True, text=True, check=True
        )
        collected = CollectedEvidence.model_validate_json(raw.stdout)
        if collected.accounting_unknown_reason != REQUEUE_UNKNOWN_REASON:
            print(f"{CONTINUATION_REASON_PREFIX}the chain's accounting is not the requeue marker")
            return "stop"
        if not rows:
            print(f"{CONTINUATION_REASON_PREFIX}the requeued chain's scheduler records are malformed")
            return "stop"
        if not collected.managed_store:
            print(f"{CONTINUATION_REASON_PREFIX}the saved artifacts have no rollout store manifest to continue from")
            return "stop"
        collected = select_managed_evidence(collected, job)
        if collected.managed_selection is None:
            print(
                f"{CONTINUATION_REASON_PREFIX}the Store could not verify the saved artifacts: "
                f"{collected.managed_reader_error or 'no authoritative selection'}"
            )
            return "stop"
        collected = reconcile_requeued_chain(collected, rows)
        if not collected.accounting_reconciled or collected.accounted_walltime_s is None:
            print(f"{CONTINUATION_REASON_PREFIX}the terminal scheduler records do not account for the requeued chain")
            return "stop"
        decision = decide_from_collected(
            collected,
            max_infra_retries=max_retries,
            max_walltime_s=max_walltime_s,
            live_job=False,
            allocation_cut=allocation_cut_from_accounting(rows, collected.coverage_allocation_id),
        )
        requeued = requeue_note(collected.job_chain, rows)
        print(CONTINUATION_REASON_PREFIX + decision.reason + (f"; {requeued}" if requeued else ""))
        if decision.action != RecoveryAction.RESUBMIT:
            return "stop"
        print(f"{ACCOUNTED_WALLTIME_PREFIX}{int(collected.accounted_walltime_s)}")
        return "retry"
    except (OSError, ValueError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"{CONTINUATION_REASON_PREFIX}saved recovery evidence is unreadable: {type(error).__name__}: {error}")
        return "stop"


def chain_accounting(accounting: str) -> list[tuple[str, str, int, int]] | None:
    """Rows of ``sacct -D -X -n -P --format=CHAIN_ACCOUNTING_FORMAT`` as (job id, state, elapsed s, restarts).

    The state keeps its first word without a trailing ``+`` ("CANCELLED by 7" is CANCELLED). None when any row is
    malformed.
    """
    rows = []
    for line in accounting.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 4 or not parts[0] or not parts[2].isdecimal() or not parts[3].isdecimal():
            return None
        state = parts[1].split()[0].rstrip("+") if parts[1].split() else ""
        rows.append((parts[0], state, int(parts[2]), int(parts[3])))
    return rows


FOLLOWUP_ACCOUNTING_FORMAT = "JobIDRaw,State,ElapsedRaw,ExitCode"
# Terminal states of a follow-up that started; a follow-up its dependency cancelled never starts (no elapsed time).
_FOLLOWUP_ENDED_STATES = frozenset(
    {"FAILED", "CANCELLED", "NODE_FAIL", "BOOT_FAIL", "PREEMPTED", "OUT_OF_MEMORY", "TIMEOUT", "DEADLINE"}
)


def stopped_followup(job_id: str, accounting: str) -> str | None:
    """Why the chain stopped when the follow-up ``job_id`` that its last allocation claimed is not in the chain.

    ``accounting`` is ``sacct -D -X -n -P --format=FOLLOWUP_ACCOUNTING_FORMAT -j job_id``. The prologue records an
    allocation in the chain before anything else, so a follow-up that started and ended without that entry ran no
    work. None when the records do not show that: the follow-up never started, is live, or its record is ambiguous.
    """
    rows = [line.strip().split("|") for line in accounting.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 4 or rows[0][0] != job_id or not rows[0][2].isdecimal():
        return None
    _, state, elapsed, exit_code = rows[0]
    state = state.split()[0].rstrip("+") if state.split() else ""
    if state not in _FOLLOWUP_ENDED_STATES or int(elapsed) == 0:
        return None
    code, _, signal = exit_code.partition(":")
    signaled = signal not in ("", "0")
    detail = f" with signal {signal}" if signaled else f" with exit code {code}" if code != "0" else ""
    # Only the scheduler's own state names the infrastructure; the record does not say who sent a signal or why.
    cause = " on infrastructure" if state in {"NODE_FAIL", "BOOT_FAIL", "PREEMPTED"} else ""
    unknown = " (cause not recorded)" if signaled and not cause else ""
    return (
        f"the chain stopped{cause}: its follow-up {job_id} ended {state}{detail} after {elapsed} s{unknown}, "
        "before it recorded itself in the chain, so no job continued the run"
    )


def allocation_cut_from_accounting(rows: list[tuple[str, str, int, int]], allocation_id: str | None) -> bool:
    """Whether the first incarnation of ``allocation_id`` has ended, by any path (``ALLOCATION_ENDED_STATES``).

    Only the first incarnation (no restarts) runs work: the prologue stops a requeued one before any. A state that is
    absent, ambiguous or not terminal is not an ending.
    """
    states = {state for job_id, state, _, restarts in rows if job_id == allocation_id and restarts == 0}
    return allocation_id is not None and len(states) == 1 and states <= ALLOCATION_ENDED_STATES


def _store_writer_ended(collected: CollectedEvidence, previous_job_id: str) -> bool:
    """In the in-job guard: whether the allocation that last wrote the Store has ended.

    The guard runs after its predecessor ended, and a chain runs one allocation at a time: every allocation the chain
    recorded before this one has ended. The writer is often an earlier allocation than the predecessor, for example
    when the predecessor timed out before it opened the Store. A writer the chain does not name stays unknown.
    """
    writer = collected.coverage_allocation_id
    if not writer or not previous_job_id:
        return False
    return writer == previous_job_id or (writer in collected.job_chain and writer != os.environ.get("SLURM_JOB_ID"))


def accounted_requeued_chain_walltime(
    job_chain: list[str], rows: list[tuple[str, str, int, int]], *, stored_walltime_s: float = 0
) -> float | None:
    """Elapsed seconds of every incarnation of a chain with a requeued allocation, when the records are complete.

    Every chain job needs terminal records numbered 0..n by restarts, and some job a requeue (n > 0). The stored
    counter must equal the elapsed time of every incarnation of the jobs before the last requeued one: an earlier
    requeue was reconciled and stored before the chain went on, and the last requeued incarnation stops before it
    adds anything, as does every later allocation. Otherwise the budget epoch is unknown.
    """
    terminal = {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "BOOT_FAIL",
        "DEADLINE",
        "PREEMPTED",
    }
    if not job_chain or len(set(job_chain)) != len(job_chain) or {row[0] for row in rows} != set(job_chain):
        return None
    restarts: dict[str, list[int]] = {}
    for job_id, state, _, restart in rows:
        if state not in terminal:
            return None
        restarts.setdefault(job_id, []).append(restart)
    if any(sorted(numbers) != list(range(len(numbers))) for numbers in restarts.values()):
        return None
    requeued = [index for index, job_id in enumerate(job_chain) if len(restarts[job_id]) > 1]
    if not requeued:
        return None
    before = set(job_chain[: requeued[-1]])
    if stored_walltime_s != sum(elapsed for job_id, _, elapsed, _ in rows if job_id in before):
        return None
    return float(sum(elapsed for _, _, elapsed, _ in rows))


def reconcile_requeued_chain(collected: CollectedEvidence, rows: list[tuple[str, str, int, int]]) -> CollectedEvidence:
    """Know a requeued chain's budget again once every incarnation has a terminal scheduler record.

    Only the requeue marker is reconciled: the counters themselves stayed readable and the missing fact is the elapsed
    time of the requeued incarnations, which complete terminal records give. Any other unknown reason stays unknown.
    """
    states = collected.counter_states or {}
    if (
        collected.accounting_unknown_reason != REQUEUE_UNKNOWN_REASON
        or states.get("infra_retries") not in ("absent", "ok")
        or states.get("walltime") not in ("absent", "ok")
    ):
        return collected
    total = accounted_requeued_chain_walltime(
        collected.job_chain, rows, stored_walltime_s=collected.walltime_used_s or 0
    )
    if total is None:
        return collected
    return collected.model_copy(
        update={
            "infra_retries_state": states["infra_retries"],
            "walltime_state": states["walltime"],
            "accounted_walltime_s": total,
            "accounting_reconciled": True,
        }
    )


def requeue_note(job_chain: list[str], rows: list[tuple[str, str, int, int]]) -> str | None:
    """The last requeued job of a chain and what its allocations account for, as a decision states it."""
    requeued = [job_id for job_id in job_chain if sum(row[0] == job_id for row in rows) > 1]
    if not requeued:
        return None
    incarnations = sorted((row for row in rows if row[0] == requeued[-1]), key=lambda row: row[3])
    cut = "was preempted" if incarnations[0][1] == "PREEMPTED" else f"ended {incarnations[0][1]}"
    return (
        f"job {requeued[-1]} {cut} and requeued; its {len(incarnations)} allocations account for "
        f"{sum(row[2] for row in incarnations)} s"
    )


def accounted_chain_walltime(
    job_chain: list[str],
    accounting: str,
    *,
    stored_walltime_s: float | None = 0,
) -> float | None:
    """Elapsed seconds of all recorded allocations, only when each has one terminal accounting row.

    ``stored_walltime_s`` None means no allocation stored a walltime: the prologue that accounts the previous
    allocation stopped first (after an operator's cancel, for one). The records' sum is the used walltime then.
    """
    terminal = {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "BOOT_FAIL",
        "DEADLINE",
        "PREEMPTED",
    }
    if not job_chain or len(set(job_chain)) != len(job_chain):
        return None
    elapsed: dict[str, int] = {}
    states: dict[str, str] = {}
    for line in accounting.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            return None
        job_id, state, seconds = parts
        state = state.split()[0].rstrip("+") if state.split() else ""
        if job_id not in job_chain or job_id in elapsed or state not in terminal:
            return None
        if not seconds.isdecimal():
            return None
        elapsed[job_id] = int(seconds)
        states[job_id] = state
    if set(elapsed) != set(job_chain):
        return None
    total = sum(elapsed.values())
    # The prologue records predecessors. A stopped guard may already have accounted for the last job too.
    # After an operator cancels a segment, the next prologue stops at the cancel before recording it, so neither
    # the cancelled segment nor the one that stopped there is stored. A segment before the cancel must have stored
    # its time, so a chain that stored nothing still has to match exactly.
    # Other values cannot identify the budget epoch (e.g. an intentional manual reset in an old chain).
    accepted = {total - elapsed[job_chain[-1]], total}
    if len(job_chain) >= 3 and states[job_chain[-2]] == "CANCELLED":
        accepted.add(total - elapsed[job_chain[-1]] - elapsed[job_chain[-2]])
    if stored_walltime_s is not None and stored_walltime_s not in accepted:
        return None
    return float(total)
