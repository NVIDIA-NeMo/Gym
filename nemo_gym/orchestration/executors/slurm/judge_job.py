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

"""Judge-only jobs: grade a run's saved answers again on a CPU or served-judge profile."""

import getpass
import json
import math
import os
import re
import shlex
from pathlib import Path

from nemo_gym.config.global_config import (
    MODEL_ENDPOINT_READINESS_TIMEOUT_KEY_NAME,
)
from nemo_gym.eval.recovery.decision import (
    IdentityState,
    RecoveryAction,
    RecoveryDecision,
    RecoveryOutcome,
    failed_judge_jobs_blocking,
    failed_judge_jobs_stop,
)
from nemo_gym.eval.recovery.evidence import (
    JUDGE_JOB_ACK_JOURNAL_NAME,
    JUDGE_JOB_JOURNAL_NAME,
    CollectedEvidence,
)
from nemo_gym.orchestration.api import (
    NodePool,
    SlurmComputeConfig,
    SubmitConfig,
)
from nemo_gym.orchestration.executors.connection import (
    Connection,
    get_connection,
)
from nemo_gym.orchestration.executors.git_ref import validate_gym_install_ref
from nemo_gym.orchestration.executors.slurm.judge_profile import (
    JUDGE_POOL_NAME,
    JudgeProfileUnsupported,
    choose_judge_partition,
    is_cpu_only_submission,
    judge_execution_profile,
    judge_job_config,
    served_judge_placement,
    served_judge_profile,
)
from nemo_gym.orchestration.executors.slurm.reader import (
    READER_SRUN_ENV,
)
from nemo_gym.orchestration.executors.slurm.resume_script import (
    _walltime_to_seconds,
)
from nemo_gym.orchestration.executors.slurm.scheduler import (
    _RESUBMIT_LOCK,
    _chain_live_check,
    _parse_sbatch_results,
    _partition_time_limit,
    _sbatch_command,
    _submission_exports,
)
from nemo_gym.orchestration.executors.slurm.script import (
    _with_default_capture_dir,
    build_sbatch_script,
)
from nemo_gym.orchestration.jobs import (
    RESOLVED_CONFIG_NAME,
    SubmissionRecord,
    installed_gym_commit,
    new_gym_job_id,
    utc_now,
    utc_timestamp,
)


# Settings of the run's own collection that a judge-only job's `gym eval reverify` replaces with its own.
_COLLECTION_ONLY_RUN_KEYS = (
    "limit",
    "output_jsonl_fpath",
    "materialized_inputs_jsonl_fpath",
    "resume_from_cache",
    "resubmit",
    "require_complete",
)
# Appends one judge-job journal entry (argv: the journal, the entry as JSON, the job id or empty, an optional error).
_JOURNAL_JUDGE_JOB = (
    "import json,sys; entry=json.loads(sys.argv[2]); entry['job_id']=sys.argv[3] or None; "
    "error=sys.argv[4] if len(sys.argv) > 4 else ''; entry.update({'error': error} if error else {}); "
    "f=open(sys.argv[1],'a'); f.write(json.dumps(entry, sort_keys=True)+chr(10)); f.close()"
)


def _stopped(decision: RecoveryDecision, reason: str) -> RecoveryOutcome:
    """The outcome of a judge-only job that was not queued: ``decision`` stopped with ``reason``."""
    return RecoveryOutcome(
        decision=decision.model_copy(update={"action": RecoveryAction.STOP, "reason": reason}),
        ok=False,
        dispatched=None,
        unsupported=reason,
    )


class JudgeJobs:
    """The judge-only jobs of ``SlurmExecutor``: grade a run's saved answers again, never its policy.

    Mixed into ``SlurmExecutor``, whose ``recovery_evidence``, ``_saved_config`` and ``installed_commit`` its
    methods use.
    """

    def reverify_judge_failures(
        self,
        record: SubmissionRecord,
        benchmark: str,
        decision: RecoveryDecision,
        *,
        judge_partition: str | None = None,
        judge_qos: str | None = None,
        repin: str | None = None,
        repin_reason: str | None = None,
        acknowledge_failed_judge_jobs: bool = False,
        acknowledge_reason: str | None = None,
    ) -> RecoveryOutcome:
        """Queue a judge-only job that grades the run's selected saved answers again, never the policy.

        A CPU-only submission (no service, no policy model, CPU node pools only) runs it on its own compute with the
        chain's remaining walltime allowance, as before. Any other submission, one that served its policy on GPUs,
        runs it on a judge execution profile derived from its saved config (``_queue_judge_job``): the run's own judge
        and verifier servers on one node of a CPU partition (``judge_partition``, a saved CPU pool or the launcher's
        reader placement, with ``judge_qos`` or a QOS derived for it), or, for a run that serves its judge on a node
        pool of its own, on that saved pool with those services started and the run's QOS, for the run's segment
        walltime bounded by the partition's MaxTime, at the run's pinned Gym commit or ``repin``.
        ``acknowledge_failed_judge_jobs`` (with ``acknowledge_reason``) lets one more judge-only job queue when the last
        ones ended without settling a saved answer (see ``_queue_judge_job``). Either way the job settles through
        ``gym eval reverify --judge-failed-only --settle-into-source``. This never starts a policy service or turns an
        unavailable judge into a model run.
        """
        from nemo_gym.orchestration.executors.utils import flatten_run_args

        current, evidence = self.recovery_evidence(
            record, benchmark, **({"reader_ref": repin} if repin is not None else {})
        )
        if current.action != RecoveryAction.REVERIFY or evidence is None:
            return _stopped(decision, "saved-answer rejudge admission changed; collect the current evidence")
        job = record.benchmark_job(benchmark)
        job_dir = Path(job.job_dir)
        with get_connection(record.hostname) as conn:
            config = self._saved_config(conn, Path(record.run_dir))
            original = config.driver.benchmarks[benchmark]
            resume = original.resume_config
            compute = config.compute[record.cluster]
            if not isinstance(compute, SlurmComputeConfig):
                return _stopped(decision, "saved rejudge compute is not Slurm")
            if not is_cpu_only_submission(config):
                # A run that served its policy on GPUs: a judge-only job on a derived CPU-only profile instead.
                return self._queue_judge_job(
                    conn,
                    record,
                    benchmark,
                    current,
                    evidence,
                    config,
                    judge_partition=judge_partition,
                    judge_qos=judge_qos,
                    repin=repin,
                    repin_reason=repin_reason,
                    acknowledge_failed_judge_jobs=acknowledge_failed_judge_jobs,
                    acknowledge_reason=acknowledge_reason,
                )
            install = config.driver.gym_install
            if (
                install is None
                or install.ref != installed_gym_commit()
                or not re.fullmatch(r"[0-9a-f]{40}", install.ref)
            ):
                return _stopped(
                    decision, "saved judge driver and installed recovery source must be the same immutable revision"
                )
            if resume is None or resume.max_walltime is None or evidence.accounting_snapshot is None:
                return _stopped(decision, "saved rejudge walltime or accounting owner is unknown")
            used = evidence.accounted_walltime_s
            retries = evidence.infra_retries_used
            if (
                used is None
                or not math.isfinite(used)
                or used < 0
                or retries is None
                or not math.isfinite(retries)
                or retries < 0
            ):
                return _stopped(decision, "actual chain walltime or retry accounting is unknown")
            if retries > resume.max_retries or not evidence.job_chain or evidence.chain_incomplete:
                return _stopped(decision, "recorded chain is incomplete or exceeds its original retry allowance")
            cap = _walltime_to_seconds(resume.max_walltime)
            remaining = math.floor(cap - used)
            if compute.walltime is not None:
                remaining = min(remaining, _walltime_to_seconds(compute.walltime))
            if remaining < 1:
                return _stopped(decision, "original walltime allowance is exhausted")
            if evidence.accounting_snapshot.get("resume_config") != resume.model_dump_json():
                return _stopped(decision, "saved resume configuration changed since accounting")
            # Use the original runtime/configuration. Only operational output/cache controls change.
            run = dict(original.run)
            for name in (
                "limit",
                "output_jsonl_fpath",
                "materialized_inputs_jsonl_fpath",
                "resume_from_cache",
                "resubmit",
                "require_complete",
            ):
                run.pop(name, None)
            if evidence.source_digests is None:
                return _stopped(decision, "saved source snapshot is unavailable")
            queued_snapshot = {k: v for k, v in evidence.source_digests.items() if Path(k).name == k}
            nonce = new_gym_job_id(utc_now())
            output = job_dir / "reverification" / nonce / "rollouts.jsonl"
            # Observability is on by default, and it needs a capture directory: the job's own folder, as a submitted
            # evaluation captures into its benchmark directory.
            run = _with_default_capture_dir(run, output.parent)
            command = " ".join(
                [
                    "gym eval reverify",
                    "--judge-failed-only",
                    "--settle-into-source",
                    # The dispatch below runs only when no allocation of the run is live.
                    "--withdraw-unresolved-admissions",
                    "--settle-source-snapshot",
                    shlex.quote(json.dumps(queued_snapshot, sort_keys=True)),
                    "--rollouts",
                    shlex.quote(str(job_dir / "artifacts/rollouts.jsonl")),
                    "--inputs",
                    shlex.quote(str(job_dir / "artifacts/rollouts_materialized_inputs.jsonl")),
                    "--output",
                    shlex.quote(str(output)),
                    *flatten_run_args(run),
                ]
            )
            # No preparation, fresh model run, reset prologue, or afternotok successor.
            bench = original.model_copy(update={"prepare": {}, "run": {}, "command": command, "resumable": False})
            bounded_compute = compute.model_copy(
                update={"walltime": f"{remaining // 3600:02}:{remaining % 3600 // 60:02}:{remaining % 60:02}"}
            )
            driver = config.driver.model_copy(update={"benchmarks": {benchmark: bench}})
            bound = config.model_copy(update={"driver": driver, "compute": {record.cluster: bounded_compute}})
            script = build_sbatch_script(bound, benchmark, bench, bounded_compute, job_dir)
            script = script.replace("#!/bin/bash", "#!/bin/bash\n#SBATCH --no-requeue", 1)
            script_path = job_dir / f"reverify-{nonce}.sh"
            expected = {k: v for k, v in evidence.accounting_snapshot.items() if k != "resume_config"}
            config_path = Path(record.run_dir) / RESOLVED_CONFIG_NAME
            config_digest = conn.run([f"sha256sum {shlex.quote(str(config_path))}"]).split()[0]
            if not re.fullmatch(r"[0-9a-f]{64}", config_digest):
                return _stopped(decision, "saved submission configuration digest is unknown")
            check = (
                "import hashlib,json,os,sys; from pathlib import Path\n"
                "expected=json.loads(sys.argv[1])\n"
                "def digest(n):\n"
                " p=Path(n)\n"
                " if not p.exists(): return None\n"
                " h=hashlib.sha256()\n"
                " with p.open('rb') as f:\n"
                "  for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)\n"
                " return h.hexdigest()\n"
                "names=set(expected)|{n for n in os.listdir('.') if n.startswith('.gym_followup_queued.')}\n"
                f"source=json.loads({json.dumps(json.dumps(evidence.source_digests))})\n"
                "if {n:digest('artifacts/'+n) for n in source} != source: raise ValueError('saved source changed')\n"
                f"if digest({str(config_path)!r}) != {config_digest!r}: raise ValueError('saved configuration changed')\n"
                "if {n:digest(n) for n in names} != expected: raise ValueError('stale accounting')\n"
                f"p=Path('.gym_accumulated_walltime'); total=max(float(p.read_text()) if p.exists() else 0, {used!r})\n"
                f"if not __import__('math').isfinite(total) or total < 0 or total >= {cap!r}: raise ValueError('spent walltime')\n"
                "tmp=p.with_suffix('.tmp'); tmp.write_text(str(total)+'\\n'); os.replace(tmp,p)\n"
            )
            commands = _submission_exports(config) + [
                'fail() { echo "Error: $*" >&2; exit 1; }',
                f"cd {shlex.quote(str(job_dir))} || fail 'job directory unavailable'",
                f"(set -o noclobber; : > {_RESUBMIT_LOCK}) 2>/dev/null || fail 'another recovery holds the lock'",
                "_gym_submit_started=0; _gym_submit_confirmed=0",
                "_gym_release_rejudge_lock() { "
                'if [[ "$_gym_submit_started" == 1 && "$_gym_submit_confirmed" != 1 ]]; then '
                "printf '%s\\n' 'GYM_ACCOUNTING_UNKNOWN: rejudge acknowledgement unknown' > .gym_chain_incomplete; fi; "
                f"rm -f {_RESUBMIT_LOCK}; }}",
                "trap _gym_release_rejudge_lock EXIT",
                *_chain_live_check(evidence.job_chain, "an original allocation is still live"),
                f"python3 -c {shlex.quote(check)} {shlex.quote(json.dumps(expected))} || fail 'accounting changed or exhausted'",
                "umask 077",
                f"printf %s {shlex.quote(script)} > {shlex.quote(str(script_path))} || fail 'cannot stage judge driver'",
                "_gym_submit_started=1",
                _sbatch_command(benchmark, script_path).replace("sbatch --parsable", "sbatch --no-requeue --parsable"),
                'if [ "$rc" != 0 ]; then fail "judge allocation not admitted"; fi',
                "jid=$(printf '%s' \"$out\" | awk '{print $NF}' | cut -d';' -f1); [[ $jid =~ ^[0-9]+$ ]] || { printf '%s\\n' 'GYM_ACCOUNTING_UNKNOWN: rejudge submission acknowledgement unknown' > .gym_chain_incomplete; fail 'unknown job identity'; }",
                'printf "%s\\n" "$jid" >> .gym_job_chain || fail "cannot preserve judge allocation accounting"',
                "_gym_submit_confirmed=1",
            ]
            results = _parse_sbatch_results(conn.run(commands))
            job_id, error = results.get(benchmark, (None, "no scheduler acknowledgement"))
            if job_id is None:
                return _stopped(decision, f"saved-answer rejudge was not admitted: {error}")
            updated = job.model_copy(update={"job_id": job_id, "error": None})
            self.persist_record(record.with_benchmark(updated), conn.write_text)
        return RecoveryOutcome(decision=current, ok=True, dispatched="reverify", job_id=job_id)

    def _queue_judge_job(
        self,
        conn: Connection,
        record: SubmissionRecord,
        benchmark: str,
        decision: RecoveryDecision,
        evidence: CollectedEvidence,
        config: SubmitConfig,
        *,
        judge_partition: str | None,
        judge_qos: str | None = None,
        repin: str | None,
        repin_reason: str | None,
        acknowledge_failed_judge_jobs: bool = False,
        acknowledge_reason: str | None = None,
    ) -> RecoveryOutcome:
        """Queue the judge-only job of a run that is not CPU-only, on the judge execution profile derived from its
        saved config (``judge_profile``), and journal it: the served-judge profile when the run serves services beside
        its policy (its judge on a node pool of its own), else the CPU profile.

        The job's command is the same ``gym eval reverify --judge-failed-only --settle-into-source`` as a CPU-only
        run's, with the run's saved settings, the policy's ``--model-type`` so the run's config composes as it did
        (the policy server it names is never called) and a receipt the job writes when it ends. Under the resubmit
        lock, no job of the run may wait or run in its job dir, no allocation of the chain may be live, and the saved
        sources, resolved config and chain accounting must be as the evidence saw them; then the script is staged
        beside the run's and queued with ``--no-requeue``, and its id, partition, walltime, rows and receipt path are
        appended to the job dir's judge-job journal. The job is not a segment of the policy chain: nothing is added
        to ``.gym_job_chain`` and no retry or walltime budget is spent or needed. When the last
        ``JUDGE_JOB_FAILURE_BOUND`` judge-only jobs ended without settling a saved answer, nothing is queued and the
        stop names them and the way on: ``acknowledge_failed_judge_jobs`` with a nonempty ``acknowledge_reason`` appends
        one line (time, user, reason, those jobs' ids) to the acknowledgement journal once the new job is accepted by the
        scheduler, so exactly one more job queues; the acknowledgement covers only the jobs it names.
        """
        from nemo_gym.orchestration.executors.utils import flatten_run_args

        job = record.benchmark_job(benchmark)
        job_dir = Path(job.job_dir)
        original = config.driver.benchmarks[benchmark]
        blocked = failed_judge_jobs_blocking(evidence.judge_jobs, evidence.judge_job_acknowledgements)
        acknowledgement = None
        if blocked is not None:
            if not acknowledge_failed_judge_jobs or None in blocked:
                return _stopped(decision, failed_judge_jobs_stop(blocked, str(job_dir / "reverification")))
            if not (acknowledge_reason or "").strip():
                raise ValueError("--acknowledge-failed-judge-jobs needs a nonempty --acknowledge-reason")
            acknowledgement = {
                "at": utc_timestamp(utc_now()),
                "user": getpass.getuser(),
                "reason": (acknowledge_reason or "").strip(),
                "job_ids": blocked,
                "gym_job_id": record.gym_job_id,
                "benchmark": benchmark,
            }
        try:
            # A run that serves its judge on a node pool of its own takes that pool; any other, a CPU partition.
            placement = served_judge_placement(config, record.cluster)
            if placement is None:
                partition, source = choose_judge_partition(
                    config, record.cluster, partition=judge_partition, reader_srun=os.environ.get(READER_SRUN_ENV)
                )
        except JudgeProfileUnsupported as error:
            return _stopped(decision, str(error))
        install = config.driver.gym_install
        ref = repin if repin is not None else (install.ref if install is not None else None)
        if install is None or ref is None or ref != installed_gym_commit() or not re.fullmatch(r"[0-9a-f]{40}", ref):
            return _stopped(
                decision,
                "saved judge driver and installed recovery source must be the same immutable revision"
                + (
                    ""
                    if repin is not None
                    else "; --repin <this launcher's full commit> --repin-reason TEXT runs the judge-only job at it"
                ),
            )
        if repin is not None and not (repin_reason or "").strip():
            raise ValueError("--repin needs a nonempty --repin-reason")
        compute = config.compute[record.cluster]
        if placement is not None:
            probe = compute.model_copy(update={"node_pools": placement[1]})
            profile = served_judge_profile(
                config, record.cluster, _partition_time_limit(conn, probe), partition=judge_partition, qos=judge_qos
            )
        else:
            probe = compute.model_copy(update={"node_pools": {JUDGE_POOL_NAME: NodePool(partition=partition)}})
            profile = judge_execution_profile(
                config,
                record.cluster,
                partition,
                source,
                _partition_time_limit(conn, probe),
                qos=judge_qos,
                reader_srun=os.environ.get(READER_SRUN_ENV),
            )
        if evidence.source_digests is None or evidence.accounting_snapshot is None:
            return _stopped(decision, "saved source or accounting snapshot is unavailable; collect the evidence again")
        queued_snapshot = {k: v for k, v in evidence.source_digests.items() if Path(k).name == k}
        run = {k: v for k, v in original.run.items() if k not in _COLLECTION_ONLY_RUN_KEYS}
        nonce = new_gym_job_id(utc_now())
        folder = job_dir / "reverification" / nonce
        output, receipt = folder / "rollouts.jsonl", folder / "receipt.json"
        # Observability is on by default, and it needs a capture directory: the job's own folder, as a submitted
        # evaluation captures into its benchmark directory.
        run = _with_default_capture_dir(run, folder)
        # The job only judges saved answers: it never calls the policy model, whose served endpoint is gone with the
        # run's allocation, so waiting for that endpoint's readiness would only time the job out.
        run = {key: value for key, value in run.items() if key != MODEL_ENDPOINT_READINESS_TIMEOUT_KEY_NAME}
        policy_type = config.driver.policy_model_type
        command = " ".join(
            [
                "gym eval reverify",
                "--judge-failed-only",
                "--settle-into-source",
                # The judge-only job is queued only when no job of the run is live, so an admission a pass that died
                # left without an outcome is withdrawn and judged again instead of refusing the job.
                "--withdraw-unresolved-admissions",
                "--settle-source-snapshot",
                shlex.quote(json.dumps(queued_snapshot, sort_keys=True)),
                "--rollouts",
                shlex.quote(str(job_dir / "artifacts/rollouts.jsonl")),
                "--inputs",
                shlex.quote(str(job_dir / "artifacts/rollouts_materialized_inputs.jsonl")),
                "--output",
                shlex.quote(str(output)),
                "--settle-receipt",
                shlex.quote(str(receipt)),
                *([f"--model-type {shlex.quote(policy_type)}"] if config.driver.policy_model and policy_type else []),
                # A judge failure whose saved attempts are spent keeps its masked row; the rest are judged.
                "+skip_spent_judge_failures=True",
                *flatten_run_args(run),
                f"++{MODEL_ENDPOINT_READINESS_TIMEOUT_KEY_NAME}=0",
            ]
        )
        bound = judge_job_config(config, record.cluster, benchmark, profile, command, gym_ref=repin)
        if repin is not None:
            validate_gym_install_ref(bound)
        bench = bound.driver.benchmarks[benchmark]
        script = build_sbatch_script(bound, benchmark, bench, bound.compute[record.cluster], job_dir)
        script = script.replace("#!/bin/bash", "#!/bin/bash\n#SBATCH --no-requeue", 1)
        script_path = job_dir / f"reverify-{nonce}.sh"
        config_path = Path(record.run_dir) / RESOLVED_CONFIG_NAME
        config_digest = conn.run([f"sha256sum {shlex.quote(str(config_path))}"]).split()[0]
        if not re.fullmatch(r"[0-9a-f]{64}", config_digest):
            return _stopped(decision, "saved submission configuration digest is unknown")
        expected = {k: v for k, v in evidence.accounting_snapshot.items() if k != "resume_config"}
        check = (
            "import hashlib,json,os,sys; from pathlib import Path\n"
            "expected=json.loads(sys.argv[1])\n"
            "def digest(n):\n"
            " p=Path(n)\n"
            " if not p.exists(): return None\n"
            " h=hashlib.sha256()\n"
            " with p.open('rb') as f:\n"
            "  for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)\n"
            " return h.hexdigest()\n"
            "names=set(expected)|{n for n in os.listdir('.') if n.startswith('.gym_followup_queued.')}\n"
            f"source=json.loads({json.dumps(json.dumps(evidence.source_digests))})\n"
            "if {n:digest('artifacts/'+n) for n in source} != source: raise ValueError('saved source changed')\n"
            f"if digest({str(config_path)!r}) != {config_digest!r}: raise ValueError('saved configuration changed')\n"
            "if {n:digest(n) for n in names} != expected: raise ValueError('stale accounting')\n"
        )
        entry = {
            "at": utc_timestamp(utc_now()),
            "job_id": None,
            "nonce": nonce,
            # Its kind, partition, pools, services, QOS and walltime, each with where it came from.
            **profile.report(),
            "rows": decision.counts.get(IdentityState.JUDGE_RECOVERABLE.value, 0),
            "output": str(output),
            "receipt": str(receipt),
            "gym_commit": ref,
            "actor": getpass.getuser(),
            **({"repin_reason": repin_reason} if repin is not None else {}),
        }
        # Only once the scheduler has accepted the job: an acknowledgement of a job that was not queued is never written.
        acknowledge = (
            f"printf '%s\\n' {shlex.quote(json.dumps(acknowledgement, sort_keys=True))} "
            f">> {shlex.quote(JUDGE_JOB_ACK_JOURNAL_NAME)} || fail 'cannot journal the acknowledgement'; "
            if acknowledgement is not None
            else ""
        )
        journal = (
            f"python3 -c {shlex.quote(_JOURNAL_JUDGE_JOB)} {shlex.quote(JUDGE_JOB_JOURNAL_NAME)} "
            f"{shlex.quote(json.dumps(entry, sort_keys=True))}"
        )
        commands = _submission_exports(config) + [
            'fail() { echo "Error: $*" >&2; exit 1; }',
            f"cd {shlex.quote(str(job_dir))} || fail 'job directory unavailable'",
            f"(set -o noclobber; : > {_RESUBMIT_LOCK}) 2>/dev/null || fail 'another recovery holds the lock'",
            f"trap 'rm -f {_RESUBMIT_LOCK}' EXIT",
            # No job of this run, a chain segment or an earlier judge-only job, may wait or run in its job dir.
            "workdirs=$(squeue --me --all --noheader --format=%Z) || fail 'scheduler unknown'",
            'for d in $workdirs; do [ "$d" != "$PWD" ] || '
            'fail "a job of this run is still pending or running in $PWD"; done',
            *(
                _chain_live_check(evidence.job_chain, "an original allocation is still live")
                if evidence.job_chain
                else []
            ),
            f"python3 -c {shlex.quote(check)} {shlex.quote(json.dumps(expected))} "
            "|| fail 'saved sources or accounting changed since the evidence was collected; collect it again'",
            "umask 077",
            f"printf %s {shlex.quote(script)} > {shlex.quote(str(script_path))} || fail 'cannot stage the judge driver'",
            _sbatch_command(benchmark, script_path).replace("sbatch --parsable", "sbatch --no-requeue --parsable"),
            'if [ "$rc" != 0 ]; then fail "judge allocation not admitted"; fi',
            "jid=$(printf '%s' \"$out\" | awk '{print $NF}' | cut -d';' -f1)",
            # The queued job is journaled by id; an acknowledgement without one is journaled as such, so inspect
            # reports it and the next recovery's scheduler check finds the job by its working directory.
            f'if [[ $jid =~ ^[0-9]+$ ]]; then {acknowledge}{journal} "$jid" || fail "cannot journal the judge-only job"; '
            f'else {journal} "" "acknowledgement unknown: $out"; fail "unknown job identity"; fi',
        ]
        try:
            results = _parse_sbatch_results(conn.run(commands))
        except (RuntimeError, OSError) as error:
            text = str(error).strip()
            return _stopped(
                decision, f"judge-only job was not queued: {text.splitlines()[-1] if text else type(error).__name__}"
            )
        job_id, error = results.get(benchmark, (None, "no scheduler acknowledgement"))
        if job_id is None:
            return _stopped(decision, f"judge-only job was not admitted: {error}")
        updated = job.model_copy(update={"job_id": job_id, "error": None})
        self.persist_record(record.with_benchmark(updated), conn.write_text)
        return RecoveryOutcome(
            decision=decision, ok=True, dispatched="reverify", job_id=job_id, judge_job={**entry, "job_id": job_id}
        )
