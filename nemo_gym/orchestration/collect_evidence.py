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
"""The evidence collector of `gym eval resubmit --recover`: one self-contained Python script, run on the run's host
over the executor's connection (or locally), that reads the job directory and prints one JSON object.

Decisions live in ``nemo_gym.orchestration.recovery``; this module holds the script and the constants it embeds.
"""

from nemo_gym.core.failure_kinds import (
    ENVIRONMENT_PROTOCOL_VIOLATION,
    JUDGE_SKIPPED,
    PROVIDER_CREDENTIALS_REJECTED,
    PROVIDER_QUOTA_EXHAUSTED,
    QUOTA_RESET_REGEX,
)
from nemo_gym.data.task_materialization import TASK_ID_FIELDS
from nemo_gym.orchestration.executors.resume_script import (
    ACCOUNTING_UNKNOWN_PREFIX,
    CANCELLED_FOLLOWUP_MARKER,
    FOLLOW_UP_SUBMISSION_REFUSED,
    PROVISIONAL_MARKER_PREFIX,
)


# The reason an environment server records when its own episode deadline ends an episode; no kind goes with it.
EPISODE_DEADLINE_REASON = "Episode timed out"
# Why COLLECT_SCRIPT reads the accounting as unknown when a follow-up claim, not the marker, is the cause.
UNOBSERVED_CLAIM_REASON = "a follow-up submission claim is unobserved"
UNREADABLE_CLAIM_REASON = "the chain marker or a follow-up claim is unreadable"
# Appended in the job dir when an operator continuation is queued: who, why, the budget, every value it wrote with its
# previous state, and the epoch (chain length and walltime baseline) a later recovery accounts the chain from.
OPERATOR_CONTINUATION_JOURNAL_NAME = "operator-continuation-journal.jsonl"
# Appended in the job dir when a judge-only job is queued to grade the run's saved answers again (see
# ``SlurmExecutor.reverify_judge_failures``): its id, partition, walltime, the rows it re-judges and its receipt path.
# Such a job is not a segment of the policy chain: it is not recorded in the chain and spends none of its budget.
JUDGE_JOB_JOURNAL_NAME = "judge-job-journal.jsonl"
# Appended in the job dir when an operator acknowledges that bound (``--acknowledge-failed-judge-jobs``): the time, user and
# reason, and the ids of the failed judge-only jobs it covers. It is a sibling of the judge-job journal, whose readers take
# every line for a job. It lets one more judge-only job queue, and covers only the failed jobs it names.
JUDGE_JOB_ACK_JOURNAL_NAME = "judge-job-acknowledgement-journal.jsonl"


# Run on the cluster over the executor's SSH connection: prints ONE JSON object with identity, failure and cause
# fields only (never answers or prompts), and the job chain's own budget files. argv[1] is the job dir.
COLLECT_SCRIPT = (
    "ACCOUNTING_UNKNOWN_PREFIX = "
    + repr(ACCOUNTING_UNKNOWN_PREFIX)
    + "\n"
    + "PROVISIONAL_MARKER_PREFIX = "
    + repr(PROVISIONAL_MARKER_PREFIX)
    + "\n"
    + "CANCELLED_FOLLOWUP_MARKER = "
    + repr(CANCELLED_FOLLOWUP_MARKER)
    + "\n"
    + "PROVIDER_CREDENTIALS_REJECTED = "
    + repr(PROVIDER_CREDENTIALS_REJECTED)
    + "\n"
    + "ENVIRONMENT_PROTOCOL_VIOLATION = "
    + repr(ENVIRONMENT_PROTOCOL_VIOLATION)
    + "\n"
    + "JUDGE_SKIPPED = "
    + repr(JUDGE_SKIPPED)
    + "\n"
    + "PROVIDER_QUOTA_EXHAUSTED = "
    + repr(PROVIDER_QUOTA_EXHAUSTED)
    + "\n"
    + "QUOTA_RESET_REGEX = "
    + repr(QUOTA_RESET_REGEX)
    + "\n"
    + "FOLLOW_UP_SUBMISSION_REFUSED = "
    + repr(FOLLOW_UP_SUBMISSION_REFUSED)
    + "\n"
    + "TASK_ID_FIELDS = "
    + repr(TASK_ID_FIELDS)
    + "\n"
    + f"UNOBSERVED_CLAIM_REASON = {UNOBSERVED_CLAIM_REASON!r}\n"
    + f"EPISODE_DEADLINE_REASON = {EPISODE_DEADLINE_REASON!r}\n"
    + f"UNREADABLE_CLAIM_REASON = {UNREADABLE_CLAIM_REASON!r}\n"
    + f"OPERATOR_CONTINUATION_JOURNAL_NAME = {OPERATOR_CONTINUATION_JOURNAL_NAME!r}\n"
    + f"JUDGE_JOB_JOURNAL_NAME = {JUDGE_JOB_JOURNAL_NAME!r}\n"
    + f"JUDGE_JOB_ACK_JOURNAL_NAME = {JUDGE_JOB_ACK_JOURNAL_NAME!r}\n"
    + r"""
import hashlib, json, math, os, re, sys
d = sys.argv[1]
def job_identity():
    try:
        stat = os.stat(d)
        return [os.path.realpath(d), stat.st_dev, stat.st_ino]
    except OSError:
        return None
initial_job_identity = job_identity()
a = os.path.join(d, "artifacts")
KEEP = ("_ng_task_index", "_ng_rollout_index", "_ng_rollout_id", "_ng_attempt_index", "_ng_failure_class", "failure_kind",
        "_ng_failure_terminal", "_ng_failure_counts_as_zero", "_ng_failure_max_attempts", "_ng_failure_id", "_ng_failure_stage",
        "_ng_failure_events", "_ng_failure_events_omitted", "_ng_omitted_rollout", "_ng_unrouted_run_error", "_ng_dispatch_drained", "_line", "reward",
        "mask_sample", "evaluation_completed", "_ng_failure_http_status", "accepted_stage_omission", "_ng_elapsed_s",
        "ng_policy_model_server", "_ng_rejudge_operation_id", "failure_outcome_action", "failure_outcome_attribution")
# Only structural response facts leave the run host; response contents stay local.
# A rollout store's attempt manifest, its attempt journal, or a run-tagged row: only the store can select those
# attempts. Only their presence leaves the run host, never their content or the run id.
managed_store = [os.path.exists(os.path.join(a, n)) for n in ("rollouts_manifest.json", "rollouts_attempts.jsonl")]
def trial_refs(items):
    if not isinstance(items, list):
        return None
    return [{"model_call_id": item.get("model_call_id"), "response_id": item.get("response_id"),
             "model_ref": {k: item["model_ref"].get(k) for k in ("type", "name")} if isinstance(item.get("model_ref"), dict) else None}
            if isinstance(item, dict) else None for item in items]
def model_call_facts(r):
    # Each captured model call's server-fault facts and timing only, in rollout_recovery.captured_model_calls's shape.
    capture = r.get("ng_model_call_capture")
    raw = capture.get("calls") if isinstance(capture, dict) else None
    if not isinstance(raw, list):
        trajectory = r.get("ng_trajectory")
        raw = trajectory.get("model_calls") if isinstance(trajectory, dict) else None
    if not isinstance(raw, list):
        return None
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        meta = item.get("response_metadata") if isinstance(item.get("response_metadata"), dict) else item
        report = meta.get("failure_report")
        events = report.get("events") if isinstance(report, dict) else None
        category = meta.get("error_category")
        out.append({"status_code": count(meta.get("status_code")), "upstream_status_code": count(meta.get("upstream_status_code")),
                    "error_category": category[:64] if isinstance(category, str) else None,
                    "failure_events": [{"failure_kind": event.get("failure_kind") if isinstance(event.get("failure_kind"), str) else None,
                                        "http_status": count(event.get("http_status")), "stage": event.get("stage")}
                                       for event in (events.values() if isinstance(events, dict) else ()) if isinstance(event, dict)],
                    "started_at": finite(item.get("started_at")), "duration_ms": finite(item.get("latency_total_ms", item.get("duration_ms")))})
    return out
def trial_metadata(value):
    if not isinstance(value, dict):
        return None
    result = {k: value.get(k) for k in ("response_id", "status_code", "response_status", "error_category",
                                        "upstream_status_code", "local_response_reason")}
    result["dialect"] = value.get("dialect") if value.get("dialect") in ("chat", "responses") else None
    result["finish_reason"] = value.get("finish_reason") if value.get("finish_reason") in ("stop", "length", "tool_calls", "content_filter", "function_call", "max_output_tokens", "max_tokens") else None
    result["model_ref"] = {k: value["model_ref"].get(k) for k in ("type", "name")} if isinstance(value.get("model_ref"), dict) else None
    refit = value.get("output_budget_refit")
    result["output_budget_refit"] = {k: refit.get(k) for k in ("requested_max_tokens", "window", "applied_max_tokens")} if isinstance(refit, dict) else None
    operations = value.get("model_operations")
    result["model_operations_recorded"] = "model_operations" in value
    result["model_operations"] = [{k: operation.get(k) for k in ("operation_id", "outcome", "final_attempt_id", "failure_id", "http_attempts", "transport_failures", "retry_wait_s", "request_elapsed_ms")}
                                  if isinstance(operation, dict) else None for operation in operations] if isinstance(operations, list) else None
    report = value.get("failure_report")
    result["failure_report"] = None
    if isinstance(report, dict) and isinstance(report.get("events"), dict):
        result["failure_report"] = {"failure_id": report.get("failure_id"), "omitted": report.get("omitted"),
            "events": {event_id: {k: event.get(k) for k in ("event_id", "boundary", "operation", "operation_id", "attempt_id", "cause_id", "failure_kind", "http_status", "stage")}
                       if isinstance(event, dict) else None for event_id, event in report["events"].items()}}
    return result
import hashlib
def finite(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None
def boolean(value):
    return value if type(value) is bool else None
def count(value):
    return value if type(value) is int else None
def message_text(item):
    # Whether a message holds non-whitespace text (an output text or a refusal); None when it saved no content.
    content = item.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return None
    def text(part):
        value = (part.get("refusal") if part.get("type") == "refusal" else part.get("text")) if isinstance(part, dict) else None
        return value if isinstance(value, str) else ""
    return any(text(part).strip() for part in content)
def answer_item(item):
    # An answer's output item: its type, role and status; an assistant message also says whether it has text
    # (``has_text``). The text stays local.
    if not isinstance(item, dict):
        return None
    projected = {k: item[k] for k in ("type", "role", "status") if k in item}
    if item.get("role") == "assistant" and (has_text := message_text(item)) is not None:
        projected["has_text"] = has_text
    return projected
def test_run(r):
    # A saved repository test run: its outcome flags only; the patch, test names and output stay local.
    if type(r.get("resolved")) is not bool:
        return None
    return {"resolved": r["resolved"], "patch_applied": boolean(r.get("patch_applied")),
            "timed_out": boolean(r.get("eval_timed_out")), "error": r.get("error") is not None,
            "test_results": isinstance(r.get("test_results"), dict), "model_patch": isinstance(r.get("model_patch"), str)}
def substep_tests(r):
    # Saved per-sub-step test results: counts and the problem verdict only; code and test output stay local.
    steps = r.get("step_results")
    if not isinstance(steps, list) or not all(type(step) is bool for step in steps):
        return None
    environments = r.get("step_environment_results")
    return {"steps": len(steps), "passed": sum(steps), "num_steps_total": count(r.get("num_steps_total")),
            "num_steps_passed": count(r.get("num_steps_passed")), "problem_accuracy": boolean(r.get("problem_accuracy")),
            "environment_results": len(environments) if isinstance(environments, list) else None}
import hashlib, json, re
JUDGE_VERDICT = re.compile(r"^(?:\[\[(A=B|A!=B)\]\]|judgement:\s*(yes|no))$", re.IGNORECASE)
def unit_tests(r):
    # Saved per-test unit test results of the answer's extracted code (True passes; False or an error code does
    # not): counts only, with the reasoning-format violation rate. Code, tests and their output stay local.
    tests = r.get("result")
    if not isinstance(tests, list) or not all(type(test) in (bool, int) for test in tests):
        return None
    return {"tests": len(tests), "passed": sum(test is True for test in tests),
            "all_passed": all(test is True for test in tests), "code": isinstance(r.get("extracted_model_code"), str),
            "reasoning_format_violation_rate": finite(r.get("reasoning_format_violation_rate"))}
def node_f1(r):
    # A saved node-set F1 (predicted_nodes, parse_failed, f1) re-graded from the answer's last non-empty line
    # ("Final Answer: [n1, n2]") against the saved expected node list, as the verifier scores it. Only the scores and
    # whether the re-parse reproduces the saved nodes leave the run host; the answer and nodes stay local.
    nodes, failed, saved = r.get("predicted_nodes"), r.get("parse_failed"), finite(r.get("f1"))
    if not isinstance(nodes, list) or type(failed) is not bool or saved is None or not isinstance(r.get("expected_answer"), str):
        return None
    response = r.get("response")
    output = response.get("output") if isinstance(response, dict) else None
    if not isinstance(output, list):
        return {"f1": saved, "regraded": None, "parse_matches": None}
    text = "".join(part["text"] for item in output if isinstance(item, dict) and item.get("type") == "message"
                   and isinstance(item.get("content"), list)
                   for part in item["content"] if isinstance(part, dict) and part.get("type") == "output_text"
                   and isinstance(part.get("text"), str))
    lines = [line for line in text.strip().split("\n") if line.strip()]
    match = re.search(r"Final Answer:\s*\[(.*)\]", lines[-1]) if lines else None
    if match is None:
        predicted, parse_failed = [], True
    else:
        predicted = [item.strip() for item in match.group(1).split(",") if item.strip()]
        parse_failed = False
    try:
        expected = set(json.loads(r["expected_answer"]))
    except (ValueError, TypeError):
        expected = set()
    chosen = set(predicted)
    if parse_failed:
        f1 = 0.0
    elif not expected and not chosen:
        f1 = 1.0
    elif not expected or not chosen or not chosen & expected:
        f1 = 0.0
    else:
        precision, recall = len(chosen & expected) / len(chosen), len(chosen & expected) / len(expected)
        f1 = 2 * precision * recall / (precision + recall)
    return {"f1": saved, "regraded": f1, "parse_matches": predicted == nodes and parse_failed is failed}
def guarded_objectives(r):
    # A guardrail-gated objective score an agent's own rubric saved beside the answer (response.metrics): the headline
    # (aa_headline) and the counts it follows from. Counts only; the world state and assertions stay local.
    response = r.get("response")
    metrics = response.get("metrics") if isinstance(response, dict) else None
    if not isinstance(metrics, dict) or "aa_headline" not in metrics:
        return None
    return {k: finite(metrics.get(k)) for k in
            ("aa_headline", "objectives_passed", "objectives_total", "guardrails_violated", "guardrails_total")}
def native_score(r):
    # A native scorer's saved metrics (native_metrics: weighted_score, execution) and whether it saved diagnostics.
    # Numbers and a flag only; the diagnostics stay local.
    metrics = r.get("native_metrics")
    if not isinstance(metrics, dict) or "weighted_score" not in metrics:
        return None
    diagnostics = r.get("native_diagnostics")
    return {"weighted_score": finite(metrics.get("weighted_score")), "execution": finite(metrics.get("execution")),
            "diagnostics": isinstance(diagnostics, dict) and bool(diagnostics)}
def saved_model_patch(r):
    # Whether the row saved the workspace patch its verifier grades, intact: model_patch hashes (UTF-8) to the saved
    # model_patch_sha256. False for a patch field without an intact patch (never collected, or not matching its
    # digest), None for a row without one. The patch itself stays local.
    if "model_patch" not in r and "model_patch_sha256" not in r:
        return None
    patch, digest = r.get("model_patch"), r.get("model_patch_sha256")
    return isinstance(patch, str) and isinstance(digest, str) and hashlib.sha256(patch.encode("utf-8")).hexdigest() == digest
def collected_patch(r):
    # Whether the row's verifier collected the agent's workspace patch itself and timed it (patch_collection_time_s and
    # verification_time_s): the verify response of a verifier that grades a collected patch on a fresh sandbox.
    return all(type(r.get(k)) in (int, float) for k in ("patch_collection_time_s", "verification_time_s"))
def judge_verdicts(value):
    # One boolean per saved judge evaluation (equal/yes is True, not equal/no is False), None where its label does
    # not parse; the label text stays local.
    out = []
    for item in value:
        label = item.get("verdict_label") if isinstance(item, dict) else None
        match = JUDGE_VERDICT.match(label.strip()) if isinstance(label, str) else None
        out.append(None if match is None else (match.group(1) or match.group(2)).lower() in ("a=b", "yes"))
    return out
# Labels a judge evaluation saves when it recorded no verdict; a null label is none either.
UNPARSED_VERDICT_LABELS = ("judge_parsing_error", "judge_error")
def judge_unparsed(value):
    # Whether a saved judge evaluation recorded no verdict. An evaluation without a verdict_label field (its verdict
    # is in the judge reply) and a label in another judge's vocabulary are not unparsed here.
    return any(isinstance(item, dict) and "verdict_label" in item and (item["verdict_label"] is None or (
        isinstance(item["verdict_label"], str) and item["verdict_label"].strip().casefold() in UNPARSED_VERDICT_LABELS))
        for item in value)
def games_unjudged(value):
    # Whether a pairwise judge's saved games include one without a verdict: its reward then averages the other games.
    return isinstance(value, list) and any(isinstance(item, dict) and "verdict" in item and item["verdict"] is None
        for item in value)
def rubric(r):
    # A rubric judge's verdict and scores as scalars only; the judge's text stays local.
    if "rubric_score" not in r and "scoring_threshold" not in r:
        return None
    verdict = r.get("verdict")
    return {"verdict": (verdict == "YES") if verdict in ("YES", "NO") else None,
            "rubric_score": finite(r.get("rubric_score")), "scoring_threshold": finite(r.get("scoring_threshold")),
            "scoring_max_points": finite(r.get("scoring_max_points")),
            "invalid_judge_response": r.get("invalid_judge_response") if type(r.get("invalid_judge_response")) is bool else None}
def grading_facts(r):
    # Scalar facts of the grading artifacts a verifier or environment saved on the row; their contents stay local.
    judgments, result = r.get("judgments"), r.get("result")
    info = result.get("reward_info") if isinstance(result, dict) else None
    return {
        "judge_response": r.get("judge_response") is not None,
        "judgments": isinstance(judgments, list) and bool(judgments),
        "judge_score": finite(r.get("judge_score")) is not None or (isinstance(judgments, list) and any(
            isinstance(item, dict) and finite(item.get("judge_score")) is not None for item in judgments)),
        "judge_evaluations": isinstance(r.get("judge_evaluations"), list) and bool(r["judge_evaluations"]),
        "judge_verdicts": judge_verdicts(r["judge_evaluations"]) if isinstance(r.get("judge_evaluations"), list) and r["judge_evaluations"] else None,
        "judge_error": isinstance(r.get("judge_evaluations"), list) and any(
            isinstance(item, dict) and item.get("verdict_label") == "JUDGE_ERROR" for item in r["judge_evaluations"]),
        "judge_unparsed": isinstance(r.get("judge_evaluations"), list) and judge_unparsed(r["judge_evaluations"]),
        "judge_error_message": isinstance(r.get("judge_error"), str) and bool(r["judge_error"].strip()),
        "invalid_judge_response": r.get("invalid_judge_response") is True,
        "games_unjudged": games_unjudged(r.get("games")),
        "rubric": rubric(r),
        "library_reward": finite(r.get("library_reward")),
        "sentence_metrics": sorted(k for k, v in r.items() if k.startswith("sentence_") and finite(v) is not None),
        "verifier_exit_code": r.get("verifier_exit_code") if type(r.get("verifier_exit_code")) is int else None,
        "verifier_error": r.get("verifier_error") is not None,
        "environment_reward": finite(info.get("reward")) if isinstance(info, dict) else None,
        "invalid_model_response": r.get("invalid_model_response") is True,
        "test_run": test_run(r),
        "substep_tests": substep_tests(r),
        "unit_tests": unit_tests(r),
        "node_f1": node_f1(r),
        "guarded_objectives": guarded_objectives(r),
        "native_score": native_score(r),
        "saved_model_patch": saved_model_patch(r),
        "collected_patch": collected_patch(r),
    }
def grading_result_binding(result):
    # Bind saved grading input before metadata projection.
    if not isinstance(result, dict) or not isinstance(result.get("response"), dict):
        return None
    response = result["response"]
    if not isinstance(response.get("id"), str) or not response["id"] or not isinstance(response.get("output"), list):
        return None
    reward = result.get("reward")
    if type(reward) not in (int, float) or (isinstance(reward, float) and not math.isfinite(reward)):
        return None
    value = {"response": response, "reward": reward}
    if "reward_components" in result:
        components = result["reward_components"]
        if not isinstance(components, dict) or not all(
            isinstance(key, str) and type(number) in (int, float)
            and (not isinstance(number, float) or math.isfinite(number))
            for key, number in components.items()
        ):
            return None
        value["reward_components"] = components

    def canonical(item):
        if item is None or type(item) in (str, bool, int):
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("nonfinite grading input")
            return int(item) if item.is_integer() else item
        if isinstance(item, list):
            return [canonical(child) for child in item]
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            return {key: canonical(child) for key, child in item.items()}
        raise ValueError("non-JSON grading input")

    try:
        payload = json.dumps(canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    except (TypeError, ValueError, UnicodeError, RecursionError):
        return None
SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
def session_reward(r):
    # The verifier's own reward file of the row's session (resources/<session_id>/verifier/reward.json, else
    # reward.txt), for a completed evaluation with no infrastructure error: its "reward" number only. A file that
    # exists but cannot be read is unreadable, never a fact.
    session = r.get("session_id")
    if (r.get("evaluation_completed") is not True or r.get("infrastructure_error") is not None
            or not isinstance(session, str) or not SESSION_ID.match(session) or ".." in session):
        return None
    directory = os.path.join(d, "resources", session, "verifier")
    for name in ("reward.json", "reward.txt"):
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as fh:
                text = fh.read(65536).decode("utf-8")
        except (OSError, UnicodeDecodeError):
            unreadable.add(os.path.relpath(path, d))
            return None
        try:
            value = json.loads(text) if name == "reward.json" else {"reward": float(text)}
        except ValueError:
            return None
        return finite(value.get("reward")) if isinstance(value, dict) else None
    return None
def trial_grading(value):
    if not isinstance(value, list):
        return None
    return [({k: item.get(k) for k in ("operation_id", "kind", "outcome", "result_binding_sha256")}
             | {"dependencies": trial_refs(item.get("dependencies"))}) if isinstance(item, dict) else None
            for item in value]
def trial_items(value, key, fields):
    if not isinstance(value, dict) or not isinstance(value.get(key), list):
        return None
    return [{k: item[k] for k in fields if k in item} if isinstance(item, dict) else None
            for item in value[key]]
# Parsed sources whose parsed bytes differ from the opening snapshot: like a changed closing snapshot, unstable.
changed_while_reading = set()
# Artifacts that exist but cannot be read, relative to the job dir: recorded as unknown, never a crash.
unreadable = set()
# A line longer than BIG_LINE bytes is read in chunks and parsed with the long strings of its bulky top-level values
# elided (``Pruner``), so a read holds what the collector keeps of a row, not the row: ``ng_trajectory`` repeats each
# model call's whole request, so one row grows with the square of its conversation and reaches gigabytes.
BIG_LINE = 16 << 20
CHUNK = 4 << 20
PRUNED_KEYS = (b"ng_trajectory", b"ng_agent_observations")
# A string is elided only when it is longer than this and is not one the collector reads (KEPT_PATHS).
ELIDE_ABOVE = 1024
# Elided text is checked as JSON in pieces of about this size.
VALIDATE = 1 << 20
# A string the collector reads is never elided; one above this size is no identity or status, so the row fails loudly.
KEPT_MAX = 1 << 20
# A key is read up to this many bytes: a name the collector reads is at most 21 characters, 126 bytes with every one
# escaped, so a longer key matches none of them.
KEY_MAX = 256
# The places in a pruned value whose string the collector projects (identities, statuses, kinds, names), as the keys
# from the row down to the string, list levels left out. A string anywhere else is uninterpreted payload (request
# input, tool output, conversation, even under a field that shares a name with one of these) and is elided when long.
def kept_paths():
    paths = set()

    def add(prefix, names):
        paths.update(tuple(part.encode() for part in prefix.split() + [name]) for name in names.split())

    add("ng_trajectory", "task_id rollout_id")
    ref = "type name"
    for group, fields in (
        ("invocations", "invocation_id status in_session verification_skipped error_type parent_invocation_id stop_reason"),
        ("verifier_invocations", "invocation_id status failure_kind"),
        ("turns", "invocation_id task_id rollout_id turn_no step_count"),
        ("tool_calls", "invocation_id tool_call_id tool_name status error_type"),
    ):
        add("ng_trajectory " + group, fields)
        if group != "tool_calls":
            add("ng_trajectory " + group + " model_calls", "model_call_id response_id")
            add("ng_trajectory " + group + " model_calls model_ref", ref)
    add("ng_trajectory verifier_invocations grading_operations", "operation_id kind outcome result_binding_sha256")
    for field in ("model_call_id", "response_id"):
        add("ng_trajectory verifier_invocations grading_operations dependencies", field)
    add("ng_trajectory verifier_invocations grading_operations dependencies model_ref", ref)
    add("ng_trajectory verifier_invocations grading_record", "verifier score verdict result_binding_sha256")
    add("ng_trajectory verifier_invocations sandbox_observations",
        "role provider sandbox_id outcome exit_code teardown_confirmed error_type")
    add("ng_trajectory gaps", "code")
    add("ng_trajectory model_calls", "model_call_id started_at status_code upstream_status_code error_category")
    add("ng_trajectory model_calls response output", "type role status call_id name")
    add("ng_trajectory model_calls request input", "type call_id")
    for call in ("model_calls", "model_calls response_metadata"):
        base = "ng_trajectory " + call
        add(base, "response_id status_code response_status error_category upstream_status_code local_response_reason "
                  "dialect finish_reason")
        add(base + " model_ref", ref)
        add(base + " output_budget_refit", "requested_max_tokens window applied_max_tokens")
        add(base + " model_operations", "operation_id outcome final_attempt_id failure_id http_attempts "
                                        "transport_failures retry_wait_s request_elapsed_ms")
        add(base + " failure_report", "failure_id omitted")
        add(base + " failure_report events *", "event_id boundary operation operation_id attempt_id cause_id "
                                               "failure_kind http_status stage")
    return frozenset(paths)


KEPT_PATHS = kept_paths()
STRUCTURE = re.compile(rb'["{}\[\],:]')
NESTED = re.compile(rb'["{}\[\]]')
HEX = frozenset(b"0123456789abcdefABCDEF")
def unescaped(key):
    # A key as the parsed row names it, so that "invocation_id" written with escapes is the same key. One that is not
    # a valid string (the row fails when it is parsed) or is cut at KEY_MAX matches no path.
    if b"\\" not in key:
        return key
    try:
        return json.loads(b'"' + key + b'"').encode()
    except ValueError:
        return b"\0"
class Pruner:
    # Copies one JSON object in chunks, replacing each long string inside a PRUNED_KEYS value that the collector does
    # not read with "<elided N bytes>". Only top-level keys are matched; any other value is copied byte for byte. An
    # elided string is still checked as a JSON string, in pieces: a row is invalid exactly when it is invalid whole.
    def __init__(self):
        self.out, self.in_string, self.escape = [], False, False
        self.expect_key, self.key, self.last_key, self.target = False, None, None, False
        self.kinds, self.keys = [], []
        self.mode, self.buffer, self.size, self.carry = None, None, 0, b""
        self.invalid = False
    def body(self, data):
        if self.mode == "elide":
            self.size += len(data)
            if self.buffer is not None:
                self.buffer += data
                if self.size > ELIDE_ABOVE:
                    self.carry, self.buffer = self.buffer, None
            else:
                self.carry += data
            if self.buffer is None and len(self.carry) >= VALIDATE:
                self.check(False)
        else:
            if self.mode == "keep":
                self.size += len(data)
                if self.size > KEPT_MAX:
                    self.invalid = True
            if self.key is not None and len(self.key) < KEY_MAX:
                self.key += data
    def check(self, final):
        # Checks the text gathered so far, up to the last complete escape and UTF-8 character unless it is the end.
        data = self.carry
        cut = len(data)
        if not final:
            if (cut - len(data.rstrip(b"\\"))) % 2:
                cut -= 1
            else:
                for width in range(2, 6):
                    tail = data[cut - width:] if cut >= width else b""
                    if tail[:2] == b"\\u" and all(byte in HEX for byte in tail[2:]):
                        head = data[: cut - width + 1]
                        if (len(head) - len(head.rstrip(b"\\"))) % 2:
                            cut -= width
                            break
            for back in range(1, min(3, cut) + 1):
                lead = data[cut - back]
                if lead < 0x80:
                    break
                if lead >= 0xC0:
                    if back < (2 if lead < 0xE0 else 3 if lead < 0xF0 else 4):
                        cut -= back
                    break
        piece, self.carry = data[:cut], data[cut:]
        if self.invalid or not piece:
            return
        try:
            json.loads('"' + piece.decode("utf-8", "surrogatepass") + '"')
        except ValueError:
            self.invalid = True
    def kept(self):
        # Whether the string that starts here sits where the collector reads it.
        path = [key for key, kind in zip(self.keys, self.kinds) if kind == b"{"]
        if any(key is None for key in path):
            return True
        for index in range(2, len(path)):
            if path[index - 1] == b"events" and path[index - 2] == b"failure_report":
                path[index] = b"*"
        return tuple(path) in KEPT_PATHS
    def open_string(self, depth):
        # How the string that starts here is handled: a top-level key is captured, a string the collector reads is
        # copied, any other string inside a pruned value is elided when long.
        self.mode = None
        if self.expect_key and (depth == 1 or self.target):
            self.key = bytearray()
        if self.target:
            key = None if self.expect_key else self.keys[-1]
            self.size = 0
            if self.expect_key or key is None or self.kept():
                self.mode = "keep"
            else:
                self.mode, self.buffer, self.carry = "elide", b"", b""
        self.expect_key = False
    def feed(self, chunk):
        pos, size, copy = 0, len(chunk), None if self.mode == "elide" else 0
        while pos < size:
            if self.in_string:
                if self.escape:
                    self.escape = False
                    self.body(chunk[pos:pos + 1])
                    pos += 1
                    continue
                # The closing quote is the first one not preceded by an odd run of backslashes. Found with find and a
                # count back over the run, so a string full of escapes costs no state beyond the chunk.
                end = chunk.find(b'"', pos)
                while end != -1:
                    start = end
                    while start > pos and chunk[start - 1] == 0x5C:
                        start -= 1
                    if (end - start) % 2 == 0:
                        break
                    end = chunk.find(b'"', end + 1)
                if end == -1:
                    start = size
                    while start > pos and chunk[start - 1] == 0x5C:
                        start -= 1
                    self.escape = (size - start) % 2 == 1
                    self.body(chunk[pos:size])
                    break
                self.body(chunk[pos:end])
                pos = end
                pos += 1
                self.in_string = False
                if self.mode == "elide":
                    if self.buffer is not None:
                        self.out.append(b'"' + self.buffer + b'"')
                    else:
                        self.check(True)
                        self.out.append(b'"<elided %d bytes>"' % self.size)
                    self.buffer, self.mode, copy = None, None, pos
                elif self.key is not None:
                    self.last_key, self.key = unescaped(bytes(self.key)), None
                    self.keys[-1] = self.last_key
                self.mode = None
                continue
            found = (STRUCTURE if self.target or len(self.keys) <= 1 else NESTED).search(chunk, pos)
            if found is None:
                break
            char, start, pos = found.group(), found.start(), found.end()
            if char == b'"':
                self.in_string = True
                self.open_string(len(self.keys))
                if self.mode == "elide":
                    if copy is not None:
                        self.out.append(chunk[copy:start])
                    copy = None
            elif char in (b"{", b"["):
                self.keys.append(None if char == b"{" else (self.keys[-1] if self.keys else None))
                self.kinds.append(char)
                self.expect_key = char == b"{"
            elif char in (b"}", b"]"):
                # An extra closer in a corrupt row has nothing to close: the parse of the row rejects it.
                if self.keys:
                    self.keys.pop()
                    self.kinds.pop()
                self.target = self.target and bool(self.keys)
            elif char == b":":
                if len(self.keys) == 1 and self.last_key is not None:
                    self.target = self.last_key in PRUNED_KEYS
            elif char == b",":
                if len(self.keys) == 1:
                    self.target = False
                self.expect_key = self.kinds[-1] == b"{"
        if copy is not None:
            self.out.append(chunk[copy:])
    def result(self, newline):
        # An invalid string makes the whole row invalid; its newline decides, as for a whole row, whether it is a
        # malformed record or a truncated last one.
        if self.invalid:
            return b"}" + (b"\n" if newline else b"")
        data, self.out = b"".join(self.out), []
        # A newline that ended an unfinished elided string is not in the copied text: keep it, so that a malformed
        # record is not mistaken for a truncated last line.
        return data + b"\n" if newline and not data.endswith(b"\n") else data
def lines(fh, digest):
    # Each line with its number; a line longer than BIG_LINE comes back pruned. Every byte read goes into the digest.
    number = 0
    while True:
        line = fh.readline(BIG_LINE + 1)
        if not line:
            return
        digest.update(line)
        if len(line) > BIG_LINE and not line.endswith(b"\n"):
            pruner = Pruner()
            pruner.feed(line)
            last = line
            while not line.endswith(b"\n"):
                line = fh.readline(CHUNK)
                if not line:
                    break
                last = line
                digest.update(line)
                pruner.feed(line)
            line = pruner.result(last.endswith(b"\n"))
            # The rebuilt row is the only copy that stays alive while it is parsed.
            del pruner
        yield number, line
        number += 1
def rows(name, skip=0):
    p = os.path.join(a, name)
    if not os.path.exists(p):
        return None
    out = []
    # The parse hashes every byte it reads: it must read exactly the opening snapshot's bytes. This catches a rewrite
    # before or during the parse, an ABA one included, without a separate hashing pass.
    read = hashlib.sha256()
    try:
        source = open(p, "rb")
    except OSError:
        unreadable.add(os.path.relpath(p, d))
        return None
    with source as fh:
        for i, line in lines(fh, read):
            if i < skip or not line.strip():
                continue
            try:
                r = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Binary iteration gives an unterminated line only at EOF; inventory must always be intact.
                if name != "rollouts_materialized_inputs.jsonl" and not line.endswith(b"\n"):
                    break
                raise
            if name == "rollouts_failures.jsonl":
                r["_line"] = i
            if isinstance(r, dict) and r.get("_ng_run_id") is not None:
                managed_store.append(True)
            if name == "rollouts.jsonl" and any(managed_store):
                r["_line"] = i
            selected = {k: r[k] for k in KEEP if k in r}
            if isinstance(r, dict) and r.get("failure_kind") == JUDGE_SKIPPED and isinstance(r.get("failure_reason"), str):
                # The skip's own reason (a bounded configuration note), never an answer.
                selected["_ng_skipped_reason"] = r["failure_reason"][:200]
            if isinstance(r, dict) and r.get("failure_kind") == PROVIDER_QUOTA_EXHAUSTED:
                # Only the reset time of a quota stop leaves the run host, never its message.
                reset = re.search(QUOTA_RESET_REGEX, str(r.get("_ng_failure_message") or r.get("failure_reason") or ""))
                if reset:
                    selected["_ng_quota_resets_at"] = reset.group(1)
            if name in ("rollouts.jsonl", "rollouts_failures.jsonl"):
                calls = model_call_facts(r)
                if calls is not None:
                    selected["_ng_collected_model_calls"] = calls
            if name == "rollouts.jsonl":
                binding = grading_result_binding(r)
                if binding is not None:
                    selected["_ng_collected_grading_binding"] = binding
                facts = grading_facts(r)
                reward_file = session_reward(r)
                if reward_file is not None:
                    facts["session_reward"] = reward_file
                # A zero is a fact (a wrong answer's library_reward, a passing exit code), so compare by identity; so
                # is a patch field without an intact patch.
                if facts["saved_model_patch"] is False or any(
                        value is not None and value is not False and value != [] for value in facts.values()):
                    selected["_ng_collected_grading_facts"] = facts
                adaptations = r.get("agent_adaptations")
                if isinstance(adaptations, dict):
                    # An agent's own per-row adaptation counts: bounded names and non-negative integers only.
                    selected["agent_adaptations"] = {k[:64]: v for k, v in adaptations.items()
                                                     if isinstance(k, str) and type(v) is int and v >= 0}
            record = r.get("_ng_failure_record")
            if isinstance(record, dict):
                failure = record.get("failure")
                selected["_ng_failure_record"] = {"source": record.get("source"), "delivery": record.get("delivery"),
                    "failure": {"stage": failure.get("stage"), "failure_kind": failure.get("failure_kind")} if isinstance(failure, dict) else None}
            if name == "rollouts_failures.jsonl":
                # A failure's recorded cause as facts: whether the environment's own episode deadline ended it, and for
                # each failed model call's events, whether the call was the policy's and whether it hit the context limit.
                policy = r.get("ng_policy_model_server")
                policy = policy if isinstance(policy, str) and policy else "policy_model"
                failure = record.get("failure") if isinstance(record, dict) else None
                capture = r.get("ng_model_call_capture")
                calls = capture.get("calls") if isinstance(capture, dict) else None
                event_calls = {}
                for call in calls if isinstance(calls, list) else []:
                    report = call.get("failure_report") if isinstance(call, dict) else None
                    events = report.get("events") if isinstance(report, dict) else None
                    if not isinstance(events, dict) or not events:
                        continue
                    model_ref = call.get("model_ref")
                    # Captures made before model_ref existed held only policy calls.
                    owner = True if model_ref is None else model_ref.get("name") == policy if isinstance(model_ref, dict) else None
                    limit = "context_length_exceeded" in (call.get("local_response_reason"), call.get("error_category"))
                    for event_id in events:
                        if isinstance(event_id, str) and len(event_calls) < 256:
                            event_calls[event_id[:64]] = {"policy": owner, "context_limit": limit}
                selected["_ng_collected_cause"] = {
                    "episode_deadline": isinstance(record, dict) and record.get("source") == "environment"
                    and isinstance(failure, dict) and failure.get("failure_reason") == EPISODE_DEADLINE_REASON
                    and not failure.get("failure_kind"),
                    "event_calls": event_calls,
                }
            for key in ("response", "_ng_failure_response", "_ng_failure_partial_response"):
                answer = r.get(key)
                if key in r and not isinstance(answer, dict):
                    selected[key] = None  # Preserve unusable-slot presence and verifier-input precedence.
                if isinstance(answer, dict):
                    output = answer.get("output")
                    selected[key] = {k: answer[k] for k in ("object", "id", "status") if k in answer}
                    # Cutoff facts only: why the answer is incomplete, its output token count and its output cap.
                    details, usage = answer.get("incomplete_details"), answer.get("usage")
                    if isinstance(details, dict) and isinstance(details.get("reason"), str):
                        selected[key]["incomplete_details"] = {"reason": details["reason"][:64]}
                    if type(answer.get("max_output_tokens")) is int:
                        selected[key]["max_output_tokens"] = answer["max_output_tokens"]
                    if isinstance(usage, dict) and type(usage.get("output_tokens")) is int:
                        selected[key]["usage"] = {"output_tokens": usage["output_tokens"]}
                    metadata = answer.get("metadata")
                    if isinstance(metadata, dict) and "ng_agent_outcome" in metadata:
                        # Export only the bounded public classification, never arbitrary metadata or JSON extras.
                        outcome = None
                        try:
                            value = json.loads(metadata["ng_agent_outcome"])
                            if (isinstance(value, dict) and set(value) == {"failure_kind", "status_code"}
                                and value["failure_kind"] == "model_request_error"
                                and type(value["status_code"]) is int and 400 <= value["status_code"] <= 599):
                                outcome = json.dumps(value)
                        except (TypeError, ValueError, RecursionError):
                            pass
                        selected[key]["metadata"] = {"ng_agent_outcome": outcome}
                    if answer.get("error") is not None:
                        selected[key]["error"] = {}  # Presence only; diagnostic messages stay on the run host.
                    if isinstance(output, list):
                        selected[key]["output"] = [answer_item(item) for item in output]

            if name == "rollouts_materialized_inputs.jsonl":
                identity = r.get("trajectory_identity")
                if isinstance(identity, dict) and all(isinstance(identity.get(k), str) and identity[k] for k in ("task_id", "rollout_id")):
                    task_id = identity["task_id"]
                else:
                    task_id = next((r[k] for k in TASK_ID_FIELDS if r.get(k) is not None), r.get("_ng_task_index"))
                # Only the scalar task identity leaves the host; dataset content stays in the saved input.
                selected["_ng_collected_task_id"] = str(task_id) if type(task_id) in (str, int) else None
            # Operation evidence only: prompts, answers, conversations and model payloads stay on the run host.
            trajectory = r.get("ng_trajectory")
            if name == "rollouts.jsonl" and isinstance(trajectory, dict):
                selected["ng_trial_observations"] = {
                    "task_id": trajectory.get("task_id"), "rollout_id": trajectory.get("rollout_id"),
                    "invocations": [({k: item.get(k) for k in ("invocation_id", "status", "in_session", "verification_skipped", "error_type", "parent_invocation_id", "stop_reason")} | {"model_calls": trial_refs(item.get("model_calls"))})
                                    if isinstance(item, dict) else None for item in trajectory.get("invocations")] if isinstance(trajectory.get("invocations"), list) else None,
                    "verifier_invocations": [({k: item.get(k) for k in ("invocation_id", "status", "failure_kind")} | {"model_calls": trial_refs(item.get("model_calls")), "grading_operations": trial_grading(item.get("grading_operations")), "grading_record": {k: item["grading_record"].get(k) for k in ("verifier", "score", "verdict", "result_binding_sha256")} if isinstance(item.get("grading_record"), dict) else None, "sandbox_observations": [{k: fact.get(k) for k in ("role", "provider", "sandbox_id", "outcome", "exit_code", "teardown_confirmed", "error_type")} if isinstance(fact, dict) else None for fact in item["sandbox_observations"]] if isinstance(item.get("sandbox_observations"), list) else item.get("sandbox_observations")})
                                            if isinstance(item, dict) else None for item in trajectory.get("verifier_invocations")] if isinstance(trajectory.get("verifier_invocations"), list) else None,
                    "model_calls": [{"model_call_id": item.get("model_call_id"), "started_at": finite(item.get("started_at")), "response_metadata": trial_metadata(item.get("response_metadata")),
                                     "output_items": trial_items(item.get("response"), "output", ("type", "role", "status", "call_id", "name")),
                                     "input_items": trial_items(item.get("request"), "input", ("type", "call_id"))}
                                    if isinstance(item, dict) else None for item in trajectory.get("model_calls")] if isinstance(trajectory.get("model_calls"), list) else None,
                    "turns": [({k: item.get(k) for k in ("invocation_id", "task_id", "rollout_id", "turn_no", "step_count")} | {"model_calls": trial_refs(item.get("model_calls"))}) if isinstance(item, dict) else None for item in trajectory.get("turns")] if isinstance(trajectory.get("turns"), list) else None,
                    "tool_calls": [{k: item.get(k) for k in ("invocation_id", "tool_call_id", "tool_name", "status", "error_type")}
                                   if isinstance(item, dict) else None for item in trajectory.get("tool_calls")] if isinstance(trajectory.get("tool_calls"), list) else None,
                    "tool_call_count": len(trajectory["tool_calls"]) if isinstance(trajectory.get("tool_calls"), list) else None,
                    "gaps": [{"code": gap.get("code")} if isinstance(gap, dict) else None for gap in trajectory.get("gaps")] if isinstance(trajectory.get("gaps"), list) else None,
                }
                for key in ("agent_ref", "_ng_environment_server", "_ng_result_type"):
                    if key in r:
                        selected[key] = r[key]
            out.append(selected)
    if read.hexdigest() != source_snapshot[name]:
        changed_while_reading.add(name)
    return out
def num(name):
    # (value, state): "absent" when the chain never wrote the file, "malformed" when it cannot be read as a number.
    path = os.path.join(d, name)
    if not os.path.exists(path):
        return None, "absent"
    try:
        value = float(open(path).read().strip())
        if not math.isfinite(value) or value < 0 or (name == ".gym_infra_retries" and not value.is_integer()):
            return None, "malformed"
        return value, "ok"
    except (OSError, ValueError):
        return None, "malformed"
try:
    offset = int(open(os.path.join(a, "rollouts_resubmit_offset")).read().strip())
except (OSError, ValueError):
    offset = 0
def accounting_unknown_reason():
    # Why the chain's own accounting is unknown, or None when it is known. An unobserved claim comes first.
    marker = os.path.join(d, ".gym_chain_incomplete")
    try:
        for name in os.listdir(d):
            if name.startswith(".gym_followup_queued."):
                value = open(os.path.join(d, name)).read().strip()
                if not value.isascii() or not value.isdecimal():
                    return UNOBSERVED_CLAIM_REASON
        text = open(marker).read() if os.path.exists(marker) else ""
    except (OSError, ValueError):
        return UNREADABLE_CLAIM_REASON
    if text.startswith(ACCOUNTING_UNKNOWN_PREFIX):
        return (text[len(ACCOUNTING_UNKNOWN_PREFIX):].strip().splitlines() or ["no reason recorded"])[0][:300]
    return None
def accounting_unknown():
    return accounting_unknown_reason() is not None
def job_ids(prefix):
    # The job id each allocation recorded in its <prefix><allocation> file; only decimal ids leave the run host.
    ids = {}
    try:
        for name in os.listdir(d):
            if name.startswith(prefix):
                value = open(os.path.join(d, name)).read().strip()
                if value.isascii() and value.isdecimal():
                    ids[name[len(prefix):]] = value
    except (OSError, ValueError):
        return {}
    return ids
def followup_claims():
    # The follow-up job id each allocation claimed, keyed by that allocation.
    return job_ids(".gym_followup_queued.")
def sha256(path):
    # One bounded pass with stable stat identity. A same-size in-place rewrite within one timestamp tick keeps the
    # identity: the opening and closing snapshots compare bytes, and rows() checks the bytes it parses against the
    # opening snapshot, so such a rewrite of a parsed source is caught even when it is undone before closing.
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    try:
        source = open(path, "rb")
    except FileNotFoundError:
        return None
    except OSError:
        # Present but unreadable (an owner-only file, for example): a stable marker, so both snapshots agree.
        unreadable.add(os.path.relpath(path, d))
        return "unreadable"
    h = hashlib.sha256()
    with source:
        before = os.fstat(source.fileno())
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
        after = os.fstat(source.fileno())
    current = os.stat(path)
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise RuntimeError("run artifact changed while hashing")
    return h.hexdigest()
# The live exporter's run id file, and the post-run export receipt checked against the same files as its version.
RECEIPT_FILES = {"rollouts_jsonl_fpath": "rollouts.jsonl", "resolved_config_yaml_fpath": "rollouts_resolved_config.yaml",
                 "aggregate_metrics_json_fpath": "rollouts_aggregate_metrics.json",
                 "failures_jsonl_fpath": "rollouts_failures.jsonl",
                 "materialized_inputs_jsonl_fpath": "rollouts_materialized_inputs.jsonl"}
source_files = (*RECEIPT_FILES.values(), "quality_summary.json", "rollouts_manifest.json", "rollouts_attempts.jsonl", "rollouts_resubmit_offset",
            "rollouts_coverage.json", "../.gym_infra_retries", "../.gym_accumulated_walltime", "../.gym_job_chain")
def followup_snapshot():
    # Keep scheduler state outside the installed Store artifact protocol, but bind both the dynamic
    # claim set and its contents. Directory timestamps also detect transient creation/removal.
    try:
        before = os.stat(d)
        claims = {name: sha256(os.path.join(d, name)) for name in sorted(os.listdir(d))
                  if name.startswith(".gym_followup_queued.")}
        after = os.stat(d)
        identity = lambda st: (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_ctime_ns)
        return (identity(after), claims) if identity(before) == identity(after) else None
    except (OSError, RuntimeError):
        return None
initial_followups = followup_snapshot()
source_snapshot = {name: sha256(os.path.join(a, name)) for name in source_files}
accounting_files = (".gym_infra_retries", ".gym_accumulated_walltime", ".gym_job_chain", ".gym_infra_retry_extension",
                    ".gym_chain_incomplete", "../resolved-config.yaml", "artifacts/rollouts_resubmit_offset")
accounting_snapshot = {name: sha256(os.path.join(d, name)) for name in accounting_files}
try:
    chain = open(os.path.join(d, ".gym_job_chain")).read().split() if os.path.exists(os.path.join(d, ".gym_job_chain")) else []
except OSError:
    unreadable.add(".gym_job_chain")
    chain = []
unknown_accounting = accounting_unknown()
mlflow = {"live_run_id": None, "receipt": None}
try:
    mlflow["live_run_id"] = open(os.path.join(a, "rollouts_mlflow_run_id")).read().strip()
except OSError:
    pass
if os.path.exists(os.path.join(a, "rollouts_mlflow_export_done")):
    try:
        text = open(os.path.join(a, "rollouts_mlflow_export_done")).read()
    except OSError:
        text = ""
    try:
        receipt = json.loads(text)
    except ValueError:
        receipt = None
    if (isinstance(receipt, dict) and type(receipt.get("version")) is int and receipt["version"] == 1
            and isinstance(receipt.get("run_id"), str) and receipt["run_id"].strip()
            and isinstance(receipt.get("artifacts"), dict)
            and set(receipt["artifacts"]) == set(RECEIPT_FILES)):
        # Bound to the opening snapshot; the closing one below reports any later change as unstable sources.
        current = {k: source_snapshot[n] for k, n in RECEIPT_FILES.items()}
        health_current = True
        try:
            aggregate = json.load(open(os.path.join(a, "rollouts_aggregate_metrics.json")))
            payloads = [entry["_ng_run_health_metrics"] for entry in aggregate if isinstance(entry, dict) and "_ng_run_health_metrics" in entry]
            if payloads:
                health_sources = {k: v for k, v in current.items() if k != "aggregate_metrics_json_fpath"}
                health_sources.update({name: source_snapshot["rollouts_" + name] for name in ("manifest.json", "attempts.jsonl")})
                try:
                    summary = json.load(open(os.path.join(a, "quality_summary.json")))
                    stats = summary.get("run", {}).get("stats", {})
                    if isinstance(stats, dict): stats.pop("acceptance_health", None)
                    health_sources["quality_summary"] = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
                except FileNotFoundError:
                    health_sources["quality_summary"] = None
                health_current = (len(payloads) == 1 and isinstance(payloads[0], dict)
                    and payloads[0].get("source_artifacts") == health_sources
                    and isinstance(payloads[0].get("metrics"), dict) and bool(payloads[0]["metrics"])
                    and all(isinstance(k, str) and k.startswith("health/") and type(v) in (int, float) and math.isfinite(v)
                            for k, v in payloads[0]["metrics"].items()))
        except (OSError, ValueError, TypeError, AttributeError):
            health_current = False
        mlflow["receipt"] = {"run_id": receipt["run_id"], "current": receipt["artifacts"] == current and health_current,
                             "uploaded_rollouts": receipt.get("uploaded_rollouts"),
                             "rollout_projection_version": receipt.get("rollout_projection_version")}
    elif not isinstance(receipt, (dict, list)) and len(text.split()) == 1:
        # A marker that names only the backend run: it carries no digests to check the artifacts against.
        mlflow["receipt"] = {"run_id": text.strip(), "current": None}
    else:
        mlflow["receipt"] = "malformed"
# A run whose health report predates `provenance.mlflow.tracking_required` still states it in its saved config.
try:
    config_match = re.search(r"^mlflow_tracking_required:[ 	]*['\"]?(true|false)['\"]?[ 	]*(?:#.*)?$",
                             open(os.path.join(a, "rollouts_resolved_config.yaml")).read(), re.MULTILINE | re.IGNORECASE)
    mlflow["config_tracking_required"] = None if config_match is None else config_match.group(1).lower() == "true"
except OSError:
    pass
# The tracking server named by the saved config, for the reader to ask the receipt's run for its status; an absent,
# null or interpolated value names none.
try:
    uri_match = re.search(r"^mlflow_tracking_uri:[ \t]*['\"]?([^\s'\"#]+)",
                          open(os.path.join(a, "rollouts_resolved_config.yaml")).read(), re.MULTILINE)
    uri = uri_match.group(1) if uri_match is not None else None
    mlflow["config_tracking_uri"] = uri if uri and "${" not in uri and uri.lower() not in ("null", "~") else None
except OSError:
    pass
quality = None
try:
    run = json.load(open(os.path.join(a, "quality_summary.json")))["run"]
    acc = run.get("artifacts") or {}
    sources = run.get("rollouts")
    source_matches = (isinstance(sources, list) and len(sources) == 1 and isinstance(sources[0], str)
                      and os.path.abspath(os.path.join(a, sources[0])) == os.path.abspath(os.path.join(a, "rollouts.jsonl")))
    provenance = run.get("provenance")
    bindings = provenance.get("rollout_artifact_digests") if isinstance(provenance, dict) else None
    binding = bindings[0] if isinstance(bindings, list) and len(bindings) == 1 else None
    expected_files = {k: v for k, v in RECEIPT_FILES.items() if k != "aggregate_metrics_json_fpath"}
    binding_valid = (isinstance(binding, dict) and set(binding) == set(expected_files)
                     and all(value is None or isinstance(value, str) and len(value) == 64
                             and all(c in "0123456789abcdef" for c in value) for value in binding.values())
                     and isinstance(binding.get("rollouts_jsonl_fpath"), str))
    source_matches = source_matches and binding_valid and binding == {k: source_snapshot[n] for k, n in expected_files.items()}
    quality = {"source_matches": source_matches, "rollouts": sources, "artifacts": {k: acc.get(k) for k in ("expected", "missing", "healthy", "unhealthy_reasons", "partial")},
               "verdicts": run.get("verdicts"), "issues": run.get("issues"), "coverage": acc.get("coverage"),
               "ignored_checks": run.get("ignored_checks"), "responses": (run.get("stats") or {}).get("responses"),
               "provenance": run.get("provenance"),
               "observed_upstream_recovery": (run.get("stats") or {}).get("observed_upstream_recovery"),
               "observed_runtime": (run.get("stats") or {}).get("observed_runtime"),
               "observed_request_effort": (run.get("stats") or {}).get("observed_request_effort"),
               "ended_on_failed_model_call": ((run.get("stats") or {}).get("model_call_errors") or {}).get("ended_on_error"),
               "verification_skipped": ((run.get("stats") or {}).get("model_call_errors") or {}).get("verification_skipped"),
               "context_window_exceeded": ((run.get("stats") or {}).get("model_call_errors") or {}).get("context_window_exceeded")}
except (OSError, ValueError, KeyError, TypeError, AttributeError):
    # Optional diagnostics cannot make the authoritative saved inventory unreadable.
    pass
benchmark_metric_coverage = None
try:
    # A benchmark may declare how many trials its headline metric covered, beside its metrics; one declaration only.
    declared = set()
    value = json.load(open(os.path.join(a, "rollouts_aggregate_metrics.json")))
    for entry in value if isinstance(value, list) else []:
        metrics = entry.get("agent_metrics") if isinstance(entry, dict) else None
        if not isinstance(metrics, dict) or "metric_coverage/trials" not in metrics:
            continue
        trials, expected = metrics.get("metric_coverage/trials"), metrics.get("metric_coverage/expected")
        name = metrics.get("metric_coverage/name")
        valid = (all(type(n) is int and n >= 0 for n in (trials, expected)) and trials <= expected
                 and (name is None or isinstance(name, str) and 0 < len(name) <= 40 and name.isprintable()))
        declared.add((trials, expected, name) if valid else None)
    if len(declared) == 1 and None not in declared:
        benchmark_metric_coverage = list(next(iter(declared)))
except (OSError, ValueError, AttributeError, TypeError):
    pass
score = None
if len(sys.argv) > 2:
    try:
        value = json.load(open(os.path.join(a, "rollouts_aggregate_metrics.json")))
        selector = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        if isinstance(value, list):
            if not selector or any(k not in ("agent", "environment") or not isinstance(v, str) or not v
                                   for k, v in selector.items()):
                raise ValueError("aggregate lists require a stable agent or environment selector")
            matches = [r for r in value if isinstance(r, dict)
                       and ("agent" not in selector or isinstance(r.get("agent_ref"), dict)
                            and r["agent_ref"].get("name") == selector["agent"])
                       and ("environment" not in selector or r.get("_ng_environment_server") == selector["environment"])]
            if len(matches) != 1:
                raise ValueError("aggregate identity is absent or ambiguous")
            value = matches[0]
        elif selector:
            raise ValueError("identity selectors require producer aggregate entries")
        for part in json.loads(sys.argv[2]):  # the metric path as a JSON list: keys may contain "/"
            # A string key into an object, a non-negative integer index into an array; anything else reads nothing.
            if isinstance(value, dict) and isinstance(part, str):
                value = value[part]
            elif isinstance(value, list) and type(part) is int and 0 <= part < len(value):
                value = value[part]
            else:
                raise KeyError(part)
        score = value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        pass
coverage_allocation_id = None
coverage_scored = None
try:
    # The allocation that last opened the Store for writing; only its id leaves the run host.
    report = json.load(open(os.path.join(a, "rollouts_coverage.json")))
    value = report.get("allocation_id")
    coverage_allocation_id = value if isinstance(value, str) and value.isascii() and value.isdecimal() else None
    # Collection's own count of the rollouts whose score entered the metrics, and its expected count.
    if all(type(report.get(k)) is int and report[k] >= 0 for k in ("scored", "expected")):
        coverage_scored = [report["scored"], report["expected"]]
except (OSError, ValueError, AttributeError):
    pass
collected = {"materialized": rows("rollouts_materialized_inputs.jsonl"), "results": rows("rollouts.jsonl"),
                  "failures": rows("rollouts_failures.jsonl") or [], "resubmit_offset": offset, "infra_retries_used": num(".gym_infra_retries")[0],
                  "infra_retries_state": "unknown" if unknown_accounting else num(".gym_infra_retries")[1],
                  "walltime_used_s": num(".gym_accumulated_walltime")[0],
                  "walltime_state": "unknown" if unknown_accounting else num(".gym_accumulated_walltime")[1], "job_chain": chain,
                  "chain_incomplete": os.path.exists(os.path.join(d, ".gym_chain_incomplete")),
                  "quality": quality, "score": score, "mlflow": mlflow, "managed_store": any(managed_store),
                  "coverage_allocation_id": coverage_allocation_id, "coverage_scored": coverage_scored,
                  "benchmark_metric_coverage": benchmark_metric_coverage}
# One closing snapshot: every source hash above is the opening one, so a change during collection shows up here.
closing_snapshot = {name: sha256(os.path.join(a, name)) for name in source_files}
if quality is not None and (source_snapshot != closing_snapshot or changed_while_reading):
    quality["source_matches"] = False
stable_sources = (not changed_while_reading and source_snapshot == closing_snapshot
                  and accounting_snapshot == {name: sha256(os.path.join(d, name)) for name in accounting_files}
                  and initial_followups is not None and initial_followups == followup_snapshot())
collected["source_digests"] = source_snapshot if stable_sources else None
# The scheduler accounting a resubmit must find unchanged under its lock: counters, chain, marker and claims.
collected["accounting_snapshot"] = (
    accounting_snapshot | initial_followups[1]
) if stable_sources else None
# Do not upgrade a stale scheduler-state read into a fresh budget. The executor and generated job
# recheck the durable unknown marker under the existing submission/admission boundary as well.
unknown_reason = accounting_unknown_reason()
if not stable_sources or unknown_reason is not None:
    collected["infra_retries_state"] = collected["walltime_state"] = "unknown"
# The counters' own states, for a caller that reconciles the unknown reason from terminal scheduler records.
collected["accounting_unknown_reason"] = unknown_reason
collected["infra_retry_extension"], collected["infra_retry_extension_state"] = num(".gym_infra_retry_extension")
collected["counter_states"] = {"infra_retries": num(".gym_infra_retries")[1],
                               "walltime": num(".gym_accumulated_walltime")[1]} if stable_sources else None
collected["source_job_identity"] = initial_job_identity if initial_job_identity == job_identity() else None
collected["followup_claims"] = followup_claims()
# The sentinel each allocation queued behind its follow-up, keyed by that allocation.
collected["followup_sentinels"] = job_ids(".gym_followup_sentinel.")
def operator_epoch():
    # The last operator continuation's epoch; None when no continuation was journaled or its entry is unreadable.
    try:
        epoch = json.loads(open(os.path.join(d, OPERATOR_CONTINUATION_JOURNAL_NAME)).read().splitlines()[-1])["epoch"]
        length, walltime = epoch["chain_length"], epoch["walltime_s"]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None
    if type(length) is not int or length < 0 or type(walltime) not in (int, float) or not math.isfinite(walltime):
        return None
    return {"chain_length": length, "walltime_s": walltime} if walltime >= 0 else None
collected["operator_epoch"] = operator_epoch()
def attempt_extension():
    # Attempts an explicit recovery added to every rollout's cap (see --extend-attempts), and the file's state.
    path = os.path.join(a, "rollouts_attempt_extension")
    if not os.path.exists(path):
        return None, "absent"
    try:
        text = open(path).read(64).strip()
    except OSError:
        return None, "malformed"
    return (int(text), "ok") if text.isascii() and text.isdecimal() else (None, "malformed")
collected["attempt_extension"], collected["attempt_extension_state"] = attempt_extension()
def resume_drift_keys():
    # The operational settings the latest resume crossed (the drift journal's last line), keys only and bounded.
    try:
        with open(os.path.join(a, "rollouts_manifest.resume-drift.jsonl"), "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 65536))
            lines = [line for line in fh.read().splitlines() if line.strip()]
        keys = json.loads(lines[-1])["operational_keys"]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None
    if not isinstance(keys, list):
        return None
    return [key[:200] for key in keys if isinstance(key, str)][:200] or None
collected["resume_drift_keys"] = resume_drift_keys()
def chain_continuation_note():
    # A continuation's provisional line, left when it ended before queuing its follow-up, or a sentinel's line for a
    # follow-up cancelled before it started, or the guard's stop on rejected sandbox credentials or on a follow-up
    # submission the scheduler refused (see resume_script).
    try:
        text = open(os.path.join(d, ".gym_chain_incomplete")).read(2000)
    except OSError:
        return None
    line = text.strip().splitlines()[0] if text.strip() else ""
    named = (
        line.startswith(PROVISIONAL_MARKER_PREFIX)
        or CANCELLED_FOLLOWUP_MARKER in line
        or PROVIDER_CREDENTIALS_REJECTED in line
        or PROVIDER_QUOTA_EXHAUSTED in line
        or ENVIRONMENT_PROTOCOL_VIOLATION in line
        or line.startswith(FOLLOW_UP_SUBMISSION_REFUSED)
    )
    return line[:500] if named else None
collected["chain_continuation_note"] = chain_continuation_note()
collected["unreadable_files"] = sorted(unreadable)
def judge_jobs():
    # The judge-only jobs queued for this run's saved answers (JUDGE_JOB_JOURNAL_NAME, oldest first), each with its
    # receipt's outcome when the job wrote one. Only bounded ids, names, counts and states leave the run host: never a
    # saved answer, and only a receipt inside this job dir is read.
    def text(value, limit):
        return value[:limit] if isinstance(value, str) else None
    def count(value):
        return value if type(value) is int and value >= 0 else None
    jobs = []
    try:
        lines = open(os.path.join(d, JUDGE_JOB_JOURNAL_NAME)).read().splitlines()
    except OSError:
        return jobs
    for line in lines[-50:]:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict):
            continue
        job = {"job_id": text(entry.get("job_id"), 64), "at": text(entry.get("at"), 40),
               "nonce": text(entry.get("nonce"), 120), "partition": text(entry.get("partition"), 120),
               "walltime": text(entry.get("walltime"), 40), "rows": count(entry.get("rows")),
               "gym_commit": text(entry.get("gym_commit"), 64), "receipt": text(entry.get("receipt"), 1000),
               "error": text(entry.get("error"), 500), "state": "queued", "kind": text(entry.get("kind"), 40),
               "services": [text(s, 120) for s in entry["services"][:20] if isinstance(s, str)]
               if isinstance(entry.get("services"), list) else []}
        path = entry.get("receipt")
        receipt = None
        inside = isinstance(path, str) and os.path.isabs(path) and os.path.realpath(path).startswith(os.path.realpath(d) + os.sep)
        if inside and os.path.exists(path):
            try:
                receipt = json.load(open(path))
            except (OSError, ValueError):
                receipt = "malformed"
        if isinstance(receipt, dict) and receipt.get("version") == 1 and receipt.get("state") in ("completed", "failed"):
            job["state"] = receipt["state"]
            for name in ("selected", "judged", "settled", "withdrawn", "unsupported", "due_after"):
                job[name] = count(receipt.get(name))
            job["error"] = text(receipt.get("error"), 500)
            job["ended_at"] = text(receipt.get("ended_at"), 40)
            job["metrics_recomputed"] = receipt.get("metrics_recomputed") is True
            job["health_summary_recomputed"] = receipt.get("health_summary_recomputed") is True
        elif receipt is not None:
            job["state"] = "unknown"  # a receipt that is not one this reader knows
        jobs.append(job)
    return jobs
collected["judge_jobs"] = judge_jobs()
def judge_job_acknowledgements():
    # The operator's acknowledgements of failed judge-only jobs (JUDGE_JOB_ACK_JOURNAL_NAME): the ids each covers, bounded.
    acknowledgements = []
    try:
        lines = open(os.path.join(d, JUDGE_JOB_ACK_JOURNAL_NAME)).read().splitlines()
    except OSError:
        return acknowledgements
    for line in lines[-50:]:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        ids = entry.get("job_ids") if isinstance(entry, dict) else None
        if isinstance(ids, list) and all(isinstance(i, str) and 0 < len(i) <= 64 for i in ids):
            acknowledgements.append({"job_ids": ids[:20], "at": str(entry.get("at"))[:40]})
    return acknowledgements
collected["judge_job_acknowledgements"] = judge_job_acknowledgements()
print(json.dumps(collected))
"""
)
