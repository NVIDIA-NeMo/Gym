#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Post-hoc pairwise comparison judge for GDPVal evaluation.

Compares outputs of two models (reference vs eval) on GDPVal tasks using
an LLM judge, then computes ELO ratings from win rates.

Uses shared core functions from ``_gdpval_comparison.py``.

Usage:
    python scripts/compare_elo.py \\
        --reference-model-dir output/gdpval/Qwen3-235B-A22B-Thinking-2507 \\
        --eval-model-dir output/gdpval/NVIDIA-Nemotron-Super-V3 \\
        --reference-model-name Qwen3-235B \\
        --eval-model-name Nemotron-Super-V3 \\
        --server-address https://api.openai.com/v1 \\
        --judge-model-name gpt-4.1-2025-04-14 \\
        --api-key $JUDGE_API_KEY \\
        --reference-model-elo 1000 \\
        --num-trials 4 \\
        --max-concurrent-judgements 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI


# Ensure project root is importable — import comparison module directly
# to avoid pulling in ray/stirrup via the package __init__.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util

_comp_path = Path(__file__).resolve().parent.parent / "responses_api_agents/stirrup_agent/tasks/_gdpval_comparison.py"
_spec = importlib.util.spec_from_file_location("_gdpval_comparison", _comp_path)
_comp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_comp)

A_WIN_RESPONSE = _comp.A_WIN_RESPONSE
B_WIN_RESPONSE = _comp.B_WIN_RESPONSE
TIE_RESPONSE = _comp.TIE_RESPONSE
build_file_section = _comp.build_file_section
calculate_elo = _comp.calculate_elo
clean_up_paths = _comp.clean_up_paths
construct_judge_messages = _comp.construct_judge_messages
send_judge_request = _comp.send_judge_request
tally_result = _comp.tally_result
task_attempted = _comp.task_attempted


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class Logger:
    def __init__(self, log_dir: str, log_name: str, log_filename: str = "log.txt", overwrite: bool = False):
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_path = os.path.join(log_dir, log_filename)
        self.log_file = open(self.log_path, "w" if overwrite else "a", encoding="utf-8")

    def log(self, message: str):
        self.log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {self.log_name}]: {message}\n")
        self.log_file.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.log_file.close()


class JudgementLogger(Logger):
    def log_trial_result(self, trial_judgement, task_id, status, swapped):
        self.log(
            f"\n{'=' * 48}\n"
            f"Task ID: {task_id}\n"
            f"Trial judgement: {trial_judgement}\n"
            f"Trial count A: {status.win_count_a}\n"
            f"Trial count B: {status.win_count_b}\n"
            f"Trial count tie: {status.tie_count}\n"
            f"Swapped: {swapped}\n"
            f"{'=' * 48}\n"
        )

    def log_judgement(self, status):
        self.log(
            f"\n{'=' * 48}\n"
            f"A Attempted: {status.a_attempted}\n"
            f"B Attempted: {status.b_attempted}\n"
            f"Reference File Issue: {status.reference_file_issue}\n"
            f"Success: {status.success}\n"
            f"Error Message: {status.error_message}\n"
            f"Trial Count A: {status.win_count_a}\n"
            f"Trial Count B: {status.win_count_b}\n"
            f"Trial Count Tie: {status.tie_count}\n"
            f"Request Failure Count: {status.request_failure_count}\n"
            f"Retryable Request Failure Count: {status.retryable_request_failure_count}\n"
            f"Request Retry Count: {status.request_retry_count}\n"
            f"Final Winner: {status.winner}\n"
            f"{'=' * 48}\n"
        )

    def log_error(self, error, task_id):
        self.log(
            f"\n{'=' * 48}\n"
            f"Error evaluating task (not included in tally).\n"
            f"Task ID: {task_id}\nError: {error}\n"
            f"{'=' * 48}\n"
        )

    def log_final_results(self, status, eval_elo, eval_normalized_elo, ref_name, eval_name):
        tc = status.task_count or 1
        self.log(
            f"\n{'=' * 48}\n"
            f"Reference Model: {ref_name}\n"
            f"Eval Model: {eval_name}\n"
            f"Reference Win Count: {status.win_count_a}\n"
            f"Eval Win Count: {status.win_count_b}\n"
            f"Task Count: {status.task_count}\n"
            f"Reference Win %: {status.win_count_a / tc:.4f}\n"
            f"Eval Win %: {status.win_count_b / tc:.4f}\n"
            f"Tie %: {status.tie_count / tc:.4f}\n"
            f"Total Request Failures: {status.request_failure_count}\n"
            f"Total Retryable Failures: {status.retryable_request_failure_count}\n"
            f"Total Retries: {status.request_retry_count}\n"
            f"Tasks with Failures: {status.tasks_with_request_failures}\n"
            f"Tasks with Retries: {status.tasks_with_retries}\n"
            f"Predicted Eval ELO: {eval_elo}\n"
            f"Predicted Eval Normalized ELO: {eval_normalized_elo}\n"
            f"{'=' * 48}\n"
        )


# ---------------------------------------------------------------------------
# JudgementStatus
# ---------------------------------------------------------------------------


@dataclass
class JudgementStatus:
    task_id: str
    task_index: int | None
    a_attempted: bool
    b_attempted: bool
    reference_file_issue: bool
    success: bool
    win_count_a: int = 0
    win_count_b: int = 0
    tie_count: int = 0
    task_count: int = 0
    request_failure_count: int = 0
    retryable_request_failure_count: int = 0
    request_retry_count: int = 0
    tasks_with_request_failures: int = 0
    tasks_with_retries: int = 0
    winner: str | None = None
    failure_reason: str | None = None
    failure_detail: str | None = None
    error_message: str | None = None

    def tally_result_from_judgement(self, judgement: str, swapped: bool = False):
        self.win_count_a, self.win_count_b, self.tie_count = tally_result(
            judgement, swapped, self.win_count_a, self.win_count_b, self.tie_count
        )
        self.task_count += 1

    def update_final_judgement(self):
        if self.win_count_a > self.win_count_b:
            self.winner = A_WIN_RESPONSE
        elif self.win_count_b > self.win_count_a:
            self.winner = B_WIN_RESPONSE
        else:
            self.winner = TIE_RESPONSE
        self.success = True

    def record_request_failure(self, retryable: bool):
        self.request_failure_count += 1
        if retryable:
            self.retryable_request_failure_count += 1

    def record_request_retry(self):
        self.request_retry_count += 1

    def accumulate_request_stats_from(self, other: JudgementStatus):
        self.request_failure_count += other.request_failure_count
        self.retryable_request_failure_count += other.retryable_request_failure_count
        self.request_retry_count += other.request_retry_count
        if other.request_failure_count > 0:
            self.tasks_with_request_failures += 1
        if other.request_retry_count > 0:
            self.tasks_with_retries += 1


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def _scan_completed_judgements(judgement_dir: str) -> dict[str, str]:
    """Scan existing judgement logs to find already-completed task_ids.

    Returns ``{task_id: winner}`` for tasks whose log contains a final winner.
    """
    import re

    completed: dict[str, str] = {}
    if not os.path.isdir(judgement_dir):
        return completed

    for filename in os.listdir(judgement_dir):
        m = re.match(r"judgements-(.+)\.log$", filename)
        if not m:
            continue
        task_id = m.group(1)
        log_path = os.path.join(judgement_dir, filename)
        try:
            content = Path(log_path).read_text(encoding="utf-8", errors="replace")
            # Look for "Final Winner:" line with a BOXED result
            for line in content.splitlines():
                line = line.strip()
                if "BOXED[" in line and ("Final Winner" in content):
                    if A_WIN_RESPONSE in line:
                        completed[task_id] = A_WIN_RESPONSE
                    elif B_WIN_RESPONSE in line:
                        completed[task_id] = B_WIN_RESPONSE
                    elif TIE_RESPONSE in line:
                        completed[task_id] = TIE_RESPONSE
        except Exception:
            pass

    return completed


class Judge:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.judge_model_name = args.judge_model_name
        self.max_output_tokens = args.max_output_tokens
        self.num_trials = args.num_trials
        self.judgement_dir = args.judgement_dir
        self.reference_model_dir = args.reference_model_dir
        self.eval_model_dir = args.eval_model_dir
        self.reference_model_name = args.reference_model_name
        self.eval_model_name = args.eval_model_name

        self._tls = threading.local()
        self.semaphore = threading.BoundedSemaphore(args.max_concurrent_judgements)
        self.executor = ThreadPoolExecutor(max_workers=args.max_concurrent_judgements)

    def _get_client(self) -> OpenAI:
        if not hasattr(self._tls, "client"):
            api_key = self.args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError("No API key. Set OPENAI_API_KEY or NVIDIA_API_KEY, or pass --api-key.")
            self._tls.client = OpenAI(base_url=self.args.server_address.rstrip("/"), api_key=api_key)
        return self._tls.client

    def _reference_file_issue(self, ref_dir_a, ref_dir_b) -> bool:
        if (ref_dir_a is not None) != (ref_dir_b is not None):
            return True
        if ref_dir_a is not None and ref_dir_b is not None:
            if not os.path.exists(ref_dir_a) or not os.path.exists(ref_dir_b):
                return True
        return False

    def _classify_failure_reason(self, status: JudgementStatus) -> str:
        if not status.a_attempted:
            return "reference_model_not_attempted"
        if not status.b_attempted:
            return "candidate_model_not_attempted"
        if status.reference_file_issue:
            return "reference_file_issue"
        error_text = (status.error_message or "").lower()
        if "permission denied" in error_text:
            return "unreadable_file_permission_denied"
        if any(m in error_text for m in ("502", "503", "504", "bad gateway", "gateway timeout")):
            return "judge_gateway_error"
        if status.request_failure_count > 0:
            return "other_request_error"
        return "other_task_error"

    def run_trials(self, task, task_dir_a, task_dir_b, refs_dir, logger, status, clean_up_list):
        task_id = task["task_id"]
        logger.log(f"Running {self.num_trials} trials for task {task_id}")

        refs = build_file_section(refs_dir, clean_up_list)
        submission_a = build_file_section(task_dir_a, clean_up_list)
        submission_b = build_file_section(task_dir_b, clean_up_list)

        for i in range(self.num_trials):
            trial_number = i + 1
            swapped = i % 2 != 0
            current_a = submission_b if swapped else submission_a
            current_b = submission_a if swapped else submission_b

            messages = construct_judge_messages(
                task_prompt=task["prompt"], refs=refs, submission_a=current_a, submission_b=current_b
            )
            logger.log(f"Trial {trial_number}/{self.num_trials}, swapped={swapped}")

            client = self._get_client()
            response_text = send_judge_request(client, self.judge_model_name, messages, self.max_output_tokens)
            status.tally_result_from_judgement(response_text, swapped=swapped)
            logger.log_trial_result(response_text, task_id, status, swapped)

        return clean_up_list, status

    def judge_task(self, task: dict, task_index: int) -> JudgementStatus:
        task_id = task["task_id"]
        log_name = f"{self.reference_model_name} vs {self.eval_model_name} | {task_id} | #{task_index}"

        logger = JudgementLogger(
            log_dir=self.judgement_dir, log_name=log_name, log_filename=f"judgements-{task_id}.log", overwrite=False
        )
        logger.log(f"Starting to evaluate task {task_id}")

        task_dir_a = os.path.join(self.reference_model_dir, f"task_{task_id}")
        task_dir_b = os.path.join(self.eval_model_dir, f"task_{task_id}")

        refs_dir_a = os.path.join(task_dir_a, "reference_files") if task.get("reference_file_urls") else None
        refs_dir_b = os.path.join(task_dir_b, "reference_files") if task.get("reference_file_urls") else None

        a_attempted = task_attempted(task_dir_a)
        b_attempted = task_attempted(task_dir_b)
        reference_file_issue = self._reference_file_issue(refs_dir_a, refs_dir_b)

        status = JudgementStatus(
            task_id=task_id,
            task_index=task_index,
            a_attempted=a_attempted,
            b_attempted=b_attempted,
            reference_file_issue=reference_file_issue,
            success=False,
        )

        if not a_attempted or not b_attempted or reference_file_issue:
            status.error_message = f"Task {task_id} not attempted by both models or reference file issue"
            status.failure_reason = self._classify_failure_reason(status)
            status.failure_detail = status.error_message
            return status

        self.semaphore.acquire()
        clean_up_list = []
        try:
            t0 = time.time()
            clean_up_list, status = self.run_trials(
                task=task,
                task_dir_a=task_dir_a,
                task_dir_b=task_dir_b,
                refs_dir=refs_dir_a,
                logger=logger,
                status=status,
                clean_up_list=clean_up_list,
            )
            status.update_final_judgement()
            logger.log_judgement(status)
            clean_up_paths(clean_up_list)
            logger.log(f"Trials took {(time.time() - t0) / 60.0:.3f} minutes")
            return status
        except Exception as e:
            logger.log_error(error=e, task_id=task_id)
            status.error_message = str(e)
            status.failure_reason = self._classify_failure_reason(status)
            status.failure_detail = status.error_message
            clean_up_paths(clean_up_list)
            return status
        finally:
            self.semaphore.release()

    def _write_postflight_reports(self, entries: list[dict], final_status: JudgementStatus, total_tasks: int):
        failed = [e for e in entries if not e["success"]]
        counts_by_reason: dict[str, int] = {}
        tasks_by_reason: dict[str, list[dict]] = {}
        for entry in failed:
            reason = entry["failure_reason"] or "unknown"
            counts_by_reason[reason] = counts_by_reason.get(reason, 0) + 1
            tasks_by_reason.setdefault(reason, []).append(entry)

        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "reference_model_name": self.reference_model_name,
            "eval_model_name": self.eval_model_name,
            "judgement_dir": self.judgement_dir,
            "total_tasks_considered": total_tasks,
            "successful_judgements": len(entries) - len(failed),
            "failed_judgements": len(failed),
            "counts_by_failure_reason": counts_by_reason,
            "total_request_failure_count": final_status.request_failure_count,
            "total_retryable_failures": final_status.retryable_request_failure_count,
            "total_retries": final_status.request_retry_count,
            "tasks_with_request_failures": final_status.tasks_with_request_failures,
            "tasks_with_retries": final_status.tasks_with_retries,
            "failed_tasks": failed,
        }

        json_path = Path(self.judgement_dir) / "postflight_report.json"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        md_lines = [
            "# Judge Postflight Report",
            "",
            f"- Generated at: {report['generated_at']}",
            f"- Reference Model: {self.reference_model_name}",
            f"- Eval Model: {self.eval_model_name}",
            f"- Judgement dir: `{self.judgement_dir}`",
            f"- Total tasks considered: {total_tasks}",
            f"- Successful judgements: {report['successful_judgements']}",
            f"- Failed judgements: {len(failed)}",
            "",
            "## Failure Reasons",
            "",
        ]
        if not counts_by_reason:
            md_lines.append("No failed judgements.")
        else:
            for reason in sorted(counts_by_reason):
                md_lines.append(f"- `{reason}`: {counts_by_reason[reason]}")
            for reason in sorted(tasks_by_reason):
                md_lines.extend(["", f"## {reason}", ""])
                for entry in tasks_by_reason[reason]:
                    md_lines.append(f"- `{entry['task_id']}` (index: {entry['task_index']})")
                    if entry.get("error_message"):
                        md_lines.append(f"  Error: {entry['error_message']}")

        md_path = Path(self.judgement_dir) / "postflight_report.md"
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    def evaluate(self):
        dataset = load_dataset("openai/gdpval", split="train")
        os.makedirs(self.judgement_dir, exist_ok=True)

        final_logger = JudgementLogger(
            log_dir=self.judgement_dir,
            log_name=f"{self.reference_model_name} vs {self.eval_model_name} | FINAL",
            log_filename="final_judgement.log",
            overwrite=False,
        )

        # Auto-resume: scan existing judgement logs for completed tasks
        already_done = _scan_completed_judgements(self.judgement_dir)
        resume_win_a = sum(1 for w in already_done.values() if w == A_WIN_RESPONSE)
        resume_win_b = sum(1 for w in already_done.values() if w == B_WIN_RESPONSE)
        resume_ties = sum(1 for w in already_done.values() if w == TIE_RESPONSE)

        all_tasks = [(idx, task) for idx, task in enumerate(dataset)]
        tasks = [(idx, task) for idx, task in all_tasks if task["task_id"] not in already_done]

        if already_done:
            print(
                f"[resume] Found {len(already_done)} completed tasks in {self.judgement_dir}/ "
                f"(A={resume_win_a}, B={resume_win_b}, ties={resume_ties}). "
                f"{len(tasks)} remaining.",
                flush=True,
            )
        final_logger.log(
            f"Judging {len(tasks)} tasks ({len(already_done)} already done, "
            f"A={resume_win_a}, B={resume_win_b}, ties={resume_ties})"
        )

        final_status = JudgementStatus(
            task_id="FINAL",
            task_index=None,
            a_attempted=True,
            b_attempted=True,
            reference_file_issue=False,
            success=False,
            win_count_a=resume_win_a,
            win_count_b=resume_win_b,
            tie_count=resume_ties,
            task_count=len(already_done),
        )

        t0 = time.time()
        futures = [self.executor.submit(self.judge_task, task, idx) for idx, task in tasks]

        completed = 0
        postflight_entries = []

        for future in as_completed(futures):
            result = future.result()
            if result.success and result.winner is not None:
                final_status.tally_result_from_judgement(result.winner)
            final_status.accumulate_request_stats_from(result)

            postflight_entries.append(
                {
                    "task_id": result.task_id,
                    "task_index": result.task_index,
                    "success": result.success,
                    "winner": result.winner,
                    "failure_reason": result.failure_reason,
                    "error_message": result.error_message,
                    "a_attempted": result.a_attempted,
                    "b_attempted": result.b_attempted,
                    "reference_file_issue": result.reference_file_issue,
                    "win_count_a": result.win_count_a,
                    "win_count_b": result.win_count_b,
                    "tie_count": result.tie_count,
                    "request_failure_count": result.request_failure_count,
                    "request_retry_count": result.request_retry_count,
                }
            )

            completed += 1
            final_logger.log(
                f"\n{'=' * 48}\n"
                f"Completed: {completed}/{len(tasks)}\n"
                f"Task {result.task_id}: success={result.success}, winner={result.winner}\n"
                f"{'=' * 48}\n"
            )

        final_status.update_final_judgement()

        if final_status.task_count == 0:
            eval_elo = eval_normalized_elo = None
            final_logger.log("No successful judgements — skipping ELO calculation.")
        else:
            win_rate = final_status.win_count_b / final_status.task_count
            eval_elo, eval_normalized_elo = calculate_elo(win_rate, self.args.reference_model_elo)

        final_logger.log_final_results(
            status=final_status,
            eval_elo=eval_elo,
            eval_normalized_elo=eval_normalized_elo,
            ref_name=self.reference_model_name,
            eval_name=self.eval_model_name,
        )

        total_time = (time.time() - t0) / 60.0
        final_logger.log(f"Total time: {total_time:.3f} minutes")

        self._write_postflight_reports(entries=postflight_entries, final_status=final_status, total_tasks=len(tasks))

        print(f"\nResults written to {self.judgement_dir}/")
        if eval_elo is not None:
            print(f"  Eval model win rate: {final_status.win_count_b / final_status.task_count:.4f}")
            print(f"  Eval model ELO: {eval_elo:.1f}")
            print(f"  Eval model normalized ELO: {eval_normalized_elo:.4f}")
        print(f"  Tasks judged: {final_status.task_count}")
        print(
            f"  Ref wins: {final_status.win_count_a}, Eval wins: {final_status.win_count_b}, Ties: {final_status.tie_count}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc pairwise comparison judge for GDPVal")
    parser.add_argument("--server-address", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--judge-model-name", type=str, default="gpt-4.1-2025-04-14")
    parser.add_argument("--api-key", type=str, default=None, help="API key (or set OPENAI_API_KEY / NVIDIA_API_KEY)")
    parser.add_argument("--reference-model-name", type=str, required=True)
    parser.add_argument("--eval-model-name", type=str, required=True)
    parser.add_argument("--reference-model-dir", type=str, required=True)
    parser.add_argument("--eval-model-dir", type=str, required=True)
    parser.add_argument("--reference-model-elo", type=int, default=1000)
    parser.add_argument("--max-output-tokens", type=int, default=65535)
    parser.add_argument("--num-trials", type=int, default=4)
    parser.add_argument("--max-concurrent-judgements", type=int, default=4)
    parser.add_argument("--judgement-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.judgement_dir is None:
        dir_name = f"{args.reference_model_name}-vs-{args.eval_model_name}-judgements"
        args.judgement_dir = os.path.join("output/gdpval/judgements", dir_name)
    judge = Judge(args)
    judge.evaluate()


if __name__ == "__main__":
    main()
