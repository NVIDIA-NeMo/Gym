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
"""Post-hoc standalone rubric scoring for GDPVal evaluation.

Scores a SINGLE model's outputs against the GDPVal rubric using an LLM judge.
This is the rubric analog of ``scripts/compare_elo.py`` (which does pairwise
comparison).


Usage:
    python scripts/run_rubric_judge.py \\
        --model-dir output/gdpval/NVIDIA-Nemotron-Super-V3 \\
        --model-name Nemotron-Super-V3 \\
        --server-address https://api.openai.com/v1 \\
        --judge-model-name gpt-4.1-2025-04-14 \\
        --api-key $JUDGE_API_KEY \\
        --num-trials 2 \\
        --max-concurrent-judgements 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI

# ---------------------------------------------------------------------------
# Import shared file handling from _gdpval_comparison.py without pulling in
# ray/stirrup via the package __init__.py (same trick as compare_elo.py).
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_comp_path = Path(__file__).resolve().parent.parent / "responses_api_agents/stirrup_agent/tasks/_gdpval_comparison.py"
_spec = importlib.util.spec_from_file_location("_gdpval_comparison", _comp_path)
_comp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_comp)

build_file_section = _comp.build_file_section
task_attempted = _comp.task_attempted
clean_up_paths = _comp.clean_up_paths
send_judge_request = _comp.send_judge_request

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUBRIC_JUDGE_PROMPT = (
    "Given a task description, reference files, an evaluation rubric, and submission file(s) "
    "for the task-- score the submission file(s) according to the rubric. Make sure the final "
    "overall score doesn't exceed the maximum score possible according to the points possible "
    "for each criterion and the sum of those points. For each criterion, give an explanation "
    "for the number of points you awarded. Then, list your awarded points in the format: "
    "'CRITERION_NUMBER[criterion_number]: GRADE[numeric_grade] out of "
    "MAX_POSSIBLE_POINTS[max_possible_points]'. Lastly, give your final overall score in the "
    "format: 'FINAL_SCORE[final_score] out of MAX_POSSIBLE_SCORE[max_possible_score]' Each "
    "value must be surrounded by the appropriate tag with square brackets [] around each number "
    "as described above. Double check that there are no math errors in any of your score "
    "calculations."
)

TASK_TEMPLATE = "<TASK_DESCRIPTION_START>\n{task}\n<TASK_DESCRIPTION_END>\n\n"
REFERENCES_OPEN = "<REFERENCES_FILES_START>\n"
REFERENCES_CLOSE = "\n<REFERENCES_FILES_END>\n\n"
SUBMISSION_OPEN = "<SUBMISSION_FILES_START>\n"
SUBMISSION_CLOSE = "\n<SUBMISSION_FILES_END>\n\n"
RUBRIC_OPEN = "<RUBRIC_START>\n"
RUBRIC_CLOSE = "\n<RUBRIC_END>\n\n"

FINAL_SCORE_RE = re.compile(r"FINAL_SCORE\[(\d+(?:\.\d+)?)\]")
MAX_POSSIBLE_SCORE_RE = re.compile(r"MAX_POSSIBLE_SCORE\[(\d+(?:\.\d+)?)\]")


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


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------


def parse_rubric_score(response_text: str) -> tuple[float | None, float | None]:
    """Extract FINAL_SCORE[x] and MAX_POSSIBLE_SCORE[y] from judge response.

    Returns (final_score, max_possible_score) or (None, None) if not parseable.
    """
    final_match = FINAL_SCORE_RE.search(response_text)
    max_match = MAX_POSSIBLE_SCORE_RE.search(response_text)
    if final_match and max_match:
        return float(final_match.group(1)), float(max_match.group(1))
    return None, None


def compute_max_from_rubric_json(rubric_json_str: str) -> float:
    """Compute expected max score from the rubric JSON."""
    rubric_items = json.loads(rubric_json_str)
    return sum(item["score"] for item in rubric_items)


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------


def construct_rubric_judge_messages(
    task_prompt: str,
    refs: list[dict],
    submission: list[dict],
    rubric_text: str,
) -> list[dict]:
    """Assemble OpenAI messages for rubric scoring: prompt + task + refs + submission + rubric."""
    content: list[dict] = []
    content.append({"type": "text", "text": RUBRIC_JUDGE_PROMPT + "\n\n" + TASK_TEMPLATE.format(task=task_prompt)})
    content.append({"type": "text", "text": REFERENCES_OPEN})
    content.extend(refs)
    content.append({"type": "text", "text": REFERENCES_CLOSE})
    content.append({"type": "text", "text": SUBMISSION_OPEN})
    content.extend(submission)
    content.append({"type": "text", "text": SUBMISSION_CLOSE})
    content.append({"type": "text", "text": RUBRIC_OPEN + rubric_text + RUBRIC_CLOSE})
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# ScoringStatus
# ---------------------------------------------------------------------------


@dataclass
class ScoringStatus:
    task_id: str
    task_index: int | None
    attempted: bool
    success: bool
    scores: list[float] = field(default_factory=list)
    max_possible_score: float = 0.0
    average_score: float = 0.0
    normalized_score: float = 0.0
    num_trials_completed: int = 0
    formatting_failures: int = 0
    error_message: str | None = None
    failure_reason: str | None = None


# ---------------------------------------------------------------------------
# Auto-resume
# ---------------------------------------------------------------------------


def _scan_completed_scores(judgement_dir: str) -> dict[str, dict]:
    """Scan existing score-summary-*.json files to find completed tasks.

    Returns {task_id: summary_dict} for tasks with a valid score summary.
    """
    completed: dict[str, dict] = {}
    if not os.path.isdir(judgement_dir):
        return completed

    for filename in os.listdir(judgement_dir):
        m = re.match(r"score-summary-(.+)\.json$", filename)
        if not m:
            continue
        task_id = m.group(1)
        fpath = os.path.join(judgement_dir, filename)
        try:
            data = json.loads(Path(fpath).read_text(encoding="utf-8"))
            if data.get("success"):
                completed[task_id] = data
        except Exception:
            pass

    return completed


# ---------------------------------------------------------------------------
# RubricJudge
# ---------------------------------------------------------------------------


class RubricJudge:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.judge_model_name = args.judge_model_name
        self.max_output_tokens = args.max_output_tokens
        self.num_trials = args.num_trials
        self.formatting_retries = args.formatting_retries
        self.model_dir = args.model_dir
        self.model_name = args.model_name
        self.judgement_dir = args.judgement_dir

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

    def _score_single_trial(
        self,
        task: dict,
        messages: list[dict],
        expected_max: float,
        logger: Logger,
        trial_number: int,
    ) -> float | None:
        """Run a single scoring trial with formatting retries.

        Returns the final_score if successfully parsed and validated, else None.
        """
        client = self._get_client()

        for attempt in range(1, self.formatting_retries + 1):
            response_text = send_judge_request(client, self.judge_model_name, messages, self.max_output_tokens)
            logger.log(f"Trial {trial_number}, attempt {attempt}: response length={len(response_text)}")

            final_score, parsed_max = parse_rubric_score(response_text)
            if final_score is None or parsed_max is None:
                logger.log(f"Trial {trial_number}, attempt {attempt}: could not parse scores from response")
                continue

            # Validate that the parsed max matches the rubric's computed max
            if abs(parsed_max - expected_max) > 0.01:
                logger.log(
                    f"Trial {trial_number}, attempt {attempt}: "
                    f"parsed max ({parsed_max}) != expected max ({expected_max}), "
                    f"using response anyway"
                )

            if final_score > expected_max:
                logger.log(
                    f"Trial {trial_number}, attempt {attempt}: "
                    f"final_score ({final_score}) > expected max ({expected_max}), clamping"
                )
                final_score = expected_max

            logger.log(f"Trial {trial_number}: score={final_score}/{expected_max}")
            return final_score

        logger.log(f"Trial {trial_number}: all {self.formatting_retries} formatting attempts failed")
        return None

    def judge_task(self, task: dict, task_index: int) -> ScoringStatus:
        task_id = task["task_id"]
        log_name = f"{self.model_name} | {task_id} | #{task_index}"

        logger = Logger(
            log_dir=self.judgement_dir, log_name=log_name, log_filename=f"judgements-{task_id}.log", overwrite=False
        )
        logger.log(f"Starting rubric scoring for task {task_id}")

        task_dir = os.path.join(self.model_dir, f"task_{task_id}")
        refs_dir = os.path.join(task_dir, "reference_files") if task.get("reference_file_urls") else None
        attempted = task_attempted(task_dir)

        status = ScoringStatus(task_id=task_id, task_index=task_index, attempted=attempted, success=False)

        if not attempted:
            status.error_message = f"Task {task_id} not attempted"
            status.failure_reason = "not_attempted"
            logger.log(status.error_message)
            return status

        # Compute expected max from rubric JSON
        rubric_json_str = task.get("rubric_json", "[]")
        try:
            expected_max = compute_max_from_rubric_json(rubric_json_str)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            status.error_message = f"Failed to parse rubric_json: {e}"
            status.failure_reason = "rubric_parse_error"
            logger.log(status.error_message)
            return status

        status.max_possible_score = expected_max
        rubric_text = task.get("rubric_pretty", rubric_json_str)

        self.semaphore.acquire()
        clean_up_list = []
        try:
            t0 = time.time()

            refs = build_file_section(refs_dir, clean_up_list)
            submission = build_file_section(task_dir, clean_up_list)

            messages = construct_rubric_judge_messages(
                task_prompt=task["prompt"],
                refs=refs,
                submission=submission,
                rubric_text=rubric_text,
            )

            for trial_num in range(1, self.num_trials + 1):
                score = self._score_single_trial(task, messages, expected_max, logger, trial_num)
                if score is not None:
                    status.scores.append(score)
                    status.num_trials_completed += 1
                else:
                    status.formatting_failures += 1

            if status.scores:
                status.average_score = sum(status.scores) / len(status.scores)
                status.normalized_score = status.average_score / expected_max if expected_max > 0 else 0.0
                status.success = True
                logger.log(
                    f"Completed: avg={status.average_score:.2f}/{expected_max} "
                    f"({status.normalized_score:.4f}), trials={status.num_trials_completed}"
                )
            else:
                status.error_message = "All trials failed to produce parseable scores"
                status.failure_reason = "all_trials_failed"
                logger.log(status.error_message)

            clean_up_paths(clean_up_list)
            logger.log(f"Task took {(time.time() - t0) / 60.0:.3f} minutes")

            # Write per-task score summary
            summary = {
                "task_id": task_id,
                "task_index": task_index,
                "success": status.success,
                "scores": status.scores,
                "average_score": status.average_score,
                "max_possible_score": status.max_possible_score,
                "normalized_score": status.normalized_score,
                "num_trials_completed": status.num_trials_completed,
                "formatting_failures": status.formatting_failures,
                "error_message": status.error_message,
            }
            summary_path = os.path.join(self.judgement_dir, f"score-summary-{task_id}.json")
            Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")

            return status

        except Exception as e:
            logger.log(f"Error evaluating task {task_id}: {e}")
            status.error_message = str(e)
            status.failure_reason = "exception"
            clean_up_paths(clean_up_list)
            return status
        finally:
            self.semaphore.release()

    def evaluate(self):
        dataset = load_dataset("openai/gdpval", split="train")
        os.makedirs(self.judgement_dir, exist_ok=True)

        final_logger = Logger(
            log_dir=self.judgement_dir,
            log_name=f"{self.model_name} | FINAL",
            log_filename="final_judgement.log",
            overwrite=False,
        )

        # Auto-resume: scan existing score summaries
        already_done = _scan_completed_scores(self.judgement_dir)

        all_tasks = [(idx, task) for idx, task in enumerate(dataset)]
        tasks = [(idx, task) for idx, task in all_tasks if task["task_id"] not in already_done]

        if already_done:
            resumed_scores = [d["normalized_score"] for d in already_done.values() if d.get("normalized_score")]
            avg_resumed = sum(resumed_scores) / len(resumed_scores) if resumed_scores else 0.0
            print(
                f"[resume] Found {len(already_done)} completed tasks in {self.judgement_dir}/ "
                f"(avg normalized={avg_resumed:.4f}). {len(tasks)} remaining.",
                flush=True,
            )
            final_logger.log(
                f"Resuming: {len(already_done)} already done (avg normalized={avg_resumed:.4f}), "
                f"{len(tasks)} remaining"
            )

        t0 = time.time()
        futures = [self.executor.submit(self.judge_task, task, idx) for idx, task in tasks]

        completed_count = 0
        new_results: list[ScoringStatus] = []

        for future in as_completed(futures):
            result = future.result()
            new_results.append(result)
            completed_count += 1
            final_logger.log(
                f"Completed: {completed_count}/{len(tasks)} | "
                f"task={result.task_id}, success={result.success}, "
                f"score={result.average_score:.2f}/{result.max_possible_score}"
            )

        total_time = (time.time() - t0) / 60.0
        final_logger.log(f"Total time: {total_time:.3f} minutes")

        # Combine resumed + new results for final outputs
        all_entries = []

        # Add resumed entries
        for task_id, summary in already_done.items():
            all_entries.append(summary)

        # Add new entries
        for result in new_results:
            all_entries.append(
                {
                    "task_id": result.task_id,
                    "task_index": result.task_index,
                    "success": result.success,
                    "scores": result.scores,
                    "average_score": result.average_score,
                    "max_possible_score": result.max_possible_score,
                    "normalized_score": result.normalized_score,
                    "num_trials_completed": result.num_trials_completed,
                    "formatting_failures": result.formatting_failures,
                    "error_message": result.error_message,
                    "failure_reason": result.failure_reason,
                }
            )

        # Write score_summary.jsonl
        jsonl_path = os.path.join(self.judgement_dir, "score_summary.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for entry in all_entries:
                f.write(json.dumps(entry) + "\n")

        # Compute aggregate stats
        successful = [e for e in all_entries if e.get("success")]
        failed = [e for e in all_entries if not e.get("success")]

        if successful:
            avg_normalized = sum(e["normalized_score"] for e in successful) / len(successful)
            avg_raw = sum(e["average_score"] for e in successful) / len(successful)
            total_max = sum(e["max_possible_score"] for e in successful)
            total_scored = sum(e["average_score"] for e in successful)
        else:
            avg_normalized = avg_raw = total_max = total_scored = 0.0

        # Write postflight report
        failure_reasons: dict[str, int] = {}
        for e in failed:
            reason = e.get("failure_reason", "unknown")
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "model_name": self.model_name,
            "model_dir": self.model_dir,
            "judgement_dir": self.judgement_dir,
            "judge_model": self.judge_model_name,
            "num_trials": self.num_trials,
            "formatting_retries": self.formatting_retries,
            "total_tasks": len(all_entries),
            "successful": len(successful),
            "failed": len(failed),
            "average_normalized_score": avg_normalized,
            "average_raw_score": avg_raw,
            "total_scored": total_scored,
            "total_max_possible": total_max,
            "failure_reasons": failure_reasons,
            "failed_tasks": [
                {"task_id": e["task_id"], "reason": e.get("failure_reason"), "error": e.get("error_message")}
                for e in failed
            ],
        }

        report_path = os.path.join(self.judgement_dir, "postflight_report.json")
        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")

        # Print summary
        print(f"\nResults written to {self.judgement_dir}/")
        print(f"  Model: {self.model_name}")
        print(f"  Tasks scored: {len(successful)}/{len(all_entries)}")
        if successful:
            print(f"  Average normalized score: {avg_normalized:.4f}")
            print(f"  Average raw score: {avg_raw:.2f}")
            print(f"  Total: {total_scored:.1f}/{total_max:.1f}")
        if failed:
            print(f"  Failed tasks: {len(failed)}")
            for reason, count in sorted(failure_reasons.items()):
                print(f"    {reason}: {count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc standalone rubric scoring for GDPVal")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to model output dir with task_xxx/ subdirs")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--server-address", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--judge-model-name", type=str, default="gpt-4.1-2025-04-14")
    parser.add_argument("--api-key", type=str, default=None, help="API key (or set OPENAI_API_KEY / NVIDIA_API_KEY)")
    parser.add_argument("--max-output-tokens", type=int, default=65535)
    parser.add_argument("--num-trials", type=int, default=2)
    parser.add_argument("--formatting-retries", type=int, default=3)
    parser.add_argument("--max-concurrent-judgements", type=int, default=4)
    parser.add_argument("--judgement-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.judgement_dir is None:
        args.judgement_dir = os.path.join(args.model_dir, "rubric-judgements")
    judge = RubricJudge(args)
    judge.evaluate()


if __name__ == "__main__":
    main()
