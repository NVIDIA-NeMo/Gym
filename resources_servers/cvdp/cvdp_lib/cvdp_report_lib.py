# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import json
import os
import subprocess
import sys

from cvdp_lib.cvdp_constants import (
    BLEU_SCORING_CATEGORIES,
    is_score_based_category,
)


def auto_generate_text_report(json_path: str) -> None:
    """
    Automatically generate a text report from a JSON file using run_reporter.py.

    Args:
        json_path: Path to the JSON report file
    """
    if not os.path.exists(json_path):
        print(f"Warning: JSON file {json_path} does not exist, skipping text report generation")
        return

    try:
        # Determine the output text file path
        json_dir = os.path.dirname(json_path)
        json_filename = os.path.basename(json_path)

        # Create text filename by replacing .json with .txt
        if json_filename.endswith(".json"):
            txt_filename = json_filename[:-5] + ".txt"
        else:
            txt_filename = json_filename + ".txt"

        txt_path = os.path.join(json_dir, txt_filename)

        # Find the cvdp_run_reporter.py script - co-located with this file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        reporter_script = os.path.join(script_dir, "cvdp_run_reporter.py")

        if not os.path.exists(reporter_script):
            print(f"Warning: run_reporter.py not found at {reporter_script}, skipping text report generation")
            return

        # Run the reporter script with output file option
        cmd = [sys.executable, reporter_script, json_path, "-o", txt_path]
        print(f"Generating text report: {txt_path}")

        # Add the cvdp server root to PYTHONPATH so cvdp_lib imports resolve
        env = os.environ.copy()
        cvdp_root = os.path.dirname(script_dir)
        env["PYTHONPATH"] = cvdp_root + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)

        print(f"Text report generated: {txt_path}")

    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to generate text report for {json_path}: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
    except Exception as e:
        print(f"Warning: Unexpected error generating text report for {json_path}: {e}")


class Report:
    def __init__(
        self,
        raw_logs=None,
        prefix="work",
        dataset_path=None,
        golden_mode=None,
        disable_patch=None,
        model_agent=None,
        force_agentic=None,
        force_agentic_include_golden=None,
        force_agentic_include_harness=None,
        force_copilot=None,
        copilot_refine=None,
    ):
        self.raw_logs = raw_logs
        self.min = float("inf")
        self.max = float("-inf")
        self.prefix = prefix
        self.dataset_path = dataset_path
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.golden_mode = golden_mode
        self.disable_patch = disable_patch
        self.model_agent = model_agent
        self.force_agentic = force_agentic
        self.force_agentic_include_golden = force_agentic_include_golden
        self.force_agentic_include_harness = force_agentic_include_harness
        self.force_copilot = force_copilot
        self.copilot_refine = copilot_refine

        if raw_logs != None:
            self.format_report()

    def read_report(self, filename):
        with open(filename, "r+") as f:
            report = f.readlines()

        self.categories = json.loads("".join(report))

        # Extract metadata if available
        if "metadata" in self.categories:
            self.dataset_path = self.categories["metadata"].get("dataset_path")
            self.timestamp = self.categories["metadata"].get("timestamp")
            self.golden_mode = self.categories["metadata"].get("golden_mode")
            self.disable_patch = self.categories["metadata"].get("disable_patch")
            self.model_agent = self.categories["metadata"].get("model_agent")
            self.force_agentic = self.categories["metadata"].get("force_agentic")
            self.force_agentic_include_golden = self.categories["metadata"].get("force_agentic_include_golden")
            self.force_agentic_include_harness = self.categories["metadata"].get("force_agentic_include_harness")
            self.force_copilot = self.categories["metadata"].get("force_copilot")
            self.copilot_refine = self.categories["metadata"].get("copilot_refine")

    def format_difficulty(self):
        all = {}
        all["hard"] = {}
        all["medium"] = {}
        all["easy"] = {}

        self.update_category(all["hard"])
        self.update_category(all["medium"])
        self.update_category(all["easy"])

        for cat in self.categories.values():
            # Skip non-category entries like metadata
            if not isinstance(cat, dict) or not all(diff in cat for diff in ["hard", "medium", "easy"]):
                continue

            for diff in ["hard", "medium", "easy"]:
                all[diff]["Passed Tests"] += cat[diff]["Passed Tests"]
                all[diff]["Failed Tests"] += cat[diff]["Failed Tests"]
                all[diff]["Total Tests"] += cat[diff]["Total Tests"]
                all[diff]["Passed Problems"] += cat[diff]["Passed Problems"]
                all[diff]["Failed Problems"] += cat[diff]["Failed Problems"]
                all[diff]["Total Problems"] += cat[diff]["Total Problems"]

                total_tests = all[diff]["Total Tests"]
                passed_tests = all[diff]["Passed Tests"]

                if total_tests != 0:
                    all[diff]["Passed Tests (%)"] = (passed_tests / total_tests) * 100
                else:
                    all[diff]["Passed Tests (%)"] = 0

                total_problems = all[diff]["Total Problems"]
                passed_problems = all[diff]["Passed Problems"]

                if total_problems != 0:
                    all[diff]["Passed Problems (%)"] = (passed_problems / total_problems) * 100
                else:
                    all[diff]["Passed Problems (%)"] = 0

        return all

    def report_categories(self):
        # Create prefix directory if it doesn't exist
        os.makedirs(self.prefix, exist_ok=True)

        # Write report to prefix directory
        report_path = os.path.join(self.prefix, "report.json")
        with open(report_path, "w+") as f:
            f.write(json.dumps(self.categories))

        # Automatically generate text report
        auto_generate_text_report(report_path)

    def report_difficulty(self):
        print(self.format_difficulty())

    def report_timers(self):
        print(f"Benchmark execution time: Min: {self.min} - Max: {self.max} - Avg: {self.avg}")

    def report_header(self):
        """Print header information including dataset path, timestamp, and run configuration."""
        print("\n=== Benchmark Report ===")
        if self.dataset_path:
            print(f"Dataset: {self.dataset_path}")
        if self.timestamp:
            print(f"Generated: {self.timestamp}")

        # Print run configuration information
        print("\n=== Run Configuration ===")
        if self.golden_mode is not None:
            print(f"Golden Mode: {'Yes' if self.golden_mode else 'No'}")
        if self.disable_patch is not None:
            print(f"Patches Disabled: {'Yes' if self.disable_patch else 'No'}")
        if self.force_agentic is not None:
            print(f"Force Agentic: {'Yes' if self.force_agentic else 'No'}")
        if self.force_copilot is not None:
            print(f"Force Copilot: {'Yes' if self.force_copilot else 'No'}")
        if self.force_agentic_include_golden is not None:
            print(f"Include Golden Patch: {'Yes' if self.force_agentic_include_golden else 'No'}")
        if self.force_agentic_include_harness is not None:
            print(f"Include Harness: {'Yes' if self.force_agentic_include_harness else 'No'}")
        if self.copilot_refine is not None:
            print(f"Copilot Refine Model: {self.copilot_refine}")
        if self.golden_mode is False and self.model_agent:
            print(f"Model/Agent: {self.model_agent}")

        print("=======================\n")

    def update_category(self, category):
        category["Passed Tests"] = 0
        category["Failed Tests"] = 0
        category["Total Tests"] = 0
        category["Passed Tests (%)"] = 0
        category["Passed Problems"] = 0
        category["Failed Problems"] = 0
        category["Total Problems"] = 0
        category["Passed Problems (%)"] = 0

    def format_report(self):
        self.categories = {}
        total_exec_time = 0
        tests = 0
        problem_results = {}  # Track results per problem ID
        failing_tests = []  # Track all failing tests
        passing_tests = []  # Track all passing tests
        scores_by_problem = {}  # Track scores per problem for score-based categories (BLEU, LLM, etc.)

        for id, report in self.raw_logs.items():
            category = report["category"]
            diff = report["difficulty"]

            if category not in self.categories:
                self.categories[category] = {}
                self.categories[category]["easy"] = {}
                self.categories[category]["medium"] = {}
                self.categories[category]["hard"] = {}
                self.categories[category]["logs"] = []

                self.update_category(self.categories[category]["easy"])
                self.update_category(self.categories[category]["medium"])
                self.update_category(self.categories[category]["hard"])

            # ----------------------------------------
            # - Update category report
            # ----------------------------------------

            # Initialize problem tracking for this ID
            if id not in problem_results:
                problem_results[id] = {"category": category, "difficulty": diff, "all_tests_pass": True}

            # Extract BLEU scores for BLEU scoring categories
            category_num = None
            try:
                if category.startswith("cid"):
                    category_num = int(category[3:])
                elif category.isdigit():
                    category_num = int(category)
                else:
                    # Try to extract numeric part from the end if it's a mixed format
                    import re

                    match = re.search(r"(\d+)$", category)
                    if match:
                        category_num = int(match.group(1))
            except (ValueError, AttributeError):
                pass

            for test_idx, test in enumerate(report["tests"]):
                exec = test["execution"]

                # Collect scores for score-based categories (BLEU, LLM, etc.)
                if category_num is not None and is_score_based_category(category_num):
                    score_value = None

                    # Collect BLEU scores for BLEU categories
                    if "bleu_score" in test:
                        score_value = test["bleu_score"]
                    # Collect LLM scores for LLM categories (normalized to 0-1)
                    elif "llm_score" in test:
                        score_value = test["llm_score"]
                    # Other score types can be added here in the future

                    if score_value is not None:
                        if id not in scores_by_problem:
                            scores_by_problem[id] = []
                        scores_by_problem[id].append(score_value)

                if test["result"] == 0:
                    self.categories[category][diff]["Passed Tests"] += 1
                    # Add to passing tests list
                    passing_tests.append(
                        {
                            "test_id": id,
                            "category": category,
                            "difficulty": diff,
                            "test_index": test_idx,
                            "log": test.get("log"),
                        }
                    )
                else:
                    self.categories[category][diff]["Failed Tests"] += 1
                    # Mark this problem as failed if any test fails
                    problem_results[id]["all_tests_pass"] = False
                    # Add to failing tests list with error info
                    failing_tests.append(
                        {
                            "test_id": id,
                            "category": category,
                            "difficulty": diff,
                            "test_index": test_idx,
                            "error_msg": test.get("error_msg"),
                            "agent_error": test.get("agent_error", None),
                            "log": test.get("log"),
                        }
                    )

                self.categories[category][diff]["Total Tests"] += 1

                # Should be changed from original folder too
                self.categories[category]["logs"].append({"id": id, "log": test["log"]})

                total = self.categories[category][diff]["Total Tests"]
                passd = self.categories[category][diff]["Passed Tests"]

                self.categories[category][diff]["Passed Tests (%)"] = (passd / total) * 100

                # ----------------------------------------
                # - Calculate average execution time
                # ----------------------------------------

                if exec < self.min:
                    self.min = exec
                if exec > self.max:
                    self.max = exec

                total_exec_time += exec
                tests += 1

        # Update problem statistics after all tests are processed
        for id, result in problem_results.items():
            category = result["category"]
            diff = result["difficulty"]

            # Get category number for BLEU scoring logic
            category_num = None
            try:
                if category.startswith("cid"):
                    category_num = int(category[3:])
                elif category.isdigit():
                    category_num = int(category)
                else:
                    # Try to extract numeric part from the end if it's a mixed format
                    import re

                    match = re.search(r"(\d+)$", category)
                    if match:
                        category_num = int(match.group(1))
            except (ValueError, AttributeError):
                pass

            # Increment total problems
            self.categories[category][diff]["Total Problems"] += 1

            # Use configurable scoring based on category mode
            if category_num is not None and is_score_based_category(category_num):
                # Score-based scoring: use average score (0-1) as fractional pass rate
                if id in scores_by_problem:
                    scores = scores_by_problem[id]
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        # Store the average score as the problem "pass" score
                        self.categories[category][diff]["Passed Problems"] += avg_score  # Fractional score
                        self.categories[category][diff]["Failed Problems"] += 1 - avg_score  # Fractional failure
                    else:
                        # No scores found, treat as failed
                        self.categories[category][diff]["Failed Problems"] += 1
                else:
                    # No scores available for score-based category, treat as failed
                    self.categories[category][diff]["Failed Problems"] += 1
            else:
                # Traditional threshold-based binary pass/fail logic
                if result["all_tests_pass"]:
                    self.categories[category][diff]["Passed Problems"] += 1
                else:
                    self.categories[category][diff]["Failed Problems"] += 1

            # Calculate pass percentage for problems
            total_problems = self.categories[category][diff]["Total Problems"]
            passed_problems = self.categories[category][diff]["Passed Problems"]

            if total_problems > 0:
                self.categories[category][diff]["Passed Problems (%)"] = (passed_problems / total_problems) * 100
            else:
                self.categories[category][diff]["Passed Problems (%)"] = 0

        # Store scores in the raw result data for future reference
        if scores_by_problem:
            for id, scores in scores_by_problem.items():
                if id in self.raw_logs:
                    category = self.raw_logs[id]["category"]
                    category_num = None
                    try:
                        if category.startswith("cid"):
                            category_num = int(category[3:])
                        elif category.isdigit():
                            category_num = int(category)
                        else:
                            import re

                            match = re.search(r"(\d+)$", category)
                            if match:
                                category_num = int(match.group(1))
                    except (ValueError, AttributeError):
                        pass

                    # Store different score types based on category
                    if category_num in BLEU_SCORING_CATEGORIES:
                        self.raw_logs[id]["bleu_scores"] = scores
                        self.raw_logs[id]["avg_bleu_score"] = sum(scores) / len(scores) if scores else 0
                    else:
                        # For LLM or other score types
                        self.raw_logs[id]["llm_scores"] = scores
                        self.raw_logs[id]["avg_llm_score"] = sum(scores) / len(scores) if scores else 0

        # Add metadata to the report
        self.categories["metadata"] = {
            "dataset_path": self.dataset_path,
            "timestamp": self.timestamp,
            "golden_mode": self.golden_mode,
            "disable_patch": self.disable_patch,
            "model_agent": self.model_agent,
            "force_agentic": self.force_agentic,
            "force_agentic_include_golden": self.force_agentic_include_golden,
            "force_agentic_include_harness": self.force_agentic_include_harness,
            "force_copilot": self.force_copilot,
            "copilot_refine": self.copilot_refine,
        }

        # Add failing and passing tests to the report
        self.categories["test_details"] = {"failing_tests": failing_tests, "passing_tests": passing_tests}

        # Average after loop
        self.avg = total_exec_time / tests


if __name__ == "__main__":
    # Parse Creation
    parser = argparse.ArgumentParser(description="Parser for report evaluation.")
    parser.add_argument(
        "-f", "--filename", required=True, type=str, help="Identify one file to run the harness evaluation."
    )

    # Arg Parsing
    args = parser.parse_args()
    filename = args.filename
    rpt = Report()
    rpt.read_report(filename)
    rpt.report_header()  # Add header with metadata
    rpt.report_difficulty()
