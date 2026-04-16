#!/usr/bin/env python3

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

import json
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List

from tabulate import tabulate

from cvdp_lib.cvdp_constants import (
    is_score_based_category,
)


def extract_category_number(category_id: str) -> int:
    """
    Extract the numeric category ID from various category string formats.

    Args:
        category_id: Category string in formats like 'cid6', '6', 'category6', etc.

    Returns:
        int: The numeric category ID

    Raises:
        ValueError: If the category format is invalid or unrecognized
    """
    if category_id.startswith("cid"):
        try:
            return int(category_id[3:])
        except ValueError:
            raise ValueError(
                f"Invalid category format: {category_id}. Expected 'cid<number>' but got malformed number."
            )
    elif category_id.isdigit():
        return int(category_id)
    else:
        # Try to extract numeric part from the end if it's a mixed format
        import re

        match = re.search(r"(\d+)$", category_id)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(
                f"Unknown category format: {category_id}. Expected 'cid<number>', '<number>', or '<text><number>' format."
            )


def extract_problem_id_from_test_id(test_id: str) -> str:
    """
    Extract problem ID from test ID, handling dots in problem IDs.

    Test IDs are expected to be in format: problem_id.test_suffix
    where problem_id may contain dots (e.g., cvdp_copilot_H.264_encoder_0020.test1)

    Args:
        test_id: Test ID string

    Returns:
        str: The problem ID portion
    """
    if "." not in test_id:
        return test_id

    # Split from right side only once - everything before the last dot is the problem ID
    return test_id.rsplit(".", 1)[0]


def is_category_score_based(category_id: str) -> bool:
    """
    Check if a category uses score-based scoring (0-1 fractional pass rates) instead of binary pass/fail.

    Args:
        category_id: Category string in formats like 'cid6', '6', 'category6', etc.

    Returns:
        bool: True if the category uses score-based scoring, False otherwise

    Raises:
        ValueError: If the category format is invalid or unrecognized
    """
    category_num = extract_category_number(category_id)
    return is_score_based_category(category_num)


class DifficultyStats:
    """Statistics for a specific difficulty level."""

    def __init__(
        self,
        passed_tests=0,
        failed_tests=0,
        total_tests=0,
        test_pass_percentage=0.0,
        passed_problems=0,
        failed_problems=0,
        total_problems=0,
        problem_pass_percentage=0.0,
    ):
        self.passed_tests = passed_tests
        self.failed_tests = failed_tests
        self.total_tests = total_tests
        self.test_pass_percentage = test_pass_percentage

        # For problems
        self.passed_problems = passed_problems
        self.failed_problems = failed_problems
        self.total_problems = total_problems
        self.problem_pass_percentage = problem_pass_percentage


class CategoryStats:
    """Statistics for a specific category."""

    def __init__(self, category_name):
        self.category_name = category_name
        self.easy = DifficultyStats()
        self.medium = DifficultyStats()
        self.hard = DifficultyStats()

        # Aggregated stats across all difficulties
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_tests = 0
        self.test_pass_percentage = 0.0

        # For problems
        self.passed_problems = 0
        self.failed_problems = 0
        self.total_problems = 0
        self.problem_pass_percentage = 0.0

        # Total stats needed by the parser
        self.total_passed_tests = 0
        self.total_failed_tests = 0
        self.total_passed_problems = 0
        self.total_failed_problems = 0
        self.overall_test_pass_percentage = 0.0
        self.overall_problem_pass_percentage = 0.0


class ResultParser:
    def __init__(self, json_file_path: str):
        """Initialize the parser with a path to the JSON results file."""
        self.json_file_path = json_file_path
        # Use a regular dictionary instead of defaultdict
        self.categories: Dict[str, CategoryStats] = {}
        self.failing_tests = []
        self.passing_tests = []
        self.failing_problems = []
        self.passing_problems = []
        self.dataset_path = None
        self.timestamp = None
        self.disable_patch = None
        self.golden_mode = None
        self.model_agent = None
        self.is_composite = False
        self.n_samples = 0
        self.k_threshold = 0
        self.sample_prefixes = []

    def load_results(self) -> None:
        """Load and parse the JSON results file."""
        if not os.path.exists(self.json_file_path):
            raise FileNotFoundError(f"Results file not found: {self.json_file_path}")

        with open(self.json_file_path, "r") as f:
            self.raw_results = json.load(f)

        # Extract metadata if available
        if "metadata" in self.raw_results:
            self.dataset_path = self.raw_results["metadata"].get("dataset_path")
            self.timestamp = self.raw_results["metadata"].get("timestamp")
            self.disable_patch = self.raw_results["metadata"].get("disable_patch")
            self.golden_mode = self.raw_results["metadata"].get("golden_mode")
            self.model_agent = self.raw_results["metadata"].get("model_agent")

            # Check if this is a composite report
            self.is_composite = self.raw_results["metadata"].get("composite", False)
            if self.is_composite:
                self.n_samples = self.raw_results["metadata"].get("n_samples", 0)
                self.k_threshold = self.raw_results["metadata"].get("k_threshold", 1)
                self.sample_prefixes = self.raw_results["metadata"].get("sample_prefixes", [])

        # Extract test detail information if available
        if "test_details" in self.raw_results and not self.is_composite:
            self.failing_tests = self.raw_results["test_details"].get("failing_tests", [])
            self.passing_tests = self.raw_results["test_details"].get("passing_tests", [])

    def parse_results(self) -> None:
        """Parse the results and organize them by category and difficulty."""
        if self.is_composite:
            self._parse_composite_results()
        else:
            self._parse_standard_results()

    def _parse_standard_results(self) -> None:
        """Parse standard (non-composite) results."""
        for cid, results in self.raw_results.items():
            # Skip the test_details and metadata sections - they're not categories
            if cid in ["test_details", "metadata"]:
                continue

            # Create CategoryStats if it doesn't exist yet
            if cid not in self.categories:
                self.categories[cid] = CategoryStats(cid)

            category_stats = self.categories[cid]

            for difficulty in ["easy", "medium", "hard"]:
                if difficulty in results:
                    stats = DifficultyStats(
                        passed_tests=results[difficulty]["Passed Tests"],
                        failed_tests=results[difficulty]["Failed Tests"],
                        total_tests=results[difficulty]["Total Tests"],
                        test_pass_percentage=results[difficulty]["Passed Tests (%)"],
                        passed_problems=results[difficulty]["Passed Problems"],
                        failed_problems=results[difficulty]["Failed Problems"],
                        total_problems=results[difficulty]["Total Problems"],
                        problem_pass_percentage=results[difficulty]["Passed Problems (%)"],
                    )

                    # Update category-specific stats
                    setattr(category_stats, difficulty, stats)

                    # Update category totals for tests
                    category_stats.total_passed_tests += stats.passed_tests
                    category_stats.total_failed_tests += stats.failed_tests
                    category_stats.total_tests += stats.total_tests

                    # Update category totals for problems
                    category_stats.total_passed_problems += stats.passed_problems
                    category_stats.total_failed_problems += stats.failed_problems
                    category_stats.total_problems += stats.total_problems

            # Calculate overall pass percentage for the category tests
            if category_stats.total_tests > 0:
                category_stats.overall_test_pass_percentage = (
                    category_stats.total_passed_tests / category_stats.total_tests
                ) * 100

            # Calculate overall pass percentage for the category problems
            if category_stats.total_problems > 0:
                category_stats.overall_problem_pass_percentage = (
                    category_stats.total_passed_problems / category_stats.total_problems
                ) * 100

    def _parse_composite_results(self) -> None:
        """Parse composite results with pass@k metrics."""
        # If pass@k was already calculated, use that
        if "pass_at_k" in self.raw_results:
            self._parse_existing_pass_at_k()
            return

        # Otherwise, calculate it from the samples
        self._calculate_pass_at_k_from_samples()

    def _parse_existing_pass_at_k(self) -> None:
        """Parse pass@k data that was already calculated."""
        pass_at_k_data = self.raw_results["pass_at_k"]

        # Process category statistics
        if "categories" in pass_at_k_data:
            for cid, results in pass_at_k_data["categories"].items():
                # Create CategoryStats if it doesn't exist yet
                if cid not in self.categories:
                    self.categories[cid] = CategoryStats(cid)

                category_stats = self.categories[cid]

                for difficulty in ["easy", "medium", "hard"]:
                    if difficulty in results:
                        stats = DifficultyStats(
                            passed_tests=results[difficulty]["Passed Tests"],
                            failed_tests=results[difficulty]["Failed Tests"],
                            total_tests=results[difficulty]["Total Tests"],
                            test_pass_percentage=results[difficulty]["Passed Tests (%)"],
                            passed_problems=results[difficulty]["Passed Problems"],
                            failed_problems=results[difficulty]["Failed Problems"],
                            total_problems=results[difficulty]["Total Problems"],
                            problem_pass_percentage=results[difficulty]["Passed Problems (%)"],
                        )

                        # Update category-specific stats
                        setattr(category_stats, difficulty, stats)

                        # Update category totals
                        # For composite reports, we only track problem statistics, not test statistics
                        # since we're dealing with pass@k metrics at the problem level
                        category_stats.total_passed_problems += stats.passed_problems
                        category_stats.total_failed_problems += stats.failed_problems
                        category_stats.total_problems += stats.total_problems

                # Calculate overall percentages
                # For composite reports, we only calculate problem percentages since we don't track test statistics
                if category_stats.total_problems > 0:
                    category_stats.overall_problem_pass_percentage = (
                        category_stats.total_passed_problems / category_stats.total_problems
                    ) * 100

    def _calculate_pass_at_k_from_samples(self) -> None:
        """Calculate pass@k metrics directly from the sample data."""
        samples = self.raw_results.get("samples", [])
        if not samples:
            print("Warning: No samples found in the composite report. Cannot calculate pass@k metrics.")
            return

        # Validate n_samples matches actual sample count
        if self.n_samples == 0:
            print(f"Warning: n_samples is 0 in metadata. Using actual sample count: {len(samples)}")
            self.n_samples = len(samples)
        elif self.n_samples != len(samples):
            print(
                f"Warning: n_samples in metadata ({self.n_samples}) doesn't match actual samples ({len(samples)}). Using actual count."
            )
            self.n_samples = len(samples)

        # Get all unique problem IDs and their metadata from the reports
        problem_ids = {}  # Maps problem_id -> {category, difficulty}

        # First, find all unique problems and their metadata
        for sample in samples:
            for category, cat_data in sample.items():
                if category in ["metadata", "sample_index", "test_details"]:
                    continue

                # Get log entries, if any
                logs = cat_data.get("logs", [])
                for log in logs:
                    if "id" in log:
                        problem_id = log["id"]
                        if problem_id not in problem_ids:
                            # Determine difficulty level
                            difficulty = None

                            # First check the test_details section for each problem, which is more reliable
                            current_problem_id = problem_id  # Save the current problem ID we're processing
                            if "test_details" in sample:
                                # Check passing tests
                                if "passing_tests" in sample["test_details"]:
                                    for test in sample["test_details"]["passing_tests"]:
                                        if "test_id" in test:
                                            # Extract problem ID from test ID (often problem.test format)
                                            test_id = test["test_id"]
                                            # Extract potential problem id
                                            extracted_id = extract_problem_id_from_test_id(test_id)

                                            # Check if this test is for our current problem
                                            if extracted_id == current_problem_id:
                                                # Only log when we find a match with difficulty
                                                if "difficulty" in test:
                                                    difficulty = test.get("difficulty")
                                                    break

                                # Check failing tests if needed
                                if not difficulty and "failing_tests" in sample["test_details"]:
                                    for test in sample["test_details"]["failing_tests"]:
                                        if "test_id" in test:
                                            # Extract problem ID from test ID (often problem.test format)
                                            test_id = test["test_id"]
                                            # Extract potential problem id
                                            extracted_id = extract_problem_id_from_test_id(test_id)

                                            # Check if this test is for our current problem
                                            if extracted_id == current_problem_id:
                                                # Only log when we find a match with difficulty
                                                if "difficulty" in test:
                                                    difficulty = test.get("difficulty")
                                                    break

                            # Log only when we fall back to choosing a default difficulty
                            if not difficulty:
                                # Try other methods first (no need to log each attempt)

                                # Then try logs in each difficulty level
                                for level in ["easy", "medium", "hard"]:
                                    if level in cat_data:
                                        level_logs = cat_data[level].get("logs", [])
                                        if any(entry.get("id") == problem_id for entry in level_logs):
                                            difficulty = level
                                            break

                                # Finally check problems in each difficulty level
                                if not difficulty:
                                    for level in ["easy", "medium", "hard"]:
                                        if level in cat_data and "problems" in cat_data[level]:
                                            if any(p.get("id") == problem_id for p in cat_data[level]["problems"]):
                                                difficulty = level
                                                break

                                    # If we still can't determine difficulty, that's an error
                                    if not difficulty:
                                        raise ValueError(
                                            f"Could not determine difficulty for problem {problem_id} in category {category}. This indicates a data structure issue."
                                        )

                            # Add the problem to the tracking dictionary
                            problem_ids[problem_id] = {"category": category, "difficulty": difficulty}

                            # Check if this category is score-based
                            if is_category_score_based(category):
                                problem_ids[problem_id]["is_score_based"] = True

                # Also check for problems in each difficulty level
                for difficulty in ["easy", "medium", "hard"]:
                    if difficulty in cat_data and "problems" in cat_data[difficulty]:
                        for problem in cat_data[difficulty]["problems"]:
                            if "id" in problem:
                                problem_id = problem["id"]
                                # If this is a new problem, add it
                                if problem_id not in problem_ids:
                                    problem_ids[problem_id] = {"category": category, "difficulty": difficulty}
                                    # Check if this category is score-based
                                    if is_category_score_based(category):
                                        problem_ids[problem_id]["is_score_based"] = True
                                # Handle cross-sample difficulty conflicts
                                # Prefer non-medium difficulties when there's a conflict
                                elif problem_ids[problem_id]["difficulty"] == "medium" and difficulty != "medium":
                                    problem_ids[problem_id]["difficulty"] = difficulty

        # Track problem pass/fail in each sample
        # Initialize tracking structure for all problems across all samples
        problem_results = {
            problem_id: {
                "category": info["category"],
                "difficulty": info["difficulty"],
                "passed_in_samples": [False] * len(samples),
                "pass_count": 0,
                "scores_in_samples": [0.0] * len(samples),  # For score-based categories
                "is_score_based": info.get("is_score_based", False),  # Set based on category
            }
            for problem_id, info in problem_ids.items()
        }

        # Flag to check if we're finding any failed problems
        any_problem_fails = False

        # For each sample, extract all the "Passed Problems" logs to identify which problems passed
        for sample_idx, sample in enumerate(samples):
            sample_passed_problems = set()

            # Check test details for passing test info, if available
            passing_test_ids = set()
            if "test_details" in sample and "passing_tests" in sample["test_details"]:
                for test in sample["test_details"]["passing_tests"]:
                    if "test_id" in test:
                        # Extract problem ID from test ID (often problem.test format)
                        test_id = test["test_id"]
                        # Extract potential problem id
                        problem_id = extract_problem_id_from_test_id(test_id)

                        passing_test_ids.add(problem_id)

            # Also collect failing test IDs
            failing_test_ids = set()
            if "test_details" in sample and "failing_tests" in sample["test_details"]:
                for test in sample["test_details"]["failing_tests"]:
                    if "test_id" in test:
                        # Extract problem ID from test ID (often problem.test format)
                        test_id = test["test_id"]
                        # Extract potential problem id
                        problem_id = extract_problem_id_from_test_id(test_id)

                        failing_test_ids.add(problem_id)
                        if problem_id in problem_ids:
                            any_problem_fails = True

            # Track problems that have explicitly passed and failed
            explicit_passes = set()
            explicit_fails = set()

            # Track score-based problems from summary statistics
            score_based_problems = {}  # problem_id -> average_score

            # For each category, extract the passing and failing problem counts from category summaries
            total_passed = 0
            total_problems = 0

            for category, cat_data in sample.items():
                if category in ["metadata", "sample_index", "test_details"]:
                    continue

                # Check if this is a score-based category
                category_is_score_based = is_category_score_based(category)

                # Process each difficulty level
                for difficulty in ["easy", "medium", "hard"]:
                    if difficulty not in cat_data:
                        continue

                    # Get pass/fail counts from summary statistics
                    passed_problems_count = cat_data[difficulty].get("Passed Problems", 0)
                    total_problems_count = cat_data[difficulty].get("Total Problems", 0)

                    # Update totals
                    total_passed += passed_problems_count
                    total_problems += total_problems_count

                    # For score-based categories, extract scores from logs or summary statistics
                    if category_is_score_based and total_problems_count > 0:
                        # Get problem IDs from logs that match this specific difficulty level
                        # We need to match problems to their exact difficulty, not just category

                        # Look for problems in the detailed problems section for this difficulty
                        difficulty_problem_ids = set()
                        if "problems" in cat_data[difficulty]:
                            for problem in cat_data[difficulty]["problems"]:
                                if "id" in problem:
                                    difficulty_problem_ids.add(problem["id"])

                        # If no detailed problems, try to infer from test_details
                        if not difficulty_problem_ids and "test_details" in sample:
                            for test_list in [
                                sample["test_details"].get("passing_tests", []),
                                sample["test_details"].get("failing_tests", []),
                            ]:
                                for test in test_list:
                                    if test.get("category") == category and test.get("difficulty") == difficulty:
                                        test_id = test.get("test_id", "")
                                        # Extract problem ID from test ID
                                        problem_id = extract_problem_id_from_test_id(test_id)
                                        difficulty_problem_ids.add(problem_id)

                        # Calculate average score per problem from summary statistics for this specific difficulty
                        if difficulty_problem_ids and total_problems_count > 0:
                            avg_score_per_problem = passed_problems_count / total_problems_count

                            # Validate that the score is reasonable for BLEU (0-1 range)
                            if avg_score_per_problem < 0 or avg_score_per_problem > 1:
                                raise ValueError(
                                    f"Invalid BLEU score {avg_score_per_problem} for category {category} difficulty {difficulty}. BLEU scores must be in range [0,1]."
                                )

                            for problem_id in difficulty_problem_ids:
                                # Store the average score for this problem in this specific difficulty
                                score_based_problems[problem_id] = avg_score_per_problem
                        elif category_is_score_based and total_problems_count > 0:
                            # Score-based category with problems but no problem IDs found - this is unexpected
                            raise ValueError(
                                f"Score-based category {category} difficulty {difficulty} has {total_problems_count} problems but no problem IDs found. This indicates a data structure issue."
                            )

                    # Check if we have detailed problem information at this difficulty level
                    if "problems" in cat_data[difficulty]:
                        for problem in cat_data[difficulty]["problems"]:
                            if "id" in problem:
                                problem_id = problem["id"]
                                is_passing = False
                                problem_score = None  # For score-based categories

                                # Check if this is a score-based category (e.g., BLEU)
                                is_score_based = is_category_score_based(category)

                                # Detect pass status
                                if problem.get("status") == "pass":
                                    is_passing = True
                                elif "tests" in problem:
                                    if is_score_based:
                                        # For score-based categories, collect all test scores
                                        test_scores = []
                                        for test in problem["tests"]:
                                            score = test.get("result", 0)
                                            # For BLEU, also check for bleu_score field
                                            if "bleu_score" in test:
                                                score = test["bleu_score"]
                                            # For LLM, also check for llm_score field
                                            elif "llm_score" in test:
                                                score = test["llm_score"]
                                            test_scores.append(score)

                                        if test_scores:
                                            # Use average score as the problem's score
                                            problem_score = sum(test_scores) / len(test_scores)

                                            # Validate score is in reasonable range (0-1)
                                            if problem_score < 0 or problem_score > 1:
                                                raise ValueError(
                                                    f"Invalid score {problem_score} for problem {problem_id}. Scores must be in range [0,1]."
                                                )

                                            # For composite reports, store the fractional score instead of binary pass/fail
                                            # We'll use this score directly in pass@k calculations
                                        else:
                                            raise ValueError(
                                                f"Score-based problem {problem_id} has no test scores. This indicates a data structure issue."
                                            )
                                    else:
                                        # Traditional binary pass/fail logic for non-score-based categories
                                        if all(t.get("result", 1) == 0 for t in problem["tests"]):
                                            is_passing = True

                                # Record problem status
                                if is_score_based and problem_score is not None:
                                    # For score-based categories, store the fractional score
                                    problem_results[problem_id]["scores_in_samples"][sample_idx] = problem_score
                                    problem_results[problem_id]["is_score_based"] = True
                                    # Don't add to explicit passes/fails since we'll handle differently
                                elif is_passing:
                                    explicit_passes.add(problem_id)
                                else:
                                    explicit_fails.add(problem_id)
                                    any_problem_fails = True

            # First handle problems with explicit pass/fail status
            for problem_id in problem_ids:
                # Check if this is a score-based problem with summary statistics
                if problem_id in score_based_problems:
                    # For score-based problems, use the score from summary statistics
                    score = score_based_problems[problem_id]
                    problem_results[problem_id]["scores_in_samples"][sample_idx] = score
                    problem_results[problem_id]["is_score_based"] = True
                    # Don't add to explicit passes/fails since we'll handle differently
                elif problem_id in explicit_passes:
                    # Explicitly marked as passing
                    if not problem_results[problem_id]["passed_in_samples"][sample_idx]:
                        problem_results[problem_id]["passed_in_samples"][sample_idx] = True
                        problem_results[problem_id]["pass_count"] += 1
                        sample_passed_problems.add(problem_id)
                elif problem_id in explicit_fails:
                    # Explicitly marked as failing - make sure it's not counted as passing
                    problem_results[problem_id]["passed_in_samples"][sample_idx] = False
                    # We already set any_problem_fails above
                elif problem_id in passing_test_ids and problem_id not in failing_test_ids:
                    # Passed based on test_details and not in failing tests
                    if not problem_results[problem_id]["passed_in_samples"][sample_idx]:
                        problem_results[problem_id]["passed_in_samples"][sample_idx] = True
                        problem_results[problem_id]["pass_count"] += 1
                        sample_passed_problems.add(problem_id)
                elif problem_id in failing_test_ids:
                    # Failed based on test_details
                    problem_results[problem_id]["passed_in_samples"][sample_idx] = False

        # For score-based categories, we might not have explicit failures, so this warning doesn't apply
        # Only check for binary categories
        has_binary_problems = any(not data.get("is_score_based", False) for data in problem_results.values())
        if not any_problem_fails and has_binary_problems:
            print("\nWARNING: No binary problems were detected as failing in any sample!")
            print(
                "This might indicate an issue with how problem pass/fail status is being detected for non-score-based categories."
            )

        # Validate that score-based problems were properly detected and processed
        for problem_id, data in problem_results.items():
            if data.get("is_score_based", False):
                # Ensure we have scores for all samples
                scores = data["scores_in_samples"]
                # if all(score == 0.0 for score in scores):
                #     raise ValueError(f"Score-based problem {problem_id} has zero scores in all samples. This indicates the score extraction failed.")
                # Ensure all scores are valid
                for i, score in enumerate(scores):
                    if score < 0 or score > 1:
                        raise ValueError(
                            f"Score-based problem {problem_id} has invalid score {score} in sample {i}. BLEU scores must be in range [0,1]."
                        )

        # Create the pass@k report structure
        categories_stats = {}
        total_success_probability = 0.0

        # Collect stats by category and difficulty
        for problem_id, data in problem_results.items():
            category = data["category"]
            difficulty = data["difficulty"]

            if category not in categories_stats:
                categories_stats[category] = {
                    "easy": {"passed": 0, "failed": 0, "total": 0, "pass_probability": 0.0},
                    "medium": {"passed": 0, "failed": 0, "total": 0, "pass_probability": 0.0},
                    "hard": {"passed": 0, "failed": 0, "total": 0, "pass_probability": 0.0},
                }

            if data.get("is_score_based", False):
                # For score-based categories, use the average score across samples as the pass probability
                scores = data["scores_in_samples"]
                avg_score = sum(scores) / len(scores)  # Average score across all samples
                pass_probability = avg_score  # Score is already in 0-1 range for BLEU

                # Update pass count for distribution tracking (use average score)
                data["pass_count"] = avg_score * self.n_samples
            else:
                # Traditional binary pass@k calculation for non-score-based categories
                # pass@k = 1 - (1 - c/n)^k
                # where c is the pass count, n is total samples, k is the threshold
                pass_rate = data["pass_count"] / self.n_samples
                pass_probability = 1.0 - ((1.0 - pass_rate) ** self.k_threshold)

            # Store probability for each problem
            data["pass_probability"] = pass_probability

            # Count the problem for the category/difficulty stats
            categories_stats[category][difficulty]["total"] += 1

            # Accumulate the probability for the category/difficulty stats
            categories_stats[category][difficulty]["pass_probability"] += pass_probability

            # We don't use a binary threshold anymore - each problem contributes its probability
            # directly to the "passed" count - this gives a fractional count of passed problems
            categories_stats[category][difficulty]["passed"] += pass_probability
            categories_stats[category][difficulty]["failed"] += 1.0 - pass_probability

            # Track the overall success probability
            total_success_probability += pass_probability

        # Calculate overall metrics
        total_problems = len(problem_results)
        avg_success_probability = (total_success_probability / total_problems) if total_problems > 0 else 0

        # Update categories stats in the ResultParser object
        for category, difficulty_data in categories_stats.items():
            # Create CategoryStats if it doesn't exist yet
            if category not in self.categories:
                self.categories[category] = CategoryStats(category)

            category_stats = self.categories[category]

            for difficulty, counts in difficulty_data.items():
                if counts["total"] == 0:
                    continue

                # Calculate average pass probability for this category/difficulty
                avg_probability = counts["pass_probability"] / counts["total"]
                pass_percent = avg_probability * 100

                # Create stats object with fractional counts
                stats = DifficultyStats(
                    passed_tests=counts["passed"],
                    failed_tests=counts["failed"],
                    total_tests=counts["total"],
                    test_pass_percentage=pass_percent,
                    passed_problems=counts["passed"],
                    failed_problems=counts["failed"],
                    total_problems=counts["total"],
                    problem_pass_percentage=pass_percent,
                )

                # Update category-specific stats
                setattr(category_stats, difficulty, stats)

                # Update category totals with fractional counts
                # For composite reports, we only track problem statistics, not test statistics
                # since we're dealing with pass@k metrics at the problem level
                category_stats.total_passed_problems += stats.passed_problems
                category_stats.total_failed_problems += stats.failed_problems
                category_stats.total_problems += stats.total_problems

            # Calculate overall percentages
            # For composite reports, we only calculate problem percentages since we don't track test statistics
            if category_stats.total_problems > 0:
                category_stats.overall_problem_pass_percentage = (
                    category_stats.total_passed_problems / category_stats.total_problems
                ) * 100

        # Store the pass@k results back into the raw_results
        # This allows saving the calculated pass@k data if needed
        problem_details = {}
        for problem_id, data in problem_results.items():
            problem_details[problem_id] = {
                "category": data["category"],
                "difficulty": data["difficulty"],
                "pass_count": data["pass_count"],
                "pass_rate": data["pass_count"] / self.n_samples,
                "pass_threshold": self.k_threshold,
                "pass_probability": data.get("pass_probability", 0.0),
                "sample_results": data["passed_in_samples"],
                "is_score_based": data.get("is_score_based", False),
            }

            # For score-based categories, also include the scores
            if data.get("is_score_based", False):
                problem_details[problem_id]["scores_in_samples"] = data["scores_in_samples"]
                problem_details[problem_id]["avg_score"] = sum(data["scores_in_samples"]) / len(
                    data["scores_in_samples"]
                )

        # Build composite categories structure for the report
        composite_categories = {}
        for category, difficulty_data in categories_stats.items():
            composite_categories[category] = {}

            for difficulty, counts in difficulty_data.items():
                if counts["total"] == 0:
                    continue

                # Calculate pass rate based on accumulated probabilities
                # This is the average pass@k probability across all problems
                avg_probability = counts["pass_probability"] / counts["total"] * 100

                # Create stats structure with fractional counts
                composite_categories[category][difficulty] = {
                    "Passed Problems": round(counts["passed"], 2),  # Round to 2 decimal places for readability
                    "Failed Problems": round(counts["failed"], 2),
                    "Total Problems": counts["total"],
                    "Passed Problems (%)": avg_probability,  # This is now directly the average pass@k probability
                    "Pass@k Probability (%)": avg_probability,
                }

        # Store the calculated pass@k data
        self.raw_results["pass_at_k"] = {
            "categories": composite_categories,
            "problems": problem_details,
            "metrics": {
                "total_problems": total_problems,
                "n_samples": self.n_samples,
                "k_threshold": self.k_threshold,
                "avg_success_probability": avg_success_probability,
            },
        }

        # No need to call _parse_existing_pass_at_k() since we already populated self.categories above

    def print_metadata_header(self) -> None:
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
        if not self.golden_mode and self.model_agent:
            print(f"Model/Agent: {self.model_agent}")

        # Print composite report information if applicable
        if self.is_composite:
            print("\n=== Composite Report ===")
            print(f"Number of samples: {self.n_samples}")
            print(
                f"Pass@{self.k_threshold}, n={self.n_samples} threshold: A problem passes if it passes in at least {self.k_threshold} out of {self.n_samples} samples"
            )

        print("=======================\n")

    def print_sample_summary(self) -> None:
        """Print summary statistics for each sample in a composite report."""
        if not self.is_composite or "samples" not in self.raw_results:
            return

        samples = self.raw_results.get("samples", [])
        if not samples:
            return

        print("\n=== Sample Statistics ===")

        # Create table header
        headers = ["Sample #", "Total Problems", "Passed Problems", "Pass Rate", "Prefix"]
        table_data = []
        pass_rates = []  # Track pass rates for stddev calculation

        # Process each sample
        for sample in samples:
            sample_index = sample.get("sample_index", -1)

            # Count totals for this sample
            total_problems = 0
            passed_problems = 0

            for cid, results in sample.items():
                if cid in ["metadata", "sample_index", "test_details"]:
                    continue

                for difficulty in ["easy", "medium", "hard"]:
                    if difficulty in results:
                        total_problems += results[difficulty].get("Total Problems", 0)
                        passed_problems += results[difficulty].get("Passed Problems", 0)

            # Calculate pass rate
            pass_rate = 0
            if total_problems > 0:
                pass_rate = (passed_problems / total_problems) * 100
                pass_rates.append(pass_rate)

            # Get prefix for this sample
            prefix = self.sample_prefixes[sample_index] if sample_index < len(self.sample_prefixes) else "default"

            # Add row to table
            table_data.append([sample_index + 1, total_problems, passed_problems, f"{pass_rate:.2f}%", prefix])

        # Sort by sample index
        table_data.sort(key=lambda x: x[0])

        # Print table
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Print summary statistics
        if len(pass_rates) > 1:
            mean_pass_rate = statistics.mean(pass_rates)
            stddev_pass_rate = statistics.stdev(pass_rates)
            print(f"\nPass Rate Statistics: Mean = {mean_pass_rate:.2f}%, StdDev = {stddev_pass_rate:.2f}%")

    def get_per_sample_statistics(self) -> Dict[str, Any]:
        """Calculate per-sample statistics for stddev calculations in composite reports."""
        if not self.is_composite or "samples" not in self.raw_results:
            return {}

        samples = self.raw_results.get("samples", [])
        if not samples:
            return {}

        # Initialize data structures to track per-sample pass rates
        per_sample_stats = {
            "overall": [],
            "by_difficulty": {"easy": [], "medium": [], "hard": []},
            "by_category": defaultdict(list),
            "by_category_difficulty": defaultdict(lambda: defaultdict(list)),
        }

        # Process each sample
        for sample in samples:
            # Track overall stats for this sample
            sample_total_problems = 0
            sample_passed_problems = 0

            # Track by difficulty for this sample
            difficulty_stats = {
                "easy": {"total": 0, "passed": 0},
                "medium": {"total": 0, "passed": 0},
                "hard": {"total": 0, "passed": 0},
            }

            # Track by category for this sample
            category_stats = defaultdict(lambda: {"total": 0, "passed": 0})

            # Track by category-difficulty for this sample
            category_difficulty_stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "passed": 0}))

            for cid, results in sample.items():
                if cid in ["metadata", "sample_index", "test_details"]:
                    continue

                for difficulty in ["easy", "medium", "hard"]:
                    if difficulty in results:
                        total = results[difficulty].get("Total Problems", 0)
                        passed = results[difficulty].get("Passed Problems", 0)

                        # Update overall
                        sample_total_problems += total
                        sample_passed_problems += passed

                        # Update by difficulty
                        difficulty_stats[difficulty]["total"] += total
                        difficulty_stats[difficulty]["passed"] += passed

                        # Update by category
                        category_stats[cid]["total"] += total
                        category_stats[cid]["passed"] += passed

                        # Update by category-difficulty
                        category_difficulty_stats[cid][difficulty]["total"] += total
                        category_difficulty_stats[cid][difficulty]["passed"] += passed

            # Calculate and store pass rates for this sample
            if sample_total_problems > 0:
                per_sample_stats["overall"].append((sample_passed_problems / sample_total_problems) * 100)

            for difficulty in ["easy", "medium", "hard"]:
                if difficulty_stats[difficulty]["total"] > 0:
                    per_sample_stats["by_difficulty"][difficulty].append(
                        (difficulty_stats[difficulty]["passed"] / difficulty_stats[difficulty]["total"]) * 100
                    )

            for cid, stats in category_stats.items():
                if stats["total"] > 0:
                    per_sample_stats["by_category"][cid].append((stats["passed"] / stats["total"]) * 100)

            for cid, diff_stats in category_difficulty_stats.items():
                for difficulty, stats in diff_stats.items():
                    if stats["total"] > 0:
                        per_sample_stats["by_category_difficulty"][cid][difficulty].append(
                            (stats["passed"] / stats["total"]) * 100
                        )

        return per_sample_stats

    def get_difficulty_totals(self) -> Dict[str, Dict[str, Any]]:
        """Calculate the total stats for each difficulty level across all categories."""
        difficulty_totals = {
            "easy": {
                "passed_tests": 0,
                "failed_tests": 0,
                "total_tests": 0,
                "test_pass_percentage": 0.0,
                "passed_problems": 0,
                "failed_problems": 0,
                "total_problems": 0,
                "problem_pass_percentage": 0.0,
                "stddev": None,
            },
            "medium": {
                "passed_tests": 0,
                "failed_tests": 0,
                "total_tests": 0,
                "test_pass_percentage": 0.0,
                "passed_problems": 0,
                "failed_problems": 0,
                "total_problems": 0,
                "problem_pass_percentage": 0.0,
                "stddev": None,
            },
            "hard": {
                "passed_tests": 0,
                "failed_tests": 0,
                "total_tests": 0,
                "test_pass_percentage": 0.0,
                "passed_problems": 0,
                "failed_problems": 0,
                "total_problems": 0,
                "problem_pass_percentage": 0.0,
                "stddev": None,
            },
        }

        # Aggregate stats across all categories for each difficulty level
        for category_stats in self.categories.values():
            for difficulty in ["easy", "medium", "hard"]:
                difficulty_stats = getattr(category_stats, difficulty)

                # Add test stats
                difficulty_totals[difficulty]["passed_tests"] += difficulty_stats.passed_tests
                difficulty_totals[difficulty]["failed_tests"] += difficulty_stats.failed_tests
                difficulty_totals[difficulty]["total_tests"] += difficulty_stats.total_tests

                # Add problem stats
                difficulty_totals[difficulty]["passed_problems"] += difficulty_stats.passed_problems
                difficulty_totals[difficulty]["failed_problems"] += difficulty_stats.failed_problems
                difficulty_totals[difficulty]["total_problems"] += difficulty_stats.total_problems

        # Calculate pass percentages
        for difficulty, stats in difficulty_totals.items():
            if stats["total_tests"] > 0:
                stats["test_pass_percentage"] = (stats["passed_tests"] / stats["total_tests"]) * 100

            if stats["total_problems"] > 0:
                stats["problem_pass_percentage"] = (stats["passed_problems"] / stats["total_problems"]) * 100

        # Calculate stddev for composite reports
        if self.is_composite:
            per_sample_stats = self.get_per_sample_statistics()
            for difficulty in ["easy", "medium", "hard"]:
                pass_rates = per_sample_stats.get("by_difficulty", {}).get(difficulty, [])
                if len(pass_rates) > 1:
                    difficulty_totals[difficulty]["stddev"] = statistics.stdev(pass_rates)

        return difficulty_totals

    def print_summary(self) -> None:
        """Print a formatted summary of the results in tabular format."""
        # First print the metadata header
        self.print_metadata_header()

        # For composite reports, print sample statistics
        if self.is_composite:
            self.print_sample_summary()

        summary = self.get_summary()

        # Overall statistics table for tests (skip for composite reports)
        if not self.is_composite:
            overall_tests_data = [
                ["Total Tests", summary["overall"]["total_tests"]],
                ["Passed Tests", summary["overall"]["total_passed_tests"]],
                ["Failed Tests", summary["overall"]["total_failed_tests"]],
                ["Test Pass Rate", f"{summary['overall']['overall_test_pass_percentage']:.2f}%"],
            ]

            print("\n=== Overall Test Statistics ===")
            print(tabulate(overall_tests_data, tablefmt="grid"))

        # Overall statistics table for problems
        overall_problems_data = [
            ["Total Problems", summary["overall"]["total_problems"]],
            ["Passed Problems", round(summary["overall"]["total_passed_problems"], 2)],
            ["Failed Problems", round(summary["overall"]["total_failed_problems"], 2)],
            ["Problem Pass Rate", f"{summary['overall']['overall_problem_pass_percentage']:.2f}%"],
        ]

        # Add stddev for composite reports
        if self.is_composite:
            per_sample_stats = self.get_per_sample_statistics()
            overall_pass_rates = per_sample_stats.get("overall", [])
            if len(overall_pass_rates) > 1:
                overall_stddev = statistics.stdev(overall_pass_rates)
                overall_problems_data.append(["Pass Rate StdDev", f"{overall_stddev:.2f}%"])

        # Add pass@k clarification for composite reports
        if self.is_composite:
            print(f"\n=== Overall Problem Statistics (Pass@{self.k_threshold}, n={self.n_samples}) ===")
        else:
            print("\n=== Overall Problem Statistics ===")

        print(tabulate(overall_problems_data, tablefmt="grid"))

        # Print difficulty totals table
        difficulty_totals = self.get_difficulty_totals()

        difficulty_totals_data = []
        for difficulty in ["easy", "medium", "hard"]:
            stats = difficulty_totals[difficulty]
            if stats["total_problems"] > 0:
                row = [
                    difficulty.capitalize(),
                    stats["total_problems"],
                    round(stats["passed_problems"], 2),
                    round(stats["failed_problems"], 2),
                    f"{stats['problem_pass_percentage']:.2f}%",
                ]
                # Add stddev for composite reports
                if self.is_composite:
                    if stats["stddev"] is not None:
                        row.append(f"{stats['stddev']:.2f}%")
                    else:
                        row.append("N/A")
                difficulty_totals_data.append(row)

        if difficulty_totals_data:
            if self.is_composite:
                print(f"\n=== Problem Results by Difficulty (Pass@{self.k_threshold}, n={self.n_samples}) ===")
                headers = ["Difficulty", "Total", "Pass", "Fail", "Rate", "StdDev"]
            else:
                print("\n=== Problem Results by Difficulty ===")
                headers = ["Difficulty", "Total", "Pass", "Fail", "Rate"]

            print(tabulate(difficulty_totals_data, headers=headers, tablefmt="grid"))

        # Category summary table for tests (skip for composite reports)
        if not self.is_composite:
            category_test_data = []
            for cid in sorted(summary["categories"].keys()):
                stats = summary["categories"][cid]
                category_test_data.append(
                    [
                        cid,
                        stats["total_tests"],
                        stats["total_passed_tests"],
                        stats["total_failed_tests"],
                        f"{stats['overall_test_pass_percentage']:.2f}%",
                    ]
                )

            print("\n=== Test Results by Category ===")
            print(tabulate(category_test_data, headers=["Cat", "Total", "Pass", "Fail", "Rate"], tablefmt="grid"))

        # Category summary table for problems
        category_problem_data = []
        score_based_categories_found = False

        # Get per-sample statistics for stddev calculation if composite
        per_sample_stats = None
        if self.is_composite:
            per_sample_stats = self.get_per_sample_statistics()

        for cid in sorted(summary["categories"].keys()):
            stats = summary["categories"][cid]
            category_name = cid

            # Add score-based scoring indicator for score-based categories
            if self._is_score_based_category(cid):
                category_name = f"{cid}*"
                score_based_categories_found = True

            row = [
                category_name,
                stats["total_problems"],
                round(stats["total_passed_problems"], 2),
                round(stats["total_failed_problems"], 2),
                f"{stats['overall_problem_pass_percentage']:.2f}%",
            ]

            # Add stddev for composite reports
            if self.is_composite:
                if per_sample_stats:
                    pass_rates = per_sample_stats.get("by_category", {}).get(cid, [])
                    if len(pass_rates) > 1:
                        stddev = statistics.stdev(pass_rates)
                        row.append(f"{stddev:.2f}%")
                    else:
                        row.append("N/A")
                else:
                    row.append("N/A")

            category_problem_data.append(row)

        # Add pass@k clarification for composite reports
        if self.is_composite:
            print(f"\n=== Problem Results by Category (Pass@{self.k_threshold}, n={self.n_samples}) ===")
            headers = ["Cat", "Total", "Pass", "Fail", "Rate", "StdDev"]
        else:
            print("\n=== Problem Results by Category ===")
            headers = ["Cat", "Total", "Pass", "Fail", "Rate"]

        print(tabulate(category_problem_data, headers=headers, tablefmt="grid"))

        # Add score-based scoring footnote if any score-based categories were found
        if score_based_categories_found:
            print("* Categories marked with asterisk use score-based scoring (0-1 range) instead of binary pass/fail")

        # Complexity breakdown table with difficulties as columns for tests (skip for composite reports)
        if not self.is_composite:
            difficulty_test_data = []
            for cid in sorted(summary["categories"].keys()):
                stats = summary["categories"][cid]
                row = [cid]

                # First add all raw numbers for tests
                for difficulty in ["easy", "medium", "hard"]:
                    if stats[difficulty]["total_tests"] > 0:
                        row.append(f"{stats[difficulty]['passed_tests']}/{stats[difficulty]['total_tests']}")
                    else:
                        row.append("-")

                # Then add all percentages for tests
                for difficulty in ["easy", "medium", "hard"]:
                    if stats[difficulty]["total_tests"] > 0:
                        row.append(f"{stats[difficulty]['test_pass_percentage']:.1f}%")
                    else:
                        row.append("-")

                difficulty_test_data.append(row)

            print("\n=== Test Results by Category and Difficulty ===")

            headers = ["Cat"]
            # Add headers for raw numbers
            for difficulty in ["Easy", "Medium", "Hard"]:
                headers.append(difficulty)
            # Add headers for percentages
            for difficulty in ["Easy", "Medium", "Hard"]:
                headers.append(f"{difficulty}%")

            print(tabulate(difficulty_test_data, headers=headers, tablefmt="grid"))

        # Complexity breakdown table with difficulties as columns for problems
        difficulty_problem_data = []
        score_based_categories_in_difficulty = False

        # Get per-sample statistics for stddev if composite
        if self.is_composite and not per_sample_stats:
            per_sample_stats = self.get_per_sample_statistics()

        for cid in sorted(summary["categories"].keys()):
            stats = summary["categories"][cid]
            category_name = cid

            # Add score-based scoring indicator for score-based categories
            if self._is_score_based_category(cid):
                category_name = f"{cid}*"
                score_based_categories_in_difficulty = True

            row = [category_name]

            # First add all raw numbers for problems
            for difficulty in ["easy", "medium", "hard"]:
                if stats[difficulty]["total_problems"] > 0:
                    passed = round(stats[difficulty]["passed_problems"], 2)
                    total = stats[difficulty]["total_problems"]
                    row.append(f"{passed}/{total}")
                else:
                    row.append("-")

            # Then add all percentages for problems
            for difficulty in ["easy", "medium", "hard"]:
                if stats[difficulty]["total_problems"] > 0:
                    row.append(f"{stats[difficulty]['problem_pass_percentage']:.2f}%")
                else:
                    row.append("-")

            # Add stddev columns for composite reports
            if self.is_composite:
                if per_sample_stats:
                    for difficulty in ["easy", "medium", "hard"]:
                        if stats[difficulty]["total_problems"] > 0:
                            pass_rates = (
                                per_sample_stats.get("by_category_difficulty", {}).get(cid, {}).get(difficulty, [])
                            )
                            if len(pass_rates) > 1:
                                stddev = statistics.stdev(pass_rates)
                                row.append(f"{stddev:.2f}%")
                            else:
                                row.append("N/A")
                        else:
                            row.append("-")
                else:
                    # If no per_sample_stats, still need placeholders for the 3 stddev columns
                    for difficulty in ["easy", "medium", "hard"]:
                        row.append("N/A")

            difficulty_problem_data.append(row)

        # Add pass@k clarification for composite reports
        if self.is_composite:
            print(
                f"\n=== Problem Results by Category and Difficulty (Pass@{self.k_threshold}, n={self.n_samples}) ==="
            )
        else:
            print("\n=== Problem Results by Category and Difficulty ===")

        headers = ["Cat"]
        # Add headers for raw numbers
        for difficulty in ["Easy", "Medium", "Hard"]:
            headers.append(difficulty)
        # Add headers for percentages
        for difficulty in ["Easy", "Medium", "Hard"]:
            headers.append(f"{difficulty}%")
        # Add headers for stddev (composite only)
        if self.is_composite:
            for difficulty in ["Easy", "Medium", "Hard"]:
                headers.append(f"{difficulty} SD")

        print(tabulate(difficulty_problem_data, headers=headers, tablefmt="grid"))

        # Add score-based scoring footnote if any score-based categories were found
        if score_based_categories_in_difficulty:
            print("* Categories marked with asterisk use score-based scoring (0-1 range) instead of binary pass/fail")

        # Skip failing/passing problem sections for composite reports
        if not self.is_composite:
            # Print failing problems report
            print("\n=== Failing Problems ===")
            failing_problems = self.print_failing_problems()
            if not failing_problems:
                print("No failing problems found.")

            # Print passing problems list
            print("\n=== Passing Problems ===")
            passing_problems = self.print_passing_problems()
            if not passing_problems:
                print("No passing problems found.")

        # For composite reports, show pass@k distribution
        if self.is_composite and "pass_at_k" in self.raw_results and "problems" in self.raw_results["pass_at_k"]:
            self.print_pass_at_k_distribution()

    def print_pass_at_k_distribution(self) -> None:
        """Print distribution of pass counts for problems in a composite report."""
        if not self.is_composite or "pass_at_k" not in self.raw_results:
            return

        problems = self.raw_results["pass_at_k"].get("problems", {})
        if not problems:
            return

        print(f"\n=== Pass@{self.k_threshold} Distribution (n={self.n_samples}) ===")
        print(
            f"Pass@{self.k_threshold} measures the probability of a problem passing in at least {self.k_threshold} of {self.n_samples} randomly-selected samples"
        )

        # Count how many problems pass in 0, 1, 2, ..., n samples
        distribution = {}
        problems_by_pass_count = defaultdict(list)

        for problem_id, problem_data in problems.items():
            pass_count = problem_data.get("pass_count", 0)
            is_score_based = problem_data.get("is_score_based", False)

            # For score-based categories, round the fractional pass count for distribution grouping
            if is_score_based:
                # Round to nearest 0.1 for reasonable grouping
                pass_count_key = round(pass_count, 1)
            else:
                # For binary categories, pass_count should be an integer
                pass_count_key = int(round(pass_count))

            # Count for distribution table
            if pass_count_key not in distribution:
                distribution[pass_count_key] = 0
            distribution[pass_count_key] += 1

            # Group problems by pass count
            problems_by_pass_count[pass_count_key].append(
                {
                    "problem_id": problem_id,
                    "category": problem_data.get("category", ""),
                    "difficulty": problem_data.get("difficulty", ""),
                    "pass_probability": problem_data.get("pass_probability", 0.0),
                    "is_score_based": is_score_based,
                    "actual_pass_count": pass_count,  # Store the original value
                }
            )

        # Create table data for distribution summary
        table_data = []
        for pass_count_key in sorted(distribution.keys()):
            count = distribution[pass_count_key]
            percentage = (count / len(problems)) * 100

            # For display, handle fractional pass counts appropriately
            if isinstance(pass_count_key, float):
                # Score-based category with fractional pass count
                display_count = f"{pass_count_key:.1f}"
                # For score-based, the "probability" is just the average score itself
                probability_pct = pass_count_key / self.n_samples * 100
            else:
                # Traditional binary category
                display_count = str(pass_count_key)
                # Calculate pass@k probability for this group
                # pass@k = 1 - (1 - pass_count/n_samples)^k
                pass_rate = pass_count_key / self.n_samples
                probability = 1.0 - ((1.0 - pass_rate) ** self.k_threshold)
                probability_pct = probability * 100

            table_data.append(
                [f"{display_count}/{self.n_samples}", count, f"{percentage:.2f}%", f"{probability_pct:.2f}%"]
            )

        headers = [
            "Pass Count",
            "Problems",
            "% of Dataset",
            f"Pass@{self.k_threshold}, n={self.n_samples} Probability",
        ]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Create consolidated problems table with all pass counts
        print(f"\n=== Problems by Pass Count (Pass@{self.k_threshold}, n={self.n_samples}) ===")

        consolidated_table = []

        # Process each pass count group in ascending order (lowest to highest)
        for i, pass_count_key in enumerate(sorted(problems_by_pass_count.keys())):
            problems_list = problems_by_pass_count[pass_count_key]

            # Check if this group contains score-based problems
            has_score_based = any(p.get("is_score_based", False) for p in problems_list)

            if has_score_based:
                # For score-based categories, use the average score as the probability
                probability_pct = pass_count_key / self.n_samples * 100
                display_count = f"{pass_count_key:.1f}"
            else:
                # Traditional binary calculation
                pass_rate = pass_count_key / self.n_samples
                probability = 1.0 - ((1.0 - pass_rate) ** self.k_threshold)
                probability_pct = probability * 100
                display_count = str(int(pass_count_key))

            # Add extra visual separation between sections (except for first section)
            if i > 0:
                consolidated_table.append(["", ""])
                consolidated_table.append(["", ""])
                consolidated_table.append(
                    ["═══════════════════════════════════", "═══════════════════════════════════"]
                )

            # Add section header row with count and status
            consolidated_table.append(
                [f"Pass Count: {display_count}/{self.n_samples}", f"Total: {len(problems_list)} problems"]
            )

            # Add probability information
            if has_score_based:
                consolidated_table.append(
                    [
                        f"Avg Score: {pass_count_key / self.n_samples:.4f}",
                        f"Score-based Probability: {probability_pct:.2f}%",
                    ]
                )
            else:
                pass_rate = pass_count_key / self.n_samples
                consolidated_table.append(
                    [
                        f"Pass Rate: {pass_rate:.4f}",
                        f"Pass@{self.k_threshold}, n={self.n_samples} Probability: {probability_pct:.2f}%",
                    ]
                )

            # Add a separator row
            consolidated_table.append(["───────────────────────────────────", "───────────────────────────────────"])

            # Add all problems in this pass count group (sorted by ID)
            for problem in sorted(problems_list, key=lambda x: x["problem_id"]):
                category_info = f"{problem['category']} ({problem['difficulty']})" if problem["category"] else "N/A"
                consolidated_table.append([problem["problem_id"], category_info])

        # Print the consolidated table
        print(
            tabulate(
                consolidated_table,
                headers=["Problem ID", "Category (Difficulty)"],
                tablefmt="grid",
                colalign=("left", "left"),
            )
        )

        # Get overall metrics
        metrics = self.raw_results["pass_at_k"].get("metrics", {})
        avg_probability = metrics.get("avg_success_probability", 0.0)

        # Summary with probability
        print(f"\nOverall Pass@{self.k_threshold}, n={self.n_samples} probability: {avg_probability * 100:.2f}%")

    def get_summary(self) -> Dict[str, Any]:
        """Generate a summary of all results."""
        summary = {
            "categories": {},
            "overall": {
                "total_passed_tests": 0,
                "total_failed_tests": 0,
                "total_tests": 0,
                "overall_test_pass_percentage": 0.0,
                "total_passed_problems": 0,
                "total_failed_problems": 0,
                "total_problems": 0,
                "overall_problem_pass_percentage": 0.0,
            },
        }

        # Aggregate results across all categories
        for cid, stats in self.categories.items():
            category_summary = {
                "easy": {
                    "passed_tests": stats.easy.passed_tests,
                    "failed_tests": stats.easy.failed_tests,
                    "total_tests": stats.easy.total_tests,
                    "test_pass_percentage": stats.easy.test_pass_percentage,
                    "passed_problems": stats.easy.passed_problems,
                    "failed_problems": stats.easy.failed_problems,
                    "total_problems": stats.easy.total_problems,
                    "problem_pass_percentage": stats.easy.problem_pass_percentage,
                },
                "medium": {
                    "passed_tests": stats.medium.passed_tests,
                    "failed_tests": stats.medium.failed_tests,
                    "total_tests": stats.medium.total_tests,
                    "test_pass_percentage": stats.medium.test_pass_percentage,
                    "passed_problems": stats.medium.passed_problems,
                    "failed_problems": stats.medium.failed_problems,
                    "total_problems": stats.medium.total_problems,
                    "problem_pass_percentage": stats.medium.problem_pass_percentage,
                },
                "hard": {
                    "passed_tests": stats.hard.passed_tests,
                    "failed_tests": stats.hard.failed_tests,
                    "total_tests": stats.hard.total_tests,
                    "test_pass_percentage": stats.hard.test_pass_percentage,
                    "passed_problems": stats.hard.passed_problems,
                    "failed_problems": stats.hard.failed_problems,
                    "total_problems": stats.hard.total_problems,
                    "problem_pass_percentage": stats.hard.problem_pass_percentage,
                },
                "total_passed_tests": stats.total_passed_tests,
                "total_failed_tests": stats.total_failed_tests,
                "total_tests": stats.total_tests,
                "overall_test_pass_percentage": stats.overall_test_pass_percentage,
                "total_passed_problems": stats.total_passed_problems,
                "total_failed_problems": stats.total_failed_problems,
                "total_problems": stats.total_problems,
                "overall_problem_pass_percentage": stats.overall_problem_pass_percentage,
            }

            summary["categories"][cid] = category_summary

            # Update overall totals for tests
            summary["overall"]["total_passed_tests"] += stats.total_passed_tests
            summary["overall"]["total_failed_tests"] += stats.total_failed_tests
            summary["overall"]["total_tests"] += stats.total_tests

            # Update overall totals for problems
            summary["overall"]["total_passed_problems"] += stats.total_passed_problems
            summary["overall"]["total_failed_problems"] += stats.total_failed_problems
            summary["overall"]["total_problems"] += stats.total_problems

        # Calculate overall pass percentage for tests
        if summary["overall"]["total_tests"] > 0:
            summary["overall"]["overall_test_pass_percentage"] = (
                summary["overall"]["total_passed_tests"] / summary["overall"]["total_tests"]
            ) * 100

        # Calculate overall pass percentage for problems
        if summary["overall"]["total_problems"] > 0:
            summary["overall"]["overall_problem_pass_percentage"] = (
                summary["overall"]["total_passed_problems"] / summary["overall"]["total_problems"]
            ) * 100

        return summary

    def get_failing_tests(self) -> List[Dict[str, Any]]:
        """Extract all failing tests with their error messages."""
        # If we have failing tests in the report, return them
        if self.failing_tests:
            return self.failing_tests

        # Otherwise, fall back to extracting from raw_results
        failing_tests = []

        for test_id, results in self.raw_results.items():
            if test_id in ["test_details", "metadata"]:
                continue

            if "tests" in results:
                # Direct test results at top level
                for i, test in enumerate(results["tests"]):
                    if test.get("result", 0) != 0:  # Non-zero result indicates failure
                        failing_tests.append(
                            {
                                "test_id": test_id,
                                "test_index": i,
                                "error_msg": test.get("error_msg"),
                                "agent_error": test.get("agent_error", None),
                                "category": test_id,  # Use the test_id as category for top-level tests
                            }
                        )
            else:
                # Need to look in the difficulty levels
                for difficulty in ["easy", "medium", "hard"]:
                    if difficulty in results and "tests" in results[difficulty]:
                        for i, test in enumerate(results[difficulty]["tests"]):
                            if test.get("result", 0) != 0:  # Non-zero result indicates failure
                                failing_tests.append(
                                    {
                                        "test_id": test_id,
                                        "difficulty": difficulty,
                                        "test_index": i,
                                        "error_msg": test.get("error_msg"),
                                        "agent_error": test.get("agent_error", None),
                                        "category": test_id,  # Use the test_id as category
                                    }
                                )

        return failing_tests

    def get_passing_tests(self) -> List[Dict[str, Any]]:
        """Extract all passing tests."""
        # If we have passing tests in the report, return them
        if self.passing_tests:
            return self.passing_tests

        # Otherwise, fall back to extracting from raw_results
        passing_tests = []

        for test_id, results in self.raw_results.items():
            if test_id in ["test_details", "metadata"]:
                continue

            if "tests" in results:
                # Direct test results at top level
                for i, test in enumerate(results["tests"]):
                    if test.get("result", 0) == 0:  # Zero result indicates success
                        passing_tests.append(
                            {
                                "test_id": test_id,
                                "test_index": i,
                                "category": test_id,  # Use the test_id as category for top-level tests
                            }
                        )
            else:
                # Need to look in the difficulty levels
                for difficulty in ["easy", "medium", "hard"]:
                    if difficulty in results and "tests" in results[difficulty]:
                        for i, test in enumerate(results[difficulty]["tests"]):
                            if test.get("result", 0) == 0:  # Zero result indicates success
                                passing_tests.append(
                                    {
                                        "test_id": test_id,
                                        "difficulty": difficulty,
                                        "test_index": i,
                                        "category": test_id,  # Use the test_id as category
                                    }
                                )

        return passing_tests

    def get_failing_problems(self) -> List[Dict[str, Any]]:
        """Extract all failing problems with associated failed test logs."""
        # If we have already processed failing problems, return them
        if self.failing_problems:
            return self.failing_problems

        # Otherwise, extract from raw results
        failing_problems = []
        failing_tests = self.get_failing_tests()

        problem_tests_map = defaultdict(list)

        # Group failing tests by problem ID
        for test in failing_tests:
            # Extract problem ID from test_id (assuming format like problem_id.test_id)
            problem_id = extract_problem_id_from_test_id(test["test_id"])

            # Get category if available - note category is now stored directly with test
            category = test.get("category", "")
            difficulty = test.get("difficulty", "unknown")

            # Construct a problem key that includes category and difficulty if available
            problem_key = f"{problem_id}:{category}:{difficulty}"

            # Add this test to the problem's test list
            problem_tests_map[problem_key].append(test)

        # Create problem entries with their associated tests
        for problem_key, tests in problem_tests_map.items():
            problem_id, category, difficulty = problem_key.split(":")

            log_path = ""

            failing_problems.append(
                {
                    "problem_id": problem_id,
                    "category": category,
                    "difficulty": difficulty,
                    "failed_tests": tests,
                    "log": log_path,
                }
            )

        # Sort problems by ID
        failing_problems.sort(key=lambda p: p["problem_id"])
        self.failing_problems = failing_problems

        return failing_problems

    def get_passing_problems(self) -> List[Dict[str, Any]]:
        """Extract all passing problems."""
        # If we have already processed passing problems, return them
        if self.passing_problems:
            return self.passing_problems

        # To identify passing problems, we need:
        # 1. Get problems that have passed tests
        # 2. Exclude problems that already exist in failing problems
        passing_problems = []
        passing_tests = self.get_passing_tests()

        # Get all failing problems to exclude them
        failing_problems = self.get_failing_problems()
        failing_problem_ids = {p["problem_id"] for p in failing_problems}

        # First, group passing tests by problem ID (similar to how we do with failing tests)
        problem_tests_map = defaultdict(list)

        # Group passing tests by problem ID
        for test in passing_tests:
            # Extract problem ID from test_id (assuming format like problem_id.test_id)
            problem_id = extract_problem_id_from_test_id(test["test_id"])

            # Skip if this is a known failing problem
            if problem_id in failing_problem_ids:
                continue

            # Get category if available - note category is now stored directly with test
            category = test.get("category", "")
            difficulty = test.get("difficulty", "unknown")

            # Construct a problem key that includes category and difficulty if available
            problem_key = f"{problem_id}:{category}:{difficulty}"

            # Add this test to the problem's test list
            problem_tests_map[problem_key].append(test)

        # Now, convert the grouped tests into problem entries
        for problem_key, tests in problem_tests_map.items():
            problem_id, category, difficulty = problem_key.split(":")

            # Construct log path for the problem
            log_path = ""

            passing_problems.append(
                {
                    "problem_id": problem_id,
                    "category": category,
                    "difficulty": difficulty,
                    "passing_tests": tests,
                    "log": log_path,
                }
            )

        # Additionally, scan through the raw results to find problems explicitly marked as passed
        for cid, results in self.raw_results.items():
            if cid in ["test_details", "metadata"]:
                continue

            for difficulty in ["easy", "medium", "hard"]:
                if difficulty in results and "problems" in results[difficulty]:
                    for problem in results[difficulty]["problems"]:
                        problem_id = problem.get("id", "")

                        # Skip if already added or in failing problems
                        if not problem_id or problem_id in failing_problem_ids:
                            continue

                        # Check if this problem ID is already in our passing problems
                        already_added = False
                        for p in passing_problems:
                            if p["problem_id"] == problem_id:
                                already_added = True
                                break

                        if already_added:
                            continue

                        # If has "pass" status or has tests and all tests passed
                        if problem.get("status") == "pass" or (
                            "tests" in problem and all(t.get("result", 1) == 0 for t in problem["tests"])
                        ):
                            # Construct log path for the problem
                            log_path = None
                            if cid:
                                # Check if the category starts with 'cat' - if not, it might be a direct category name
                                if cid.startswith("cat"):
                                    category_num = cid.replace("cat", "")
                                else:
                                    category_num = cid  # Use the category as is
                                log_path = f"work/cvdp_cat{category_num}/reports/{problem_id}"

                            # Extract any tests for this problem
                            tests = problem.get("tests", [])
                            passing_test_objects = []

                            for i, test in enumerate(tests):
                                if test.get("result", 1) == 0:  # Passing test
                                    passing_test_objects.append(
                                        {
                                            "test_id": f"{problem_id}.test{i + 1}",
                                            "test_index": i,
                                            "category": cid,
                                            "difficulty": difficulty,
                                        }
                                    )

                            passing_problems.append(
                                {
                                    "problem_id": problem_id,
                                    "category": cid,
                                    "difficulty": difficulty,
                                    "passing_tests": passing_test_objects,
                                    "log": log_path,
                                }
                            )

        # Sort problems by ID
        passing_problems.sort(key=lambda p: p["problem_id"])
        self.passing_problems = passing_problems

        return passing_problems

    def print_failing_problems(self) -> List[Dict[str, Any]]:
        """Print a detailed report of failing problems with their failed tests."""
        failing_problems = self.get_failing_problems()

        # Print the failing problems in tabular format
        if failing_problems:
            table_data = []
            for i, problem in enumerate(failing_problems):
                row = [i + 1, problem["problem_id"]]

                # Add category and difficulty if available
                category = problem.get("category", "")
                difficulty = problem.get("difficulty", "")
                if category and difficulty and difficulty != "unknown":
                    row.append(f"{category} ({difficulty})")
                elif category:
                    row.append(category)
                else:
                    row.append("N/A")

                # Count the number of failed tests
                num_failed_tests = len(problem.get("failed_tests", []))
                row.append(num_failed_tests)

                # Add log location
                row.append(problem.get("log", "N/A"))

                table_data.append(row)

                # For each failed test within this problem, add a subrow with test details
                for j, test in enumerate(problem.get("failed_tests", [])):
                    # Use agent_error if available, otherwise use error_msg
                    error_msg = None
                    if test.get("agent_error"):
                        error_msg = test.get("agent_error")
                    else:
                        error_msg = test.get("error_msg")

                    # Get test log location
                    test_log = None
                    if "log" in test:
                        test_log = test["log"]

                    test_row = [
                        "",  # Empty first column
                        f"   ↳ Test {j + 1}: {test.get('test_id', '')}",
                        "",  # Empty category column
                        "",  # Empty test count column
                        f"Log: {test_log if test_log else 'Not Available'}",
                    ]
                    table_data.append(test_row)

                    # Add error message in a separate row if available
                    if error_msg:
                        error_row = [
                            "",  # Empty first column
                            "      Error: ",
                            "",  # Empty category column
                            "",  # Empty test count column
                            error_msg,
                        ]
                        table_data.append(error_row)

            print(
                tabulate(
                    table_data, headers=["#", "Problem ID", "Category", "Failed Tests", "Log/Error"], tablefmt="grid"
                )
            )

        return failing_problems

    def print_passing_problems(self) -> List[Dict[str, Any]]:
        """Print a list of passing problems with their passing tests."""
        passing_problems = self.get_passing_problems()

        # Print the passing problems in tabular format
        if passing_problems:
            table_data = []
            for i, problem in enumerate(passing_problems):
                row = [i + 1, problem["problem_id"]]

                # Add category and difficulty if available
                category = problem.get("category", "")
                difficulty = problem.get("difficulty", "")
                if category and difficulty and difficulty != "unknown":
                    row.append(f"{category} ({difficulty})")
                elif category:
                    row.append(category)
                else:
                    row.append("N/A")

                # Count the number of passing tests
                num_passing_tests = len(problem.get("passing_tests", []))
                row.append(num_passing_tests)

                # Add log location
                row.append(problem.get("log", "N/A"))

                table_data.append(row)

                # For each passing test within this problem, add a subrow with test details
                for j, test in enumerate(problem.get("passing_tests", [])):
                    # Get test log location
                    test_log = None
                    if "log" in test:
                        test_log = test["log"]
                    test_row = [
                        "",  # Empty first column
                        f"   ↳ Test {j + 1}: {test.get('test_id', '')}",
                        "",  # Empty category column
                        "",  # Empty test count column
                        f"Log: {test_log if test_log else 'Not Available'}",
                    ]
                    table_data.append(test_row)

            print(
                tabulate(
                    table_data, headers=["#", "Problem ID", "Category", "Passed Tests", "Log/Status"], tablefmt="grid"
                )
            )

        return passing_problems

    def _is_score_based_category(self, category_id: str) -> bool:
        """Check if a category uses score-based scoring (0-1 fractional pass rates) instead of binary pass/fail."""
        return is_category_score_based(category_id)


def main():
    """Example usage of the ResultParser."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Parse and analyze benchmark results")
    parser.add_argument("json_file", help="Path to the JSON results file")
    parser.add_argument("-k", type=int, help="Override the k threshold for pass@k metrics")
    parser.add_argument("--save", action="store_true", help="Save calculated pass@k results back to the file")
    parser.add_argument("-o", "--output", help="Output file to save the text report (default: stdout)")
    args = parser.parse_args()

    result_parser = ResultParser(args.json_file)
    result_parser.load_results()

    # Override k threshold if specified
    if args.k is not None:
        result_parser.k_threshold = args.k
        print(f"Overriding k threshold to {args.k}")

    result_parser.parse_results()

    # Redirect output to file if specified
    original_stdout = sys.stdout
    if args.output:
        try:
            sys.stdout = open(args.output, "w")
        except Exception as e:
            print(f"Error opening output file {args.output}: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        result_parser.print_summary()
    finally:
        # Restore stdout and close output file if needed
        if args.output:
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"Report saved to {args.output}")

    # Save the updated pass@k results if requested
    if args.save and result_parser.is_composite and "pass_at_k" in result_parser.raw_results:
        print(f"Saving calculated pass@k results to {args.json_file}")
        with open(args.json_file, "w") as f:
            json.dump(result_parser.raw_results, f, indent=2)


if __name__ == "__main__":
    main()
