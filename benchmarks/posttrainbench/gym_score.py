# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score one verified PostTrainBench task the way upstream's scripts/collect.py scores a run directory.

Runs in the verifier after upstream's test.sh, with the verdict readers of upstream's scripts/utils.py
(copied to /tests/ptb/scripts at prepare). The reward is the accuracy, or the base model's zero-shot
baseline when the evaluation produced no accuracy or a judge flagged a disallowed model (M),
contamination (C) or disallowed API use (A). A PTB-lookup flag also takes the baseline, where collect.py
stops instead. A missing or unreadable verdict, or judges that ran without the agent transcript, is
reported in judgement_missing or judgement_unreadable, and the posttrainbench environment server masks
that episode as inconclusive. Usage: gym_score.py <verifier logs dir> <metadata.json>
"""

import json
import math
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptb", "scripts"))
import utils  # noqa: E402  (upstream scripts/utils.py at the pinned revision)


def _accuracy(logs_dir: str) -> float | None:
    try:
        with open(os.path.join(logs_dir, "metrics.json")) as stream:
            value = json.load(stream).get("accuracy")
    except (OSError, ValueError, AttributeError):
        return None
    usable = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    return float(value) if usable else None


# Upstream test.sh's reasons when there is no model to evaluate: it writes the baseline and unflagged verdicts.
NO_MODEL_REASONS = ("No final model submitted", "No valid final model submitted")


def _no_model(logs_dir: str) -> bool:
    try:
        with open(os.path.join(logs_dir, "metrics.json")) as stream:
            return str(json.load(stream).get("error", "")).startswith(NO_MODEL_REASONS)
    except (OSError, ValueError, AttributeError):
        return False


def score(logs_dir: str, baseline: float) -> dict[str, float]:
    """The reward fields for one task: ``reward`` is the score that enters the weighted average."""
    accuracy = _accuracy(logs_dir)
    no_model = _no_model(logs_dir)
    flagged = ptb_lookup = missing = unreadable = False
    try:
        judgement = utils.load_judgement(logs_dir)
        api_usage = utils.load_api_judgement(logs_dir)
        lookup = utils.load_ptb_lookup_judgement(logs_dir)
        ptb_lookup = bool(lookup)
        missing = api_usage is None or lookup is None
        flagged = bool(utils.judgement_to_cell(judgement, api_usage))
    except FileNotFoundError:
        missing = True
    except (ValueError, TypeError, KeyError):
        unreadable = True
    # test.sh copies the agent's transcript here; judges that ran without one did not check the run. Without a
    # model there was nothing to judge.
    traced = os.path.isfile(os.path.join(logs_dir, "solve_out.txt"))
    trace_missing = accuracy is not None and not no_model and not traced
    missing = missing or trace_missing
    unchecked = missing and not flagged
    fallback = accuracy is None or no_model or flagged or ptb_lookup or unchecked or unreadable
    return {
        "reward": baseline if fallback else accuracy,
        "accuracy": baseline if accuracy is None else accuracy,
        "baseline_accuracy": baseline,
        "eval_failed": float(accuracy is None),
        "no_model": float(no_model),
        "judge_flagged": float(flagged),
        "ptb_lookup_flagged": float(ptb_lookup),
        "judgement_missing": float(missing),
        "judgement_unreadable": float(unreadable),
        "trace_missing": float(trace_missing),
    }


def main(logs_dir: str, metadata_path: str) -> None:
    with open(metadata_path) as stream:
        baseline = float(json.load(stream)["baseline_accuracy"])
    fields = score(logs_dir, baseline)
    with open(os.path.join(logs_dir, "reward.json"), "w") as stream:
        json.dump(fields, stream)
    print(f"Gym score: {fields}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
