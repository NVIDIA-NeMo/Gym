# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official leaf-level AIR-Bench Foundation accuracy reporting.

The public scorer prints one accuracy for each ``task_name_dataset_name`` leaf
and does not define an overall Foundation accuracy. Consequently a calculator
that observes multiple leaves reports only the official ``total_sum`` and
``fail_num`` counts; it never invents a micro or macro aggregate.
"""

from resources_servers.audio_eval._scoring.evaluation.metrics.base import BaseMetrics


class AIRBenchFoundationMetrics(BaseMetrics):
    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        if max_k != 1:
            raise ValueError("AIRBenchFoundationMetrics supports one greedy generation only")
        super().__init__(compute_no_answer=compute_no_answer)

    def reset(self):
        super().reset()
        self.max_k = 1
        self.official_total = 0
        self.official_correct = 0
        self.fail_num = 0
        self.leaves: set[str] = set()

    def _get_score_dict(self, prediction):
        return {"correct": bool(prediction.get("symbolic_correct", False))}

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("AIRBenchFoundationMetrics expects exactly one prediction per sample")
        super().update(predictions)
        prediction = predictions[0]
        leaf = prediction.get("air_bench_foundation_leaf", prediction.get("subset_for_metrics"))
        if not isinstance(leaf, str) or not leaf:
            raise ValueError("AIR-Bench Foundation prediction is missing its official leaf identifier")
        self.leaves.add(leaf)
        if bool(prediction.get("official_parse_valid", False)):
            self.official_total += 1
            self.official_correct += int(bool(prediction.get("symbolic_correct", False)))
        else:
            self.fail_num += 1

    def get_metrics(self):
        values = {}
        self.update_common_metrics(values)
        values.update(
            {
                "total_sum": self.official_total,
                "correct_num": self.official_correct,
                "fail_num": self.fail_num,
                "num_leaves": len(self.leaves),
            }
        )
        # ComputeMetrics creates one calculator per subset and an _all_
        # calculator. Only the single-leaf calculators reproduce an official
        # accuracy; the mixed _all_ calculator intentionally has none.
        if len(self.leaves) == 1:
            values["accuracy"] = (
                round(100.0 * self.official_correct / self.official_total, 2) if self.official_total else 0.0
            )
        return {"pass@1": values}
