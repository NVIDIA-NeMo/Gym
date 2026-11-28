# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AudioBench metrics: the audio metrics plus the Spoken-MQA accuracy and parse-failure diagnostics."""

from resources_servers.audio_eval._scoring.evaluation.metrics.audio_metrics import AudioMetrics


class AudioBenchMetrics(AudioMetrics):
    """``AudioMetrics`` with Spoken-MQA's official metric name (``accuracy``, ``acc`` upstream) and diagnostics."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer, max_k=max_k)
        self.has_spoken_mqa = False
        self.spoken_mqa_parse_failures = 0
        self.spoken_mqa_parse_diagnostics = 0

    def update(self, predictions):
        for pred in predictions:
            if pred.get("task_type") == "MathQA":
                self.has_spoken_mqa = True
                if "spoken_mqa_parse_failure" in pred:
                    self.spoken_mqa_parse_diagnostics += 1
                    self.spoken_mqa_parse_failures += int(bool(pred["spoken_mqa_parse_failure"]))
        super().update(predictions)

    def get_metrics(self):
        metrics_dict = super().get_metrics()
        for agg_metrics in metrics_dict.values():
            if self.has_spoken_mqa and "success_rate" in agg_metrics:
                agg_metrics["accuracy"] = agg_metrics["success_rate"]
            if self.spoken_mqa_parse_diagnostics:
                # Attempt-level diagnostic only; it does not participate in
                # official accuracy or pass@k correctness.
                agg_metrics["no_numeric_answer_rate"] = round(
                    100.0 * self.spoken_mqa_parse_failures / self.spoken_mqa_parse_diagnostics, 2
                )
        return metrics_dict
