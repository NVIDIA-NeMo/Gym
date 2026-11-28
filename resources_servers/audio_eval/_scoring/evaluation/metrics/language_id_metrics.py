# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable top-1 language-ID micro/macro and group metrics."""

from collections import defaultdict

from resources_servers.audio_eval._scoring.evaluation.metrics.base import BaseMetrics


class LanguageIDMetrics(BaseMetrics):
    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError("LanguageIDMetrics supports top-1 evaluation only")
        self.max_k = 1
        self.correct = 0
        self.label_correct = defaultdict(int)
        self.label_total = defaultdict(int)
        self.group_correct = defaultdict(int)
        self.group_total = defaultdict(int)
        self.label_group: dict[str, str] = {}

    def _get_score_dict(self, prediction):
        return {"correct": bool(prediction["lid_correct"])}

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("LanguageIDMetrics expects exactly one prediction per sample")
        super().update(predictions)
        prediction = predictions[0]
        label = str(prediction.get("label", prediction["expected_answer"]))
        group_value = prediction.get("label_group")
        group = str(group_value) if group_value is not None else None
        is_correct = int(bool(prediction["lid_correct"]))
        self.correct += is_correct
        self.label_correct[label] += is_correct
        self.label_total[label] += 1
        if group is not None:
            previous_group = self.label_group.setdefault(label, group)
            if previous_group != group:
                raise ValueError(f"Language-ID label {label!r} appears in multiple groups")
            self.group_correct[group] += is_correct
            self.group_total[group] += 1
        answer = (prediction.get("generation") or "").strip() or None
        self._compute_pass_at_k(predictions, [answer])
        self._compute_majority_at_k(predictions, [answer])

    def get_metrics(self):
        metrics = super().get_metrics()
        label_accuracy = {
            label: self.label_correct[label] / count for label, count in sorted(self.label_total.items())
        }
        group_labels = defaultdict(list)
        for label, accuracy in label_accuracy.items():
            if label in self.label_group:
                group_labels[self.label_group[label]].append(accuracy)

        for values in metrics.values():
            values.pop("correct", None)
            values["accuracy_micro"] = round(100.0 * self.correct / self.total, 2) if self.total else 0.0
            values["accuracy_macro"] = (
                round(100.0 * sum(label_accuracy.values()) / len(label_accuracy), 2) if label_accuracy else 0.0
            )
            values["num_labels"] = len(label_accuracy)
            for label, accuracy in label_accuracy.items():
                values[f"accuracy::{label}"] = round(100.0 * accuracy, 2)
            for group, total in sorted(self.group_total.items()):
                values[f"accuracy_group::{group}"] = round(100.0 * self.group_correct[group] / total, 2)
                values[f"accuracy_group_macro::{group}"] = round(
                    100.0 * sum(group_labels[group]) / len(group_labels[group]), 2
                )
        return metrics
