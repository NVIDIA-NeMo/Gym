# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate official VoiceCodeBench WER, CTEM, TSR, and per-type CTEM."""

from resources_servers.audio_eval._scoring.evaluation.metrics.base import BaseMetrics


class VoiceCodeBenchMetrics(BaseMetrics):
    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError("VoiceCodeBench uses one raw batch transcript per recording")
        self.max_k = 1
        self.wer_errors = 0
        self.wer_ref_words = 0
        self.wer_hyp_words = 0
        self.substitutions = 0
        self.deletions = 0
        self.insertions = 0
        self.verified_entries = 0
        self.correct_entities = 0
        self.total_entities = 0
        self.successful_tasks = 0
        self.correct_by_type: dict[str, int] = {}
        self.total_by_type: dict[str, int] = {}
        self.revisions: set[str] = set()
        self.verifier_ids: set[str] = set()
        self.verifier_digests: set[str] = set()

    def reset(self):
        super().reset()
        self.wer_errors = 0
        self.wer_ref_words = 0
        self.wer_hyp_words = 0
        self.substitutions = 0
        self.deletions = 0
        self.insertions = 0
        self.verified_entries = 0
        self.correct_entities = 0
        self.total_entities = 0
        self.successful_tasks = 0
        self.correct_by_type = {}
        self.total_by_type = {}
        self.revisions = set()
        self.verifier_ids = set()
        self.verifier_digests = set()

    def _get_score_dict(self, prediction):
        return {"correct": bool(prediction.get("is_correct"))}

    def update_common_metrics(self, agg_dict):
        agg_dict["num_entries"] = self.total
        if self.avg_tokens > 0 and self.total:
            agg_dict["avg_tokens"] = int(self.avg_tokens / self.total)
        if self.max_end_time > float("-inf") and self.min_start_time < float("inf"):
            agg_dict["gen_seconds"] = int(self.max_end_time - self.min_start_time)

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("VoiceCodeBenchMetrics expects exactly one prediction per recording")
        super().update(predictions)
        prediction = predictions[0]
        self.wer_errors += int(prediction.get("voice_code_wer_errors") or 0)
        self.wer_ref_words += int(prediction.get("voice_code_wer_ref_words") or 0)
        self.wer_hyp_words += int(prediction.get("voice_code_wer_hyp_words") or 0)
        self.substitutions += int(prediction.get("voice_code_wer_substitutions") or 0)
        self.deletions += int(prediction.get("voice_code_wer_deletions") or 0)
        self.insertions += int(prediction.get("voice_code_wer_insertions") or 0)
        revision = prediction.get("voice_code_bench_revision")
        if revision:
            self.revisions.add(str(revision))
        if prediction.get("voice_code_entity_verifier_status") == "complete":
            self.verified_entries += 1
            self.correct_entities += int(prediction["voice_code_correct_entities"])
            self.total_entities += int(prediction["voice_code_total_entities"])
            self.successful_tasks += int(bool(prediction["voice_code_task_success"]))
            self.verifier_ids.add(str(prediction["voice_code_verifier_id"]))
            self.verifier_digests.add(str(prediction["voice_code_verifier_config_digest"]))
            for entity_type, count in prediction["voice_code_correct_by_type"].items():
                self.correct_by_type[entity_type] = self.correct_by_type.get(entity_type, 0) + int(count)
            for entity_type, count in prediction["voice_code_total_by_type"].items():
                self.total_by_type[entity_type] = self.total_by_type.get(entity_type, 0) + int(count)

        answers = [(prediction.get("generation") or "").strip() or None]
        self._compute_pass_at_k(predictions=predictions, predicted_answers=answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=answers)

    def get_metrics(self):
        metrics_dict = super().get_metrics()
        if len(self.revisions) != 1:
            raise ValueError(f"VoiceCodeBench metrics mixed revisions: {sorted(self.revisions)}")
        for metrics in metrics_dict.values():
            metrics["official_revision"] = next(iter(self.revisions))
            metrics["wer"] = round(100.0 * self.wer_errors / self.wer_ref_words, 2) if self.wer_ref_words else 0.0
            metrics["reference_words"] = self.wer_ref_words
            metrics["hypothesis_words"] = self.wer_hyp_words
            metrics["substitutions"] = self.substitutions
            metrics["deletions"] = self.deletions
            metrics["insertions"] = self.insertions
            metrics["entity_verified_entries"] = self.verified_entries
            if self.verified_entries:
                if self.verified_entries != self.total:
                    raise ValueError(
                        f"Partial VoiceCodeBench entity scoring: {self.verified_entries}/{self.total} entries"
                    )
                if len(self.verifier_ids) != 1 or len(self.verifier_digests) != 1:
                    raise ValueError("VoiceCodeBench metrics mixed verifier versions")
                metrics["verifier_id"] = next(iter(self.verifier_ids))
                metrics["verifier_config_digest"] = next(iter(self.verifier_digests))
                metrics["ctem"] = round(100.0 * self.correct_entities / self.total_entities, 2)
                metrics["tsr"] = round(100.0 * self.successful_tasks / self.verified_entries, 2)
                metrics["correct_entities"] = self.correct_entities
                metrics["total_entities"] = self.total_entities
                metrics["successful_tasks"] = self.successful_tasks
                for entity_type in sorted(self.total_by_type):
                    metrics[f"ctem::{entity_type}"] = round(
                        100.0 * self.correct_by_type.get(entity_type, 0) / self.total_by_type[entity_type], 2
                    )
        return metrics_dict
