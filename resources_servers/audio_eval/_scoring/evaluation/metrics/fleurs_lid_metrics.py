# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-1 FLEURS-LID metrics with locale and geographic reporting."""

from resources_servers.audio_eval._scoring.evaluation.metrics.language_id_metrics import LanguageIDMetrics


class FleursLIDMetrics(LanguageIDMetrics):
    def get_metrics(self):
        metrics = super().get_metrics()
        for values in metrics.values():
            values["num_locales"] = values.pop("num_labels")
        return metrics
