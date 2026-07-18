# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LAB-local compatibility wrapper around NeMo Gym's shared Harbor bridge."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import FastAPI

from responses_api_agents.harbor_agent.app import HarborAgent


REPO_ROOT = Path(__file__).resolve().parents[2]


class LegalAgentBenchHarborBridge(HarborAgent):
    """Keep LAB-specific route and path compatibility out of the shared bridge."""

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        return app

    def _get_results_output_dir(self, policy_model_name: str, dataset_alias: str, run_timestamp: datetime) -> Path:
        date_key = run_timestamp.strftime("%Y%m%d")
        dataset_key = self._sanitize_path_component(dataset_alias)
        model_key = self._sanitize_path_component(self._extract_model_name(policy_model_name))
        return REPO_ROOT / "results" / "legal_agent_bench" / "runs" / date_key / dataset_key / model_key

    def _get_jobs_output_dir(self, policy_model_name: str, dataset_alias: str, run_timestamp: datetime) -> Path:
        date_key = run_timestamp.strftime("%Y%m%d")
        dataset_key = self._sanitize_path_component(dataset_alias)
        model_key = self._sanitize_path_component(self._extract_model_name(policy_model_name))
        root = Path(self.config.harbor_jobs_dir).expanduser()
        if not root.is_absolute():
            root = REPO_ROOT / root
        return root / date_key / dataset_key / model_key


if __name__ == "__main__":
    LegalAgentBenchHarborBridge.run_webserver()
