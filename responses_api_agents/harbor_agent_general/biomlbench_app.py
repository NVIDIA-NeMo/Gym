# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run BioML-bench's Terminus-2 agent beside Gym's model server, with tools in OpenSandbox."""

from harbor.models.job.config import JobConfig

from responses_api_agents.harbor_agent_general.app import HarborRunRequest
from responses_api_agents.harbor_agent_general.skillsbench_app import SkillsBenchHarborAgent


class BioMLBenchHarborAgent(SkillsBenchHarborAgent):
    """SkillsBench's wiring (model server URL per rollout, sandbox config for the worker) for Terminus-2."""

    # Gym starts Ray only for a class that declares it itself.
    ray_enabled = True

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        sandbox_config = self.config.harbor_worker_env.pop("SKILLSBENCH_SANDBOX_CONFIG", None)
        if sandbox_config is not None:
            self.config.harbor_worker_env["BIOMLBENCH_SANDBOX_CONFIG"] = sandbox_config

    def _configure_job_for_run(self, job_config: JobConfig, body: HarborRunRequest) -> None:
        super()._configure_job_for_run(job_config, body)
        # Terminus-2 takes temperature directly and other sampling settings as LiteLLM call kwargs.
        kwargs = job_config.agents[0].kwargs
        params = kwargs.pop("responses_create_params", {})
        if "temperature" in params:
            kwargs["temperature"] = params["temperature"]
        call_kwargs = dict(kwargs.get("llm_call_kwargs") or {})
        if "top_p" in params:
            call_kwargs["top_p"] = params["top_p"]
        if "max_output_tokens" in params:
            call_kwargs["max_tokens"] = params["max_output_tokens"]
        if call_kwargs:
            kwargs["llm_call_kwargs"] = call_kwargs


if __name__ == "__main__":
    BioMLBenchHarborAgent.run_webserver()
