# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire SkillsBench's worker-side OpenHands agent to Gym's runtime services."""

import json
from pathlib import Path

from harbor.models.job.config import JobConfig

from nemo_gym.config import ModelServerRef, get_global_config_dict
from nemo_gym.sandbox import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import get_first_server_config_dict
from responses_api_agents.harbor_agent_general.app import HarborAgent, HarborAgentConfig, HarborRunRequest


class SkillsBenchHarborAgentConfig(HarborAgentConfig):
    model_server: ModelServerRef
    sandbox_provider: str | None = None


class SkillsBenchHarborAgent(HarborAgent):
    """Keep inference beside Gym while Harbor runs tools in remote sandboxes."""

    ray_enabled = True
    config: SkillsBenchHarborAgentConfig

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        global_config = get_global_config_dict()
        model = get_first_server_config_dict(global_config, self.config.model_server.name)
        self.config.job.agents[0].kwargs["api_base"] = f"http://{model['host']}:{model['port']}/v1"
        if self.config.sandbox_provider is not None:
            provider = resolve_provider_config(self.config.sandbox_provider, global_config)
            metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config)
            self.config.harbor_worker_env["SKILLSBENCH_SANDBOX_CONFIG"] = json.dumps(
                {**provider, "default_metadata": metadata}
            )
        # The pinned checkout may be read-only in a batch job. Keep Harbor's
        # evidence with Gym's model captures so the run collector retains it.
        capture_dir = global_config.get("model_call_capture_dir")
        if capture_dir and Path(capture_dir).is_absolute():
            self.config.job.jobs_dir = Path(capture_dir).parent / "harbor-jobs"

    def _configure_job_for_run(self, job_config: JobConfig, body: HarborRunRequest) -> None:
        root = self.config.job.agents[0].kwargs["api_base"].removesuffix("/v1")
        identity = body.model_dump(by_alias=True)
        # This internal field is deliberately excluded from request serialization.
        identity["_ng_rollout_id"] = body.capture_rollout_id
        job_config.agents[0].kwargs["api_base"] = self.base_url_for_run(root, identity) + "/v1"
        kwargs = job_config.agents[0].kwargs
        kwargs["responses_create_params"] = {
            **(kwargs.get("responses_create_params") or {}),
            **body.responses_create_params.model_dump(
                include={"temperature", "top_p", "max_output_tokens"}, exclude_none=True
            ),
        }


if __name__ == "__main__":
    SkillsBenchHarborAgent.run_webserver()
