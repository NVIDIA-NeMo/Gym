# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire SkillsBench's worker-side OpenHands agent to Gym's runtime services."""

import json
from pathlib import Path

from harbor.models.job.config import JobConfig
from harbor.models.trial.result import TrialResult
from ray.exceptions import RayTaskError

from benchmarks.skillsbench.observations import agent_observations, verifier_exit_code
from benchmarks.skillsbench.verifier import verifier_infrastructure_failure
from nemo_gym.config import ModelServerRef, get_first_server_config_dict, get_global_config_dict
from nemo_gym.sandbox import resolve_provider_config, resolve_provider_metadata
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.harbor_agent_general.app import (
    HarborAgent,
    HarborAgentConfig,
    HarborRunRequest,
    HarborTrialError,
    HarborVerifyResponse,
)


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
        # Gym materializes repeat seeds in Responses metadata; the third-party
        # chat client cannot inherit that metadata automatically.
        metadata = body.responses_create_params.metadata or {}
        extra_body = json.loads(metadata.get("extra_body", "{}"))
        if "seed" in extra_body:
            kwargs["responses_create_params"]["seed"] = extra_body["seed"]

    def success_response(self, body: HarborRunRequest, trial_dir: Path) -> HarborVerifyResponse:
        reason = verifier_infrastructure_failure(trial_dir)
        if reason is not None:
            # Preserve the official reward as evidence, but invalidate resume of
            # this inconclusive trial. Gym's existing bounded retry policy starts
            # a fresh sandbox; it must not count this result as a model zero.
            (trial_dir / "result.json").rename(trial_dir / "result.failed.json")
            raise HarborTrialError("VerifierInfrastructureError", reason, str(trial_dir.resolve()))
        response = super().success_response(body, trial_dir)
        trial = TrialResult.model_validate_json((trial_dir / "result.json").read_text())
        updates = {}
        code = verifier_exit_code(trial_dir, trial)
        if code is not None:
            # The existing test-execution evidence contract binds these facts
            # to the collected response and reward. A task test can exit nonzero
            # and still grade the answer; use Harbor's reward unchanged.
            updates.update(evaluation_completed=True, verifier_exit_code=code)
        observations = agent_observations(trial_dir, trial, response, self.config.model_server)
        if observations is not None:
            updates["ng_agent_observations"] = observations
        return response.model_copy(update=updates)

    def failure_response(self, body: HarborRunRequest, err: Exception) -> HarborVerifyResponse:
        response = super().failure_response(body, err)
        cause = err.cause if isinstance(err, RayTaskError) else err
        details = {"verification_completed": False, "status": "infrastructure_error"}
        updates = {
            "_ng_failure_class": "skillsbench_infrastructure",
            "mask_sample": True,
            "failure_kind": "skillsbench:infrastructure",
        }
        if isinstance(cause, HarborTrialError):
            details.update(exception_type=cause.exception_type, trial_dir=cause.trial_dir)
            # Only a typed agent deadline qualifies. Never infer a deadline
            # from a substring in a provider error or from a deleted sandbox.
            if cause.exception_type == "AgentTimeoutError":
                details.update(status="agent_timeout", score_origin="timeout_policy")
                updates.update(failure_kind="agent_timeout", evaluation_completed=True, mask_sample=False)
            elif cause.exception_type == "SkillsBenchResourceUnattributedError":
                details.update(status="resource_unattributed", requires_review=True)
                updates.update(_ng_failure_class="skillsbench_resource_unattributed", _ng_failure_terminal=True)
        # Explicit deadlines are completed zero-score outcomes.
        # Unattributed resource deaths stay unscored and terminal pending review.

        # Infrastructure and verifier failures remain unscored and retryable.
        payload = response.model_dump() | updates | {"evaluation_outcome": details}
        if updates.get("evaluation_completed"):
            payload.pop("_ng_failure_class", None)
        return HarborVerifyResponse.model_validate(payload)


if __name__ == "__main__":
    SkillsBenchHarborAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = SkillsBenchHarborAgent.run_webserver()  # noqa: F401
