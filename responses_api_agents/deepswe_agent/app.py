# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gym Responses API agent that runs DeepSWE with Pier and OpenSandbox."""

import asyncio
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from responses_api_agents.harbor_agent.app import (
    HarborAgent,
    HarborAgentConfig,
    runner_ray_remote,
)


OPENSANDBOX_API_KEY_ENV = "OPENSANDBOX_API_KEY"  # pragma: allowlist secret
POLICY_API_KEY_ENV = "POLICY_API_KEY"  # pragma: allowlist secret


class DeepSWEAgentConfig(HarborAgentConfig):
    sandbox_provider: dict[str, Any]
    sandbox_spec: dict[str, Any]
    harbor_agent_env: dict[str, str] | None = None


def _ensure_pier_litellm_compat() -> None:
    """Supply the one model registry Pier 0.3 expects from newer LiteLLM."""
    import litellm

    if not hasattr(litellm, "zai_models"):
        litellm.zai_models = set()


async def run_pier_job(job_config_dict: dict[str, Any]) -> str:
    """Run a one-task Pier job and recover its trial directory on failure."""
    _ensure_pier_litellm_compat()

    from pier.job import Job
    from pier.models.job.config import JobConfig

    config = JobConfig(**job_config_dict)
    job = await Job.create(config)
    job_error: Exception | None = None
    try:
        await job.run()
    except Exception as exc:
        job_error = exc

    job_dir = config.jobs_dir / config.job_name
    if job_dir.exists():
        for trial_dir in job_dir.iterdir():
            if trial_dir.is_dir() and (trial_dir / "result.json").exists():
                # Ray workers and the API server may use different working directories.
                return str(trial_dir.resolve())
    if job_error is not None:
        raise job_error
    raise FileNotFoundError(f"No Pier trial result found in {job_dir}")


_RAY_WORKER_EVENT_LOOP: asyncio.AbstractEventLoop | None = None


def _run_pier_job_sync(job_config_dict: dict[str, Any]) -> str:
    global _RAY_WORKER_EVENT_LOOP
    if _RAY_WORKER_EVENT_LOOP is None or _RAY_WORKER_EVENT_LOOP.is_closed():
        _RAY_WORKER_EVENT_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_RAY_WORKER_EVENT_LOOP)
    return _RAY_WORKER_EVENT_LOOP.run_until_complete(run_pier_job(job_config_dict))


def _provider_without_secret(provider: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    sanitized = deepcopy(provider)
    opensandbox = sanitized.get("opensandbox")
    if not isinstance(opensandbox, dict):
        return sanitized, None
    connection = opensandbox.get("connection")
    if not isinstance(connection, dict):
        return sanitized, None
    api_key = connection.pop("api_key", None)
    return sanitized, str(api_key) if api_key else None


class DeepSWEAgent(HarborAgent):
    """Harbor-compatible DeepSWE runner using Pier's v1.1 task lifecycle."""

    config: DeepSWEAgentConfig

    async def _run_job(self, job_config_dict: dict[str, Any]) -> str:
        runner = runner_ray_remote
        runner_options: dict[str, Any] = {}
        if self.config.harbor_ray_task_num_cpus is not None:
            runner_options["num_cpus"] = self.config.harbor_ray_task_num_cpus
        _, configured_key = _provider_without_secret(self.config.sandbox_provider)
        api_key = configured_key or os.getenv(OPENSANDBOX_API_KEY_ENV)
        runtime_env_vars: dict[str, str] = {}
        if api_key:
            runtime_env_vars[OPENSANDBOX_API_KEY_ENV] = api_key
        policy_api_key = os.getenv(POLICY_API_KEY_ENV)
        if policy_api_key:
            runtime_env_vars[POLICY_API_KEY_ENV] = policy_api_key
        if runtime_env_vars:
            runner_options["runtime_env"] = {
                "py_executable": os.sys.executable,
                "env_vars": runtime_env_vars,
            }
        if runner_options:
            runner = runner.options(**runner_options)
        params = {"job_config_dict": job_config_dict}
        return await runner.remote(_run_pier_job_sync, params)

    def _build_job_config(
        self,
        dataset_alias: str,
        task_name: str,
        model_name: str,
        api_base: str,
        job_name: str,
        jobs_dir: Path,
        responses_create_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from pier.models.job.config import DatasetConfig, JobConfig
        from pier.models.trial.config import AgentConfig, EnvironmentConfig, VerifierConfig

        del api_base, responses_create_params
        dataset_source = self.config.harbor_datasets.get(dataset_alias)
        if dataset_source is None or not dataset_source.local_dataset_path:
            available = ", ".join(sorted(self.config.harbor_datasets))
            raise ValueError(f"Unknown local dataset {dataset_alias!r}. Available datasets: [{available}]")

        agent_name = self.config.harbor_agent_name
        if not agent_name:
            raise ValueError("DeepSWE requires a Pier agent name")

        agent_env = deepcopy(self.config.harbor_agent_env or {})
        if agent_name == "mini-swe-agent":
            agent_env.setdefault("OPENAI_API_KEY", "${POLICY_API_KEY}")
            agent_env.setdefault("MSWEA_API_KEY", "${POLICY_API_KEY}")

        provider, _ = _provider_without_secret(self.config.sandbox_provider)
        environment_kwargs = {
            "provider": provider,
            "spec": deepcopy(self.config.sandbox_spec),
        }

        config = JobConfig(
            job_name=job_name,
            jobs_dir=jobs_dir,
            n_concurrent_trials=1,
            quiet=True,
            timeout_multiplier=self.config.harbor_timeout_multiplier or 1.0,
            environment=EnvironmentConfig(
                import_path=("responses_api_agents.deepswe_agent.opensandbox_environment:PierOpenSandboxEnvironment"),
                kwargs=environment_kwargs,
            ),
            verifier=VerifierConfig(
                override_timeout_sec=self.config.harbor_verifier_override_timeout,
                max_timeout_sec=self.config.harbor_verifier_max_timeout,
            ),
            agents=[
                AgentConfig(
                    name=agent_name,
                    model_name=model_name,
                    override_timeout_sec=self.config.harbor_agent_override_timeout,
                    max_timeout_sec=self.config.harbor_agent_max_timeout,
                    kwargs=deepcopy(self.config.harbor_agent_kwargs or {}),
                    env=agent_env,
                )
            ],
            datasets=[
                DatasetConfig(
                    path=Path(dataset_source.local_dataset_path),
                    task_names=[task_name],
                )
            ],
        )
        return config.model_dump(mode="json")


if __name__ == "__main__":
    DeepSWEAgent.run_webserver()
