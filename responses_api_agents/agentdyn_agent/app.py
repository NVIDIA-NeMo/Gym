# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentDyn backend for the shared AgentDojo-family NeMo Gym adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import agentdojo
import agentdojo.agent_pipeline.agent_pipeline as agentdyn_pipeline
import openai
from fastapi import Body
from huggingface_hub import snapshot_download
from pydantic import ConfigDict

from responses_api_agents.agentdojo_family.app import (
    AgentDojoFamilyAgent,
    AgentDojoFamilyAgentConfig,
    AgentDojoFamilyRunRequest,
    AgentDojoFamilyVerifyResponse,
)
from responses_api_agents.agentdojo_family.model_bridge import NeMoGymOpenAIProxy


AGENTDYN_SUITES = ("shopping", "github", "dailylife")
# Upstream's full DEFENSES list, in upstream's own names.
AGENTDYN_DEFENSES = (
    "prompt_guard_2_detector",
    "piguard_detector",
    "camel",
    "progent",
    "drift",
    "transformers_pi_detector",
    "spotlighting_with_delimiting",
    "repeat_user_prompt",
    "tool_filter",
)
PROMPT_GUARD_2_UPSTREAM_MODEL = "meta-llama/Llama-Prompt-Guard-2-86M"
PIGUARD_UPSTREAM_MODEL = "leolee99/PIGuard"
TRANSFORMERS_PI_UPSTREAM_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
# Detector defenses hard-code their classifier upstream and resolve moving Hub `main` at
# construction time. Both are pinned here instead, so a published treatment names the exact
# weights it scored with. PIGuard additionally ships custom modeling code that upstream loads
# with `trust_remote_code=True`, which makes the pin a supply-chain control, not just provenance.
DETECTOR_UPSTREAM_MODELS = {
    "prompt_guard_2_detector": PROMPT_GUARD_2_UPSTREAM_MODEL,
    "piguard_detector": PIGUARD_UPSTREAM_MODEL,
    "transformers_pi_detector": TRANSFORMERS_PI_UPSTREAM_MODEL,
}
ROUTED_DEFENSES = frozenset({"camel", "progent", "drift"})


class AgentDynAgentConfig(AgentDojoFamilyAgentConfig):
    # One rollout at a time, enforced rather than advised. The routed defenses redirect their
    # internal OpenAI clients with `mock.patch` over process globals (`openai.OpenAI`,
    # `os.environ`), and those scopes are not thread-safe: overlapping rollouts restore each
    # other's patches out of order, which can record calls on the wrong rollout or hand a
    # defense the real OpenAI client. Scale out with more agent processes instead.
    concurrency: Literal[1] = 1
    metric_prefix: str = "agentdyn"
    defense_model_alias: str | None = "gpt-4o-2024-08-06"
    prompt_guard_2_model_name: str = PROMPT_GUARD_2_UPSTREAM_MODEL
    prompt_guard_2_model_revision: str | None = None
    prompt_guard_2_local_path: str | None = None
    piguard_model_name: str = PIGUARD_UPSTREAM_MODEL
    piguard_model_revision: str | None = None
    piguard_local_path: str | None = None
    transformers_pi_model_name: str = TRANSFORMERS_PI_UPSTREAM_MODEL
    transformers_pi_model_revision: str | None = None
    transformers_pi_local_path: str | None = None

    def detector_source(self, defense: str) -> tuple[str, str | None, str | None]:
        """Return the (repository, revision, local cache override) backing a detector defense."""
        if defense == "prompt_guard_2_detector":
            return (
                self.prompt_guard_2_model_name,
                self.prompt_guard_2_model_revision,
                self.prompt_guard_2_local_path,
            )
        if defense == "transformers_pi_detector":
            return (
                self.transformers_pi_model_name,
                self.transformers_pi_model_revision,
                self.transformers_pi_local_path,
            )
        return (self.piguard_model_name, self.piguard_model_revision, self.piguard_local_path)


class AgentDynRunRequest(AgentDojoFamilyRunRequest):
    model_config = ConfigDict(extra="allow")

    suite: Literal["shopping", "github", "dailylife"]
    defense: (
        Literal[
            "prompt_guard_2_detector",
            "piguard_detector",
            "camel",
            "progent",
            "drift",
            "transformers_pi_detector",
            "spotlighting_with_delimiting",
            "repeat_user_prompt",
            "tool_filter",
        ]
        | None
    ) = None
    external_model_base_url: str | None = None
    detector_model_name: str | None = None
    detector_model_revision: str | None = None


class AgentDynAgent(AgentDojoFamilyAgent):
    config: AgentDynAgentConfig

    async def run(self, body: AgentDynRunRequest = Body()) -> AgentDojoFamilyVerifyResponse:
        defense = body.defense or self.config.default_defense
        if defense in ROUTED_DEFENSES:
            body = body.model_copy(
                update={
                    "external_model_base_url": self.resolve_model_base_url(
                        self.config.model_server.name,
                        self.rollout_id_from_run(body),
                    )
                }
            )
        elif defense in DETECTOR_UPSTREAM_MODELS:
            detector_model_name, detector_model_revision, _ = self.config.detector_source(defense)
            body = body.model_copy(
                update={
                    "detector_model_name": detector_model_name,
                    "detector_model_revision": detector_model_revision,
                }
            )
        return await super().run(body)

    def _run_agentdojo(self, body, bridge, benchmark_version):
        if body.defense in DETECTOR_UPSTREAM_MODELS:
            upstream_model_name = DETECTOR_UPSTREAM_MODELS[body.defense]
            repo_id, revision, local_path = self.config.detector_source(body.defense)
            model_path = local_path or snapshot_download(repo_id=repo_id, revision=revision)
            original_detector = agentdyn_pipeline.TransformersBasedPIDetector

            def build_pinned_detector(*args, **kwargs):
                if kwargs.get("model_name") == upstream_model_name:
                    kwargs["model_name"] = model_path
                return original_detector(*args, **kwargs)

            with patch.object(agentdyn_pipeline, "TransformersBasedPIDetector", build_pinned_detector):
                return super()._run_agentdojo(body, bridge, benchmark_version)
        if body.defense not in ROUTED_DEFENSES:
            return super()._run_agentdojo(body, bridge, benchmark_version)
        if body.external_model_base_url is None:
            raise RuntimeError(f"{body.defense} requires the routed NeMo model-server URL")
        model_alias = self.config.defense_model_alias or self.config.attack_model_alias
        routed_environment = {
            "OPENAI_BASE_URL": body.external_model_base_url,
            "OPENAI_API_KEY": "EMPTY",
            "SECAGENT_POLICY_MODEL": model_alias,
            "SECAGENT_IGNORE_UPDATE_ERROR": "False",
        }
        installed_defenses = Path(agentdojo.__file__).resolve().parent / "defenses"
        with (
            patch.dict(os.environ, routed_environment, clear=False),
            patch.object(agentdyn_pipeline, "_defenses_root", return_value=installed_defenses),
            patch.object(openai, "OpenAI", side_effect=lambda *args, **kwargs: NeMoGymOpenAIProxy(bridge)),
        ):
            return super()._run_agentdojo(body, bridge, benchmark_version)


if __name__ == "__main__":
    AgentDynAgent.run_webserver()
