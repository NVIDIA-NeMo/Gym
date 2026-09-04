# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""swe_if_agents: SWE-bench rollouts with injected instruction-following (IF) constraints, graded in the gym.

A thin wrapper around ``responses_api_agents.swe_agents`` (the OpenHands SWE agent). Every dataset row is a SWE
task plus, in its request metadata, the exact instruction surfaces the episode must run under and the constraints
to grade:

* ``tool_name_overrides``: the tool-name binding the episode exposes (exported to nv-OpenHands as
  ``TOOL_NAME_OVERRIDES``);
* ``system_prompt_template_text`` / ``user_prompt_template_text``: the prompts with the injected instruction,
  mounted per row;
* ``replay_observation_suffix``: an instruction appended to one recorded tool output of a replayed prefix
  (mid-task injection);
* ``sdg_item``: the constraints (verifier parameters), graded on the model-generated turns after the episode.

The task outcome reward is untouched: ``reward`` is the SWE-bench verdict. The IF grades are attached as
``if_constraints`` (one record per constraint, per gradable step) for downstream aggregation. Grading semantics
live in ``if_constraints/`` (a vendored copy of the recipe verifier; see the README).

Requires an nv-OpenHands checkout that understands ``TOOL_NAME_OVERRIDES`` and ``observation_suffix`` (the fork
pinned in ``configs/swebench_opencode_if.yaml``).
"""

from typing import Any, Dict, List, Optional

import orjson
from pydantic import Field

from responses_api_agents.swe_agents import app as swe
from responses_api_agents.swe_if_agents.hooks import (
    normalize_tool_name_overrides,
    tag_replay_observation_suffix,
    write_row_templates,
)
from responses_api_agents.swe_if_agents.if_constraints import grade_row


class SWEIFWrapperConfig(swe.SWEBenchWrapperConfig):
    if_grading: bool = Field(
        default=True,
        description=(
            "Grade the row's sdg_item constraints on the model-generated turns and attach if_constraints to the "
            "verify response."
        ),
    )
    empty_response_retries: int = Field(
        default=0,
        description=(
            "Exported to the agent as OPENCODE_EMPTY_RESPONSE_RETRIES: the OpenCode agent of the pinned nv-OpenHands fork "
            "re-issues the identical request up to this many times when the model returns neither content nor tool calls "
            "(a reasoning-only turn), instead of ending the episode on it. 0 (default) keeps the harness behaviour."
        ),
    )


class SWEIFVerifyResponse(swe.SWEBenchVerifyResponse):
    # One record per constraint: {id, trigger, match, no_answer, instruction, n_steps, n_pass, n_silent, step_avg,
    # all_pass, graded_turns, continuation_only, steps: [{turn, reward, detail}]}. None when the row has no
    # constraints.
    if_constraints: Optional[List[Dict[str, Any]]] = None


class SWEIFWrapper(swe.SWEBenchWrapper):
    config: SWEIFWrapperConfig

    # ---- mid-task injection: tag the replayed tool message the instruction was appended to
    def _maybe_build_replay_messages(self, body: swe.NeMoGymResponseCreateParamsNonStreaming) -> Optional[str]:
        replay_json = super()._maybe_build_replay_messages(body)
        spec = (body.metadata or {}).get("replay_observation_suffix") if body.metadata else None
        if not spec:
            return replay_json
        if replay_json is None:
            raise ValueError("replay_observation_suffix was given but the request carries no trajectory to replay")
        messages = orjson.loads(replay_json)
        tag_replay_observation_suffix(messages, spec)
        return orjson.dumps(messages).decode()

    # ---- per-row instruction surfaces (tool binding, prompt templates) and the agent environment
    def _setup_params(self, body: swe.NeMoGymResponseCreateParamsNonStreaming):
        params, dataset_processor = super()._setup_params(body)
        md = body.metadata or {}
        changed = False
        agent_env: Dict[str, str] = {}
        overrides = normalize_tool_name_overrides(md.get("tool_name_overrides"))
        if overrides:
            agent_env["TOOL_NAME_OVERRIDES"] = overrides
        if self.config.empty_response_retries > 0:
            agent_env["OPENCODE_EMPTY_RESPONSE_RETRIES"] = str(self.config.empty_response_retries)
        if agent_env:
            params.resolved_agent_env = agent_env
            changed = True
        sp_path, up_path = write_row_templates(
            params.persistent_dir, md.get("system_prompt_template_text"), md.get("user_prompt_template_text")
        )
        if sp_path:
            params.resolved_system_prompt_template = sp_path
            changed = True
        if up_path:
            params.resolved_user_prompt_template = up_path
            changed = True
        if changed:
            if self.config.agent_framework != "openhands":
                raise ValueError(
                    "swe_if_agents needs agent_framework: openhands "
                    "(per-row tool names and templates are OpenHands features)"
                )
            # The base built the agent command before it knew about the row's surfaces and the agent environment:
            # rebuild from the amended params so the mounts and exports see them.
            params.agent_command = swe.OpenHandsHarnessProcessor(config=params).get_run_command()
            params.agent_apptainer_command_str = self._build_apptainer_command(params, params.agent_command)
            params.agent_script = params.agent_script_path.read_text()
        return params, dataset_processor

    # ---- grading: the outcome reward stays; the IF grades ride along
    async def run(self, body: swe.BaseRunRequest) -> SWEIFVerifyResponse:
        base = await super().run(body)
        records = None
        if self.config.if_grading:
            has_output = base.response is not None and base.response.output
            output_items = [o.model_dump() for o in base.response.output] if has_output else []
            # responses_create_params is validated into a model by the verify response; accept a dict too
            rcp = base.responses_create_params
            raw_input = rcp.get("input") if isinstance(rcp, dict) else getattr(rcp, "input", None)
            input_items = [i.model_dump() if hasattr(i, "model_dump") else i for i in (raw_input or [])]
            records = grade_row(body.responses_create_params.metadata or {}, input_items, output_items)
        return SWEIFVerifyResponse(**base.model_dump(), if_constraints=records)


if __name__ == "__main__":
    SWEIFWrapper.run_webserver()
