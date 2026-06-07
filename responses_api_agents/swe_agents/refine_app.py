# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multi-turn (refine) SWE agent.
#
# Runs the same task as several sequential OpenHands attempts. Between attempts
# the prior attempt's context is compressed into a seed for the next one. Each
# attempt is an independent OpenHands episode (independent event stream → an
# internally-monotonic, independently-trainable sample). All attempts of one
# task share a `group_hash`, and the chain's final reward is broadcast to every
# attempt. Early-solved chains are padded to a fixed length with
# `loss_multiplier=0` so the per-task sample count is constant.
#
# This reuses all of SWEBenchWrapper's OpenHands machinery via `responses()`;
# only the per-attempt loop and the cross-attempt context handling are new.
#
# Status: orchestration + training contract (list return, group_hash, padding)
# are implemented. Two pieces are intentionally left as seams and require
# cluster iteration (see TODOs in _summarize_prior / _build_attempt_body):
#   1. Container persistence ("refine without restart"): keep one apptainer
#      instance alive across attempts and skip `git reset --hard` on attempts>1
#      (run_infer.py SKIP_INITIAL_RESET). Until that lands, attempts start from a
#      clean repo and only the textual seed carries over (== summary-style).
#   2. Real compression (strip history thinking + truncate tool outputs, or a
#      reasoning digest / SWE-Pruner-style line selection).

import copy
import hashlib
import json
from typing import Any, Optional

from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest

from .app import (
    SWEBenchMetrics,
    SWEBenchVerifyResponse,
    SWEBenchWrapper,
    SWEBenchWrapperConfig,
    SWEBenchWrapperInstanceConfig,
)


class SWEBenchRefineConfig(SWEBenchWrapperConfig):
    max_attempts: int = Field(
        default=2, description="Number of sequential OpenHands attempts per task (MVP=2)."
    )
    carry_over_token_budget: int = Field(
        default=40000,
        description="Target upper bound (tokens) for the context carried into the next attempt.",
    )
    skip_reset_after_first: bool = Field(
        default=False,
        description=(
            "If True, keep the workspace (accumulated patch) across attempts instead of "
            "git-resetting (true 'refine'). Requires the persistent-instance + "
            "SKIP_INITIAL_RESET change in the OpenHands runner; not yet wired."
        ),
    )


class SWEBenchRefineVerifyResponse(SWEBenchVerifyResponse):
    # extra="allow" so per-element fields consumed by the trainer (group_hash,
    # loss_multiplier, turn_idx) pass through model_dump into the gym result dict.
    model_config = ConfigDict(extra="allow")


def _input_to_jsonable(input_value: Any) -> Any:
    """Render responses_create_params.input to a stable JSON-able structure for hashing."""
    if isinstance(input_value, str):
        return input_value
    out = []
    for item in input_value or []:
        out.append(item.model_dump() if hasattr(item, "model_dump") else item)
    return out


class SWEBenchRefineWrapper(SWEBenchWrapper):
    config: SWEBenchRefineConfig

    async def run(self, body: BaseRunRequest) -> list[SWEBenchRefineVerifyResponse]:
        async with self._sem:
            base_params = body.responses_create_params
            base_params.parallel_tool_calls = True
            base_params.tool_choice = "auto"

            # Stable group key shared by every attempt and rollout of this task.
            group_hash = hashlib.md5(
                json.dumps(
                    _input_to_jsonable(base_params.input), sort_keys=True, ensure_ascii=False
                ).encode("utf-8")
            ).hexdigest()

            max_attempts = self.config.max_attempts
            attempts: list[dict] = []
            prior_summary: Optional[str] = None

            for k in range(max_attempts):
                attempt_body = self._build_attempt_body(body, k, prior_summary)
                response = await self.responses(attempt_body.responses_create_params)

                metadata, response.metadata = response.metadata, None
                metrics = SWEBenchMetrics.model_validate_json(metadata["metrics"])
                responses_create_params = attempt_body.responses_create_params.model_dump() | {
                    "input": json.loads(metadata["input"]),
                    "tools": [t.model_dump() for t in response.tools] if response.tools else [],
                }
                attempts.append(
                    {
                        "responses_create_params": responses_create_params,
                        "response": response,
                        "metrics": metrics,
                        "instance_config": metadata["instance_config"],
                    }
                )

                if metrics.resolved:
                    break  # chain solved → stop early, pad the rest
                if k < max_attempts - 1:
                    prior_summary = self._summarize_prior(attempts)

            # Chain-level reward = resolved by the (cumulative) final state; broadcast
            # to every attempt's sample. With fixed-length padding this keeps each
            # rollout's per-group count constant, so the GRPO baseline stays unbiased.
            chain_resolved = any(a["metrics"].resolved for a in attempts)
            reward = 1.0 if chain_resolved else 0.0

            results: list[SWEBenchRefineVerifyResponse] = []
            for k in range(max_attempts):
                is_padded = k >= len(attempts)
                src = attempts[k] if not is_padded else attempts[-1]
                metrics = src["metrics"]
                # Padded slots reuse the last real attempt; deep-copy the response so the
                # trainer's postprocess (which pops token-id fields) can't corrupt the original.
                response = (
                    src["response"].model_copy(deep=True) if is_padded else src["response"]
                )
                results.append(
                    SWEBenchRefineVerifyResponse(
                        responses_create_params=copy.deepcopy(src["responses_create_params"]),
                        response=response,
                        reward=reward,
                        **metrics.model_dump(),
                        instance_config=SWEBenchWrapperInstanceConfig.model_validate_json(
                            src["instance_config"]
                        ).model_dump(),
                        group_hash=group_hash,
                        loss_multiplier=0.0 if is_padded else 1.0,
                        turn_idx=k,
                        is_padded=is_padded,
                    )
                )
            return results

    def _build_attempt_body(
        self, body: BaseRunRequest, attempt_idx: int, prior_summary: Optional[str]
    ) -> BaseRunRequest:
        """Construct the BaseRunRequest for one attempt.

        Attempt 0 is the original task. For attempt>0 the compressed seed from the
        prior attempt(s) is injected as an extra user message.

        TODO(refine): when skip_reset_after_first is wired, the workspace already
        carries the accumulated patch; the seed should then describe *what changed
        and what failed* rather than restate the whole task.
        """
        if attempt_idx == 0 or not prior_summary:
            return body

        new_body = body.model_copy(deep=True)
        params = new_body.responses_create_params
        seed_msg = {"role": "user", "content": prior_summary}
        if isinstance(params.input, str):
            params.input = [{"role": "user", "content": params.input}, seed_msg]
        else:
            params.input = list(params.input) + [seed_msg]
        return new_body

    def _summarize_prior(self, attempts: list[dict]) -> str:
        """Build the seed carried into the next attempt from prior attempt(s).

        MVP: a minimal textual handoff (last patch + outcome). This is the seam for
        the real compression strategy (strip history thinking + truncate tool outputs
        to <= carry_over_token_budget, or a reasoning digest / SWE-Pruner line
        selection). Keep deterministic and bounded.
        """
        last = attempts[-1]
        metrics = last["metrics"]
        patch = (metrics.model_patch or "").strip()
        # TODO(refine): compress to carry_over_token_budget; add failing-test feedback.
        return (
            "Your previous attempt did not resolve the issue. "
            "Here is the diff you produced so far; continue refining it:\n\n"
            f"```diff\n{patch}\n```\n"
        )


if __name__ == "__main__":
    SWEBenchRefineWrapper.run_webserver()
