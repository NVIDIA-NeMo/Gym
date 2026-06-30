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
from pathlib import Path
from typing import Any, Optional

from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest

from app import (
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
    dump_dir: Optional[str] = Field(
        default=None,
        description=(
            "If set, write each attempt's input prompt + output + metrics + carried "
            "verify_feedback as readable JSON to <dump_dir>/<group_hash>/attempt_<k>.json, "
            "for debugging the constructed refine context. Off (None) by default."
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
                        "verify_feedback": metadata.get("verify_feedback"),
                    }
                )
                self._dump_attempt(group_hash, k, attempts[-1])

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

        Attempt 0 is the original task. For attempt>0 the seed (prior diff + test
        failures) is injected by APPENDING it to the SWE instance's
        ``problem_statement`` inside ``responses_create_params.metadata["instance_dict"]``.
        That is the field the OpenHands runner renders into the agent's task prompt
        (run_infer.py:get_instruction -> swe_default.j2 ``{{ instance.problem_statement }}``).
        NOTE: appending to ``responses_create_params.input`` does NOT reach the
        OpenHands SWE agent (it rebuilds the prompt from the instance), so the seed
        must go through problem_statement.

        TODO(refine): when skip_reset_after_first is wired, the workspace already
        carries the accumulated patch; the seed should then describe *what changed
        and what failed* rather than restate the whole task.
        """
        if attempt_idx == 0 or not prior_summary:
            return body

        new_body = body.model_copy(deep=True)
        metadata = new_body.responses_create_params.metadata or {}
        instance_raw = metadata.get("instance_dict")
        if not instance_raw:
            print(
                "[refine] WARNING: no instance_dict in metadata; refine seed NOT injected",
                flush=True,
            )
            return new_body

        instance = json.loads(instance_raw)
        original = instance.get("problem_statement", "")
        instance["problem_statement"] = (
            f"{original}\n\n"
            "=== Refinement feedback from your previous attempt ===\n"
            f"{prior_summary}"
        )
        metadata["instance_dict"] = json.dumps(instance)
        # Keep the top-level mirror (if present) consistent with the instance copy.
        if "problem_statement" in metadata:
            metadata["problem_statement"] = instance["problem_statement"]
        new_body.responses_create_params.metadata = metadata
        return new_body

    def _summarize_prior(self, attempts: list[dict]) -> str:
        """Build the seed carried into the next attempt from prior attempt(s).

        Carries the prior attempt's diff plus the raw verify output (the failing
        tests / tracebacks from the eval step), so the next attempt can fix the
        actual failures instead of guessing. The verify log is already bounded
        upstream (tail, see _inner_responses). Keep deterministic.

        TODO(refine): when the workspace persists across attempts (skip_reset),
        the seed should describe what changed and what still fails rather than
        restate the whole diff; also consider stripping history thinking.
        """
        last = attempts[-1]
        metrics = last["metrics"]
        patch = (metrics.model_patch or "").strip()
        verify_feedback = (last.get("verify_feedback") or "").strip()

        # Bound the carried-over context to carry_over_token_budget. Without this,
        # a large prior diff (common on hard problems) makes the next attempt's
        # first-turn prompt exceed the model context window; vLLM then rejects the
        # request, the attempt produces NO generation, the rollout is dropped, and
        # the async replay buffer dead-locks waiting for the missing group. We keep
        # the (small, already-tail-bounded) verify_feedback intact and truncate the
        # diff in the middle to fit. ~4 chars/token (conservative) since no tokenizer here.
        budget_chars = max(2000, self.config.carry_over_token_budget * 4)
        verify_feedback = verify_feedback[: budget_chars // 2]
        patch = self._truncate_middle(patch, max(1000, budget_chars - len(verify_feedback) - 500))

        parts = [
            "Your previous attempt did not resolve the issue.",
            "Here is the diff you produced so far:",
            f"```diff\n{patch}\n```",
        ]
        if verify_feedback:
            parts.append(
                "Running the tests on that diff produced the following output. "
                "Use the failures below to fix the patch:"
            )
            parts.append(f"```\n{verify_feedback}\n```")
        parts.append("Continue refining the patch so the failing tests pass.")
        return "\n\n".join(parts) + "\n"

    @staticmethod
    def _truncate_middle(text: str, max_chars: int) -> str:
        """Truncate the middle of text to <= max_chars, keeping head + tail (diff
        head shows the changed files/hunks, tail shows the latest edits)."""
        if len(text) <= max_chars:
            return text
        keep = max(0, (max_chars - 60) // 2)
        omitted = len(text) - 2 * keep
        return f"{text[:keep]}\n... [diff truncated: {omitted} chars omitted to fit context budget] ...\n{text[-keep:]}"

    def _dump_attempt(self, group_hash: str, attempt_idx: int, attempt: dict) -> None:
        """Best-effort debug dump of one attempt's prompt/output to disk.

        Writes <dump_dir>/<group_hash>/attempt_<k>.json (readable text) so the
        constructed refine context can be inspected — in particular that attempt
        k>0's input carries the prior diff + verify_feedback. No-op unless
        config.dump_dir is set. Never raises (debug side-channel must not break
        the rollout).
        """
        dump_dir = self.config.dump_dir
        if not dump_dir:
            return
        try:
            response = attempt["response"]
            metrics = attempt["metrics"]
            record = {
                "group_hash": group_hash,
                "turn_idx": attempt_idx,
                "resolved": metrics.resolved,
                "model_patch": metrics.model_patch,
                # raw verify output produced by THIS attempt (fed into the next one)
                "verify_feedback": attempt.get("verify_feedback") or "",
                # the prompt this attempt actually ran on (for k>0 includes the
                # diff + verify_feedback seed appended by _build_attempt_body)
                "input": attempt["responses_create_params"].get("input"),
                "output": [o.model_dump() for o in response.output]
                if response.output
                else [],
            }
            out_dir = Path(dump_dir) / group_hash
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"attempt_{attempt_idx}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False, default=str)
            )
        except Exception as e:  # best-effort debug dump; never break the rollout
            print(f"[refine dump] failed for {group_hash} attempt {attempt_idx}: {e}", flush=True)


if __name__ == "__main__":
    SWEBenchRefineWrapper.run_webserver()
