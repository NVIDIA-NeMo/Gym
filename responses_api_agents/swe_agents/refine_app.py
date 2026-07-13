# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multi-turn (refine) SWE agent.
#
# Runs the same task as several sequential OpenHands refinement rounds. Between
# rounds the prior round's context is compressed into a seed for the next one. Each
# round is an independent OpenHands episode (independent event stream → an
# internally-monotonic, independently-trainable sample). All rounds of one
# task share a `group_hash`, and the chain's final reward is broadcast to every
# round. Early-solved chains are padded to a fixed length with
# `loss_multiplier=0` so the per-task sample count is constant.
#
# This reuses all of SWEBenchWrapper's OpenHands machinery via `responses()`;
# only the per-round loop and the cross-round context handling are new.
#
# Status: orchestration + training contract (list return, group_hash, padding)
# are implemented. Two pieces are intentionally left as seams and require
# cluster iteration (see TODOs in _summarize_prior / _build_refine_round_body):
#   1. Container persistence ("refine without restart"): keep one apptainer
#      instance alive across rounds and skip `git reset --hard` on later rounds
#      (run_infer.py SKIP_INITIAL_RESET). Until that lands, rounds start from a
#      clean repo and only the textual seed carries over (== summary-style).
#   2. Real compression (strip history thinking + truncate tool outputs, or a
#      reasoning digest / SWE-Pruner-style line selection).

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import AliasChoices, ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest

if __package__:
    from .app import (
        SWEBenchMetrics,
        SWEBenchVerifyResponse,
        SWEBenchWrapper,
        SWEBenchWrapperConfig,
        SWEBenchWrapperInstanceConfig,
    )
else:
    # Gym launches this file directly as an entrypoint, without package context.
    from app import (
        SWEBenchMetrics,
        SWEBenchVerifyResponse,
        SWEBenchWrapper,
        SWEBenchWrapperConfig,
        SWEBenchWrapperInstanceConfig,
    )


class SWEBenchRefineConfig(SWEBenchWrapperConfig):
    max_refine_rounds: int = Field(
        default=2,
        ge=1,
        validation_alias=AliasChoices("max_refine_rounds", "max_attempts"),
        description=(
            "Number of sequential OpenHands refinement rounds per task. "
            "The legacy max_attempts name is accepted for compatibility."
        ),
    )
    carry_over_token_budget: int = Field(
        default=40000,
        ge=0,
        description="Target upper bound (tokens) for context carried into the next round.",
    )
    refine_strategy: Literal["baseline"] = Field(
        default="baseline",
        description="Refine seed strategy. 'baseline' is eval-compatible refine v1.",
    )
    skip_reset_after_initial_round: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "skip_reset_after_initial_round", "skip_reset_after_first"
        ),
        description=(
            "If True, keep the workspace (accumulated patch) across rounds instead of "
            "git-resetting (true 'refine'). Requires the persistent-instance + "
            "SKIP_INITIAL_RESET change in the OpenHands runner; not yet wired."
        ),
    )
    dump_dir: Optional[str] = Field(
        default=None,
        description=(
            "If set, write each round's input prompt + output + metrics + carried "
            "verify_feedback as readable JSON to <dump_dir>/<group_hash>/round_<k>.json, "
            "for debugging the constructed refine context. Off (None) by default."
        ),
    )


class SWEBenchRefineVerifyResponse(SWEBenchVerifyResponse):
    # extra="allow" so per-element fields consumed by the trainer (group_hash,
    # loss_multiplier, refine_round_idx) pass through model_dump into the result dict.
    model_config = ConfigDict(extra="allow")


def _input_to_jsonable(input_value: Any) -> Any:
    """Render responses_create_params.input to a stable JSON-able structure for hashing."""
    if isinstance(input_value, str):
        return input_value
    out = []
    for item in input_value or []:
        out.append(item.model_dump() if hasattr(item, "model_dump") else item)
    return out


def _truncate_middle(text: str, max_tokens: int) -> str:
    """Middle-truncate text to approximately max_tokens using four chars/token."""
    if max_tokens <= 0:
        return text
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return (
        text[:head]
        + "\n...[diff truncated to fit carry-over budget]...\n"
        + text[-tail:]
    )


def _build_refine_v1_seed(patch: str, verify_feedback: str, max_patch_tokens: int) -> str:
    """Build the eval-compatible refine v1 text seed."""
    patch = _truncate_middle((patch or "").strip(), max_patch_tokens)
    verify_feedback = (verify_feedback or "").strip()

    parts = [
        "\n\n---\n"
        "Your previous automated attempt did NOT resolve the issue.",
        "Here is the diff you produced so far:",
        f"```diff\n{patch}\n```",
    ]
    if verify_feedback:
        parts.append(
            "Running the tests on that diff produced the following output. "
            "Use the failures below to fix the patch:"
        )
        parts.append(f"```\n{verify_feedback}\n```")
    parts.append(
        "Continue refining the patch so the failing tests pass. "
        "Review the diff, fix what is wrong, and produce a correct, complete patch."
    )
    return "\n\n".join(parts) + "\n"


def _append_seed_to_problem_metadata(metadata: dict[str, Any], seed: str) -> dict[str, Any]:
    """Append a refine seed to both task representations consumed by the harness."""
    refined_metadata = copy.deepcopy(metadata)
    original_problem = str(refined_metadata.get("problem_statement") or "")
    refined_metadata["problem_statement"] = original_problem.rstrip() + seed

    raw_instance_dict = refined_metadata.get("instance_dict")
    try:
        instance_dict = (
            json.loads(raw_instance_dict)
            if isinstance(raw_instance_dict, str)
            else copy.deepcopy(raw_instance_dict)
        )
    except (TypeError, json.JSONDecodeError):
        instance_dict = None

    if isinstance(instance_dict, dict):
        nested_problem = str(instance_dict.get("problem_statement") or original_problem)
        instance_dict["problem_statement"] = nested_problem.rstrip() + seed
        refined_metadata["instance_dict"] = json.dumps(instance_dict, ensure_ascii=False)

    return refined_metadata


def _build_chain_metrics(
    refine_rounds: list[dict], max_refine_rounds: int
) -> dict[str, Any]:
    """Return core metrics for one refinement chain."""
    resolved_at_refine_round = next(
        (
            idx
            for idx, refine_round in enumerate(refine_rounds)
            if refine_round["metrics"].resolved
        ),
        None,
    )
    initial_round_resolved = bool(refine_rounds[0]["metrics"].resolved)
    chain_resolved = resolved_at_refine_round is not None
    refine_continued = len(refine_rounds) > 1
    return {
        "refine_strategy": "baseline",
        "num_refine_rounds": len(refine_rounds),
        "max_refine_rounds": max_refine_rounds,
        "chain_resolved": chain_resolved,
        "resolved_at_refine_round": resolved_at_refine_round,
        "initial_round_resolved": initial_round_resolved,
        "refine_continued": refine_continued,
        "refine_rescued": bool(chain_resolved and not initial_round_resolved),
    }


class SWEBenchRefineWrapper(SWEBenchWrapper):
    config: SWEBenchRefineConfig

    async def run(self, body: BaseRunRequest) -> list[SWEBenchRefineVerifyResponse]:
        async with self._sem:
            base_params = body.responses_create_params
            base_params.parallel_tool_calls = True
            base_params.tool_choice = "auto"

            # Stable group key shared by every refinement round and rollout of this task.
            group_hash = hashlib.md5(
                json.dumps(
                    _input_to_jsonable(base_params.input), sort_keys=True, ensure_ascii=False
                ).encode("utf-8")
            ).hexdigest()

            max_refine_rounds = self.config.max_refine_rounds
            refine_rounds: list[dict] = []
            prior_summary: Optional[str] = None

            for round_idx in range(max_refine_rounds):
                round_body = self._build_refine_round_body(body, round_idx, prior_summary)
                response = await self.responses(round_body.responses_create_params)

                metadata, response.metadata = response.metadata, None
                metrics = SWEBenchMetrics.model_validate_json(metadata["metrics"])
                responses_create_params = round_body.responses_create_params.model_dump() | {
                    "input": json.loads(metadata["input"]),
                    "tools": [t.model_dump() for t in response.tools] if response.tools else [],
                }
                refine_rounds.append(
                    {
                        "responses_create_params": responses_create_params,
                        "response": response,
                        "metrics": metrics,
                        "instance_config": metadata["instance_config"],
                        "verify_feedback": metadata.get("verify_feedback"),
                    }
                )
                self._dump_refine_round(group_hash, round_idx, refine_rounds[-1])

                if metrics.resolved:
                    break  # chain solved → stop early, pad the rest
                if round_idx < max_refine_rounds - 1:
                    prior_summary = self._summarize_prior(refine_rounds)

            # Chain-level reward = resolved by the (cumulative) final state; broadcast
            # to every round's sample. With fixed-length padding this keeps each
            # rollout's per-group count constant, so the GRPO baseline stays unbiased.
            chain_metrics = _build_chain_metrics(refine_rounds, max_refine_rounds)
            reward = 1.0 if chain_metrics["chain_resolved"] else 0.0

            results: list[SWEBenchRefineVerifyResponse] = []
            for round_idx in range(max_refine_rounds):
                is_padded = round_idx >= len(refine_rounds)
                src = (
                    refine_rounds[round_idx] if not is_padded else refine_rounds[-1]
                )
                metrics = src["metrics"]
                # Padded slots reuse the last real round; deep-copy the response so the
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
                        refine_round_idx=round_idx,
                        is_padded=is_padded,
                        **chain_metrics,
                    )
                )
            return results

    def _build_refine_round_body(
        self, body: BaseRunRequest, round_idx: int, prior_summary: Optional[str]
    ) -> BaseRunRequest:
        """Construct the BaseRunRequest for one refinement round.

        Round 0 is the original task. For later rounds the eval-compatible seed is
        appended to problem_statement in the metadata and nested instance_dict.
        The latter is the representation written to the OpenHands dataset JSONL.

        TODO(refine): when skip_reset_after_initial_round is wired, the workspace
        already carries the accumulated patch; the seed should then describe
        *what changed and what failed* rather than restate the whole task.
        """
        if round_idx == 0 or not prior_summary:
            return body

        new_body = body.model_copy(deep=True)
        params = new_body.responses_create_params
        params.metadata = _append_seed_to_problem_metadata(
            params.metadata or {}, prior_summary
        )
        return new_body

    def _summarize_prior(self, refine_rounds: list[dict]) -> str:
        """Build the seed carried from the prior refinement round.

        Carries the prior round's diff plus the raw verify output (the failing
        tests / tracebacks from the eval step), so the next round can fix the
        actual failures instead of guessing. The verify log is already bounded
        upstream (tail, see _inner_responses). Keep deterministic.

        TODO(refine): when the workspace persists across rounds, the seed should
        describe what changed and what still fails rather than restate the whole
        diff; also consider stripping history thinking.
        """
        last = refine_rounds[-1]
        metrics = last["metrics"]
        return _build_refine_v1_seed(
            patch=metrics.model_patch or "",
            verify_feedback=last.get("verify_feedback") or "",
            max_patch_tokens=self.config.carry_over_token_budget,
        )

    def _dump_refine_round(
        self, group_hash: str, round_idx: int, refine_round: dict
    ) -> None:
        """Best-effort debug dump of one refinement round to disk.

        Writes <dump_dir>/<group_hash>/round_<k>.json (readable text) so the
        constructed refine context can be inspected — in particular that a later
        round carries the prior diff + verify_feedback. No-op unless
        config.dump_dir is set. Never raises (debug side-channel must not break
        the rollout).
        """
        dump_dir = self.config.dump_dir
        if not dump_dir:
            return
        try:
            response = refine_round["response"]
            metrics = refine_round["metrics"]
            record = {
                "group_hash": group_hash,
                "refine_round_idx": round_idx,
                "resolved": metrics.resolved,
                "model_patch": metrics.model_patch,
                # Raw verify output produced by this round (fed into the next one).
                "verify_feedback": refine_round.get("verify_feedback") or "",
                "input": refine_round["responses_create_params"].get("input"),
                "output": [o.model_dump() for o in response.output]
                if response.output
                else [],
            }
            out_dir = Path(dump_dir) / group_hash
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"round_{round_idx}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False, default=str)
            )
        except Exception as e:  # best-effort debug dump; never break the rollout
            print(f"[refine dump] failed for {group_hash} round {round_idx}: {e}", flush=True)


if __name__ == "__main__":
    SWEBenchRefineWrapper.run_webserver()
