# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate Megatron Inference ledger payloads into Gym capture material."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nemo_gym.token_id_capture.staging.records import CaptureAdmission


PREFIX_IDS_FIELD = "required_prefix_token_ids"


def _single_choice(response_payload: dict[str, Any]) -> dict[str, Any]:
    choices = response_payload.get("choices") or []
    if len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError(f"Megatron token capture requires exactly one object choice, got {len(choices)}")
    return choices[0]


def _message(response_payload: dict[str, Any]) -> dict[str, Any]:
    message = _single_choice(response_payload).get("message") or {}
    if not isinstance(message, dict):
        raise ValueError("Megatron response choice.message must be an object")
    return message


def _token_ids(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"Megatron capture field {field_name} must be a token-id list")
    token_ids: list[int] = []
    for token_id in value:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError(f"Megatron capture field {field_name} must contain only integer token ids")
        token_ids.append(token_id)
    if any(token_id < 0 for token_id in token_ids):
        raise ValueError(f"Megatron capture field {field_name} contains a negative token id")
    return token_ids


@dataclass(frozen=True)
class PendingMegatronCapture:
    """One served call waiting for its MInf ledger payload to be staged."""

    request_uid: str
    token_ids_delta: tuple[int, ...]
    cumulative_token_ids: tuple[int, ...]

    @property
    def delta_len(self) -> int:
        return len(self.token_ids_delta)

    @property
    def cum_len(self) -> int:
        return len(self.cumulative_token_ids)


class MegatronCaptureAdapter:
    """Adapt the HTTP token lineage and the later MInf ledger record.

    The HTTP response supplies exact token IDs immediately so Gym can resolve
    multi-turn lineage. Log probabilities remain in the MInf ledger and are
    joined by the response UID when the owning framework flushes the rollout.
    """

    def enter_prefix(self, request_payload: dict[str, Any], prefix_ids: list[int]) -> dict[str, Any]:
        request_payload[PREFIX_IDS_FIELD] = list(prefix_ids)
        return request_payload

    def pending_capture(
        self,
        response_payload: dict[str, Any],
        admission: CaptureAdmission,
    ) -> PendingMegatronCapture:
        """Build the token-light reference Gym persists before batch staging."""
        request_uid = response_payload.get("id")
        if not isinstance(request_uid, str) or not request_uid:
            raise ValueError("Megatron ledger capture response carries no request id")
        message = _message(response_payload)
        if "generation_log_probs" in message:
            raise ValueError(
                "Megatron returned generation_log_probs on the HTTP path; "
                "local_metadata_ledger_offload_enabled is not active"
            )
        prompt_token_ids = _token_ids(message.get("prompt_token_ids"), "prompt_token_ids")
        generated_token_ids = _token_ids(message.get("generation_token_ids"), "generation_token_ids")
        if admission.mode == "token_in" and prompt_token_ids[: admission.prev_len] != list(
            admission.required_prefix_token_ids
        ):
            raise ValueError("Megatron generation prompt does not begin with the admitted token prefix")
        token_ids_delta = prompt_token_ids[admission.prev_len :] + generated_token_ids
        if not token_ids_delta:
            raise ValueError("Megatron capture produced an empty token delta")
        cumulative_token_ids = list(admission.required_prefix_token_ids) + token_ids_delta
        return PendingMegatronCapture(
            request_uid=request_uid,
            token_ids_delta=tuple(token_ids_delta),
            cumulative_token_ids=tuple(cumulative_token_ids),
        )

    def extract_prompt_ids(self, ledger_payload: dict[str, Any]) -> list[int]:
        return _token_ids(ledger_payload.get("prompt_token_ids"), "prompt_token_ids")

    def extract_generation(self, ledger_payload: dict[str, Any]) -> tuple[list[int], list[float]]:
        generated_token_ids = _token_ids(ledger_payload.get("generated_token_ids"), "generated_token_ids")
        raw_logprobs = ledger_payload.get("generated_log_probs")
        if not isinstance(raw_logprobs, list):
            raise ValueError("Megatron ledger record carries no generated_log_probs list")
        generated_logprobs = [float(logprob) for logprob in raw_logprobs]
        if len(generated_token_ids) != len(generated_logprobs):
            raise ValueError(
                "Megatron generated token and log-probability lengths differ: "
                f"{len(generated_token_ids)} != {len(generated_logprobs)}"
            )
        return generated_token_ids, generated_logprobs

    def extract_extras(self, ledger_payload: dict[str, Any]) -> dict[str, Any] | None:
        # MInf routes have total_tokens - 1 rows while the current staging
        # contract is delta-token aligned. NeMo-RL rejects router replay with
        # this adapter until that shift has a first-class representation.
        return None

    def extract_weight_version(self, ledger_payload: dict[str, Any]) -> int:
        """Return the single policy epoch that produced the call."""
        policy_epoch = ledger_payload.get("policy_epoch")
        if not isinstance(policy_epoch, list) or not policy_epoch:
            raise ValueError("Megatron ledger record carries no policy_epoch")
        try:
            versions = {int(boundary[1]) for boundary in policy_epoch}
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError("Megatron policy_epoch must contain (token_offset, version) pairs") from error
        if len(versions) != 1:
            raise ValueError(f"Megatron call spans multiple policy epochs: {sorted(versions)}")
        (version,) = versions
        if version < 0:
            raise ValueError(f"Megatron policy epoch must be non-negative, got {version}")
        return version
