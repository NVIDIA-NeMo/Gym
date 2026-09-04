# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenCode-specific model-call observation enrichment."""

from collections.abc import Iterable

from nemo_gym.base_responses_api_model import ModelCallRecord
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ModelCallRef


SOURCE = "opencode"


def _model_ref_key(reference: object) -> tuple[str, str] | None:
    model_type = getattr(reference, "type", None)
    model_name = getattr(reference, "name", None)
    return (model_type, model_name) if isinstance(model_type, str) and isinstance(model_name, str) else None


def _reference_key(reference: ModelCallRef) -> tuple[str | None, tuple[str, str] | None, str | None]:
    return reference.model_call_id, _model_ref_key(reference.model_ref), reference.response_id


def _matching_reference_keys(
    call: ModelCallRecord,
) -> tuple[tuple[str | None, tuple[str, str] | None, str | None], ...]:
    model_ref = _model_ref_key(call.model_ref)
    keys = [
        (call.model_call_id, None, None),
        (call.model_call_id, model_ref, None),
        (call.model_call_id, None, call.response_id),
        (call.model_call_id, model_ref, call.response_id),
    ]
    if model_ref is not None and call.response_id is not None:
        keys.append((None, model_ref, call.response_id))
    return tuple(dict.fromkeys(keys))


def associate_opencode_session_calls(
    bundle: AgentObservationBundle,
    calls: Iterable[ModelCallRecord],
) -> AgentObservationBundle:
    """Associate calls with the exact OpenCode session declared by the client.

    The session identifier is correlation evidence, not an authentication boundary.
    """
    if bundle.source != SOURCE:
        return bundle

    result = bundle.model_copy()
    result.records = [
        record.model_copy(update={"model_calls": list(record.model_calls)})
        if isinstance(record, AgentInvocation)
        else record
        for record in bundle.records
    ]
    result.gaps = list(bundle.gaps)
    invocations = {record.invocation_id: record for record in result.records if isinstance(record, AgentInvocation)}
    reference_keys = {
        invocation_id: {_reference_key(reference) for reference in invocation.model_calls}
        for invocation_id, invocation in invocations.items()
    }

    for call in calls:
        if not call.client_session_id or not call.model_call_id:
            continue
        invocation = invocations.get(call.client_session_id)
        if invocation is None:
            continue
        invocation_keys = reference_keys[invocation.invocation_id]
        if any(key in invocation_keys for key in _matching_reference_keys(call)):
            continue
        reference = ModelCallRef(
            model_call_id=call.model_call_id,
            model_ref=call.model_ref,
            response_id=call.response_id,
        )
        invocation.model_calls.append(reference)
        invocation_keys.add(_reference_key(reference))

    return result
