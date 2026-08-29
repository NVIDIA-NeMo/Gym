# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attribute a finished rollout's verified terminal model call from its manifest.

Every agent's ``/run`` result carries ``response``: the object the verifier
scored. The token-free capture ledger, by contrast, holds one ``CallRecord``
per call the model server served — including auxiliary calls, abandoned
retries, and sub-agent branches. Terminal attribution joins the scored
response to exactly one manifest row, so receipt assembly can anchor
``terminal_model_call_id`` to the call that earned the reward instead of
guessing among chains or masking a healthy rollout.

The join is **independent witnesses with corroboration**, not a trust
hierarchy (ported from the legacy-path design in Gym PR #2676):

  declared     — the harness names the response it kept (the SWE agent's
                 ``terminal_logical_request_id``, which is a served response
                 id). A declaration is authoritative: a declared id that
                 matches no row attributes nothing and never falls back.
  response_id  — the scored ``response.id`` equals the served envelope id
                 recorded on exactly one row. Possession of the id proves
                 which response the agent actually received.
  content      — the ``assistant_fingerprint`` of the scored response's
                 model-authored items matches one row's recorded
                 fingerprints. Three readings pool before the ambiguity
                 decision: the row's cumulative ``continuation_fingerprint``
                 (a full-transcript response), the row's own
                 ``output_fingerprint`` (a final-turn-only response), and the
                 transcript's trailing model-authored block (a merged
                 multi-turn transcript).

Each witness abstains rather than guesses (ambiguity inside a witness is an
abstention, not a vote). Witnesses that agree — or that name rows whose full
token sequences are identical — attribute; witnesses that contradict each
other attribute nothing and persist the disagreement, because a contradiction
is evidence of a real defect (a stale declaration, backend id reuse, a
transcript-synthesis bug) that outranking would silently bury. A rollout with
no witness falls back to the caller's strict parent-link policy
(``select_terminal_call``).

This module never reads tokens: a wrong attribution can only name a chain
whose every digest ``verify_and_linearize`` still checks downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from nemo_gym.token_id_capture.lineage import (
    LINEAGE_FINGERPRINT_VERSION,
    _is_assistant_authored,
    assistant_fingerprint,
)
from nemo_gym.token_id_capture.staging.records import CallRecord


@dataclass(frozen=True)
class TerminalAttribution:
    """The joined terminal call, or the reasons no witness could name one."""

    model_call_id: str | None
    # The naming witness: "declared", "response_id", "content", or "" when
    # unattributed. Precedence orders naming only — never outranking.
    method: str = ""
    # The abstention/disagreement trail, kept on success and failure alike,
    # plus corroboration notes when several witnesses agreed.
    reason: str = ""

    @property
    def attributed(self) -> bool:
        return self.model_call_id is not None


def resolve_terminal(
    records: Sequence[CallRecord],
    response: dict | None,
    *,
    declared_response_id: str | None = None,
) -> TerminalAttribution:
    """Join the scored ``/run`` response to one manifest row.

    Args:
        records: The rollout's committed manifest rows.
        response: The result's scored response object, or ``None`` when the
            result carries none.
        declared_response_id: The response id the harness explicitly declared
            it kept, when it declared one.

    Returns:
        The attribution verdict. This function never raises: malformed
        content is an abstention, not an error.
    """
    reasons: list[str] = []
    witnesses: list[tuple[str, CallRecord]] = []

    if declared_response_id:
        declared_matches = [record for record in records if record.response_id == declared_response_id]
        declared_winner = _collapse_identical(declared_matches)
        if declared_winner is None:
            # A declaration is authoritative: a miss (or an ambiguous match)
            # masks and never falls back to weaker evidence — the harness
            # claimed a specific response and the ledger cannot confirm it.
            reasons.append("declared_ambiguous" if declared_matches else "declared_terminal_not_captured")
            return TerminalAttribution(None, reason=",".join(reasons))
        witnesses.append(("declared", declared_winner))

    if isinstance(response, dict):
        response_id = str(response.get("id") or "")
        if response_id:
            id_matches = [record for record in records if record.response_id == response_id]
            if id_matches:
                winner = _collapse_identical(id_matches)
                if winner is not None:
                    witnesses.append(("response_id", winner))
                else:
                    reasons.append("response_id_ambiguous")
            else:
                reasons.append("response_id_no_match")
        else:
            reasons.append("response_has_no_id")
        content = _content_witness(records, response, reasons)
        if content is not None:
            witnesses.append(content)
    else:
        reasons.append("no_response_object")

    if not witnesses:
        return TerminalAttribution(None, reason=",".join(reasons))

    # Corroborate: all witnesses must name the same row, or rows whose full
    # token sequences are identical (interchangeable for training). A
    # contradiction attributes nothing — it is evidence of a stale
    # declaration or a synthesis defect, and outranking would bury it.
    if _collapse_identical([record for _, record in witnesses]) is None:
        detail = ";".join(f"{method}={record.model_call_id}" for method, record in witnesses)
        reasons.append(f"witness_disagreement[{detail}]")
        return TerminalAttribution(None, reason=",".join(reasons))

    method, named = witnesses[0]
    if len(witnesses) > 1:
        reasons.append("corroborated_by=" + "+".join(other for other, _ in witnesses[1:]))
    # The trail is kept even on success: a witness that abstained (e.g. a
    # duplicated response id) is a diagnosable defect even when another
    # witness attributes the rollout.
    return TerminalAttribution(named.model_call_id, method=method, reason=",".join(reasons))


def _sequence_identity(record: CallRecord) -> tuple:
    """Identify a manifest row by its full token sequence, token-free.

    The worker's whole-sequence ``cumulative_hash`` (with the chained
    ``chain_hash`` as a secondary key) plus the cumulative length identify
    the delivered sequence, which is what training consumes. Identical
    retries produce identical hashes.
    """
    return (record.cumulative_hash, record.chain_hash, record.cum_len)


def _collapse_identical(candidates: list[CallRecord]) -> CallRecord | None:
    """Reduce candidates that carry the same full token sequence to one.

    Identical retries stage identical sequences; any of them linearizes to
    the same training row, so the smallest call id wins deterministically.
    Candidates with different sequences are genuinely ambiguous and collapse
    to ``None``.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    identities = {_sequence_identity(record) for record in candidates}
    if len(identities) == 1:
        return min(candidates, key=lambda record: record.model_call_id)
    return None


def _content_witness(
    records: Sequence[CallRecord],
    response: dict,
    reasons: list[str],
) -> tuple[str, CallRecord] | None:
    """The content witness: fingerprint the response's model-authored items.

    Three readings of one response are possible and must compete, not race.
    A full transcript matches a row's ``continuation_fingerprint`` (the
    model-authored spine of request context + that call's output); a
    final-turn-only response matches one row's ``output_fingerprint``; a
    merged transcript's trailing model-authored block matches the terminal
    call's own output. The readings can name different rows — a first call's
    continuation fingerprint IS its own-output fingerprint, because
    non-model turns never contribute — so candidates from all readings pool
    before the ambiguity decision. Rows whose ``fingerprint_version``
    differs from this process's canonicalization never match.
    """
    output = response.get("output")
    if not isinstance(output, list) or not output:
        reasons.append("response_has_no_output")
        return None
    items = [item for item in output if isinstance(item, dict)]
    try:
        target = assistant_fingerprint(items)
    except (TypeError, ValueError):
        reasons.append("response_output_unfingerprintable")
        return None
    if not target:
        reasons.append("no_model_authored_output")
        return None

    # Third reading: the transcript's trailing model-authored block. The
    # final run of consecutive model-authored items is the terminal call's
    # own output. A transcript ending in a non-model item (a pending tool
    # result) has no trailing block and skips this reading rather than
    # matching the wrong call.
    trailing: list[dict] = []
    for item in reversed(items):
        if not _is_assistant_authored(item):
            break
        trailing.append(item)
    trailing.reverse()
    tail = ""
    if trailing and len(trailing) != len(items):
        try:
            tail = assistant_fingerprint(trailing)
        except (TypeError, ValueError):
            tail = ""

    matches: dict[str, CallRecord] = {}
    for record in records:
        if record.fingerprint_version != LINEAGE_FINGERPRINT_VERSION:
            continue
        if record.continuation_fingerprint and record.continuation_fingerprint == target:
            matches[record.model_call_id] = record
        if record.output_fingerprint and (
            record.output_fingerprint == target or (tail and record.output_fingerprint == tail)
        ):
            matches[record.model_call_id] = record
    winner = _collapse_identical(list(matches.values()))
    if winner is not None:
        return ("content", winner)
    reasons.append("content_ambiguous" if matches else "no_content_match")
    return None
