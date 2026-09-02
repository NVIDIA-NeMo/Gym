# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attribute a finished rollout's verified terminal model call.

Every agent's ``/run`` result carries ``response``: the object the verifier
scored. Capture, by contrast, records every call the model server served,
including auxiliary calls, abandoned retries, and sub-agent branches. Terminal
attribution joins the verified response to exactly one captured call, so the
builder can deliver the chain that earned the reward instead of guessing among
chains or masking a healthy rollout.

The join is **independent witnesses with corroboration**, not a trust
hierarchy. The fact "which call was kept" is emitted in different places by
different agent classes, so up to three witnesses testify:

  explicit     — the caller names the kept call directly (a gate seal or an
                 agent-declared terminal id).
  response_id  — the ``/run`` response's ``id`` equals the served id recorded
                 on exactly one entry. Possession of the id proves which
                 response the client actually received.
  content      — the fingerprint of the response's model-authored items
                 matches one entry. Three readings are pooled: the entry's
                 cumulative ``continuation_fingerprint`` when a lineage-aware
                 writer recorded one (a full-transcript response), the
                 fingerprint of the entry's own output (a final-turn-only
                 response), and the transcript's trailing model-authored
                 block (a merged multi-turn transcript).

Each witness abstains rather than guesses (ambiguity inside a witness is an
abstention, not a vote). The verdict then follows the stack's rule that claims
are verified, never ranked: witnesses that agree — or that name calls with
identical full token sequences — attribute; witnesses that contradict each
other attribute nothing and persist the disagreement, because a contradiction
is evidence of a real defect (a stale seal mapping, backend id reuse, a
transcript-synthesis bug) that outranking would silently bury. A rollout with
no witness, or with disagreeing witnesses, falls back to the builder's strict
single-chain policy.

The content witness deliberately uses ``assistant_fingerprint`` alone, without
a request-context digest. Attribution selects a chain whose tokens the builder
verifies independently; it never reuses tokens across the matched boundary.
Requiring context verification would spuriously refuse synthesized transcripts
that reformat tool output, without adding safety.
"""

from __future__ import annotations

from dataclasses import dataclass

from nemo_gym.token_id_capture.fingerprint import (
    FINGERPRINT_VERSION,
    _is_assistant_authored,
    assistant_fingerprint,
)
from nemo_gym.token_id_capture.records import TokenEntry


@dataclass(frozen=True)
class TerminalAttribution:
    """The joined terminal call, or the reasons no witness could name one."""

    model_call_id: str | None
    # The strongest agreeing witness: "explicit", "response_id",
    # "content_cumulative", "content_output", or "" when unattributed.
    method: str = ""
    # The abstention/disagreement trail, kept on success and failure alike,
    # plus corroboration notes when several witnesses agreed.
    reason: str = ""

    @property
    def attributed(self) -> bool:
        return self.model_call_id is not None


def resolve_terminal(
    entries: list[TokenEntry],
    response: dict | None,
    explicit_call_id: str | None = None,
) -> TerminalAttribution:
    """Join the verified ``/run`` response to one captured model call.

    ``entries`` is the frozen snapshot. ``response`` is the result's scored
    response object (or ``None`` when the record carries none). This function
    never raises: malformed content is an abstention, not an error.
    """
    reasons: list[str] = []
    by_call_id = {entry.model_call_id: entry for entry in entries}
    # Witnesses in precedence order for *naming* only (which call id and
    # method label an agreeing verdict reports) — never for outranking.
    witnesses: list[tuple[str, TokenEntry]] = []

    if explicit_call_id:
        named = by_call_id.get(explicit_call_id)
        if named is not None:
            witnesses.append(("explicit", named))
        else:
            reasons.append("explicit_terminal_not_captured")

    if isinstance(response, dict):
        response_id = str(response.get("id") or "")
        if response_id:
            id_matches = [entry for entry in entries if entry.response_id and entry.response_id == response_id]
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
        content = _content_witness(entries, response, reasons)
        if content is not None:
            witnesses.append(content)
    else:
        reasons.append("no_response_object")

    if not witnesses:
        return TerminalAttribution(None, reason=",".join(reasons))

    # Corroborate: all witnesses must name the same call, or calls whose full
    # token sequences are identical (interchangeable for training). A
    # contradiction attributes nothing — it is evidence of a stale mapping or
    # a synthesis defect, and outranking would bury it.
    if _collapse_identical([entry for _, entry in witnesses]) is None:
        detail = ";".join(f"{method}={entry.model_call_id}" for method, entry in witnesses)
        reasons.append(f"witness_disagreement[{detail}]")
        return TerminalAttribution(None, reason=",".join(reasons))

    method, named = witnesses[0]
    if len(witnesses) > 1:
        reasons.append("corroborated_by=" + "+".join(other for other, _ in witnesses[1:]))
    # The trail is kept even on success: a witness that abstained (e.g. a
    # duplicated response id) is a diagnosable defect even when another
    # witness attributes the rollout.
    return TerminalAttribution(named.model_call_id, method=method, reason=",".join(reasons))


def _sequence_identity(entry: TokenEntry) -> tuple:
    """Identify an entry by its full token sequence.

    A lineage-aware writer records a cumulative digest and length.
    Records without one compare their token arrays directly.
    Both identify the delivered sequence, which is what training consumes.
    """
    digest = getattr(entry, "digest", None)
    cum_len = getattr(entry, "cum_len", None)
    if digest and cum_len is not None:
        return (digest, cum_len)
    return (tuple(entry.prompt_token_ids), tuple(entry.generation_token_ids))


def _collapse_identical(candidates: list[TokenEntry]) -> TokenEntry | None:
    """Reduce candidates that carry the same full token sequence to one.

    Identical retries produce entries whose sequences match; any of them
    yields the same delivered chain, so the smallest call id wins
    deterministically. Candidates with different sequences are genuinely
    ambiguous and collapse to ``None``.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    identities = {_sequence_identity(entry) for entry in candidates}
    if len(identities) == 1:
        return min(candidates, key=lambda entry: entry.model_call_id)
    return None


def _content_witness(entries: list[TokenEntry], response: dict, reasons: list[str]) -> tuple[str, TokenEntry] | None:
    """The content witness: fingerprint the response's model-authored items.

    Three readings of one response are possible and must compete, not race. A
    full transcript matches an entry's cumulative fingerprint (the
    model-authored spine of request context + that call's output); a
    final-turn-only response matches the fingerprint of one entry's own
    output; a merged transcript's trailing model-authored block matches the
    terminal call's own output. The readings can name different calls — a
    first call's cumulative fingerprint IS its own-output fingerprint, because
    non-model turns never contribute — so candidates from all keys pool before
    the ambiguity decision.
    """
    output = response.get("output")
    if not isinstance(output, list) or not output:
        reasons.append("response_has_no_output")
        return None
    items = [item for item in output if isinstance(item, dict)]
    try:
        target = assistant_fingerprint(items)
    except ValueError:
        reasons.append("response_output_unfingerprintable")
        return None
    if not target:
        reasons.append("no_model_authored_output")
        return None

    # Third reading: the transcript's trailing model-authored block.
    # A merged multi-turn transcript hashes over every model turn, so it can
    # only match a cumulative fingerprint — which requires a lineage-aware
    # writer. The final block of consecutive model-authored items is the
    # terminal call's own output, matchable on any base. A transcript ending
    # in a non-model item (a pending tool result) has no trailing block and
    # skips this reading rather than matching the wrong call.
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
        except ValueError:
            tail = ""

    matches: dict[str, TokenEntry] = {}
    cumulative_hit = False
    for entry in entries:
        continuation = getattr(entry, "continuation_fingerprint", None)
        version = getattr(entry, "fingerprint_version", None)
        if continuation and continuation == target and (version is None or version == FINGERPRINT_VERSION):
            matches[entry.model_call_id] = entry
            cumulative_hit = True
        if entry.output_items:
            try:
                own = assistant_fingerprint(list(entry.output_items))
            except ValueError:
                continue
            if own and (own == target or (tail and own == tail)):
                matches[entry.model_call_id] = entry
    winner = _collapse_identical(list(matches.values()))
    if winner is not None:
        return ("content_cumulative" if cumulative_hit else "content_output", winner)
    reasons.append("content_ambiguous" if matches else "no_content_match")
    return None
