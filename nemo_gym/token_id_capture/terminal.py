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
hierarchy. A witness here is one independent reading of the response that can
name the captured call. The fact "which call was kept" is emitted in different
places by different agent classes, so up to five witnesses testify:

  explicit     — the caller names the kept call directly (a gate seal or an
                 agent-declared terminal id). Soft: a miss is an abstention
                 and other witnesses may still attribute.
  declared     — the harness reports the response id it retained. Selection
                 stops if that id does not match exactly one captured row.
  response_id  — the ``/run`` response's ``id`` equals the served id recorded
                 on exactly one entry. Possession of the id proves which
                 response the client actually received.
  item_id      — the ``/run`` response's last model-authored item carries the
                 id the model server assigned to it (a message's ``id``, or a
                 tool call's ``call_id``), and exactly one entry's recorded
                 output holds that id. A harness that rebuilds its transcript
                 from its own session files keeps these per-item ids even when
                 it never saw the response id, and the last model-authored item
                 names the terminal call even when tool outputs follow it.
  content      — the fingerprint of the response's model-authored items
                 matches one entry. Three readings are pooled: the entry's
                 cumulative ``continuation_fingerprint`` when a lineage-aware
                 writer recorded one (a full-transcript response), the
                 fingerprint of the entry's own output (a final-turn-only
                 response), and the transcript's trailing model-authored
                 block (a merged multi-turn transcript).

External staging rows omit token arrays because those tokens remain in framework storage.
They carry a ``staging_key`` and precomputed content fingerprints instead.
This module identifies external staging rows by the presence of ``staging_key``.

Each witness abstains rather than guesses (ambiguity inside a witness is an
abstention, not a vote). The verdict then follows the stack's rule that claims
are verified, never ranked: witnesses that agree — or that name calls with
identical full token sequences — attribute; witnesses that contradict each
other attribute nothing and persist the disagreement, because a contradiction
is evidence of a real defect (a stale seal mapping, backend id reuse, a
transcript-synthesis bug) that outranking would silently bury. If no terminal
response ID is declared, a rollout with no witness or with disagreeing witnesses
falls back to the builder's strict single-chain policy. If a declared terminal
response ID cannot be attributed, the consumer masks the rollout instead.

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
    _content_of,
    _is_assistant_authored,
    assistant_fingerprint,
    canonicalize_tool_arguments,
)
from nemo_gym.token_id_capture.records import NAMESPACE_TOOL_DELIMITER, TokenEntry


@dataclass(frozen=True)
class TerminalAttribution:
    """The joined terminal call, or the reasons no witness could name one."""

    model_call_id: str | None
    # The strongest agreeing witness: "explicit", "response_id", "item_id",
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
    *,
    declared_response_id: str | None = None,
) -> TerminalAttribution:
    """Join the verified ``/run`` response to one captured model call.

    ``entries`` is the frozen snapshot (``TokenEntry`` records or token-free
    custody rows). ``response`` is the result's scored response object (or
    ``None`` when the record carries none). This function never raises:
    malformed content is an abstention, not an error.
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

    if declared_response_id:
        declared_matches = [entry for entry in entries if entry.response_id == declared_response_id]
        declared_winner = _collapse_identical(declared_matches)
        if declared_winner is None:
            # The harness reported a specific response ID.
            # Do not select a different call when that ID has no unique match.
            reasons.append("declared_ambiguous" if declared_matches else "declared_terminal_not_captured")
            return TerminalAttribution(None, reason=",".join(reasons))
        witnesses.append(("declared", declared_winner))

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
        item = _item_id_witness(entries, response, reasons)
        if item is not None:
            witnesses.append(item)
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


def _is_custody_row(entry: TokenEntry) -> bool:
    """A token-free custody row stages its tokens externally under a key."""
    return getattr(entry, "staging_key", None) is not None


def _sequence_identity(entry: TokenEntry) -> tuple:
    """Identify an entry by its full token sequence.

    A custody row identifies by the worker's whole-sequence
    ``cumulative_hash`` (with the chained ``chain_hash`` as a secondary key)
    plus the cumulative length. A lineage-aware ``TokenEntry`` writer records
    a cumulative digest and length; records without one compare their token
    arrays directly. All identify the delivered sequence, which is what
    training consumes.
    """
    if _is_custody_row(entry):
        return (entry.cumulative_hash, entry.chain_hash, entry.cum_len)
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


def _served_item_id(item: dict) -> str:
    """Return the id the model server assigned to an output item, or "" when it carries none.

    A tool call is identified by its ``call_id``, the id the serving engine
    assigned to the call; both the chat and the Responses formats keep that
    id. Any other item is identified by its ``id``.
    """
    if item.get("type") == "function_call":
        return str(item.get("call_id") or "")
    return str(item.get("id") or "")


def _tool_name_key(name: object) -> str:
    """Return a tool name without the namespace prefix the streaming sanitizer flattens in.

    A namespaced Responses tool reaches the chat backend as
    ``<namespace>__<name>`` and is recorded that way in the entry, while the
    client, and a transcript rebuilt from the client's files, keeps the bare
    name with the namespace in a separate field. Two bare names that differ
    only before their own last ``__`` compare equal here; the witness also
    requires the served id and the canonical arguments to match, so such a
    pair cannot be confused on its own.
    """
    return str(name or "").rsplit(NAMESPACE_TOOL_DELIMITER, 1)[-1]


def _item_content_key(item: dict) -> tuple:
    """Return the content of one model-authored item in a dialect-independent form.

    A tool call is its name (without any namespace prefix) and canonical
    arguments; a message is its typed content parts (``_content_of`` reads a
    plain string and a list of typed parts alike).
    """
    if item.get("type") == "function_call":
        return ("function_call", _tool_name_key(item.get("name")), canonicalize_tool_arguments(item.get("arguments")))
    return ("message", tuple(_content_of(item.get("content"))))


def _entry_items_by_served_id(entry: TokenEntry) -> dict[str, tuple]:
    """Map every served id in an entry's recorded output to that item's content key.

    The map covers Responses items (a message's ``id``, a tool call's
    ``call_id``) and the tool calls nested in a chat assistant message's
    ``tool_calls`` list.
    """
    keyed: dict[str, tuple] = {}
    for item in entry.output_items:
        if not isinstance(item, dict):
            continue
        served = _served_item_id(item)
        if served:
            keyed[served] = _item_content_key(item)
        for call in item.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("id"):
                function = call.get("function") or {}
                keyed[str(call["id"])] = (
                    "function_call",
                    _tool_name_key(function.get("name")),
                    canonicalize_tool_arguments(function.get("arguments")),
                )
    return keyed


def _item_id_witness(entries: list[TokenEntry], response: dict, reasons: list[str]) -> tuple[str, TokenEntry] | None:
    """The item-id witness: the served id of the response's last model-authored item.

    The model server assigns an id to every output item it returns. A harness
    that rebuilds its transcript from its own session files keeps those ids
    even when it never records the response id, so the transcript's last
    model-authored item names the call that produced it. Tool outputs or user
    turns that follow it are not model output, so they do not change which
    item is last.

    An id alone does not name a call. The entry's item with that id must also
    carry the same content (the tool's name and canonical arguments, or the
    message's content parts), so that a reused or colliding id cannot name a
    call whose output the transcript does not end with. An item whose id no
    entry recorded, such as a message the harness composed itself, abstains
    rather than naming an earlier call.
    """
    output = response.get("output")
    if not isinstance(output, list):
        return None
    last = next((item for item in reversed(output) if isinstance(item, dict) and _is_assistant_authored(item)), None)
    if last is None:
        return None
    item_id = _served_item_id(last)
    if not item_id:
        reasons.append("last_item_has_no_id")
        return None
    try:
        wanted = _item_content_key(last)
    except ValueError:
        reasons.append("last_item_unfingerprintable")
        return None
    matches: list[TokenEntry] = []
    content_mismatch = False
    for entry in entries:
        # A token-free custody row records fingerprints and ids, not output
        # items, so this witness cannot read it and abstains from it.
        if _is_custody_row(entry):
            continue
        try:
            recorded = _entry_items_by_served_id(entry).get(item_id)
        except ValueError:
            continue
        if recorded is None:
            continue
        if recorded == wanted:
            matches.append(entry)
        else:
            content_mismatch = True
    if not matches:
        reasons.append("item_id_content_mismatch" if content_mismatch else "item_id_no_match")
        return None
    winner = _collapse_identical(matches)
    if winner is None:
        reasons.append("item_id_ambiguous")
        return None
    return ("item_id", winner)


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
    except (TypeError, ValueError):
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
        except (TypeError, ValueError):
            tail = ""

    matches: dict[str, TokenEntry] = {}
    cumulative_hit = False
    custody_hit = False
    for entry in entries:
        continuation = getattr(entry, "continuation_fingerprint", None)
        version = getattr(entry, "fingerprint_version", None)
        if _is_custody_row(entry):
            # Custody rows record both fingerprints at commit time, so a row
            # stamped with a different canonicalization version never matches.
            if version != FINGERPRINT_VERSION:
                continue
            output_fingerprint = getattr(entry, "output_fingerprint", None)
            if continuation and continuation == target:
                matches[entry.model_call_id] = entry
                custody_hit = True
            if output_fingerprint and (output_fingerprint == target or (tail and output_fingerprint == tail)):
                matches[entry.model_call_id] = entry
                custody_hit = True
            continue
        if continuation and continuation == target and (version is None or version == FINGERPRINT_VERSION):
            matches[entry.model_call_id] = entry
            cumulative_hit = True
        if entry.output_items:
            try:
                own = assistant_fingerprint(list(entry.output_items))
            except (TypeError, ValueError):
                continue
            if own and (own == target or (tail and own == tail)):
                matches[entry.model_call_id] = entry
    winner = _collapse_identical(list(matches.values()))
    if winner is not None:
        if custody_hit:
            # Custody manifests label all three readings as one witness.
            return ("content", winner)
        return ("content_cumulative" if cumulative_hit else "content_output", winner)
    reasons.append("content_ambiguous" if matches else "no_content_match")
    return None
