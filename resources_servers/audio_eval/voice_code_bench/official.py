# SPDX-FileCopyrightText: Copyright (c) 2026 BeSimple AI
# SPDX-License-Identifier: MIT
#
# Vendored from the VoiceCodeBench release (MIT License for all files outside ``data/audio/``).
#   Source:   https://huggingface.co/datasets/besimple-ai/voice-code-bench
#   Commit:   8edfa398aa5575816d940793e4e2b8c6430184da
#   Files:    https://huggingface.co/datasets/besimple-ai/voice-code-bench/blob/8edfa398aa5575816d940793e4e2b8c6430184da/scripts/voice_code_bench/metrics.py
#             (the format-invariant WER lattice) and
#             https://huggingface.co/datasets/besimple-ai/voice-code-bench/blob/8edfa398aa5575816d940793e4e2b8c6430184da/scripts/voice_code_bench/verifier.py
#             (verifier config, request and response parsing)
#   License:  MIT, https://huggingface.co/datasets/besimple-ai/voice-code-bench/blob/8edfa398aa5575816d940793e4e2b8c6430184da/LICENSE
#
# Modifications: the function bodies below are copied unchanged from the two upstream files. NVIDIA merged the two
# files into one module; dropped the verifier transport (``post_json``, the on-disk verifier cache and the provider
# call) and the report writers, because Gym issues the verifier request through a judge model server; and added
# ``load_verifier_config``, which reads the packaged ``verifier_openai_gpt_5_5_v1.json`` (an unchanged copy of
# upstream ``scripts/voice_code_bench/verifiers/openai_gpt_5_5_v1.json``). The upstream line formatting is kept
# (``fmt: off`` below keeps this file out of ``ruff format``).
#
# MIT License
#
# Copyright (c) 2026 BeSimple AI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# fmt: off
"""Official VoiceCodeBench scoring: format-invariant WER and the entity-verifier request/response contract."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_VERIFIER_ID = "openai_gpt_5_5_v1"
VERIFIER_CONFIG_PATH = Path(__file__).with_name("verifier_openai_gpt_5_5_v1.json")

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
TEMPLATE_ENTITY_RE = re.compile(r"<([^>]+)>")
FORMAT_INVARIANT_WER_REFERENCE = "acoustic_or_canonical_entity_lattice"


@dataclass(frozen=True)
class WerResult:
    reference_words: int
    hypothesis_words: int
    substitutions: int
    deletions: int
    insertions: int
    wer: float


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def compute_format_invariant_wer(datapoint: dict[str, Any], hypothesis_text: str) -> WerResult:
    reference = tokenize_words(datapoint["transcripts"]["acoustic"])
    hypothesis = tokenize_words(hypothesis_text)
    chunks = format_invariant_reference_chunks(datapoint, reference)
    return compute_alternative_reference_wer(chunks, hypothesis, reference_word_count=len(reference))


def format_invariant_reference_chunks(
    datapoint: dict[str, Any],
    reference: list[str],
) -> list[list[list[str]]]:
    """Build ordered reference chunks with acoustic/canonical entity alternatives."""
    entities = datapoint.get("entities")
    transcripts = datapoint.get("transcripts")
    if not isinstance(entities, list):
        raise ValueError(f"Datapoint {datapoint.get('audio_id', '<unknown>')} has invalid entities.")
    if not isinstance(transcripts, dict) or not isinstance(transcripts.get("template"), str):
        raise ValueError(f"Datapoint {datapoint.get('audio_id', '<unknown>')} is missing its template transcript.")

    entities_by_id: dict[str, dict[str, Any]] = {}
    for entity in entities:
        entity_id = entity.get("id")
        if not isinstance(entity_id, str) or not entity_id:
            raise ValueError("WER entity is missing its ID.")
        if entity_id in entities_by_id:
            raise ValueError(f"WER entities contain duplicate ID {entity_id}.")
        entities_by_id[entity_id] = entity
    entity_ids = TEMPLATE_ENTITY_RE.findall(transcripts["template"])
    if len(entity_ids) != len(entities) or set(entity_ids) != set(entities_by_id):
        entity_ids = list(entities_by_id)

    chunks: list[list[list[str]]] = []
    cursor = 0
    for entity_id in entity_ids:
        entity = entities_by_id[entity_id]
        acoustic = entity.get("acoustic")
        canonical = entity.get("canonical")
        if not isinstance(acoustic, str) or not isinstance(canonical, str):
            raise ValueError(f"Entity {entity_id} is missing acoustic or canonical text.")
        acoustic_tokens = tokenize_words(acoustic)
        canonical_tokens = tokenize_words(canonical)
        if not acoustic_tokens or not canonical_tokens:
            raise ValueError(f"Entity {entity_id} has an empty acoustic or canonical WER rendering.")
        start = find_subsequence(reference, acoustic_tokens, cursor)
        if start is None:
            raise ValueError(
                f"Datapoint {datapoint.get('audio_id', '<unknown>')} acoustic transcript "
                f"does not contain entity {entity_id} after token {cursor}."
            )
        if start > cursor:
            chunks.append([reference[cursor:start]])
        alternatives = [acoustic_tokens]
        if canonical_tokens != acoustic_tokens:
            alternatives.append(canonical_tokens)
        chunks.append(alternatives)
        cursor = start + len(acoustic_tokens)
    if cursor < len(reference):
        chunks.append([reference[cursor:]])
    return chunks


def find_subsequence(tokens: list[str], subsequence: list[str], start: int = 0) -> int | None:
    width = len(subsequence)
    for index in range(start, len(tokens) - width + 1):
        if tokens[index : index + width] == subsequence:
            return index
    return None


def compute_token_wer(reference: list[str], hypothesis: list[str]) -> WerResult:
    return compute_alternative_reference_wer([[reference]], hypothesis, reference_word_count=len(reference))


def compute_alternative_reference_wer(
    reference_chunks: list[list[list[str]]],
    hypothesis: list[str],
    *,
    reference_word_count: int,
) -> WerResult:
    """Find minimum edit distance through an acyclic alternative-reference lattice."""
    cols = len(hypothesis) + 1
    prev_cost = list(range(cols))
    prev_sub = [0] * cols
    prev_del = [0] * cols
    prev_ins = list(range(cols))

    for alternatives in reference_chunks:
        candidates = [
            advance_wer_state(
                prev_cost,
                prev_sub,
                prev_del,
                prev_ins,
                alternative,
                hypothesis,
            )
            for alternative in alternatives
        ]
        prev_cost = list(candidates[0][0])
        prev_sub = list(candidates[0][1])
        prev_del = list(candidates[0][2])
        prev_ins = list(candidates[0][3])
        for cost, substitutions, deletions, insertions in candidates[1:]:
            for col in range(cols):
                if cost[col] < prev_cost[col]:
                    prev_cost[col] = cost[col]
                    prev_sub[col] = substitutions[col]
                    prev_del[col] = deletions[col]
                    prev_ins[col] = insertions[col]

    substitutions = prev_sub[-1]
    deletions = prev_del[-1]
    insertions = prev_ins[-1]
    return WerResult(
        reference_words=reference_word_count,
        hypothesis_words=len(hypothesis),
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        wer=0.0
        if reference_word_count == 0
        else (substitutions + deletions + insertions) / reference_word_count,
    )


def advance_wer_state(
    initial_cost: list[int],
    initial_sub: list[int],
    initial_del: list[int],
    initial_ins: list[int],
    reference: list[str],
    hypothesis: list[str],
) -> tuple[list[int], list[int], list[int], list[int]]:
    cols = len(hypothesis) + 1
    prev_cost = list(initial_cost)
    prev_sub = list(initial_sub)
    prev_del = list(initial_del)
    prev_ins = list(initial_ins)

    for reference_word in reference:
        curr_cost = [prev_cost[0] + 1] + [0] * (cols - 1)
        curr_sub = [0] * cols
        curr_del = [prev_del[0] + 1] + [0] * (cols - 1)
        curr_ins = [0] * cols

        for col, hypothesis_word in enumerate(hypothesis, start=1):
            substitution_increment = 0 if reference_word == hypothesis_word else 1

            best_cost = prev_cost[col - 1] + substitution_increment
            best_sub = prev_sub[col - 1] + substitution_increment
            best_del = prev_del[col - 1]
            best_ins = prev_ins[col - 1]

            deletion_cost = prev_cost[col] + 1
            if deletion_cost < best_cost:
                best_cost = deletion_cost
                best_sub = prev_sub[col]
                best_del = prev_del[col] + 1
                best_ins = prev_ins[col]

            insertion_cost = curr_cost[col - 1] + 1
            if insertion_cost < best_cost:
                best_cost = insertion_cost
                best_sub = curr_sub[col - 1]
                best_del = curr_del[col - 1]
                best_ins = curr_ins[col - 1] + 1

            curr_cost[col] = best_cost
            curr_sub[col] = best_sub
            curr_del[col] = best_del
            curr_ins[col] = best_ins

        prev_cost = curr_cost
        prev_sub = curr_sub
        prev_del = curr_del
        prev_ins = curr_ins
    return prev_cost, prev_sub, prev_del, prev_ins


@dataclass(frozen=True)
class VerifierConfig:
    id: str
    provider: str
    model: str
    endpoint_or_api: str
    timeout_seconds: float
    request_options: dict[str, Any]
    system_prompt: str
    response_format: dict[str, Any]

    @property
    def digest(self) -> str:
        payload = {
            "id": self.id,
            "provider": self.provider,
            "model": self.model,
            "endpoint_or_api": self.endpoint_or_api,
            "timeout_seconds": self.timeout_seconds,
            "request_options": self.request_options,
            "system_prompt": self.system_prompt,
            "response_format": self.response_format,
        }
        return stable_digest(payload)


def parse_verifier_config(data: Any, *, path: Path | None = None) -> VerifierConfig:
    location = str(path) if path is not None else "verifier config"
    if not isinstance(data, dict):
        raise ValueError(f"{location} must be a JSON object.")
    if data.get("schema_version") != 1:
        raise ValueError(f"{location} must use schema_version=1.")
    verifier_id = required_string(data, "id", location)
    provider = required_string(data, "provider", location)
    model = required_string(data, "model", location)
    endpoint_or_api = required_string(data, "endpoint_or_api", location)
    timeout_seconds = data.get("timeout_seconds")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError(f"{location} must define a positive timeout_seconds.")
    request_options = data.get("request_options", {})
    if not isinstance(request_options, dict):
        raise ValueError(f"{location} request_options must be an object.")
    system_prompt = data.get("system_prompt")
    if isinstance(system_prompt, list) and all(isinstance(line, str) for line in system_prompt):
        prompt = "\n".join(system_prompt)
    elif isinstance(system_prompt, str):
        prompt = system_prompt
    else:
        raise ValueError(f"{location} system_prompt must be a string or array of strings.")
    if not prompt.strip():
        raise ValueError(f"{location} system_prompt must be non-empty.")
    response_format = data.get("response_format")
    if not isinstance(response_format, dict):
        raise ValueError(f"{location} response_format must be an object.")
    return VerifierConfig(
        id=verifier_id,
        provider=provider,
        model=model,
        endpoint_or_api=endpoint_or_api,
        timeout_seconds=float(timeout_seconds),
        request_options=dict(request_options),
        system_prompt=prompt,
        response_format=response_format,
    )


def required_string(data: dict[str, Any], key: str, location: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must define non-empty {key}.")
    return value


def target_entity_payload(datapoint: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "target_index": index,
            "id": entity["id"],
            "type": entity["type"],
            "role": entity["role"],
            "acoustic": entity["acoustic"],
            "canonical": entity["canonical"],
        }
        for index, entity in enumerate(datapoint["entities"])
    ]


def verifier_input(datapoint: dict[str, Any], transcript: str) -> dict[str, Any]:
    return {
        "datapoint_id": datapoint["audio_id"],
        "target_entities": target_entity_payload(datapoint),
        "transcript": transcript,
    }


def build_openai_verify_request(config: VerifierConfig, datapoint: dict[str, Any], transcript: str) -> dict[str, Any]:
    request = {
        "model": config.model,
        **config.request_options,
        "response_format": config.response_format,
        "messages": [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": json.dumps(verifier_input(datapoint, transcript), ensure_ascii=False)},
        ],
    }
    if "temperature" not in request:
        temperature = verifier_temperature(config.model)
        if temperature is not None:
            request["temperature"] = temperature
    return request


def verifier_temperature(model: str) -> int | None:
    if model.strip().lower().startswith("gpt-5"):
        return None
    return 0


def parse_llm_entity_matches(
    value: Any,
    target_entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("entities"), list):
        raise ValueError("Expected verifier response with an entities array.")

    targets = []
    for index, entity in enumerate(target_entities):
        targets.append(
            {
                "target_index": index,
                "type": entity["type"],
                "canonical": entity["canonical"],
            }
        )
    by_index = {target["target_index"]: target for target in targets}
    out_by_index: dict[int, dict[str, Any]] = {}
    for item in value["entities"]:
        if not isinstance(item, dict):
            raise ValueError("Expected verifier entity item to be an object.")
        target_index = item.get("target_index")
        entity_type = item.get("type")
        canonical = item.get("canonical")
        present = item.get("present")
        evidence = item.get("evidence")
        reason = item.get("reason")
        if not isinstance(target_index, int) or target_index not in by_index:
            raise ValueError(f"Unsupported target_index from verifier: {target_index}")
        if target_index in out_by_index:
            raise ValueError(f"Duplicate target_index from verifier: {target_index}")
        target = by_index[target_index]
        if entity_type != target["type"]:
            raise ValueError(f"Verifier changed type for target_index {target_index}.")
        if not isinstance(canonical, str):
            raise ValueError(f"Verifier returned non-string canonical for target_index {target_index}.")
        if not isinstance(present, bool):
            raise ValueError(f"Verifier returned non-boolean present for target_index {target_index}.")
        if not isinstance(evidence, str):
            raise ValueError(f"Verifier returned non-string evidence for target_index {target_index}.")
        if present and not evidence.strip():
            raise ValueError(f"Verifier returned present=true without evidence for target_index {target_index}.")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Verifier returned empty reason for target_index {target_index}.")
        out_by_index[target_index] = {
            "target_index": target_index,
            "type": target["type"],
            "canonical": target["canonical"],
            "present": present,
            "evidence": evidence,
            "reason": reason,
        }

    missing = [target["target_index"] for target in targets if target["target_index"] not in out_by_index]
    if missing:
        raise ValueError(f"Verifier omitted target_index values: {missing}")
    return [out_by_index[index] for index in range(len(targets))]


def stable_digest(value: Any) -> str:
    if isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_verifier_config(verifier_id: str = DEFAULT_VERIFIER_ID) -> VerifierConfig:
    if verifier_id != DEFAULT_VERIFIER_ID:
        raise ValueError(f"Unknown verifier id {verifier_id!r}; only {DEFAULT_VERIFIER_ID!r} is packaged.")
    return parse_verifier_config(json.loads(VERIFIER_CONFIG_PATH.read_text(encoding="utf-8")), path=VERIFIER_CONFIG_PATH)
