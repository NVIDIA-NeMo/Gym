# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Domain candidate generation."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from pydantic import TypeAdapter

from resources_servers.synthetic_tool_use.common.artifacts import (
    RunArtifactStore,
    atomic_write_json,
    atomic_write_jsonl,
)
from resources_servers.synthetic_tool_use.common.clients import (
    AsyncTextGenerator,
    ProviderGenerationError,
)
from resources_servers.synthetic_tool_use.common.models import (
    DomainApplication,
    DomainCandidate,
    GenerationMetadata,
    SeedGenerationConfig,
    normalize_domain_name,
)
from resources_servers.synthetic_tool_use.common.parsing import parse_json_value
from resources_servers.synthetic_tool_use_domain_generation.rendering import (
    domain_followup_prompt,
)


APPLICATIONS_ADAPTER = TypeAdapter(list[DomainApplication])


def _parse_candidates(text: str) -> list[dict[str, Any]]:
    value = parse_json_value(text)
    if isinstance(value, dict):
        value = value.get("domains")
    if not isinstance(value, list):
        raise ValueError("domain response must be a JSON array or an object containing `domains`")
    return value


class DomainGenerationStage:
    def __init__(
        self,
        config: SeedGenerationConfig,
        prompt: str,
        store: RunArtifactStore,
        generator: AsyncTextGenerator,
    ) -> None:
        self.config = config
        self.prompt = prompt
        self.store = store
        self.generator = generator

    async def run(self, *, resume: bool = True) -> list[DomainCandidate]:
        accepted_path = self.store.run_dir / "domains.accepted.jsonl"
        if resume and accepted_path.is_file():
            candidates = [DomainCandidate.model_validate_json(line) for line in accepted_path.read_text().splitlines()]
            self.store.register_domains(candidates)
            return candidates

        request_results = await asyncio.gather(
            *(self._run_request(request_index) for request_index in range(self.config.domains.request_count))
        )
        raw_candidates = [candidate for result in request_results for candidate in result]

        raw_candidates.sort(key=lambda item: (item.request_index, item.candidate_index))
        seen_names: set[str] = set()
        accepted: list[DomainCandidate] = []
        for candidate in raw_candidates:
            normalized = normalize_domain_name(candidate.name)
            if normalized in seen_names:
                candidate.accepted = False
                candidate.rejection_reason = "duplicate_normalized_name"
            else:
                seen_names.add(normalized)
                accepted.append(candidate)

        atomic_write_jsonl(self.store.run_dir / "domains.raw.jsonl", raw_candidates)
        atomic_write_jsonl(accepted_path, accepted)
        self.store.register_domains(accepted)
        return accepted

    async def _run_request(self, request_index: int) -> list[DomainCandidate]:
        first = await self._run_prompt(
            request_index=request_index,
            phase="initial",
            prompt=self.prompt,
            candidate_offset=0,
        )
        followup_prompt = domain_followup_prompt(
            self.prompt,
            [candidate.name for candidate in first],
        )
        second = await self._run_prompt(
            request_index=request_index,
            phase="followup",
            prompt=followup_prompt,
            candidate_offset=len(first),
        )
        return [*first, *second]

    async def _run_prompt(
        self,
        *,
        request_index: int,
        phase: str,
        prompt: str,
        candidate_offset: int,
    ) -> list[DomainCandidate]:
        for semantic_attempt in range(1, self.config.domains.semantic_attempts + 1):
            attempt_path = (
                self.store.run_dir
                / "attempts"
                / "domains"
                / (f"request_{request_index:04d}_{phase}_attempt_{semantic_attempt:02d}.json")
            )
            try:
                response = await self.generator.generate([{"role": "user", "content": prompt}])
            except ProviderGenerationError as exc:
                atomic_write_json(
                    attempt_path,
                    {
                        "request_index": request_index,
                        "phase": phase,
                        "semantic_attempt": semantic_attempt,
                        "provider_error": str(exc),
                    },
                )
                return []
            try:
                parsed = _parse_candidates(response.text)
                atomic_write_json(
                    attempt_path,
                    {
                        "request_index": request_index,
                        "phase": phase,
                        "semantic_attempt": semantic_attempt,
                        "provider_attempts": response.provider_attempts,
                        "raw_response": response.raw_response,
                        "parsed_count": len(parsed),
                    },
                )
                generation = GenerationMetadata(
                    model=self.config.domain_model.model,
                    sampling=self.config.domain_model.sampling,
                    prompt_name=f"domains_{phase}",
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                )
                return [
                    DomainCandidate(
                        name=item["name"],
                        applications=APPLICATIONS_ADAPTER.validate_python(item["applications"]),
                        generation_profile=self.config.generation_profile,
                        request_index=request_index,
                        candidate_index=candidate_offset + candidate_index,
                        generation=generation,
                    )
                    for candidate_index, item in enumerate(parsed)
                ]
            except Exception as exc:
                atomic_write_json(
                    attempt_path,
                    {
                        "request_index": request_index,
                        "phase": phase,
                        "semantic_attempt": semantic_attempt,
                        "provider_attempts": response.provider_attempts,
                        "raw_response": response.raw_response,
                        "parse_error": str(exc),
                    },
                )
        return []
