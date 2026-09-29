# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the speed_bench server.

speed_bench measures speculative-decoding throughput by scraping the model server's Prometheus
``/metrics`` endpoint; ``verify()`` returns a hardcoded 0.0 reward and reads only
``response.usage.output_tokens`` — never any task field — for scoring. The fields below live
inside ``verifier_metadata`` on today's wire (``SpeedBenchVerifyRequest`` at app.py:305 declares
it as an untyped ``Optional[Dict[str, Any]]`` passthrough) solely so they survive into the
rollout JSONL, where the cross-pipeline diff script (``debug_compare_specdec.py``) matches
Skills<->Gym rollouts on ``src_id``. Everything is Optional because the wire never requires the
bucket or any subfield.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    src_id: Optional[str] = Field(
        default=None,
        description="Source-row identifier (e.g. 'synthetic-1'); join key for the Skills<->Gym rollout diff.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    source: Optional[str] = Field(
        default=None,
        description="Dataset source label (e.g. 'example').",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    speed_config: Optional[str] = Field(
        default=None,
        description="Speed-bench prompt-generation config name (e.g. 'qualitative').",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    num_turns: Optional[int] = Field(
        default=None,
        description="Number of conversation turns in the prepared prompt.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    sub_category: Optional[str] = Field(
        default=None,
        description="Prompt sub-category emitted by prepare.py; null in all committed example rows.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
