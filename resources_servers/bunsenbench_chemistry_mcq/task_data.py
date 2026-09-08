# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the bunsenbench_chemistry_mcq server.

Extends the mcqa family schema exactly as the wire does (``BunsenChemVerifyRequest`` extends
``MCQAVerifyRequest``), adding five Optional bunsen fields. Committed rows use the mcqa
"preferred dataset format": grading fields at top level (options/expected_answer/uuid) and a
16-key provenance ``metadata`` dict. None of the added fields appear top-level in committed data
— the verifier's ``_metadata_value`` helper resolves each (bunsen_id/source/bct_field/
bct_subfield) from the top-level attr first, then falls back to ``metadata[key]`` — but they are
declared on the wire, so they are declared here. Do not over-constrain: the server deliberately
tolerates two option encodings (``options`` letter->text dicts vs ``choices`` with
letter/label/key/id + text/content/value/choice key aliases).
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from resources_servers.mcqa.task_data import TaskData as MCQATaskData


class TaskData(MCQATaskData):
    bunsen_id: Optional[str] = Field(
        default=None,
        description=(
            "BunsenBench task identifier; echoed into the verify response. Committed rows carry "
            "it inside `metadata` instead (the verifier falls back there)."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    choices: Optional[List[Union[str, Dict[str, Any]]]] = Field(
        default=None,
        description=(
            "Alternate option encoding used when `options` is empty: bare option strings, or "
            "dicts with a letter/label/key/id key plus a text/content/value/choice key."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    source: Optional[str] = Field(
        default=None,
        description="Source dataset name; drives compute_metrics grouping (metadata fallback applies).",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    bct_field: Optional[str] = Field(
        default=None,
        description="BunsenBench chemistry taxonomy field; drives compute_metrics grouping.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    bct_subfield: Optional[str] = Field(
        default=None,
        description="BunsenBench chemistry taxonomy subfield; drives compute_metrics grouping.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
