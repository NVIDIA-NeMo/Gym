# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the format_verification server.

The task datum is a single top-level ``verifier`` dict (this server predates the
``verifier_metadata`` convention, so nothing is spliced — ``verifier`` is a genuine flat field)
carrying a discriminated union keyed on ``verifier['type']``: 'regex' and 'inline_prose' both route
to the line-matching regex path (``_verify_regex``), 'string_match' to the marker-presence path,
and any other value raises ``NotImplementedError``. 'inline_prose' is code-reachable (app.py:48)
but exercised by no committed data. The wire model (``FormatVerificationVerifyRequest``,
app.py:32) types ``verifier`` as a required ``Dict[str, Any]`` and every subkey is read via
``.get`` with a default, so within each variant only the discriminator is required and the
variants stay ``extra="allow"``.
"""

from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class RegexVerifier(BaseModel):
    """Line-matching variant: reward 1.0 iff at least ``verify_min_matches`` lines match a pattern."""

    model_config = ConfigDict(extra="allow")

    type: Literal["regex", "inline_prose"] = Field(
        description="Verify-path discriminator; 'inline_prose' is routed to the same regex path as 'regex'.",
    )
    pattern_id: Optional[str] = Field(
        default=None,
        description="Human-readable pattern label (e.g. 'kv_equals'); never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    verify_regex: List[str] = Field(
        default_factory=list,
        description=(
            "Regex patterns tried per response line; a line counts at most once even if several patterns "
            "match. Read as verifier.get('verify_regex', [])."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    verify_min_matches: int = Field(
        default=1,
        description="Minimum number of matching lines for reward 1.0; read as .get('verify_min_matches', 1).",
        json_schema_extra={"consumed_by": ["verify"]},
    )


class StringMatchVerifier(BaseModel):
    """Marker-presence variant: reward 1.0 iff every expected marker appears and no spurious ones do."""

    model_config = ConfigDict(extra="allow")

    type: Literal["string_match"] = Field(description="Verify-path discriminator.")
    expected_markers: List[str] = Field(
        default_factory=list,
        description=(
            "Literal strings that must all appear in the response text (e.g. '[ref:2]'); read as "
            ".get('expected_markers', [])."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    patterns: List[str] = Field(
        default_factory=list,
        description=(
            "Regex patterns used to detect SPURIOUS markers: any match whose text is not in "
            "expected_markers fails the row. Read as .get('patterns', [])."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    verifier: Annotated[Union[RegexVerifier, StringMatchVerifier], Field(discriminator="type")] = Field(
        description=(
            "Wire-required verification spec dict, discriminated on 'type'; an unknown type raises "
            "NotImplementedError in verify()."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
