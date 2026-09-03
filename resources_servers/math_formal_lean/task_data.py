# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the math_formal_lean server (Lean 4 proof generation, sandbox-compiled).

Task fields ride at the row top level; there is no verifier_metadata. Required-ness mirrors
``MathFormalLeanRunRequest`` (app.py:353): ``header`` and ``formal_statement`` required,
``informal_prefix``/``name`` Optional. ``turn_index`` (MathFormalLeanVerifyRequest, default 0) is
a verify-time multi-turn field injected by the agent, not a dataset column, so it stays declared
in app.py only. ``split`` is carried by every committed task row but silently dropped by today's
wire model (default ``extra="ignore"``); declared here as provenance.

Only the uniform task files are task rows: data/example.jsonl and data/minif2f_test.jsonl
(minif2f_valid.jsonl is committed but empty). data/multi_turn_full_example.jsonl and
data/multi_turn_success_examples.jsonl are rollout/verify-response DUMPS (keys like all_attempts,
compiler_output, proof_status, reward) — they carry no header/formal_statement and must not be
validated or fed as task data.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    header: str = Field(
        description=(
            "Lean 4 file preamble (imports, set_options, open ...) prepended when assembling the "
            "proof file sent to the compiler sandbox (build_lean4_proof, app.py:125)."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    formal_statement: str = Field(
        description=(
            "The theorem statement ending in ':= by\\n' that the model must complete; restated in "
            "front of the predicted proof when config.restate_formal_statement is set."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    informal_prefix: Optional[str] = Field(
        default=None,
        description=(
            "Natural-language problem statement as a '/-- ... -/' doc comment. Typed on the wire but "
            "never read by verify(); prompt-building provenance."
        ),
        json_schema_extra={"consumed_by": ["prompt", "provenance"]},
    )
    name: Optional[str] = Field(
        default=None,
        description="Theorem/problem name (e.g. 'mathd_algebra_478'). Typed on the wire, never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    split: Optional[str] = Field(
        default=None,
        description=(
            "Upstream miniF2F split label (e.g. 'test'). Present in every committed task row but "
            "silently dropped by today's wire model (extra='ignore')."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
