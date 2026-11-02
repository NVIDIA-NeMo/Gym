# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the swemer_oml (OML SWE-bench Extended) resources server.

Mirrors ``app.SwemerOmlInstanceRequest``: required-ness follows the wire contract. The three
``tests/`` files are what the verifier actually runs; ``FAIL_TO_PASS``/``PASS_TO_PASS`` are
copied from the package's ``config.json`` for provenance only (the package's own ``grade.py``
enforces them).
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """One OML SWE-bench Extended task."""

    model_config = ConfigDict(extra="allow")

    instance_id: str = Field(json_schema_extra={"consumed_by": ["verify", "provenance"]})
    delivery: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    language: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    repo: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    workdir: str = Field(default="/workspace/repo", json_schema_extra={"consumed_by": ["verify"]})
    image_ref: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    base_commit: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    patch: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    test_patch: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    problem_statement: str = Field(default="", json_schema_extra={"consumed_by": ["prompt", "provenance"]})
    test_sh: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    config_json: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    grade_py: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    test_framework: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    FAIL_TO_PASS: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["provenance"]})
    PASS_TO_PASS: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["provenance"]})
    # Delivery metadata carried through from task.toml / config.json / the image manifest; none of
    # it is read by verify().
    nydus_ref: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    prompt_statement: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    f2p_synthetic: bool = Field(default=False, json_schema_extra={"consumed_by": ["provenance"]})
    keywords: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["provenance"]})
    difficulty: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    task_type: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    pass_at_k_glm_5_2: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    pass_at_k_opus_4_8: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    pass_at_k_gpt_5_5: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    agent_timeout_sec: float | None = Field(default=None, json_schema_extra={"consumed_by": ["provenance"]})
    verifier_timeout_sec: float | None = Field(default=None, json_schema_extra={"consumed_by": ["provenance"]})
    toml_workdir: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
