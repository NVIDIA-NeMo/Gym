# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the swerl_gen server.

SWE-RL patch-generation rows. ``mode`` branches the entire verify path ('eval' scores a predicted
patch against the instance's tests; 'repro-gen' scores a generated reproduction test against the
gold ``instance['patch']``), but both modes share one row shape, so this is a single model rather
than a union — a discriminated union would add no field-level typing while making the
wire-optional ``mode`` (default 'eval') effectively required.

``instance`` is deliberately an open dict, mirroring the wire (``SWEGenRunRequest`` at app.py:47
types it ``dict[str, Any]``): its core is the SWE-bench instance family documented in
``resources_servers.swebench.task_data.TaskData`` (committed rows carry 14 str keys — that core
minus environment_setup_commit/difficulty/subset/split — plus setup_script/test_script/
regression_script), but the eval harness (``eval/eval_instance.py``) tolerates per-dataset
variants: alternate test-list key spellings (FAIL_TO_PASS | fail_to_pass_select | fail_to_pass,
same for PASS_TO_PASS), an optional ``base_dir`` (default '/testbed'), and dynamic filename-style
keys ('run_script.sh', 'parsing_script.py'). Do not chase closure on it.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    instance: Dict[str, Any] = Field(
        description=(
            "Open SWE-bench-style instance dict (see module docstring): the swebench task_data core "
            "plus setup_script/test_script/regression_script; FAIL_TO_PASS/PASS_TO_PASS are "
            "JSON-encoded list strings. Serialized whole (base64) into the Ray evaluation job; "
            "repro-gen mode additionally reads instance['patch'] as the gold patch."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    dataset_name: Optional[str] = Field(
        default=None,
        description="Source dataset name, e.g. 'SWE-Gym/SWE-Gym'; never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    dataset_split: Optional[str] = Field(
        default=None,
        description="Source dataset split, e.g. 'train'; never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Verification context: 'image' (sandbox image, both modes) plus eval-mode keys "
            "'relevant_file_contents' (JSON-encoded string) and 'remove_repo_name' (bool)."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    partial_similarity: Optional[bool] = Field(
        default=None,
        description="Accepted by the wire but never read anywhere in the server.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    mode: str = Field(
        default="eval",
        description=(
            "Verify-path switch: 'repro-gen' scores a generated reproduction test, 'eval' (the default) "
            "scores a predicted patch; any other value makes verify() raise ValueError. Plain str on "
            "the wire, so not narrowed to a Literal."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
