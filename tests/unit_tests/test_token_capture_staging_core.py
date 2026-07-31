# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Staging core (gate-authoritative capture): wire records, digest, purity.

Ported from the fork-era ``test_token_capture_gate_primitives.py``; grows with
the staging subpackage (lineage, rebuild, and conformance sections land with
their modules). The digest vectors are frozen against the sync prototype's
implementation and must never change.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from nemo_gym.token_id_capture.staging.digest import (
    EMPTY_PREFIX_HASH,
    build_staging_delta,
    compute_staging_digest,
    encode_token_ids,
    hash_token_ids,
)
from nemo_gym.token_id_capture.staging.records import (
    SCHEMA_VERSION,
    CommitCoords,
    StagedCallRecord,
    staging_key,
)


# ---------------------------------------------------------------------------
# digest: golden vectors frozen against the sync prototype (rollout_writer.py)
# ---------------------------------------------------------------------------


def test_digest_golden_vectors_match_prototype() -> None:
    # Verified byte-identical against the prototype donor implementation
    # (compute_staging_digest at pranav/tq_gym_prototype 05e0adfa0).
    assert (
        compute_staging_digest(
            rollout_id="g7_r0",
            call_id="c1",
            prev_len=0,
            token_ids_delta=[10, 11, 12, 13, 14],
            token_mask_delta=[0.0, 0.0, 0.0, 1.0, 1.0],
            logprobs_delta=[0.0, 0.0, 0.0, -0.125, -1.75],
        )
        == "6629f7662bf8fbc7de3ddb8c49048c8de9e8d2e8d63cfcfb84556b05c1323aa7"
    )
    assert (
        compute_staging_digest(
            rollout_id="g0_r1",
            call_id="a1",
            prev_len=0,
            token_ids_delta=[5, 6, 7, 8, 9],
            token_mask_delta=[0.0, 0.0, 0.0, 0.0, 1.0],
            logprobs_delta=[0.0, 0.0, 0.0, 0.0, -0.25],
        )
        == "f4cc7cf930dcc821fbc9b87da523f1b487a51cb65adc7cec2eefb97b5cd54884"
    )
    assert EMPTY_PREFIX_HASH == hash_token_ids([])
    assert encode_token_ids([1]) != encode_token_ids([1, 0])  # length-delimited


def test_digest_is_sensitive_to_every_field() -> None:
    base = dict(
        rollout_id="r",
        call_id="c",
        prev_len=3,
        token_ids_delta=[1, 2],
        token_mask_delta=[0.0, 1.0],
        logprobs_delta=[0.0, -1.0],
    )
    reference = compute_staging_digest(**base)
    for mutation in (
        {"rollout_id": "r2"},
        {"call_id": "c2"},
        {"prev_len": 4},
        {"token_ids_delta": [1, 3]},
        {"token_mask_delta": [1.0, 1.0]},
        {"logprobs_delta": [0.0, -1.0000001]},
        {"logprobs_delta": [-0.0, -1.0]},  # -0.0 vs 0.0 must not alias
    ):
        assert compute_staging_digest(**{**base, **mutation}) != reference


def test_encode_token_ids_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        encode_token_ids([-1])


def test_build_staging_delta_layout_and_errors() -> None:
    ids, mask, lps = build_staging_delta(
        prompt_token_ids=[1, 2, 3, 4],
        generated_token_ids=[5, 6],
        generated_logprobs=[-0.5, -0.25],
        prev_len=2,
    )
    assert ids == [3, 4, 5, 6]
    assert mask == [0.0, 0.0, 1.0, 1.0]
    assert lps == [0.0, 0.0, -0.5, -0.25]
    with pytest.raises(ValueError, match="outside prompt length"):
        build_staging_delta(prompt_token_ids=[1], generated_token_ids=[2], generated_logprobs=[-1.0], prev_len=5)
    with pytest.raises(ValueError, match="lengths differ"):
        build_staging_delta(prompt_token_ids=[1], generated_token_ids=[2], generated_logprobs=[], prev_len=0)
    with pytest.raises(ValueError, match="at least one token"):
        build_staging_delta(prompt_token_ids=[1], generated_token_ids=[], generated_logprobs=[], prev_len=1)


# ---------------------------------------------------------------------------
# records: wire-schema sanity
# ---------------------------------------------------------------------------


def test_staging_key_and_record_roundtrip() -> None:
    record = StagedCallRecord(
        rollout_id="r0",
        call_id="c0",
        prev_len=0,
        new_len=2,
        weight_version=1,
        digest="d",
        token_ids_delta=[1, 2],
        token_mask_delta=[0.0, 1.0],
        generation_logprobs_delta=[0.0, -0.5],
    )
    assert record.staging_key == staging_key("r0", "c0") == "r0/c0"
    assert record.schema_version == SCHEMA_VERSION
    assert StagedCallRecord.model_validate(record.model_dump()) == record


def test_records_reject_unknown_fields() -> None:
    with pytest.raises(Exception, match="extra|unexpected"):
        CommitCoords(
            rollout_id="r",
            call_id="c",
            parent_call_id=None,
            delta_len=1,
            cum_len=1,
            digest="d",
            staging_key="r/c",
            weight_version=0,
            not_a_field=True,
        )


# ---------------------------------------------------------------------------
# purity: the staging core must import with no heavy dependencies
# ---------------------------------------------------------------------------


def _staging_modules() -> list[str]:
    """Every module in the ``staging`` subpackage, discovered from disk so a
    new core module is covered automatically (no hand-maintained list)."""
    import nemo_gym.token_id_capture.staging as staging

    root = Path(staging.__file__).parent
    modules = ["nemo_gym.token_id_capture.staging"]
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).with_suffix("")
        parts = [p for p in rel.parts if p != "__init__"]
        if parts:
            modules.append(".".join(["nemo_gym.token_id_capture.staging", *parts]))
    return modules


CORE_MODULES = _staging_modules()

FORBIDDEN_IMPORTS = ("fastapi", "ray", "torch", "transfer_queue", "aiohttp")


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_module_purity(module: str) -> None:
    """The § 3.0 purity rule: core modules must be importable inside any
    framework's worker process, so importing one may not pull fastapi, ray,
    torch, TQ, or an HTTP client into the process."""
    code = (
        f"import {module}, sys; "
        f"bad = sorted(set(sys.modules) & set({FORBIDDEN_IMPORTS!r})); "
        "assert not bad, f'forbidden imports: {bad}'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
