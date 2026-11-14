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
import asyncio
import json
from pathlib import Path

from nemo_gym.server_runtime import verifier_observability
from nemo_gym.trajectory import VerifierInvocation
from nemo_gym.verification import exercise_verifier_fixture

from ..app import VERIFIER_FIXTURE, OcsrVerifier, OcsrVerifyRequest
from ..chem import extract_answer, score, strip_reasoning


CASES = [json.loads(line) for line in (Path(__file__).parent / "verifier_cases.jsonl").open()]


def test_verifier_fixture() -> None:
    asyncio.run(
        exercise_verifier_fixture(
            VERIFIER_FIXTURE,
            reward_range=(0.0, 1.0),
            higher_is_better=True,
            determinism="unknown",
        )
    )


def _request(gold: str) -> OcsrVerifyRequest:
    body = dict(CASES[0]["request"])
    body["verifier_metadata"] = {"id": "x", "gold_smiles": gold}
    return OcsrVerifyRequest.model_validate(body)


def test_unusable_gold_is_masked_not_scored() -> None:
    for gold in ("", "C1CC("):
        out = asyncio.run(OcsrVerifier().verify(_request(gold)))
        assert out.mask_sample and out.reward == 0.0 and out.failure_kind == "ocsr:bad_gold"


def test_diagnostics_on_a_correct_answer() -> None:
    out = asyncio.run(OcsrVerifier().verify(OcsrVerifyRequest.model_validate(CASES[0]["request"])))
    assert (out.reward, out.exact, out.graph, out.inchikey, out.valid, out.format_ok) == (1.0,) * 6
    assert out.tanimoto == 1.0 and not out.mask_sample


def test_stereo_error_keeps_graph_credit() -> None:
    s = score("C[C@@H](C(=O)O)c1ccc2cc(OC)ccc2c1", "C[C@H](C(=O)O)c1ccc2cc(OC)ccc2c1")
    assert s["exact"] == 0.0 and s["graph"] == 1.0 and s["inchikey"] == 0.0


def test_extraction_rules() -> None:
    assert extract_answer('{"smiles": "CCO", "confidence": 1}') == ("CCO", "json")
    assert extract_answer('first {"smiles": "CC"} then {"smiles": "CCO"}') == ("CCO", "json")
    assert extract_answer("answer: `CCO` and `not a smiles((`") == ("CCO", "salvage")
    assert extract_answer('<think>{"smiles": "CC"}</think>no answer') == (None, None)
    assert extract_answer("") == (None, None)
    assert strip_reasoning('reasoning only closed</think>{"smiles": "C"}') == '{"smiles": "C"}'


def _verify_under_invocation(request: OcsrVerifyRequest):
    invocation = VerifierInvocation(invocation_id="verify-test")
    token = verifier_observability._active_verifier.set(invocation)
    try:
        response = asyncio.run(OcsrVerifier().verify(request))
    finally:
        verifier_observability._active_verifier.reset(token)
    return response, invocation


def _with_answer(text: str) -> OcsrVerifyRequest:
    request = OcsrVerifyRequest.model_validate(CASES[0]["request"])
    request.response.output[0].content[0].text = text
    return request


def test_a_scored_verify_records_one_programmatic_grade_bound_to_the_response() -> None:
    right, right_invocation = _verify_under_invocation(_with_answer('{"smiles": "COc1ccc2cc([C@H](C)C(O)=O)ccc2c1"}'))
    wrong, wrong_invocation = _verify_under_invocation(_with_answer('{"smiles": "CCO"}'))
    assert right.reward == 1.0
    assert (
        wrong.reward == 0.0 and not wrong.mask_sample and wrong.failure_kind is None
    )  # a wrong answer is a plain zero
    [right_op] = right_invocation.grading_operations
    [wrong_op] = wrong_invocation.grading_operations
    for op in (right_op, wrong_op):
        assert op.kind == "programmatic_calculation" and op.outcome == "completed" and op.dependencies == []
    # The record is bound to the graded result: a different response gives a different binding.
    assert right_op.result_binding_sha256 != wrong_op.result_binding_sha256


def test_an_unusable_gold_label_records_no_grade() -> None:
    response, invocation = _verify_under_invocation(_request("C1CC("))
    assert response.mask_sample and response.failure_kind == "ocsr:bad_gold"
    assert invocation.grading_operations is None
