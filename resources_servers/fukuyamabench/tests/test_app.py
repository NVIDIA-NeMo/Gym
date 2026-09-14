# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from unittest.mock import MagicMock

from app import (
    FukuyamaBenchResourcesServer,
    FukuyamaBenchStatus,
    FukuyamaBenchVerifyRequest,
    _extract_pathway,
)
from metrics import canonical_set, canonical_smiles, compare_step_products, score_pathway, strip_atom_mapping

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


# A three-step oxidation with the middle step excluded from the checkpoints, which
# is the shape upstream produces when it drops trivial steps.
GT_PATHWAY = [
    {"step_id": "1", "products": ["CCO"]},
    {"step_id": "2", "products": ["CC=O"]},
    {"step_id": "3", "products": ["CC(=O)O"]},
]
CHECKPOINTS = [[1], [3]]


def _make_server() -> FukuyamaBenchResourcesServer:
    config = BaseResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return FukuyamaBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_test",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_request(text: str, **meta_overrides) -> FukuyamaBenchVerifyRequest:
    meta = {
        "case_id": "B001",
        "case_set": "B",
        "gt_pathway": GT_PATHWAY,
        "checkpoints": CHECKPOINTS,
        "lenient": True,
    }
    meta.update(meta_overrides)
    return FukuyamaBenchVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "Q"}]},
        response=_make_response(text),
        verifier_metadata=meta,
    )


def _fenced(steps: list[dict]) -> str:
    return f"## Reasoning\nsome analysis\n\n## Result\n\n```json\n{json.dumps(steps)}\n```"


def _pathway(*smiles: str) -> list[dict]:
    return [{"step_id": i, "product_smiles": s, "is_terminal": False} for i, s in enumerate(smiles, 1)]


class TestVerify:
    def test_sanity(self) -> None:
        _make_server()

    async def test_gold_pathway_scores_one(self) -> None:
        result = await _make_server().verify(_make_request(_fenced(_pathway("CCO", "CC=O", "CC(=O)O"))))
        assert result.reward == 1.0
        assert result.status == FukuyamaBenchStatus.SCORED.value
        assert result.checkpoints_correct == 2
        assert result.checkpoints_total == 2
        assert result.checkpoint_accuracy == 1.0
        assert result.case_id == "B001"

    async def test_noncanonical_writing_still_matches(self) -> None:
        """Structural equality, not string equality."""
        result = await _make_server().verify(_make_request(_fenced(_pathway("OCC", "O=CC", "OC(C)=O"))))
        assert result.reward == 1.0

    async def test_atom_mapped_prediction_matches_unmapped_gold(self) -> None:
        """Prompts hand the model atom-mapped reactants, so mapping comes back in the answer."""
        result = await _make_server().verify(
            _make_request(_fenced(_pathway("[CH3:1][CH2:2][OH:3]", "[CH3:1][CH:2]=[O:3]", "CC(=O)O")))
        )
        assert result.reward == 1.0

    async def test_finer_grained_prediction_still_matches(self) -> None:
        """Extra intermediate steps between checkpoints are the reason checkpoints exist."""
        result = await _make_server().verify(
            _make_request(_fenced(_pathway("CCO", "CCOO", "CC=O", "C1CO1", "CC(=O)O")))
        )
        assert result.reward == 1.0

    async def test_missing_final_checkpoint_scores_zero(self) -> None:
        result = await _make_server().verify(_make_request(_fenced(_pathway("CCO", "CC=O"))))
        assert result.reward == 0.0
        assert result.status == FukuyamaBenchStatus.SCORED.value
        assert result.checkpoints_correct == 1
        assert result.any_checkpoint_correct is True

    async def test_out_of_order_pathway_scores_zero(self) -> None:
        """Checkpoints are matched in order; the cursor never goes backwards."""
        result = await _make_server().verify(_make_request(_fenced(_pathway("CC(=O)O", "CC=O", "CCO"))))
        assert result.reward == 0.0

    async def test_wrong_but_valid_chemistry_scores_zero(self) -> None:
        result = await _make_server().verify(_make_request(_fenced(_pathway("c1ccccc1", "CCCCCC"))))
        assert result.reward == 0.0
        assert result.any_checkpoint_correct is False

    async def test_unparseable_smiles_scores_zero(self) -> None:
        result = await _make_server().verify(_make_request(_fenced(_pathway("not_a_smiles((", "))"))))
        assert result.reward == 0.0
        assert result.status == FukuyamaBenchStatus.SCORED.value

    async def test_empty_output_is_flagged(self) -> None:
        result = await _make_server().verify(_make_request(""))
        assert result.reward == 0.0
        assert result.status == FukuyamaBenchStatus.EMPTY_OUTPUT.value
        assert result.checkpoints_correct is None

    async def test_prose_without_a_pathway_is_flagged(self) -> None:
        result = await _make_server().verify(_make_request("I cannot determine this mechanism."))
        assert result.reward == 0.0
        assert result.status == FukuyamaBenchStatus.NO_PATHWAY.value

    async def test_missing_gold_is_flagged_not_scored_as_a_miss(self) -> None:
        result = await _make_server().verify(_make_request(_fenced(_pathway("CCO")), gt_pathway=None))
        assert result.reward == 0.0
        assert result.status == FukuyamaBenchStatus.BAD_GOLD.value

    async def test_missing_checkpoints_is_flagged(self) -> None:
        result = await _make_server().verify(_make_request(_fenced(_pathway("CCO")), checkpoints=None))
        assert result.status == FukuyamaBenchStatus.BAD_GOLD.value


class TestLeniency:
    """Upstream credits a prediction that omits a co-product; --strict does not."""

    GT = [{"step_id": "1", "products": ["CCO", "O"]}]
    CKPT = [[1]]

    def test_subset_matches_when_lenient(self) -> None:
        result = score_pathway([{"product_smiles": "CCO"}], self.GT, self.CKPT, lenient=True)
        assert result["exact_match"] is True

    def test_subset_fails_when_strict(self) -> None:
        result = score_pathway([{"product_smiles": "CCO"}], self.GT, self.CKPT, lenient=False)
        assert result["exact_match"] is False

    def test_superset_never_matches(self) -> None:
        result = score_pathway([{"product_smiles": "CCO.O.c1ccccc1"}], self.GT, self.CKPT, lenient=True)
        assert result["exact_match"] is False


class TestDottedProducts:
    """Regression: gold and prediction must be split on '.' the same way.

    Upstream splits only the prediction, so a gold entry written as one dotted
    string can never be matched. That silently caps the attainable score.
    """

    def test_dotted_gold_matches_dotted_prediction(self) -> None:
        gt = [{"step_id": "1", "products": ["CCO.[Pd]"]}]
        result = score_pathway([{"product_smiles": "CCO.[Pd]"}], gt, [[1]])
        assert result["exact_match"] is True

    def test_dotted_gold_matches_split_prediction(self) -> None:
        gt = [{"step_id": "1", "products": ["CCO.[Pd]"]}]
        assert canonical_set(["CCO.[Pd]"]) == canonical_set(["CCO", "[Pd]"])
        result = score_pathway([{"product_smiles": "CCO"}, {"product_smiles": "[Pd]"}], gt, [[1]])
        # A subset of the gold species, so lenient matching credits it.
        assert result["exact_match"] is True


class TestOrganometallics:
    """Transition-metal complexes fail RDKit's default valence check.

    Without the permissive fallback these gold products are unparseable, which
    makes their checkpoints unmatchable by any prediction — five set-B/C cases
    could not be scored at all.
    """

    CHROMIUM = "COC(c1ccccc1)=[Cr](C#O)(C#O)(C#O)C#O"

    def test_metal_complex_is_parseable(self) -> None:
        assert canonical_smiles(self.CHROMIUM) is not None

    def test_metal_complex_matches_itself(self) -> None:
        gt = [{"step_id": "1", "products": [self.CHROMIUM]}]
        result = score_pathway([{"product_smiles": self.CHROMIUM}], gt, [[1]])
        assert result["exact_match"] is True

    def test_metal_complex_does_not_match_a_different_species(self) -> None:
        gt = [{"step_id": "1", "products": [self.CHROMIUM]}]
        result = score_pathway([{"product_smiles": "CCO"}], gt, [[1]])
        assert result["exact_match"] is False


class TestEquivalentSteps:
    """A checkpoint listing several step IDs accepts any one of them."""

    GT = [
        {"step_id": "1", "products": ["CCO"]},
        {"step_id": "2", "products": ["CC=O"]},
    ]

    def test_either_equivalent_step_satisfies_the_checkpoint(self) -> None:
        for smiles in ("CCO", "CC=O"):
            result = score_pathway([{"product_smiles": smiles}], self.GT, [[1, 2]])
            assert result["exact_match"] is True, smiles


class TestExtraction:
    def test_fenced_json_array(self) -> None:
        assert _extract_pathway(_fenced(_pathway("CCO"))) == _pathway("CCO")

    def test_last_fenced_block_wins(self) -> None:
        text = _fenced(_pathway("CCO")) + "\n\nOn reflection:\n" + _fenced(_pathway("CC=O"))
        extracted = _extract_pathway(text)
        assert extracted is not None
        assert extracted[0]["product_smiles"] == "CC=O"

    def test_unfenced_array_after_result_header(self) -> None:
        text = f"## Result\n\n{json.dumps(_pathway('CCO'))}"
        assert _extract_pathway(text) == _pathway("CCO")

    def test_loose_step_objects(self) -> None:
        text = 'Step one: {"step_id": 1, "product_smiles": "CCO"} and that is all.'
        extracted = _extract_pathway(text)
        assert extracted is not None
        assert extracted[0]["product_smiles"] == "CCO"

    def test_prose_yields_nothing(self) -> None:
        assert _extract_pathway("No JSON here at all.") is None

    def test_malformed_json_yields_nothing(self) -> None:
        assert _extract_pathway('```json\n[{"step_id": 1, "product_smiles":}]\n```') is None


class TestHelpers:
    def test_strip_atom_mapping(self) -> None:
        assert strip_atom_mapping("[CH3:1][O:22]") == "[CH3][O]"

    def test_canonical_set_drops_unparseable(self) -> None:
        assert canonical_set(["CCO", "not_a_smiles(("]) == {"CCO"}

    def test_compare_reports_why_it_failed(self) -> None:
        assert compare_step_products(["CCO"], ["CC=O"])[1] == "mismatch"
        assert compare_step_products(["CCO", "c1ccccc1"], ["CCO"])[1] == "superset_match"
        assert compare_step_products([], ["CCO"])[1] == "invalid_pred"
