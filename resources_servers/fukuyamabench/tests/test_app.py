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

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from app import (
    FukuyamaBenchResourcesServer,
    FukuyamaBenchStatus,
    FukuyamaBenchVerifyRequest,
    _extract_last_assistant_text,
    _extract_pathway,
)
from metrics import canonical_smiles, canonical_species, compare_step_products, score_pathway, strip_atom_mapping
from task_data import TaskData

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.server_utils import ServerClient


def _load_prepare_module():
    """Load the preparation script, which lives outside the import path."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_fukuyamabench.py"
    spec = importlib.util.spec_from_file_location("prepare_fukuyamabench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _load_prepare_module()

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "synthetic_mechanisms"


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

    def test_dotted_gold_matches_under_strict_scoring(self) -> None:
        """Strict mode removes the subset escape hatch.

        Under lenient matching a partial answer could mask a broken split, so
        this pins the behaviour where only true set equality passes.
        """
        gt = [{"step_id": "1", "products": ["CCO.[Pd]"]}]
        result = score_pathway([{"product_smiles": "CCO.[Pd]"}], gt, [[1]], lenient=False)
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


class TestMalformedPathway:
    """A well-formed array can still hold a malformed element.

    Extraction keeps an array when any element is a step, so scoring must treat
    the rest as ordinary misses rather than raising — a model formatting error
    must not become a server error.
    """

    GT = [{"step_id": "1", "products": ["CCO"]}, {"step_id": "2", "products": ["CC=O"]}]

    def test_assorted_non_mappings_do_not_raise(self) -> None:
        result = score_pathway(["a string", 42, [], None], self.GT, [[1]])
        assert result["exact_match"] is False

    async def test_verify_scores_a_malformed_element_as_a_miss(self) -> None:
        body = _make_request('```json\n[{"step_id": 1, "product_smiles": "CCO"}, null]\n```')
        result = await _make_server().verify(body)
        assert result.status == FukuyamaBenchStatus.SCORED.value
        assert result.reward == 0.0


class TestPerTierMetrics:
    """Set B and Set C differ by nearly an order of magnitude upstream.

    A mean pooled across tiers describes no published benchmark. Asserting on
    ``compute_metrics`` alone is not enough — emitting tier keys while the
    inherited selection still promotes ``mean/reward`` leaves the invalid
    cross-tier number as the headline, so these go through the real aggregate
    path that consumers read.
    """

    # One perfect B case and one failed C case: pooling them gives 0.5, a number
    # that describes neither tier.
    VERIFY_RESPONSES = [
        {"case_set": "B", "case_id": "B001", "reward": 1.0, "_ng_task_index": 0, "_ng_rollout_index": 0},
        {"case_set": "C", "case_id": "C001", "reward": 0.0, "_ng_task_index": 1, "_ng_rollout_index": 0},
    ]

    def _aggregate(self):
        server = _make_server()
        return compute_aggregate_metrics(
            self.VERIFY_RESPONSES,
            compute_metrics_fn=server.compute_metrics,
            get_key_metrics_fn=server.get_key_metrics,
        )

    def test_pooled_reward_is_not_a_headline_metric(self) -> None:
        key_metrics = self._aggregate().key_metrics
        assert "mean/reward" not in key_metrics, key_metrics

    def test_tiers_carry_different_results(self) -> None:
        key_metrics = self._aggregate().key_metrics
        b = {k.split("/", 1)[1]: v for k, v in key_metrics.items() if k.startswith("B/")}
        c = {k.split("/", 1)[1]: v for k, v in key_metrics.items() if k.startswith("C/")}
        assert b and c and b != c


class TestSingleTierMetrics:
    """A single-tier run has no cross-tier average to suppress."""

    def test_key_metrics_fall_back_when_no_tier_is_present(self) -> None:
        metrics = {"mean/reward": 0.5, "mean/checkpoint_accuracy": 0.25}
        assert _make_server().get_key_metrics(metrics) == metrics


class TestInvalidPredictedProducts:
    """A prediction is only correct if everything it wrote is a real structure.

    Unparseable components used to be dropped before comparison, so the right
    product plus one malformed fragment scored an exact match — in strict mode
    too, which made "strict" mean something other than set equality.
    """

    GT = [{"step_id": "1", "products": ["CCO"]}]

    def test_extra_unparseable_component_fails_when_lenient(self) -> None:
        result = score_pathway([{"product_smiles": "CCO.not_a_smiles(("}], self.GT, [[1]], lenient=True)
        assert result["exact_match"] is False

    def test_extra_unparseable_component_fails_when_strict(self) -> None:
        result = score_pathway([{"product_smiles": "CCO.not_a_smiles(("}], self.GT, [[1]], lenient=False)
        assert result["exact_match"] is False

    def test_error_names_the_unparseable_component(self) -> None:
        assert compare_step_products(["CCO", "not_a_smiles(("], ["CCO"])[1] == "invalid_pred_component"

    def test_a_wholly_valid_prediction_still_matches(self) -> None:
        """The guard must not reject good answers."""
        result = score_pathway([{"product_smiles": "CCO"}], self.GT, [[1]])
        assert result["exact_match"] is True

    def test_non_string_list_member_fails_product_smiles(self) -> None:
        result = score_pathway([{"product_smiles": ["CCO", None]}], self.GT, [[1]], lenient=False)
        assert result["exact_match"] is False

    def test_non_string_list_member_fails_products(self) -> None:
        result = score_pathway([{"products": ["CCO", None]}], self.GT, [[1]], lenient=False)
        assert result["exact_match"] is False

    @pytest.mark.parametrize("field", ["product_smiles", "products"])
    def test_a_product_field_of_the_wrong_type_fails(self, field) -> None:
        """Neither a string nor a list — the shape is not a product at all."""
        result = score_pathway([{field: 42}], self.GT, [[1]], lenient=False)
        assert result["exact_match"] is False

    async def test_verify_rejects_a_malformed_list_member_end_to_end(self) -> None:
        body = _make_request(
            _fenced([{"step_id": 1, "product_smiles": ["CCO", None]}]),
            gt_pathway=self.GT,
            checkpoints=[[1]],
        )
        result = await _make_server().verify(body)
        assert result.status == FukuyamaBenchStatus.SCORED.value
        assert result.reward == 0.0

    async def test_verify_rejects_it_end_to_end(self) -> None:
        body = _make_request(
            _fenced([{"step_id": 1, "product_smiles": "CCO.not_a_smiles(("}]),
            gt_pathway=self.GT,
            checkpoints=[[1]],
        )
        result = await _make_server().verify(body)
        assert result.status == FukuyamaBenchStatus.SCORED.value
        assert result.reward == 0.0


class TestCorpusCompleteness:
    """A short corpus silently changes the denominator of every score.

    Counting directories is not enough: a complete listing whose contents are
    unreadable skips those cases during parsing and still produced a short split.
    """

    @staticmethod
    def _cases(tier: str, count: int) -> list[dict]:
        return [{"case_id": f"{tier}{i:03d}", "case_set": tier} for i in range(1, count + 1)]

    def test_short_tier_is_rejected(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            prepare.check_corpus_complete(self._cases("B", 1), {"B"})
        assert "loaded 1" in str(excinfo.value)

    def test_complete_tier_is_accepted(self) -> None:
        prepare.check_corpus_complete(self._cases("B", prepare.EXPECTED_CASES["B"]), {"B"})

    def test_synthetic_fixtures_are_exempt_from_the_manifest(self) -> None:
        prepare.check_corpus_complete(self._cases("S", 1), {"S"})

    def test_cli_fails_when_a_present_case_cannot_be_read(self, tmp_path, monkeypatch) -> None:
        """The directory is there and counted; its required content is not."""
        source = tmp_path / "mechanisms"
        shutil.copytree(FIXTURES, source)
        (source / "S003" / "ckpt.txt").unlink()
        output = tmp_path / "out.jsonl"
        monkeypatch.setattr(prepare, "EXPECTED_CASES", {"S": 5})
        monkeypatch.setattr(
            sys,
            "argv",
            ["prepare", "--output", str(output), "--sets", "S", "--source-dir", str(source)],
        )
        with pytest.raises(SystemExit) as excinfo:
            prepare.main()
        assert "loaded 4" in str(excinfo.value)
        assert not output.exists(), "a short split must not be left on disk"

    def test_cli_rejects_an_unknown_set_before_fetching(self, tmp_path, monkeypatch) -> None:
        """Rejecting the argument must not cost a download.

        Asserting only on SystemExit would still pass if the validation moved
        back below the fetch, because a warm cache or a working network hides
        the ordering. Stubbing the fetch makes the test independent of both.
        """

        def _must_not_fetch(*_args, **_kwargs):
            raise AssertionError("fetch_mechanisms() must not be called for an unknown set")

        monkeypatch.setattr(prepare, "fetch_mechanisms", _must_not_fetch)
        monkeypatch.setattr(
            sys,
            "argv",
            ["prepare", "--output", str(tmp_path / "out.jsonl"), "--sets", "Z"],
        )
        with pytest.raises(SystemExit) as excinfo:
            prepare.main()
        assert "Unknown set" in str(excinfo.value)

    @pytest.mark.parametrize("limit", ["0", "-1"])
    def test_cli_rejects_a_non_positive_limit(self, limit, tmp_path, monkeypatch) -> None:
        """0 previously meant "no limit" and -1 truncated to a single row."""
        output = tmp_path / "out.jsonl"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prepare",
                "--output",
                str(output),
                "--sets",
                "S",
                "--source-dir",
                str(FIXTURES),
                "--limit",
                limit,
            ],
        )
        with pytest.raises(SystemExit):
            prepare.main()
        assert not output.exists(), "a rejected limit must not write a split"

    def test_cli_accepts_a_positive_limit(self, tmp_path, monkeypatch) -> None:
        """A positive limit writes that many rows and skips the manifest check.

        The skip is asserted rather than assumed: the synthetic tier is exempt
        from the manifest anyway, so without the fail-if-called stub this would
        pass even if completeness were still being enforced.
        """

        def _must_not_check(*_args, **_kwargs):
            raise AssertionError("check_corpus_complete() must not run for an explicit subset")

        monkeypatch.setattr(prepare, "check_corpus_complete", _must_not_check)
        output = tmp_path / "out.jsonl"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prepare",
                "--output",
                str(output),
                "--sets",
                "S",
                "--source-dir",
                str(FIXTURES),
                "--limit",
                "3",
            ],
        )
        prepare.main()
        assert len(output.read_text().strip().splitlines()) == 3


class TestTaskDataSchema:
    def test_every_documented_field_is_declared(self) -> None:
        assert set(TaskData.model_fields) == {
            "gt_pathway",
            "checkpoints",
            "lenient",
            "case_id",
            "case_set",
            "n_gt_steps",
            "conditions",
            "starting_reactants",
        }

    def test_extra_metadata_is_tolerated(self) -> None:
        """verifier_metadata carries provenance the scorer never reads."""
        assert TaskData(case_id="B001", unknown_future_field=1).case_id == "B001"


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

    def test_a_json_array_that_is_not_a_pathway_yields_nothing(self) -> None:
        """A results array of bare values is not a step list."""
        assert _extract_pathway("## Result\n\n[1, 2, 3]") is None

    def test_plain_string_content_is_read(self) -> None:
        """Some responses carry content as a string rather than a parts list."""
        response = _make_response("ignored")
        response.output[0].content = _fenced(_pathway("CCO"))
        request = FukuyamaBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "Q"}]},
            response=response,
            verifier_metadata={"gt_pathway": GT_PATHWAY, "checkpoints": [[1]]},
        )
        assert _extract_last_assistant_text(request).startswith("## Reasoning")

    def test_malformed_json_yields_nothing(self) -> None:
        assert _extract_pathway('```json\n[{"step_id": 1, "product_smiles":}]\n```') is None


class TestHelpers:
    def test_strip_atom_mapping(self) -> None:
        assert strip_atom_mapping("[CH3:1][O:22]") == "[CH3][O]"

    def test_canonical_species_reports_unparseable_components(self) -> None:
        assert canonical_species(["CCO", "not_a_smiles(("]) == ({"CCO"}, False)
        assert canonical_species(["CCO"]) == ({"CCO"}, True)

    def test_empty_and_blank_input_is_handled(self) -> None:
        """Models emit empty strings and stray separators; neither is an error."""
        assert strip_atom_mapping("") == ""
        assert canonical_smiles("") is None
        assert canonical_species(["", None, "CCO..", "  "]) == ({"CCO"}, True)

    def test_compare_reports_why_it_failed(self) -> None:
        assert compare_step_products(["CCO"], ["CC=O"])[1] == "mismatch"
        assert compare_step_products(["CCO", "c1ccccc1"], ["CCO"])[1] == "superset_match"
        assert compare_step_products([], ["CCO"])[1] == "invalid_pred"

    def test_unparseable_gold_is_never_a_match(self) -> None:
        """An organometallic RDKit still cannot read must not pass by default."""
        assert compare_step_products(["CCO"], ["not_a_smiles(("]) == (False, "invalid_gt")
        assert compare_step_products([], []) == (False, "both_invalid")


class TestAlternativeStepShapes:
    """Upstream's extractor accepts `products` as well as `product_smiles`.

    A model emitting the alternative shape must still be scored rather than
    silently reading as an empty prediction.
    """

    GT = [{"step_id": "1", "products": ["CCO"]}]

    def test_product_smiles_as_a_list(self) -> None:
        result = score_pathway([{"product_smiles": ["CCO"]}], self.GT, [[1]])
        assert result["exact_match"] is True

    def test_products_as_a_list_of_strings(self) -> None:
        result = score_pathway([{"products": ["CCO"]}], self.GT, [[1]])
        assert result["exact_match"] is True

    def test_products_as_a_list_of_dicts(self) -> None:
        result = score_pathway([{"products": [{"smiles": "CCO"}]}], self.GT, [[1]])
        assert result["exact_match"] is True

    def test_a_step_with_no_recognised_product_field_scores_nothing(self) -> None:
        result = score_pathway([{"step_id": 1, "reaction_type": "addition"}], self.GT, [[1]])
        assert result["exact_match"] is False
