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

"""Reward-function level tests for the graders Gym overrides (see graders.py)."""

from unittest.mock import patch

import pytest
from ether0 import rewards

from resources_servers.ether0 import graders


class TestSolubilityGroups:
    REQUIRED = ["arylchloride", "carboxylic acid", "alkyne"]

    @pytest.mark.parametrize(
        ("prediction", "kind", "data", "oracle_value", "expected", "reason", "oracle_calls"),
        [
            pytest.param("c1ccccc1Cl", "groups", REQUIRED, -3.0, 0.0, "failed_constraint", 0, id="one-of-three"),
            pytest.param("O=C(O)c1ccccc1Cl", "groups", REQUIRED, -3.0, 0.0, "failed_constraint", 0, id="two-of-three"),
            pytest.param("O=C(O)c1cc(Cl)ccc1C#C", "groups", REQUIRED, -3.0, 1.0, None, 1, id="all-three"),
            pytest.param("CCO", "groups", REQUIRED, -3.0, 0.0, "failed_constraint", 0, id="none"),
            pytest.param("c1ccccc1Cl", "groups", ["ARYLCHLORIDE"], -3.0, 1.0, None, 1, id="case-insensitive"),
            pytest.param("CCL", "groups", REQUIRED, -3.0, 0.0, "invalid_mol", 0, id="invalid-smiles"),
            pytest.param(
                "O=C(O)c1cc(Cl)ccc1C#C", "groups", REQUIRED, -2.0, 0.0, "wrong_numerical_answer", 1, id="wrong-value"
            ),
            pytest.param("CCO", "scaffold", "CC", -3.0, 1.0, None, 1, id="scaffold-present"),
            pytest.param("CCO", "scaffold", "c1ccccc1", -3.0, 0.0, "failed_constraint", 0, id="scaffold-absent"),
            pytest.param("CCO", "tanimoto", "CCO", -3.0, 1.0, None, 1, id="tanimoto-identical"),
            pytest.param("CCO", "tanimoto", "c1ccccc1", -3.0, 0.0, "failed_constraint", 0, id="tanimoto-different"),
        ],
    )
    def test_constraints(self, prediction, kind, data, oracle_value, expected, reason, oracle_calls) -> None:
        metadata: dict = {}
        target = repr((kind, data, "-3.0", "increase"))
        # Isolate the constraint logic from the admissibility filter and the HTTP oracle;
        # RDKit parsing and exmol group detection stay real.
        with (
            patch.object(rewards, "is_reasonable_molecule", return_value=True),
            patch.object(rewards, "fetch_solubility", return_value={"solubility": oracle_value}) as oracle,
        ):
            actual = graders.oracle_solubility_eval(prediction, target, metadata=metadata)
        assert actual == expected
        assert metadata.get("reward_reason") == reason
        assert oracle.call_count == oracle_calls

    def test_registered(self) -> None:
        assert graders.EVAL_FUNCTIONS["sol_eval"] is graders.oracle_solubility_eval


class TestStrEval:
    @pytest.mark.parametrize(
        ("prediction", "reference", "expected"),
        [
            pytest.param("OCC", "CCO", 1.0, id="same-molecule"),
            pytest.param("CCL", "CCl", 0.0, id="invalid-smiles"),
            pytest.param("C1CCCCC1", "c1ccccc1", 0.0, id="different-molecule"),
            pytest.param("Suzuki-Miyaura coupling", "suzuki-miyaura coupling", 1.0, id="text-case-insensitive"),
            pytest.param("Suzuki-Miyaura coupling", "Heck reaction", 0.0, id="text-different"),
            pytest.param("F[C@H](Cl)Br", "F[C@@H](Cl)Br", 0.0, id="stereo-differs"),
        ],
    )
    def test_dispatch(self, prediction: str, reference: str, expected: float) -> None:
        assert graders.str_eval(prediction, reference) == expected

    def test_registered(self) -> None:
        assert graders.EVAL_FUNCTIONS["str_eval"] is graders.str_eval


class TestProductFragments:
    @pytest.mark.parametrize(
        ("prediction", "reference", "expected"),
        [
            pytest.param("CCCC.", "CCCC", 0.0, id="trailing-empty"),
            pytest.param(".CCCC", "CCCC", 0.0, id="leading-empty"),
            pytest.param("CCCC..", "CCCC", 0.0, id="double-dot"),
            pytest.param("CCCC. .CO", "CCCC.CO", 0.0, id="blank-fragment"),
            pytest.param("CCCC.this_is_not_SMILES", "CCCC", 0.0, id="garbage-fragment"),
            pytest.param("CCCC.CO", "CCCC", 1.0, id="smaller-valid-component-ungraded"),
            pytest.param("CCO", "CCO", 1.0, id="exact"),
            pytest.param("CCCC.[Na+]", "CCCC.[Na+]", 1.0, id="salt-both-sides"),
            pytest.param("CCCC.CCCO", "CCCO.CCCC", 1.0, id="order-independent"),
        ],
    )
    def test_product_eval(self, prediction: str, reference: str, expected: float) -> None:
        assert rewards.product_eval(prediction, reference) == expected

    @pytest.mark.parametrize(
        ("prediction", "reference"),
        [
            # A prompt may ask for an ammonium iodide while the reference stores only the cation.
            pytest.param("o1c(C[N+](C)(C)C)ccc1C.[I-]", "CC1=CC=C(C[N+](C)(C)C)O1", id="anion-last"),
            pytest.param("[I-].o1c(C[N+](C)(C)C)ccc1C", "CC1=CC=C(C[N+](C)(C)C)O1", id="anion-first"),
            pytest.param("CCN(CC)CC.Cl", "CCN(CC)CC", id="hydrochloride"),
            pytest.param("CCO.[Na+]", "CCO", id="sodium"),
            pytest.param("CCCCCCCCN.O=S(=O)(O)O", "CCCCCCCCN", id="sulfate"),
        ],
    )
    @pytest.mark.parametrize("evaluate", [rewards.product_eval, rewards.caption_eval], ids=lambda f: f.__name__)
    @pytest.mark.parametrize("soft", [False, True], ids=["exact", "soft"])
    def test_salt_components_are_tolerated(self, prediction, reference, evaluate, soft) -> None:
        metadata: dict = {}
        assert evaluate(prediction, reference, soft=soft, metadata=metadata) == 1.0
        assert "reward_reason" not in metadata
        assert metadata["full_smiles"] == rewards.MolToSmiles(graders.get_largest_mol(reference))

    @pytest.mark.parametrize("suffix", ["I-", "not_SMILES", ""])
    @pytest.mark.parametrize("evaluate", [rewards.product_eval, rewards.caption_eval], ids=lambda f: f.__name__)
    def test_invalid_fragments_report_reason(self, suffix: str, evaluate) -> None:
        reference = "CC1=CC=C(C[N+](C)(C)C)O1"
        metadata: dict = {}
        assert evaluate(reference + "." + suffix, reference, metadata=metadata) == 0.0
        assert metadata["reward_reason"] == "invalid_mol"

    def test_helper_is_rebound_in_ether0(self) -> None:
        assert rewards.get_largest_mol is graders.get_largest_mol
