# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the in-process oracles (see local_oracles.py).

The adapter and configuration tests run everywhere. The tests under ``TestRealOracles`` load the
real models and need the two oracle files: set ``ETHER0_REMOTES_MOLTRANS_MODEL_PATH`` and
``ETHER0_ZINC20_BLOOM_PATH`` to run them.
"""

import os
from unittest.mock import MagicMock

import pytest
from ether0 import rewards
from fastapi import HTTPException

from resources_servers.ether0 import graders, local_oracles


MOLTRANS_PATH = os.environ.get("ETHER0_REMOTES_MOLTRANS_MODEL_PATH")
ZINC20_PATH = os.environ.get("ETHER0_ZINC20_BLOOM_PATH")


class TestInstallRejectsBadConfiguration:
    def test_download_failure_stops_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        download = MagicMock(side_effect=RuntimeError("download failed"))
        monkeypatch.setattr(local_oracles, "ensure_moltrans", download)
        with pytest.raises(RuntimeError, match="download failed"):
            local_oracles.install(None, None)
        download.assert_called_once_with(None)

    def test_file_missing(self, tmp_path) -> None:
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"checkpoint")
        with pytest.raises(FileNotFoundError, match="zinc20.bloom"):
            local_oracles.install(str(checkpoint), str(tmp_path / "zinc20.bloom"))

    def test_checkpoint_is_not_the_pinned_file(self, tmp_path) -> None:
        checkpoint, bloom = tmp_path / "model.pt", tmp_path / "zinc20.bloom"
        checkpoint.write_bytes(b"not the checkpoint")
        bloom.write_bytes(b"filter")
        with pytest.raises(ValueError, match="not the pinned MolTrans checkpoint"):
            local_oracles.install(str(checkpoint), str(bloom))
        assert rewards.fetch_forward_rxn is not local_oracles.fetch_forward_rxn


class TestSolubilityAdapter:
    def test_unpredictable_molecule_is_an_error_reply(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(local_oracles, "_solubility", MagicMock(run=MagicMock(return_value=False)))
        assert local_oracles.fetch_solubility("CCO") == {"smiles": "CCO", "error": "Solubility prediction failed."}

    def test_model_failure_is_not_an_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(local_oracles, "_solubility", MagicMock(run=MagicMock(side_effect=RuntimeError("boom"))))
        with pytest.raises(local_oracles.OracleError, match="boom"):
            local_oracles.fetch_solubility("CCO")


class TestForwardReactionAdapter:
    def test_unparsable_reactants_are_an_error_reply(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = MagicMock()
        model.canonicalize_smiles.side_effect = HTTPException(status_code=400, detail="invalid: xx")
        monkeypatch.setattr(local_oracles, "_moltrans", model)
        assert local_oracles.fetch_forward_rxn("xx>>") == {"smiles": "xx>>", "error": "invalid: xx"}
        model.run.assert_not_called()

    def test_reactants_and_reagents_are_translated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = MagicMock()
        model.canonicalize_smiles.side_effect = lambda smiles: smiles
        model.run.return_value = ("CCOC(C)=O", "job")
        monkeypatch.setattr(local_oracles, "_moltrans", model)
        reply = local_oracles.fetch_forward_rxn("CC(=O)Cl.CCO > CN > CCOC(C)=O")
        assert reply == {"smiles": "CC(=O)Cl.CCO > CN > CCOC(C)=O", "product": "CCOC(C)=O"}
        model.run.assert_called_once_with("CC(=O)Cl.CCO.CN")

    def test_model_failure_is_not_an_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = MagicMock()
        model.canonicalize_smiles.side_effect = lambda smiles: smiles
        model.run.side_effect = RuntimeError("boom")
        monkeypatch.setattr(local_oracles, "_moltrans", model)
        with pytest.raises(local_oracles.OracleError, match="boom"):
            local_oracles.fetch_forward_rxn("CCO>>")


@pytest.mark.skipif(not (MOLTRANS_PATH and ZINC20_PATH), reason="the ether0 oracle files are not configured")
class TestRealOracles:
    @pytest.fixture(autouse=True, scope="class")
    def _installed(self):
        saved = {name: getattr(rewards, name) for name in local_oracles.ORACLE_FUNCTIONS}
        local_oracles.install(MOLTRANS_PATH, ZINC20_PATH)
        yield
        for name, function in saved.items():
            setattr(rewards, name, function)

    def test_reward_functions_use_the_local_oracles(self) -> None:
        for name in local_oracles.ORACLE_FUNCTIONS:
            assert getattr(rewards, name) is getattr(local_oracles, name)

    def test_purchasability_reads_the_full_catalog(self) -> None:
        reply = local_oracles.fetch_purchasable(["CCO", "CC(=O)O", "C1CC1C2CC2C3CC3C4CC4N(=O)=O"])
        assert reply == {"CCO": True, "CC(=O)O": True, "C1CC1C2CC2C3CC3C4CC4N(=O)=O": False}

    def test_solubility_reward_on_and_off_target(self) -> None:
        smiles = "COc1ccc2nc(NC(=O)c3c(F)cccc3F)sc2c1"
        predicted = local_oracles.fetch_solubility(smiles)["solubility"]
        on_target = repr(("tanimoto", smiles, str(predicted), "decrease"))
        off_target = repr(("tanimoto", smiles, str(predicted + 1.0), "decrease"))
        assert graders.EVAL_FUNCTIONS["sol_eval"](smiles, on_target) == 1.0
        metadata: dict = {}
        assert graders.EVAL_FUNCTIONS["sol_eval"](smiles, off_target, metadata=metadata) == 0.0
        assert metadata["reward_reason"] == "wrong_numerical_answer"

    def test_retrosynthesis_reward_right_and_wrong_product(self) -> None:
        reaction = "CC(=O)Cl.CCO>>"
        product = local_oracles.fetch_forward_rxn(reaction)["product"]
        assert graders.EVAL_FUNCTIONS["rxn_forward"](reaction, product) == 1.0
        metadata: dict = {}
        assert graders.EVAL_FUNCTIONS["rxn_forward"](reaction, "c1ccccc1", metadata=metadata) == 0.0
        assert metadata["reward_reason"] == "wrong_product"
