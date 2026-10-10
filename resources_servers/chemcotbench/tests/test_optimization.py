# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib.util
import os
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from nemo_gym.server_utils import ServerClient
from resources_servers.chemcotbench.app import (
    ChemCoTBenchResourcesServer,
    ChemCoTBenchResourcesServerConfig,
    ChemCoTBenchVerifyRequest,
)
from resources_servers.chemcotbench.setup_upstream import ensure_repository
from resources_servers.chemcotbench.tests.test_app import response


pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def optimization_server():
    server = ChemCoTBenchResourcesServer(
        config=ChemCoTBenchResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="chemcotbench",
            repo_path=str(ensure_repository(os.environ.get("CHEMCOTBENCH_TEST_REPO"))),
            run_layer3=False,
            molopt_python=os.environ.get("CHEMCOTBENCH_TEST_MOLOPT_PYTHON"),
            oracle_dir=os.environ.get("CHEMCOTBENCH_TEST_ORACLE_DIR"),
        ),
        server_client=MagicMock(spec=ServerClient),
    )

    yield server
    await server.close()


def optimization_request(text, subtask="logp", src="CCO"):
    return ChemCoTBenchVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "Optimize the source molecule."}]},
        response=response(text),
        verifier_metadata={
            "id": "synthetic",
            "task_family": "mol_opt",
            "subtask": subtask,
            "upstream_record": {
                "src": src,
                "tgt": "CCCC",
                "answer_smiles": "CCCC",
                "predicted_smiles": "CCCC",
                "step3_predicted_smiles": "CCCC",
                "step1_scaffold_smiles": "C",
                "step2_fg_removed": "O",
                "step2_fg_added": "C",
                "step4_scaffold_claimed": "yes",
                "step5_fg_consistent_claimed": "yes",
            },
        },
    )


@pytest.mark.parametrize(
    "text,prediction,reward",
    [
        ("[PRODUCT_CONSTRUCTION]\nPredicted SMILES: CCCC\nAnswer: CCCC", "CCCC", 1.0),
        ('Step 3 [PRODUCT_CONSTRUCTION]: PREDICTED_SMILES("CCCC")\nAnswer: CCCC', "CCCC", 1.0),
        ("[PRODUCT_CONSTRUCTION]\nPredicted SMILES: CCO\nAnswer: CCO", "CCO", 0.0),
        ("[PRODUCT_CONSTRUCTION]\nPredicted SMILES: invalid\nAnswer: invalid", "invalid", 0.0),
        ("I cannot answer.", None, 0.0),
        ("Answer: CCCC", None, 0.0),
        ("", None, 0.0),
    ],
)
async def test_optimization_outcomes(optimization_server, text, prediction, reward):
    body = optimization_request(text)
    original = copy.deepcopy(body.model_dump())
    result = await optimization_server.verify(body)
    assert result.scoring_error is None, result.failure_reason
    assert result.reward == reward
    assert result.predicted_answer == prediction
    assert result.layer3_step_score is None
    assert body.model_dump() == original
    if prediction == "CCCC":
        assert result.optimization_metrics["layer1_delta"] > 0
        assert result.layer1_fts == 1


@pytest.mark.parametrize(
    "subtask",
    [
        "logp",
        "qed",
        "solubility",
        "drd",
        "gsk",
        "jnk",
        "logp_qed",
        "logp_solubility",
        "qed_solubility",
        "drd_logp",
        "drd_solubility",
        "gsk_logp",
    ],
)
async def test_unchanged_molecule_never_improves(optimization_server, subtask):
    body = optimization_request("[PRODUCT_CONSTRUCTION]\nPredicted SMILES: CCO\nAnswer: CCO", subtask)
    result = await optimization_server.verify(body)
    assert result.scoring_error is None, result.failure_reason
    assert result.reward == 0
    metrics = result.optimization_metrics
    deltas = [v for k, v in metrics.items() if k == "layer1_delta" or k.startswith("delta_")]
    assert deltas and all(delta == 0 for delta in deltas)


async def test_optimization_layer3_does_not_inherit_gold(optimization_server):
    optimization_server.config.run_layer3 = True
    try:
        result = await optimization_server.verify(optimization_request("I cannot answer."))
        assert result.scoring_error is None, result.failure_reason
        assert result.reward == 0 and result.layer3_step_score == 0
        assert result.layer3_type1 is None and result.layer3_type2 is None
    finally:
        optimization_server.config.run_layer3 = False


@pytest.mark.skipif(importlib.util.find_spec("rdkit") is None, reason="Requires RDKit")
@pytest.mark.parametrize("source,expected", [("mutated", "Different"), ("permutated", "Same")])
async def test_equivalence_answers(optimization_server, source, expected):
    body = optimization_request("Answer: " + expected)
    body.verifier_metadata.task_family = "mol_und"
    body.verifier_metadata.subtask = "smiles_equivalent"
    body.verifier_metadata.upstream_record = {
        "source_subtask": source,
        "smiles": "CCO",
        "smiles_a": "CCO",
        "smiles_b": "OCC" if source == "permutated" else "CCC",
        "gt_answer": expected,
        "expected_predict": expected,
    }
    correct = await optimization_server.verify(body)
    body.response = response("Answer: " + ("Same" if expected == "Different" else "Different"))
    incorrect = await optimization_server.verify(body)
    assert correct.scoring_error is None and incorrect.scoring_error is None
    assert correct.reward == 1 and incorrect.reward == 0
