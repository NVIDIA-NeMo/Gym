# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import get_args

import pytest

from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.vision2code.app import VERIFIER_FIXTURE
from resources_servers.vision2code.evaluation import RubricName
from resources_servers.vision2code.generic_rubric import GENERIC_RUBRIC
from resources_servers.vision2code.rubrics import DATASET_RUBRICS
from resources_servers.vision2code.tests.verifier_fixture import RubricRequest, RubricVerifier


async def test_verifier_fixture():
    results = await exercise_verifier_fixture(VERIFIER_FIXTURE, reward_range=(0.0, 1.0), determinism="stochastic")
    assert len(results) == 51


@pytest.mark.parametrize("name", get_args(RubricName))
def test_fixture_penalizes_partial_scores(name):
    rubric = GENERIC_RUBRIC if name == "generic" else DATASET_RUBRICS[name]
    body = RubricRequest(
        dataset=name,
        parsed={"category_scores": {c["id"]: 3 for c in rubric["categories"]}, "rationales": {}},
        inspection={"candidate_readable": True, "is_blank_or_degenerate": False},
    )
    assert RubricVerifier().verify(body) == pytest.approx(0.36)
