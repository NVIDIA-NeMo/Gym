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

"""Boundary contracts the scorer relies on but no other test reaches.

Each case here is a branch that decides whether a bad input becomes a loud
failure or a quiet wrong number. They are cheap, and the quiet-wrong-number
outcome is the expensive one on a benchmark.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from app import MolDeToxResourcesServer, MolDeToxStatus, MolDeToxVerifyRequest
from metrics import decode_safe, exact_match, fragment_levenshtein, parse_fragments, score_fragments, score_safe

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_moldetox  # noqa: E402


class TestGoldIsTrusted:
    """An unusable gold must raise, not score zero.

    A zero would be indistinguishable from the model missing, so a corrupt row
    would quietly depress the benchmark instead of failing the run.
    """

    def test_unparseable_smiles_gold_raises(self) -> None:
        with pytest.raises(ValueError, match="not valid SMILES"):
            exact_match("CCO", "not-a-molecule")

    def test_undecodable_safe_gold_raises(self) -> None:
        with pytest.raises(ValueError, match="does not decode"):
            score_safe("CCO", "%%%not-safe%%%")


class TestNoneGuards:
    """Absent input must return an absent metric, never a flattering number."""

    @pytest.mark.parametrize("bad", [None, 123, ""])
    def test_decode_safe_rejects_non_strings(self, bad) -> None:
        assert decode_safe(bad) is None

    @pytest.mark.parametrize("bad", [None, 123, ""])
    def test_parse_fragments_rejects_non_strings(self, bad) -> None:
        assert parse_fragments(bad) == []

    def test_empty_prediction_scores_zero_f1_not_perfect(self) -> None:
        scores = score_fragments("", "A.B", task="task1")
        assert scores["fragment_f1"] == 0.0
        assert scores["exact_match"] == 0.0

    def test_fragment_distance_is_none_when_there_is_nothing_to_average(self) -> None:
        """Zero would read as a perfect match; None is excluded from the mean instead."""
        assert fragment_levenshtein([], ["A"]) is None
        assert fragment_levenshtein(["A"], []) is None


class TestAssistantTextShapes:
    def test_plain_string_content_is_read(self) -> None:
        """Some providers return `content` as a bare string rather than a parts list."""
        config = BaseResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        server = MolDeToxResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        response = NeMoGymResponse(
            id="r",
            created_at=0.0,
            model="m",
            object="response",
            output=[
                {
                    "id": "m1",
                    "content": '{"answer": "CCO"}',
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        request = MolDeToxVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "Q"}]},
            response=response,
            verifier_metadata={"answer": "CCO", "scoring_mode": "smiles"},
        )
        import asyncio

        result = asyncio.run(server.verify(request))
        assert result.status == MolDeToxStatus.SCORED.value
        assert result.reward == 1.0


class TestTaskDataSchema:
    """`task_data.py` is the published description of the row format.

    Nothing imports it at runtime, so it can drift from what `verify()` actually
    reads and then documents the wrong thing to every future reader.
    """

    def test_schema_covers_every_field_the_preparer_emits(self) -> None:
        from task_data import TaskData

        row = {
            "question": "Q",
            "answer": {"answer": "CCO"},
            "endpoint": "e",
            "dataset_name": "d",
            "source_index": 1,
            "id": 7,
            "common_safe_fragments": "A.B",
        }
        emitted = prepare_moldetox.format_row(row, "task2_single", {1: "CBr"})["verifier_metadata"]
        assert set(emitted) <= set(TaskData.model_fields)

    def test_schema_accepts_a_prepared_row(self) -> None:
        from task_data import TaskData

        row = {
            "question": "Q",
            "answer": {"answer": "CCO"},
            "endpoint": "e",
            "dataset_name": "d",
            "source_index": 1,
            "id": 7,
        }
        emitted = prepare_moldetox.format_row(row, "task3_smiles_gen_single", {1: "CBr"})["verifier_metadata"]
        parsed = TaskData(**emitted)
        assert parsed.answer == "CCO"
        assert parsed.scoring_mode == "smiles"
        assert parsed.toxic_smiles == "CBr"


class TestTask2CarriesSharedFragments:
    def test_common_safe_fragments_is_carried_when_upstream_ships_it(self) -> None:
        """Only the Task 2 configs have it, and the splice limitation depends on it surviving."""
        row = {
            "question": "Q",
            "answer": {"answer": "A.B"},
            "endpoint": "e",
            "dataset_name": "d",
            "source_index": 1,
            "id": 7,
            "common_safe_fragments": "C.D",
        }
        meta = prepare_moldetox.format_row(row, "task2_single", {1: "CBr"})["verifier_metadata"]
        assert meta["common_safe_fragments"] == "C.D"

    def test_absent_on_configs_that_do_not_ship_it(self) -> None:
        row = {
            "question": "Q",
            "answer": {"answer": "CCO"},
            "endpoint": "e",
            "dataset_name": "d",
            "source_index": 1,
            "id": 7,
        }
        meta = prepare_moldetox.format_row(row, "task3_smiles_gen_single", {1: "CBr"})["verifier_metadata"]
        assert "common_safe_fragments" not in meta
