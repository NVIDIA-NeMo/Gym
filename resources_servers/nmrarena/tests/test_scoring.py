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

"""Extraction and scoring against upstream's rules, on fixtures written for this file.

Expected values marked "upstream" were produced by running the functions in
``dataset/llm_track.ipynb`` and ``analysis/data_analysis.ipynb`` of
odanchem/NMRArena at ``8b4ca8a`` on the same strings, not by this implementation.
"""

import json

import pytest
from scoring import (
    canonical,
    extract_raw_candidates,
    hit_rank,
    parse_candidates,
    salvage_truncated_candidates,
    score_candidates,
    tanimoto,
)


GOLD = "CC(CCI)C"  # upstream canonical: CC(C)CCI
GARBAGE = "C(C)(C)(C)(C)C"  # pentavalent carbon, never parses


def cands(*smiles: str) -> str:
    return json.dumps({"candidates": [{"rank": i + 1, "smiles": s} for i, s in enumerate(smiles)]})


class TestCanonical:
    def test_strips_stereo_and_backticks(self) -> None:
        # upstream: canonical("C[C@H](O)CC") == "CCC(C)O"
        assert canonical("C[C@H](O)CC") == "CCC(C)O"
        assert canonical("`CCO`") == "CCO"
        assert canonical("ICCC(C)C") == canonical(GOLD) == "CC(C)CCI"

    @pytest.mark.parametrize("bad", ["", "   ", GARBAGE, None, 3, ["CCO"], "C1CC"])
    def test_unparseable_is_none(self, bad) -> None:
        assert canonical(bad) is None

    def test_length_cap_is_applied_before_rdkit(self, monkeypatch) -> None:
        import scoring

        def boom(*a, **k):
            raise AssertionError("RDKit must not see an oversize string")

        monkeypatch.setattr(scoring.Chem, "MolFromSmiles", boom)
        assert canonical("C" * 501, max_chars=500) is None
        assert tanimoto("C" * 501, "C") is None


class TestExtraction:
    def test_last_candidates_block_wins_and_rank_orders(self) -> None:
        """A draft block followed by a corrected one: the rightmost is the conclusion."""
        text = (
            cands("CCO")
            + "\nActually:\n"
            + json.dumps({"candidates": [{"rank": 2, "smiles": "CCC"}, {"rank": 1, "smiles": "CCCC"}]})
        )
        assert extract_raw_candidates(text) == ["CCCC", "CCC"]

    def test_comments_and_trailing_commas_are_tolerated(self) -> None:
        text = '{"candidates": [ // best\n {"rank": 1, "smiles": "CCO"}, ], }'
        assert extract_raw_candidates(text) == ["CCO"]

    def test_bare_strings_and_missing_smiles_keep_their_position(self) -> None:
        # upstream: ranked objects sort first by rank, then unranked members in file order
        text = json.dumps({"candidates": ["CCO", {"rank": 2}, 7, {"rank": 3, "SMILES": "CCC"}]})
        assert extract_raw_candidates(text) == [None, "CCC", "CCO", None]

    @pytest.mark.parametrize("text", ["", "no json here", '{"candidates": []}', '"candidates": [1', "{ }"])
    def test_nothing_usable_is_none(self, text) -> None:
        assert extract_raw_candidates(text) is None

    def test_salvage_recovers_complete_objects_from_a_cut_block(self) -> None:
        text = (
            '```json\n{"candidates": [{"rank": 1, "smiles": "CCO"}, {"rank": 2, "smiles": "CC\\"C"}, {"rank": 3, "smi'
        )
        assert extract_raw_candidates(text) is None
        assert salvage_truncated_candidates(text) == ["CCO", 'CC"C']

    def test_salvage_keeps_a_smiles_whose_backslash_is_not_a_json_escape(self) -> None:
        text = '{"candidates": [{"rank": 1, "smiles": "C/C=C\\C"}, {"rank": 2, "smiles": "CC'
        assert salvage_truncated_candidates(text) == ["C/C=C\\C"]

    def test_salvage_needs_the_key_and_one_complete_object(self) -> None:
        assert salvage_truncated_candidates('{"rank": 1, "smiles": "CCO"}') is None
        assert salvage_truncated_candidates('{"candidates": [{"rank": 1, "smiles": "CC') is None


class TestParseCandidates:
    def test_positions_are_ranks_and_invalid_entries_occupy_them(self) -> None:
        parsed = parse_candidates(cands(GARBAGE, GOLD, "ICCC(C)C"))
        assert parsed.candidates == [None, "CC(C)CCI", "CC(C)CCI"]
        assert (parsed.n_raw, parsed.n_invalid, parsed.n_duplicate, parsed.n_valid) == (3, 1, 1, 2)
        assert not parsed.salvaged and parsed.disqualified is None

    def test_only_the_first_n_positions_are_kept(self) -> None:
        parsed = parse_candidates(cands(*(["C"] * 10 + [GOLD])), n=10)
        assert len(parsed.candidates) == 10 and "CC(C)CCI" not in parsed.candidates

    def test_no_json_is_not_found(self) -> None:
        parsed = parse_candidates("I cannot determine the structure.")
        assert not parsed.found_json and parsed.candidates == []

    def test_json_with_nothing_valid_is_found_but_empty_of_structures(self) -> None:
        parsed = parse_candidates(cands(GARBAGE, "xyz"))
        assert parsed.found_json and parsed.n_valid == 0 and parsed.candidates == [None, None]

    def test_oversize_entry_is_counted_and_never_parsed(self) -> None:
        parsed = parse_candidates(cands("C" * 20000, GOLD), max_smiles_chars=500)
        assert parsed.candidates == [None, "CC(C)CCI"]
        assert parsed.n_oversize == 1 and parsed.n_invalid == 0

    @pytest.mark.parametrize(
        ("text", "reason"),
        [
            (cands(GOLD, GARBAGE), "invalid"),
            (cands(GOLD, "C" * 501), "oversize"),
            (cands(GOLD, "ICCC(C)C"), "duplicate"),
            (
                cands(GOLD, "C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC", "CCO"),
                "too_many",
            ),
        ],
    )
    def test_strict_disqualifies_right_plus_garbage(self, text, reason) -> None:
        lenient = parse_candidates(text)
        strict = parse_candidates(text, strict=True)
        assert lenient.candidates[0] == "CC(C)CCI"
        assert strict.disqualified == reason and strict.candidates == []

    def test_strict_accepts_ten_distinct_valid_entries(self) -> None:
        text = cands(*([GOLD] + ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC"]))
        parsed = parse_candidates(text, strict=True)
        assert parsed.disqualified is None and parsed.candidates[0] == "CC(C)CCI" and len(parsed.candidates) == 10

    def test_surrogate_escape_inside_the_json_is_neutralised_after_parsing(self) -> None:
        text = '{"candidates": [{"rank": 1, "smiles": "\\udcff"}, {"rank": 2, "smiles": "%s"}]}' % GOLD
        parsed = parse_candidates(text)
        assert parsed.candidates == [None, "CC(C)CCI"] and parsed.n_invalid == 1

    def test_salvage_is_used_only_when_the_block_does_not_load(self) -> None:
        cut = '{"candidates": [{"rank": 1, "smiles": "%s"}, {"rank": 2, "smi' % GOLD
        assert parse_candidates(cut).candidates == ["CC(C)CCI"]
        assert parse_candidates(cut).salvaged
        assert parse_candidates(cut, salvage=False).candidates == []
        assert not parse_candidates(cands(GOLD)).salvaged


class TestScores:
    def test_top1_top10_and_tanimoto_follow_upstream(self) -> None:
        truth = canonical(GOLD)
        # upstream: hit_rank("CC(CCI)C", ["CC(C)CCBr", "CC(C)CCI"]) == 2
        assert hit_rank(truth, ["CC(C)CCBr", "CC(C)CCI"]) == 2
        s = score_candidates(truth, ["CC(C)CCBr", "CC(C)CCI"])
        assert (s.top1, s.top10, s.answered) == (0.0, 1.0, 1.0)
        # upstream: tanimoto("CC(CCI)C", "CC(C)CCBr") == 0.4117647058823529
        assert s.tanimoto_top1 == pytest.approx(0.4117647058823529)

    def test_rank_eleven_is_not_top10_and_position_one_none_has_no_tanimoto(self) -> None:
        truth = canonical(GOLD)
        s = score_candidates(truth, [None] * 10 + [truth])
        assert (s.hit_rank, s.top10, s.tanimoto_top1, s.answered) == (11, 0.0, None, 1.0)
        assert score_candidates(truth, []).answered == 0.0
        assert score_candidates(truth, [None]).tanimoto_top1 is None

    def test_identity_scores_one(self) -> None:
        truth = canonical("C[C@H](O)CC")
        s = score_candidates(truth, [truth])
        assert (s.hit_rank, s.top1, s.tanimoto_top1) == (1, 1.0, 1.0)
