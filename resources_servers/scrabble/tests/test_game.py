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
import itertools
import random

import pytest

from resources_servers.scrabble import game


WORDS = frozenset({"AT", "CA", "CAT", "CATS", "QI", "TO", "DOG", "RETAINS", "TA", "AS", "SAT", "TAO", "OAT", "ACT"})
TRIE = game.build_trie(WORDS)


def board_with(word: str, row: int, col: int, direction: str = "across") -> list[str]:
    """Board holding ``word`` from 0-based (row, col)."""
    dr, dc = (0, 1) if direction == "across" else (1, 0)
    return game.apply_move(game.empty_board(), [(row + dr * i, col + dc * i, ch) for i, ch in enumerate(word)])


CAT_BOARD = board_with("CAT", 7, 7)  # H8-J8


class TestLayout:
    def test_premium_square_counts(self):
        assert (len(game.TRIPLE_WORD), len(game.DOUBLE_WORD)) == (8, 17)
        assert (len(game.TRIPLE_LETTER), len(game.DOUBLE_LETTER)) == (12, 24)

    def test_premiums_are_disjoint_and_symmetric_about_the_diagonal(self):
        premiums = [game.TRIPLE_WORD, game.DOUBLE_WORD, game.TRIPLE_LETTER, game.DOUBLE_LETTER]
        for a, b in itertools.combinations(premiums, 2):
            assert not a & b
        for squares in premiums:
            assert squares == {(c, r) for r, c in squares}

    def test_center_is_double_word(self):
        assert game.WORD_MULTIPLIER[7][7] == 2

    def test_bag_has_100_tiles_and_187_points(self):
        bag = game.full_bag()
        assert len(bag) == 100
        assert sum(game.LETTER_VALUES.get(t, 0) for t in bag) == 187


class TestScoring:
    def test_first_move_doubles_on_center(self):
        result = game.evaluate_move(game.empty_board(), "QI", "QI", 7, 6, "across", WORDS)
        assert result.valid and result.score == (10 + 1) * 2

    def test_blank_scores_zero(self):
        result = game.evaluate_move(game.empty_board(), "Q?", "QI", 7, 6, "across", WORDS)
        assert result.valid and result.score == 10 * 2
        assert (7, 7, "i") in result.placed

    def test_bingo_adds_fifty(self):
        # R E T A I N S on H8-N8: center doubles the word and L8 doubles the I.
        result = game.evaluate_move(game.empty_board(), "AEINRST", "RETAINS", 7, 7, "across", WORDS)
        assert result.valid and result.score == (1 + 1 + 1 + 1 + 2 + 1 + 1) * 2 + 50

    def test_existing_tiles_count_without_premiums(self):
        # CAT sits on the center DW, which no longer applies; K8 is a plain square.
        result = game.evaluate_move(CAT_BOARD, "S", "CATS", 7, 7, "across", WORDS)
        assert result.valid and result.score == 3 + 1 + 1 + 1

    def test_cross_words_are_scored(self):
        # AT on H9-I9 forms CA (H8-H9) and AT (I8-I9); I9 is a DL under the new T.
        result = game.evaluate_move(CAT_BOARD, "AT", "AT", 8, 7, "across", WORDS)
        assert result.valid
        assert sorted(result.words) == ["AT", "AT", "CA"]
        assert result.score == (1 + 1 * 2) + (3 + 1) + (1 + 1 * 2)

    def test_blank_goes_where_it_costs_least(self):
        # Both A squares need a tile and there is one real A: the real one goes on the heavier square.
        placed = game._assign_tiles("A?", [(0, 0, "A"), (0, 1, "A")], {(0, 0): 1, (0, 1): 6})
        assert placed == {(0, 1): "A", (0, 0): "a"}


class TestIllegalMoves:
    @pytest.mark.parametrize(
        "board, rack, word, row, col, direction, message",
        [
            (game.empty_board(), "QI", "QI", 0, 0, "across", "center square"),
            (CAT_BOARD, "DOG", "DOG", 0, 0, "across", "connect"),
            (CAT_BOARD, "O", "COT", 7, 7, "across", "already holds A"),
            (CAT_BOARD, "S", "ATS", 7, 8, "across", "touch the ends"),
            (CAT_BOARD, "X", "CATX", 7, 7, "across", "not in the dictionary: CATX"),
            (CAT_BOARD, "Q", "CATS", 7, 7, "across", "lacks tiles"),
            (CAT_BOARD, "S", "CATS", 7, 7, "sideways", "direction"),
            (game.empty_board(), "AEINRST", "RETAINS", 7, 10, "across", "off the board"),
            (CAT_BOARD, "", "CAT", 7, 7, "across", "at least one tile"),
            (game.empty_board(), "A", "A", 7, 7, "across", "at least 2 letters"),
        ],
    )
    def test_rejected_with_reason(self, board, rack, word, row, col, direction, message):
        result = game.evaluate_move(board, rack, word, row, col, direction, WORDS)
        assert not result.valid
        assert message in result.error

    def test_touching_tiles_error_spells_the_whole_line(self):
        # ATS on I8-K8 runs into the C on H8, so the line actually reads CATS.
        result = game.evaluate_move(CAT_BOARD, "S", "ATS", 7, 8, "across", WORDS)
        assert "would read 'CATS' from H8" in result.error

    def test_conflicting_square_is_reported_before_touching_tiles(self):
        # BAT on G8-I8 touches the T on J8, but the clash with the C on H8 is the actionable error.
        result = game.evaluate_move(CAT_BOARD, "B", "BAT", 7, 6, "across", WORDS)
        assert "square H8 already holds C" in result.error

    def test_letter_case_is_ignored(self):
        assert game.evaluate_move(CAT_BOARD, "S", "cats", 7, 7, "across", WORDS).score == 6


class TestMoveGeneration:
    def test_finds_best_opening(self):
        moves = game.generate_moves(game.empty_board(), "QI", TRIE, WORDS)
        assert max(m.score for m in moves) == 22
        # Every opening covers the center.
        assert all(any((r, c) == game.CENTER for r, c, _ in m.placed) for m in moves)

    def test_every_generated_move_is_legal_with_the_same_score(self):
        rack = "AOST?"
        moves = game.generate_moves(CAT_BOARD, rack, TRIE, WORDS)
        assert moves
        for move in moves:
            result = game.evaluate_move(CAT_BOARD, rack, move.word, move.row, move.col, move.direction, WORDS)
            assert result.valid, move
            # The checker assigns blanks optimally, so it never scores a placement lower than the generator.
            assert result.score >= move.score, move

    def test_matches_brute_force_on_small_racks(self):
        # Every placement of every permutation of the rack, checked one by one, finds the same move set.
        rack = "AOST"
        generated = {frozenset(m.placed) for m in game.generate_moves(CAT_BOARD, rack, TRIE, WORDS)}
        brute = set()
        for direction in game.DIRECTIONS:
            dr, dc = (0, 1) if direction == "across" else (1, 0)
            for r, c, k in itertools.product(range(15), range(15), range(1, len(rack) + 1)):
                for tiles in set(itertools.permutations(rack, k)):
                    placed, rr, cc = [], r, c
                    while len(placed) < k and rr < 15 and cc < 15:
                        if CAT_BOARD[rr][cc] == game.EMPTY:
                            placed.append((rr, cc, tiles[len(placed)]))
                        rr, cc = rr + dr, cc + dc
                    if len(placed) < k:
                        continue
                    after = game.apply_move(CAT_BOARD, placed)
                    if r - dr >= 0 and c - dc >= 0 and after[r - dr][c - dc] != game.EMPTY:
                        continue  # The word starts earlier; it is enumerated from there.
                    if after[r][c] == game.EMPTY:
                        continue
                    end_r, end_c = r, c
                    while end_r + dr < 15 and end_c + dc < 15 and after[end_r + dr][end_c + dc] != game.EMPTY:
                        end_r, end_c = end_r + dr, end_c + dc
                    word = "".join(after[r + dr * i][c + dc * i] for i in range(max(end_r - r, end_c - c) + 1))
                    if game.evaluate_move(CAT_BOARD, rack, word, r, c, direction, WORDS).valid:
                        brute.add(frozenset(placed))
        assert generated == brute

    def test_self_play_stays_consistent(self):
        rng = random.Random(0)
        bag = game.full_bag()
        board, rack = game.empty_board(), "ACOST"
        for _ in range(3):
            moves = game.generate_moves(board, rack, TRIE, WORDS)
            if not moves:
                break
            move = max(moves, key=lambda m: m.score)
            board = game.apply_move(board, move.placed)
            rack = game.draw(bag, game.remove_from_rack(rack, move.placed), rng)
        assert sum(ch != game.EMPTY for row in board for ch in row) >= 3
        game.validate_board(board)


class TestHelpers:
    def test_remove_from_rack_returns_blanks_as_question_marks(self):
        assert game.remove_from_rack("AB?", [(0, 0, "A"), (0, 1, "z")]) == "B"

    def test_render_board_labels_center_and_tiles(self):
        text = game.render_board(CAT_BOARD)
        assert " C  A  T " in text
        assert text.splitlines()[1].startswith(" 1  3W ")

    @pytest.mark.parametrize("board", [["." * 15] * 14, ["." * 14] * 15, ["1" * 15] * 15])
    def test_validate_board_rejects_malformed(self, board):
        with pytest.raises(ValueError):
            game.validate_board(board)

    @pytest.mark.parametrize("rack", ["", "ABCDEFGH", "ab", "A1"])
    def test_validate_rack_rejects_malformed(self, rack):
        with pytest.raises(ValueError):
            game.validate_rack(rack)
