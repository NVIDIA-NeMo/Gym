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
"""Scrabble rules: board layout, move validation, scoring, and exhaustive move generation.

Standard library only, so the data generator and the server share one implementation of the rules.

A board is 15 strings of 15 characters: ``.`` is an empty square, ``A``-``Z`` a lettered tile, and
``a``-``z`` a blank tile played as that letter (worth 0 points). A rack is a string of ``A``-``Z`` and
``?`` (an unplayed blank). Rows and columns are 0-based internally; the model sees rows 1-15 and
columns A-O.
"""

import random
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal


BOARD_SIZE = 15
CENTER = (7, 7)
RACK_SIZE = 7
BINGO_BONUS = 50
BLANK = "?"
EMPTY = "."
COLUMNS = "ABCDEFGHIJKLMNO"

Direction = Literal["across", "down"]
DIRECTIONS: tuple[Direction, ...] = ("across", "down")

LETTER_VALUES: dict[str, int] = {
    **dict.fromkeys("AEIOULNSTR", 1),
    **dict.fromkeys("DG", 2),
    **dict.fromkeys("BCMP", 3),
    **dict.fromkeys("FHVWY", 4),
    "K": 5,
    **dict.fromkeys("JX", 8),
    **dict.fromkeys("QZ", 10),
}

TILE_COUNTS: dict[str, int] = {
    "A": 9, "B": 2, "C": 2, "D": 4, "E": 12, "F": 2, "G": 3, "H": 2, "I": 9, "J": 1, "K": 1, "L": 4, "M": 2,
    "N": 6, "O": 8, "P": 2, "Q": 1, "R": 6, "S": 4, "T": 6, "U": 4, "V": 2, "W": 2, "X": 1, "Y": 2, "Z": 1,
    BLANK: 2,
}  # fmt: skip


def _mirror(quadrant: Iterable[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    """Reflect top-left-quadrant coordinates across both board axes."""
    last = BOARD_SIZE - 1
    return frozenset(
        square for r, c in quadrant for square in ((r, c), (r, last - c), (last - r, c), (last - r, last - c))
    )


# The standard layout is symmetric about both axes, so each premium type is listed for the
# top-left quadrant (rows and columns 0-7) only.
TRIPLE_WORD = _mirror([(0, 0), (0, 7), (7, 0)])
DOUBLE_WORD = _mirror([(1, 1), (2, 2), (3, 3), (4, 4), (7, 7)])
TRIPLE_LETTER = _mirror([(1, 5), (5, 1), (5, 5)])
DOUBLE_LETTER = _mirror([(0, 3), (2, 6), (3, 0), (3, 7), (6, 2), (6, 6), (7, 3)])

LETTER_MULTIPLIER: tuple[tuple[int, ...], ...] = tuple(
    tuple(3 if (r, c) in TRIPLE_LETTER else 2 if (r, c) in DOUBLE_LETTER else 1 for c in range(BOARD_SIZE))
    for r in range(BOARD_SIZE)
)
WORD_MULTIPLIER: tuple[tuple[int, ...], ...] = tuple(
    tuple(3 if (r, c) in TRIPLE_WORD else 2 if (r, c) in DOUBLE_WORD else 1 for c in range(BOARD_SIZE))
    for r in range(BOARD_SIZE)
)


def tile_value(tile: str) -> int:
    """Points for a tile on the board: lowercase marks a blank, which scores 0."""
    return 0 if tile.islower() else LETTER_VALUES[tile]


def square_name(row: int, col: int) -> str:
    """Human-facing square name, for example ``H8`` for the center."""
    return f"{COLUMNS[col]}{row + 1}"


def empty_board() -> list[str]:
    return [EMPTY * BOARD_SIZE for _ in range(BOARD_SIZE)]


def is_board_empty(board: list[str]) -> bool:
    return all(set(row) == {EMPTY} for row in board)


def validate_board(board: object) -> list[str]:
    """Return ``board`` if it is a well-formed 15x15 board, else raise ``ValueError``."""
    if not isinstance(board, list) or len(board) != BOARD_SIZE:
        raise ValueError(f"board must be a list of {BOARD_SIZE} strings")
    for i, row in enumerate(board):
        if not isinstance(row, str) or len(row) != BOARD_SIZE:
            raise ValueError(f"board row {i + 1} must be a string of {BOARD_SIZE} characters")
        if any(ch != EMPTY and not ("A" <= ch.upper() <= "Z" and ch.isascii()) for ch in row):
            raise ValueError(f"board row {i + 1} has a character other than '.', A-Z or a-z")
    return board


def validate_rack(rack: object) -> str:
    """Return ``rack`` if it holds 1-7 tiles of ``A``-``Z`` or ``?``, else raise ``ValueError``."""
    if not isinstance(rack, str) or not 1 <= len(rack) <= RACK_SIZE:
        raise ValueError(f"rack must be a string of 1-{RACK_SIZE} tiles")
    if any(ch not in LETTER_VALUES and ch != BLANK for ch in rack):
        raise ValueError("rack tiles must be A-Z or '?' (blank)")
    return rack


def _step(direction: Direction) -> tuple[int, int]:
    return (0, 1) if direction == "across" else (1, 0)


def _run_through(board: list[str], row: int, col: int, dr: int, dc: int, placed: dict) -> tuple[int, int, int, int]:
    """Return the start and end squares of the contiguous run through (row, col) along (dr, dc)."""

    def occupied(r: int, c: int) -> bool:
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and (board[r][c] != EMPTY or (r, c) in placed)

    sr, sc = row, col
    while occupied(sr - dr, sc - dc):
        sr, sc = sr - dr, sc - dc
    er, ec = row, col
    while occupied(er + dr, ec + dc):
        er, ec = er + dr, ec + dc
    return sr, sc, er, ec


def _score_run(board: list[str], start: tuple[int, int], end: tuple[int, int], placed: dict) -> tuple[str, int]:
    """Spell and score one word; premiums apply only under newly placed tiles."""
    (sr, sc), (er, ec) = start, end
    dr, dc = (1, 0) if sc == ec and sr != er else (0, 1)
    letters, total, multiplier = [], 0, 1
    r, c = sr, sc
    while True:
        if (r, c) in placed:
            tile = placed[(r, c)]
            total += tile_value(tile) * LETTER_MULTIPLIER[r][c]
            multiplier *= WORD_MULTIPLIER[r][c]
        else:
            tile = board[r][c]
            total += tile_value(tile)
        letters.append(tile.upper())
        if (r, c) == (er, ec):
            break
        r, c = r + dr, c + dc
    return "".join(letters), total * multiplier


@dataclass(frozen=True)
class MoveResult:
    """Outcome of checking a proposed move. ``error`` is set exactly when the move is illegal."""

    valid: bool
    score: int = 0
    words: tuple[str, ...] = ()
    invalid_words: tuple[str, ...] = ()
    placed: tuple[tuple[int, int, str], ...] = ()
    error: str | None = None


def _assign_tiles(
    rack: str, needed: list[tuple[int, int, str]], weights: dict[tuple[int, int], int]
) -> dict[tuple[int, int], str] | None:
    """Choose rack tiles for the empty squares a move fills, or return None if the rack cannot cover them.

    A score is linear in tile values, so using real tiles on the squares that count most (the highest
    ``weights``) and blanks on the rest gives the highest score for this placement.
    """
    available = Counter(rack)
    blanks = available.pop(BLANK, 0)
    by_letter: dict[str, list[tuple[int, int]]] = {}
    for r, c, letter in needed:
        by_letter.setdefault(letter, []).append((r, c))
    placed: dict[tuple[int, int], str] = {}
    for letter, squares in by_letter.items():
        squares.sort(key=lambda sq: weights[sq], reverse=True)
        real = min(available[letter], len(squares))
        for i, sq in enumerate(squares):
            placed[sq] = letter if i < real else letter.lower()
        blanks -= len(squares) - real
    return placed if blanks >= 0 else None


def evaluate_move(
    board: list[str], rack: str, word: str, row: int, col: int, direction: str, dictionary: frozenset[str]
) -> MoveResult:
    """Check a move given as the full word it spells (including tiles already on the board) and score it.

    ``row`` and ``col`` are the 0-based first square of the word. Letter case in ``word`` is ignored: the
    move uses real rack tiles where it can and blanks for the rest, assigned for the highest score.
    """
    if direction not in DIRECTIONS:
        return MoveResult(False, error=f"direction must be 'across' or 'down', got {direction!r}")
    word = word.strip().upper()
    if len(word) < 2 or not word.isascii() or not word.isalpha():
        return MoveResult(False, error="word must be at least 2 letters A-Z")
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        return MoveResult(False, error=f"start square must be within A1-O15, got row {row + 1}, column {col + 1}")
    dr, dc = _step(direction)
    end_r, end_c = row + dr * (len(word) - 1), col + dc * (len(word) - 1)
    if end_r >= BOARD_SIZE or end_c >= BOARD_SIZE:
        return MoveResult(False, error=f"'{word}' runs off the board from {square_name(row, col)} going {direction}")

    def occupied(r: int, c: int) -> bool:
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] != EMPTY

    needed: list[tuple[int, int, str]] = []
    for i, letter in enumerate(word):
        r, c = row + dr * i, col + dc * i
        existing = board[r][c]
        if existing == EMPTY:
            needed.append((r, c, letter))
        elif existing.upper() != letter:
            return MoveResult(
                False,
                error=f"square {square_name(r, c)} already holds {existing.upper()}, but '{word}' needs {letter} there",
            )
    if occupied(row - dr, col - dc) or occupied(end_r + dr, end_c + dc):
        # Spell the whole line the move would make, so the model sees what it actually formed.
        shadow = {(r, c): letter for r, c, letter in needed}
        sr, sc, er, ec = _run_through(board, row, col, dr, dc, shadow)
        line, _ = _score_run(board, (sr, sc), (er, ec), shadow)
        return MoveResult(
            False,
            error=f"tiles already on the board touch the ends of '{word}', so that line would read '{line}' "
            f"from {square_name(sr, sc)}. Give the complete word along the line, including letters already "
            "played before or after yours, or play somewhere else.",
        )

    if not needed:
        return MoveResult(False, error="a move must place at least one tile from your rack")

    first_move = is_board_empty(board)
    if first_move and not any((r, c) == CENTER for r, c, _ in needed):
        return MoveResult(False, error="the first move must cover the center square H8")
    if not first_move and len(needed) == len(word):
        connected = any(occupied(r + a, c + b) for r, c, _ in needed for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        if not connected:
            return MoveResult(False, error="the move must connect to tiles already on the board")

    # Weight of each square = how many times its tile's value is counted, after multipliers.
    main_multiplier = 1
    for r, c, _ in needed:
        main_multiplier *= WORD_MULTIPLIER[r][c]
    shadow = {(r, c): letter for r, c, letter in needed}
    cross_runs = {}
    weights = {}
    for r, c, _ in needed:
        sr, sc, er, ec = _run_through(board, r, c, dc, dr, shadow)
        cross = (sr, sc) != (er, ec)
        if cross:
            cross_runs[(r, c)] = ((sr, sc), (er, ec))
        weights[(r, c)] = LETTER_MULTIPLIER[r][c] * (main_multiplier + (WORD_MULTIPLIER[r][c] if cross else 0))

    placed = _assign_tiles(rack, needed, weights)
    if placed is None:
        rack_counts = Counter(rack)
        short = Counter(letter for _, _, letter in needed) - rack_counts
        return MoveResult(
            False, error=f"your rack {rack!r} lacks tiles for this move (missing {''.join(sorted(short.elements()))})"
        )

    main_word, score = _score_run(board, (row, col), (end_r, end_c), placed)
    words = [main_word]
    for start, end in cross_runs.values():
        cross_word, cross_score = _score_run(board, start, end, placed)
        words.append(cross_word)
        score += cross_score
    if len(placed) == RACK_SIZE:
        score += BINGO_BONUS

    placed_tiles = tuple((r, c, tile) for (r, c), tile in placed.items())
    invalid = tuple(w for w in words if w not in dictionary)
    if invalid:
        return MoveResult(
            False,
            words=tuple(words),
            invalid_words=invalid,
            placed=placed_tiles,
            error=f"not in the dictionary: {', '.join(invalid)}",
        )
    return MoveResult(True, score=score, words=tuple(words), placed=placed_tiles)


def apply_move(board: list[str], placed: Iterable[tuple[int, int, str]]) -> list[str]:
    """Return a new board with ``placed`` tiles added."""
    rows = [list(row) for row in board]
    for r, c, tile in placed:
        rows[r][c] = tile
    return ["".join(row) for row in rows]


def remove_from_rack(rack: str, placed: Iterable[tuple[int, int, str]]) -> str:
    remaining = Counter(rack)
    for _, _, tile in placed:
        remaining[BLANK if tile.islower() else tile] -= 1
    return "".join(sorted(remaining.elements()))


# --- Exhaustive move generation (Appel & Jacobson, 1988) ------------------------------------------------------
#
# Used offline to find each puzzle's best score. The trie is nested dicts with a "$" key on word ends.

_END = "$"


def build_trie(words: Iterable[str]) -> dict:
    root: dict = {}
    for word in words:
        node = root
        for letter in word:
            node = node.setdefault(letter, {})
        node[_END] = True
    return root


@dataclass(frozen=True)
class GeneratedMove:
    word: str  # Full word, blanks lowercase
    row: int
    col: int
    direction: Direction
    score: int
    placed: tuple[tuple[int, int, str], ...]

    def to_json(self) -> dict:
        return {
            "word": self.word,
            "row": self.row + 1,
            "col": COLUMNS[self.col],
            "direction": self.direction,
            "score": self.score,
        }


def _transpose(board: list[str]) -> list[str]:
    return ["".join(board[r][c] for r in range(BOARD_SIZE)) for c in range(BOARD_SIZE)]


def _generate_across(grid: list[str], rack: str, trie: dict, dictionary: frozenset[str], transposed: bool):
    """Yield across moves on ``grid``. When ``transposed``, ``grid`` is the transposed board and premiums swap."""

    def lm(r: int, c: int) -> int:
        return LETTER_MULTIPLIER[c][r] if transposed else LETTER_MULTIPLIER[r][c]

    def wm(r: int, c: int) -> int:
        return WORD_MULTIPLIER[c][r] if transposed else WORD_MULTIPLIER[r][c]

    empty_board_ = is_board_empty(grid)
    rack_counts = Counter(rack)

    # Cross-checks: letters allowed on each empty square by the vertical word they would form, and the
    # points of that word's existing tiles (-1 when no vertical word forms).
    allowed: list[list[frozenset[str] | None]] = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    cross_points = [[-1] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    anchors = [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if grid[r][c] != EMPTY:
                continue
            up, rr = [], r - 1
            while rr >= 0 and grid[rr][c] != EMPTY:
                up.append(grid[rr][c])
                rr -= 1
            down, rr = [], r + 1
            while rr < BOARD_SIZE and grid[rr][c] != EMPTY:
                down.append(grid[rr][c])
                rr += 1
            up.reverse()
            if up or down:
                prefix, suffix = "".join(up).upper(), "".join(down).upper()
                allowed[r][c] = frozenset(L for L in LETTER_VALUES if prefix + L + suffix in dictionary)
                cross_points[r][c] = sum(tile_value(t) for t in up + down)
            horizontal_neighbor = (c > 0 and grid[r][c - 1] != EMPTY) or (
                c < BOARD_SIZE - 1 and grid[r][c + 1] != EMPTY
            )
            anchors[r][c] = bool(up or down or horizontal_neighbor)
    if empty_board_:
        anchors[CENTER[0]][CENTER[1]] = True

    for r in range(BOARD_SIZE):
        line = grid[r]
        for anchor in range(BOARD_SIZE):
            if not anchors[r][anchor]:
                continue
            placed: list[tuple[int, int, str]] = []

            def extend_right(node: dict, c: int, start: int, main: int, mult: int, cross: int) -> Iterator:
                if c >= BOARD_SIZE or line[c] == EMPTY:
                    if c > anchor and _END in node and placed:
                        total = main * mult + cross + (BINGO_BONUS if len(placed) == RACK_SIZE else 0)
                        yield start, total, tuple(placed)
                if c >= BOARD_SIZE:
                    return
                if line[c] != EMPTY:
                    letter = line[c].upper()
                    if letter in node:
                        yield from extend_right(node[letter], c + 1, start, main + tile_value(line[c]), mult, cross)
                    return
                yield from place_tile(node, c, start, main, mult, cross, extend_right)

            def place_tile(node: dict, c: int, start: int, main: int, mult: int, cross: int, then) -> Iterator:
                checks = allowed[r][c]
                for letter, child in node.items():
                    if letter == _END or (checks is not None and letter not in checks):
                        continue
                    for tile in (letter, letter.lower()):
                        source = letter if tile == letter else BLANK
                        if rack_counts[source] <= 0:
                            continue
                        value = tile_value(tile) * lm(r, c)
                        new_cross = cross
                        if cross_points[r][c] >= 0:
                            new_cross += (cross_points[r][c] + value) * wm(r, c)
                        rack_counts[source] -= 1
                        placed.append((r, c, tile))
                        yield from then(child, c + 1, start, main + value, mult * wm(r, c), new_cross)
                        placed.pop()
                        rack_counts[source] += 1

            if anchor > 0 and line[anchor - 1] != EMPTY:
                # The left part is the fixed run of tiles already on the board.
                start = anchor
                while start > 0 and line[start - 1] != EMPTY:
                    start -= 1
                node, main = trie, 0
                for c in range(start, anchor):
                    node = node.get(line[c].upper())
                    if node is None:
                        break
                    main += tile_value(line[c])
                if node is not None:
                    yield from extend_right(node, anchor, start, main, 1, 0)
                continue

            limit = 0
            while anchor - limit - 1 >= 0 and line[anchor - limit - 1] == EMPTY and not anchors[r][anchor - limit - 1]:
                limit += 1

            # Build left parts of length k on squares anchor-k .. anchor-1 (none of which are anchors), then
            # extend right from the anchor.
            def left_part(node: dict, k: int, length: int, main: int, mult: int, cross: int) -> Iterator:
                if k == length:
                    yield from extend_right(node, anchor, anchor - length, main, mult, cross)
                    return
                c = anchor - length + k

                def next_left(child: dict, _c: int, _s: int, m: int, w: int, x: int) -> Iterator:
                    yield from left_part(child, k + 1, length, m, w, x)

                yield from place_tile(node, c, anchor - length, main, mult, cross, next_left)

            for length in range(min(limit, len(rack) - 1) + 1):
                yield from left_part(trie, 0, length, 0, 1, 0)


def generate_moves(board: list[str], rack: str, trie: dict, dictionary: frozenset[str]) -> list[GeneratedMove]:
    """Return every legal move for ``rack`` on ``board``, in both directions.

    A one-tile move that forms words both ways appears once per direction, with the same score.
    """
    moves = []
    for direction in DIRECTIONS:
        grid = board if direction == "across" else _transpose(board)
        for start, score, placed in _generate_across(grid, rack, trie, dictionary, direction == "down"):
            if direction == "down":
                placed = tuple((c, r, t) for r, c, t in placed)
            first_r, first_c = placed[0][0], placed[0][1]
            row, col = (first_r, start) if direction == "across" else (start, first_c)
            dr, dc = _step(direction)
            final = apply_move(board, placed)
            letters = []
            r, c = row, col
            while r < BOARD_SIZE and c < BOARD_SIZE and final[r][c] != EMPTY:
                letters.append(final[r][c])
                r, c = r + dr, c + dc
            moves.append(GeneratedMove("".join(letters), row, col, direction, score, placed))
    return moves


# --- Tile bag -------------------------------------------------------------------------------------------------


def full_bag() -> list[str]:
    return [tile for tile, count in TILE_COUNTS.items() for _ in range(count)]


def draw(bag: list[str], rack: str, rng: random.Random) -> str:
    """Refill ``rack`` to 7 tiles from ``bag`` in place; return the sorted rack."""
    tiles = list(rack)
    while len(tiles) < RACK_SIZE and bag:
        tiles.append(bag.pop(rng.randrange(len(bag))))
    return "".join(sorted(tiles))


# --- Rendering ------------------------------------------------------------------------------------------------


def _premium_label(r: int, c: int) -> str:
    if (r, c) == CENTER:
        return " * "
    if (r, c) in TRIPLE_WORD:
        return "3W "
    if (r, c) in DOUBLE_WORD:
        return "2W "
    if (r, c) in TRIPLE_LETTER:
        return "3L "
    if (r, c) in DOUBLE_LETTER:
        return "2L "
    return " . "


def render_board(board: list[str]) -> str:
    """Render the board as a fixed-width grid with premium squares labeled."""
    lines = ["    " + "".join(f" {letter} " for letter in COLUMNS)]
    for r in range(BOARD_SIZE):
        cells = "".join(
            _premium_label(r, c) if board[r][c] == EMPTY else f" {board[r][c]} " for c in range(BOARD_SIZE)
        )
        lines.append(f"{r + 1:>2}  {cells}")
    return "\n".join(lines)


def render_rack(rack: str) -> str:
    return "  ".join(f"{tile}({0 if tile == BLANK else LETTER_VALUES[tile]})" for tile in rack)
