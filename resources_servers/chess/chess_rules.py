# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standard chess rules: FEN, UCI moves, legal move generation, and game outcomes.

Written here rather than depending on python-chess, which is GPL-3.0. Squares are 0 (a1) to 63 (h8), pieces are
FEN letters, and threefold repetition and the fifty-move rule end the game without a claim.
"""

import re
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from typing import Iterator, Literal, Optional, Sequence


Color = Literal["white", "black"]
Termination = Literal["checkmate", "stalemate", "insufficient_material", "fifty_moves", "threefold_repetition"]

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
FILES = "abcdefgh"
PIECE_NAMES = {"p": "pawn", "n": "knight", "b": "bishop", "r": "rook", "q": "queen", "k": "king"}
PROMOTION_PIECES = "qrbn"

_UCI_PATTERN = re.compile(r"([a-h][1-8])([a-h][1-8])([qrbn]?)")
_CASTLING_ORDER = "KQkq"


class InvalidFenError(ValueError):
    """A FEN string is malformed or describes a position that cannot arise in a game."""


class IllegalMoveError(ValueError):
    """A move is malformed or not legal in the current position; the message says which and why."""


def square_name(square: int) -> str:
    return FILES[square % 8] + str(square // 8 + 1)


def parse_square(name: str) -> int:
    return FILES.index(name[0]) + 8 * (int(name[1]) - 1)


def opposite(color: Color) -> Color:
    return "black" if color == "white" else "white"


def piece_color(piece: str) -> Color:
    return "white" if piece.isupper() else "black"


def _squares_at(offsets: tuple[tuple[int, int], ...]) -> tuple[tuple[int, ...], ...]:
    """For every square, the squares one jump away by each (file, rank) offset that stays on the board."""
    table = []
    for square in range(64):
        file, rank = square % 8, square // 8
        table.append(
            tuple((rank + dr) * 8 + file + df for df, dr in offsets if 0 <= file + df < 8 and 0 <= rank + dr < 8)
        )
    return tuple(table)


def _rays(directions: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """For every square, the squares along each direction, nearest first, up to the board edge."""
    table = []
    for square in range(64):
        rays = []
        for df, dr in directions:
            file, rank = square % 8 + df, square // 8 + dr
            ray = []
            while 0 <= file < 8 and 0 <= rank < 8:
                ray.append(rank * 8 + file)
                file, rank = file + df, rank + dr
            if ray:
                rays.append(tuple(ray))
        table.append(tuple(rays))
    return tuple(table)


_KNIGHT_SQUARES = _squares_at(((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)))
_KING_SQUARES = _squares_at(((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)))
_ROOK_RAYS = _rays(((1, 0), (-1, 0), (0, 1), (0, -1)))
_BISHOP_RAYS = _rays(((1, 1), (1, -1), (-1, 1), (-1, -1)))
_SLIDER_RAYS = {
    "r": _ROOK_RAYS,
    "b": _BISHOP_RAYS,
    "q": tuple(rook + bishop for rook, bishop in zip(_ROOK_RAYS, _BISHOP_RAYS)),
}


@dataclass(frozen=True)
class _Castle:
    right: str
    king_from: int
    king_to: int
    rook_from: int
    rook_to: int
    # Squares between king and rook, which must be empty.
    between: tuple[int, ...]
    # Squares the king stands on, crosses, and lands on, none of which may be attacked.
    king_path: tuple[int, ...]


def _squares(names: str) -> tuple[int, ...]:
    return tuple(parse_square(name) for name in names.split())


_CASTLES = tuple(
    _Castle(
        right,
        parse_square(king_from),
        parse_square(king_to),
        parse_square(rook_from),
        parse_square(rook_to),
        between=_squares(between),
        king_path=_squares(king_path),
    )
    for right, king_from, king_to, rook_from, rook_to, between, king_path in (
        ("K", "e1", "g1", "h1", "f1", "f1 g1", "e1 f1 g1"),
        ("Q", "e1", "c1", "a1", "d1", "b1 c1 d1", "e1 d1 c1"),
        ("k", "e8", "g8", "h8", "f8", "f8 g8", "e8 f8 g8"),
        ("q", "e8", "c8", "a8", "d8", "b8 c8 d8", "e8 d8 c8"),
    )
)
_CASTLE_BY_KING_MOVE = {(castle.king_from, castle.king_to): castle for castle in _CASTLES}
# A move from or to one of these squares removes these castling rights.
_RIGHTS_LOST_ON_SQUARE = {
    parse_square(name): rights
    for name, rights in (("e1", "KQ"), ("h1", "K"), ("a1", "Q"), ("e8", "kq"), ("h8", "k"), ("a8", "q"))
}


@dataclass(frozen=True)
class Move:
    from_square: int
    to_square: int
    # Lowercase promotion piece letter, set only when a pawn reaches the last rank.
    promotion: Optional[str] = None

    def uci(self) -> str:
        return square_name(self.from_square) + square_name(self.to_square) + (self.promotion or "")


@dataclass(frozen=True)
class Outcome:
    termination: Termination
    # None for a draw.
    winner: Optional[Color]


@dataclass(frozen=True)
class Position:
    """Everything a FEN records: the board, side to move, castling rights, en passant square, and move counters.

    Positions are immutable; ``apply`` returns the position after a move.
    """

    board: tuple[Optional[str], ...]
    turn: Color
    castling_rights: str
    en_passant: Optional[int]
    halfmove_clock: int
    fullmove_number: int

    @classmethod
    def from_fen(cls, fen: str) -> "Position":
        fields = fen.split()
        if len(fields) != 6:
            raise InvalidFenError(f"a FEN has 6 space-separated fields, got {len(fields)}: {fen!r}")
        placement, turn, castling, en_passant, halfmove, fullmove = fields

        ranks = placement.split("/")
        if len(ranks) != 8:
            raise InvalidFenError(f"the piece placement must describe 8 ranks, got {len(ranks)}")
        board: list[Optional[str]] = [None] * 64
        for index, rank_text in enumerate(ranks):
            rank, file = 7 - index, 0
            for char in rank_text:
                if char in "12345678":
                    file += int(char)
                    continue
                if char.lower() not in PIECE_NAMES:
                    raise InvalidFenError(f"unknown character {char!r} in rank {rank + 1}")
                if file < 8:
                    board[rank * 8 + file] = char
                file += 1
            if file != 8:
                raise InvalidFenError(f"rank {rank + 1} describes {file} squares, expected 8")

        if turn not in ("w", "b"):
            raise InvalidFenError(f"the side to move must be 'w' or 'b', got {turn!r}")
        if castling != "-" and (set(castling) - set(_CASTLING_ORDER) or len(set(castling)) != len(castling)):
            raise InvalidFenError(f"castling rights must be '-' or distinct letters from KQkq, got {castling!r}")
        if en_passant != "-" and not re.fullmatch(r"[a-h][36]", en_passant):
            raise InvalidFenError(f"the en passant square must be '-' or a square on rank 3 or 6, got {en_passant!r}")
        if not (halfmove.isdigit() and fullmove.isdigit() and int(fullmove) >= 1):
            raise InvalidFenError(
                f"the move counters must be a halfmove clock >= 0 and a fullmove number >= 1, got {halfmove!r} and "
                f"{fullmove!r}"
            )

        position = cls(
            board=tuple(board),
            turn="white" if turn == "w" else "black",
            castling_rights="".join(right for right in _CASTLING_ORDER if right in castling),
            en_passant=None if en_passant == "-" else parse_square(en_passant),
            halfmove_clock=int(halfmove),
            fullmove_number=int(fullmove),
        )
        position._check_consistency()
        return position

    def _check_consistency(self) -> None:
        for king in ("K", "k"):
            count = self.board.count(king)
            if count != 1:
                raise InvalidFenError(f"{piece_color(king)} must have exactly one king, found {count}")
        for square in (*range(8), *range(56, 64)):
            if self.board[square] in ("P", "p"):
                raise InvalidFenError(f"a pawn cannot stand on the first or last rank ({square_name(square)})")
        for castle in _CASTLES:
            if castle.right in self.castling_rights:
                king, rook = ("K", "R") if castle.right.isupper() else ("k", "r")
                if self.board[castle.king_from] != king or self.board[castle.rook_from] != rook:
                    raise InvalidFenError(
                        f"castling right {castle.right} needs the king on {square_name(castle.king_from)} and the "
                        f"rook on {square_name(castle.rook_from)}"
                    )
        if self.en_passant is not None:
            forward = -8 if self.turn == "white" else 8
            expected_rank = 5 if self.turn == "white" else 2
            pawn = "p" if self.turn == "white" else "P"
            if (
                self.en_passant // 8 != expected_rank
                or self.board[self.en_passant] is not None
                or self.board[self.en_passant + forward] != pawn
                or self.board[self.en_passant - forward] is not None
            ):
                raise InvalidFenError(
                    f"en passant square {square_name(self.en_passant)} does not follow a two-square pawn advance"
                )
        if self.is_attacked(self.king_square(opposite(self.turn)), by=self.turn):
            raise InvalidFenError(f"{opposite(self.turn)} is in check but it is {self.turn} to move")

    def fen(self) -> str:
        ranks = []
        for rank in range(7, -1, -1):
            text, empty = "", 0
            for piece in self.board[rank * 8 : rank * 8 + 8]:
                if piece is None:
                    empty += 1
                    continue
                text += (str(empty) if empty else "") + piece
                empty = 0
            ranks.append(text + (str(empty) if empty else ""))
        return " ".join(
            (
                "/".join(ranks),
                self.turn[0],
                self.castling_rights or "-",
                square_name(self.en_passant) if self.en_passant is not None else "-",
                str(self.halfmove_clock),
                str(self.fullmove_number),
            )
        )

    def board_rows(self) -> list[str]:
        """The board as text, White at the bottom: one line per rank, then the file letters."""
        rows = [
            f"{rank + 1} " + " ".join(piece or "." for piece in self.board[rank * 8 : rank * 8 + 8])
            for rank in range(7, -1, -1)
        ]
        return rows + ["  " + " ".join(FILES)]

    def king_square(self, color: Color) -> int:
        return self.board.index("K" if color == "white" else "k")

    def is_attacked(self, square: int, *, by: Color) -> bool:
        """Whether a piece of color ``by`` attacks ``square`` (whether or not that piece could legally move there)."""
        board = self.board
        white = by == "white"
        pawn, knight, king = ("P", "N", "K") if white else ("p", "n", "k")
        # A pawn attacks diagonally forward, so look one rank behind the square from the attacker's side.
        pawn_rank = square // 8 - 1 if white else square // 8 + 1
        if 0 <= pawn_rank < 8:
            file = square % 8
            if any(0 <= file + df < 8 and board[pawn_rank * 8 + file + df] == pawn for df in (-1, 1)):
                return True
        if any(board[target] == knight for target in _KNIGHT_SQUARES[square]):
            return True
        if any(board[target] == king for target in _KING_SQUARES[square]):
            return True
        for rays, white_attackers in ((_ROOK_RAYS, "RQ"), (_BISHOP_RAYS, "BQ")):
            attackers = white_attackers if white else white_attackers.lower()
            for ray in rays[square]:
                for target in ray:
                    occupant = board[target]
                    if occupant is not None:
                        if occupant in attackers:
                            return True
                        break
        return False

    def in_check(self) -> bool:
        return self.is_attacked(self.king_square(self.turn), by=opposite(self.turn))

    def captured_piece(self, move: Move) -> Optional[str]:
        """The piece ``move`` captures, including a pawn taken en passant."""
        if move.to_square == self.en_passant and self.board[move.from_square] in ("P", "p"):
            return "p" if self.turn == "white" else "P"
        return self.board[move.to_square]

    def apply(self, move: Move) -> "Position":
        """The position after ``move``, which must be one of this position's legal moves."""
        board = list(self.board)
        piece = board[move.from_square]
        assert piece is not None, "apply() takes a legal move"
        kind = piece.lower()
        captured = self.captured_piece(move)
        board[move.from_square] = None
        if kind == "p" and move.to_square == self.en_passant:
            board[move.to_square + (-8 if self.turn == "white" else 8)] = None
        if move.promotion is not None:
            piece = move.promotion.upper() if self.turn == "white" else move.promotion
        board[move.to_square] = piece
        if kind == "k" and abs(move.to_square - move.from_square) == 2:
            castle = _CASTLE_BY_KING_MOVE[move.from_square, move.to_square]
            board[castle.rook_to], board[castle.rook_from] = board[castle.rook_from], None

        lost = _RIGHTS_LOST_ON_SQUARE.get(move.from_square, "") + _RIGHTS_LOST_ON_SQUARE.get(move.to_square, "")
        double_push = kind == "p" and abs(move.to_square - move.from_square) == 16
        return Position(
            board=tuple(board),
            turn=opposite(self.turn),
            castling_rights="".join(right for right in self.castling_rights if right not in lost),
            en_passant=(move.from_square + move.to_square) // 2 if double_push else None,
            halfmove_clock=0 if kind == "p" or captured is not None else self.halfmove_clock + 1,
            fullmove_number=self.fullmove_number + (1 if self.turn == "black" else 0),
        )

    @cached_property
    def legal_moves(self) -> tuple[Move, ...]:
        return tuple(
            move
            for move in self._pseudo_legal_moves()
            if not (after := self.apply(move)).is_attacked(after.king_square(self.turn), by=after.turn)
        )

    def _pseudo_legal_moves(self) -> Iterator[Move]:
        """Moves that follow the pieces' movement rules but may leave the king in check (castling excepted)."""
        board = self.board
        for square, piece in enumerate(board):
            if piece is None or piece_color(piece) != self.turn:
                continue
            kind = piece.lower()
            if kind == "p":
                yield from self._pawn_moves(square)
                continue
            targets: Sequence[int]
            if kind in _SLIDER_RAYS:
                targets = []
                for ray in _SLIDER_RAYS[kind][square]:
                    for target in ray:
                        targets.append(target)
                        if board[target] is not None:
                            break
            else:
                targets = _KNIGHT_SQUARES[square] if kind == "n" else _KING_SQUARES[square]
            for target in targets:
                occupant = board[target]
                if occupant is None or piece_color(occupant) != self.turn:
                    yield Move(square, target)
            if kind == "k":
                yield from self._castling_moves()

    def _pawn_moves(self, square: int) -> Iterator[Move]:
        board = self.board
        white = self.turn == "white"
        forward = 8 if white else -8
        targets = []
        one_step = square + forward
        if board[one_step] is None:
            targets.append(one_step)
            start_rank = 1 if white else 6
            if square // 8 == start_rank and board[one_step + forward] is None:
                targets.append(one_step + forward)
        file = square % 8
        for df in (-1, 1):
            if 0 <= file + df < 8:
                target = one_step + df
                victim = board[target]
                if (victim is not None and piece_color(victim) != self.turn) or target == self.en_passant:
                    targets.append(target)
        last_rank = 7 if white else 0
        for target in targets:
            if target // 8 == last_rank:
                yield from (Move(square, target, promotion) for promotion in PROMOTION_PIECES)
            else:
                yield Move(square, target)

    def _castling_moves(self) -> Iterator[Move]:
        for castle in _CASTLES:
            if castle.right in self.castling_rights and self._castling_obstacle(castle) is None:
                yield Move(castle.king_from, castle.king_to)

    def _castling_obstacle(self, castle: _Castle) -> Optional[str]:
        """Why ``castle`` is not legal now, or None when it is."""
        if castle.right not in self.castling_rights:
            return "the king or that rook has already moved"
        if any(self.board[square] is not None for square in castle.between):
            return "there are pieces between the king and the rook"
        if self.in_check():
            return "the king is in check"
        if any(self.is_attacked(square, by=opposite(self.turn)) for square in castle.king_path):
            return "the king would pass through or land on an attacked square"
        return None

    def parse_move(self, text: str) -> Move:
        """The legal move ``text`` names in UCI notation; raises IllegalMoveError saying why when there is none."""
        match = _UCI_PATTERN.fullmatch(text.strip().lower())
        if match is None:
            raise IllegalMoveError(
                f"{text!r} is not a UCI move: write the from-square and the to-square, such as e2e4, and add the "
                "promotion piece (q, r, b or n) when a pawn promotes, such as e7e8q"
            )
        move = Move(parse_square(match[1]), parse_square(match[2]), match[3] or None)
        if move in self.legal_moves:
            return move
        raise IllegalMoveError(f"{move.uci()} is illegal: {self._illegal_reason(move)}")

    def _illegal_reason(self, move: Move) -> str:
        origin, target = square_name(move.from_square), square_name(move.to_square)
        piece = self.board[move.from_square]
        if piece is None:
            return f"there is no piece on {origin}"
        if piece_color(piece) != self.turn:
            return f"the piece on {origin} is {piece_color(piece)} and it is {self.turn} to move"
        if move.promotion is None and Move(move.from_square, move.to_square, "q") in self.legal_moves:
            return f"a pawn reaching the last rank must promote; add q, r, b or n, such as {move.uci()}q"
        if move.promotion is not None and Move(move.from_square, move.to_square) in self.legal_moves:
            return f"only a pawn reaching the last rank can promote; to make this move, play {move.uci()[:4]}"
        castle = _CASTLE_BY_KING_MOVE.get((move.from_square, move.to_square))
        if castle is not None and piece == ("K" if castle.right.isupper() else "k"):
            return f"cannot castle because {self._castling_obstacle(castle)}"
        name = PIECE_NAMES[piece.lower()]
        if move in self._pseudo_legal_moves():
            return f"moving the {name} from {origin} to {target} would leave your king in check"
        return f"the {name} on {origin} cannot move to {target}" + (
            " (your king is in check)" if self.in_check() else ""
        )

    def repetition_key(self) -> tuple:
        """What identifies a position for the repetition rule; the en passant square counts only if it can be used."""
        en_passant = self.en_passant
        if en_passant is not None and not any(
            move.to_square == en_passant and self.board[move.from_square] in ("P", "p") for move in self.legal_moves
        ):
            en_passant = None
        return self.board, self.turn, self.castling_rights, en_passant

    def has_insufficient_material(self) -> bool:
        """Whether neither side can ever checkmate: bare kings, one minor piece, or only same-colored bishops."""
        others = [(square, piece.lower()) for square, piece in enumerate(self.board) if piece and piece not in "Kk"]
        if len(others) <= 1:
            return not any(kind in "prq" for _, kind in others)
        if all(kind == "b" for _, kind in others):
            return len({(square % 8 + square // 8) % 2 for square, _ in others}) == 1
        return False


class ChessGame:
    """A game in progress: the current position plus the history the draw rules need."""

    def __init__(self, fen: str = STARTING_FEN) -> None:
        self.starting_position = Position.from_fen(fen)
        self.position = self.starting_position
        self.moves: list[Move] = []
        self._occurrences = Counter([self.position.repetition_key()])

    def play(self, text: str) -> Move:
        """Play a UCI move for the side to move; raises IllegalMoveError, changing nothing, if it cannot be played."""
        outcome = self.outcome()
        if outcome is not None:
            raise IllegalMoveError(f"the game is over ({outcome.termination})")
        move = self.position.parse_move(text)
        self.position = self.position.apply(move)
        self.moves.append(move)
        self._occurrences[self.position.repetition_key()] += 1
        return move

    def outcome(self) -> Optional[Outcome]:
        """How the game ended, or None while it continues."""
        position = self.position
        if not position.legal_moves:
            if position.in_check():
                return Outcome("checkmate", winner=opposite(position.turn))
            return Outcome("stalemate", winner=None)
        if position.has_insufficient_material():
            return Outcome("insufficient_material", winner=None)
        if position.halfmove_clock >= 100:
            return Outcome("fifty_moves", winner=None)
        if self._occurrences[position.repetition_key()] >= 3:
            return Outcome("threefold_repetition", winner=None)
        return None

    def numbered_moves(self) -> list[str]:
        """The moves played so far, numbered as in a score sheet: ``1. e2e4``, ``1... e7e5``."""
        numbered = []
        number, turn = self.starting_position.fullmove_number, self.starting_position.turn
        for move in self.moves:
            numbered.append(f"{number}{'.' if turn == 'white' else '...'} {move.uci()}")
            if turn == "black":
                number += 1
            turn = opposite(turn)
        return numbered
