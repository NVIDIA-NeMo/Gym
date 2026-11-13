# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from resources_servers.chess.chess_rules import (
    STARTING_FEN,
    ChessGame,
    IllegalMoveError,
    InvalidFenError,
    Outcome,
    Position,
    parse_square,
)


# Morphy vs. Duke of Brunswick and Count Isouard, Paris 1858: queenside castling, then mate on d8.
OPERA_GAME = (
    "e2e4 e7e5 g1f3 d7d6 d2d4 c8g4 d4e5 g4f3 d1f3 d6e5 f1c4 g8f6 f3b3 d8e7 b1c3 c7c6 c1g5 b7b5 c3b5 c6b5 "
    "c4b5 b8d7 e1c1 a8d8 d1d7 d8d7 h1d1 e7e6 b5d7 f6d7 b3b8 d7b8 d1d8"
).split()
# Sam Loyd's ten-move stalemate from the initial position.
LOYD_STALEMATE = (
    "e2e3 a7a5 d1h5 a8a6 h5a5 h7h5 h2h4 a6h6 a5c7 f7f6 c7d7 e8f7 d7b7 d8d3 b7b8 d3h7 b8c8 f7g6 c8e6".split()
)


def perft(position: Position, depth: int) -> int:
    if depth == 1:
        return len(position.legal_moves)
    return sum(perft(position.apply(move), depth - 1) for move in position.legal_moves)


def legal_ucis(fen: str) -> set[str]:
    return {move.uci() for move in Position.from_fen(fen).legal_moves}


def play_all(moves: list[str], fen: str = STARTING_FEN) -> ChessGame:
    game = ChessGame(fen)
    for move in moves:
        game.play(move)
    return game


class TestMoveGeneration:
    # Reference node counts from https://www.chessprogramming.org/Perft_Results. Together these positions cover
    # castling, en passant (including pinned en passant), promotion, and checks.
    @pytest.mark.parametrize(
        "fen, counts",
        [
            (STARTING_FEN, [20, 400, 8902]),
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", [48, 2039]),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812]),
            ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [6, 264, 9467]),
            ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486]),
            ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", [46, 2079]),
        ],
    )
    def test_perft_matches_reference_counts(self, fen, counts):
        position = Position.from_fen(fen)
        assert [perft(position, depth) for depth in range(1, len(counts) + 1)] == counts

    def test_starting_position_moves(self):
        assert legal_ucis(STARTING_FEN) == {
            *(f"{file}2{file}{rank}" for file in "abcdefgh" for rank in "34"),
            "b1a3",
            "b1c3",
            "g1f3",
            "g1h3",
        }

    def test_pieces_move_and_capture(self):
        game = play_all(["e2e4", "d7d5", "e4d5", "d8d5", "b1c3"])
        board = game.position.board
        assert board[parse_square("e4")] is None
        assert board[parse_square("d5")] == "q"
        assert board[parse_square("c3")] == "N"
        assert game.position.turn == "black"
        # The queen on d5 can now take the pawn on a2 along the open diagonal.
        assert "d5a2" in {move.uci() for move in game.position.legal_moves}
        assert game.position.halfmove_clock == 1
        assert game.position.fullmove_number == 3

    def test_turns_alternate(self):
        game = ChessGame()
        assert game.position.turn == "white"
        game.play("e2e4")
        assert game.position.turn == "black"
        with pytest.raises(IllegalMoveError, match="the piece on d2 is white and it is black to move"):
            game.play("d2d4")
        game.play("e7e5")
        assert game.position.turn == "white"


class TestCastling:
    FEN = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"

    def test_kingside_castling_moves_king_and_rook(self):
        game = play_all(["e1g1"], self.FEN)
        board = game.position.board
        assert (board[parse_square("g1")], board[parse_square("f1")]) == ("K", "R")
        assert board[parse_square("e1")] is None and board[parse_square("h1")] is None
        assert game.position.castling_rights == "kq"

    def test_queenside_castling_moves_king_and_rook(self):
        game = play_all(["e1g1", "e8c8"], self.FEN)
        board = game.position.board
        assert (board[parse_square("c8")], board[parse_square("d8")]) == ("k", "r")
        assert board[parse_square("a8")] is None and board[parse_square("e8")] is None
        assert game.position.castling_rights == ""

    def test_cannot_castle_through_an_attacked_square(self):
        # The bishop on c4 covers f1, which the king would cross.
        fen = "4k3/8/8/8/2b5/8/8/4K2R w K - 0 1"
        assert "e1g1" not in legal_ucis(fen)
        with pytest.raises(IllegalMoveError, match="cannot castle because the king would pass through"):
            ChessGame(fen).play("e1g1")

    def test_cannot_castle_out_of_check(self):
        fen = "4r1k1/8/8/8/8/8/8/R3K2R w KQ - 0 1"
        assert not {"e1g1", "e1c1"} & legal_ucis(fen)
        with pytest.raises(IllegalMoveError, match="the king is in check"):
            ChessGame(fen).play("e1c1")

    def test_queenside_castling_ignores_an_attack_on_b1(self):
        # Only b1 is attacked; the king never crosses it, so queenside castling stays legal.
        assert "e1c1" in legal_ucis("1r2k3/8/8/8/8/8/8/R3K3 w Q - 0 1")

    def test_cannot_castle_with_pieces_in_between(self):
        with pytest.raises(IllegalMoveError, match="pieces between the king and the rook"):
            ChessGame(STARTING_FEN).play("e1g1")

    def test_moving_the_king_or_a_rook_loses_the_rights(self):
        fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        assert play_all(["h1h2", "a8a7"], fen).position.castling_rights == "Qk"
        game = play_all(["e1e2", "e8d8", "e2e1", "d8e8"], fen)
        assert game.position.castling_rights == ""
        with pytest.raises(IllegalMoveError, match="already moved"):
            game.play("e1g1")

    def test_capturing_a_rook_removes_its_right(self):
        game = play_all(["g2a8"], "r3k2r/p1pppppp/8/8/8/8/PPPPPPBP/R3K2R w KQkq - 0 1")
        assert game.position.castling_rights == "KQk"


class TestEnPassant:
    def test_capture_removes_the_passed_pawn(self):
        game = play_all(["e2e4", "a7a6", "e4e5", "d7d5"])
        assert game.position.en_passant == parse_square("d6")
        game.play("e5d6")
        assert game.position.board[parse_square("d5")] is None
        assert game.position.board[parse_square("d6")] == "P"
        assert game.position.halfmove_clock == 0

    def test_only_available_immediately(self):
        game = play_all(["e2e4", "a7a6", "e4e5", "d7d5", "g1f3", "a6a5"])
        assert "e5d6" not in {move.uci() for move in game.position.legal_moves}

    def test_capture_that_exposes_the_king_along_the_rank_is_illegal(self):
        # Taking on c6 would remove both pawns from rank 5 and open the rook's line to the king on a5.
        fen = "8/8/8/KPp4r/8/8/8/7k w - c6 0 1"
        assert legal_ucis(fen) == {"a5a4", "a5a6", "a5b6", "b5b6"}
        with pytest.raises(IllegalMoveError, match="would leave your king in check"):
            ChessGame(fen).play("b5c6")

    def test_capture_that_removes_the_checking_pawn(self):
        # White's d2-d4 checks the king on e5; taking the pawn en passant is one of the ways out.
        fen = "8/8/8/4k3/3Pp3/8/8/4K3 b - d3 0 1"
        assert Position.from_fen(fen).in_check()
        assert "e4d3" in legal_ucis(fen)
        game = play_all(["e4d3"], fen)
        assert not game.position.in_check()


class TestPromotion:
    @pytest.mark.parametrize("piece", ["q", "r", "b", "n"])
    def test_each_promotion_piece(self, piece):
        game = play_all([f"a7a8{piece}"], "8/P7/8/8/8/8/k7/4K3 w - - 0 1")
        assert game.position.board[parse_square("a8")] == piece.upper()

    def test_black_promotes_by_capture(self):
        game = play_all(["b2a1n"], "4k3/8/8/8/8/8/1p6/R3K3 b Q - 0 1")
        assert game.position.board[parse_square("a1")] == "n"
        assert game.position.castling_rights == ""

    def test_promotion_piece_is_required(self):
        with pytest.raises(IllegalMoveError, match="must promote; add q, r, b or n, such as a7a8q"):
            ChessGame("8/P7/8/8/8/8/k7/4K3 w - - 0 1").play("a7a8")

    def test_promotion_piece_is_rejected_elsewhere(self):
        with pytest.raises(IllegalMoveError, match="only a pawn reaching the last rank can promote; .* play e2e4$"):
            ChessGame().play("e2e4q")


class TestOutcomes:
    def test_checkmate_ends_a_full_game(self):
        game = play_all(OPERA_GAME)
        assert game.outcome() == Outcome("checkmate", winner="white")
        assert game.position.in_check()
        assert game.position.fen() == "1n1Rkb1r/p4ppp/4q3/4p1B1/4P3/8/PPP2PPP/2K5 b k - 1 17"

    def test_fools_mate(self):
        assert play_all(["f2f3", "e7e5", "g2g4", "d8h4"]).outcome() == Outcome("checkmate", winner="black")

    def test_stalemate_ends_a_full_game(self):
        game = play_all(LOYD_STALEMATE)
        assert game.outcome() == Outcome("stalemate", winner=None)
        assert not game.position.in_check()

    def test_threefold_repetition(self):
        shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"]
        game = play_all(shuffle)
        assert game.outcome() is None
        game = play_all(shuffle * 2)
        assert game.outcome() == Outcome("threefold_repetition", winner=None)

    def test_repetition_ignores_an_en_passant_square_nobody_can_use(self):
        # After 1. e4 the FEN records e3, but no black pawn can take en passant, so the position after the knight
        # shuffles is the same one; it occurs for the third time after the second shuffle.
        game = play_all(["e2e4", "g8f6", "g1f3", "f6g8", "f3g1", "g8f6", "g1f3", "f6g8", "f3g1"])
        assert game.outcome() == Outcome("threefold_repetition", winner=None)

    def test_fifty_move_rule(self):
        game = ChessGame("4k3/8/8/8/8/8/8/R3K3 w Q - 99 80")
        assert game.outcome() is None
        game.play("a1a2")
        assert game.outcome() == Outcome("fifty_moves", winner=None)

    def test_checkmate_takes_precedence_over_the_fifty_move_rule(self):
        game = play_all(["a1a8"], "6k1/5ppp/8/8/8/8/8/R5K1 w - - 99 80")
        assert game.outcome() == Outcome("checkmate", winner="white")

    def test_pawn_move_resets_the_fifty_move_count(self):
        game = play_all(["e2e4"], "4k3/8/8/8/8/8/4P3/R3K3 w Q - 99 80")
        assert game.position.halfmove_clock == 0
        assert game.outcome() is None

    @pytest.mark.parametrize(
        "fen",
        [
            "8/8/4k3/8/8/4K3/8/8 w - - 0 1",
            "8/8/4k3/8/8/4K3/8/5B2 w - - 0 1",
            "8/8/4k3/8/8/4K3/8/5n2 w - - 0 1",
            "2b5/8/4k3/8/8/4K3/8/5B2 w - - 0 1",
        ],
    )
    def test_insufficient_material(self, fen):
        assert ChessGame(fen).outcome() == Outcome("insufficient_material", winner=None)

    @pytest.mark.parametrize(
        "fen",
        [
            "8/8/4k3/8/8/4K3/8/R7 w - - 0 1",
            "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
            "8/8/4k3/8/8/4K3/8/5N1n w - - 0 1",
            "1b6/8/4k3/8/8/4K3/8/5B2 w - - 0 1",
            "8/8/4k3/8/8/4K3/8/4NN2 w - - 0 1",
        ],
    )
    def test_sufficient_material(self, fen):
        assert ChessGame(fen).outcome() is None

    def test_capture_into_insufficient_material(self):
        game = play_all(["e1d1"], "8/8/4k3/8/8/8/8/3rK3 w - - 0 1")
        assert game.outcome() == Outcome("insufficient_material", winner=None)

    def test_no_moves_after_the_game_ends(self):
        game = play_all(OPERA_GAME)
        with pytest.raises(IllegalMoveError, match="the game is over"):
            game.play("e8d8")
        assert len(game.moves) == len(OPERA_GAME)


class TestIllegalMoves:
    @pytest.mark.parametrize("text", ["", "e2", "e2e9", "i2i4", "e2-e4", "Nf3", "e7e8k", "0000", "e2e4e5"])
    def test_malformed_moves(self, text):
        with pytest.raises(IllegalMoveError, match="is not a UCI move"):
            ChessGame().play(text)

    def test_moves_are_case_and_whitespace_insensitive(self):
        assert ChessGame().play("  E2E4 ").uci() == "e2e4"

    @pytest.mark.parametrize(
        "text, reason",
        [
            ("e3e4", "there is no piece on e3"),
            ("e7e5", "the piece on e7 is black and it is white to move"),
            ("g1g3", "the knight on g1 cannot move to g3"),
            ("e2e5", "the pawn on e2 cannot move to e5"),
            ("a1a3", "the rook on a1 cannot move to a3"),
        ],
    )
    def test_illegal_moves_explain_why(self, text, reason):
        with pytest.raises(IllegalMoveError, match=reason):
            ChessGame().play(text)

    def test_pinned_piece_cannot_leave_the_line(self):
        with pytest.raises(IllegalMoveError, match="moving the knight from e2 to c3 would leave your king in check"):
            ChessGame("4r1k1/8/8/8/8/8/4N3/4K3 w - - 0 1").play("e2c3")

    def test_illegal_move_leaves_the_game_unchanged(self):
        game = play_all(["e2e4"])
        before = game.position
        with pytest.raises(IllegalMoveError):
            game.play("e4e5")
        assert game.position == before
        assert [move.uci() for move in game.moves] == ["e2e4"]


class TestFen:
    @pytest.mark.parametrize(
        "fen",
        [
            STARTING_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "8/8/8/8/8/8/8/K6k b - - 47 112",
        ],
    )
    def test_round_trip(self, fen):
        assert Position.from_fen(fen).fen() == fen

    def test_moves_update_the_fen(self):
        game = play_all(["e2e4", "c7c5", "g1f3"])
        assert game.position.fen() == "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"

    def test_castling_rights_are_normalized(self):
        assert Position.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w qkQK - 0 1").castling_rights == "KQkq"

    @pytest.mark.parametrize(
        "fen, message",
        [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -", "6 space-separated fields"),
            ("rnbqkbnr/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "8 ranks"),
            ("rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "unknown character '9'"),
            ("rnbqkbnr/ppppxppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "unknown character 'x'"),
            ("rnbqkbnr/ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "rank 7 describes 7 squares"),
            ("rnbqkbnr/ppppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "rank 7 describes 9 squares"),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1", "side to move"),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkx - 0 1", "castling rights"),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KKq - 0 1", "castling rights"),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e4 0 1", "en passant square"),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - -1 1", "move counters"),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0", "move counters"),
            ("rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1", "black must have exactly one king, found 0"),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKKNR w kq - 0 1", "white must have exactly one king, found 2"),
            ("rnbqkbnP/pppppppp/8/8/8/8/PPPPPPP1/RNBQKBNR w KQq - 0 1", "first or last rank"),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w KQkq - 0 1", "castling right K needs"),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 1", "does not follow a two-square pawn"),
            ("4k3/8/8/8/8/8/8/4K2r b - - 0 1", "white is in check but it is black to move"),
        ],
    )
    def test_invalid_fen(self, fen, message):
        with pytest.raises(InvalidFenError, match=message):
            ChessGame(fen)

    def test_numbered_moves_follow_the_starting_move_number(self):
        game = play_all(["e7e5", "g1f3"], "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        assert game.numbered_moves() == ["1... e7e5", "2. g1f3"]
