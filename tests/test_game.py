import numpy as np
import pytest

from connect4.game import Connect4


def play_moves(moves):
    """Play a sequence of columns written 1-7, e.g. "1122"."""
    game = Connect4()
    for ch in moves:
        game.play(int(ch) - 1)
    return game


def test_stones_stack_from_the_bottom():
    game = play_moves("44")
    assert game.board[0, 3] == 1 and game.board[1, 3] == -1
    assert game.current_player == 1


def test_full_column_is_not_legal():
    game = play_moves("444444")
    assert 3 not in game.legal_moves()
    with pytest.raises(ValueError):
        game.play(3)


def test_horizontal_win():
    game = play_moves("1122334")
    assert game.winner == 1


def test_vertical_win():
    game = play_moves("1212121")
    assert game.winner == 1


def test_diagonal_win():
    # X builds a "/" diagonal from the bottom-left corner
    game = play_moves("1223343447")
    assert game.winner is None
    game.play(3)  # column 4
    assert game.winner == 1


def test_draw():
    # fill the board in a pattern where nobody gets four in a row
    game = Connect4()
    order = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 2, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2,
             4, 5, 4, 5, 4, 5, 5, 4, 5, 4, 5, 4, 6, 6, 6, 6, 6, 6]
    for col in order:
        assert game.winner is None
        game.play(col)
    assert game.winner == 0


def test_winning_moves_does_not_change_the_board():
    game = play_moves("112233")
    before = game.board.copy()
    assert game.winning_moves(1) == [3]
    assert np.array_equal(game.board, before)


def test_encode_is_from_the_movers_point_of_view():
    game = play_moves("4")
    planes = game.encode()  # O to move: plane 0 = O's stones, plane 1 = X's stones
    assert planes.shape == (2, 6, 7)
    assert planes[1, 0, 3] == 1 and planes[0].sum() == 0
