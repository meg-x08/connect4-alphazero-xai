"""Connect-4 rules."""

import numpy as np

ROWS = 6
COLS = 7


class Connect4:
    """A game of Connect-4.

    The board is a 6x7 numpy array: 1 = first player, -1 = second player, 0 = empty.
    Row 0 is the bottom row, so a stone dropped into a column lands at the lowest empty row.
    """

    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = 1
        self.num_moves = 0
        self.winner = None  # set to 1, -1, or 0 (draw) when the game ends

    def copy(self):
        new = Connect4()
        new.board = self.board.copy()
        new.current_player = self.current_player
        new.num_moves = self.num_moves
        new.winner = self.winner
        return new

    def legal_moves(self):
        """Columns that are not full."""
        return [col for col in range(COLS) if self.board[ROWS - 1, col] == 0]

    def is_over(self):
        return self.winner is not None

    def play(self, col):
        if self.is_over():
            raise ValueError("the game is already over")
        if self.board[ROWS - 1, col] != 0:
            raise ValueError(f"column {col} is full")
        row = np.count_nonzero(self.board[:, col])  # stones stack up from the bottom
        self.board[row, col] = self.current_player
        self.num_moves += 1
        if self._makes_four(row, col):
            self.winner = self.current_player
        elif self.num_moves == ROWS * COLS:
            self.winner = 0
        self.current_player = -self.current_player

    def _makes_four(self, row, col):
        """Does the stone at (row, col) belong to a line of four or more?"""
        player = self.board[row, col]
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:  # horizontal, vertical, two diagonals
            count = 1
            for sign in (1, -1):  # look in both directions along the line
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 4:
                return True
        return False

    def winning_moves(self, player):
        """Columns where `player` would complete four by dropping a stone right now."""
        moves = []
        for col in self.legal_moves():
            row = np.count_nonzero(self.board[:, col])
            self.board[row, col] = player  # try the move...
            if self._makes_four(row, col):
                moves.append(col)
            self.board[row, col] = 0  # ...and undo it
        return moves

    def encode(self):
        """The board as the network sees it: two 6x7 planes, from the point of view of
        the player to move (plane 0 = my stones, plane 1 = opponent's stones)."""
        mine = self.board == self.current_player
        theirs = self.board == -self.current_player
        return np.stack([mine, theirs]).astype(np.float32)

    def __str__(self):
        symbols = {1: "X", -1: "O", 0: "."}
        lines = [" ".join(symbols[v] for v in self.board[r]) for r in range(ROWS - 1, -1, -1)]
        lines.append(" ".join(str(c + 1) for c in range(COLS)))
        return "\n".join(lines)
