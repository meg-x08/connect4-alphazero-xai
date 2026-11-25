"""Different players, and a function to play matches between them."""

import random

import numpy as np

from .game import Connect4
from .mcts import mcts


class RandomPlayer:
    name = "random"

    def choose_move(self, game):
        return random.choice(game.legal_moves())


class GreedyPlayer:
    """Looks one move ahead: wins if it can, blocks if it must, otherwise prefers the centre."""

    name = "greedy"

    def choose_move(self, game):
        wins = game.winning_moves(game.current_player)
        if wins:
            return wins[0]
        blocks = game.winning_moves(-game.current_player)
        if blocks:
            return blocks[0]
        moves = game.legal_moves()
        weights = [4 - abs(3 - col) for col in moves]  # centre column gets the most weight
        return random.choices(moves, weights=weights)[0]


class NetworkPlayer:
    """Uses only the network's policy, with no search: the network's "intuition"."""

    name = "network only"

    def __init__(self, net):
        self.net = net

    def choose_move(self, game):
        policy, _ = self.net.predict(game)
        legal = game.legal_moves()
        return max(legal, key=lambda m: policy[m])


class AlphaZeroPlayer:
    """Network + tree search: the full agent."""

    def __init__(self, net, num_simulations=50):
        self.net = net
        self.num_simulations = num_simulations
        self.name = f"alphazero ({num_simulations} sims)"

    def choose_move(self, game):
        visits = mcts(game, self.net, self.num_simulations)
        return int(np.argmax(visits))


def play_game(first, second):
    """Play one game. Returns 1 if `first` wins, -1 if `second` wins, 0 for a draw."""
    game = Connect4()
    players = {1: first, -1: second}
    while not game.is_over():
        game.play(players[game.current_player].choose_move(game))
    return game.winner


def play_match(player, opponent, num_games=20):
    """Play `num_games`, swapping who starts each game. Returns (wins, draws, losses) for `player`."""
    wins = draws = losses = 0
    for i in range(num_games):
        if i % 2 == 0:
            result = play_game(player, opponent)
        else:
            result = -play_game(opponent, player)
        if result == 1:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1
    return wins, draws, losses
