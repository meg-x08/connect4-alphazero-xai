import numpy as np
import torch

from connect4.game import Connect4
from connect4.mcts import mcts
from connect4.network import Connect4Net
from connect4.players import GreedyPlayer, RandomPlayer, play_match
from connect4.probing import CONCEPTS, collect_positions, concept_labels, probe_all
from connect4.train import self_play_game


class UniformNet:
    """A fake network with no opinion: equal probabilities, value 0."""

    def predict(self, game):
        return np.full(7, 1 / 7), 0.0


def play_moves(moves):
    game = Connect4()
    for ch in moves:
        game.play(int(ch) - 1)
    return game


def test_search_finds_a_winning_move():
    game = play_moves("112233")  # X can win in column 4
    visits = mcts(game, UniformNet(), num_simulations=200)
    assert np.argmax(visits) == 3


def test_search_blocks_the_opponent():
    game = play_moves("11223")  # O must block column 4
    visits = mcts(game, UniformNet(), num_simulations=400)
    assert np.argmax(visits) == 3


def test_greedy_beats_random():
    wins, draws, losses = play_match(GreedyPlayer(), RandomPlayer(), num_games=20)
    assert wins >= 16


def test_network_output_shapes():
    net = Connect4Net()
    policy, value = net.predict(Connect4())
    assert policy.shape == (7,) and abs(policy.sum() - 1) < 1e-5
    assert -1 <= value <= 1


def test_self_play_examples_have_results():
    torch.manual_seed(0)
    examples = self_play_game(Connect4Net(), num_simulations=5)
    assert len(examples) % 2 == 0  # each position plus its mirror image
    assert all(result in (-1.0, 0.0, 1.0) for _, _, result in examples)


def test_concepts_on_known_positions():
    game = play_moves("112233")  # X to move and can win
    assert CONCEPTS["can_win_now"](game) and not CONCEPTS["must_block"](game)
    game = play_moves("11223")  # O to move and must block
    assert CONCEPTS["must_block"](game) and not CONCEPTS["can_win_now"](game)
    assert CONCEPTS["controls_centre"](play_moves("41"))  # X to move, X owns the centre


def test_probing_runs_and_reads_simple_concepts():
    positions, game_ids = collect_positions(num_games=60)
    labels = concept_labels(positions)
    scores = probe_all(Connect4Net(), positions, labels, game_ids)
    assert set(scores) == set(CONCEPTS)
    assert scores["controls_centre"]["board"] > 0.9  # easy to read straight off the board
