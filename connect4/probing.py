"""Looking inside the network: does it represent simple Connect-4 concepts?

A "probe" is a simple classifier (logistic regression) that tries to read a
concept, e.g. "the opponent threatens to win", from the numbers inside one
layer of the network. If it works on positions from games it has never seen,
the information is stored in that layer.

To make sure the result means something, every probe is compared with:
- the same probe on the raw board (what is easy to read directly from the input), and
- the same probe on an untrained network (what random features already give for free).
"""

import random

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .game import Connect4
from .players import GreedyPlayer, RandomPlayer

# ----------------------------------------------------------------- concepts ---


def can_win_now(game):
    """The player to move can win with their next move."""
    return len(game.winning_moves(game.current_player)) > 0


def must_block(game):
    """The opponent threatens to win on their next move."""
    return len(game.winning_moves(-game.current_player)) > 0


def controls_centre(game):
    """The player to move has more stones in the centre column than the opponent."""
    centre = game.board[:, 3]
    return (centre == game.current_player).sum() > (centre == -game.current_player).sum()


CONCEPTS = {
    "can_win_now": can_win_now,
    "must_block": must_block,
    "controls_centre": controls_centre,
}

# ---------------------------------------------------------------- positions ---


def collect_positions(num_games=600, seed=0):
    """Positions from games between random and greedy players (a mix of calm and tactical play).

    Returns the positions and, for each one, the id of the game it came from.
    """
    random.seed(seed)
    np.random.seed(seed)
    players = [RandomPlayer(), GreedyPlayer()]
    positions, game_ids = [], []
    for game_id in range(num_games):
        game = Connect4()
        first, second = random.choice(players), random.choice(players)
        while not game.is_over():
            positions.append(game.copy())
            game_ids.append(game_id)
            player = first if game.current_player == 1 else second
            game.play(player.choose_move(game))
    return positions, np.array(game_ids)


def concept_labels(positions):
    return {name: np.array([fn(g) for g in positions]) for name, fn in CONCEPTS.items()}


def layer_features(net, positions):
    """For each layer, a (num_positions x num_features) array. Includes the raw board as "board"."""
    x = torch.tensor(np.array([g.encode() for g in positions]))
    net.eval()
    with torch.no_grad():
        layers = net.layer_outputs(x)
    features = {"board": x.flatten(1).numpy()}
    features.update({name: out.numpy() for name, out in layers.items()})
    return features


# ------------------------------------------------------------------- probes ---


def probe_score(features, labels, game_ids, seed=0):
    """Train a logistic-regression probe on 80% of the games, test it on the other 20%.

    Splitting by game (not by position) means test positions really are unseen.
    Returns balanced accuracy: 0.5 = guessing, 1.0 = perfect.
    """
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(split.split(features, labels, groups=game_ids))
    probe = make_pipeline(
        StandardScaler(), LogisticRegression(C=0.05, max_iter=500, class_weight="balanced")
    )
    probe.fit(features[train_idx], labels[train_idx])
    return balanced_accuracy_score(labels[test_idx], probe.predict(features[test_idx]))


def probe_all(net, positions, labels, game_ids):
    """Probe score for every (concept, layer) pair. Returns {concept: {layer: score}}."""
    features = layer_features(net, positions)
    return {
        concept: {layer: probe_score(f, y, game_ids) for layer, f in features.items()}
        for concept, y in labels.items()
    }
