"""Monte Carlo Tree Search guided by the network (the AlphaZero version of MCTS).

Each simulation walks down the tree of possible moves, asks the network to
evaluate the position it reaches, and passes that evaluation back up the path.
After many simulations, the most-visited move is the one the search trusts most.
"""

import math

import numpy as np


class Node:
    def __init__(self, prior):
        self.prior = prior  # the network's probability for the move that leads here
        self.visits = 0
        self.value_sum = 0.0  # from the point of view of the player who made that move
        self.children = {}  # move -> Node

    def mean_value(self):
        return self.value_sum / self.visits if self.visits else 0.0


def mcts(game, net, num_simulations=50, c_puct=1.5, add_noise=False):
    """Search from `game` and return how many times each of the 7 columns was visited."""
    root = Node(prior=1.0)
    expand(root, game, net)
    if add_noise:  # extra randomness at the root so self-play games are not all the same
        add_dirichlet_noise(root)

    for _ in range(num_simulations):
        node = root
        scratch = game.copy()
        path = [node]

        # 1. Selection: keep choosing the most promising child until we reach a new position
        while node.children and not scratch.is_over():
            move, node = select_child(node, c_puct)
            scratch.play(move)
            path.append(node)

        # 2. Evaluation: how good is this position for the player to move?
        if scratch.is_over():
            value = 0.0 if scratch.winner == 0 else -1.0  # the previous player just won
        else:
            value = expand(node, scratch, net)

        # 3. Backup: each node stores the value for the player who moved into it,
        #    so the sign flips at every step up the path
        for n in reversed(path):
            n.visits += 1
            n.value_sum -= value
            value = -value

    visits = np.zeros(7)
    for move, child in root.children.items():
        visits[move] = child.visits
    return visits


def expand(node, game, net):
    """Add one child per legal move, using the network's policy as priors. Returns the value."""
    policy, value = net.predict(game)
    legal = game.legal_moves()
    total = sum(policy[m] for m in legal)
    for move in legal:
        node.children[move] = Node(prior=policy[move] / total)
    return value


def select_child(node, c_puct):
    """PUCT rule: good average value (exploit) plus a bonus for promising, rarely tried moves (explore)."""
    best_score, best_move, best_child = -math.inf, None, None
    for move, child in node.children.items():
        explore = c_puct * child.prior * math.sqrt(node.visits + 1) / (1 + child.visits)
        score = child.mean_value() + explore
        if score > best_score:
            best_score, best_move, best_child = score, move, child
    return best_move, best_child


def add_dirichlet_noise(node, alpha=1.0, fraction=0.25):
    moves = list(node.children)
    noise = np.random.dirichlet([alpha] * len(moves))
    for move, eta in zip(moves, noise):
        child = node.children[move]
        child.prior = (1 - fraction) * child.prior + fraction * eta
