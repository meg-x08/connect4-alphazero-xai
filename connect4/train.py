"""Self-play training: the AI plays against itself and learns from those games."""

import glob
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from .game import Connect4
from .mcts import mcts
from .network import Connect4Net, save_checkpoint
from .players import AlphaZeroPlayer, GreedyPlayer, RandomPlayer, play_match


def self_play_game(net, num_simulations=50, random_moves=8):
    """Play one game against itself. Returns training examples (state, policy, result).

    - policy: how often the search visited each column (what the network should learn to predict)
    - result: +1 if the player to move in that position went on to win, -1 if they lost, 0 for a draw
    """
    game = Connect4()
    history = []
    while not game.is_over():
        visits = mcts(game, net, num_simulations, add_noise=True)
        policy = visits / visits.sum()
        history.append((game.encode(), policy, game.current_player))
        if game.num_moves < random_moves:
            move = np.random.choice(7, p=policy)  # explore different openings
        else:
            move = int(np.argmax(visits))
        game.play(move)

    examples = []
    for state, policy, player in history:
        result = 0.0 if game.winner == 0 else (1.0 if game.winner == player else -1.0)
        examples.append((state, policy, result))
        # Connect-4 is symmetric left-right, so the mirrored position is a free extra example
        examples.append((state[:, :, ::-1].copy(), policy[::-1].copy(), result))
    return examples


def train_network(net, optimizer, examples, steps=100, batch_size=128):
    """A few gradient steps on random batches. Returns the average (policy loss, value loss)."""
    net.train()
    states = torch.tensor(np.array([e[0] for e in examples]))
    policies = torch.tensor(np.array([e[1] for e in examples]), dtype=torch.float32)
    results = torch.tensor([e[2] for e in examples], dtype=torch.float32)
    policy_losses, value_losses = [], []
    for _ in range(steps):
        idx = torch.randint(len(examples), (batch_size,))
        logits, values = net(states[idx])
        # policy loss: cross-entropy between the search's choices and the network's guesses
        policy_loss = -(policies[idx] * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        # value loss: squared error between the predicted and the real result
        value_loss = F.mse_loss(values, results[idx])
        optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        optimizer.step()
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
    net.eval()
    return float(np.mean(policy_losses)), float(np.mean(value_losses))


def latest_checkpoint(save_dir):
    files = sorted(glob.glob(os.path.join(save_dir, "iter_*.pt")))
    return files[-1] if files else None


def train(
    save_dir="checkpoints",
    iterations=40,
    games_per_iteration=40,
    num_simulations=50,
    train_steps=100,
    buffer_size=30000,
    learning_rate=1e-3,
    eval_every=5,
    eval_games=20,
):
    """The full training loop. Saves a checkpoint after every iteration, so if it stops
    (for example a Colab disconnect) running it again continues where it left off."""
    os.makedirs(save_dir, exist_ok=True)
    net = Connect4Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    buffer = deque(maxlen=buffer_size)  # most recent training examples
    log = []
    start = 1

    checkpoint = latest_checkpoint(save_dir)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        net.load_state_dict(state["model"])
        if state["optimizer"]:
            optimizer.load_state_dict(state["optimizer"])
        log = state["log"]
        start = state["iteration"] + 1
        print(f"Resuming from {checkpoint} (iteration {state['iteration']})")
    else:
        save_checkpoint(os.path.join(save_dir, "iter_000.pt"), net, iteration=0)  # untrained network

    for iteration in range(start, iterations + 1):
        t0 = time.time()
        for _ in range(games_per_iteration):
            buffer.extend(self_play_game(net, num_simulations))
        policy_loss, value_loss = train_network(net, optimizer, buffer, steps=train_steps)

        entry = {
            "iteration": iteration,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "minutes": (time.time() - t0) / 60,
        }
        if iteration % eval_every == 0 or iteration == iterations:
            agent = AlphaZeroPlayer(net, num_simulations)
            for opponent in (RandomPlayer(), GreedyPlayer()):
                wins, draws, losses = play_match(agent, opponent, eval_games)
                entry[f"win_rate_vs_{opponent.name}"] = wins / eval_games
        log.append(entry)
        save_checkpoint(
            os.path.join(save_dir, f"iter_{iteration:03d}.pt"), net, optimizer, iteration, log
        )

        msg = f"iteration {iteration:3d} | policy loss {policy_loss:.3f} | value loss {value_loss:.3f}"
        for key in ("win_rate_vs_random", "win_rate_vs_greedy"):
            if key in entry:
                msg += f" | {key} {entry[key]:.0%}"
        print(msg + f" | {entry['minutes']:.1f} min", flush=True)
    return net, log
