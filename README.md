# Connect-4 AlphaZero + Explainable AI

A small AlphaZero-style agent that learns Connect-4 only by playing against itself,
followed by an experiment to check which game concepts its neural network has learned.

| Notebook | |
|---|---|
| 1. Train the agent | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meg-x08/connect4-alphazero-xai/blob/main/notebooks/1_train.ipynb) |
| 2. What did it learn? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meg-x08/connect4-alphazero-xai/blob/main/notebooks/2_analysis.ipynb) |

## Idea

AlphaZero (DeepMind) learned chess and Go without human games: a neural network suggests
moves, a tree search checks them, and the network is trained on the games it plays against
itself. This project applies the same idea to Connect-4 at a scale that trains on a normal CPU.

The second part looks inside the trained network. Neural networks are often treated as black
boxes, so the question is: does this one represent concepts a human player would use, such as
"I can win now" or "I must block"?

## How it works

| File | What it does |
|---|---|
| `connect4/game.py` | Rules of the game. The board is a 6×7 array |
| `connect4/network.py` | Convolutional network (4 conv layers) with two outputs: a **policy** (which columns look good) and a **value** (who is winning, from -1 to 1) |
| `connect4/mcts.py` | Monte Carlo Tree Search guided by the network: it looks ahead, using the policy to decide which moves to explore and the value to judge positions |
| `connect4/train.py` | Self-play loop: play games with MCTS, train the network to predict the search's moves (policy) and the final result (value), repeat |
| `connect4/players.py` | Random player, greedy player (wins or blocks if it can), network-only player, full agent, and a function to play matches |
| `connect4/probing.py` | Concept definitions and probes for looking inside the network |

## Looking inside the network

Three concepts are computed for each position:

- `can_win_now`: the player to move can win with their next move
- `must_block`: the opponent threatens to win on their next move
- `controls_centre`: the player to move has more stones in the centre column

A **probe** is a logistic-regression classifier trained to predict a concept from the
activations of one layer. If it works on positions from games it has not seen, the concept is
represented in that layer. Each probe is compared with:

- the same probe on the **raw board**, and
- the same probe on an **untrained network**, because random features can already make some
  concepts easier to read. Only the improvement over this baseline comes from training.

## Results

_Added after the training run._

## Run it

**Colab:** open notebook 1 with the button above, then *Runtime → Run all* (about 1 hour on the
free CPU runtime; checkpoints are saved to Google Drive). Then do the same with notebook 2.

**Locally:**

```bash
pip install -r requirements.txt
python -m pytest            # tests
jupyter notebook notebooks/
```

## Limitations and possible improvements

- One short training run on a CPU; a longer run or a bigger network would play better.
- Probes show that information is present in a layer, not that the network uses it to choose moves.
- Possible next steps: more concepts (e.g. counting threats), testing whether removing a concept
  changes the agent's moves, and a longer training run on a GPU.

## References

- Silver et al. (2017). *Mastering the game of Go without human knowledge.* Nature.
- Silver et al. (2018). *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play.* Science.
- Alain & Bengio (2016). *Understanding intermediate layers using linear classifier probes.*
- McGrath et al. (2022). *Acquisition of chess knowledge in AlphaZero.* PNAS.
