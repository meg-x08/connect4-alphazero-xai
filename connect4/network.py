"""The neural network: looks at a board and outputs a policy and a value."""

import torch
from torch import nn

from .game import COLS, ROWS


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class Connect4Net(nn.Module):
    """A small convolutional network with two outputs ("heads"):

    - policy: one score per column, i.e. which moves look promising
    - value: a number between -1 and 1, i.e. how good the position is for the player to move
    """

    LAYERS = ["conv1", "conv2", "conv3", "conv4"]

    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = conv_block(2, channels)
        self.conv2 = conv_block(channels, channels)
        self.conv3 = conv_block(channels, channels)
        self.conv4 = conv_block(channels, channels)
        size = channels * ROWS * COLS
        self.policy_head = nn.Sequential(nn.Flatten(), nn.Linear(size, COLS))
        self.value_head = nn.Sequential(
            nn.Flatten(), nn.Linear(size, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh()
        )

    def forward(self, x):
        x = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        return self.policy_head(x), self.value_head(x).squeeze(-1)

    def layer_outputs(self, x):
        """The output of every conv layer, flattened. Used to look inside the network."""
        outputs = {}
        for name in self.LAYERS:
            x = getattr(self, name)(x)
            outputs[name] = x.flatten(1)
        return outputs

    def predict(self, game):
        """Move probabilities (7 numbers) and value for a single game position."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(game.encode()).unsqueeze(0)
            logits, value = self(x)
        return torch.softmax(logits, dim=1)[0].numpy(), value.item()


def save_checkpoint(path, net, optimizer=None, iteration=0, log=None):
    torch.save(
        {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "iteration": iteration,
            "log": log or [],
        },
        path,
    )


def load_network(path):
    net = Connect4Net()
    net.load_state_dict(torch.load(path, map_location="cpu")["model"])
    net.eval()
    return net
