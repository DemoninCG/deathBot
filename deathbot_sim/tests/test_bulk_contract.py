"""Regression test: bulk-path contract for line-reporting game loops."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
import random
import torch
from model import TAPNet
from two_ply import play_game_2ply, play_game_2ply_style

W = HERE.parent.parent / "model_weights_snapshot9.pt"


def _net():
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(W, map_location="cpu"))
    net.eval()
    return net


def test_style_game_reports_lines_and_clears():
    net = _net()
    p, lines, clears, reason = play_game_2ply_style(
        net, random.Random(20194), beam=0, max_pieces=300)
    assert p == 300 and reason == "max_pieces"
    assert lines > 0
    assert sum(clears.values()) == 300
    assert sum(k * v for k, v in clears.items()) == lines


def test_full_game_reports_lines():
    net = _net()
    p, lines, reason = play_game_2ply(net, random.Random(20194), beam=0,
                                      max_pieces=300)
    assert p == 300 and reason == "max_pieces"
    assert lines > 0
