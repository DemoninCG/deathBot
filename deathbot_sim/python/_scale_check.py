import sys, random, statistics
sys.path.insert(0, ".")
import torch
torch.set_num_threads(4)
from model import TAPNet
from neural_scorer import NeuralScorer
from selfplay import play_game

net = TAPNet(output="linear")
net.load_state_dict(torch.load(r"..\..\model_weights_snapshot7.pt", map_location="cpu"))
net.eval()
scorer = NeuralScorer(net=net, device="cpu")

# Play 3 greedy games, log the winner-minus-median score gap at every step.
gaps, tops, all_scores = [], [], []
for seed in (42, 43, 44):
    traj, lines = play_game(scorer, random.Random(seed), epsilon=0.0, max_pieces=300)
    # replay the game to capture per-step score spreads
    import deathbot_sim as ds
    from piece_history import PieceQueue
    from model import piece_to_index
    from afterstate import make_afterstate
    board = ds.Board(); pq = PieceQueue(random.Random(seed)); pos = 0
    while pos < 300:
        cands = board.search_locks(pq.current)
        if not cands: break
        cur = piece_to_index(pq.current); nxt = piece_to_index(pq.next_piece)
        scores = scorer.score_batch(board, cands, current_piece_idx=cur, next_piece_idx=nxt)
        s_sorted = sorted(scores, reverse=True)
        gaps.append(s_sorted[0] - statistics.median(scores))
        tops.append(s_sorted[0]); all_scores.extend(scores)
        pick = max(range(len(cands)), key=lambda i: scores[i])
        a = make_afterstate(board, cands[pick]); board = a.board; pq.advance(); pos += 1

print(f"snapshot7 output stats over {len(all_scores)} candidate scores:")
print(f"  range      : [{min(all_scores):.1f}, {max(all_scores):.1f}]")
print(f"  mean |s|   : {statistics.fmean(abs(s) for s in all_scores):.1f}")
print(f"  top1-median gap: mean={statistics.fmean(gaps):.2f}  p10={sorted(gaps)[len(gaps)//10]:.2f}")
print()
print(f"Current-code normalized targets live in approx [-3.2, +1.5]")
print(f"Legacy raw targets live in approx [-250, +55]")
print(f"=> snapshot7's outputs (typ |s| ~{statistics.fmean(abs(s) for s in all_scores):.0f}, gap ~{statistics.fmean(gaps):.1f}) are ~100x the normalized target scale")
