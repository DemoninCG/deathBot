"""Speed benchmark harness: before/after timing for the 12-item plan."""
import random, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import deathbot_sim as ds
from model import TAPNet, piece_to_index
from afterstate import rows_to_tensor, flat_rows_to_tensor

W = str((HERE / "../../model_weights_snapshot9.pt").resolve())
if not Path(W).is_file():
    W = str((HERE.parent / "model_weights.pt").resolve())

def load():
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(W, map_location="cpu"))
    net.eval()
    torch.set_num_threads(1)
    return net

def bench_search_lock():
    b = ds.Board()
    for kind in [ds.PieceType.T, ds.PieceType.I, ds.PieceType.S]:
        t0 = time.perf_counter(); N = 300
        for _ in range(N):
            b.search_locks(kind)
        print(f"search_locks {kind} x{N}: {(time.perf_counter()-t0)*1000/N:.3f} ms/call")
        t0 = time.perf_counter()
        for _ in range(N):
            b.lock_candidates_full(kind)
        print(f"lock_full {kind} x{N}: {(time.perf_counter()-t0)*1000/N:.3f} ms/call")

def bench_nn(net):
    for B in [1, 17, 34, 200, 600]:
        boards = torch.zeros((B, 1, 20, 10))
        cur = torch.zeros(B, dtype=torch.long); nxt = torch.zeros(B, dtype=torch.long)
        # warmup
        with torch.no_grad():
            net(boards, cur, nxt)
        t0 = time.perf_counter(); N = 20
        with torch.no_grad():
            for _ in range(N):
                net(boards, cur, nxt)
        print(f"nn forward B={B}: {(time.perf_counter()-t0)*1000/N:.3f} ms/call")

def bench_rows():
    b = ds.Board()
    _, _, _, _, flat, _ = b.lock_candidates_full(ds.PieceType.T)
    n = 34
    flat = np.asarray(flat, dtype=np.uint32)
    t0 = time.perf_counter(); N = 200
    for _ in range(N):
        flat_rows_to_tensor(flat, n)
    print(f"flat_rows_to_tensor n={n} x{N}: {(time.perf_counter()-t0)*1000/N:.3f} ms/call")

def bench_twoply(net, pieces=30, beam=100):
    from two_ply import play_game_2ply_style
    t0 = time.perf_counter()
    p, l, c, r = play_game_2ply_style(net, random.Random(1), beam=beam, max_pieces=pieces)
    dt = time.perf_counter() - t0
    print(f"2-ply game beam={beam} pieces={pieces}: {dt:.2f}s total = {dt/max(1,p)*1000:.1f} ms/piece")
    return dt

def bench_heuristic():
    from heuristic import HeuristicScorer
    h = HeuristicScorer()
    b = ds.Board()
    _, _, _, _, flat, lines = b.lock_candidates_full(ds.PieceType.T)
    packed = np.asarray(flat, dtype=np.uint32).reshape(len(lines), 21)
    t0 = time.perf_counter(); N = 50
    for _ in range(N):
        h._weighted_reward_batch(packed, [int(x) for x in lines])
    print(f"heuristic batch n={len(lines)} x{N}: {(time.perf_counter()-t0)*1000/N:.3f} ms/call")

def bench_p2t_step(net=None):
    # simulate current per-state loop cost with fake data
    import torch.nn.functional as F
    B = 64; C = 17
    cand_boards = torch.zeros((B*C, 1, 20, 10))
    cand_cur = torch.zeros(B*C, dtype=torch.long); cand_nxt = torch.zeros(B*C, dtype=torch.long)
    if net is None: net = load()
    net.eval()
    t0 = time.perf_counter(); N = 5
    for _ in range(N):
        total = 0
        for s in range(B):
            lo, hi = s*C, (s+1)*C
            scores = net(cand_boards[lo:hi], cand_cur[lo:hi], cand_nxt[lo:hi]).flatten()
            logp = F.log_softmax(scores, dim=0)
            total = total + (-logp[0])
    print(f"p2t per-state loop B={B} C={C} x{N}: {(time.perf_counter()-t0)/N:.3f} s/step-batch")

if __name__ == "__main__":
    net = load()
    print("== search/lock =="); bench_search_lock()
    print("== nn =="); bench_nn(net)
    print("== rows =="); bench_rows()
    print("== heuristic =="); bench_heuristic()
    print("== p2t loop =="); bench_p2t_step(net)
    print("== 2-ply game =="); bench_twoply(net, pieces=30, beam=100)
