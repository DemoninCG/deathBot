import json, statistics
d = json.load(open(r"c:\My Things\HTML\Demonin's Item Shop\other\deathBot\deathbot_sim\_diag_out\dense_sweep.json"))
for k in sorted(d["arms"], key=float):
    a = d["arms"][k]
    print(k, round(a["mean_pieces"], 1), round(a["mean_lines"], 1),
          a["tetrises"], round(a["tetris_per_100"], 3), round(a["tetris_share"], 4),
          round(a["lines_in_tetris_frac"], 4), a["capped"], round(a["mean_phi"], 3))
b = json.load(open(r"c:\My Things\HTML\Demonin's Item Shop\other\deathBot\deathbot_sim\_diag_out\rollout_probe.json"))
print("rollout:", b["mean_pieces"], b["mean_lines"], b["tetrises"],
      round(b["tetris_per_100"], 3), round(b["tetris_share"], 4), b["capped"])
c = json.load(open(r"c:\My Things\HTML\Demonin's Item Shop\other\deathBot\deathbot_sim\_diag_out\rollout_baseline.json"))
print("baseline:", c["mean_pieces"], c["mean_lines"], c["tetrises"],
      round(c["tetris_per_100"], 3), round(c["tetris_share"], 4), c["capped"])
