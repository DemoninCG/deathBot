import json

d = json.load(open("../_diag_out/_dose_sweep.json"))
print("=== FIT CELLS ===")
for name, c in d.get("cells", {}).items():
    print(f"{name:8s} lr {c['lr']:.0e} steps {c['steps']:5d} "
          f"dW {c['dW']:6.2f} agree {c['post_train_agree']:.4f}/"
          f"{c['post_val_agree']:.4f} loss {c['final_loss']}")
print()
for tag in ("eval_train26000", "eval_report31000"):
    leg = d.get(tag)
    if not leg:
        continue
    print("===", tag, "===")
    for name, r in leg.items():
        if "delta_vs_snap9" not in r:
            print(f"  {name:8s} mean {r['mean']:7.1f} (baseline)")
            continue
        extra = (f"delta {r['delta_vs_snap9']:+7.1f} t {r['t']:+.2f} "
                 f"cap {r['capped']} W/L {r['wins']}/{r['losses']}")
        if "delta_vs_base" in r:
            extra += f"  vs_base {r['delta_vs_base']:+7.1f} t {r['t_vs_base']:+.2f}"
        print(f"  {name:8s} mean {r['mean']:7.1f} {extra}")
print()
print("=== HISTORY (last 3 checkpoints per cell) ===")
for name, c in d.get("cells", {}).items():
    tail = c.get("history", [])[-3:]
    pts = ", ".join(f"s{h['step']} loss {h['loss']} dW {h['dW']} ag {h['agree3k']}"
                    for h in tail)
    print(f"  {name:8s} {pts}")