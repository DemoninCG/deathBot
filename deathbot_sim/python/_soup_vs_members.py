import json
import statistics

d = json.load(open("../_diag_out/_soup_screen.json"))["eval"]
for tag in ("train26000", "report31000"):
    leg = d[tag]
    print("===", tag, "===")
    for name in ("snap9", "s42", "s43", "s44", "soup2_43", "soup3"):
        r = leg[name]
        extra = (f"delta {r['delta_vs_snap9']:+7.1f} t {r['t']:+.2f} "
                 f"W/L {r['wins']}/{r['losses']}"
                 if "delta_vs_snap9" in r else "(baseline)")
        print(f"  {name:10s} mean {r['mean']:7.1f} "
              f"cap {str(r.get('capped','-')):>4} {extra}")
    for member in ("s43", "s42"):
        a = leg["soup3"]["per_seed"]
        b = leg[member]["per_seed"]
        dl = [x - y for x, y in zip(a, b)]
        m = statistics.fmean(dl)
        sd = statistics.stdev(dl)
        t = m / (sd / len(dl) ** 0.5) if sd > 0 else float("nan")
        print(f"  soup3 - {member}: {m:+7.1f}  t {t:+.2f}  "
              f"W/L {sum(x > 0 for x in dl)}/{sum(x < 0 for x in dl)}")
    print()
