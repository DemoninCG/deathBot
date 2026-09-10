import json
d = json.load(open('deathbot_sim/_diag_out/commit_a120_followup.json'))
for leg, cap in [('cap300', 300), ('cap1500', 1500)]:
    print('====', leg, '====')
    L = d['legs'][leg]
    base = L['a110_r60']['per_seed']
    ch = L['a120_r50']['per_seed']
    for s in sorted(base, key=int):
        b = base[s]; a = ch[s]
        dt = int(a['clears'].get('4', 0)) - int(b['clears'].get('4', 0))
        dp = a['pieces'] - b['pieces']
        if dt != 0 or dp != 0 or a['pieces'] < cap or b['pieces'] < cap:
            print(f"seed {s}: base {b['pieces']}/{b['clears'].get('4',0)}({b['reason']}) vs a120 {a['pieces']}/{a['clears'].get('4',0)}({a['reason']}) dT{dt:+d} dP{dp:+d}")
    print('deaths base:', [(s, v['pieces'], v['reason']) for s, v in base.items() if v['pieces'] < cap])
    print('deaths a120:', [(s, v['pieces'], v['reason']) for s, v in ch.items() if v['pieces'] < cap])
