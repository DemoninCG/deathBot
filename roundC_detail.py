import json
d = json.load(open('deathbot_sim/_diag_out/commit_roundC.json'))
for leg, cap in [('cap300', 300), ('cap1500', 1500)]:
    print('====', leg, '====')
    L = d['legs'][leg]
    base = L['baseline']['per_seed']
    for arm in L:
        if arm == 'baseline':
            continue
        print('---', arm, 'vs baseline ---')
        for s in sorted(base, key=int):
            b = base[s]
            a = L[arm]['per_seed'][s]
            dt = int(a['clears'].get('4', 0)) - int(b['clears'].get('4', 0))
            dp = a['pieces'] - b['pieces']
            if dt != 0 or dp != 0 or a['pieces'] < cap or b['pieces'] < cap:
                print(f"seed {s}: base {b['pieces']}/{b['clears'].get('4',0)}({b['reason']}) vs {arm} {a['pieces']}/{a['clears'].get('4',0)}({a['reason']}) dT{dt:+d} dP{dp:+d}")
    print('deaths baseline:', [(s, v['pieces'], v['reason']) for s, v in base.items() if v['pieces'] < cap])
    for arm in L:
        if arm == 'baseline':
            continue
        print('deaths', arm, ':', [(s, v['pieces'], v['reason']) for s, v in L[arm]['per_seed'].items() if v['pieces'] < cap])
