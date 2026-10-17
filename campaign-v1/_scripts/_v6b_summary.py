# _v6b_summary.py DIR — V6b: llama.cpp F16 per setting (median of 5 rounds, -r 1 each), its best setting and its
# default (-b 2048 -ub 512) against Herbert f16 of the same rounds; decode at -fa 0/1.
import json, re, sys, glob, statistics as st
d = sys.argv[1]; v = {}; h = {}; errs = []
for f in glob.glob(f'{d}/bench/*'):
    b = f.rsplit('/', 1)[1]
    m = re.match(r'r(\d)-([\d.]+B)-p(\d+)-herbert-f16\.log$', b)
    if m:
        for l in open(f, errors='replace'):
            if l.startswith('[turn'):
                h.setdefault((m.group(2), int(m.group(3))), []).append((float(re.search(r'prompt tokens in [\d.]+ s \(([\d.]+) tok/s\)', l).group(1)), float(re.search(r'generated in [\d.]+ s \(([\d.]+) tok/s\)', l).group(1))))
        continue
    m = re.match(r'r(\d)-([\d.]+B)-p(\d+)-(cuda|vk)-(.+)-(pp|tg)\.json$', b)
    if m:
        try: x = json.load(open(f))
        except Exception: x = []
        if not x: errs.append(b); continue
        v.setdefault((m.group(2), int(m.group(3)), m.group(4), m.group(5), m.group(6)), []).append(x[0]['avg_ts'])
med = {k: (st.median(s), len(s), (max(s) - min(s)) / st.median(s) * 100) for k, s in v.items()}
hm = {k: (st.median(x[0] for x in s), st.median(x[1] for x in s), len(s)) for k, s in h.items()}
print(f"V6b, 4080: llama.cpp F16 per setting, medians of 5 rounds; Herbert f16 in the same rounds. Empty outputs: {len(errs)} {errs[:6]}\n")
print("| model | N | Herbert f16 pp | backend | llama default b2048/ub512 (fa0 · fa1) | llama best (setting) | Herbert / best | Herbert / default best-fa |\n|---|---|---|---|---|---|---|---|")
for (mo, N), (hp, ht, hn) in sorted(hm.items(), key=lambda x: (float(x[0][0][:-1]), x[0][1])):
    for be in ('cuda', 'vk'):
        pp = {k[3]: val for k, val in med.items() if k[0] == mo and k[1] == N and k[2] == be and k[4] == 'pp'}
        if not pp: continue
        best = max(pp, key=lambda s: pp[s][0]); d0, d1 = pp.get('b2048-ub512-fa0'), pp.get('b2048-ub512-fa1')
        dbest = max(x[0] for x in (d0, d1) if x)
        print(f"| {mo} | {N} | {hp:.0f} | {be} | {d0[0]:.0f} · {d1[0]:.0f} | {pp[best][0]:.0f} ({best}, spread {pp[best][2]:.1f} %) | ×{hp / pp[best][0]:.2f} | ×{hp / dbest:.2f} |")
print("\n| model | N | Herbert f16 tg | backend | llama tg fa0 · fa1 | Herbert / best |\n|---|---|---|---|---|---|")
for (mo, N), (hp, ht, hn) in sorted(hm.items(), key=lambda x: (float(x[0][0][:-1]), x[0][1])):
    for be in ('cuda', 'vk'):
        t0, t1 = med.get((mo, N, be, 'fa0', 'tg')), med.get((mo, N, be, 'fa1', 'tg'))
        if t0 and t1: print(f"| {mo} | {N} | {ht:.1f} | {be} | {t0[0]:.1f} · {t1[0]:.1f} | {ht / max(t0[0], t1[0]) * 100:.1f} % |")
