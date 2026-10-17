# _v4_summary.py — llama.cpp in F16 (V4) against Herbert bf16 (V1 bench, V2 client), per machine.
import json, os, re, glob, statistics as st
T = os.path.expanduser('~/bench-runs')
def lb(f):
    try: return json.load(open(f))
    except Exception: return []
def bench(d):
    r = {}
    for f in glob.glob(f'{d}/bench/r*-*-p*-llama-*-f16-*.json'):
        m = re.match(r'r\d+-([\d.]+B)-p(\d+)-llama-(\w+)-f16-(pp|tg)\.json', os.path.basename(f))
        for e in lb(f): r.setdefault((m.group(1), int(m.group(2)), m.group(3), m.group(4)), []).append(e['avg_ts'])
    return {k: st.median(v) for k, v in r.items()}
def herbert_v1(d, shape):  # V1 vs.sh logs: r*-{m}-vk-*-warm.log (p3) or p4/r*-{m}-herbert.log
    r = {}
    pat = f'{d}/p3/r*-*-vk-*-warm.log' if shape == 759 else f'{d}/p4/r*-*-herbert.log'
    for f in glob.glob(pat):
        m = re.match(r'r\d+-([\d.]+B)-', os.path.basename(f))
        for l in open(f, errors='replace'):
            if l.startswith('[turn'):
                p = float(re.search(r'prompt tokens in [\d.]+ s \(([\d.]+) tok/s\)', l).group(1)); g = float(re.search(r'generated in [\d.]+ s \(([\d.]+) tok/s\)', l).group(1))
                r.setdefault(m.group(1), []).append((p, g))
    return {k: (st.median(x[0] for x in v), st.median(x[1] for x in v)) for k, v in r.items()}
def http(f):
    return {r['prompt_key']: r for r in lb(f)}
print("V4: llama.cpp F16 (08ffba21e) against Herbert bf16, the same machine, the same day. Bench: prefill N / decode 255 at depth N, medians of 5; Herbert from V1 (#541, warm). Client: TTFT ms / decode tok/s, medians of 15; Herbert from V2 (#542, #543).\n")
for name, v4, v1, shapes, eng in [('Raphael iGPU (beast)', f'{T}/v4/beast', f'{T}/v1/beast/p3-raphael/..', [759], ['vk']),
                                  ('Arc 140V', f'{T}/v4/arc', f'{T}/v1/omnibook-arc', [759, 2880], ['vk']),
                                  ('Cezanne iGPU', f'{T}/v4/cezanne', f'{T}/v1/white2-cezanne', [759], ['vk'])]:
    b = bench(v4)
    print(f"## {name}, bench\n\n| model | N | Herbert bf16 pp / tg | llama.cpp Vulkan F16 pp / tg | Herbert / llama pp | tg |\n|---|---|---|---|---|---|")
    for N in shapes:
        if name.startswith('Raphael'):
            h = {}
            for f in glob.glob(f'{T}/v1/beast/p3-raphael/r*-*-vk-raphael-warm.log'):
                m = re.match(r'r\d+-([\d.]+B)-', os.path.basename(f))
                for l in open(f, errors='replace'):
                    if l.startswith('[turn'):
                        h.setdefault(m.group(1), []).append((float(re.search(r'prompt tokens in [\d.]+ s \(([\d.]+) tok/s\)', l).group(1)), float(re.search(r'generated in [\d.]+ s \(([\d.]+) tok/s\)', l).group(1))))
            h = {k: (st.median(x[0] for x in v), st.median(x[1] for x in v)) for k, v in h.items()}
        else:
            h = herbert_v1(v1, N)
        for m in ('0.8B', '4B'):
            if m in h and (m, N, 'vk', 'pp') in b:
                pp, tg = b[(m, N, 'vk', 'pp')], b[(m, N, 'vk', 'tg')]
                print(f"| {m} | {N} | {h[m][0]:.1f} / {h[m][1]:.2f} | {pp:.1f} / {tg:.2f} | {h[m][0]/pp*100:.1f} % | {h[m][1]/tg*100:.1f} % |")
    print()
for name, v4, v2, pairs in [('RTX 4080 (beast)', f'{T}/v4/beast', f'{T}/v2/beast', [('llama-cuda-f16', 'llama-cuda'), ('llama-vk-f16', 'llama-vk')]),
                            ('Arc 140V', f'{T}/v4/arc', f'{T}/v2b/omnibook-arc', [('llama-vk-f16', 'llama-vk')])]:
    print(f"## {name}, client\n")
    for m in ('0.8B', '4B'):
        hf = f'{v2}/http-herbert-{m}.json'
        if not os.path.exists(hf): continue
        h = http(hf)
        cols = []
        for e16, e in pairs:
            f16 = http(f'{v4}/http-{e16}-{m}.json'); bf = http(f'{v2}/http-{e}-{m}.json')
            cols.append((e16, f16, bf))
        print(f"| {m} prompt tokens | Herbert bf16 | " + " | ".join(f"{e} (BF16 → F16)" for e, _, _ in cols) + " | Herbert TTFT / F16 | decode / F16 |\n|---|---|" + "---|" * len(cols) + "---|---|")
        for k in sorted(h):
            cells = [f"{bf[k]['ttft_ms_median']:.1f} → {f16[k]['ttft_ms_median']:.1f} ms / {bf[k]['decode_tps_median']:.1f} → {f16[k]['decode_tps_median']:.1f}" for _, f16, bf in cols]
            best = min(cols, key=lambda c: c[1][k]['ttft_ms_median'])[1]
            print(f"| {h[k]['input_tokens'][0]} | {h[k]['ttft_ms_median']:.1f} ms / {h[k]['decode_tps_median']:.1f} | " + " | ".join(cells) + f" | {h[k]['ttft_ms_median']/best[k]['ttft_ms_median']*100:.1f} % | {h[k]['decode_tps_median']/max(c[1][k]['decode_tps_median'] for c in cols)*100:.1f} % |")
        print()
