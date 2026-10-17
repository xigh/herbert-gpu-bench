# _v5_summary.py DIR... — V5, f16 against f16: Herbert f16 (and Herbert bf16, same rounds) against llama.cpp F16.
# Bench: medians of the 5 rounds (Herbert's [turn] line, warm; llama-bench avg_ts). Client: medians of _v2_http.py.
import json, os, re, glob, sys, statistics as st
def lb(f):
    try: return json.load(open(f))
    except Exception: return []
def turn(f):
    for l in open(f, errors='replace'):
        if l.startswith('[turn'):
            return (float(re.search(r'prompt tokens in [\d.]+ s \(([\d.]+) tok/s\)', l).group(1)),
                    float(re.search(r'generated in [\d.]+ s \(([\d.]+) tok/s\)', l).group(1)))
def pct(a, b): return f"{a / b * 100:.1f} %" if a and b else "-"
for d in sys.argv[1:]:
    name = os.path.basename(os.path.normpath(d))
    v = {}
    for f in glob.glob(f'{d}/bench/r*-*'):
        b = os.path.basename(f)
        m = re.match(r'r\d+-([\d.]+B)-p(\d+)-herbert-(bf16|f16)\.log$', b)
        if m:
            t = turn(f)
            if t: v.setdefault((m.group(1), int(m.group(2)), 'herbert-' + m.group(3)), []).append(t)
            continue
        m = re.match(r'r\d+-([\d.]+B)-p(\d+)-llama-(vk|cuda)-f16-(pp|tg)\.json$', b)
        if m:
            for e in lb(f): v.setdefault((m.group(1), int(m.group(2)), 'llama-' + m.group(3), m.group(4)), []).append(e['avg_ts'])
    med = {k: (tuple(st.median(x[i] for x in s) for i in (0, 1)) if len(k) == 3 else st.median(s)) for k, s in v.items()}
    n = {k: len(s) for k, s in v.items()}
    cuda = any(k[2] == 'llama-cuda' for k in med)
    print(f"## {name}, bench (tok/s, medians; prefill N / decode 255 at depth N)\n")
    head = "| model | N | Herbert f16 pp / tg | Herbert bf16 pp / tg | llama.cpp Vulkan F16 pp / tg |" + (" llama.cpp CUDA F16 pp / tg |" if cuda else "") + " Herbert f16 / llama Vulkan pp, tg |" + (" / llama CUDA pp, tg |" if cuda else "") + " rounds |"
    print(head); print("|" + "---|" * (head.count('|') - 1))
    for m, N in sorted({(k[0], k[1]) for k in med}, key=lambda x: (float(x[0][:-1]), x[1])):
        hf, hb = med.get((m, N, 'herbert-f16')), med.get((m, N, 'herbert-bf16'))
        row = [m, str(N), f"{hf[0]:.1f} / {hf[1]:.2f}" if hf else "-", f"{hb[0]:.1f} / {hb[1]:.2f}" if hb else "-"]
        ls = ['llama-vk'] + (['llama-cuda'] if cuda else [])
        for e in ls:
            pp, tg = med.get((m, N, e, 'pp')), med.get((m, N, e, 'tg'))
            row.append(f"{pp:.1f} / {tg:.2f}" if pp and tg else "-")
        for e in ls:
            pp, tg = med.get((m, N, e, 'pp')), med.get((m, N, e, 'tg'))
            row.append(f"{pct(hf and hf[0], pp)}, {pct(hf and hf[1], tg)}")
        row.append(str(min(n.get((m, N, 'herbert-f16'), 0), n.get((m, N, 'llama-vk', 'pp'), 0))))
        print("| " + " | ".join(row) + " |")
    print()
    hs = sorted(glob.glob(f'{d}/http-*.json'))
    if not hs: continue
    print(f"## {name}, client (TTFT ms / decode tok/s, medians)\n")
    for m in ('0.8B', '2B', '4B'):
        eng = {re.match(rf'.*/http-(.+)-{re.escape(m)}\.json$', f).group(1): {r['prompt_key']: r for r in lb(f)} for f in hs if f.endswith(f'-{m}.json')}
        if 'herbert-f16' not in eng: continue
        ls = [e for e in ('llama-vk-f16', 'llama-cuda-f16') if e in eng]
        head = f"| {m} prompt tokens | Herbert f16 | Herbert bf16 | " + " | ".join(ls) + " | Herbert f16 TTFT / best llama | decode / best llama |"
        print(head); print("|" + "---|" * (head.count('|') - 1))
        for k in sorted(eng['herbert-f16']):
            c = lambda e: f"{eng[e][k]['ttft_ms_median']:.1f} ms / {eng[e][k]['decode_tps_median']:.1f}" if e in eng and k in eng[e] else "-"
            h = eng['herbert-f16'][k]
            bt = min(eng[e][k]['ttft_ms_median'] for e in ls); bd = max(eng[e][k]['decode_tps_median'] for e in ls)
            print(f"| {h['input_tokens'][0]} | {c('herbert-f16')} | {c('herbert-bf16')} | " + " | ".join(c(e) for e in ls) + f" | {pct(h['ttft_ms_median'], bt)} | {pct(h['decode_tps_median'], bd)} |")
        print()
