# _v7_summary.py DIR — V7: start-up to the first token, medians of rounds 1..N (round 0 discarded), per mode,
# model and engine; Herbert's split (SPIR-V emission / driver build / load) from its logs.
import json, re, sys, os, glob, statistics as st
d = sys.argv[1]
rows = [json.loads(l) for l in open(f'{d}/startup.jsonl')]
res = {}
for r in rows:
    if r['round'] == 0: continue
    x = r['result']
    res.setdefault((r['mode'], r['model'], r['engine']), []).append(x)
print("V7, 4080: from the launch of the server to its first token (s, medians of 5; ready = health answered, then 1 token).\n")
for mode in ('cold', 'warm'):
    for m in ('0.8B', '4B'):
        print(f"## {mode}, {m}\n\n| engine | ready (s) | first token (s) | errors |\n|---|---|---|---|")
        for e in ('herbert-bf16', 'llama-vk-bf16', 'llama-cuda-bf16', 'vllm-bf16', 'herbert-f16', 'llama-vk-f16', 'llama-cuda-f16', 'vllm-f16'):
            v = res.get((mode, m, e))
            if not v: continue
            ok = [x for x in v if x['first_token_s']]
            err = [x['error'] for x in v if not x['first_token_s']]
            rd = st.median(x['ready_s'] for x in ok) if ok else None; ft = st.median(x['first_token_s'] for x in ok) if ok else None
            print(f"| {e} | {rd:.2f} | {ft:.2f} | {len(err)}{' ' + err[0][:60] if err else ''} |" if ok else f"| {e} | - | - | {len(err)} {err[0][:80] if err else ''} |")
        print()
print("## Herbert's split (ms, medians): SPIR-V emitted / built by the driver / pipelines total; weights loaded (s)\n")
print("| mode | model | build | emitted | driver | total | loaded (s) |\n|---|---|---|---|---|---|---|")
for mode in ('cold', 'warm'):
    for m in ('0.8B', '4B'):
        for e in ('herbert-bf16', 'herbert-f16'):
            em, dr, to, ld = [], [], [], []
            for f in glob.glob(f'{d}/logs/{mode}-r[1-9]-{m}-{e}.log'):
                t = open(f, errors='replace').read()
                p = re.search(r'emitted and built in ([\d.]+) ms: SPIR-V emitted in ([\d.]+) ms \((\d+) KiB\), built by the driver in ([\d.]+) ms', t)
                l = re.search(r'loaded in ([\d.]+) s', t)
                if p: to.append(float(p.group(1))); em.append(float(p.group(2))); dr.append(float(p.group(4)))
                if l: ld.append(float(l.group(1)))
            if to: print(f"| {mode} | {m} | {e} | {st.median(em):.1f} | {st.median(dr):.1f} | {st.median(to):.1f} | {st.median(ld):.2f} |")
