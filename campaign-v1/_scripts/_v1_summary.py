# _v1_summary.py DIR... — V1 tables from the raw outputs of one or more machine directories.
# Herbert: its [turn] line (prompt tok/s, generated tok/s), median of the rounds.
# llama-bench: avg_ts of each JSON entry (prefill: n_prompt > 0; decode: n_gen > 0), median of the rounds.
# vLLM (latency bench): prefill = input / L(out 1); decode = 254 / (L(out 255) - L(out 1)); fixed cost = L at 1, 16, 64.
import sys, os, re, json, glob, statistics as st

def med(v):
    return st.median(v) if v else None

def spread(v):
    return (max(v) - min(v)) / st.median(v) * 100 if len(v) > 1 else 0.0

def turn(f):
    for l in open(f, errors="replace"):
        if l.startswith("[turn"):
            p = re.search(r"(\d+) prompt tokens in [\d.]+ s \(([\d.]+) tok/s\)", l)
            g = re.search(r"(\d+) generated in [\d.]+ s \(([\d.]+) tok/s\)", l)
            return float(p.group(2)), float(g.group(2)), int(g.group(1))
    return None

def lb(f):
    try:
        return json.load(open(f))
    except Exception:
        return []

def fmt(x, d=1):
    return "-" if x is None else f"{x:,.{d}f}".replace(",", " ")

def prefill_sweep(d):
    rows = {}
    for f in glob.glob(os.path.join(d, "pp-r*-*-herbert-pp*.log")):
        m = re.match(r"pp-r\d+-([\d.]+B)-herbert-pp(\d+)\.log", os.path.basename(f))
        t = turn(f)
        if t:
            rows.setdefault((m.group(1), "herbert"), {}).setdefault(int(m.group(2)), []).append(t[0])
    for f in glob.glob(os.path.join(d, "pp-*-llama-*.json")):
        m = re.match(r"pp-([\d.]+B)-llama-(cuda|vk)\.json", os.path.basename(f))
        for e in lb(f):
            rows.setdefault((m.group(1), f"llama.cpp {m.group(2)}"), {}).setdefault(e["n_prompt"], []).extend(e.get("samples_ts") or [e["avg_ts"]])
    fixed = {}
    for f in glob.glob(os.path.join(d, "pp-vllm-*-pp*.json")):
        m = re.match(r"pp-vllm-([\d.]+B)-pp(\d+)\.json", os.path.basename(f))
        lat = json.load(open(f))["avg_latency"]
        n = int(m.group(2))
        if n in (1, 16, 64):
            fixed[(m.group(1), n)] = lat * 1000
        else:
            rows.setdefault((m.group(1), "vLLM (end to end)"), {}).setdefault(n, []).append(n / lat)
    if not rows:
        return
    pps = [128, 256, 512, 1024, 2048, 4000]
    print(f"\n## {d}: prefill against the length (tok/s, median; spread max-min/median in %)\n")
    print("| model | engine | " + " | ".join(map(str, pps)) + " |")
    print("|---|---|" + "---|" * len(pps))
    for (m, e), v in sorted(rows.items()):
        print(f"| {m} | {e} | " + " | ".join(f"{fmt(med(v[p]))} ({spread(v[p]):.1f})" if p in v else "-" for p in pps) + " |")
    for m in sorted({k[0] for k in rows}):
        h = rows.get((m, "herbert"), {})
        for e in sorted({k[1] for k in rows if k[0] == m and k[1] != "herbert"}):
            o = rows[(m, e)]
            print(f"herbert / {e} {m}: " + " ".join(f"{med(h[p]) / med(o[p]) * 100:.1f}%" if p in h and p in o else "-" for p in pps))
    if fixed:
        print("\nvLLM latency with a short prompt and one generated token (ms): " + ", ".join(f"{m} pp{n} {v:.2f}" for (m, n), v in sorted(fixed.items())))

def shape(d, name, n_prompt):
    if not os.path.isdir(d):
        return
    eng = {}
    for f in glob.glob(os.path.join(d, "r*-*.log")):
        m = re.match(r"r\d+-([\d.]+B)-(.+)\.log", os.path.basename(f))
        t = turn(f)
        if t:
            k = (m.group(1), "herbert " + m.group(2))
            eng.setdefault(k, ([], [], []))
            eng[k][0].append(t[0]); eng[k][1].append(t[1]); eng[k][2].append(t[2])
    for f in glob.glob(os.path.join(d, "r*-*-llama-*.json")):
        m = re.match(r"r\d+-([\d.]+B)-llama-(.+)-(pp|tg)\.json", os.path.basename(f))
        k = (m.group(1), "llama.cpp " + m.group(2))
        eng.setdefault(k, ([], [], []))
        for e in lb(f):
            (eng[k][0] if m.group(3) == "pp" else eng[k][1]).append(e["avg_ts"])
    for m in ("0.8B", "2B", "4B"):
        a, b = os.path.join(d, f"vllm-{m}-out1.json"), os.path.join(d, f"vllm-{m}-out255.json")
        if os.path.exists(a) and os.path.exists(b):
            l1, l255 = json.load(open(a))["avg_latency"], json.load(open(b))["avg_latency"]
            eng[(m, "vLLM (end to end)")] = ([n_prompt / l1], [254 / (l255 - l1)], [])
    if not eng:
        return
    print(f"\n## {d}: {name} (prefill {n_prompt} tokens / decode 255 at depth {n_prompt}, tok/s, median; spread %)\n")
    print("| model | engine | prefill | decode | N |")
    print("|---|---|---|---|---|")
    for (m, e), (p, g, n) in sorted(eng.items()):
        print(f"| {m} | {e} | {fmt(med(p))} ({spread(p):.1f}) | {fmt(med(g), 2)} ({spread(g):.1f}) | {max(len(p), len(g))} |")

def sweep(d):
    for f in sorted(glob.glob(os.path.join(d, "q-*-llama-*.json"))):
        best = {}
        for e in lb(f):
            k = e["n_prompt"]
            if k not in best or e["avg_ts"] > best[k][0]:
                best[k] = (e["avg_ts"], e["n_ubatch"], e["flash_attn"])
        dflt = {e["n_prompt"]: e["avg_ts"] for e in lb(f) if e["n_ubatch"] == 512 and e["flash_attn"] == 1}
        print(f"\n{os.path.basename(f)}: best (tok/s, ubatch, fa) per length: " + "; ".join(f"{k}: {v[0]:.0f} ub {v[1]} fa {v[2]}" for k, v in sorted(best.items())))

for d in sys.argv[1:]:
    prefill_sweep(d)
    shape(os.path.join(d, "p3"), "p3", 759)
    shape(os.path.join(d, "p3-raphael"), "p3, Raphael iGPU", 759)
    shape(os.path.join(d, "p4"), "p4", 2880)
    sweep(d)
