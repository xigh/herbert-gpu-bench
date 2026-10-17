# _f16_summary.py DIR... — f16 against bf16 from bench/_f16-ab.sh: speed at the p3 shape (medians of the
# rounds, warm turn) and, per prompt, the first token that differs from the CPU oracle with the margin there
# (the GPU's logits of its own choice and of the CPU's, from HERBERT_VK_TOP).
import sys, os, re, glob, statistics as st

def turn(f):
    for l in open(f, errors="replace"):
        if l.startswith("[turn"):
            p = re.search(r"prompt tokens in [\d.]+ s \(([\d.]+) tok/s\)", l)
            g = re.search(r"generated in [\d.]+ s \(([\d.]+) tok/s\)", l)
            return float(p.group(1)), float(g.group(1))

def ids(f):
    for l in open(f, errors="replace"):
        if l.startswith("[ids]"):
            return [int(x) for x in l.split()[1:]]

def tops(f):
    t = {}
    for m in re.finditer(r"\[top (\d+)\] ((?:\d+:-?[\d.]+ ?)+)", open(f, errors="replace").read()):
        t[int(m.group(1))] = dict((int(a), float(b)) for a, b in (x.split(":") for x in m.group(2).split()))
    return t

for d in sys.argv[1:]:
    runs = {}
    for f in glob.glob(os.path.join(d, "r*-*.log")):
        m = re.match(r"r\d+-([\d.]+B)-(f16|bf16)\.log", os.path.basename(f))
        t = turn(f) if m else None
        if t:
            runs.setdefault((m.group(1), m.group(2)), []).append(t)
    print(f"\n## {d}\n\n| model | format | prefill 759 tok/s | decode tok/s | N |\n|---|---|---|---|---|")
    for (m, v), r in sorted(runs.items()):
        print(f"| {m} | {v} | {st.median(x[0] for x in r):,.1f} | {st.median(x[1] for x in r):.2f} | {len(r)} |".replace(",", " "))
    for m in sorted({k[0] for k in runs}):
        if (m, "f16") in runs and (m, "bf16") in runs:
            pf = st.median(x[0] for x in runs[(m, "f16")]) / st.median(x[0] for x in runs[(m, "bf16")]) * 100
            dc = st.median(x[1] for x in runs[(m, "f16")]) / st.median(x[1] for x in runs[(m, "bf16")]) * 100
            print(f"\nf16 / bf16 {m}: prefill {pf:.1f} %, decode {dc:.1f} %")
    for c in sorted(glob.glob(os.path.join(d, "cmp-*"))):
        print(f"\n{os.path.basename(c)} — first token that differs from the CPU oracle (gap in the GPU's logits):")
        for cpu in sorted(glob.glob(os.path.join(c, "*-cpu.log"))):
            name = os.path.basename(cpu)[:-8]
            ref = ids(cpu)
            line = [f"  {name:11}"]
            for v in ("bf16", "f16"):
                f = os.path.join(c, f"{name}-vk-{v}-top.log")
                got = ids(f) if os.path.exists(f) else None
                if not ref or not got:
                    line.append(f"{v}: missing"); continue
                n = min(len(ref), len(got))
                i = next((k for k in range(n) if ref[k] != got[k]), None)
                if i is None:
                    line.append(f"{v}: identical ({len(got)}/{len(ref)})"); continue
                t = tops(f).get(i, {})
                gap = (t[got[i]] - t[ref[i]]) if ref[i] in t and got[i] in t else None
                line.append(f"{v}: token {i}" + (f" (gap {gap:g})" if gap is not None else " (cpu choice not in top 5)"))
            print(" · ".join(line))
