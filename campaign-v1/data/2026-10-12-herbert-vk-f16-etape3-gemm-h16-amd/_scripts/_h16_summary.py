# _h16_summary.py DIR... — bench/_h16-ab.sh: the SIMT GEMM in packed f16 (h1, h2, h4, h64 = flush period)
# against f32: p3 speed (medians of the rounds) and the first token that differs from the CPU oracle.
import sys, os, re, glob, statistics as st
V = ["f32", "h1", "h2", "h4", "h64"]
def turn(f):
    for l in open(f, errors="replace"):
        if l.startswith("[turn"):
            return (float(re.search(r"prompt tokens in [\d.]+ s \(([\d.]+) tok/s\)", l).group(1)), float(re.search(r"generated in [\d.]+ s \(([\d.]+) tok/s\)", l).group(1)))
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
        m = re.match(r"r\d+-([\d.]+B)-(\w+)\.log", os.path.basename(f))
        t = turn(f) if m else None
        if t: runs.setdefault((m.group(1), m.group(2)), []).append(t)
    print(f"\n## {d}\n\n| model | GEMM | prefill 759 tok/s | / f32 | decode tok/s | N |\n|---|---|---|---|---|---|")
    for mdl in ("0.8B", "4B"):
        base = st.median(x[0] for x in runs.get((mdl, "f32"), [(1, 1)]))
        for v in V:
            r = runs.get((mdl, v))
            if r: print(f"| {mdl} | {v} | {st.median(x[0] for x in r):.1f} | {st.median(x[0] for x in r)/base*100:.1f} % | {st.median(x[1] for x in r):.2f} | {len(r)} |")
    for c in sorted(glob.glob(os.path.join(d, "cmp-*"))):
        print(f"\n{os.path.basename(c)} — first token that differs from the CPU oracle (gap in the GPU's logits):")
        for cpu in sorted(glob.glob(os.path.join(c, "*-cpu.log"))):
            name = os.path.basename(cpu)[:-8]; ref = ids(cpu); cells = []
            for v in V:
                f = os.path.join(c, f"{name}-vk-{v}-top.log"); got = ids(f) if os.path.exists(f) else None
                if not ref or not got: cells.append(f"{v} -"); continue
                n = min(len(ref), len(got)); i = next((k for k in range(n) if ref[k] != got[k]), None)
                if i is None: cells.append(f"{v} =({len(got)})"); continue
                t = tops(f).get(i, {}); gap = (t[got[i]] - t[ref[i]]) if ref[i] in t and got[i] in t else None
                cells.append(f"{v} {i}" + (f"({gap:g})" if gap is not None else "(cpu∉top5)"))
            print(f"  {name:11} " + " · ".join(cells))
