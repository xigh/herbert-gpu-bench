#!/usr/bin/env python3
# V8 summary: medians over the rounds of Herbert's measured turn, per model, prompt and variant; the ratio to `base`.
# Usage: _v8_summary.py <O>  (the directory holding logs/)
import re, statistics, sys
from collections import defaultdict
from pathlib import Path

d = Path(sys.argv[1])
TURN = re.compile(r"\[turn 1\] (\d+) prompt tokens in ([\d.]+) s \(([\d.]+) tok/s\) · (\d+) generated in ([\d.]+) s \(([\d.]+) tok/s\)")
STEP = re.compile(r"step of (\d+) operations")
V = ["base", "nofuse", "split1", "nopipe", "block256", "alloff"]
pre, dec, ops, fails = defaultdict(list), defaultdict(list), {}, []
for f in sorted((d / "logs").glob("r*.log")):
    parts = f.stem.split("-")
    m, v = parts[1], parts[-1]
    p = "-".join(parts[2:-1])
    t = f.read_text(errors="replace")
    g = TURN.search(t)
    if not g:
        fails.append(f.name); continue
    pre[(m, p, v)].append(int(g[1]) / float(g[3]) * 1000)  # tokens / tok/s: the printed seconds have 3 decimals only
    if int(g[4]) > 1:
        dec[(m, p, v)].append(float(g[6]))
    s = STEP.search(t)
    if s: ops[(m, v)] = int(s[1])
P = ["p1-short", "pp512", "pp2048", "pp4000", "p3-long"]
print("V8, 4080: Herbert's own counter, medians of the rounds (prefill in ms, lower is better; decode in tok/s), and the ratio to base.\n")
for m in ["0.8B", "4B"]:
    print(f"## {m}\n")
    print("| variant | step ops | " + " | ".join(f"prefill {p} (ms)" for p in P) + " | decode p3-long (tok/s) |")
    print("|---|---|" + "---|" * (len(P) + 1))
    for v in V:
        row = [v, str(ops.get((m, v), "?"))]
        for p in P:
            x = pre.get((m, p, v))
            b = pre.get((m, p, "base"))
            row.append("—" if not x else f"{statistics.median(x):.3f} (×{statistics.median(x) / statistics.median(b):.2f}, n={len(x)})")
        x, b = dec.get((m, "p3-long", v)), dec.get((m, "p3-long", "base"))
        row.append("—" if not x else f"{statistics.median(x):.1f} (×{statistics.median(x) / statistics.median(b):.3f}, n={len(x)})")
        print("| " + " | ".join(row) + " |")
    print()
if fails:
    print("no measured turn in: " + ", ".join(fails))
