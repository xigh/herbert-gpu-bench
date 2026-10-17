# _speed.py OUTDIR — medians of prefill and decode tok/s per engine, from the [turn] lines.
import sys, glob, os, re, statistics
rows = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "r*-*.log"))):
    eng = re.sub(r"^r\d+-", "", os.path.basename(f))[:-4]
    for l in open(f, errors="replace"):
        if l.startswith("[turn"):
            p = re.search(r"(\d+) prompt tokens in ([\d.]+) s \(([\d.]+) tok/s\)", l)
            g = re.search(r"(\d+) generated in ([\d.]+) s \(([\d.]+) tok/s\)", l)
            rows.setdefault(eng, []).append((float(p.group(3)), float(g.group(3)), int(g.group(1))))
for eng, v in rows.items():
    pre = [x[0] for x in v]; dec = [x[1] for x in v]
    print(f"{eng:12} N={len(v)}  prefill median {statistics.median(pre):8.1f} tok/s (min {min(pre):.1f} max {max(pre):.1f})  "
          f"decode median {statistics.median(dec):8.2f} tok/s (min {min(dec):.2f} max {max(dec):.2f})  generated {sorted(set(x[2] for x in v))}")
