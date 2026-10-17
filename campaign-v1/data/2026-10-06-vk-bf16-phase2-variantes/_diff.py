# _diff.py OUTDIR — first divergence of each Vulkan run against the CPU oracle.
import sys, glob, os, re
out = sys.argv[1]
def ids(f):
    for line in open(f, errors="replace"):
        if line.startswith("[ids]"):
            return [int(x) for x in line.split()[1:]]
    return None
def turn(f):
    for line in open(f, errors="replace"):
        if line.startswith("[turn"):
            m = re.search(r"(\d+) generated in ([\d.]+) s \(([\d.]+) tok/s\)", line)
            p = re.search(r"(\d+) prompt tokens in ([\d.]+) s \(([\d.]+) tok/s\)", line)
            return f"prompt {p.group(1)} @ {p.group(3)} tok/s · gen {m.group(1)} @ {m.group(3)} tok/s"
    return "no [turn] line"
for cpu in sorted(glob.glob(os.path.join(out, "*-cpu.log"))):
    name = os.path.basename(cpu)[:-len("-cpu.log")]
    ref = ids(cpu)
    print(f"{name}: cpu {len(ref) if ref else None} tokens · {turn(cpu)}")
    for vk in sorted(glob.glob(os.path.join(out, f"{name}-vk-*.log"))):
        got = ids(vk)
        dev = os.path.basename(vk)[len(name) + 4:-4]
        if got is None:
            print(f"  {dev}: no [ids] line"); continue
        n = min(len(ref), len(got))
        first = next((i for i in range(n) if ref[i] != got[i]), None)
        if first is None and len(ref) == len(got):
            verdict = f"identical ({len(got)} tokens)"
        elif first is None:
            verdict = f"identical prefix of {n}, lengths {len(ref)} vs {len(got)}"
        else:
            verdict = f"first divergence at token {first} of {len(ref)}"
        print(f"  {dev}: {verdict} · {turn(vk)}")
