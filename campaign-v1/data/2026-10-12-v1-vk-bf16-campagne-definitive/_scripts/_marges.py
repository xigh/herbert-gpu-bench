# _marges.py OUTDIR — at the first divergence of each Vulkan run against the CPU oracle,
# the GPU logits of its own choice and of the CPU's choice (HERBERT_VK_TOP lines), in bf16 quanta.
import sys, glob, os, re, math
def ids(f):
    for l in open(f, errors="replace"):
        if l.startswith("[ids]"):
            return [int(x) for x in l.split()[1:]]
def tops(f):
    t = {}
    for m in re.finditer(r"\[top (\d+)\] ((?:\d+:-?[\d.]+ ?)+)", open(f, errors="replace").read()):
        t[int(m.group(1))] = [(int(a), float(b)) for a, b in (x.split(":") for x in m.group(2).split())]
    return t
def quantum(x):  # bf16 spacing at |x|: 2^(exponent - 7)
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7) if x else 0
for d in sorted(glob.glob(os.path.join(sys.argv[1], "cmp-*"))):
    for cpu in sorted(glob.glob(os.path.join(d, "*-cpu.log"))):
        name = os.path.basename(cpu)[:-8]
        vk = os.path.join(d, f"{name}-vk-nvidia-top.log")
        ref, got = ids(cpu), (ids(vk) if os.path.exists(vk) else None)
        tag = f"{os.path.basename(d)[4:]:5} {name:11}"
        if not ref or not got:
            print(f"{tag} missing ids (cpu {bool(ref)}, vk {bool(got)})"); continue
        n = min(len(ref), len(got))
        i = next((k for k in range(n) if ref[k] != got[k]), None)
        if i is None:
            print(f"{tag} identical ({len(got)} of {len(ref)} tokens)"); continue
        t = dict(tops(vk).get(i, []))
        g, c = t.get(got[i]), t.get(ref[i])
        if c is None:
            print(f"{tag} token {i}: gpu {got[i]} {g} · cpu choice {ref[i]} not in gpu top 5"); continue
        q = quantum(g)
        print(f"{tag} token {i}: gpu {got[i]} {g} · cpu choice {ref[i]} {c} · gap {g - c:g} = {(g - c) / q:g} bf16 quantum(s) of {q:g}")
