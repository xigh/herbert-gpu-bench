# _pp.py OUTDIR — prefill tok/s against the prompt length (pp1): herbert (median of the
# rounds, its [turn] line), llama.cpp CUDA and Vulkan (llama-bench avg_ts per n_prompt),
# vLLM (its latency bench: prompt / L(out 1)).
import sys, glob, os, re, json, statistics
out = sys.argv[1]
pps = [128, 256, 512, 1024, 2048, 4000]
rows = {}
for f in glob.glob(os.path.join(out, "r*-*-herbert-pp*.log")):
    m = re.match(r"r\d+-([\d.]+B)-herbert-pp(\d+)\.log", os.path.basename(f))
    for l in open(f, errors="replace"):
        if l.startswith("[turn"):
            p = re.search(r"(\d+) prompt tokens in [\d.]+ s \(([\d.]+) tok/s\)", l)
            rows.setdefault((m.group(1), "herbert-vk (bf16)"), {}).setdefault(int(m.group(2)), []).append(float(p.group(2)))
for m in ["0.8B", "4B"]:
    for eng, tag in [("llama.cpp CUDA", "cuda"), ("llama.cpp Vulkan", "vk")]:
        try:
            for e in json.load(open(os.path.join(out, f"{m}-llama-{tag}.json"))):
                rows.setdefault((m, eng), {}).setdefault(e["n_prompt"], []).append(e["avg_ts"])
        except Exception as e:
            print("llama", m, tag, e)
    for p in pps:
        try:
            l1 = json.load(open(os.path.join(out, f"vllm-{m}-pp{p}.json")))["avg_latency"]
            rows.setdefault((m, "vLLM (own bench)"), {}).setdefault(p, []).append(p / l1)
        except Exception as e:
            print("vllm", m, p, e)
print(f"{'model':5} {'engine':20} " + " ".join(f"{p:>9}" for p in pps))
for (m, eng), d in sorted(rows.items()):
    print(f"{m:5} {eng:20} " + " ".join(f"{statistics.median(d[p]):9.1f}" if p in d else f"{'-':>9}" for p in pps))
