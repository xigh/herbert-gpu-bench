import json,os,statistics as st,re,glob
J=os.path.expanduser('~/bench-runs/v3/formats')
V1=os.path.expanduser('~/bench-runs/v1/beast')
def lb(f):
    try: return json.load(open(f))
    except Exception: return []
H={}
for f in glob.glob(V1+'/pp-r*-*-herbert-pp*.log'):
    m=re.search(r'pp-r\d+-([\d.]+B)-herbert-pp(\d+)',f)
    for l in open(f):
        if l.startswith('[turn'):
            H.setdefault((m.group(1),int(m.group(2))),[]).append(float(re.search(r'prompt tokens in [\d.]+ s \(([\d.]+) tok/s\)',l).group(1)))
P=[128,256,512,1024,2048,4000]
print("V3, 4080 locked, llama-bench 08ffba21e: prefill (tok/s, median of the samples) and decode of 255 tokens at depth 759 / 2880. Herbert bf16 from V1 (#541) for reference.\n")
for m in ('0.8B','4B'):
    print(f"## {m} prefill\n\n| engine | format | "+" | ".join(map(str,P))+" |\n|---|---|"+"---|"*len(P))
    print("| herbert | bf16 | "+" | ".join(f"{st.median(H[(m,p)]):.0f}" for p in P)+" |")
    for be in ('cuda','vk'):
        for q in ('BF16','F16','Q8_0'):
            d={e['n_prompt']:st.median(e['samples_ts']) for e in lb(f'{J}/pp-{m}-{q}-{be}.json')}
            print(f"| llama.cpp {be} | {q} | "+" | ".join(f"{d.get(p,float('nan')):.0f}" for p in P)+" |")
    print(f"\n## {m} decode (tok/s, 255 tokens at depth 759 / 2880)\n\n| engine | format | 759 | 2880 |\n|---|---|---|---|")
    for be in ('cuda','vk'):
        for q in ('BF16','F16','Q8_0'):
            v=[st.median(e['samples_ts']) for d in (759,2880) for e in lb(f'{J}/tg{d}-{m}-{q}-{be}.json')]
            print(f"| llama.cpp {be} | {q} | {v[0]:.1f} | {v[1]:.1f} |")
    print()
