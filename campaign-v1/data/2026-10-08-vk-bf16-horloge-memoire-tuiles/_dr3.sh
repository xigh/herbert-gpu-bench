#!/usr/bin/env bash
# dr3: the 4 000-token prompt (context 4 608) with gemm.direct and 2 048-row blocks, SM and memory
# clocks locked, bf16 build, warm, N = 5 interleaved; tokens of pp4000 against the CPU oracle.
cd ~/herbert-vk-bf16
mkdir -p runs/dr3
nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv > runs/dr3/clocks-start.txt
B=@../target-bf16/release/herbert-vk-chat
V="base:$B --ctx 4608 --warmup 1;direct:$B --ctx 4608 --warmup 1 --tune gemm.direct=1;direct-b2048:$B --ctx 4608 --warmup 1 --tune gemm.direct=1 --tune prefill.block=2048"
for m in 4B 0.8B; do
  PROMPT=pp4000 NOCPU=1 VARIANTS="$V" bench/speed.sh /opt3/models/Qwen3.5-$m ~/herbert-vk-bf16/runs/dr3/pp4000-$m 5 nvidia
done
# tokens: only pp4000 and pp2048 (the CPU logs of dr2 reused)
mkdir -p runs/dr3/prompts; cp bench/prompts/pp4000.txt bench/prompts/pp2048.txt runs/dr3/prompts/
for m in 0.8B 4B; do
  mkdir -p runs/dr3/cmp-$m; cp runs/dr2/cmp-$m/pp4000-cpu.log runs/dr2/cmp-$m/pp2048-cpu.log runs/dr3/cmp-$m/
  for p in pp2048 pp4000; do
    for v in "b1024:--tune gemm.direct=1" "b2048:--tune gemm.direct=1 --tune prefill.block=2048"; do
      n=${v%%:*}; o=${v#*:}
      target-bf16/release/herbert-vk-chat /opt3/models/Qwen3.5-$m --device nvidia --no-system --no-think --max-tokens 200 --ctx 4608 $o --ask "$(cat runs/dr3/prompts/$p.txt)" > runs/dr3/cmp-$m/$p-vk-nvidia-$n.log 2>&1
    done
  done
done
nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv > runs/dr3/clocks-end.txt
echo ALLDONE-DR3
