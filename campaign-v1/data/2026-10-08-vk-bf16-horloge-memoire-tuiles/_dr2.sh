#!/usr/bin/env bash
# dr2: the GEMM epilogue straight from the accumulators (gemm.direct) and larger prefill
# blocks, 4080 with SM 2 805 MHz and memory 11 201 MHz locked, bf16 build, warm, N = 5 interleaved.
cd ~/herbert-vk-bf16
cp runs/pp1/prompts/pp*.txt bench/prompts/
mkdir -p runs/dr2
nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv > runs/dr2/clocks-start.txt
B=@../target-bf16/release/herbert-vk-chat
V4="base:$B --warmup 1;direct:$B --warmup 1 --tune gemm.direct=1;direct-b2048:$B --warmup 1 --tune gemm.direct=1 --tune prefill.block=2048"
V8="base:$B --warmup 1;direct:$B --warmup 1 --tune gemm.direct=1;direct-b2048:$B --warmup 1 --tune gemm.direct=1 --tune prefill.block=2048"
for p in pp512 pp2048 pp4000; do
  PROMPT=$p NOCPU=1 VARIANTS="$V4" bench/speed.sh /opt3/models/Qwen3.5-4B ~/herbert-vk-bf16/runs/dr2/$p-4B 5 nvidia
done
for p in pp1024 pp4000; do
  PROMPT=$p NOCPU=1 VARIANTS="$V8" bench/speed.sh /opt3/models/Qwen3.5-0.8B ~/herbert-vk-bf16/runs/dr2/$p-0.8B 5 nvidia
done
for m in 0.8B 4B; do
  mkdir -p runs/dr2/cmp-$m; cp runs/fa1/cmp-$m/*-cpu.log runs/dr2/cmp-$m/
  VARIANTS="base:$B;direct:$B --tune gemm.direct=1" bench/compare.sh /opt3/models/Qwen3.5-$m ~/herbert-vk-bf16/runs/dr2/cmp-$m nvidia > runs/dr2/cmp-$m.out 2>&1
done
nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv > runs/dr2/clocks-end.txt
echo ALLDONE-DR2
