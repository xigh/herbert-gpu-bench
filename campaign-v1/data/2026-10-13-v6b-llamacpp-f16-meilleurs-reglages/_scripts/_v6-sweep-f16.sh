#!/usr/bin/env bash
# V6, after V5 on beast (4080, clocks locked): llama.cpp 08ffba21e in F16 at its best settings, against Herbert
# f16 of V5. The micro-batch / flash attention sweep done before was on the BF16 GGUF; Herbert's lead on the
# 0.8B in V5 (×1.5 to ×2 at prefill) asks for it again in F16. llama-bench grid in one invocation per model and
# backend: prefill 759 and 2 880 tokens, -b 2048/4096, -ub 256/512/1024/2048, -fa 0/1, -r 5; decode of 255 tokens
# at depth 759 and 2 880, -fa 0/1, -r 5. CUDA0 and the 4080 through Vulkan.
set -u
L=/opt3/llama.cpp-08ffba21e; G=/opt3/bench-qwen35; O=${O:-$HOME/v6-sweep}
until grep -q ALLDONE-V5-BEAST $HOME/v5-beast.out 2>/dev/null; do sleep 60; done
mkdir -p $O
VK=$($L/build-vk/bin/llama-bench --list-devices 2>&1 | grep -E '^ *Vulkan[0-9]+:' | grep -i -- "RTX 4080" | head -1 | sed -E 's/^ *(Vulkan[0-9]+):.*/\1/')
{ date -u; (cd $L && git log -1 --format='%h %ci'); echo "Vulkan device $VK"; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate --format=csv; sha256sum $G/Qwen3.5-*-F16.gguf; } > $O/env-start.txt 2>&1
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu --format=csv -lms 500 > $O/clocks.csv 2>&1 & SMI=$!
for m in 0.8B 4B; do
  g=$G/Qwen3.5-$m-F16.gguf
  for be in cuda vk; do
    case $be in cuda) B=$L/build-cuda/bin/llama-bench; D=CUDA0;; vk) B=$L/build-vk/bin/llama-bench; D=$VK;; esac
    $B -m $g -dev $D -ngl 99 -p 759,2880 -n 0 -b 2048,4096 -ub 256,512,1024,2048 -fa 0,1 -r 5 -o json > $O/$be-$m-pp.json 2> $O/$be-$m-pp.err
    echo "$m $be pp exit $? · $(date -u +%T)"
    $B -m $g -dev $D -ngl 99 -p 0 -n 255 -d 759,2880 -fa 0,1 -r 5 -o json > $O/$be-$m-tg.json 2> $O/$be-$m-tg.err
    echo "$m $be tg exit $? · $(date -u +%T)"
  done
done
kill $SMI
{ date -u; nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv; } > $O/env-end.txt 2>&1
echo ALLDONE-V6
