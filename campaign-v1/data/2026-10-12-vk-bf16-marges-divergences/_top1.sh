#!/usr/bin/env bash
# top1: margins at the first divergence against the CPU oracle, NVIDIA defaults of e692613
# (bf16 build), 4080 of beast, after the reboot on the 595.99 driver. Same prompts and
# options as compare.sh (p3 with thinking and 256 tokens, the others --no-think, 200 tokens;
# pp4000 with --ctx 4608 as in dr3). HERBERT_VK_TOP=5 prints the 5 best logits of every
# generated position. CPU logs reused from dr2/dr3 (same oracle binary, CPU unchanged).
cd ~/herbert-vk-bf16
O=runs/top1; mkdir -p $O
{ uptime; cat /proc/driver/nvidia/version; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate --format=csv; } > $O/env.txt 2>&1
for m in 0.8B 4B; do
  mkdir -p $O/cmp-$m
  cp runs/dr2/cmp-$m/*-cpu.log $O/cmp-$m/
  for p in bench/prompts/*.txt; do
    name=$(basename "$p" .txt)
    think="--no-think"; max=200; ctx=""
    case $name in p3-*) think=""; max=256;; pp4000) ctx="--ctx 4608";; esac
    HERBERT_VK_TOP=5 target-bf16/release/herbert-vk-chat /opt3/models/Qwen3.5-$m --device nvidia --no-system $think --max-tokens $max $ctx --ask "$(cat "$p")" > $O/cmp-$m/$name-vk-nvidia-top.log 2>&1
    echo "$m $name exit $?"
  done
done
echo ALLDONE-TOP1
